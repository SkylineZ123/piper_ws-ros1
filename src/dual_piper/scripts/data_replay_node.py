#!/home/zzq/.conda/envs/robomlt/bin/python3
# -*- coding: utf-8 -*-
"""
LeRobot v3 Data Replay Node

Loads a LeRobot v3 dataset episode (Parquet + MP4), visualizes joint-position /
action curves and stitched camera videos, and optionally replays the trajectory
on the real Piper arms.

Usage (standalone)::

    python data_replay_node.py --dataset_dir /path/to/lerobot_v3 --episode_idx 0

Usage (via rosrun)::

    rosrun dual_piper data_replay_node.py --dataset_dir /path/to/lerobot_v3 --episode_idx 0
"""

import argparse
import json
import os
import sys

import av
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# ── Optional ROS ──────────────────────────────────────────────────────────────
try:
    import rospy
    _HAS_ROS = True
except ImportError:
    _HAS_ROS = False


def _log(msg: str, level: str = 'info'):
    if _HAS_ROS and rospy.core.is_initialized():
        getattr(rospy, level)(msg)
    else:
        prefix = {'info': '[INFO]', 'warn': '[WARN]', 'error': '[ERROR]'}.get(level, '[INFO]')
        print(f'{prefix} {msg}', flush=True)


# ── Constants ─────────────────────────────────────────────────────────────────
# Piper SDK unit conversions (for robot replay)
RAD_TO_DEG_X1000 = 180.0 / np.pi * 1000.0
GRIPPER_SCALE = 1_000_000.0

# Camera stitch order (left → right in the output frame)
_STITCH_ORDER = ('cam_top', 'cam_left', 'cam_right')


# ═════════════════════════════════════════════════════════════════════════════
#  Dataset helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_info(dataset_dir: str) -> dict:
    path = os.path.join(dataset_dir, 'meta', 'info.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'meta/info.json not found: {path}')
    with open(path) as f:
        return json.load(f)


def _load_subtasks(dataset_dir: str) -> dict:
    """Return {subtask_index (int): subtask_name (str)}, empty dict if file absent."""
    path = os.path.join(dataset_dir, 'meta', 'subtasks.parquet')
    if not os.path.isfile(path):
        return {}
    df = pq.read_table(path).to_pandas()
    # index column = subtask_index, 'subtask' is the row index label
    return {int(row['subtask_index']): idx for idx, row in df.iterrows()}


def _load_episode_meta(dataset_dir: str, episode_idx: int) -> pd.Series:
    """Return the episode metadata row for *episode_idx*."""
    ep_dir = os.path.join(dataset_dir, 'meta', 'episodes')
    rows = []
    for chunk_dir in sorted(os.listdir(ep_dir)):
        for fname in sorted(os.listdir(os.path.join(ep_dir, chunk_dir))):
            tbl = pq.read_table(os.path.join(ep_dir, chunk_dir, fname)).to_pandas()
            rows.append(tbl)
    all_eps = pd.concat(rows, ignore_index=True)
    match = all_eps[all_eps['episode_index'] == episode_idx]
    if match.empty:
        raise ValueError(f'Episode {episode_idx} not found (total: {len(all_eps)})')
    return match.iloc[0]


def _load_episode_data(dataset_dir: str, ep_meta: pd.Series) -> pd.DataFrame:
    """Load the data parquet rows that belong to this episode."""
    chunk_idx = int(ep_meta['data/chunk_index'])
    file_idx = int(ep_meta['data/file_index'])
    parquet_path = os.path.join(
        dataset_dir, 'data',
        f'chunk-{chunk_idx:03d}', f'file-{file_idx:03d}.parquet'
    )
    df = pq.read_table(parquet_path).to_pandas()
    ep_idx = int(ep_meta['episode_index'])
    return df[df['episode_index'] == ep_idx].reset_index(drop=True)


def _load_camera_frames(dataset_dir: str, ep_meta: pd.Series,
                        camera_name: str, fps: float) -> list:
    """
    Read all frames for *camera_name* in this episode from its MP4 file.

    Uses PyAV (libdav1d) so AV1-encoded videos are decoded correctly, unlike
    the default OpenCV build which lacks a software AV1 decoder.

    Returns a list of BGR uint8 ndarrays (H, W, 3).
    """
    key = f'videos/observation.images.{camera_name}'
    if f'{key}/chunk_index' not in ep_meta.index:
        _log(f'Camera {camera_name} not found in episode metadata — skipping.', 'warn')
        return []

    chunk_idx = int(ep_meta[f'{key}/chunk_index'])
    file_idx = int(ep_meta[f'{key}/file_index'])
    from_ts = float(ep_meta[f'{key}/from_timestamp'])
    to_ts = float(ep_meta[f'{key}/to_timestamp'])

    video_path = os.path.join(
        dataset_dir, 'videos', f'observation.images.{camera_name}',
        f'chunk-{chunk_idx:03d}', f'file-{file_idx:03d}.mp4'
    )
    if not os.path.isfile(video_path):
        _log(f'Video file not found: {video_path}', 'warn')
        return []

    n_frames = int(round((to_ts - from_ts) * fps))

    container = av.open(video_path)
    stream = container.streams.video[0]

    # Seek to approximately the right position (AV_TIME_BASE = microseconds)
    seek_pts = int(from_ts * 1e6)
    container.seek(seek_pts)

    frames = []
    for pkt in container.decode(stream):
        frame_ts = float(pkt.pts * stream.time_base)
        if frame_ts < from_ts - 1e-3:
            continue
        # PyAV returns RGB; convert to BGR for consistency with cv2
        frames.append(pkt.to_ndarray(format='bgr24'))
        if len(frames) >= n_frames:
            break

    container.close()
    return frames


# ═════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════════════════════

def _joint_label(info: dict, idx: int, prefix: str) -> str:
    """Return a human-readable label for joint *idx* from info.json feature names."""
    try:
        return info['features'][prefix]['names'][idx]
    except (KeyError, IndexError, TypeError):
        return f'j{idx}'


def plot_qpos_action(qpos: np.ndarray, action: np.ndarray,
                     info: dict, output_path: str) -> None:
    """Plot joint-position (qpos) and action curves; save to *output_path*."""
    T = qpos.shape[0]
    t = np.arange(T) / float(info.get('fps', 20))

    dim = qpos.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(dim):
        lbl = _joint_label(info, i, 'observation.state')
        axes[0].plot(t, qpos[:, i], label=lbl)
    axes[0].set_title('Observation State (qpos) over Time')
    axes[0].set_ylabel('Position (rad)')
    axes[0].legend(fontsize=6, ncol=4)
    axes[0].grid(True, alpha=0.4)

    for i in range(action.shape[1]):
        lbl = _joint_label(info, i, 'action')
        axes[1].plot(t, action[:, i], label=lbl)
    axes[1].set_title('Action over Time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Position (rad)')
    axes[1].legend(fontsize=6, ncol=4)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    _log(f'Saved joint curve plot → {output_path}')


def stitch_and_save_video(image_dict: dict, output_path: str, fps: float) -> None:
    """
    Horizontally stitch multi-camera frames and write an MP4 video.

    Args:
        image_dict: {camera_name: list[BGR ndarray]}
        output_path: output .mp4 path
        fps: output video frame rate
    """
    cam_names = [c for c in _STITCH_ORDER if c in image_dict and image_dict[c]]
    if not cam_names:
        _log('No camera frames to stitch.', 'warn')
        return

    frame_lists = [image_dict[c] for c in cam_names]
    n_frames = min(len(fl) for fl in frame_lists)
    if n_frames == 0:
        _log('All camera frame lists are empty — skipping video.', 'warn')
        return

    h = max(fl[0].shape[0] for fl in frame_lists)
    w = sum(fl[0].shape[1] for fl in frame_lists)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i in tqdm(range(n_frames), desc='Stitching video', ncols=80):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        x = 0
        for fl in frame_lists:
            frame = fl[i] if i < len(fl) else np.zeros_like(fl[0])
            fh, fw = frame.shape[:2]
            canvas[:fh, x:x + fw] = frame
            x += fw
        writer.write(canvas)

    writer.release()
    _log(f'Saved stitched video → {output_path}')


# ═════════════════════════════════════════════════════════════════════════════
#  Real-robot replay (optional)
# ═════════════════════════════════════════════════════════════════════════════

def replay_on_robot(actions: np.ndarray, fps: float,
                    can_left: str, can_right: str) -> None:
    """Send action sequence to real Piper arms via the SDK."""
    try:
        from piper_sdk import C_PiperInterface
    except ImportError:
        _log('piper_sdk not installed — cannot replay on robot.', 'error')
        return

    import time

    _log(f'Connecting arms: can_left={can_left}  can_right={can_right}')
    piper_left = C_PiperInterface(can_name=can_left)
    piper_left.ConnectPort()
    piper_right = C_PiperInterface(can_name=can_right)
    piper_right.ConnectPort()

    piper_left.EnableArm(7)
    piper_right.EnableArm(7)
    time.sleep(1.0)

    # Move to zero first
    for arm in (piper_left, piper_right):
        arm.MotionCtrl_2(0x01, 0x01, 20)
        arm.JointCtrl(0, 0, 0, 0, 0, 0)
    time.sleep(2.0)

    dt = 1.0 / fps
    _log(f'Replaying {len(actions)} frames at {fps} Hz …')
    for action in tqdm(actions, desc='Replaying', ncols=80):
        left_action = action[:7]
        right_action = action[7:14]

        left_joints = [int(v * RAD_TO_DEG_X1000) for v in left_action[:6]]
        left_grip = int(abs(left_action[6] * GRIPPER_SCALE))

        right_joints = [int(v * RAD_TO_DEG_X1000) for v in right_action[:6]]
        right_grip = int(abs(right_action[6] * GRIPPER_SCALE))

        piper_left.MotionCtrl_2(0x01, 0x01, 60)
        piper_left.JointCtrl(*left_joints)
        piper_left.GripperCtrl(left_grip, 1000, 0x01, 0)

        piper_right.MotionCtrl_2(0x01, 0x01, 60)
        piper_right.JointCtrl(*right_joints)
        piper_right.GripperCtrl(right_grip, 1000, 0x01, 0)

        time.sleep(dt)

    _log('Trajectory replay finished.')


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Replay / visualize a LeRobot v3 dataset episode.')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/user/hdd_4t/data/data_lerobot3/dualpiper/BloodGasAnalysis_20260621',
                        help='Root directory of the LeRobot v3 dataset.')
    parser.add_argument('--episode_idx', type=int, default=150,
                        help='Episode index to replay / visualize.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots and videos. '
                             'Defaults to <dataset_dir>/replay_output.')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip joint-curve plot generation.')
    parser.add_argument('--no_video', action='store_true',default=True,
                        help='Skip stitched video generation.')
    parser.add_argument('--replay_on_robot', action='store_true',
                        help='Send actions to real Piper arms via CAN bus.')
    parser.add_argument('--can_left', type=str, default='can_left',
                        help='CAN port for the left arm (robot replay only).')
    parser.add_argument('--can_right', type=str, default='can_right',
                        help='CAN port for the right arm (robot replay only).')
    return parser.parse_args()


def main():
    if _HAS_ROS:
        rospy.init_node('data_replay_node', anonymous=True)

    args = _parse_args()

    dataset_dir = args.dataset_dir
    episode_idx = args.episode_idx
    output_dir = args.output_dir or os.path.join(dataset_dir, 'replay_output')

    # ── Load dataset metadata ──────────────────────────────────────────────
    _log(f'Loading dataset: {dataset_dir}')
    info = _load_info(dataset_dir)
    fps = float(info.get('fps', 5))

    # fps =5
    camera_names = [
        k.replace('observation.images.', '')
        for k in info.get('features', {})
        if k.startswith('observation.images.')
    ]
    subtasks = _load_subtasks(dataset_dir)

    _log(f'  total_episodes : {info.get("total_episodes")}')
    _log(f'  total_frames   : {info.get("total_frames")}')
    _log(f'  fps            : {fps}')
    _log(f'  cameras        : {camera_names}')
    _log(f'  subtasks       : {subtasks}')

    # ── Load episode ───────────────────────────────────────────────────────
    _log(f'Loading episode {episode_idx} …')
    ep_meta = _load_episode_meta(dataset_dir, episode_idx)
    ep_df = _load_episode_data(dataset_dir, ep_meta)

    n_frames = len(ep_df)
    _log(f'  frames: {n_frames}  length: {ep_meta["length"]}')

    # ── Extract arrays ─────────────────────────────────────────────────────
    qpos = np.stack(ep_df['observation.state'].values).astype(np.float32)
    action = np.stack(ep_df['action'].values).astype(np.float32)

    if subtasks and 'subtask_index' in ep_df.columns:
        unique_st = ep_df['subtask_index'].dropna().unique()
        _log(f'  subtasks in episode: {[subtasks.get(int(i), i) for i in sorted(unique_st)]}')

    # ── Plot joint curves ──────────────────────────────────────────────────
    if not args.no_plot:
        plot_path = os.path.join(output_dir, f'qpos_action_ep{episode_idx:04d}.png')
        plot_qpos_action(qpos, action, info, plot_path)

    # ── Load and stitch camera videos ──────────────────────────────────────
    if not args.no_video:
        image_dict = {}
        for cam in camera_names:
            _log(f'  loading camera: {cam}')
            frames = _load_camera_frames(dataset_dir, ep_meta, cam, fps)
            _log(f'    → {len(frames)} frames')
            image_dict[cam] = frames

        video_path = os.path.join(output_dir, f'stitched_ep{episode_idx:04d}.mp4')
        stitch_and_save_video(image_dict, video_path, fps)

    # ── Optional robot replay ──────────────────────────────────────────────
    if args.replay_on_robot:
        replay_on_robot(action, fps, args.can_left, args.can_right)

    _log(f'Done. Output dir: {output_dir}')

    if _HAS_ROS and rospy.core.is_initialized():
        rospy.spin()


if __name__ == '__main__':
    main()
