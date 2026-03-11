#!/home/user/anaconda3/envs/lerobotv3/bin/python
# -*- coding: utf-8 -*-
"""
Dual-arm Data Replay Node

Loads an HDF5 episode recorded by the data-collection node, visualizes
joint-position / action curves and stitched camera videos, and optionally
replays the trajectory on the real Piper arms.

Usage::

    rosrun dual_piper data_replay_node.py \\
        --dataset_dir /path/to/data \\
        --task_name my_task \\
        --episode_idx 0
"""
import os
import sys
# Ensure the scripts directory is on sys.path so that the utils package is
# importable regardless of the working directory used by rosrun / roslaunch.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse

import numpy as np
import cv2
import h5py
import rospy
from sensor_msgs.msg import JointState
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import RAD_TO_DEG_X1000, GRIPPER_SCALE


class DataReplayNode:
    """
    ROS node for replaying HDF5 episodes.

    Provides:
    * ``load_hdf5``  — load observations, images, actions
    * ``plot_qpos_action`` — joint-position / action curve plots
    * ``stitch_and_save_videos`` — side-by-side camera video export
    * ``replay_trajectory_on_robot`` — send actions to real arms via SDK
    """

    # Camera ordering used when stitching frames.
    _STITCH_ORDER = ('cam_top', 'cam_left', 'cam_right')


    def __init__(self, args):
        rospy.init_node('data_replay_node', anonymous=True)
        self.args = args

        # ── Publishers (for optional ROS-level replay) ───────────────────
        self._slave_pub = rospy.Publisher(
            args.slave_arm_topic, JointState, queue_size=10)
        self._master_pub = rospy.Publisher(
            args.master_arm_topic, JointState, queue_size=10)

        # ── Load data ────────────────────────────────────────────────────
        dataset_dir = os.path.join(args.dataset_dir, args.task_name)
        dataset_path = os.path.join(dataset_dir, f'episode_{args.episode_idx}.hdf5')
        (self.qpos, self.qvels, self.efforts,
         self.actions, self.image_dicts, self.depth_dicts) = \
            self.load_hdf5(dataset_path)


        # ── Visualize ────────────────────────────────────────────────────
        output_dir = os.path.join(dataset_dir, 'output')
        self.plot_qpos_action(self.qpos, self.actions, output_dir=output_dir)
        self.stitch_and_save_videos(
            self.image_dicts,
            os.path.join(output_dir,
                         f'stitched_video_episode{args.episode_idx}.mp4'))
        if self.depth_dicts is not None:
            self.stitch_and_save_videos(
                self.depth_dicts,
                os.path.join(output_dir,
                             f'stitched_depth_episode{args.episode_idx}.mp4'),
                mode='depth')

    # ── HDF5 Loading ─────────────────────────────────────────────────────

    @staticmethod
    def load_hdf5(dataset_path, use_depth=False):
        """
        Load robot trajectory data from an HDF5 file.

        Returns:
            tuple: (qpos, qvel, effort, actions, image_dict, depth_dict)
        """
        if not os.path.isfile(dataset_path):
            rospy.logerr("Dataset not found: %s", dataset_path)
            sys.exit(1)

        with h5py.File(dataset_path, 'r') as root:
            compressed = root.attrs.get('compress', False)
            qpos = root['/observations/qpos'][:, :]
            qvel = root['/observations/qvel'][:, :]
            action = root['/action'][:, :]
            effort = (root['/observations/effort'][:, :]
                      if 'effort' in root['/observations/'] else None)

            image_dict = {
                cam: root[f'/observations/images/{cam}'][()]
                for cam in root['/observations/images/'].keys()
            }

            depth_dict = None
            if use_depth and '/observations/images_depth/' in root:
                depth_dict = {
                    cam: root[f'/observations/images_depth/{cam}'][()]
                    for cam in root['/observations/images_depth/'].keys()
                }

            compress_len = (root['/compress_len'][()]
                            if compressed else None)

        # Decompress JPEG-encoded images if needed
        if compressed:
            for cam_name in list(image_dict.keys()):
                decoded = []
                for frame in image_dict[cam_name]:
                    img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    if img is not None:
                        decoded.append(img)
                image_dict[cam_name] = decoded

        return qpos, qvel, effort, action, image_dict, depth_dict

    # ── Plotting ─────────────────────────────────────────────────────────

    def plot_qpos_action(self, qpos, action, output_dir=None):
        """Plot joint-position and action curves; optionally save to *output_dir*."""
        timesteps = range(qpos.shape[0])
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        for i in range(qpos.shape[1]):
            axs[0].plot(timesteps, qpos[:, i], label=f'Joint {i + 1}')
        axs[0].set_title('Joint Position (qpos) Over Time')
        axs[0].set_xlabel('Timestep')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        axs[0].grid(True)

        for i in range(action.shape[1]):
            axs[1].plot(timesteps, action[:, i], label=f'Action {i + 1}')
        axs[1].set_title('Action Over Time')
        axs[1].set_xlabel('Timestep')
        axs[1].set_ylabel('Action Value')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(
                output_dir, f'qpos_action_{self.args.episode_idx}.png'))
        plt.close(fig)

    # ── Video Stitching ──────────────────────────────────────────────────
    def stitch_and_save_videos(self, image_dict, output_path,
                               mode='rgb', fps=10):
        """
        Horizontally stitch multi-camera frames and write an MP4 video.

        Args:
            image_dict: ``{cam_name: list_of_frames}``
            output_path: path for the output ``.mp4`` file.
            mode: ``'rgb'`` or ``'depth'``.
            fps: output video frame rate.
        """
        if mode not in ('rgb', 'depth'):
            raise ValueError("mode must be 'rgb' or 'depth'")

        cam_names = [c for c in self._STITCH_ORDER if c in image_dict]
        if not cam_names:
            rospy.logwarn("No camera data to stitch.")
            return

        frames_list = [image_dict[c] for c in cam_names]

        # Normalize depth frames to 8-bit BGR
        if mode == 'depth':
            for i, frames in enumerate(frames_list):
                frames_list[i] = [
                    cv2.cvtColor(
                        cv2.normalize(f, None, 0, 255,
                                      cv2.NORM_MINMAX).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)
                    for f in frames
                ]
        heights = [frames[0].shape[0] for frames in frames_list]
        widths = [frames[0].shape[1] for frames in frames_list]
        out_h = max(heights)
        out_w = sum(widths)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        for frame_set in tqdm(zip(*frames_list),
                              desc=f"Stitching {mode} video"):
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            x = 0
            for frame in frame_set:
                h, w = frame.shape[:2]
                canvas[:h, x:x + w] = frame
                x += w
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        writer.release()
        rospy.loginfo("Saved stitched %s video → %s", mode, output_path)

    # ── Real-Robot Replay (optional) ─────────────────────────────────────

    def replay_trajectory_on_robot(self):
        """Replay loaded actions on physical Piper arms via the SDK."""
        from piper_sdk import C_PiperInterface
        import time

        rospy.loginfo("Connecting arms for replay …")
        piper_left = C_PiperInterface(can_name='can_left')
        piper_left.ConnectPort()
        piper_right = C_PiperInterface(can_name='can_right')
        piper_right.ConnectPort()

        piper_left.EnableArm(7)
        piper_right.EnableArm(7)

        # Move to zero position first
        for piper in (piper_left, piper_right):
            piper.MotionCtrl_2(0x01, 0x01, 20)
            piper.JointCtrl(0, 0, 0, 0, 0, 0)
        time.sleep(2)

        rate = rospy.Rate(self.args.frame_rate)
        for action in tqdm(self.actions, desc="Replaying trajectory"):
            left_action = action[:7]
            right_action = action[7:]

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

            rate.sleep()

        rospy.loginfo("Trajectory replay finished.")


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Replay an HDF5 episode and visualize data.')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/user/zzq/piper_ws/data')
    parser.add_argument('--task_name', type=str, default='bloodgas_0306')
    parser.add_argument('--episode_idx', type=int, default=3)
    parser.add_argument('--camera_names', type=str,
                        default='cam_top,cam_left,cam_right')
    parser.add_argument('--master_arm_topic', type=str,
                        default='joint_ctrl_single')
    parser.add_argument('--slave_arm_topic', type=str,
                        default='/joint_states')
    parser.add_argument('--frame_rate', type=int, default=20)
    args = parser.parse_args()
    if isinstance(args.camera_names, str):
        args.camera_names = [c.strip() for c in args.camera_names.split(',')]
    return args


def main():
    try:
        args = _parse_args()
        rospy.loginfo("data_replay_node starting …")
        DataReplayNode(args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()






