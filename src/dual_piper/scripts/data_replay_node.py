#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data_replay_node.py — 回放与验证 v6 数据集 (lerobot v6)

功能:
    1. 绘制 observation.state / action 关节曲线图
    2. 拼接多相机视频 (cam_top | cam_left | cam_right)
    3. 可选: 在真实 Piper 机械臂上复现轨迹 (--replay_on_robot)

用法:
    conda activate zhl_lerobot
    rosrun dual_piper data_replay_node.py --config src/dual_piper/config/collection.yaml
    rosrun dual_piper data_replay_node.py --config ... --episode 0
    rosrun dual_piper data_replay_node.py --config ... --episode 0 --replay_on_robot
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
import yaml

from lerobot.datasets import LeRobotDataset

# 单位换算 (与 data_replay_node_zhl.py / state_get_node.py 一致)
RAD_TO_DEG_X1000 = 180.0 / np.pi * 1000.0
GRIPPER_SCALE = 1_000_000.0

CAM_KEYS = ["cam_top", "cam_left", "cam_right"]
ROBOT_TYPE_ARMS = {
    "dual_piper": ["left", "right"],
    "single_piper_left": ["left"],
    "single_piper_right": ["right"],
}
ARM_JOINT_OFFSET = {"left": 0, "right": 7}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_episode(cfg: dict, episode: int | None):
    dcfg = cfg["dataset"]
    ds = LeRobotDataset(
        repo_id=dcfg["repo_id"],
        root=Path(dcfg["target_dir"]),
        episodes=[1] if episode is not None else None,
        return_uint8=True,
    )
    return ds


def plot_curves(ds, episode: int | None, robot_type: str, out_dir: Path):
    """绘制 state/action 曲线, 每列一个关节 (双臂时左右各一条线)."""
    arms = ROBOT_TYPE_ARMS[robot_type]
    n_joints_per_arm = 7
    ncols = 7
    start_idx = min(ARM_JOINT_OFFSET[a] for a in arms)
    indices = range(ds.num_frames)

    frames = []
    for i in indices:
        frames.append(ds[i])

    t = np.arange(len(frames)) / ds.fps
    state = np.stack([f["observation.state"].numpy() for f in frames])
    action = np.stack([f["action"].numpy() for f in frames])

    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6), sharex=True)
    colors = {"left": "tab:blue", "right": "tab:red"}
    for col in range(ncols):
        for arm in arms:
            j = ARM_JOINT_OFFSET[arm] + col
            label = arm if j == start_idx else None
            axes[0, col].plot(t, state[:, j], color=colors[arm], label=label, linewidth=0.8)
            axes[1, col].plot(t, action[:, j], color=colors[arm], linewidth=0.8)
            if col == 0:
                axes[0, col].set_ylabel("observation.state")
                axes[1, col].set_ylabel("action")
                # 图例只在第 1 列有带 label 的曲线, 其他列调用 legend() 会触发
                # "No artists with labels found" 警告
                axes[0, col].legend(loc="upper right", fontsize=8)
            axes[0, col].set_title(f"joint_{start_idx + col + 1}")
            axes[0, col].grid(alpha=0.3)
            axes[1, col].grid(alpha=0.3)
    axes[1, 0].set_xlabel("time (s)")
    fig.suptitle(f"episode {episode if episode is not None else 'all'}: {ds.num_frames} frames, task='{ds[0]['task']}'")
    fig.tight_layout()
    out = out_dir / f"replay_curves_ep{episode if episode is not None else 'all'}.png"
    fig.savefig(out, dpi=150)
    print(f"📈 曲线图已保存: {out}")


def stitch_video(ds, episode: int | None, out_dir: Path):
    """拼接三路相机视频并保存."""
    first = ds[0]
    img0 = first["observation.images.cam_top"].numpy()  # (C, H, W)
    h, w = img0.shape[1:]
    out_path = out_dir / f"replay_stitched_ep{episode if episode is not None else 'all'}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), ds.fps, (w * 3, h)
    )
    for i in range(ds.num_frames):
        item = ds[i]
        tiles = [np.ascontiguousarray(item[f"observation.images.{cam}"].numpy().transpose(1, 2, 0)) for cam in CAM_KEYS]
        view = np.hstack(tiles)
        writer.write(view)
    writer.release()
    print(f"🎬 拼接视频已保存: {out_path} ({ds.num_frames} 帧 @ {ds.fps}fps)")


# ═════════════════════════════════════════════════════════════════════════════
#  真机回放 (piper_sdk 直控, 沿用 data_replay_node_zhl.py 已验证模式)
# ═════════════════════════════════════════════════════════════════════════════

def _send_arm_command(piper, cmd: np.ndarray, speed: int = 100) -> None:
    """发送 [j1..j6 (rad), gripper (m)] 到单臂 (MIT 模式 0xAD)."""
    piper.MotionCtrl_2(0x01, 0x01, speed, 0xAD)
    piper.JointCtrl(*[int(v * RAD_TO_DEG_X1000) for v in cmd[:6]])
    piper.GripperCtrl(int(abs(cmd[6] * GRIPPER_SCALE)), 1000, 0x01, 0)


def _move_to_start(piper, current: np.ndarray, target: np.ndarray, hz: int = 30) -> None:
    """平滑移动到起始位姿 (每步限幅 0.01 rad, MIT 模式无插值)."""
    n_steps = max(int(5.0 * hz), 1)
    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        step = current + (target - current) * alpha
        _send_arm_command(piper, step)
        time.sleep(1.0 / hz)


def replay_on_robot(ds, episode: int | None, robot_type: str, can_ports: dict) -> None:
    """在真实机械臂上复现轨迹 (动作须先移动到起始位姿)."""
    from piper_sdk import C_PiperInterface

    arms = ROBOT_TYPE_ARMS[robot_type]
    pipers = {}
    for arm in arms:
        piper = C_PiperInterface(can_name=can_ports[arm])
        piper.ConnectPort()
        piper.EnableArm(7)
        pipers[arm] = piper
        print(f"[replay] {arm} arm connected on {can_ports[arm]}")

    try:
        frames = [ds[i] for i in range(ds.num_frames)]
        first = frames[0]["observation.state"].numpy()
        for arm in arms:
            off = ARM_JOINT_OFFSET[arm]
            _move_to_start(pipers[arm], np.zeros(7), first[off : off + 7])

        print("▶ 开始复现轨迹...")
        for i, item in enumerate(frames):
            q = item["action"].numpy()
            for arm in arms:
                off = ARM_JOINT_OFFSET[arm]
                _send_arm_command(pipers[arm], q[off : off + 7])
            time.sleep(1.0 / ds.fps)
        print("✅ 轨迹复现完成")
    finally:
        for arm in arms:
            try:
                pipers[arm].EnableArm(0)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="回放与验证 lerobot v6 数据集")
    parser.add_argument("--config", required=True, help="collection.yaml 路径")
    parser.add_argument("--episode", type=int, default=None, help="episode 索引 (默认全部)")
    parser.add_argument("--out_dir", default=None, help="输出目录 (默认 target_dir/replay_output)")
    parser.add_argument("--replay_on_robot", action="store_true", help="在真实机械臂上复现")
    parser.add_argument("--can_left", default="can_left")
    parser.add_argument("--can_right", default="can_right")
    args = parser.parse_args()

    cfg = load_config(args.config)
    robot_type = cfg["robot"]["robot_type"]
    ds = load_episode(cfg, args.episode)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["dataset"]["target_dir"]) / "replay_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"数据集: {ds.num_episodes} episodes / {ds.num_frames} 帧, "
        f"robot_type={robot_type}, fps={ds.fps}"
    )

    if ds.num_frames == 0:
        print("数据集为空")
        sys.exit(1)

    plot_curves(ds, args.episode, robot_type, out_dir)
    stitch_video(ds, args.episode, out_dir)

    if args.replay_on_robot:
        replay_on_robot(ds, args.episode, robot_type, {"left": args.can_left, "right": args.can_right})


if __name__ == "__main__":
    main()
