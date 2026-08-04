#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data_to_lerobot3_node_repeat.py — episode 复制增强 (lerobot v6)

将已录制的高质量 episode 复制 N 份, 每份经 v6 重新编码为独立的视频 + 数据,
用于数据增强。原版"暂停时按 d"交互改为干净 CLI:

用法:
    conda activate zhl_lerobot
    rosrun dual_piper data_to_lerobot3_node_repeat.py \
        --config src/dual_piper/config/collection.yaml --episode 0 --copies 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from lerobot.datasets import LeRobotDataset

# 复用采集节点的特征/编码器/修复逻辑
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from data_to_lerobot3_node import build_encoder, repair_dataset_consistency  # noqa: E402

CAM_KEYS = ["cam_top", "cam_left", "cam_right"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="复制 episode 进行数据增强 (lerobot v6)")
    parser.add_argument("--config", required=True, help="collection.yaml 路径")
    parser.add_argument("--episode", type=int, required=True, help="要复制的 episode 索引")
    parser.add_argument("--copies", type=int, default=2, help="复制份数")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    root = Path(dcfg["target_dir"])
    repo_id = dcfg["repo_id"]

    if not (root / "meta" / "info.json").exists():
        print(f"数据集不存在: {root}")
        sys.exit(1)

    # 读端 (视频解码, uint8)
    src = LeRobotDataset(repo_id=repo_id, root=root, episodes=[args.episode], return_uint8=True)
    n_frames = src.num_frames
    task = str(src[0]["task"])
    print(f"源 episode #{args.episode}: {n_frames} 帧, task='{task}'")

    # 写端 (resume 追加; v6 save_episode 自动重新编码视频)
    repair_dataset_consistency(root, repo_id)
    dst = LeRobotDataset.resume(
        repo_id=repo_id,
        root=root,
        rgb_encoder=build_encoder(dcfg.get("video_codec", "h264_nvenc")),
        streaming_encoding=bool(dcfg.get("streaming_encoding", True)),
        encoder_queue_maxsize=30,
    )
    print(f"追加模式: 当前共 {dst.meta.total_episodes} 个 episode")

    try:
        for k in range(args.copies):
            for i in range(n_frames):
                item = src[i]
                frame = {
                    "observation.state": np.asarray(item["observation.state"], dtype=np.float32),
                    "action": np.asarray(item["action"], dtype=np.float32),
                }
                for cam in CAM_KEYS:
                    # 读端解码返回 (C, H, W), add_frame 需要 (H, W, C)
                    img = np.ascontiguousarray(item[f"observation.images.{cam}"].numpy().transpose(1, 2, 0))
                    frame[f"observation.images.{cam}"] = img
                frame["task"] = task
                dst.add_frame(frame)
            dst.save_episode()
            print(f"💾 已追加副本 #{k + 1}/{args.copies} ({n_frames} 帧)")
    finally:
        dst.finalize()
    print(f"✅ 完成: 数据集现有 {dst.meta.total_episodes} 个 episode")


if __name__ == "__main__":
    main()
