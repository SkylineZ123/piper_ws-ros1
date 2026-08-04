#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data_to_lerobot3_node.py — Piper 双臂遥操作数据采集节点 (lerobot v6)

订阅 (由 teleoperation.launch 启动):
    /cam_top/color/image_raw, /cam_leftwrist/color/image_raw, /cam_rightwrist/color/image_raw
    /left/{master,slave}_joint_states, /right/{master,slave}_joint_states

写入 lerobot v6 (v3.0) 格式数据集:
    <target_dir>/{data/, videos/, meta/info.json, meta/tasks.parquet, meta/episodes/}
    单一任务 (task_name), 无 subtask 逻辑。

键盘控制:
    c     开始/继续采集
    空格  暂停采集
    s     保存当前 episode
    r     丢弃当前缓存帧
    q     保存并退出

用法:
    conda activate zhl_lerobot
    rosrun dual_piper data_to_lerobot3_node.py
    rosrun dual_piper data_to_lerobot3_node.py _config_file:=/path/to/collection.yaml
"""

import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rospy
import yaml
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import JointState

import cv2

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep

# ═════════════════════════════════════════════════════════════════════════════
#  常量 / 特征定义
# ═════════════════════════════════════════════════════════════════════════════

# 话题契约 (与 teleoperation.launch / state_get_node 一致)
CAM_TOPICS = {
    "cam_top": "/cam_top/color/image_raw",
    "cam_left": "/cam_leftwrist/color/image_raw",
    "cam_right": "/cam_rightwrist/color/image_raw",
}
JOINT_TOPIC_TMPL = "/{side}/{source}_joint_states"  # source: master | slave

# robot_type -> (arms, 每臂 7 关节含夹爪)
ROBOT_TYPE_ARMS = {
    "dual_piper": ["left", "right"],
    "single_piper_left": ["left"],
    "single_piper_right": ["right"],
}
# 右臂关节编号偏移 (与旧数据集 joint_1..14 命名一致)
ARM_JOINT_OFFSET = {"left": 0, "right": 7}


def build_features(robot_type: str, height: int, width: int) -> dict:
    """构造 lerobot v6 features 字典 (不含 DEFAULT_FEATURES, 由 v6 自动合并)."""
    arms = ROBOT_TYPE_ARMS[robot_type]
    start = min(ARM_JOINT_OFFSET[a] for a in arms) + 1
    end = max(ARM_JOINT_OFFSET[a] for a in arms) + 7
    state_names = [f"joint_{i}.pos" for i in range(start, end + 1)]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
    }
    for cam in CAM_TOPICS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def build_encoder(codec: str) -> RGBEncoderConfig:
    """按配置 codec 构造视频编码器, 失败时逐级回退 (h264_nvenc -> h264 -> libsvtav1)."""
    # 兼容旧配置写法: libx264 -> h264
    if codec == "libx264":
        codec = "h264"
    candidates = list(dict.fromkeys([codec, "h264_nvenc", "h264", "libsvtav1"]))
    for cand in candidates:
        try:
            # h264_nvenc 在 PyAV 中 preset 需为数值 (如 4 = P4 档)
            enc = RGBEncoderConfig(vcodec=cand, preset=4 if "nvenc" in cand else None)
            print(f"[encoder] 使用视频编码器: {cand}")
            return enc
        except ValueError as e:
            print(f"[encoder] {cand} 不可用: {e}")
    raise RuntimeError("没有可用的视频编码器, 请安装 ffmpeg 编解码支持")


# ═════════════════════════════════════════════════════════════════════════════
#  崩溃恢复
# ═════════════════════════════════════════════════════════════════════════════

def repair_dataset_consistency(root: Path, repo_id: str = "local") -> None:
    """修复 v6 数据集崩溃残留的不一致 (resume 前调用).

    崩溃可能残留:
      1. 流式编码临时目录 root/tmp* (未完成的 mp4)
      2. meta/episodes/*.parquet 落后于 info.json (episode 元数据每
         metadata_buffer_size 集才落盘, 崩溃会丢失最近若干集的元数据行)
      3. 已保存但元数据缺失的 data/*.parquet 与 videos/*.mp4 悬空文件

    修复策略 (与旧节点 repair_dataset_consistency 一致): 截断到已落盘的
    episode 元数据为准, 删除悬空数据/视频, 重写 info.json 计数。
    """
    import pyarrow.parquet as pq

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return

    # 1) 清理流式编码崩溃残留的临时目录
    for d in root.glob("tmp*"):
        if d.is_dir():
            print(f"[repair] 删除残留临时目录: {d.name}")
            shutil.rmtree(d, ignore_errors=True)

    # 2) 收集已落盘的 episode 元数据行
    ep_rows = []
    for f in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        ep_rows.extend(pq.read_table(f).to_pylist())
    ep_rows.sort(key=lambda r: r["episode_index"])

    with open(info_path) as f:
        import json

        info = json.load(f)
    persisted_eps = len(ep_rows)
    info_eps = info.get("total_episodes", 0)
    if persisted_eps == info_eps:
        print("[repair] 数据集一致, 无需修复")
        return

    print(
        f"[repair] 检测到不一致: info.json total_episodes={info_eps}, "
        f"已落盘 episode 元数据={persisted_eps}, 截断到 {persisted_eps}"
    )

    # 3) 截断 data parquet: 只保留 episode_index < persisted_eps 的行
    trunc_frames = ep_rows[-1]["dataset_to_index"] if ep_rows else 0
    for f in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        tbl = pq.read_table(f)
        mask = tbl.column("episode_index").to_numpy() < persisted_eps
        if mask.all():
            continue
        new_tbl = tbl.filter(tbl.column("episode_index").to_numpy() < persisted_eps)
        pq.write_table(new_tbl, f, compression="snappy", use_dictionary=True)
        print(f"[repair] 截断数据文件: {f} ({len(tbl)} -> {new_tbl.num_rows} 行)")

    # 4) 删除未被 episode 元数据引用的视频文件
    referenced = set()
    for r in ep_rows:
        for cam in ("cam_top", "cam_left", "cam_right"):
            ck = f"videos/observation.images.{cam}/chunk_index"
            fk = f"videos/observation.images.{cam}/file_index"
            if ck in r and fk in r:
                referenced.add(
                    f"videos/observation.images.{cam}/chunk-{int(r[ck]):03d}/file-{int(r[fk]):03d}.mp4"
                )
    for vf in (root / "videos").rglob("*.mp4"):
        if vf.relative_to(root).as_posix() not in referenced:
            print(f"[repair] 删除悬空视频: {vf.relative_to(root)}")
            vf.unlink(missing_ok=True)

    # 5) 重写 info.json 计数
    info["total_episodes"] = persisted_eps
    info["total_frames"] = trunc_frames
    info["splits"] = {"train": f"0:{persisted_eps}"}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[repair] info.json 已重写: total_episodes={persisted_eps}, total_frames={trunc_frames}")

    # 6) 重算 stats.json (尾部被截断, 统计略偏; 失败可容忍)
    try:
        ds = LeRobotDataset(repo_id=repo_id, root=root)
        from lerobot.datasets import recompute_stats

        recompute_stats(ds)
        print("[repair] stats.json 已重算")
    except Exception as e:
        print(f"[repair] 警告: 重算 stats 失败 (可忽略): {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  键盘输入 (stdin 线程, 无头/有窗口统一)
# ═════════════════════════════════════════════════════════════════════════════

import queue
import select
import termios
import tty


class KeyboardReader(threading.Thread):
    """cbreak 模式读取 stdin 按键 -> queue."""

    def __init__(self, key_q: queue.Queue):
        super().__init__(daemon=True)
        self.key_q = key_q
        self._stop = threading.Event()
        self.saved = None
        try:
            self.saved = termios.tcgetattr(sys.stdin.fileno())
        except (termios.error, AttributeError):
            pass

    def run(self):
        if self.saved is None:
            print("[kb] stdin 不是终端, 按键控制不可用 (预览窗口内可按 q/ESC 退出)")
            return
        tty.setcbreak(sys.stdin.fileno())
        try:
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = os.read(sys.stdin.fileno(), 1).decode("utf-8", "ignore")
                    if ch:
                        self.key_q.put(ch)
        finally:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.saved)


# ═════════════════════════════════════════════════════════════════════════════
#  采集节点
# ═════════════════════════════════════════════════════════════════════════════

class CollectionNode:
    """订阅相机/关节话题, 按 v6 LeRobotDataset 写入数据集."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        dcfg = cfg["dataset"]
        rcfg = cfg["robot"]
        ccfg = cfg.get("camera", {})

        self.task_name = dcfg["task_name"]
        self.fps = int(dcfg["fps"])
        self.max_frames = int(dcfg.get("max_frames_per_episode", 10000))
        self.robot_type = rcfg["robot_type"]
        self.joint_source = rcfg.get("joint_source", "slave")  # slave | master
        self.arms = ROBOT_TYPE_ARMS[self.robot_type]
        self.root = Path(dcfg["target_dir"])
        self.repo_id = dcfg["repo_id"]
        self.enable_view = bool(ccfg.get("enable_view", True))
        self.height = int(ccfg.get("height", 480))
        self.width = int(ccfg.get("width", 640))

        # 编码器
        self.encoder = build_encoder(dcfg.get("video_codec", "h264_nvenc"))

        # 特征 + 数据集
        self.features = build_features(self.robot_type, self.height, self.width)
        self.dataset = self._open_dataset(dcfg)

        # 话题订阅
        self.latest = {}  # topic -> 最新消息
        for cam, topic in CAM_TOPICS.items():
            rospy.Subscriber(topic, ImageMsg, self._cb_image, callback_args=topic)
            self.latest[topic] = None
        self.joint_topics = [
            JOINT_TOPIC_TMPL.format(side=s, source=self.joint_source) for s in self.arms
        ]
        for topic in self.joint_topics:
            rospy.Subscriber(topic, JointState, self._cb_joint, callback_args=topic)
            self.latest[topic] = None

        # 状态
        self.key_q = queue.Queue()
        KeyboardReader(self.key_q).start()
        self.recording = False
        self.recorded = 0        # 当前 episode 已录帧数
        self.total_saved = 0     # 已保存 episode 数
        self.last_t = time.perf_counter()
        self.quit = False
        self._shutdown_done = False

    # ---------- 数据集打开 ----------

    def _open_dataset(self, dcfg: dict):
        resume = bool(dcfg.get("resume", False))
        if self.root.exists() and (self.root / "meta" / "info.json").exists() and resume:
            repair_dataset_consistency(self.root, self.repo_id)
            ds = LeRobotDataset.resume(
                repo_id=self.repo_id,
                root=self.root,
                rgb_encoder=self.encoder,
                streaming_encoding=bool(dcfg.get("streaming_encoding", True)),
                encoder_queue_maxsize=30,
            )
            print(f"▶ 已恢复数据集, 当前共 {ds.meta.total_episodes} 个 episode")
            return ds

        try:
            ds = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=self.features,
                root=self.root,
                robot_type=self.robot_type,
                use_videos=True,
                rgb_encoder=self.encoder,
                streaming_encoding=bool(dcfg.get("streaming_encoding", True)),
                encoder_queue_maxsize=30,
            )
            print(f"📦 LeRobot dataset created at {self.root}")
            return ds
        except FileExistsError:
            ans = input(f"[dataset] 目录 {self.root} 已存在, 覆盖并重建? [y/N] ").strip().lower()
            if ans != "y":
                print("用户拒绝覆盖, 退出。")
                sys.exit(0)
            shutil.rmtree(self.root)
            ds = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=self.features,
                root=self.root,
                robot_type=self.robot_type,
                use_videos=True,
                rgb_encoder=self.encoder,
                streaming_encoding=bool(dcfg.get("streaming_encoding", True)),
                encoder_queue_maxsize=30,
            )
            print(f"📦 LeRobot dataset recreated at {self.root}")
            return ds

    # ---------- 订阅回调 ----------

    def _cb_image(self, msg: ImageMsg, topic: str):
        self.latest[topic] = msg

    def _cb_joint(self, msg: JointState, topic: str):
        self.latest[topic] = msg

    # ---------- 帧组装 ----------

    def _to_rgb(self, msg: ImageMsg) -> np.ndarray | None:
        """sensor_msgs/Image -> RGB uint8 (H,W,3), 纯 numpy 实现。

        不用 cv_bridge: 其编译扩展 cv_bridge_boost 按 NumPy 1.x C API 编译,
        在本环境 (python 3.12 + numpy 2.x) 无法加载, 且会拖入系统 opencv
        依赖链引发 conda/system 库混用问题。相机输出 rgb8 / bgr8 / mono8,
        numpy 直接解析即可。
        """
        try:
            if msg.encoding in ("rgb8", "bgr8", "mono8"):
                h, w = msg.height, msg.width
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                if msg.encoding == "mono8":
                    arr = arr.reshape(h, w, 1).repeat(3, axis=2)
                else:
                    arr = arr.reshape(h, w, 3)
                    if msg.encoding == "bgr8":
                        arr = arr[..., ::-1]
                return np.ascontiguousarray(arr)
            rospy.logwarn_throttle(5.0, f"image convert failed: 不支持的编码 {msg.encoding}")
            return None
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"image convert failed: {e}")
            return None

    def build_frame(self) -> dict | None:
        images = {}
        for cam, topic in CAM_TOPICS.items():
            msg = self.latest[topic]
            if msg is None:
                return None
            rgb = self._to_rgb(msg)
            if rgb is None:
                return None
            if rgb.shape[:2] != (self.height, self.width):
                rospy.logwarn_throttle(
                    5.0, f"相机 {cam} 尺寸 {rgb.shape} != 特征尺寸 ({self.height},{self.width}), 跳过该帧"
                )
                return None
            images[cam] = rgb

        q_parts = []
        for topic in self.joint_topics:
            msg = self.latest[topic]
            if msg is None or msg.position is None or len(msg.position) < 7:
                return None
            q_parts.append(np.array(msg.position[:7], dtype=np.float32))
        q = np.concatenate(q_parts)

        return {
            "observation.state": q,
            "action": q,  # 遥操作采集: action == state
            **{f"observation.images.{cam}": img for cam, img in images.items()},
            "task": self.task_name,
        }

    # ---------- 按键处理 ----------

    def _drain_keys(self):
        while not self.key_q.empty():
            ch = self.key_q.get()
            if ch == "c":
                if not self.recording:
                    self.recording = True
                    self.last_t = time.perf_counter()
                    print(f"▶ 开始采集 (episode 帧 {self.recorded}/{self.max_frames})")
            elif ch == " ":
                if self.recording:
                    self.recording = False
                    print(f"⏸ 已暂停 (帧 {self.recorded}, 按 s 保存 / r 丢弃)")
            elif ch == "s":
                self.save_episode()
            elif ch == "r":
                if self.dataset.has_pending_frames():
                    self.dataset.clear_episode_buffer()
                    self.recorded = 0
                    print("🗑 已丢弃当前缓存帧")
                else:
                    print("  无缓存帧可丢弃")
            elif ch == "q":
                self.quit = True
                print("📦 收到退出指令")

    def save_episode(self):
        self.recording = False
        if not self.dataset.has_pending_frames():
            print("  无缓存帧, 未保存")
            return
        n = self.dataset.writer.episode_buffer["size"]
        self.dataset.save_episode()
        self.total_saved += 1
        self.recorded = 0
        print(f"💾 已保存 episode #{self.total_saved} ({n} 帧)")

    # ---------- 预览 ----------

    def show_preview(self):
        tiles = []
        for cam, topic in CAM_TOPICS.items():
            msg = self.latest[topic]
            if msg is None:
                tiles.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
                continue
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                tiles.append(cv2.resize(bgr, (self.width, self.height)))
            except Exception:
                tiles.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        view = np.hstack(tiles)
        status = "REC" if self.recording else "PAUSED"
        fps = self.recorded / max(time.perf_counter() - self.last_t, 1e-6) if self.recording else 0
        cv2.putText(
            view,
            f"{self.task_name} | {status} | ep#{self.total_saved} f{self.recorded} | {fps:.1f}fps",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if self.recording else (0, 165, 255),
            2,
        )
        cv2.imshow("collection_preview", view)
        cv2.waitKey(1)

    # ---------- 主循环 ----------

    def run(self):
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        try:
            while not self.quit and not rospy.is_shutdown():
                self._drain_keys()
                if self.quit:
                    break

                if self.recording:
                    frame = self.build_frame()
                    if frame is None:
                        rospy.logwarn_throttle(5.0, "等待话题数据 (相机/关节状态未就绪)...")
                    else:
                        self.dataset.add_frame(frame)
                        self.recorded += 1
                        if self.recorded >= self.max_frames:
                            self.recording = False
                            print(f"⏸ 已达单集最大帧数 {self.max_frames}, 按 s 保存 / r 丢弃")

                if self.enable_view:
                    self.show_preview()

                next_t += period
                precise_sleep(next_t - time.perf_counter())
        finally:
            self.shutdown()

    def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True
        print("📦 正在收尾 (finalize)...")
        try:
            if self.dataset is not None:
                if self.dataset.has_pending_frames():
                    self.dataset.save_episode()
                self.dataset.finalize()
            if self.enable_view:
                cv2.destroyAllWindows()
            print(
                f"✅ 完成: 共 {self.total_saved} 个 episode, 数据集位于 {self.root}\n"
                f"   总帧数: {self.dataset.meta.total_frames}"
            )
        except Exception as e:
            print(f"⚠️ finalize 异常: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  入口
# ═════════════════════════════════════════════════════════════════════════════

def find_config_path() -> str:
    """定位配置文件: _config_file 参数 -> 环境变量 -> 包内默认."""
    p = rospy.get_param("~config_file", "")
    if p:
        return p
    p = os.environ.get("COLLECTION_CONFIG", "")
    if p:
        return p
    try:
        import rospkg

        return str(Path(rospkg.RosPack().get_path("dual_piper")) / "config" / "collection.yaml")
    except Exception:
        return str(Path(__file__).resolve().parent.parent / "config" / "collection.yaml")


def main():
    rospy.init_node("data_to_lerobot3_node", anonymous=True)
    config_path = find_config_path()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    print(
        f"robot_type={cfg['robot']['robot_type']}: "
        f"arms={ROBOT_TYPE_ARMS[cfg['robot']['robot_type']]}, "
        f"cameras={list(CAM_TOPICS)}, task='{cfg['dataset']['task_name']}'"
    )

    node = CollectionNode(cfg)
    rospy.on_shutdown(node.shutdown)
    node.run()


if __name__ == "__main__":
    main()
