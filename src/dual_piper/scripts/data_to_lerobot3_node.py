#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot v3 Direct Data Collection Node

Subscribes to camera / joint-state / point-cloud topics, synchronises
frames, and writes each episode directly into a LeRobot v3.0 dataset
(Parquet + MP4), eliminating the intermediate HDF5 conversion step.

All parameters are managed via a YAML config file (config/collection.yaml).
ROS parameters (~key) can still override individual values.

Controls:
    c     - Start / resume collecting
    space - Pause collecting
    e     - Toggle error mode while collecting (flags frames back_event=1 → System-2 replan trigger)
    1-N   - Switch subtask (while paused)
    s     - Save current episode
    r     - Discard buffered frames
    q     - Finalize dataset & exit

Usage:
    rosrun dual_piper data_to_lerobot3_node.py
    rosrun dual_piper data_to_lerobot3_node.py _config_file:=/path/to/collection.yaml
"""
import os
import sys
import json
import signal
import atexit
import queue
import shutil
import threading
import collections
import time
import types
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import rospy
from tqdm import tqdm
from pynput import keyboard
from sensor_msgs.msg import Image, JointState, PointCloud2

from utils import imgmsg_to_cv2, sample_pointcloud, StatusView

# LeRobot v3.0
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_SUBTASKS_PATH


_key_queue: queue.Queue = queue.Queue()


def _on_key_press(e):
    """Keyboard listener callback: enqueue char or special-key name."""
    if hasattr(e, 'char') and e.char is not None:
        _key_queue.put(e.char)
    elif hasattr(e, 'name'):
        _key_queue.put(e.name)


def _load_yaml_config(yaml_path: str) -> dict:
    """Load and return the YAML configuration file as a nested dict."""
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f'Config file not found: {yaml_path}')
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f) or {}


def _confirm_overwrite(target_dir: str) -> bool:
    """Ask the user via stdin whether to overwrite an existing dataset.

    Returns True if the user confirms deletion, False otherwise.
    """
    print(f'\n{"=" * 60}')
    print(f'  WARNING: Dataset already exists at:')
    print(f'    {target_dir}')
    print(f'  resume is set to False.')
    print(f'{"=" * 60}')
    try:
        answer = input('  Overwrite existing dataset? [y/N]: ').strip().lower()
    except EOFError:
        answer = ''
    return answer in ('y', 'yes')


# ──────────────────────────── ROS Collector Node ──────────────────────────
class LeRobotCollectorNode:
    """
    ROS node that subscribes to camera / joint-state / point-cloud topics,
    streams synchronised frames live into a LeRobot v3.0 dataset (Parquet +
    MP4) as they are collected, so video encoding overlaps with recording and
    saving an episode only finalises the already-encoded data.
    """
    def __init__(self):
        # ── Load YAML Config ─────────────────────────────────────────────
        _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _default_config = os.path.join(_pkg_dir, 'config', 'collection.yaml')
        config_file = rospy.get_param('~config_file', _default_config)
        ycfg = _load_yaml_config(config_file)
        rospy.loginfo(f'Loaded config from {config_file}')

        ds   = ycfg.get('dataset', {})
        rob  = ycfg.get('robot', {})
        col  = ycfg.get('collection', {})
        cam  = ycfg.get('camera', {})
        pc   = ycfg.get('pointcloud', {})
        sub  = ycfg.get('subtask', {})
        top  = ycfg.get('topics', {})

        # ── Build cfg namespace (YAML defaults, ROS param overrides) ──────
        P = rospy.get_param  # shorthand
        self.cfg = cfg = types.SimpleNamespace(
            # dataset
            repo_id          = P('~repo_id',          ds.get('repo_id', 'user/DualPiper_Dataset')),
            target_dir       = P('~target_dir',       ds.get('target_dir', '/home/user/data')),
            task_name        = P('~task_name',        ds.get('task_name', 'default_task')),
            robot_type       = P('~robot_type',       ds.get('robot_type', 'dual_piper')),
            fps              = P('~fps',              ds.get('fps', 20)),
            max_timesteps    = P('~max_timesteps',    col.get('max_timesteps', 5000)),
            state_dim        = P('~state_dim',        rob.get('state_dim', 14)),
            action_dim       = P('~action_dim',       rob.get('action_dim', 14)),
            image_h          = P('~image_height',     cam.get('image_height', 180)),
            image_w          = P('~image_width',      cam.get('image_width', 320)),
            camera_names     = P('~camera_names',     cam.get('names', ['cam_top', 'cam_left', 'cam_right'])),
            use_depth        = P('~use_depth_image',  cam.get('use_depth', False)),
            save_pointcloud  = P('~save_pointcloud',  pc.get('enabled', False)),
            pc_target_points = P('~pc_target_points', pc.get('target_points', 5000)),
            use_subtask      = P('~use_subtask',      sub.get('enabled', True)),
            use_back_event   = P('~use_back_event',   col.get('use_back_event', True)),
            use_videos       = P('~use_videos',       cam.get('use_videos', True)),
            resume           = P('~resume',           ds.get('resume', True)),
            # Video encoding speed-ups:
            #   video_codec        — 'h264_nvenc'/'hevc_nvenc' use the GPU (NVENC) and are
            #                        ~3x faster than the software 'libsvtav1' default.
            #   streaming_encoding — encode frames in real time during collection instead of
            #                        writing PNGs to disk and re-encoding at save time. This
            #                        removes the disk round-trip and makes 's' near-instant.
            #   encoder_threads    — threads per software encoder (ignored for NVENC).
            video_codec        = P('~video_codec',        ds.get('video_codec', 'h264_nvenc')),
            streaming_encoding = P('~streaming_encoding', col.get('streaming_encoding', True)),
            encoder_threads    = P('~encoder_threads',    col.get('encoder_threads', 4)),
            # Live operator view (OpenCV window with camera feeds + status overlay)
            enable_view        = P('~enable_view',        col.get('enable_view', True)),
            # view_bgr=False: ROS/OpenCV images are already BGR — no conversion needed.
            # Set True only if your image source delivers RGB frames.
            view_bgr           = P('~view_bgr',           col.get('view_bgr', False)),
            view_max_width     = P('~view_max_width',     col.get('view_max_width', 1280)),

            img_top_topic              = P('~img_top_topic',              top.get('img_top', '/cam_top/cam_top/color/image_raw')),
            img_leftwrist_topic        = P('~img_leftwrist_topic',        top.get('img_left', '/cam_leftwrist/cam_leftwrist/color/image_raw')),
            img_rightwrist_topic       = P('~img_rightwrist_topic',       top.get('img_right', '/cam_rightwrist/cam_rightwrist/color/image_raw')),
            img_top_depth_topic        = P('~img_top_depth_topic',        top.get('img_top_depth', '/cam_top/cam_top/aligned_depth_to_color/image_raw')),
            img_leftwrist_depth_topic  = P('~img_leftwrist_depth_topic',  top.get('img_left_depth', '/cam_leftwrist/cam_leftwrist/aligned_depth_to_color/image_raw')),
            img_rightwrist_depth_topic = P('~img_rightwrist_depth_topic', top.get('img_right_depth', '/cam_rightwrist/cam_rightwrist/aligned_depth_to_color/image_raw')),
            left_master_topic          = P('~left_master_arm_topic',      top.get('left_master', 'left/master_joint_states')),
            left_slave_topic           = P('~left_slave_arm_topic',       top.get('left_slave', 'left/slave_joint_states')),
            right_master_topic         = P('~right_master_arm_topic',     top.get('right_master', 'right/master_joint_states')),
            right_slave_topic          = P('~right_slave_arm_topic',      top.get('right_slave', 'right/slave_joint_states')),
            pc_top_topic               = P('~pc_top_topic',               top.get('pc_top', '/cam_top/cam_top/depth/color/points')),
        )

        # ── Load Subtask Definitions from dataset config ─────────────────
        raw_subtasks = ds.get('subtasks', {})
        if cfg.use_subtask and not raw_subtasks:
            raise ValueError(
                "subtask.enabled is true but dataset.subtasks is empty. "
                "Add subtask definitions to collection.yaml under dataset.subtasks."
            )
        # Ensure keys are strings (YAML may parse '1' as int)
        self.subtasks = {str(k): v for k, v in raw_subtasks.items()}
        rospy.loginfo(f'Loaded {len(self.subtasks)} subtasks from config')
        # Build subtask_name → int index mapping for subtask_index feature
        names = list(self.subtasks.values())
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate subtask names found: {names}')
        self.subtask_name_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(names)
        }
        self._first_subtask_key = next(iter(self.subtasks))

        # ── Data Queues (topic → deque) ──────────────────────────────────
        self.queues: dict = {}
        self._init_queues()
        # ── ROS Subscribers ──────────────────────────────────────────────
        self._init_subscribers()
        # Frames are streamed straight into the LeRobot episode buffer during
        # collection (see _append_frame), so video encoding overlaps with
        # recording and 's' (save) only has to finalize the already-encoded
        # episode instead of processing every frame in one burst.
        self.collecting = False
        # Error-mode: when True, frames recorded during collection are flagged back_event=1
        # (states from which the robot should trigger a System-2 replan). Toggled with 'e' while
        # collecting; auto-reset on pause / save / discard / subtask-switch so it never leaks.
        self._error_mode = False
        self.current_subtask = self.subtasks[self._first_subtask_key]
        # ── LeRobot Dataset ──────────────────────────────────────────────
        self.dataset = None
        self._finalized = False
        self._create_dataset()
        # ── Live Operator View (OpenCV) ───────────────────────────────
        self._view = None
        if cfg.enable_view:
            self._view = StatusView(
                camera_names=cfg.camera_names,
                bgr=cfg.view_bgr,
                max_width=cfg.view_max_width,
                window_name='DualPiper Collect',
            ).start()
        # ── Shutdown Safety: ensure finalize runs on abnormal exit ─────
        rospy.on_shutdown(self._safe_finalize)
        atexit.register(self._safe_finalize)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        # ── Thread-Safe Keyboard State ───────────────────────────────────
        self._key_lock = threading.Lock()
        self._key_value = None
        for t in (
                threading.Thread(target=self._key_poll_loop, daemon=True),
                threading.Thread(target=self._command_loop, daemon=True),
        ):
            t.start()

        self._print_usage()

    # ─────────────────────── Thread-Safe Key Property ──────────────────
    @property
    def _key(self):
        with self._key_lock:
            return self._key_value

    @_key.setter
    def _key(self, val):
        with self._key_lock:
            self._key_value = val

    def _consume_key(self):
        """Atomically read and clear the current key."""
        with self._key_lock:
            val = self._key_value
            self._key_value = None
            return val

    # ─────────────────────── Dataset Creation ────────────────────────────
    def _create_dataset(self):
        """Create a new LeRobotDataset or resume an existing one."""
        cfg = self.cfg
        target = cfg.target_dir

        # ── Resume existing dataset ───────────────────────────────────
        if os.path.exists(target):
            if cfg.resume:
                rospy.loginfo(f'Resuming existing dataset at {target}')
                # Repair any corrupted parquet files left by a previous crash
                self._repair_corrupted_parquets(target)
                # Reuse the codec the dataset was originally created with so new
                # episodes stay consistent with the existing video files. We can
                # still enable streaming encoding (it works with any codec).
                existing_codec = self._get_local_video_codec(target, cfg.video_codec)
                if existing_codec != cfg.video_codec:
                    rospy.logwarn(
                        f'Existing dataset uses codec "{existing_codec}"; keeping it on '
                        f'resume (config requested "{cfg.video_codec}"). Start a fresh '
                        f'dataset to switch codecs.'
                    )
                self.dataset = LeRobotDataset(
                    repo_id=cfg.repo_id,
                    root=cfg.target_dir,
                    tolerance_s=1e-4,
                    # Use local revision to avoid contacting HuggingFace Hub
                    revision=self._get_local_codebase_version(target),
                    vcodec=existing_codec,
                    streaming_encoding=cfg.streaming_encoding,
                    encoder_threads=cfg.encoder_threads,
                )
                # Initialize recording-related attributes that __init__ doesn't set
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                # Streaming encoder feeds frames directly; the PNG image writer is
                # only needed for non-video (image) features.
                if not cfg.streaming_encoding:
                    self.dataset.start_image_writer(num_processes=0, num_threads=4)
                rospy.loginfo(
                    f'  codec={existing_codec}  streaming={cfg.streaming_encoding}  '
                    f'existing episodes={self.dataset.meta.total_episodes}  '
                    f'frames={self.dataset.meta.total_frames}'
                )
                # A resumed dataset cannot gain a new column mid-stream. If back_event was requested
                # but the existing dataset lacks it, disable it so add_frame doesn't fail on the extra
                # key — back-event collection requires a fresh dataset.
                if cfg.use_back_event and 'back_event' not in self.dataset.meta.features:
                    rospy.logwarn(
                        "use_back_event=True but the resumed dataset has no 'back_event' column; "
                        "disabling it for this run. Start a fresh dataset to collect back events."
                    )
                    cfg.use_back_event = False
                return

            # ── Confirm before overwriting ─────────────────────────────
            if not _confirm_overwrite(target):
                rospy.loginfo('User declined overwrite. Exiting.')
                rospy.signal_shutdown('User declined dataset overwrite')
                sys.exit(0)

            rospy.logwarn(f'Overwriting existing dataset: {target}')
            shutil.rmtree(target)

        # ── Build feature schema ──────────────────────────────────────
        features = {
            'observation.state': {
                'dtype': 'float32',
                'shape': (cfg.state_dim,),
                'names': [f'joint_{i+1}.pos' for i in range(cfg.state_dim)],
            },
            'action': {
                'dtype': 'float32',
                'shape': (cfg.action_dim,),
                'names': [f'joint_{i+1}.pos' for i in range(cfg.action_dim)],
            },
        }
        for cam in cfg.camera_names:
            features[f'observation.images.{cam}'] = {
                'dtype': 'video' if cfg.use_videos else 'image',
                'shape': (cfg.image_h, cfg.image_w, 3),
                'names': ['height', 'width', 'channel'],
            }
        if cfg.save_pointcloud:
            features['observation.pointcloud'] = {
                'dtype': 'float32',
                'shape': (cfg.pc_target_points, 6),
                'names': ['x', 'y', 'z', 'r', 'g', 'b'],
            }
        # Subtask as int64 index (aligned with LeRobot subtask_index system)
        if cfg.use_subtask:
            features['subtask_index'] = {
                'dtype': 'int64',
                'shape': (1,),
                'names': None,
            }
        # Per-frame back-event flag (0/1): 1 marks an erroneous, unrecoverable state that should
        # trigger a System-2 replan. Used as the back-head positive label by System 2, and as the
        # exclusion mask for System 1 (the VLA must not imitate these bad actions).
        if cfg.use_back_event:
            features['back_event'] = {
                'dtype': 'int64',
                'shape': (1,),
                'names': None,
            }

        self.dataset = LeRobotDataset.create(
            repo_id=cfg.repo_id,
            root=cfg.target_dir,
            fps=cfg.fps,
            robot_type=cfg.robot_type,
            features=features,
            use_videos=cfg.use_videos,
            tolerance_s=1e-4,
            image_writer_processes=0,
            image_writer_threads=4,
            vcodec=cfg.video_codec,
            streaming_encoding=cfg.streaming_encoding,
            encoder_threads=cfg.encoder_threads,
        )
        rospy.loginfo(
            f'Video encoding: codec={cfg.video_codec}  '
            f'streaming={cfg.streaming_encoding}'
        )

        # ── Write meta/subtasks.parquet ───────────────────────────────
        if cfg.use_subtask:
            self._write_subtasks_parquet()

        rospy.loginfo(f'LeRobot dataset created at {cfg.target_dir}')

    @staticmethod
    def _get_local_codebase_version(target_dir: str) -> str:
        """Read codebase_version from local info.json to avoid hub access."""
        info_path = os.path.join(target_dir, 'meta', 'info.json')
        if os.path.isfile(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            return info.get('codebase_version', 'v3.0')
        return 'v3.0'

    @staticmethod
    def _get_local_video_codec(target_dir: str, default: str) -> str:
        """Read the video codec an existing dataset was encoded with.

        LeRobot stores the codec under each video feature's ``info`` block as
        ``video.codec`` (e.g. 'av1', 'h264', 'hevc'). We map the stream codec
        name back to the ffmpeg encoder name so it can be reused on resume.
        Returns *default* if no codec can be determined.
        """
        info_path = os.path.join(target_dir, 'meta', 'info.json')
        if not os.path.isfile(info_path):
            return default
        with open(info_path, 'r') as f:
            info = json.load(f)
        # Stream-codec-name → ffmpeg-encoder-name
        codec_to_encoder = {'av1': 'libsvtav1', 'h264': 'h264_nvenc', 'hevc': 'hevc_nvenc'}
        for feat in info.get('features', {}).values():
            codec = feat.get('info', {}).get('video.codec')
            if codec:
                return codec_to_encoder.get(codec, codec)
        return default

    @staticmethod
    def _repair_corrupted_parquets(target_dir: str):
        """Remove parquet files left without footer by a previous crash.

        When the process is killed (e.g. ctrl+f2), the ParquetWriter may not
        have written the footer bytes.  PyArrow cannot read such files and will
        raise an ArrowInvalid error on the next load.  We detect and remove
        them here so that the remaining *valid* data can still be resumed.
        """
        import pyarrow.parquet as pq

        removed = []
        for pq_path in Path(target_dir).rglob('*.parquet'):
            try:
                pq.read_metadata(pq_path)
            except Exception:
                removed.append(str(pq_path))
                pq_path.unlink()

        if removed:
            rospy.logwarn(
                f'Removed {len(removed)} corrupted parquet file(s) from '
                f'previous crash:\n  ' + '\n  '.join(removed)
            )

            # Sync info.json counters with the surviving episode metadata,
            # since the corrupted file may have contained the latest episodes.
            info_path = os.path.join(target_dir, 'meta', 'info.json')
            episodes_dir = Path(target_dir) / 'meta' / 'episodes'
            if os.path.isfile(info_path) and episodes_dir.exists():
                surviving = sorted(episodes_dir.rglob('*.parquet'))
                if surviving:
                    total_eps = 0
                    total_frames = 0
                    for ep_pq in surviving:
                        tbl = pq.read_table(ep_pq)
                        total_eps += len(tbl)
                        if 'dataset_to_index' in tbl.column_names:
                            col = tbl.column('dataset_to_index').to_pylist()
                            if col:
                                total_frames = max(total_frames, max(col))
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    info['total_episodes'] = total_eps
                    info['total_frames'] = total_frames
                    info['splits'] = {'train': f'0:{total_eps}'}
                    with open(info_path, 'w') as f:
                        json.dump(info, f, indent=2)
                    rospy.loginfo(
                        f'Repaired info.json: total_episodes={total_eps}, '
                        f'total_frames={total_frames}'
                    )
                else:
                    # All episode metadata was corrupted — reset counters
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    info['total_episodes'] = 0
                    info['total_frames'] = 0
                    info['splits'] = {'train': '0:0'}
                    with open(info_path, 'w') as f:
                        json.dump(info, f, indent=2)
                    rospy.logwarn('All episode metadata lost — counters reset to 0.')

    def _write_subtasks_parquet(self):
        """Write subtask name → index mapping to meta/subtasks.parquet."""
        subtask_names = list(self.subtasks.values())
        df = pd.DataFrame(
            {'subtask_index': range(len(subtask_names))},
            index=pd.Index(subtask_names, name='subtask'),
        )
        parquet_path = Path(self.cfg.target_dir) / DEFAULT_SUBTASKS_PATH
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path)
        rospy.loginfo(f'Wrote {len(df)} subtasks to {parquet_path}')

    # ─────────────────────── Queue / Subscriber Init ─────────────────────
    def _init_queues(self):
        cfg = self.cfg
        for cam in cfg.camera_names:
            self.queues[cam] = collections.deque(maxlen=5)
            if cfg.use_depth:
                self.queues[f'{cam}_depth'] = collections.deque(maxlen=5)
        for arm in ('left/master', 'left/slave', 'right/master', 'right/slave'):
            self.queues[arm] = collections.deque(maxlen=50)
        if cfg.save_pointcloud:
            self.queues['pc_top'] = collections.deque(maxlen=10)

    def _init_subscribers(self):
        cfg = self.cfg
        Q = 50  # subscriber queue size
        # Camera image topics → queue mapping
        _cam_topic_map = {
            'cam_top':   cfg.img_top_topic,
            'cam_left':  cfg.img_leftwrist_topic,
            'cam_right': cfg.img_rightwrist_topic,
        }
        _cam_depth_topic_map = {
            'cam_top':   cfg.img_top_depth_topic,
            'cam_left':  cfg.img_leftwrist_depth_topic,
            'cam_right': cfg.img_rightwrist_depth_topic,
        }
        for cam in cfg.camera_names:
            if cam in _cam_topic_map:
                rospy.Subscriber(_cam_topic_map[cam], Image,
                                 lambda msg, c=cam: self.queues[c].append(msg), queue_size=Q)
            if cfg.use_depth and cam in _cam_depth_topic_map:
                rospy.Subscriber(_cam_depth_topic_map[cam], Image,
                                 lambda msg, c=cam: self.queues[f'{c}_depth'].append(msg), queue_size=Q)

        # Arm joint-state topics
        _arm_topic_map = {
            'left/master':  cfg.left_master_topic,
            'left/slave':   cfg.left_slave_topic,
            'right/master': cfg.right_master_topic,
            'right/slave':  cfg.right_slave_topic,
        }
        for arm, topic in _arm_topic_map.items():
            rospy.Subscriber(topic, JointState,
                             lambda msg, a=arm: self.queues[a].append(msg), queue_size=Q)

        # Point cloud
        if cfg.save_pointcloud:
            rospy.Subscriber(cfg.pc_top_topic, PointCloud2,
                             lambda msg: self.queues['pc_top'].append(msg), queue_size=10)

    # ─────────────────────── Frame Synchronisation ───────────────────────
    def _get_synced_frame(self):
        """
        Try to pop one time-synchronised frame from the queues.
        Returns dict with 'images', 'depths', 'pointcloud', 'left/master', ... or None.
        """
        cfg = self.cfg
        needed_keys = (list(cfg.camera_names)
                       + ['left/master', 'left/slave', 'right/master', 'right/slave'])
        if cfg.use_depth:
            needed_keys += [f'{c}_depth' for c in cfg.camera_names]

        # All queues must be non-empty
        if any(len(self.queues.get(k, [])) == 0 for k in needed_keys):
            return None

        # Sync time = latest among the oldest messages
        def _stamp(msg):
            return msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs

        sync_t = min(_stamp(self.queues[k][-1]) for k in needed_keys)

        # Drop older messages
        for k in needed_keys:
            while self.queues[k] and _stamp(self.queues[k][0]) < sync_t:
                self.queues[k].popleft()

        if any(len(self.queues.get(k, [])) == 0 for k in needed_keys):
            return None

        # Pop images
        images = {}
        for cam in cfg.camera_names:
            images[cam] = imgmsg_to_cv2(self.queues[cam].popleft())

        # Pop depth
        depths = {}
        if cfg.use_depth:
            for cam in cfg.camera_names:
                depths[cam] = imgmsg_to_cv2(self.queues[f'{cam}_depth'].popleft())

        # Pop point cloud  
        pointcloud = None
        if cfg.save_pointcloud and self.queues['pc_top']:
            pointcloud = sample_pointcloud(self.queues['pc_top'].popleft(), cfg.pc_target_points)

        return {
            'images': images,
            'depths': depths,
            'pointcloud': pointcloud,
            'left/master':  self.queues['left/master'].popleft(),
            'left/slave':   self.queues['left/slave'].popleft(),
            'right/master': self.queues['right/master'].popleft(),
            'right/slave':  self.queues['right/slave'].popleft(),
        }

    # ─────────────────────── Collection Loop ─────────────────────────────
    def _collect(self):
        """
        Stream synchronised frames straight into the LeRobot episode buffer
        until paused or max_timesteps reached.

        Feeding frames live (via _append_frame → dataset.add_frame) makes the
        streaming video encoder run *concurrently* with collection: each frame
        is encoded as it arrives instead of in one burst at save time. This
        also avoids encoder-queue overflow (feed_frame drops frames when the
        queue fills), and means 's' only finalises an already-encoded episode.
        """
        cfg = self.cfg
        rate = rospy.Rate(cfg.fps)
        count = self.dataset.episode_buffer['size']
        rospy.loginfo(f'▶ Collecting from frame {count} (max {cfg.max_timesteps})')
        rospy.loginfo(f'  Current subtask: {self.current_subtask}')

        pbar = tqdm(total=cfg.max_timesteps, desc='Collecting', ncols=80, initial=count)

        # FPS / elapsed tracking for the live view
        collect_start = time.time()
        meas_fps = float(cfg.fps)
        prev_loop_t = None

        while not rospy.is_shutdown() and count < cfg.max_timesteps:
            loop_t = time.time()
            if prev_loop_t is not None and (loop_t - prev_loop_t) > 0:
                meas_fps = 0.9 * meas_fps + 0.1 / (loop_t - prev_loop_t)
            prev_loop_t = loop_t

            # Handle in-collection keys (atomic read+clear): 'space' pauses, 'e' toggles error mode.
            k = self._consume_key()
            if k == 'space':
                rospy.loginfo(f'⏸ Paused at frame {count}')
                self._error_mode = False  # never leak error mode across a pause
                self.collecting = False
                pbar.close()
                return
            elif k == 'e' and cfg.use_back_event:
                self._error_mode = not self._error_mode
                rospy.logwarn(
                    f"{'🔴 ERROR-mode ON (back_event=1)' if self._error_mode else '🟢 error-mode OFF'} "
                    f"at frame {count}"
                )

            frame = self._get_synced_frame()
            if frame is None:
                rate.sleep()
                continue

            self._append_frame(frame)
            count += 1
            pbar.update(1)

            # Push the latest frame + collection state to the live view (non-blocking).
            if self._view is not None:
                self._view.update(frame['images'], {
                    'task': cfg.task_name,
                    'skill_text': self.current_subtask,
                    'recording': self.collecting,
                    'back': self._error_mode,
                    'step': count,
                    'elapsed': time.time() - collect_start,
                    'fps': meas_fps,
                    'episode_index': self.dataset.meta.total_episodes,
                    'buffered_frames': count,
                    'saved_episodes': self.dataset.meta.total_episodes,
                    'saved_frames': self.dataset.meta.total_frames,
                })

            rate.sleep()

        pbar.close()
        if count >= cfg.max_timesteps:
            rospy.loginfo(f'⏹ Reached max_timesteps ({cfg.max_timesteps})')
            self.collecting = False

    def _append_frame(self, frame):
        """Convert one synced frame and stream it into the episode buffer.

        Builds the LeRobot frame dict and calls ``dataset.add_frame`` so the
        per-frame cost (BGR→RGB conversion + handing the image to the streaming
        encoder) is paid during recording, at the natural FPS pace, rather than
        all at once when the episode is saved.
        """
        cfg = self.cfg
        ls = frame['left/slave']
        rs = frame['right/slave']
        lm = frame['left/master']
        rm = frame['right/master']

        # Joint state vectors (7 DOF per arm)
        qpos = np.concatenate([np.array(ls.position)[:7], np.array(rs.position)[:7]])
        action = np.concatenate([np.array(lm.position)[:7], np.array(rm.position)[:7]])

        lr_frame = {
            'observation.state': qpos.astype(np.float32),
            'action': action.astype(np.float32),
            'task': cfg.task_name,
        }
        for cam in cfg.camera_names:
            img = frame['images'].get(cam)
            if img is not None:
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                # Convert BGR (from OpenCV/ROS) → RGB (expected by LeRobot/PIL)
                lr_frame[f'observation.images.{cam}'] = img[:, :, ::-1].copy()

        if cfg.save_pointcloud and frame.get('pointcloud') is not None:
            lr_frame['observation.pointcloud'] = frame['pointcloud'].astype(np.float32)

        # Per-frame subtask_index — shape must be (1,) to match feature schema
        if cfg.use_subtask:
            lr_frame['subtask_index'] = np.array(
                [self.subtask_name_to_idx[self.current_subtask]], dtype=np.int64
            )
        # Per-frame back-event flag — 1 while error mode is on, else 0.
        if cfg.use_back_event:
            lr_frame['back_event'] = np.array(
                [1 if self._error_mode else 0], dtype=np.int64
            )

        self.dataset.add_frame(lr_frame)

    # ─────────────────────── Save Episode ────────────────────────────────
    def _save_episode(self):
        """
        Finalise the in-progress episode.

        Frames were already streamed into the LeRobot episode buffer (and the
        video encoder) live during collection by _append_frame, so this only
        has to flush the encoder and write the episode's parquet/metadata —
        making 's' near-instant regardless of episode length.

        Task/subtask strategy (aligned with LeRobot v3):
          - task:  high-level task name (cfg.task_name), same for all frames.
                   LeRobot converts this to task_index automatically.
          - subtask_index:  per-frame int64 index into meta/subtasks.parquet,
                   allowing fine-grained step tracking within an episode.
        """
        n = self.dataset.episode_buffer['size']
        if n == 0:
            rospy.logwarn('Nothing to save — episode buffer is empty.')
            return

        if n < 10:
            rospy.logwarn(f'Episode has only {n} frames — consider discarding (r).')

        rospy.loginfo(f'💾 Finalizing episode with {n} frames …')

        self.dataset.save_episode()
        # Flush metadata buffer to disk immediately so that episode files
        # are written after every save, not only every N episodes.
        # This also makes crash-resume reliable (no buffered-only episodes lost).
        self.dataset.meta._flush_metadata_buffer()

        ep_idx = self.dataset.meta.total_episodes - 1
        rospy.loginfo(
            f'✅ Episode {ep_idx} saved  |  total_episodes={self.dataset.meta.total_episodes}  '
            f'total_frames={self.dataset.meta.total_frames}'
        )
        # save_episode() already resets the episode buffer; just reset the subtask + error mode.
        self.current_subtask = self.subtasks[self._first_subtask_key]
        self._error_mode = False

    # ─────────────────────── Keyboard & Command Loop ─────────────────────
    def _key_poll_loop(self):
        """Start pynput listener and relay events to self._key."""
        listener = keyboard.Listener(on_press=_on_key_press)
        listener.start()
        while not rospy.is_shutdown():
            try:
                self._key = _key_queue.get(timeout=0.1)
            except queue.Empty:
                continue

    def _command_loop(self):
        """Main state-machine driven by keyboard commands."""
        while not rospy.is_shutdown():
            k = self._consume_key()
            if k is None:
                rospy.sleep(0.05)
                continue

            if k == 'c' and not self.collecting:
                self.collecting = True
                self._collect()
            elif k == 's' and not self.collecting:
                self._save_episode()
            elif k == 'r':
                # Cancel the streaming encoder + drop partial video/images and
                # reset the episode buffer (no-op if nothing buffered yet).
                if self.dataset.episode_buffer['size'] > 0:
                    self.dataset.clear_episode_buffer()
                self.current_subtask = self.subtasks[self._first_subtask_key]
                self._error_mode = False
                self.collecting = False
                rospy.loginfo('🗑 Buffer cleared, subtask + error mode reset.')
            elif k in self.subtasks and not self.collecting:
                self.current_subtask = self.subtasks[k]
                self._error_mode = False
                rospy.loginfo(f'🔄 Subtask → {self.current_subtask}')
            elif k == 'q':
                self._finalize_and_exit()
                return

            rospy.sleep(0.05)

    # ─────────────────────── Finalize ────────────────────────────────────
    def _safe_finalize(self):
        """Idempotent finalize — safe to call from multiple shutdown hooks."""
        if self._finalized:
            return
        self._finalized = True
        if self._view is not None:
            self._view.stop()
            self._view = None
        try:
            rospy.loginfo('📦 Finalizing dataset …')
            if self.dataset is not None:
                if self.dataset.image_writer is not None:
                    self.dataset.stop_image_writer()
                self.dataset.finalize()
            rospy.loginfo(f'✅ Dataset finalized at {self.cfg.target_dir}')
        except Exception as e:
            # Last resort: at least try to print the error
            sys.stderr.write(f'[data_to_lerobot3] finalize error: {e}\n')

    def _signal_handler(self, signum, _frame):
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        self._safe_finalize()
        rospy.signal_shutdown(f'Received signal {signum}')

    def _finalize_and_exit(self):
        """Stop image writers, close parquet writers, write final metadata."""
        self._safe_finalize()
        rospy.signal_shutdown('User requested exit')

    # ─────────────────────── Usage Banner ────────────────────────────────
    def _print_usage(self):
        lines = [
            '═══════════════════════════════════════════',
            '  LeRobot v3 Direct Collection Node',
            '═══════════════════════════════════════════',
            '  c     → Start / resume collecting',
            '  space → Pause',
        ]
        if self.cfg.use_back_event:
            lines.append('  e     → Toggle error mode (flag back_event=1) while collecting')
        for key, desc in self.subtasks.items():
            lines.append(f'  {key}     → Subtask: {desc}')
        lines += [
            '  s     → Save episode',
            '  r     → Discard buffer',
            '  q     → Finalize dataset & exit',
            '═══════════════════════════════════════════',
        ]
        rospy.loginfo('\033[32m' + '\n'.join(lines) + '\033[0m')

    # ─────────────────────── Main-Thread Spin ────────────────────────────
    def spin(self):
        """Block the main thread until shutdown.

        When the live view is enabled we drive its rendering here, because
        OpenCV's GUI must be pumped from the main thread (a background GUI
        thread silently fails to display under Anaconda's Qt). Subscriber and
        command threads keep running independently. Falls back to plain
        rospy.spin() when the view is disabled/unavailable.
        """
        if self._view is None or not self._view.enabled:
            rospy.spin()
            return
        rate = rospy.Rate(15)  # GUI refresh rate
        while not rospy.is_shutdown():
            # While NOT collecting, keep the view live with the latest camera
            # frames + current subtask so the operator can frame the scene and
            # pick a subtask before pressing 'c'. During collection, _collect()
            # already feeds the view with the synced frames it records, so we
            # leave it alone to avoid overwriting the recording status.
            if not self.collecting:
                self._view.update(self._peek_latest_images(), self._build_idle_status())
            if not self._view.render_once():  # user pressed q/ESC in the window
                self._finalize_and_exit()
                break
            rate.sleep()

    # ─────────────────────── Live-View Helpers ───────────────────────────
    def _peek_latest_images(self):
        """Latest frame from each camera queue for live preview (non-destructive).

        Reads (without popping) the newest ROS image in each camera deque and
        converts it to a cv2 array, so the preview never consumes frames the
        collector needs. Returns an empty dict until cameras start publishing.
        """
        images = {}
        for cam in self.cfg.camera_names:
            q = self.queues.get(cam)
            if q:
                try:
                    images[cam] = imgmsg_to_cv2(q[-1])
                except Exception:
                    pass
        return images

    def _build_idle_status(self):
        """Status overlay shown when not actively collecting (recording=False)."""
        buffered = self.dataset.episode_buffer['size'] if self.dataset is not None else 0
        total_eps = self.dataset.meta.total_episodes if self.dataset is not None else 0
        total_frames = self.dataset.meta.total_frames if self.dataset is not None else 0
        return {
            'task': self.cfg.task_name,
            'skill_text': self.current_subtask,
            'recording': False,
            'back': self._error_mode,
            'episode_index': total_eps,
            'buffered_frames': buffered,
            'saved_episodes': total_eps,
            'saved_frames': total_frames,
        }


def main():
    try:
        rospy.init_node('data_to_lerobot3_node', anonymous=True)
        node = LeRobotCollectorNode()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down LeRobot collection node.')


if __name__ == '__main__':
    main()