#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot v3 Direct Data Collection Node

Subscribes to camera / joint-state / point-cloud topics, synchronises
frames, and writes each episode directly into a LeRobot v3.0 dataset
(Parquet + MP4), eliminating the intermediate HDF5 conversion step.

Controls:
    c     - Start / resume collecting
    space - Pause collecting
    1-N   - Switch subtask (while paused)
    s     - Save current episode
    r     - Discard buffered frames
    q     - Finalize dataset & exit

Usage:
    rosrun dual_piper data_to_lerobot3_node.py
"""
import os
import copy
import queue
import shutil
import threading
import collections
import types

import numpy as np
import rospy
from tqdm import tqdm
from pynput import keyboard
from sensor_msgs.msg import Image, JointState, PointCloud2

from utils import imgmsg_to_cv2, sample_pointcloud

# LeRobot v3.0
from lerobot.datasets.lerobot_dataset import LeRobotDataset


_key_queue: queue.Queue = queue.Queue()


def _on_key_press(e):
    """Keyboard listener callback: enqueue char or special-key name."""
    if hasattr(e, 'char') and e.char is not None:
        _key_queue.put(e.char)
    elif hasattr(e, 'name'):
        _key_queue.put(e.name)


# ──────────────────────────── ROS Collector Node ──────────────────────────
class LeRobotCollectorNode:
    """
    ROS node that subscribes to camera / joint-state / point-cloud topics,
    buffers synchronised frames, and writes each episode directly into a
    LeRobot v3.0 dataset (Parquet + MP4).
    """
    # ── Subtask Definitions ──────────────────────────────────────────────
    SUBTASKS = {
        '1': 'pick up beakers from the table.',
        '2': 'pour the liquid from the left beaker into the right beaker.',
        '3': 'place the beakers to their initial position.',
        '4': 'return to the initial position.',
    }

    def __init__(self):
        # ── ROS Parameters ───────────────────────────────────────────────
        P = rospy.get_param  # shorthand
        self.cfg = cfg = types.SimpleNamespace(
            # dataset
            repo_id          = P('~repo_id',          'user/DualPiper_Dataset'),
            target_dir       = P('~target_dir',       '/home/user/hdd_4t/data/data_lerobot3/dualpiper/demo_dataset'),
            task_name        = P('~task_name',        'dual_arm_manipulation'),
            robot_type       = P('~robot_type',       'dual_piper'),
            fps              = P('~fps',              20),
            max_timesteps    = P('~max_timesteps',    5000),
            state_dim        = P('~state_dim',        14),
            action_dim       = P('~action_dim',       14),
            image_h          = P('~image_height',     180),
            image_w          = P('~image_width',      320),
            camera_names     = P('~camera_names',     ['cam_top', 'cam_left', 'cam_right']),
            use_depth        = P('~use_depth_image',  False),
            save_pointcloud  = P('~save_pointcloud',  False),
            pc_target_points = P('~pc_target_points', 5000),
            use_subtask      = P('~use_subtask',      True),
            use_videos       = P('~use_videos',       True),
            # ROS topics
            img_top_topic              = P('~img_top_topic',              '/cam_top/cam_top/color/image_raw'),
            img_leftwrist_topic        = P('~img_leftwrist_topic',        '/cam_leftwrist/cam_leftwrist/color/image_raw'),
            img_rightwrist_topic       = P('~img_rightwrist_topic',       '/cam_rightwrist/cam_rightwrist/color/image_raw'),
            img_top_depth_topic        = P('~img_top_depth_topic',        '/cam_top/cam_top/aligned_depth_to_color/image_raw'),
            img_leftwrist_depth_topic  = P('~img_leftwrist_depth_topic',  '/cam_leftwrist/cam_leftwrist/aligned_depth_to_color/image_raw'),
            img_rightwrist_depth_topic = P('~img_rightwrist_depth_topic', '/cam_rightwrist/cam_rightwrist/aligned_depth_to_color/image_raw'),
            left_master_topic          = P('~left_master_arm_topic',      'left/master_joint_states'),
            left_slave_topic           = P('~left_slave_arm_topic',       'left/slave_joint_states'),
            right_master_topic         = P('~right_master_arm_topic',     'right/master_joint_states'),
            right_slave_topic          = P('~right_slave_arm_topic',      'right/slave_joint_states'),
            pc_top_topic               = P('~pc_top_topic',               '/cam_top/cam_top/depth/color/points'),
        )

        # ── Data Queues (topic → deque) ──────────────────────────────────
        self.queues: dict = {}
        self._init_queues()
        # ── ROS Subscribers ──────────────────────────────────────────────
        self._init_subscribers()
        # ── Frame Buffer (per-episode, before save_episode) ──────────────
        self.frame_buffer: list = []        # list of {obs, action} dicts
        self.collecting = False
        self.current_subtask = self.SUBTASKS['1']
        # ── LeRobot Dataset (created once) ───────────────────────────────
        self.dataset = None
        self._create_dataset()
        # ── Keyboard Threads ─────────────────────────────────────────────
        self._key = None
        for t in (
                threading.Thread(target=self._key_poll_loop, daemon=True),
                threading.Thread(target=self._command_loop, daemon=True),
        ):
            t.start()

        self._print_usage()

    # ─────────────────────── Dataset Creation ────────────────────────────
    def _create_dataset(self):
        """Build a fresh LeRobotDataset with the correct feature schema."""
        cfg = self.cfg
        target = cfg.target_dir

        if os.path.exists(target):
            rospy.logwarn(f'Target dir already exists: {target}  — removing it.')
            shutil.rmtree(target)

        features = {
            'observation.state': {
                'dtype': 'float32',
                'shape': (cfg.state_dim,),
                'names': [f'joint_{i}.pos' for i in range(cfg.state_dim)],
            },
            'action': {
                'dtype': 'float32',
                'shape': (cfg.action_dim,),
                'names': [f'joint_{i}.pos' for i in range(cfg.action_dim)],
            },
        }
        # Camera image features
        for cam in cfg.camera_names:
            features[f'observation.images.{cam}'] = {
                'dtype': 'video' if cfg.use_videos else 'image',
                'shape': (cfg.image_h, cfg.image_w, 3),
                'names': ['height', 'width', 'channel'],
            }
        # Point cloud feature (optional)
        if cfg.save_pointcloud:
            features['observation.pointcloud'] = {
                'dtype': 'float32',
                'shape': (cfg.pc_target_points, 6),
                'names': ['x', 'y', 'z', 'r', 'g', 'b'],
            }
        # Subtask feature (optional)
        if cfg.use_subtask:
            features['subtask'] = {
                'dtype': 'string',
                'shape': (1,),
                'names': ['subtask'],
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
        )
        rospy.loginfo(f'LeRobot dataset created at {cfg.target_dir}')

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
        Collect frames into self.frame_buffer until paused or max_timesteps reached.
        Each buffer entry = {qpos, action, images, subtask} dict.
        """
        cfg = self.cfg
        rate = rospy.Rate(cfg.fps)
        count = len(self.frame_buffer)
        rospy.loginfo(f'▶ Collecting from frame {count} (max {cfg.max_timesteps})')
        rospy.loginfo(f'  Current subtask: {self.current_subtask}')

        pbar = tqdm(total=cfg.max_timesteps, desc='Collecting', ncols=80, initial=count)

        while not rospy.is_shutdown() and count < cfg.max_timesteps:
            # Check pause
            if self._key == 'space':
                rospy.loginfo(f'⏸ Paused at frame {count}')
                self.collecting = False
                self._key = None
                pbar.close()
                return

            frame = self._get_synced_frame()
            if frame is None:
                rate.sleep()
                continue

            ls = frame['left/slave']
            rs = frame['right/slave']
            lm = frame['left/master']
            rm = frame['right/master']

            # Joint state vectors (7 DOF per arm)
            qpos = np.concatenate([np.array(ls.position)[:7], np.array(rs.position)[:7]])
            action = np.concatenate([np.array(lm.position)[:7], np.array(rm.position)[:7]])

            # Build the buffered frame dict
            entry = {
                'qpos': qpos.astype(np.float32),
                'action': action.astype(np.float32),
                'images': {cam: img.copy() for cam, img in frame['images'].items()},
                'subtask': self.current_subtask,
            }
            if cfg.save_pointcloud and frame['pointcloud'] is not None:
                entry['pointcloud'] = frame['pointcloud']

            self.frame_buffer.append(entry)
            count += 1
            pbar.update(1)
            rate.sleep()

        pbar.close()
        if count >= cfg.max_timesteps:
            rospy.loginfo(f'⏹ Reached max_timesteps ({cfg.max_timesteps})')
            self.collecting = False

    # ─────────────────────── Save Episode ────────────────────────────────
    def _save_episode(self):
        """
        Flush self.frame_buffer into the LeRobot dataset as one episode.
        Uses dataset.add_frame() + dataset.save_episode().
        """
        if not self.frame_buffer:
            rospy.logwarn('Nothing to save — frame buffer is empty.')
            return

        cfg = self.cfg
        n = len(self.frame_buffer)
        rospy.loginfo(f'💾 Saving episode with {n} frames …')

        # Determine the task text (most-frequent subtask label)
        subtask_labels = [f['subtask'] for f in self.frame_buffer]
        task_text = max(set(subtask_labels), key=subtask_labels.count)

        for i, entry in enumerate(self.frame_buffer):
            frame = {
                'observation.state': entry['qpos'],
                'action': entry['action'],
                'task': task_text,
            }
            for cam in cfg.camera_names:
                img = entry['images'].get(cam)
                if img is not None:
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    frame[f'observation.images.{cam}'] = img

            # Point cloud
            if cfg.save_pointcloud and 'pointcloud' in entry:
                frame['observation.pointcloud'] = entry['pointcloud'].astype(np.float32)

            # Subtask
            if cfg.use_subtask:
                frame['subtask'] = entry['subtask']

            self.dataset.add_frame(frame)

        self.dataset.save_episode()

        ep_idx = self.dataset.meta.total_episodes - 1
        rospy.loginfo(
            f'✅ Episode {ep_idx} saved  |  total_episodes={self.dataset.meta.total_episodes}  '
            f'total_frames={self.dataset.meta.total_frames}'
        )
        # Clear buffer and reset subtask
        self.frame_buffer.clear()
        self.current_subtask = self.SUBTASKS['1']

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
            k = self._key
            if k == 'c' and not self.collecting:
                self.collecting = True
                self._key = None
                self._collect()

            elif k == 's' and not self.collecting:
                self._key = None
                self._save_episode()

            elif k == 'r':
                self._key = None
                self.frame_buffer.clear()
                self.current_subtask = self.SUBTASKS['1']
                self.collecting = False
                rospy.loginfo('🗑 Buffer cleared, subtask reset.')

            elif k in self.SUBTASKS and not self.collecting:
                self.current_subtask = self.SUBTASKS[k]
                rospy.loginfo(f'🔄 Subtask → {self.current_subtask}')
                self._key = None

            elif k == 'q':
                self._key = None
                self._finalize_and_exit()
                return

            rospy.sleep(0.05)

    # ─────────────────────── Finalize ────────────────────────────────────
    def _finalize_and_exit(self):
        """Stop image writers, close parquet writers, write final metadata."""
        rospy.loginfo('📦 Finalizing dataset …')
        if self.dataset is not None:
            if self.dataset.image_writer is not None:
                self.dataset.stop_image_writer()
            self.dataset.finalize()
        rospy.loginfo(f'✅ Dataset finalized at {self.cfg.target_dir}')
        rospy.signal_shutdown('User requested exit')

    # ─────────────────────── Usage Banner ────────────────────────────────
    @staticmethod
    def _print_usage():
        rospy.loginfo('\033[32m'
                      '═══════════════════════════════════════════\n'
                      '  LeRobot v3 Direct Collection Node\n'
                      '═══════════════════════════════════════════\n'
                      '  c     → Start / resume collecting\n'
                      '  space → Pause\n'
                      '  1-4   → Switch subtask (while paused)\n'
                      '  s     → Save episode\n'
                      '  r     → Discard buffer\n'
                      '  q     → Finalize dataset & exit\n'
                      '═══════════════════════════════════════════'
                      '\033[0m')


def main():
    try:
        rospy.init_node('data_to_lerobot3_node', anonymous=True)
        node = LeRobotCollectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down LeRobot collection node.')


if __name__ == '__main__':
    main()
