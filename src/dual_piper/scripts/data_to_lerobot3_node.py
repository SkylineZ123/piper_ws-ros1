#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot v3 Direct Data Collection Node

Subscribes to camera / joint-state / point-cloud topics, synchronises
frames, and writes each episode directly into a LeRobot v3.0 dataset
(Parquet + MP4), eliminating the intermediate HDF5 conversion step.

Controls (keyboard)::

    c     — start / resume collecting
    space — pause collecting
    1-N   — switch subtask (while paused)
    s     — save current episode
    r     — discard buffered frames
    q     — finalize dataset & exit

Usage::

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
from pynput import keyboard
from sensor_msgs.msg import Image, JointState, PointCloud2

from utils import imgmsg_to_cv2, sample_pointcloud

# LeRobot v3.0
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── Global keyboard queue ────────────────────────────────────────────────

_key_queue: queue.Queue = queue.Queue()


def _on_key_press(event):
    """Enqueue character or special-key name."""
    if hasattr(event, 'char') and event.char is not None:
        _key_queue.put(event.char)
    elif hasattr(event, 'name'):
        _key_queue.put(event.name)


class LeRobotCollectorNode:
    """
    ROS node that collects dual-arm data and writes LeRobot v3 datasets.

    Subtask definitions are configurable via ROS parameters.
    """

    SUBTASKS = {
        '1': 'pick up beakers from the table.',
        '2': 'pour the liquid from the left beaker into the right beaker.',
        '3': 'place the beakers to their initial position.',
        '4': 'return to the initial position.',
    }

    def __init__(self):
        # ── ROS Parameters ───────────────────────────────────────────────
        P = rospy.get_param
        self.cfg = cfg = types.SimpleNamespace(
            repo_id=P('~repo_id', 'user/DualPiper_Dataset'),
            target_dir=P('~target_dir',
                         '/home/user/hdd_4t/data/data_lerobot3/dualpiper/demo_dataset'),
            task_name=P('~task_name', 'dual_arm_manipulation'),
            robot_type=P('~robot_type', 'dual_piper'),
            fps=P('~fps', 20),
            max_timesteps=P('~max_timesteps', 5000),
            state_dim=P('~state_dim', 14),
            action_dim=P('~action_dim', 14),
            image_h=P('~image_height', 180),
            image_w=P('~image_width', 320),
            camera_names=P('~camera_names',
                           ['cam_top', 'cam_left', 'cam_right']),
            use_depth=P('~use_depth_image', False),
            save_pointcloud=P('~save_pointcloud', False),
            pc_target_points=P('~pc_target_points', 5000),
            use_subtask=P('~use_subtask', True),
            use_videos=P('~use_videos', True),
            # ROS topics
            img_top_topic=P('~img_top_topic',
                            '/cam_top/cam_top/color/image_raw'),
            img_leftwrist_topic=P('~img_leftwrist_topic',
                                  '/cam_leftwrist/cam_leftwrist/color/image_raw'),
            img_rightwrist_topic=P('~img_rightwrist_topic',
                                   '/cam_rightwrist/cam_rightwrist/color/image_raw'),
            img_top_depth_topic=P('~img_top_depth_topic',
                                  '/cam_top/cam_top/aligned_depth_to_color/image_raw'),
            img_leftwrist_depth_topic=P('~img_leftwrist_depth_topic',
                                        '/cam_leftwrist/cam_leftwrist/aligned_depth_to_color/image_raw'),
            img_rightwrist_depth_topic=P('~img_rightwrist_depth_topic',
                                         '/cam_rightwrist/cam_rightwrist/aligned_depth_to_color/image_raw'),
            left_master_topic=P('~left_master_arm_topic',
                                'left/master_joint_states'),
            left_slave_topic=P('~left_slave_arm_topic',
                               'left/slave_joint_states'),
            right_master_topic=P('~right_master_arm_topic',
                                 'right/master_joint_states'),
            right_slave_topic=P('~right_slave_arm_topic',
                                'right/slave_joint_states'),
            pc_top_topic=P('~pc_top_topic',
                           '/cam_top/cam_top/depth/color/points'),
        )

        # ── Data Queues ──────────────────────────────────────────────────
        self.queues = {}
        self._init_queues()
        self._init_subscribers()

        # ── Frame Buffer ─────────────────────────────────────────────────
        self.frame_buffer = []
        self.collecting = False
        self.current_subtask = self.SUBTASKS['1']

        # ── LeRobot Dataset ──────────────────────────────────────────────
        self.dataset = None
        self._create_dataset()

        # ── Keyboard ─────────────────────────────────────────────────────
        self._key = None
        for t in (
            threading.Thread(target=self._key_poll_loop, daemon=True),
            threading.Thread(target=self._command_loop, daemon=True),
        ):
            t.start()

        self._print_usage()

    # ── Dataset Creation ─────────────────────────────────────────────────

    def _create_dataset(self):
        cfg = self.cfg
        if os.path.exists(cfg.target_dir):
            rospy.logwarn("Removing existing dir: %s", cfg.target_dir)
            shutil.rmtree(cfg.target_dir)

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
        rospy.loginfo("LeRobot dataset created at %s", cfg.target_dir)

    # ── Queue / Subscriber Init ──────────────────────────────────────────

    def _init_queues(self):
        cfg = self.cfg
        for cam in cfg.camera_names:
            self.queues[cam] = collections.deque(maxlen=5)
            if cfg.use_depth:
                self.queues[f'{cam}_depth'] = collections.deque(maxlen=5)
        for arm in ('left/master', 'left/slave',
                     'right/master', 'right/slave'):
            self.queues[arm] = collections.deque(maxlen=50)
        if cfg.save_pointcloud:
            self.queues['pc_top'] = collections.deque(maxlen=10)

    def _init_subscribers(self):
        cfg = self.cfg
        Q = 50

        _cam_topics = {
            'cam_top': cfg.img_top_topic,
            'cam_left': cfg.img_leftwrist_topic,
            'cam_right': cfg.img_rightwrist_topic,
        }
        _cam_depth_topics = {
            'cam_top': cfg.img_top_depth_topic,
            'cam_left': cfg.img_leftwrist_depth_topic,
            'cam_right': cfg.img_rightwrist_depth_topic,
        }
        for cam in cfg.camera_names:
            if cam in _cam_topics:
                rospy.Subscriber(
                    _cam_topics[cam], Image,
                    lambda msg, c=cam: self.queues[c].append(msg),
                    queue_size=Q)
            if cfg.use_depth and cam in _cam_depth_topics:
                rospy.Subscriber(
                    _cam_depth_topics[cam], Image,
                    lambda msg, c=cam: self.queues[f'{c}_depth'].append(msg),
                    queue_size=Q)

        _arm_topics = {
            'left/master': cfg.left_master_topic,
            'left/slave': cfg.left_slave_topic,
            'right/master': cfg.right_master_topic,
            'right/slave': cfg.right_slave_topic,
        }
        for arm, topic in _arm_topics.items():
            rospy.Subscriber(
                topic, JointState,
                lambda msg, a=arm: self.queues[a].append(msg),
                queue_size=Q)

        if cfg.save_pointcloud:
            rospy.Subscriber(
                cfg.pc_top_topic, PointCloud2,
                lambda msg: self.queues['pc_top'].append(msg),
                queue_size=10)

    # ── Frame Synchronisation ────────────────────────────────────────────

    def _get_synced_frame(self):
        """Pop one time-synchronised frame from queues."""
        cfg = self.cfg
        needed = (list(cfg.camera_names)
                  + ['left/master', 'left/slave',
                     'right/master', 'right/slave'])
        if cfg.use_depth:
            needed += [f'{c}_depth' for c in cfg.camera_names]

        if any(len(self.queues.get(k, [])) == 0 for k in needed):
            return None

        def _stamp(msg):
            return msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs

        sync_t = min(_stamp(self.queues[k][-1]) for k in needed)
        for k in needed:
            while self.queues[k] and _stamp(self.queues[k][0]) < sync_t:
                self.queues[k].popleft()

        if any(len(self.queues.get(k, [])) == 0 for k in needed):
            return None

        images = {}
        for cam in cfg.camera_names:
            images[cam] = imgmsg_to_cv2(self.queues[cam].popleft())

        depths = {}
        if cfg.use_depth:
            for cam in cfg.camera_names:
                depths[cam] = imgmsg_to_cv2(
                    self.queues[f'{cam}_depth'].popleft())

        pointcloud = None
        if cfg.save_pointcloud and self.queues['pc_top']:
            pointcloud = sample_pointcloud(
                self.queues['pc_top'].popleft(), cfg.pc_target_points)

        return {
            'images': images,
            'depths': depths,
            'pointcloud': pointcloud,
            'left/master': self.queues['left/master'].popleft(),
            'left/slave': self.queues['left/slave'].popleft(),
            'right/master': self.queues['right/master'].popleft(),
            'right/slave': self.queues['right/slave'].popleft(),
        }

    # ── Collection ───────────────────────────────────────────────────────

    def _collect(self):
        cfg = self.cfg
        rate = rospy.Rate(cfg.fps)
        count = len(self.frame_buffer)
        rospy.loginfo("Collecting from frame %d (max %d)",
                      count, cfg.max_timesteps)
        rospy.loginfo("  Subtask: %s", self.current_subtask)

        while not rospy.is_shutdown() and count < cfg.max_timesteps:
            if self._key == 'space':
                rospy.loginfo("Paused at frame %d", count)
                self.collecting = False
                self._key = None
                return

            frame = self._get_synced_frame()
            if frame is None:
                rate.sleep()
                continue

            ls = frame['left/slave']
            rs = frame['right/slave']
            lm = frame['left/master']
            rm = frame['right/master']

            qpos = np.concatenate([
                np.array(ls.position)[:7],
                np.array(rs.position)[:7],
            ]).astype(np.float32)
            action = np.concatenate([
                np.array(lm.position)[:7],
                np.array(rm.position)[:7],
            ]).astype(np.float32)

            entry = {
                'qpos': qpos,
                'action': action,
                'images': {cam: img.copy()
                           for cam, img in frame['images'].items()},
                'subtask': self.current_subtask,
            }
            if cfg.save_pointcloud and frame['pointcloud'] is not None:
                entry['pointcloud'] = frame['pointcloud']

            self.frame_buffer.append(entry)
            count += 1
            if count % 100 == 0:
                rospy.loginfo("  … %d frames", count)
            rate.sleep()

        if count >= cfg.max_timesteps:
            rospy.loginfo("Reached max_timesteps (%d)", cfg.max_timesteps)
            self.collecting = False

    # ── Save Episode ─────────────────────────────────────────────────────

    def _save_episode(self):
        if not self.frame_buffer:
            rospy.logwarn("Nothing to save — buffer is empty.")
            return

        cfg = self.cfg
        n = len(self.frame_buffer)
        rospy.loginfo("Saving episode with %d frames …", n)

        labels = [f['subtask'] for f in self.frame_buffer]
        task_text = max(set(labels), key=labels.count)

        for entry in self.frame_buffer:
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

            if cfg.save_pointcloud and 'pointcloud' in entry:
                frame['observation.pointcloud'] = \
                    entry['pointcloud'].astype(np.float32)
            if cfg.use_subtask:
                frame['subtask'] = entry['subtask']

            self.dataset.add_frame(frame)

        self.dataset.save_episode()

        ep = self.dataset.meta.total_episodes - 1
        rospy.loginfo("Episode %d saved  |  total_episodes=%d  "
                      "total_frames=%d",
                      ep, self.dataset.meta.total_episodes,
                      self.dataset.meta.total_frames)
        self.frame_buffer.clear()
        self.current_subtask = self.SUBTASKS['1']

    # ── Keyboard ─────────────────────────────────────────────────────────

    def _key_poll_loop(self):
        listener = keyboard.Listener(on_press=_on_key_press)
        listener.start()
        while not rospy.is_shutdown():
            try:
                self._key = _key_queue.get(timeout=0.1)
            except queue.Empty:
                continue

    def _command_loop(self):
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
                rospy.loginfo("Buffer cleared, subtask reset.")
            elif k in self.SUBTASKS and not self.collecting:
                self.current_subtask = self.SUBTASKS[k]
                rospy.loginfo("Subtask → %s", self.current_subtask)
                self._key = None
            elif k == 'q':
                self._key = None
                self._finalize()
                return
            rospy.sleep(0.05)

    # ── Finalize ─────────────────────────────────────────────────────────

    def _finalize(self):
        rospy.loginfo("Finalizing dataset …")
        if self.dataset is not None:
            if self.dataset.image_writer is not None:
                self.dataset.stop_image_writer()
            self.dataset.finalize()
        rospy.loginfo("Dataset finalized at %s", self.cfg.target_dir)
        rospy.signal_shutdown("User requested exit")

    # ── Usage ────────────────────────────────────────────────────────────

    @staticmethod
    def _print_usage():
        rospy.loginfo(
            '\033[32m'
            '═══════════════════════════════════════════\n'
            '  LeRobot v3 Direct Collection Node\n'
            '═══════════════════════════════════════════\n'
            '  c     → start / resume collecting\n'
            '  space → pause\n'
            '  1-4   → switch subtask (while paused)\n'
            '  s     → save episode\n'
            '  r     → discard buffer\n'
            '  q     → finalize & exit\n'
            '═══════════════════════════════════════════'
            '\033[0m'
        )


def main():
    try:
        rospy.init_node('data_to_lerobot3_node', anonymous=True)
        LeRobotCollectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down LeRobot collection node.")


if __name__ == '__main__':
    main()
