#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-level Dual-arm HDF5 Data Collection Node

Extends the basic HDF5 collector with **subtask labelling**.  The operator
can pause collection, switch the active subtask via number keys (1-N), then
resume.  Each frame is tagged with the current subtask string, which is
stored in ``/observations/subtask`` as variable-length strings in the HDF5
file.  Episodes are saved into per-subtask sub-directories.

Controls (keyboard)::

    c      — start / resume collecting
    space  — pause collecting
    1-N    — switch subtask (while paused)
    s      — save episode
    r      — discard buffered data

Usage::

    rosrun dual_piper data_to_hdf5_highlevel_node.py
"""
import os
import copy
import threading
import collections
from collections import deque

import numpy as np
import yaml
import rospy
from sensor_msgs.msg import JointState, Image, PointCloud2
from cv_bridge import CvBridge
from tqdm import tqdm
import dm_env

from utils import (
    KeyboardHandler,
    save_hdf5_data,
    get_ros_stamp_sec,
    NUM_JOINTS_PER_ARM,
)


def _load_subtasks(yaml_path=None):
    """Load subtask definitions from a YAML file or return defaults."""
    if yaml_path and os.path.isfile(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('subtasks', {})
    return {
        '1': 'pick up beakers from the table.',
        '2': 'pour the liquid from the left beaker into the right beaker.',
        '3': 'place the beakers to their initial position.',
        '4': 'return to the initial position.',
    }


class HighLevelHDF5CollectorNode:
    """
    HDF5 collector with subtask annotation support.

    Subtask definitions are loaded from a YAML config file
    (``~subtasks_file``) or fall back to built-in defaults.
    """

    def __init__(self):
        rospy.init_node('data_to_hdf5_highlevel_node', anonymous=True)

        # ── Parameters ───────────────────────────────────────────────────
        P = lambda name, default: rospy.get_param('~' + name, default)

        subtasks_file = P('subtasks_file', '')
        self.subtasks = _load_subtasks(subtasks_file)
        self.current_subtask = self.subtasks.get('1', '')
        self._subtask_list = []

        dataset_dir = P('dataset_dir', '/home/user/hdd_4t/data/data_hdf5')
        task_name = P('task_name', 'dual_arm_task')
        episode_idx = P('episode_idx', 0)
        max_timesteps = P('max_timesteps', 5000)
        camera_names_raw = P('camera_names', 'cam_left,cam_top')
        if isinstance(camera_names_raw, str):
            camera_names = [c.strip() for c in camera_names_raw.split(',')]
        else:
            camera_names = list(camera_names_raw)

        self.args = collections.namedtuple('Args', [
            'dataset_dir', 'task_name', 'episode_idx', 'max_timesteps',
            'camera_names', 'use_depth_image', 'save_pointcloud',
            'pc_target_points', 'frame_rate',
            'img_top_topic', 'img_leftwrist_topic', 'img_rightwrist_topic',
            'img_top_depth_topic', 'img_leftwrist_depth_topic',
            'img_rightwrist_depth_topic',
            'left_master_arm_topic', 'left_slave_arm_topic',
            'right_master_arm_topic', 'right_slave_arm_topic',
            'pc_top_topic',
        ])(
            dataset_dir=dataset_dir,
            task_name=task_name,
            episode_idx=episode_idx,
            max_timesteps=max_timesteps,
            camera_names=camera_names,
            use_depth_image=P('use_depth_image', False),
            save_pointcloud=P('save_pointcloud', False),
            pc_target_points=P('pc_target_points', 5000),
            frame_rate=P('frame_rate', 20),
            img_top_topic=P('img_top_topic',
                            '/cam_top/cam_top/color/image_raw'),
            img_leftwrist_topic=P('img_leftwrist_topic',
                                  '/cam_leftwrist/cam_leftwrist/color/image_raw'),
            img_rightwrist_topic=P('img_rightwrist_topic',
                                   '/cam_rightwrist/cam_rightwrist/color/image_raw'),
            img_top_depth_topic=P('img_top_depth_topic',
                                  '/cam_top/cam_top/aligned_depth_to_color/image_raw'),
            img_leftwrist_depth_topic=P('img_leftwrist_depth_topic',
                                        '/cam_leftwrist/cam_leftwrist/aligned_depth_to_color/image_raw'),
            img_rightwrist_depth_topic=P('img_rightwrist_depth_topic',
                                         '/cam_rightwrist/cam_rightwrist/aligned_depth_to_color/image_raw'),
            left_master_arm_topic=P('left_master_arm_topic',
                                    'left/master_joint_states'),
            left_slave_arm_topic=P('left_slave_arm_topic',
                                   'left/slave_joint_states'),
            right_master_arm_topic=P('right_master_arm_topic',
                                     'right/master_joint_states'),
            right_slave_arm_topic=P('right_slave_arm_topic',
                                    'right/slave_joint_states'),
            pc_top_topic=P('pc_top_topic',
                           '/cam_top/cam_top/depth/color/points'),
        )

        self._bridge = CvBridge()

        # ── Queues ───────────────────────────────────────────────────────
        self._queues = {
            'cam_top': deque(maxlen=5),
            'cam_left': deque(maxlen=5),
            'cam_right': deque(maxlen=5),
            'left/master_arm': deque(maxlen=50),
            'left/slave_arm': deque(maxlen=50),
            'right/master_arm': deque(maxlen=50),
            'right/slave_arm': deque(maxlen=50),
        }
        if self.args.use_depth_image:
            for cam in ('cam_top', 'cam_left', 'cam_right'):
                self._queues[f'{cam}_depth'] = deque(maxlen=5)
        if self.args.save_pointcloud:
            self._queues['pc_top'] = deque(maxlen=10)

        # ── Subscribers ──────────────────────────────────────────────────
        Q = 50
        rospy.Subscriber(self.args.img_top_topic, Image,
                         lambda m: self._queues['cam_top'].append(m),
                         queue_size=Q)
        if 'cam_left' in camera_names:
            rospy.Subscriber(self.args.img_leftwrist_topic, Image,
                             lambda m: self._queues['cam_left'].append(m),
                             queue_size=Q)
        if 'cam_right' in camera_names:
            rospy.Subscriber(self.args.img_rightwrist_topic, Image,
                             lambda m: self._queues['cam_right'].append(m),
                             queue_size=Q)

        rospy.Subscriber(self.args.left_master_arm_topic, JointState,
                         lambda m: self._queues['left/master_arm'].append(m),
                         queue_size=Q)
        rospy.Subscriber(self.args.left_slave_arm_topic, JointState,
                         lambda m: self._queues['left/slave_arm'].append(m),
                         queue_size=Q)
        rospy.Subscriber(self.args.right_master_arm_topic, JointState,
                         lambda m: self._queues['right/master_arm'].append(m),
                         queue_size=Q)
        rospy.Subscriber(self.args.right_slave_arm_topic, JointState,
                         lambda m: self._queues['right/slave_arm'].append(m),
                         queue_size=Q)

        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_top_depth_topic, Image,
                             lambda m: self._queues['cam_top_depth'].append(m),
                             queue_size=Q)
            if 'cam_left' in camera_names:
                rospy.Subscriber(
                    self.args.img_leftwrist_depth_topic, Image,
                    lambda m: self._queues['cam_left_depth'].append(m),
                    queue_size=Q)
            if 'cam_right' in camera_names:
                rospy.Subscriber(
                    self.args.img_rightwrist_depth_topic, Image,
                    lambda m: self._queues['cam_right_depth'].append(m),
                    queue_size=Q)

        if self.args.save_pointcloud:
            rospy.Subscriber(self.args.pc_top_topic, PointCloud2,
                             lambda m: self._queues['pc_top'].append(m),
                             queue_size=10)

        # ── Keyboard ─────────────────────────────────────────────────────
        valid_keys = {'c', 's', 'r', 'space'}
        valid_keys.update(self.subtasks.keys())
        self._kb = KeyboardHandler(valid_keys=valid_keys)
        self._kb.start()

        self._timesteps = []
        self._actions = []
        self._collecting = False
        self._episode_idx = self.args.episode_idx

        rospy.loginfo('\033[32m High-level HDF5 collector ready.\033[0m')
        rospy.loginfo('\033[32m  c → start/resume  space → pause  '
                      '1-N → switch subtask  s → save  r → discard\033[0m')

        cmd_thread = threading.Thread(target=self._command_loop, daemon=True)
        cmd_thread.start()

    # ── Command Loop ─────────────────────────────────────────────────────

    def _command_loop(self):
        while not rospy.is_shutdown():
            key = self._kb.get(timeout=0.1)
            if key is None:
                continue

            if key == 'c' and not self._collecting:
                rospy.loginfo("Collecting episode %d  (subtask: %s)",
                              self._episode_idx, self.current_subtask)
                self._collecting = True
                self._collect()

            elif key == 's' and not self._collecting:
                self._save_episode()

            elif key == 'r':
                rospy.loginfo("Buffer cleared.")
                self._timesteps.clear()
                self._actions.clear()
                self._subtask_list.clear()
                self.current_subtask = self.subtasks.get('1', '')
                self._collecting = False

            elif key in self.subtasks and not self._collecting:
                self.current_subtask = self.subtasks[key]
                rospy.loginfo("Subtask → %s", self.current_subtask)

    # ── Frame Sync ───────────────────────────────────────────────────────

    def _get_synced_frame(self):
        keys = list(self.args.camera_names)
        if self.args.use_depth_image:
            keys += [f'{c}_depth' for c in self.args.camera_names
                     if f'{c}_depth' in self._queues]
        keys += ['left/master_arm', 'left/slave_arm',
                 'right/master_arm', 'right/slave_arm']

        if any(len(self._queues.get(k, [])) == 0 for k in keys):
            return False

        sync_t = min(get_ros_stamp_sec(self._queues[k][-1]) for k in keys)
        for k in keys:
            while (self._queues[k]
                   and get_ros_stamp_sec(self._queues[k][0]) < sync_t):
                self._queues[k].popleft()
        if any(len(self._queues.get(k, [])) == 0 for k in keys):
            return False

        images = {}
        for cam in self.args.camera_names:
            if self._queues.get(cam):
                images[cam] = self._bridge.imgmsg_to_cv2(
                    self._queues[cam].popleft(),
                    desired_encoding='passthrough')

        depths = None
        if self.args.use_depth_image:
            depths = {}
            for k in keys:
                if 'depth' in k and self._queues.get(k):
                    name = k.rsplit('_', 1)[0]
                    depths[name] = self._bridge.imgmsg_to_cv2(
                        self._queues[k].popleft(),
                        desired_encoding='16UC1')

        pointclouds = None
        if self.args.save_pointcloud and self._queues.get('pc_top'):
            from utils import sample_and_expand_pointcloud
            pc_msg = self._queues['pc_top'].popleft()
            pointclouds = {'cam_top': sample_and_expand_pointcloud(
                pc_msg, self.args.pc_target_points)}

        return {
            'images': images,
            'depths': depths,
            'pointclouds': pointclouds,
            'left/master_arm': (self._queues['left/master_arm'].popleft()
                                if self._queues['left/master_arm'] else None),
            'left/slave_arm': (self._queues['left/slave_arm'].popleft()
                               if self._queues['left/slave_arm'] else None),
            'right/master_arm': (self._queues['right/master_arm'].popleft()
                                 if self._queues['right/master_arm'] else None),
            'right/slave_arm': (self._queues['right/slave_arm'].popleft()
                                if self._queues['right/slave_arm'] else None),
        }

    # ── Collection ───────────────────────────────────────────────────────

    def _collect(self):
        image_dict = {
            name: np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
            for name in self.args.camera_names
        }
        depth_dict = {}
        if self.args.use_depth_image:
            depth_dict = {
                name: np.random.random((180, 320, 1))
                for name in self.args.camera_names
                if name not in ('cam_wrist',)
            }

        initial = len(self._timesteps)
        pbar = tqdm(total=int(self.args.max_timesteps),
                    desc='Collecting', ncols=80, initial=initial)
        count = initial
        rate = rospy.Rate(self.args.frame_rate)

        while not rospy.is_shutdown() and count < self.args.max_timesteps:
            key = self._kb.get(timeout=0)
            if key == 'space':
                rospy.loginfo("Paused at frame %d", count)
                self._collecting = False
                break

            result = self._get_synced_frame()
            if not result:
                rospy.logwarn_throttle(5, "Waiting for data …")
                rate.sleep()
                continue

            count += 1
            pbar.update(1)

            lm = result['left/master_arm']
            ls = result['left/slave_arm']
            rm = result['right/master_arm']
            rs = result['right/slave_arm']

            for name in self.args.camera_names:
                if name in result['images']:
                    image_dict[name] = result['images'][name].copy()
                if (self.args.use_depth_image
                        and result.get('depths')
                        and name in result['depths']):
                    depth_dict[name] = result['depths'][name].copy()

            obs = collections.OrderedDict()
            obs['images'] = copy.deepcopy(image_dict)
            if self.args.use_depth_image:
                obs['images_depth'] = copy.deepcopy(depth_dict)

            l_pos = np.array(ls.position)[:NUM_JOINTS_PER_ARM]
            l_vel = np.array(ls.velocity)[:NUM_JOINTS_PER_ARM]
            l_eff = np.array(ls.effort)[:NUM_JOINTS_PER_ARM]
            r_pos = np.array(rs.position)[:NUM_JOINTS_PER_ARM]
            r_vel = np.array(rs.velocity)[:NUM_JOINTS_PER_ARM]
            r_eff = np.array(rs.effort)[:NUM_JOINTS_PER_ARM]

            obs['qpos'] = np.concatenate([l_pos, r_pos])
            obs['qvel'] = np.concatenate([l_vel, r_vel])
            obs['effort'] = np.concatenate([l_eff, r_eff])
            obs['subtask'] = self.current_subtask
            self._subtask_list.append(self.current_subtask)

            if (self.args.save_pointcloud
                    and result.get('pointclouds')
                    and 'cam_top' in result['pointclouds']):
                obs.setdefault('pointclouds', {})['cam_top'] = \
                    result['pointclouds']['cam_top'].astype(np.float32)

            step_type = (dm_env.StepType.FIRST if count == 1
                         else dm_env.StepType.MID)
            ts = dm_env.TimeStep(step_type=step_type, reward=None,
                                 discount=None, observation=obs)

            if count == 1:
                self._timesteps.append(ts)
                rate.sleep()
                continue

            action = np.concatenate([
                np.array(lm.position)[:NUM_JOINTS_PER_ARM],
                np.array(rm.position)[:NUM_JOINTS_PER_ARM],
            ])
            self._actions.append(action)
            self._timesteps.append(ts)
            rate.sleep()

        if count >= self.args.max_timesteps:
            rospy.loginfo("Max timesteps reached (%d)", count)
            self._collecting = False

        pbar.close()
        rospy.loginfo("timesteps: %d  |  actions: %d  |  subtask labels: %d",
                      len(self._timesteps), len(self._actions),
                      len(self._subtask_list))

    # ── Save ─────────────────────────────────────────────────────────────

    def _save_episode(self):
        if not self._subtask_list:
            label = self.current_subtask
        else:
            label = max(set(self._subtask_list),
                        key=self._subtask_list.count)

        folder = label.replace(' ', '_').replace('/', '_').replace('\\', '_')
        dataset_dir = os.path.join(self.args.dataset_dir,
                                   self.args.task_name, folder)
        os.makedirs(dataset_dir, exist_ok=True)
        path = os.path.join(dataset_dir, f'episode_{self._episode_idx}')

        rospy.loginfo("Saving to subtask dir: %s", folder)
        save_hdf5_data(self.args, self._timesteps, self._actions, path)

        self._episode_idx += 1
        self._timesteps.clear()
        self._actions.clear()
        self._subtask_list.clear()
        self.current_subtask = self.subtasks.get('1', '')


def main():
    try:
        HighLevelHDF5CollectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down high-level HDF5 collector.")


if __name__ == '__main__':
    main()
