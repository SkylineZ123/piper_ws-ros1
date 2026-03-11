#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-arm HDF5 Data Collection Node

Subscribes to camera images, depth images, point clouds, and joint states
from left/right master/slave arms, synchronizes frames, and saves episodes
as HDF5 files.

Controls (keyboard)::

    c  — start / resume collecting
    x  — stop collecting (mark episode ready to save)
    s  — save buffered episode to HDF5
    r  — discard buffered data

Usage::

    rosrun dual_piper data_to_hdf5_node.py
"""
import os
import copy
import collections
from collections import deque
from types import SimpleNamespace

import numpy as np
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


class HDF5CollectorNode:
    """
    ROS node that collects synchronised dual-arm data and writes HDF5
    episodes.
    """

    def __init__(self):
        rospy.init_node('data_to_hdf5_node', anonymous=True)

        # ── Parameters ───────────────────────────────────────────────────
        P = lambda name, default: rospy.get_param('~' + name, default)

        self.args = SimpleNamespace(
            dataset_dir=P('dataset_dir', '/home/user/zzq/piper_ws/data'),
            task_name=P('task_name', 'bloodgas_0306'),
            episode_idx=P('episode_idx', 3),
            max_timesteps=P('max_timesteps', 5000),
            camera_names=[c.strip() for c in
                          P('camera_names', 'cam_top,cam_left,cam_right').split(',')],
            use_depth_image=P('use_depth_image', False),
            save_pointcloud=P('save_pointcloud', False),
            pc_target_points=P('pc_target_points', 20000),
            frame_rate=P('frame_rate', 20),
            # Topics
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
            self._queues['pc_top'] = deque(maxlen=3)

        # ── Subscribers ──────────────────────────────────────────────────
        Q = 50
        rospy.Subscriber(self.args.img_top_topic, Image,
                         lambda m: self._queues['cam_top'].append(m),
                         queue_size=Q)
        if 'cam_left' in self.args.camera_names:
            rospy.Subscriber(self.args.img_leftwrist_topic, Image,
                             lambda m: self._queues['cam_left'].append(m),
                             queue_size=Q)
        if 'cam_right' in self.args.camera_names:
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
            if 'cam_left' in self.args.camera_names:
                rospy.Subscriber(
                    self.args.img_leftwrist_depth_topic, Image,
                    lambda m: self._queues['cam_left_depth'].append(m),
                    queue_size=Q)
            if 'cam_right' in self.args.camera_names:
                rospy.Subscriber(
                    self.args.img_rightwrist_depth_topic, Image,
                    lambda m: self._queues['cam_right_depth'].append(m),
                    queue_size=Q)

        if self.args.save_pointcloud:
            rospy.Subscriber(self.args.pc_top_topic, PointCloud2,
                             lambda m: self._queues['pc_top'].append(m),
                             queue_size=10)

        # ── Keyboard & state ─────────────────────────────────────────────
        self._kb = KeyboardHandler(valid_keys={'c', 'x', 's', 'r'})
        self._kb.start()

        self._timesteps = []
        self._actions = []
        self._collecting = False
        self._episode_idx = self.args.episode_idx

        rospy.loginfo('\033[32m HDF5 data collection node ready.\033[0m')
        rospy.loginfo('\033[32m  c → start   x → stop   s → save   r → reset\033[0m')

        # ── Main command loop (runs in a daemon thread) ──────────────────
        import threading
        cmd_thread = threading.Thread(target=self._command_loop, daemon=True)
        cmd_thread.start()

    # ── Command Loop ─────────────────────────────────────────────────────

    def _command_loop(self):
        while not rospy.is_shutdown():
            key = self._kb.get(timeout=0.1)
            if key is None:
                continue

            if key == 'c' and not self._collecting:
                rospy.loginfo("Start collecting episode %d",
                              self._episode_idx)
                self._collect()

            elif key == 's' and self._collecting:
                self._save_episode()

            elif key == 'r':
                rospy.loginfo("Buffer cleared.")
                self._timesteps.clear()
                self._actions.clear()
                self._collecting = False

    # ── Frame Sync ───────────────────────────────────────────────────────

    def _get_synced_frame(self):
        """
        Pop one time-synchronised frame from all required queues.

        Returns:
            dict or ``False`` if not all queues are ready.
        """
        keys = list(self.args.camera_names)
        if self.args.use_depth_image:
            keys += [f'{c}_depth' for c in self.args.camera_names
                     if f'{c}_depth' in self._queues]
        keys += ['left/master_arm', 'left/slave_arm',
                 'right/master_arm', 'right/slave_arm']

        if any(len(self._queues.get(k, [])) == 0 for k in keys):
            return False

        # Sync to the newest "oldest" message
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
            from utils import sample_pointcloud
            pc_msg = self._queues['pc_top'].popleft()
            pointclouds = {'cam_top': sample_pointcloud(
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
        """Collect frames until ``max_timesteps`` or user presses ``x``."""
        self._timesteps.clear()
        self._actions.clear()

        image_dict = {
            name: np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
            for name in self.args.camera_names
        }
        depth_dict = {}
        if self.args.use_depth_image:
            depth_dict = {
                name: np.random.random((180, 320, 1))
                for name in self.args.camera_names
                if name not in ('cam_wrist', 'cam_left', 'cam_right')
            }

        pbar = tqdm(total=int(self.args.max_timesteps),
                    desc='Collecting', ncols=80)
        count = 0
        rate = rospy.Rate(self.args.frame_rate)

        while not rospy.is_shutdown() and count <= self.args.max_timesteps:
            result = self._get_synced_frame()
            if not result:
                rospy.logwarn_throttle(5, 'Waiting for data …')
                rate.sleep()
                continue

            count += 1
            lm = result['left/master_arm']
            ls = result['left/slave_arm']
            rm = result['right/master_arm']
            rs = result['right/slave_arm']

            for name in self.args.camera_names:
                if name in result['images']:
                    image_dict[name] = result['images'][name].copy()
                if (self.args.use_depth_image
                        and result['depths']
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
            pbar.update(1)

            rate.sleep()

            # Check stop key
            key = self._kb.get(timeout=0)
            if key == 'x' or count > self.args.max_timesteps:
                rospy.loginfo("Stopped at frame %d", count - 1)
                self._collecting = True
                break

        pbar.close()
        rospy.loginfo("timesteps: %d  |  actions: %d",
                      len(self._timesteps), len(self._actions))

    # ── Save ─────────────────────────────────────────────────────────────

    def _save_episode(self):
        dataset_dir = os.path.join(self.args.dataset_dir,
                                   self.args.task_name)
        os.makedirs(dataset_dir, exist_ok=True)
        path = os.path.join(dataset_dir, f'episode_{self._episode_idx}')
        save_hdf5_data(self.args, self._timesteps, self._actions, path)
        self._collecting = False
        self._episode_idx += 1


def main():
    try:
        HDF5CollectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down HDF5 data collection node.")


if __name__ == '__main__':
    main()
