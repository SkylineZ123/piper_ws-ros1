#!/home/user/anaconda3/envs/lerobotv3/bin/python
# -*- coding: utf-8 -*-
import sys
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path not in sys.path:
    sys.path.insert(0, ros_path)
import os
import time
import json
import copy
import queue
import threading
import collections
import numpy as np
import cv2
import rospy
from pynput import keyboard
from sensor_msgs.msg import Image, JointState, PointCloud2
from cv_bridge import CvBridge
import dm_env
from tqdm import tqdm
import h5py
import sensor_msgs.point_cloud2 as pc2  # 添加缺少的导入

# 全局变量
key_event_queue = queue.Queue()


# 键盘事件处理函数
def on_press_terminal(e):
    # 首先检查字符键（字母、数字等）
    if hasattr(e, 'char') and e.char is not None:
        key_event_queue.put(e.char)
    # 然后检查特殊键的名称
    elif hasattr(e, 'name'):
        key_event_queue.put(e.name)


# # 点云转换函数
# def pointcloud2_to_xyz_array(cloud_msg, remove_nans=True):
#     points_list = []
#     for point in pc2.read_points(cloud_msg, skip_nans=remove_nans):
#         points_list.append([point[0], point[1], point[2]])
#     return np.array(points_list)
#
#
# # 数据保存函数
# def save_data(args, timesteps, actions, dataset_path):
#     # 创建HDF5文件
#     os.makedirs(dataset_path, exist_ok=True)
#     hdf5_path = os.path.join(dataset_path, 'traj.hdf5')
#     metadata_path = os.path.join(dataset_path, 'metadata.json')
#
#     with h5py.File(hdf5_path, 'w') as root:
#         # 保存动作数据
#         root.create_dataset('action', data=np.array(actions, dtype=np.float32))
#
#         # 计算数据大小
#         data_size = len(timesteps)
#
#         # 创建观测组
#         obs = root.create_group('observation')
#
#         # 保存关节位置、速度和力矩
#         obs.create_dataset('qpos', (data_size, 14))
#         obs.create_dataset('qvel', (data_size, 14))
#         obs.create_dataset('effort', (data_size, 14))
#
#         # 如果存在子任务数据，保存子任务
#         if hasattr(timesteps[0].observation, 'get') and 'subtask' in timesteps[0].observation:
#             subtask_dataset = obs.create_dataset('subtask', (data_size,), dtype=h5py.special_dtype(vlen=str))
#
#         # 保存点云数据（如果有）
#         if hasattr(timesteps[0].observation, 'get') and 'pointclouds' in timesteps[0].observation:
#             pointclouds_grp = obs.create_group('pointclouds')
#             for cam_name in timesteps[0].observation['pointclouds'].keys():
#                 target_points = args.pc_target_points
#                 pointclouds_grp.create_dataset(cam_name, (data_size, target_points, 3), dtype=np.float32)
#
#         # 保存图像数据
#         images_grp = obs.create_group('images')
#         for cam_name in args.camera_names:
#             img_shape = timesteps[0].observation['images'][cam_name].shape
#             images_grp.create_dataset(cam_name, (data_size,) + img_shape, dtype=np.uint8)
#
#         # 保存深度图像数据（如果有）
#         if hasattr(timesteps[0].observation, 'get') and 'images_depth' in timesteps[0].observation:
#             images_depth_grp = obs.create_group('images_depth')
#             for cam_name in timesteps[0].observation['images_depth'].keys():
#                 depth_shape = timesteps[0].observation['images_depth'][cam_name].shape
#                 images_depth_grp.create_dataset(cam_name, (data_size,) + depth_shape, dtype=np.uint16)
#
#         # 填充数据
#         for i, ts in enumerate(tqdm(timesteps, desc='Saving data')):
#             obs['qpos'][i] = ts.observation['qpos']
#             obs['qvel'][i] = ts.observation['qvel']
#             obs['effort'][i] = ts.observation['effort']
#
#             # 保存子任务
#             if hasattr(ts.observation, 'get') and 'subtask' in ts.observation:
#                 subtask_dataset[i] = ts.observation['subtask']
#
#             # 保存点云
#             if hasattr(ts.observation, 'get') and 'pointclouds' in ts.observation:
#                 for cam_name, pc_data in ts.observation['pointclouds'].items():
#                     # 确保点云数据大小一致
#                     if len(pc_data) >= args.pc_target_points:
#                         pointclouds_grp[cam_name][i] = pc_data[:args.pc_target_points]
#                     else:
#                         # 不足则填充零
#                         padded_pc = np.zeros((args.pc_target_points, 3), dtype=np.float32)
#                         padded_pc[:len(pc_data)] = pc_data
#                         pointclouds_grp[cam_name][i] = padded_pc
#
#             # 保存图像
#             for cam_name, img_data in ts.observation['images'].items():
#                 images_grp[cam_name][i] = img_data
#
#             # 保存深度图像
#             if hasattr(ts.observation, 'get') and 'images_depth' in ts.observation:
#                 for cam_name, depth_data in ts.observation['images_depth'].items():
#                     images_depth_grp[cam_name][i] = depth_data
#
#     # 保存元数据
#     metadata = {
#         'action_space': {
#             'shape': actions[0].shape if actions else [],
#             'dtype': str(actions[0].dtype) if actions else '',
#         },
#         'observation_space': {
#             'qpos': {'shape': (14,), 'dtype': 'float32'},
#             'qvel': {'shape': (14,), 'dtype': 'float32'},
#             'effort': {'shape': (14,), 'dtype': 'float32'},
#         },
#         'episode_length': len(timesteps),
#     }
#
#     # 添加图像元数据
#     if timesteps:
#         for cam_name, img_data in timesteps[0].observation['images'].items():
#             metadata['observation_space'][f'images/{cam_name}'] = {
#                 'shape': img_data.shape,
#                 'dtype': str(img_data.dtype),
#             }
#
#         # 添加深度图像元数据
#         if 'images_depth' in timesteps[0].observation:
#             for cam_name, depth_data in timesteps[0].observation['images_depth'].items():
#                 metadata['observation_space'][f'images_depth/{cam_name}'] = {
#                     'shape': depth_data.shape,
#                     'dtype': str(depth_data.dtype),
#                 }
#
#         # 添加点云元数据
#         if 'pointclouds' in timesteps[0].observation:
#             for cam_name in timesteps[0].observation['pointclouds'].keys():
#                 metadata['observation_space'][f'pointclouds/{cam_name}'] = {
#                     'shape': (args.pc_target_points, 3),
#                     'dtype': 'float32',
#                 }
#
#         # 添加子任务元数据
#         if 'subtask' in timesteps[0].observation:
#             metadata['observation_space']['subtask'] = {
#                 'dtype': 'string',
#             }
#
#     with open(metadata_path, 'w') as f:
#         json.dump(metadata, f, indent=4)
#
#     rospy.loginfo(f'Successfully saved data to {dataset_path}')
#
# 点云转换函数
# python
def _sample_and_expand_pointcloud(pc, target_points):
    """
    接受 pc (N,3) 或 (N,6)，返回 (target_points,6) 的 np.float32。
    若为 (N,3) 则补零 RGB；若点数不足则重复采样。
    """
    if pc is None:
        return np.zeros((target_points, 6), dtype=np.float32)
    pc = np.asarray(pc)
    if pc.size == 0:
        return np.zeros((target_points, 6), dtype=np.float32)

    # 规范为 (N,6)
    if pc.ndim == 1:
        pc = pc.reshape(-1, pc.shape[0])
    if pc.shape[1] == 3:
        zeros_rgb = np.zeros((pc.shape[0], 3), dtype=np.float32)
        pc6 = np.concatenate((pc[:, :3].astype(np.float32), zeros_rgb), axis=1)
    elif pc.shape[1] >= 6:
        pc6 = pc[:, :6].astype(np.float32)
    else:
        pad = np.zeros((pc.shape[0], 6 - pc.shape[1]), dtype=np.float32)
        pc6 = np.concatenate((pc.astype(np.float32), pad), axis=1)

    # 过滤非法值
    mask = np.isfinite(pc6).all(axis=1)
    pc6 = pc6[mask]
    if pc6.shape[0] == 0:
        return np.zeros((target_points, 6), dtype=np.float32)

    n = pc6.shape[0]
    if n >= target_points:
        idx = np.random.choice(n, target_points, replace=False)
    else:
        idx = np.random.choice(n, target_points, replace=True)
    return pc6[idx].astype(np.float32)


def save_data(args, timesteps, actions, dataset_path):
    """
    使用一次性写入方式保存数据（仿照另一个脚本的保存格式）。
    增加对子任务 subtask 的保存，存放在 /observations/subtask，dtype 为 vlen string。
    """
    # 使用本地副本，避免修改传入列表
    actions_list = list(actions)
    timesteps_list = list(timesteps)

    data_size = len(actions_list)
    if data_size == 0:
        rospy.logwarn("No actions to save, aborting save.")
        return

    # 检查是否包含 subtask 字段（用第一个 timestep 判断）
    has_subtask = False
    if timesteps_list and isinstance(timesteps_list[0].observation, dict) and 'subtask' in timesteps_list[0].observation:
        has_subtask = True

    # 预备字典（字符串项用 list 存储）
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }
    if has_subtask:
        data_dict['/observations/subtask'] = []

    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image and cam_name not in ['cam_wrist', 'cam_wrist_left', 'cam_wrist_right']:
            data_dict[f'/observations/images_depth/{cam_name}'] = []
        if args.save_pointcloud and cam_name == 'cam_top':
            data_dict[f'/observations/pointclouds/{cam_name}'] = []

    # 把每一帧收集到列表
    while actions_list:
        action = actions_list.pop(0)
        ts = timesteps_list.pop(0)

        data_dict['/observations/qpos'].append(np.asarray(ts.observation['qpos'], dtype=np.float32))
        data_dict['/observations/qvel'].append(np.asarray(ts.observation['qvel'], dtype=np.float32))
        data_dict['/observations/effort'].append(np.asarray(ts.observation['effort'], dtype=np.float32))
        data_dict['/action'].append(np.asarray(action, dtype=np.float32))

        if has_subtask:
            # 确保以字符串形式保存（避免 None）
            sub = ts.observation.get('subtask', '') if isinstance(ts.observation, dict) else ''
            data_dict['/observations/subtask'].append(str(sub))

        for cam_name in args.camera_names:
            img = ts.observation['images'][cam_name]
            data_dict[f'/observations/images/{cam_name}'].append(np.asarray(img, dtype=np.uint8))
            if args.use_depth_image and cam_name not in ['cam_wrist', 'cam_wrist_left', 'cam_wrist_right']:
                data_dict[f'/observations/images_depth/{cam_name}'].append(
                    np.asarray(ts.observation['images_depth'][cam_name], dtype=np.uint16)
                )
            if args.save_pointcloud and cam_name == 'cam_top':
                pc_raw = ts.observation.get('pointclouds', {}).get(cam_name, None)
                pc6 = _sample_and_expand_pointcloud(pc_raw, args.pc_target_points)
                data_dict[f'/observations/pointclouds/{cam_name}'].append(pc6)

    # 写入 HDF5（一次性写入）
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        root.attrs['data_size'] = data_size

        obs = root.create_group('observations')

        # images
        image_grp = obs.create_group('images')
        for cam_name in args.camera_names:
            image_grp.create_dataset(cam_name, (data_size, 180, 320, 3), dtype='uint8',
                                     chunks=(1, 180, 320, 3))

        # depth images
        if args.use_depth_image:
            image_depth_grp = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                if cam_name not in ['cam_wrist', 'cam_wrist_left', 'cam_wrist_right']:
                    image_depth_grp.create_dataset(cam_name, (data_size, 180, 320), dtype='uint16',
                                                   chunks=(1, 180, 320))

        # pointclouds (保存为 6 列)
        if args.save_pointcloud:
            pc_grp = obs.create_group('pointclouds')
            for cam_name in args.camera_names:
                if cam_name == 'cam_top':
                    pc_grp.create_dataset(cam_name, (data_size, args.pc_target_points, 6), dtype='float32',
                                          chunks=(1, args.pc_target_points, 6))

        # qpos/qvel/effort/action
        obs.create_dataset('qpos', (data_size, 14), dtype='float32')
        obs.create_dataset('qvel', (data_size, 14), dtype='float32')
        obs.create_dataset('effort', (data_size, 14), dtype='float32')
        root.create_dataset('action', (data_size, len(data_dict['/action'][0]) if data_dict['/action'] else 14),
                            dtype='float32')

        # 如果需要，创建 subtask 数据集（vlen str）
        if has_subtask:
            vlen_str = h5py.special_dtype(vlen=str)
            obs.create_dataset('subtask', (data_size,), dtype=vlen_str)

        # 把列表一次性写入
        for name, arr_list in data_dict.items():
            # 特殊处理字符串列表（subtask）
            if name == '/observations/subtask':
                ds = root.require_dataset(name, shape=(data_size,), dtype=h5py.special_dtype(vlen=str))
                # 直接写入字符串列表
                ds[...] = arr_list
                continue

            # 其余为数值数组：转换为 np.array 后写入
            np_arr = np.array(arr_list)
            try:
                root[name][...] = np_arr
            except Exception:
                ds = root.require_dataset(name, shape=np_arr.shape, dtype=np_arr.dtype)
                ds[...] = np_arr

    rospy.loginfo(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)

# ROS操作类
class RosOperator:
    def __init__(self):
        # 子任务定义
        # self.subtasks = {
        #     '1': 'Reach for the transparent test tube containing brown liquid in the right rack',
        #     '2': 'Grasp the transparent test tube.',
        #     '3': 'Elevate the transparent test tube with brown liquid from the rack.',
        #     '4': 'Transport the transparent test tube with brown liquid to the analysis station.',
        #     '5': 'Push the transparent test tube with brown liquid into the analyzer.',
        #     '6': 'Retract the transparent tube with brown liquid from the analyzer.',
        #     '7': 'Carry the transparent tube with brown liquid to the right rack.',
        #     '8': 'Place the transparent tube into the right rack.',
        #     '9': 'Move arm back to the starting point.',
        # }
        self.subtasks = {
            # '1': 'move the right beaker to the center.',
            '1': 'pick up beakers from the table.',
            '2': 'pour the liquid from the left beaker into right beaker.',
            '3': 'place the beakers to their 1c position.',
            '4': 'return to the initial position.'
        }
        # self.subtasks = {
        #     '1': 'grasp the test tube from right rack',
        #     '2': 'move the test tube close to the blood gas analyser',
        #     '3': 'align the test tube with the sampling needlepass the test tube from the right arm to the left arm'',
        #     '4': 'withdraw the test tube from the sampling needle',
        #     '5': 'place the test tube into the left rack',
        #     '6': 'return to the initial position',
        #     '7': ',
        #     '8': 'place the test tube into the left rack using left arm',
        # }
        # self.subtasks = {
        #     '1': 'Open the second drawer',
        #     '2': 'Close the second drawer',
        #     '3': 'Pick up the red block and place it in the drawer',
        #     '4': 'Pick up the red block and place it on the plate',
        #     '5': 'Pick up the red block',
        #
        #
        # }
        # self.subtasks = {
        #     '1': 'open the orange drawer',
        #     '2': 'pick up the blue block and place it in the orange drawer',
        #     '3': 'close the orange drawer',
        #     '4': 'move back to initial pose',
        #     # '5': 'pick up the test tube and place it in the orange drawer'
        #     '5': 'pick up the red block and place it into the orange drawer',
        # }
        # self.subtasks = {
        #     '1': 'left arm pick up the test tube from left rack',
        #     '2': 'left arm move to middle position',
        #     '3': 'right arm grasp the test tube from left arm',
        #     '4': 'right arm place the test tube in right rack',
        #     '5': 'right arm pick up the test tube from right rack',
        #     '6': 'right arm move to middle position',
        #     '7': 'left arm grasp the test tube from right arm',
        #     '8': 'left arm place the test tube in left rack',
        #     '9': 'back to initial pose'
        #     # '2': 'Close the second drawer',
        #     # '3': 'Pick up the red block and place it in the drawer',
        #     # '4': 'Pick up the red block and place it on the plate',
        #     # '5': 'Pick up the red block',
        # }
        self.current_subtask = self.subtasks['1']  # 默认子任务
        self.subtask_list = []  # 存储每一帧的子任务

       # ROS参数初始化c sc sc sc sc s c sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc sc
        dataset_dir = rospy.get_param('dataset_dir', '/home/user/hdd_4t/data/data_hdf5')
        task_name = rospy.get_param('task_name', 'tzj_PickRedBlock_20260206')
        episode_idx = rospy.get_param('episode_idx', 0)
        max_timesteps = rospy.get_param('max_timesteps', 5000)

        camera_names = rospy.get_param('camera_names', ['cam_left', 'cam_top'])
        # 修改第344-355行的话题路径参数
        img_top_topic = rospy.get_param('img_top_topic', '/cam_top/cam_top/color/image_raw')
        img_leftwrist_topic = rospy.get_param('img_leftwrist_topic', '/cam_leftwrist/cam_leftwrist/color/image_raw')
        img_rightwrist_topic = rospy.get_param('img_rightwrist_topic', '/cam_rightwrist/cam_rightwrist/color/image_raw')
        img_top_depth_topic = rospy.get_param('img_top_depth_topic', '/cam_top/cam_top/aligned_depth_to_color/image_raw')
        img_leftwrist_depth_topic = rospy.get_param('img_leftwrist_depth_topic', '/cam_leftwrist/cam_leftwrist/aligned_depth_to_color/image_raw')
        img_rightwrist_depth_topic = rospy.get_param('img_rightwrist_depth_topic', '/cam_rightwrist/cam_rightwrist/aligned_depth_to_color/image_raw')
        left_master_arm_topic = rospy.get_param('left_master_arm_topic', 'left/master_joint_states')
        left_slave_arm_topic = rospy.get_param('left_slave_arm_topic', 'left/slave_joint_states')
        right_master_arm_topic = rospy.get_param('right_master_arm_topic', 'right/master_joint_states')
        right_slave_arm_topic = rospy.get_param('right_slave_arm_topic', 'right/slave_joint_states')
        # 修改点云话题路径
        pc_top_topic = rospy.get_param('pc_top_topic', '/cam_top/cam_top/depth/color/points')
        save_pointcloud = rospy.get_param('save_pointcloud', False)
        pc_queue_size = rospy.get_param('pc_queue_size', 10)
        pc_target_points = rospy.get_param('pc_target_points', 5000)
        # 添加缺失的参数定义
        use_depth_image = rospy.get_param('use_depth_image', False)
        frame_rate = rospy.get_param('frame_rate', 20)

        # 整合参数
        args = collections.namedtuple('Args', [
            'dataset_dir', 'task_name', 'episode_idx', 'max_timesteps',
            'camera_names', 'img_top_topic', 'img_leftwrist_topic', 'img_rightwrist_topic',
            'img_top_depth_topic', 'img_leftwrist_depth_topic', 'img_rightwrist_depth_topic',
            'left_master_arm_topic', 'left_slave_arm_topic', 'right_master_arm_topic', 'right_slave_arm_topic',
            'use_depth_image', 'frame_rate', 'pc_top_topic', 'save_pointcloud', 'pc_queue_size', 'pc_target_points'
        ])

        self.args = args(
            dataset_dir=dataset_dir,
            task_name=task_name,
            episode_idx=episode_idx,
            max_timesteps=max_timesteps,
            camera_names=camera_names,
            img_top_topic=img_top_topic,
            img_leftwrist_topic=img_leftwrist_topic,
            img_rightwrist_topic=img_rightwrist_topic,
            img_top_depth_topic=img_top_depth_topic,
            img_leftwrist_depth_topic=img_leftwrist_depth_topic,
            img_rightwrist_depth_topic=img_rightwrist_depth_topic,
            left_master_arm_topic=left_master_arm_topic,
            left_slave_arm_topic=left_slave_arm_topic,
            right_master_arm_topic=right_master_arm_topic,
            right_slave_arm_topic=right_slave_arm_topic,
            use_depth_image=use_depth_image,
            frame_rate=frame_rate,
            pc_top_topic=pc_top_topic,
            save_pointcloud=save_pointcloud,
            pc_queue_size=pc_queue_size,
            pc_target_points=pc_target_points
        )

        # 初始化CvBridge
        self.bridge = CvBridge()
        self.camera_names = self.args.camera_names

        # 图像和关节状态队列
        self.img_queues = {
            'cam_left': collections.deque(maxlen=5),
            'cam_right': collections.deque(maxlen=5),
            'cam_top': collections.deque(maxlen=5),
            'left/master_arm': collections.deque(maxlen=50),
            'left/slave_arm': collections.deque(maxlen=50),
            'right/master_arm': collections.deque(maxlen=50),
            'right/slave_arm': collections.deque(maxlen=50),
        }

        # 如果使用深度图像，添加深度图像队列
        if self.args.use_depth_image:
            self.img_queues['cam_left_depth'] = collections.deque(maxlen=5)
            self.img_queues['cam_right_depth'] = collections.deque(maxlen=5)
            self.img_queues['cam_top_depth'] = collections.deque(maxlen=5)

        # 点云队列
        self.pc_raw_queue = collections.deque(maxlen=self.args.pc_queue_size)

        # 订阅者
        self.img_top_sub = rospy.Subscriber(self.args.img_top_topic, Image, self.img_top_callback, queue_size=50)

        if 'cam_left' in self.camera_names:
            self.img_leftwrist_sub = rospy.Subscriber(self.args.img_leftwrist_topic, Image, self.img_leftwrist_callback,
                                                      queue_size=50)
        if 'cam_right' in self.camera_names:
            self.img_rightwrist_sub = rospy.Subscriber(self.args.img_rightwrist_topic, Image,
                                                       self.img_rightwrist_callback,
                                                       queue_size=50)

        self.left_master_arm_sub = rospy.Subscriber(self.args.left_master_arm_topic, JointState,
                                                    self.left_master_arm_callback,
                                                    queue_size=50)
        self.left_slave_arm_sub = rospy.Subscriber(self.args.left_slave_arm_topic, JointState,
                                                   self.left_slave_arm_callback,
                                                   queue_size=50)
        self.right_master_arm_sub = rospy.Subscriber(self.args.right_master_arm_topic, JointState,
                                                     self.right_master_arm_callback, queue_size=50)
        self.right_slave_arm_sub = rospy.Subscriber(self.args.right_slave_arm_topic, JointState,
                                                    self.right_slave_arm_callback,
                                                    queue_size=50)

        if self.args.use_depth_image:
            self.img_top_depth_sub = rospy.Subscriber(self.args.img_top_depth_topic, Image, self.img_top_depth_callback,
                                                      queue_size=50)
            if 'cam_left' in self.camera_names:
                self.img_leftwrist_depth_sub = rospy.Subscriber(self.args.img_leftwrist_depth_topic, Image,
                                                                self.img_leftwrist_depth_callback, queue_size=50)
            if 'cam_right' in self.camera_names:
                self.img_rightwrist_depth_sub = rospy.Subscriber(self.args.img_rightwrist_depth_topic, Image,
                                                                 self.img_rightwrist_depth_callback, queue_size=50)

        if self.args.save_pointcloud:
            self.pc_top_sub = rospy.Subscriber(self.args.pc_top_topic, PointCloud2, self.pc_top_callback, queue_size=10)

        # 初始化线程和变量
        self.key = None
        self.term_thread = threading.Thread(target=self.terminal_thread)
        self.key_process_thread = threading.Thread(target=self.key_process_loop)
        self.term_thread.daemon = True
        self.key_process_thread.daemon = True
        self.term_thread.start()
        self.key_process_thread.start()

        self.timesteps, self.actions = [], []
        self.collecting_data = False

        # 打印使用说明
        rospy.loginfo(f'\033[32m 机器人数据收集节点已启动。 \033[0m')
        rospy.loginfo(f'\033[32m 按 "c" 开始收集数据。 \033[0m')
        rospy.loginfo(f'\033[32m 按 "空格" 暂停收集数据。 \033[0m')
        rospy.loginfo(f'\033[32m 暂停后按数字键 1-5 切换子任务。 \033[0m')
        rospy.loginfo(f'\033[32m 按 "s" 保存收集的数据。 \033[0m')
        rospy.loginfo(f'\033[32m 按 "r" 清除已收集的数据。 \033[0m')

    def key_process_loop(self):
        listener = keyboard.Listener(on_press=on_press_terminal)
        listener.start()
        while not rospy.is_shutdown():
            try:
                self.key = key_event_queue.get(timeout=0.1)
            except queue.Empty:
                pass

    def terminal_thread(self):
        episode_idx = self.args.episode_idx
        while not rospy.is_shutdown():
            if self.key == 'c' and not self.collecting_data:

                rospy.loginfo(f'开始为 episode {episode_idx} 收集数据')
                rospy.loginfo(f'当前子任务: {self.current_subtask}')
                self.collecting_data = True
                self.process()
                self.key = None
            elif self.key == 's' and not self.collecting_data:
                rospy.loginfo(f'保存 episode {episode_idx} 的数据')
                # 获取当前子任务的文本标签（使用收集的数据中的子任务）
                if self.subtask_list:
                    # 使用最常见的子任务作为该episode的标签
                    subtask_label = max(set(self.subtask_list), key=self.subtask_list.count)
                else:
                    subtask_label = self.current_subtask

                # 清理子任务标签，移除特殊字符，用下划线替换空格
                subtask_folder_name = subtask_label.replace(' ', '_').replace('/', '_').replace('\\', '_')

                # 创建按子任务分类的目录结构
                dataset_dir = os.path.join(self.args.dataset_dir, self.args.task_name, subtask_folder_name)
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)

                dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}")
                rospy.loginfo(f'保存到子任务目录: {subtask_folder_name}')
                save_data(self.args, self.timesteps, self.actions, dataset_path)
                episode_idx += 1
                # 清空子任务、时间步、动作列表
                self.timesteps = []
                self.actions = []
                self.subtask_list = []
                # 添加：重置当前子任务为默认值（subtask1）
                self.current_subtask = self.subtasks['1']
                self.key = None
            elif self.key == 'r':
                rospy.loginfo(f'清除已收集的数据标志')
                self.timesteps = []
                self.actions = []
                self.subtask_list = []  # 清空子任务列表
                # 添加：重置当前子任务为默认值（subtask1）
                self.current_subtask = self.subtasks['1']
                self.collecting_data = False
                self.key = None
            elif self.key in self.subtasks.keys() and not self.collecting_data:
                # 切换子任务
                self.current_subtask = self.subtasks[self.key]
                rospy.loginfo(f'已切换到子任务: {self.current_subtask}')
                self.key = None

    def get_frame(self, use_arm=True):
        def _get_latest_time(keys):
            return min(
                self.img_queues[k][-1].header.stamp.secs + 1e-9 * self.img_queues[k][-1].header.stamp.nsecs
                for k in keys if self.img_queues[k]
            ) if keys else None

        def _sync_queues(sync_time, keys):
            for k in keys:
                while self.img_queues[k] and (
                        self.img_queues[k][0].header.stamp.secs + 1e-9 * self.img_queues[k][
                    0].header.stamp.nsecs < sync_time
                ):
                    self.img_queues[k].popleft()

        keys = []
        if 'cam_left' in self.args.camera_names:
            keys.append('cam_left')
            if self.args.use_depth_image:
                keys.append('cam_left_depth')
        if 'cam_right' in self.args.camera_names:
            keys.append('cam_right')
            if self.args.use_depth_image:
                keys.append('cam_right_depth')
        if 'cam_top' in self.args.camera_names:
            keys.append('cam_top')
            if self.args.use_depth_image:
                keys.append('cam_top_depth')

        if use_arm:
            keys.extend(['left/master_arm', 'left/slave_arm', 'right/master_arm', 'right/slave_arm'])

        # 检查队列是否为空
        if any(len(self.img_queues.get(k, [])) == 0 for k in keys):
            return False

        # 计算同步时间
        sync_time = _get_latest_time(keys)
        if sync_time is None:
            return False

        # 同步队列
        _sync_queues(sync_time, keys)

        # 再次检查队列是否为空
        if any(len(self.img_queues.get(k, [])) == 0 for k in keys):
            return False

        # 获取图像数据
        images = {}
        for key in self.camera_names:
            if key in self.img_queues and self.img_queues[key]:
                images[key] = self.bridge.imgmsg_to_cv2(self.img_queues[key].popleft(), desired_encoding='passthrough')

        # 获取深度图像数据
        images_depth = None
        if self.args.use_depth_image:
            images_depth = {}
            for key in keys:
                if 'depth' in key and key in self.img_queues and self.img_queues[key]:
                    name = key.rsplit('_', 1)[0]
                    images_depth[name] = self.bridge.imgmsg_to_cv2(self.img_queues[key].popleft(),
                                                                   desired_encoding='16UC1')

        # 获取点云数据
        pointclouds = None
        if self.args.save_pointcloud and self.pc_raw_queue:
            pc_msg = self.pc_raw_queue.popleft()
            try:
                pointclouds_arr = _sample_and_expand_pointcloud(pc_msg, remove_nans=True)
                pointclouds = {"cam_top": pointclouds_arr}
            except Exception as e:
                rospy.logerr(f'点云转换错误: {e}')
                pointclouds = None

        # 获取关节状态数据
        left_master_arm, left_slave_arm, right_master_arm, right_slave_arm = None, None, None, None
        if use_arm:
            if self.img_queues['left/master_arm']:
                left_master_arm = self.img_queues['left/master_arm'].popleft()
            if self.img_queues['left/slave_arm']:
                left_slave_arm = self.img_queues['left/slave_arm'].popleft()
            if self.img_queues['right/master_arm']:
                right_master_arm = self.img_queues['right/master_arm'].popleft()
            if self.img_queues['right/slave_arm']:
                right_slave_arm = self.img_queues['right/slave_arm'].popleft()

        return {
            "images": images,
            "depths": images_depth,
            "pointclouds": pointclouds,
            "left/master_arm": left_master_arm,
            "left/slave_arm": left_slave_arm,
            "right/master_arm": right_master_arm,
            "right/slave_arm": right_slave_arm
        }

    def img_leftwrist_callback(self, msg):
        self.img_queues['cam_left'].append(msg)

    def img_rightwrist_callback(self, msg):
        self.img_queues['cam_right'].append(msg)

    def img_top_callback(self, msg):
        self.img_queues['cam_top'].append(msg)

    def img_top_depth_callback(self, msg):
        self.img_queues['cam_top_depth'].append(msg)

    def img_leftwrist_depth_callback(self, msg):
        self.img_queues['cam_left_depth'].append(msg)

    def img_rightwrist_depth_callback(self, msg):
        self.img_queues['cam_right_depth'].append(msg)

    def pc_top_callback(self, msg):
        self.pc_raw_queue.append(msg)

    def left_master_arm_callback(self, msg):
        self.img_queues['left/master_arm'].append(msg)

    def left_slave_arm_callback(self, msg):
        self.img_queues['left/slave_arm'].append(msg)

    def right_master_arm_callback(self, msg):
        self.img_queues['right/master_arm'].append(msg)

    def right_slave_arm_callback(self, msg):
        self.img_queues['right/slave_arm'].append(msg)

    def process(self):
        # self.timesteps = []
        # self.actions = []
        # self.subtask_list = []  # 初始化子任务列表

        # 初始化图像字典
        image_dict = {name: np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
                      for name in self.args.camera_names}

        # 初始化深度图像字典
        image_depth_dict = {}
        if self.args.use_depth_image:
            image_depth_dict = {}
            for name in self.args.camera_names:
                if name not in ['cam_wrist']:  # 确保不会处理不存在的相机
                    image_depth_dict[name] = np.random.random((180, 320, 1))

        total_steps = int(self.args.max_timesteps)
        # 设置进度条的初始值为当前已有timesteps的长度
        initial_count = len(self.timesteps)
        pbar = tqdm(total=total_steps, desc='收集帧数据', ncols=80, initial=initial_count)

        # 从现有timesteps长度开始计数，而不是从0开始
        count = initial_count
        rate = rospy.Rate(self.args.frame_rate)

        while not rospy.is_shutdown() and (count < self.args.max_timesteps):
            # 检查是否暂停
            if self.key == 'space':
                rospy.loginfo(f'暂停收集数据，当前帧数: {count - 1}')
                self.collecting_data = False
                break

            # 获取一帧数据
            result = self.get_frame()
            if not result:
                rospy.logwarn('数据未准备好，等待中...')
                rate.sleep()
                continue

            count += 1
            pbar.update(1)
            left_master_arm, left_slave_arm = result['left/master_arm'], result['left/slave_arm']
            right_master_arm, right_slave_arm = result['right/master_arm'], result['right/slave_arm']

            # 更新图像数据
            for name in self.camera_names:
                if name in result['images']:
                    image_dict[name] = result['images'][name].copy()
                if self.args.use_depth_image and name in result['depths']:
                    image_depth_dict[name] = result['depths'][name].copy()

            # 构建观测数据
            obs = collections.OrderedDict()
            obs['images'] = copy.deepcopy(image_dict)

            # 添加深度图像数据
            if self.args.use_depth_image:
                obs['images_depth'] = copy.deepcopy(image_depth_dict)

            # 处理关节数据
            left_position = np.array(left_slave_arm.position)[:7]
            left_velocity = np.array(left_slave_arm.velocity)[:7]
            left_effort = np.array(left_slave_arm.effort)[:7]
            right_position = np.array(right_slave_arm.position)[:7]
            right_velocity = np.array(right_slave_arm.velocity)[:7]
            right_effort = np.array(right_slave_arm.effort)[:7]

            position = np.concatenate((left_position, right_position), axis=0)
            velocity = np.concatenate((left_velocity, right_velocity), axis=0)
            effort = np.concatenate((left_effort, right_effort), axis=0)

            # 添加关节数据到观测
            obs['qpos'] = position
            obs['qvel'] = velocity
            obs['effort'] = effort

            # 添加子任务信息
            obs['subtask'] = self.current_subtask
            self.subtask_list.append(self.current_subtask)

            # 添加点云数据（如果有）
            if self.args.save_pointcloud and result.get('pointclouds') and 'cam_top' in result['pointclouds']:
                pc6 = result['pointclouds']['cam_top']
                obs_pc = pc6[:, :].astype(np.float32)
                obs.setdefault('pointclouds', {})['cam_top'] = obs_pc

            # 创建时间步
            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                self.timesteps.append(ts)
                #pbar.update(1)
                rate.sleep()
                continue

            # 对于非第一帧，添加动作数据
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)

            # 提取动作数据
            left_position = np.array(left_master_arm.position)[:7]
            right_position = np.array(right_master_arm.position)[:7]
            combined_position = np.concatenate((left_position, right_position), axis=0)
            self.actions.append(np.array(combined_position))
            self.timesteps.append(ts)
            #pbar.update(1)
            rate.sleep()

        # 检查是否达到最大帧数
        if count >= self.args.max_timesteps:
            rospy.loginfo(f'达到最大帧数，停止收集数据，当前帧数: {count}')
            self.collecting_data = False

        # 打印收集的帧数信息
        rospy.loginfo(f"时间步数量: {len(self.timesteps)}")
        rospy.loginfo(f"动作数量: {len(self.actions)}")
        rospy.loginfo(f"子任务数量: {len(self.subtask_list)}")

# 主函数
def main():
    try:
        rospy.init_node('dual_arm_data_collection_node', anonymous=True)
        ros_operator = RosOperator()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("关闭数据收集节点")


# 入口点
if __name__ == '__main__':
    main()