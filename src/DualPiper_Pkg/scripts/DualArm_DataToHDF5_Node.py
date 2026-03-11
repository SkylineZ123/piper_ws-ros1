#coding=utf-8
import os
import time
import open3d as o3d
from tqdm import tqdm
import copy
import cv2
import h5py
import ros_numpy
import numpy as np
import dm_env
import collections
import sensor_msgs.point_cloud2 as pc2
from collections import deque
import rospy
from sensor_msgs.msg import JointState, Image,PointCloud2
from cv_bridge import CvBridge
from pynput import keyboard
import queue
import threading
from typing import Dict
from types import SimpleNamespace
import sys

# 全局队列，用于存储键盘事件
key_event_queue = queue.Queue()
def on_press_terminal(key):
    """
    键盘事件处理函数，将有效按键存入队列
    """
    try:
        if key.char in ['c', 'x', 's', 'r']:
            key_event_queue.put(key.char)
    except AttributeError:
        pass

def pointcloud2_to_xyz_array(pc_msg, remove_nans=True,visualize=True,
                             viz_window_name='PointCloud',max_distance=1.3,random_sample_num=30000):
    """更高效的 PointCloud2 → numpy (N,3)"""
    # 转为结构化数组
    pc = ros_numpy.numpify(pc_msg)
    # 提取 x, y, z 字段
    points = np.zeros((pc.shape[0], 6), dtype=np.float32)

    points[:, 0] = pc['x']
    points[:, 1] = pc['y']
    points[:, 2] = pc['z']

    rgb_raw = pc['rgb']
    raw_u32 = rgb_raw.view(np.uint32)
    r = ((raw_u32 >> 16) & 0xFF).astype(np.float32)
    g = ((raw_u32 >> 8) & 0xFF).astype(np.float32)
    b = (raw_u32 & 0xFF).astype(np.float32)
    points[:, 3:6] = np.stack((r, g, b), axis=-1)/255.0
    if remove_nans:
        mask = np.isfinite(points).all(axis=1)
        points = points[mask]
        # 5. 可视化（如果需要）
    # 计算欧氏距离并保留 0.6 m 以内的点
    dist = np.linalg.norm(points[:, :3], axis=1)
    mask_near = dist < max_distance
    points = points[mask_near]
    # 5. 随机采样 20000 个点（若点数不足则重复采样）
    if points.shape[0] > 0:
        if points.shape[0] >= random_sample_num:
            idx = np.random.choice(points.shape[0], random_sample_num, replace=False)
        else:
            # 点数不足时，允许重复采样
            idx = np.random.choice(points.shape[0], random_sample_num, replace=True)
        points = points[idx]
    if visualize:
        # Open3D 要求 colors 为 Nx3 且值在 [0,1]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(np.clip(points[:, 3:], 0.0, 1.0))
        pcd_down =pcd.farthest_point_down_sample(4096)
        pcd_down.paint_uniform_color([0, 1, 0])
        # 显示（阻塞式）
        o3d.visualization.draw_geometries([pcd], window_name=viz_window_name)

        o3d.visualization.draw_geometries([pcd_down], window_name=viz_window_name)
    return points


# def save_data(args, timesteps, actions, dataset_path):
#     """
#     将收集到的机器人数据（观测和动作）保存到HDF5文件。
#     Args:
#         args:命令行参数，包含相机名称、是否使用深度图像等。
#         timesteps(list):存储 dm_env.TimeStep 对象的列表，包含机器人观测数据。
#         actions(list):存储机器人实际发出的动作的列表
#         dataset_path(str):数据集保存的路径(不包含.hdf5后缀)
#     """
#     data_size = len(actions)
#     data_dict = {
#         '/observations/qpos': [],
#         '/observations/qvel': [],
#         '/observations/effort': [],
#         '/action': [],
#     }
#
#     for cam_name in args.camera_names:
#         data_dict[f'/observations/images/{cam_name}'] = []
#         if args.use_depth_image and cam_name not in ['cam_wrist','cam_wrist_left','cam_wrist_right']:
#             data_dict[f'/observations/images_depth/{cam_name}'] = []
#
#         if args.save_pointcloud and cam_name == 'cam_top':
#             data_dict[f'/observations/pointclouds/{cam_name}'] = []
#
#     while actions:
#         action = actions.pop(0)
#         ts = timesteps.pop(0)
#
#         data_dict['/observations/qpos'].append(ts.observation['qpos'])
#         data_dict['/observations/qvel'].append(ts.observation['qvel'])
#         data_dict['/observations/effort'].append(ts.observation['effort'])
#         data_dict['/action'].append(action)
#
#         for cam_name in args.camera_names:
#             data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
#             if args.use_depth_image and cam_name not in ['cam_wrist','cam_wrist_left','cam_wrist_right']:
#                 data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])
#             if args.save_pointcloud and cam_name == 'cam_top':
#                 # ts.observation['pointclouds'][cam_name] 应为 shape (4096,3) float32
#                 data_dict[f'/observations/pointclouds/{cam_name}'].append(ts.observation['pointclouds'][cam_name])
#
#     t0 = time.time()
#
#     with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
#         root.attrs['sim'] = False
#         root.attrs['compress'] = False
#         root.attrs['data_size'] = data_size
#
#         obs = root.create_group('observations')
#         image = obs.create_group('images')
#         for cam_name in args.camera_names:
#             _ = image.create_dataset(cam_name, (data_size, 180, 320, 3), dtype='uint8',
#                                      chunks=(1, 180, 320, 3), )
#
#         if args.use_depth_image:
#             image_depth = obs.create_group('images_depth')
#             for cam_name in args.camera_names:
#                 if cam_name not in ['cam_wrist','cam_wrist_left','cam_wrist_right']:
#                     _ = image_depth.create_dataset(cam_name, (data_size, 180, 320), dtype='uint16',
#                                                    chunks=(1, 180, 320), )
#
#         if args.save_pointcloud:
#             pc_group = obs.create_group('pointclouds')
#             for cam_name in args.camera_names:
#                 if cam_name == 'cam_top':  # 目前只对 cam_top 建立 pointcloud 存储，如需多相机，可去掉限制
#                     _ = pc_group.create_dataset(cam_name, (data_size, args.pc_target_points, 6), dtype='float32',
#                                                 chunks=(1, args.pc_target_points, 6), )
#
#         _ = obs.create_dataset('qpos', (data_size, 14))
#         _ = obs.create_dataset('qvel', (data_size, 14))
#         _ = obs.create_dataset('effort', (data_size, 14))
#         _ = root.create_dataset('action', (data_size, 14))
#
#         for name, array in data_dict.items():
#             root[name][...] = array
#
#     rospy.loginfo(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)
#

def save_data(args, timesteps, actions, dataset_path):
    """
    将收集到的机器人数据（观测和动作）保存到HDF5文件。
    """
    data_size = len(actions)
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }

    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image and cam_name not in ['cam_wrist', 'cam_wrist_left', 'cam_wrist_right']:
            data_dict[f'/observations/images_depth/{cam_name}'] = []
        if args.save_pointcloud and cam_name == 'cam_top':
            data_dict[f'/observations/pointclouds/{cam_name}'] = []

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)

        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)

        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image and cam_name not in ['cam_wrist', 'cam_wrist_left', 'cam_wrist_right']:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])
            if args.save_pointcloud and cam_name == 'cam_top':
                data_dict[f'/observations/pointclouds/{cam_name}'].append(ts.observation['pointclouds'][cam_name])

    # 动态推断各相机分辨率（从首帧读取），避免写死 180x320
    image_shapes = {}
    for cam_name in args.camera_names:
        first_img = data_dict[f'/observations/images/{cam_name}'][0]
        if first_img.ndim == 2:
            h, w = first_img.shape
            c = 1
        else:
            h, w, c = first_img.shape
        image_shapes[cam_name] = (h, w, c)

    depth_shapes = {}
    if args.use_depth_image:
        for cam_name in args.camera_names:
            if cam_name in ['cam_wrist', 'cam_wrist_left', 'cam_wrist_right']:
                continue
            first_depth = data_dict[f'/observations/images_depth/{cam_name}'][0]
            if first_depth.ndim == 3:
                h, w = first_depth.shape[:2]
            else:
                h, w = first_depth.shape
            depth_shapes[cam_name] = (h, w)

    t0 = time.time()

    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        root.attrs['data_size'] = data_size

        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            h, w, c = image_shapes[cam_name]
            if c == 1:
                _ = image.create_dataset(
                    cam_name, (data_size, h, w), dtype='uint8',
                    chunks=(1, h, w)
                )
            else:
                _ = image.create_dataset(
                    cam_name, (data_size, h, w, c), dtype='uint8',
                    chunks=(1, h, w, c)
                )

        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                if cam_name in ['cam_wrist', 'cam_wrist_left', 'cam_wrist_right']:
                    continue
                h, w = depth_shapes[cam_name]
                _ = image_depth.create_dataset(
                    cam_name, (data_size, h, w), dtype='uint16',
                    chunks=(1, h, w)
                )

        if args.save_pointcloud:
            pc_group = obs.create_group('pointclouds')
            for cam_name in args.camera_names:
                if cam_name == 'cam_top':
                    _ = pc_group.create_dataset(
                        cam_name, (data_size, args.pc_target_points, 6), dtype='float32',
                        chunks=(1, args.pc_target_points, 6)
                    )

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))

        for name, array in data_dict.items():
            root[name][...] = array

    rospy.loginfo(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n' % dataset_path)


class RosOperator():
    """
    ROS节点，用于收集机器人数据（图像、关节状态）并保存到HDF5文件。
    """
    def __init__(self):
        """
        初始化 RosOperator 节点
        """
        rospy.init_node('collect_data_node', anonymous=True)
        # --- 获取参数 ---
        p = lambda name, default: rospy.get_param("~" + name, default)
        # 话题
        img_top_topic = p('img_top_topic', '/cam_top/cam_top/color/image_raw')
        img_leftwrist_topic = p('img_leftwrist_topic', '/cam_leftwrist/cam_leftwrist/color/image_raw')
        img_rightwrist_topic = p('img_rightwrist_topic', '/cam_rightwrist/cam_rightwrist/color/image_raw')
        img_top_depth_topic = p('img_top_depth_topic', '/cam_top/cam_top/aligned_depth_to_color/image_raw')
        img_leftwrist_depth_topic = p('img_leftwrist_depth_topic', '/cam_leftwrist/cam_leftwrist/aligned_depth_to_color/image_raw')
        img_rightwrist_depth_topic = p('img_rightwrist_depth_topic', '/cam_rightwrist/cam_rightwrist/aligned_depth_to_color/image_raw')
        # 机械臂关节状态话题
        left_master_arm_topic = p('left_master_arm_topic', 'left/master_joint_states')
        left_slave_arm_topic = p('left_slave_arm_topic', 'left/slave_joint_states')
        right_master_arm_topic = p('right_master_arm_topic', 'right/master_joint_states')
        right_slave_arm_topic = p('right_slave_arm_topic', 'right/slave_joint_states')
        # 新增：点云 topic（默认使用你指定的 topic）

        pc_top_topic = p('pc_top_topic', '/cam_top/cam_top/depth/color/points')
        # 是否保存点云
        save_pointcloud = p('save_pointcloud', False)
        pc_queue_size = p('pc_queue_size', 3)  # raw pointcloud 队列深度（尽量小）
        target_pc_points = p('pc_target_points', 20000)  # 最终要保存的点数（例如 4096）
        dataset_dir = p('dataset_dir', '/home/user/zzq/piper_ws/data')
        task_name = p('task_name', 'bloodgas_0306')
        episode_idx = p('episode_idx', 3)
        max_timesteps = p('max_timesteps', 5000)
        camera_names_str = p('camera_names', 'cam_top,cam_left,cam_right')
        camera_names = [c.strip() for c in camera_names_str.split(',') if c.strip()]
        use_depth_image = p('use_depth_image', False)
        frame_rate = p('frame_rate', 20)

        args = SimpleNamespace(
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
            pc_top_topic=pc_top_topic,  # <-- 新增
            save_pointcloud=save_pointcloud,  # <-- 新增
            pc_queue_size = pc_queue_size,
            pc_target_points=target_pc_points
        )
        self.args = args
        self.bridge = CvBridge()
        self.camera_names = self.args.camera_names
        
        # 图像和关节状态队列
        self.img_queues = {
            'cam_left': deque(maxlen=5),
            'cam_right': deque(maxlen=5),
            'cam_top': deque(maxlen=5),
            'left/master_arm': deque(maxlen=50),
            'left/slave_arm': deque(maxlen=50),
            'right/master_arm': deque(maxlen=50),
            'right/slave_arm': deque(maxlen=50),
        }
        self.pc_raw_queue = collections.deque(maxlen=self.args.pc_queue_size)

        # 订阅者
        self.img_top_sub = rospy.Subscriber(args.img_top_topic, Image, self.img_top_callback, queue_size=50)

        if 'cam_left' in self.camera_names:
            self.img_leftwrist_sub = rospy.Subscriber(args.img_leftwrist_topic, Image, self.img_leftwrist_callback, queue_size=50)
        if 'cam_right' in self.camera_names:
            self.img_rightwrist_sub = rospy.Subscriber(args.img_rightwrist_topic,Image,self.img_rightwrist_callback,queue_size=50)

        self.left_master_arm_sub = rospy.Subscriber(args.left_master_arm_topic, JointState, self.left_master_arm_callback, queue_size=50)
        self.left_slave_arm_sub = rospy.Subscriber(args.left_slave_arm_topic, JointState, self.left_slave_arm_callback, queue_size=50)
        self.right_master_arm_sub = rospy.Subscriber(args.right_master_arm_topic, JointState, self.right_master_arm_callback, queue_size=50)
        self.right_slave_arm_sub = rospy.Subscriber(args.right_slave_arm_topic, JointState, self.right_slave_arm_callback, queue_size=50)
        
        if self.args.use_depth_image:
            self.img_top_depth_sub = rospy.Subscriber(args.img_top_depth_topic, Image, self.img_top_depth_callback, queue_size=50)
            if 'cam_left' in self.camera_names:
                self.img_leftwrist_depth_sub = rospy.Subscriber(args.img_leftwrist_depth_topic, Image, self.img_leftwrist_depth_callback, queue_size=50)

        if self.args.save_pointcloud:
            self.pc_top_sub = rospy.Subscriber(self.args.pc_top_topic, PointCloud2, self.pc_top_callback, queue_size=10)

        self.key = None
        self.term_thread = threading.Thread(target=self.terminal_thread)
        self.key_process_thread = threading.Thread(target=self.key_process_loop)
        self.term_thread.daemon = True
        self.key_process_thread.daemon = True
        self.term_thread.start()
        self.key_process_thread.start()

        self.timesteps, self.actions = [], []
        self.collecting_data = False
        
        rospy.loginfo(f'\033[32m The Node of Robot data collection. \033[0m')
        rospy.loginfo(f'\033[32m Press "c" to start collecting data. \033[0m')
        rospy.loginfo(f'\033[32m Press "x" to stop collecting data. \033[0m')
        rospy.loginfo(f'\033[32m Press "s" to save collecting data. \033[0m')

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
                rospy.loginfo(f'Start collecting data for episode {episode_idx}')
                self.process()
                self.key = None
            elif self.key == 's' and self.collecting_data:
                rospy.loginfo(f'To Save collecting data to file for episode {episode_idx}')
                dataset_dir = os.path.join(self.args.dataset_dir, self.args.task_name)
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}")
                save_data(self.args, self.timesteps, self.actions, dataset_path)
                self.collecting_data = False
                episode_idx += 1
                self.key = None
            elif self.key == 'r':
                rospy.loginfo(f'Clear the flag of collectted data')
                self.timesteps = []
                self.actions = []
                self.collecting_data = False
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
                        self.img_queues[k][0].header.stamp.secs + 1e-9 * self.img_queues[k][0].header.stamp.nsecs < sync_time
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
        # # 如果保存点云，则把点云队列也加入同步判定
        # if self.args.save_pointcloud:
        #     keys.append('cam_top_points')

        if use_arm:
            keys.extend(['left/master_arm', 'left/slave_arm', 'right/master_arm', 'right/slave_arm'])
        
        if any(len(self.img_queues[k]) == 0 for k in keys):
            return False
        
        sync_time = _get_latest_time(keys)
        if sync_time is None:
            return False

        _sync_queues(sync_time, keys)

        if any(len(self.img_queues[k]) == 0 for k in keys):
            return False

        images = {}
        for key in self.camera_names:
            if key in self.img_queues and self.img_queues[key]:
                images[key] = self.bridge.imgmsg_to_cv2(self.img_queues[key].popleft(), desired_encoding='passthrough')
        
        images_depth = None
        if self.args.use_depth_image:
            images_depth = {}
            for key in keys:
                if 'depth' in key and self.img_queues[key]:
                    name = key.rsplit('_', 1)[0]
                    images_depth[name] = self.bridge.imgmsg_to_cv2(self.img_queues[key].popleft(), desired_encoding='16UC1')
        # 新增点云读取：把 PointCloud2 msg 转成 Nx3 numpy array（x,y,z）
        pointclouds = None
        if self.args.save_pointcloud:
            # 目前只处理 cam_top 的点云（若需要支持更多相机，可遍历 camera_names）
            pc_msg = self.pc_raw_queue.popleft()
            #帮我计算一下读取point_list的消耗时间
            pointclouds_arr = pointcloud2_to_xyz_array(pc_msg,remove_nans=True)
            pointclouds = {"cam_top":pointclouds_arr}

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
        self.timesteps = []
        self.actions = []

        image_dict = {name: np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
                      for name in self.args.camera_names}
        image_depth_dict = {
            name: np.random.random((180, 320, 1))
            for name in self.args.  camera_names
            if name not in ['cam_wrist','cam_left', 'cam_right']
        } if self.args.use_depth_image else {}

        total_steps = int(self.args.max_timesteps)
        pbar = tqdm(total=total_steps, desc='Collecting frames', ncols=80)

        count, rate = 0, rospy.Rate(self.args.frame_rate)
        
        while not rospy.is_shutdown() and (count <= self.args.max_timesteps):
            result = self.get_frame()
            if not result:
                rospy.logwarn('State not ready yet, waiting for data...')
                rate.sleep()
                continue
            
            count += 1
            left_master_arm, left_slave_arm = result['left/master_arm'],result['left/slave_arm']
            right_master_arm, right_slave_arm = result['right/master_arm'],result['right/slave_arm']


            for i, name in enumerate(self.camera_names):
                image_dict[name] = result['images'][name].copy()
                if self.args.use_depth_image and name in result['depths']:
                    image_depth_dict[name] = result['depths'][name].copy()
            obs = collections.OrderedDict()
            obs['images']=copy.deepcopy(image_dict)
            if self.args.use_depth_image:
                obs['images_depth'] = copy.deepcopy(image_depth_dict)
            left_position = np.array(left_slave_arm.position)[:7]
            left_velocity = np.array(left_slave_arm.velocity)[:7]
            left_effort = np.array(left_slave_arm.effort)[:7]
            right_position = np.array(right_slave_arm.position)[:7]
            right_velocity = np.array(right_slave_arm.velocity)[:7]
            right_effort = np.array(right_slave_arm.effort)[:7]

            position = np.concatenate((left_position, right_position), axis=0)
            velocity = np.concatenate((left_velocity, right_velocity), axis=0)
            effort = np.concatenate((left_effort, right_effort), axis=0)

            if self.args.save_pointcloud and result.get('pointclouds') and 'cam_top' in result['pointclouds']:
                pc6 = result['pointclouds']['cam_top']
                obs_pc = pc6[:, :].astype(np.float32)
                obs.setdefault('pointclouds', {})['cam_top'] = obs_pc
            obs['qpos'] = position
            obs['qvel'] = velocity
            obs['effort'] = effort

            if count == 1:
                ts = dm_env.TimeStep(
                    step_type=dm_env.StepType.FIRST,
                    reward=None,
                    discount=None,
                    observation=obs)
                self.timesteps.append(ts)
                continue
            
            ts = dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=None,
                discount=None,
                observation=obs)
            
            left_position = np.array(left_master_arm.position)[:7]
            right_position = np.array(right_master_arm.position)[:7]
            combined_position = np.concatenate((left_position, right_position), axis=0)
            self.actions.append(np.array(combined_position))
            self.timesteps.append(ts)
            pbar.update(1)

            
            rate.sleep()
            if count > self.args.max_timesteps:
                rospy.loginfo('Stop collecting data, frame_index:{}'.format(count - 1))
                self.collecting_data = True
                break
            if self.key == 'x':
                rospy.loginfo('Stop collecting data, frame_index:{}'.format(count - 1))
                self.collecting_data = True
                break

        rospy.loginfo("len(timesteps): " + str(len(self.timesteps)))
        rospy.loginfo("len(actions)  : " + str(len(self.actions)))
        
def main():
    try:
        ros_operator = RosOperator()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down data collection node.")

if __name__ == '__main__':
    main()