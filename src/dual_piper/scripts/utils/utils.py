#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for the dual_piper package.

Provides:
- Image conversion (ROS ↔ OpenCV)
- Point-cloud helpers
- Keyboard event handler
- HDF5 save routines
- Common constants
"""
import os
import time
import queue
import threading

import numpy as np
import h5py
import rospy
import sensor_msgs.point_cloud2 as pc2
from pynput import keyboard

# ═══════════════════════════════ Constants ═══════════════════════════════
# Piper SDK uses milli-degrees internally; convert to radians.
DEG_TO_RAD = 0.01745329251994  # π / 180
JOINT_SCALE = 1000.0           # SDK joint unit → milli-degrees
GRIPPER_SCALE = 1_000_000.0    # SDK gripper unit → angle
EFFORT_SCALE = 1000.0          # SDK effort unit
RAD_TO_DEG_X1000 = 180.0 / np.pi * 1000.0  # rad → SDK milli-degrees
ENABLE_ALL_JOINTS = 7          # EnableArm(7) enables all 6 joints + gripper
PUBLISH_RATE_HZ = 200          # Default joint-state publish rate
NUM_JOINTS_PER_ARM = 7         # 6 joints + 1 gripper
NUM_JOINTS_DUAL = 14           # 7 × 2 arms
WRIST_CAMERAS = frozenset(['cam_wrist', 'cam_wrist_left', 'cam_wrist_right'])


# ═════════════════════════ Image Conversion ══════════════════════════════
def imgmsg_to_cv2(img_msg):
    """
    Pure-Python ROS Image → OpenCV (NumPy) conversion.

    Supported encodings: ``rgb8``, ``bgr8``.

    Returns:
        numpy.ndarray: ``(H, W, 3)`` BGR uint8 image.
    """
    if img_msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f"Unsupported encoding: {img_msg.encoding}")

    image = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3
    )
    if img_msg.encoding == 'rgb8':
        image = image[:, :, ::-1]  # RGB → BGR
    return image


# ═════════════════════════ Point Cloud ═══════════════════════════════════
def sample_pointcloud(pc_msg, target_points=5000):
    """
    Convert ``PointCloud2`` → ``(target_points, 6)`` float32 ``[x,y,z,r,g,b]``.

    If the cloud has only XYZ, the RGB columns are zero-filled.
    Points are randomly down-/up-sampled to exactly *target_points*.
    """
    pts = [list(p[:6]) for p in pc2.read_points(pc_msg, skip_nans=True)]
    if not pts:
        return np.zeros((target_points, 6), dtype=np.float32)

    arr = np.asarray(pts, dtype=np.float32)
    # Pad to 6 columns if fewer
    if arr.shape[1] < 6:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 6 - arr.shape[1]), dtype=np.float32)])
    else:
        arr = arr[:, :6]

    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    if arr.shape[0] == 0:
        return np.zeros((target_points, 6), dtype=np.float32)

    n = arr.shape[0]
    idx = np.random.choice(n, target_points, replace=(n < target_points))
    return arr[idx]


def sample_and_expand_pointcloud(pc, target_points):
    """
    Normalize a raw point array ``(N, 3|6)`` → ``(target_points, 6)`` float32.

    Pads RGB with zeros if only XYZ is provided; up-/down-samples to
    *target_points*.
    """
    if pc is None:
        return np.zeros((target_points, 6), dtype=np.float32)
    pc = np.asarray(pc)
    if pc.size == 0:
        return np.zeros((target_points, 6), dtype=np.float32)

    if pc.ndim == 1:
        pc = pc.reshape(-1, pc.shape[0])
    cols = pc.shape[1]
    if cols == 3:
        pc6 = np.hstack([pc.astype(np.float32),
                          np.zeros((pc.shape[0], 3), dtype=np.float32)])
    elif cols >= 6:
        pc6 = pc[:, :6].astype(np.float32)
    else:
        pc6 = np.hstack([pc.astype(np.float32),
                          np.zeros((pc.shape[0], 6 - cols), dtype=np.float32)])

    mask = np.isfinite(pc6).all(axis=1)
    pc6 = pc6[mask]
    if pc6.shape[0] == 0:
        return np.zeros((target_points, 6), dtype=np.float32)

    n = pc6.shape[0]
    idx = np.random.choice(n, target_points, replace=(n < target_points))
    return pc6[idx].astype(np.float32)


# ═════════════════════════ Keyboard Handler ══════════════════════════════
class KeyboardHandler:
    """
    Thread-safe keyboard event handler using ``pynput``.

    Usage::

        kb = KeyboardHandler(valid_keys={'c', 'x', 's', 'r', 'space'})
        kb.start()
        ...
        key = kb.get()  # non-blocking, returns str or None
    """

    def __init__(self, valid_keys=None):
        self._queue = queue.Queue()
        self._valid = set(valid_keys) if valid_keys else None
        self._listener = None

    # --- callbacks ---
    def _on_press(self, event):
        key_str = None
        if hasattr(event, 'char') and event.char is not None:
            key_str = event.char
        elif hasattr(event, 'name'):
            key_str = event.name

        if key_str and (self._valid is None or key_str in self._valid):
            self._queue.put(key_str)

    # --- public API ---
    def start(self):
        """Start the keyboard listener in a daemon thread."""
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def get(self, timeout=0.1):
        """Return the next key press, or ``None`` if the queue is empty."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ══════════════════════ ROS Timestamp Helper ═════════════════════════════
def get_ros_stamp_sec(msg):
    """Extract a float-seconds timestamp from a ROS message header."""
    return msg.header.stamp.secs + 1e-9 * msg.header.stamp.nsecs


# ═══════════════════════ HDF5 Save (basic) ══════════════════════════════
def save_hdf5_data(args, timesteps, actions, dataset_path):
    """
    Save collected robot data (observations + actions) to an HDF5 file.

    Dynamically infers image resolution from the first frame so that
    the caller does not need to hard-code ``180×320``.

    Args:
        args: Namespace with at least ``camera_names``, ``use_depth_image``,
              ``save_pointcloud``, ``pc_target_points``.
        timesteps: list of ``dm_env.TimeStep`` objects.
        actions: list of action arrays.
        dataset_path: str — output path **without** ``.hdf5`` suffix.
    """
    data_size = len(actions)
    if data_size == 0:
        rospy.logwarn("No actions to save — aborting.")
        return

    # Check for optional subtask field
    has_subtask = (
        timesteps
        and isinstance(timesteps[0].observation, dict)
        and 'subtask' in timesteps[0].observation
    )

    # ── Collect lists ────────────────────────────────────────────────────
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }
    if has_subtask:
        data_dict['/observations/subtask'] = []

    for cam in args.camera_names:
        data_dict[f'/observations/images/{cam}'] = []
        if args.use_depth_image and cam not in WRIST_CAMERAS:
            data_dict[f'/observations/images_depth/{cam}'] = []
        if args.save_pointcloud and cam == 'cam_top':
            data_dict[f'/observations/pointclouds/{cam}'] = []

    ts_list = list(timesteps)
    act_list = list(actions)
    while act_list:
        action = act_list.pop(0)
        ts = ts_list.pop(0)
        obs = ts.observation

        data_dict['/observations/qpos'].append(np.asarray(obs['qpos'], dtype=np.float32))
        data_dict['/observations/qvel'].append(np.asarray(obs['qvel'], dtype=np.float32))
        data_dict['/observations/effort'].append(np.asarray(obs['effort'], dtype=np.float32))
        data_dict['/action'].append(np.asarray(action, dtype=np.float32))

        if has_subtask:
            data_dict['/observations/subtask'].append(str(obs.get('subtask', '')))

        for cam in args.camera_names:
            data_dict[f'/observations/images/{cam}'].append(
                np.asarray(obs['images'][cam], dtype=np.uint8)
            )
            if args.use_depth_image and cam not in WRIST_CAMERAS:
                data_dict[f'/observations/images_depth/{cam}'].append(
                    np.asarray(obs['images_depth'][cam], dtype=np.uint16)
                )
            if args.save_pointcloud and cam == 'cam_top':
                pc_raw = obs.get('pointclouds', {}).get(cam)
                pc6 = sample_and_expand_pointcloud(pc_raw, args.pc_target_points)
                data_dict[f'/observations/pointclouds/{cam}'].append(pc6)

    # ── Infer shapes from first frame ────────────────────────────────────
    img_shapes = {}
    for cam in args.camera_names:
        first = data_dict[f'/observations/images/{cam}'][0]
        img_shapes[cam] = first.shape  # (H, W, C) or (H, W)

    depth_shapes = {}
    if args.use_depth_image:
        for cam in args.camera_names:
            if cam in WRIST_CAMERAS:
                continue
            first = data_dict[f'/observations/images_depth/{cam}'][0]
            depth_shapes[cam] = first.shape[:2]  # (H, W)

    # ── Write HDF5 ───────────────────────────────────────────────────────
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        root.attrs['data_size'] = data_size

        obs_grp = root.create_group('observations')

        # Images
        img_grp = obs_grp.create_group('images')
        for cam in args.camera_names:
            shape = img_shapes[cam]
            img_grp.create_dataset(
                cam, (data_size, *shape), dtype='uint8',
                chunks=(1, *shape),
            )

        # Depth images
        if args.use_depth_image:
            depth_grp = obs_grp.create_group('images_depth')
            for cam in args.camera_names:
                if cam in WRIST_CAMERAS:
                    continue
                h, w = depth_shapes[cam]
                depth_grp.create_dataset(
                    cam, (data_size, h, w), dtype='uint16', chunks=(1, h, w),
                )

        # Point clouds
        if args.save_pointcloud:
            pc_grp = obs_grp.create_group('pointclouds')
            for cam in args.camera_names:
                if cam == 'cam_top':
                    pc_grp.create_dataset(
                        cam,
                        (data_size, args.pc_target_points, 6),
                        dtype='float32',
                        chunks=(1, args.pc_target_points, 6),
                    )

        # Scalar observations & actions
        obs_grp.create_dataset('qpos', (data_size, NUM_JOINTS_DUAL), dtype='float32')
        obs_grp.create_dataset('qvel', (data_size, NUM_JOINTS_DUAL), dtype='float32')
        obs_grp.create_dataset('effort', (data_size, NUM_JOINTS_DUAL), dtype='float32')
        root.create_dataset('action', (data_size, NUM_JOINTS_DUAL), dtype='float32')

        # Subtask (variable-length strings)
        if has_subtask:
            vlen_str = h5py.special_dtype(vlen=str)
            obs_grp.create_dataset('subtask', (data_size,), dtype=vlen_str)

        # Bulk write
        for name, arr_list in data_dict.items():
            if name == '/observations/subtask':
                root[name][...] = arr_list
            else:
                root[name][...] = np.array(arr_list)

    elapsed = time.time() - t0
    rospy.loginfo(f'\033[32mSaved {data_size} frames in {elapsed:.1f}s → {dataset_path}.hdf5\033[0m')
