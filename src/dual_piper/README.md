# dual_piper

A **ROS 1 (Noetic)** package for dual-arm [Piper](https://github.com/agilexrobotics) robot teleoperation, data collection, replay, and dataset conversion.

> This package replaces the legacy `DualPiper_Pkg` with ROS-standard snake_case naming, cleaner code, shared utility modules, and a proper launch / config structure.

---

## Features

| Node | Script | Description |
|------|--------|-------------|
| **State Publisher** | `state_get_node.py` | Connects to two Piper arms via CAN, publishes `JointState` at 200 Hz |
| **HDF5 Collector** | `data_to_hdf5_node.py` | Keyboard-driven data collection → HDF5 episodes |
| **HDF5 Collector (High-level)** | `data_to_hdf5_highlevel_node.py` | Adds subtask labelling, pause/resume, per-subtask directories |
| **Data Replay** | `data_replay_node.py` | Loads HDF5 episodes, plots joint curves, stitches camera videos, optional real-robot replay |
| **LeRobot v3 Collector** | `data_to_lerobot3_node.py` | Writes episodes directly into LeRobot v3 (Parquet + MP4) format |

---

## Package Layout

```
dual_piper/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── subtasks.yaml          # Default subtask definitions
├── launch/
│   ├── teleoperation.launch   # Arms + cameras
│   ├── collect_data.launch    # Teleoperation + HDF5 collector
│   └── handeye_calib.launch   # Teleoperation + ArUco + easy_handeye
└── scripts/
    ├── state_get_node.py
    ├── data_replay_node.py
    ├── data_to_hdf5_node.py
    ├── data_to_hdf5_highlevel_node.py
    ├── data_to_lerobot3_node.py
    └── utils/
        ├── __init__.py
        └── utils.py           # Shared constants, helpers, keyboard handler
```

---

## Prerequisites

* **ROS Noetic** on Ubuntu 20.04
* Python ≥ 3.8
* [piper_sdk](https://github.com/agilexrobotics) — Piper arm CAN SDK
* Intel RealSense cameras + `realsense2_camera` ROS package
* `piper_msgs`, `piper_description` packages (available in this workspace)

### Python Dependencies

```bash
pip install numpy opencv-python h5py matplotlib tqdm dm-env pynput pyyaml
# For LeRobot v3 support:
pip install lerobot
```

---

## Quick Start

### 1. Start the Teleoperation System

```bash
roslaunch dual_piper teleoperation.launch
```

This brings up both arms and three RealSense cameras. You can override CAN ports:

```bash
roslaunch dual_piper teleoperation.launch \
    piper_can_left_port:=can0 \
    piper_can_right_port:=can1
```

### 2. Collect Data (HDF5)

```bash
roslaunch dual_piper collect_data.launch
```

Or run the collector standalone with custom parameters:

```bash
rosrun dual_piper data_to_hdf5_node.py \
    _dataset_dir:=/path/to/data \
    _task_name:=my_task \
    _frame_rate:=20
```

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `c` | Start collecting |
| `x` | Stop collecting |
| `s` | Save episode to HDF5 |
| `r` | Discard buffered data |

### 3. Collect Data with Subtask Labels

```bash
rosrun dual_piper data_to_hdf5_highlevel_node.py \
    _subtasks_file:=$(rospack find dual_piper)/config/subtasks.yaml
```

**Additional controls:**

| Key | Action |
|-----|--------|
| `space` | Pause collection |
| `1`–`N` | Switch active subtask (while paused) |

### 4. Replay an Episode

```bash
rosrun dual_piper data_replay_node.py \
    --dataset_dir /path/to/data \
    --task_name my_task \
    --episode_idx 0
```

This generates:
* Joint-position / action plots (`output/qpos_action_*.png`)
* Stitched multi-camera video (`output/stitched_video_episode*.mp4`)

### 5. LeRobot v3 Direct Collection

```bash
rosrun dual_piper data_to_lerobot3_node.py \
    _repo_id:=user/MyDataset \
    _target_dir:=/path/to/lerobot_dataset
```

| Key | Action |
|-----|--------|
| `c` | Start / resume |
| `space` | Pause |
| `1`–`4` | Switch subtask |
| `s` | Save episode |
| `r` | Discard buffer |
| `q` | Finalize dataset & exit |

---

## Configuration

### ROS Parameters

All nodes support private ROS parameters (`~param_name`). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~left_can_port` | `can_left` | CAN interface for left arm |
| `~right_can_port` | `can_right` | CAN interface for right arm |
| `~auto_enable` | `true` | Auto-enable arms on startup |
| `~dataset_dir` | `/home/user/zzq/piper_ws/data` | Base directory for datasets |
| `~task_name` | `bloodgas_0306` | Task name (subdirectory) |
| `~episode_idx` | `3` | Starting episode index |
| `~max_timesteps` | `5000` | Max frames per episode |
| `~frame_rate` | `20` | Collection rate (Hz) |
| `~camera_names` | `cam_top,cam_left,cam_right` | Comma-separated camera list |
| `~use_depth_image` | `false` | Collect depth frames |
| `~save_pointcloud` | `false` | Collect point-cloud data |

### Subtask Definitions

Edit `config/subtasks.yaml` to define your own subtask set:

```yaml
subtasks:
  '1': 'pick up the object'
  '2': 'place the object'
  '3': 'return to home position'
```

---

## Migration from DualPiper_Pkg

| Old (DualPiper_Pkg) | New (dual_piper) |
|---------------------|------------------|
| `DualPiper_StateGet_Node.py` | `state_get_node.py` |
| `DualArm_DataReplay_Node.py` | `data_replay_node.py` |
| `DualArm_DataToHDF5_Node.py` | `data_to_hdf5_node.py` |
| `DualArm_DataToHDF5_Node_High_level_1.py` | `data_to_hdf5_highlevel_node.py` |
| `DualArm_DataToLerobot3_Node.py` | `data_to_lerobot3_node.py` |
| `DualPiper_TeleoperationSystem.launch` | `teleoperation.launch` |
| `DualPiper_CollectData.launch` | `collect_data.launch` |
| `single_piper_handeye_calib.launch` | `handeye_calib.launch` |

---

## License

MIT — see `package.xml` for details.
