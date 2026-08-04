# Piper 双臂数据采集操作指南 (lerobot v6)

本工作区仿写自 `piper_ws-ros1--master1`, 数据集写入改为 **lerobot v6**
(`codebase_version: "v3.0"` 格式: `data/*.parquet` + `videos/` + `meta/`),
并**去除了全部 subtask 逻辑** (单任务采集, 无 `subtask_index`/`back_event` 列)。

## 环境要求

- Ubuntu 20.04 + ROS Noetic
- conda 环境 `zhl_lerobot` (Python 3.12, 含 lerobot v6、piper_sdk、rospy、cv_bridge)
- Intel RealSense 相机（3 台：顶部 + 左右手腕）
- Piper 双臂机器人（CAN 总线通信）
- NVIDIA GPU（可选，用于 NVENC 硬件编码）

## 1. 工作空间编译

注意：构建与运行都必须在 `zhl_lerobot` conda 环境内进行（构建时 catkin 会把
脚本 shebang 改写为该环境的 python，保证 lerobot 可导入）：

```bash
cd ~/piper_ws_lerobotv6
source /opt/ros/noetic/setup.bash
conda activate zhl_lerobot
catkin_make
source devel/setup.bash
```

（conda 环境需已安装 `empy==3.3.4`：`pip install empy==3.3.4`）

每次运行采集/回放节点前：

```bash
source /opt/ros/noetic/setup.bash
conda activate zhl_lerobot
source ~/piper_ws_lerobotv6/devel/setup.bash
```

## 2. 配置 CAN 总线

两个 Piper 机械臂通过 CAN 总线通信，需要先配置接口：

```bash
sudo bash can_config.sh
```

该脚本会：
- 检测两个 CAN 模块的 USB 端口地址
- 自动重命名为 `can_left`（左臂）和 `can_right`（右臂）
- 设置波特率为 1 Mbps

验证 CAN 接口是否正常：

```bash
bash find_all_can_port.sh
```

## 3. 修改配置文件

编辑 `src/dual_piper/config/collection.yaml`：

### 3.1 数据存储路径（必须！）

```yaml
dataset:
  target_dir: "/home/zhl/data/CubeTransfer0804"
```

### 3.2 数据集参数

```yaml
dataset:
  repo_id: "RoboMLT/CubeTransfer"   # 数据集名称
  task_name: "cube transfer"        # 任务名称（单一任务，无子任务）
  fps: 25                           # 采集帧率
  resume: false                     # false=新建数据集, true=追加到已有
  video_codec: "h264_nvenc"         # 视频编码器（无 GPU 改为 h264/libsvtav1）
```

### 3.3 相机序列号（确保与实物一致）

```yaml
camera:
  serial_no_top:   "327122074756"
  serial_no_left:  "148522072680"
  serial_no_right: "241222073777"
```

## 4. 启动遥操作系统

此命令会启动双臂状态发布 + 三台相机：

```bash
roslaunch dual_piper teleoperation.launch
```

启动成功后，可使用以下命令检查所有 topic：

```bash
rostopic list | grep -E "cam_|joint_states"
```

## 5. 运行数据采集

```bash
rosrun dual_piper data_to_lerobot3_node.py
```

或指定自定义配置文件：

```bash
rosrun dual_piper data_to_lerobot3_node.py _config_file:=/home/zhl/piper_ws_lerobotv6/src/dual_piper/config/collection.yaml
```

## 6. 键盘操控说明

| 按键 | 功能 | 使用时机 |
|------|------|----------|
| `c` | 开始 / 恢复采集 | 就绪后 |
| `space` | 暂停采集 | 完成一段动作后 |
| `s` | 保存当前 episode | 完成一个完整流程后 |
| `r` | 丢弃当前缓存帧 | 不满意当前录制时 |
| `q` | 保存并退出程序 | 全部采集结束时 |

## 7. 典型采集流程

```
1. 双手遥操作控制机械臂到初始位置
2. 按 c 开始录制 → 操控手臂完成整个任务流程 → 按 space 暂停
3. 按 s 保存该 episode
4. 开始下一个 episode 时重复步骤 1-3
5. 所有采集完成后按 q 退出
```

> **Episode 概念**：一次完整的任务流程 = 1 个 episode。每个 episode 独立保存为
> 三段 MP4 视频（每台相机一段）+ Parquet 数据。

## 8. 实时预览窗口

采集过程中会弹出 OpenCV 窗口，显示：

- 三台相机的实时画面（平铺排列）
- 当前任务名称、录制状态、实时 FPS、已录制帧数、已保存 episode 数量

也可直接在预览窗口按 `q` 或 `ESC` 退出程序。

## 9. 崩溃恢复

如果程序异常崩溃（断电、进程被 kill 等）：

```yaml
# collection.yaml 中设置
dataset:
  resume: true
```

重新启动采集节点，脚本会自动：
1. 检测已有数据集
2. 运行 `repair_dataset_consistency()` 修复不一致的元数据
   （清理崩溃残留临时目录、截断到已落盘的 episode 元数据、重算统计）
3. 从上次中断处继续追加数据

## 10. 数据回放与验证

采集完成后可回放检查数据质量：

```bash
rosrun dual_piper data_replay_node.py --config src/dual_piper/config/collection.yaml
```

功能：
- 加载 v6 数据集
- 生成 joint-position / action 曲线图
- 拼接多相机视频
- 可选：在真实机械臂上复现轨迹（`--replay_on_robot`）

## 11. 数据增强（可选）

对高质量的 episode 进行复制增强：

```bash
rosrun dual_piper data_to_lerobot3_node_repeat.py \
    --config src/dual_piper/config/collection.yaml --episode 0 --copies 2
```

每份副本经 v6 重新编码为独立视频。

## 12. 常见问题

### Q: 相机无法启动？
检查 USB 3.0 带宽，确保三台相机都插在 USB 3.0 接口上。可降低 FPS 或分辨率。

### Q: CAN 接口配置失败？
```bash
sudo ethtool -i can0 | grep bus   # 查看当前 CAN 模块的 USB 地址
```
确认 USB 地址与 `can_config.sh` 中预定义的一致，如不一致需修改脚本。

### Q: 数据采集帧率不稳定？
- 关闭视频预览：`camera.enable_view: false`
- 确保数据存储路径是 SSD 或高速硬盘
- 使用 NVENC 硬件编码（需要 NVIDIA GPU）
- 检查 CPU 负载：`htop`

### Q: 需要单臂采集？
在 `collection.yaml` 中修改：
```yaml
robot:
  robot_type: "single_piper_left"   # 仅左臂
  # robot_type: "single_piper_right"  # 仅右臂
```

### Q: 图像转换报 `undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0`？
conda 环境自带的 `libffi.so.7` / `libffi.7.so` 是指向 libffi 8.x 的兼容链接，
与系统库（libp11-kit/gnutls 等）要求的 `LIBFFI_BASE_7.0` 符号版本冲突，导致
相机图像转换失败、采集 0 帧。删除环境里这两个多余链接即可（已确认无任何
conda 库真正需要它们）：

```bash
rm /home/zhl/.conda/envs/zhl_lerobot/lib/libffi.so.7 /home/zhl/.conda/envs/zhl_lerobot/lib/libffi.7.so
```

注意：以后若在该环境执行 `conda install`，链接可能被恢复，报错重现时重复执行一次。
另外采集节点自 2026-08 起**不再依赖 cv_bridge**（其编译扩展与 numpy 2.x 不兼容），
图像解析改为纯 numpy 实现，支持 rgb8 / bgr8 / mono8。

## 13. 无硬件自检（可选）

不接机械臂/相机时，可用测试脚本验证数据集写入全流程：

```bash
conda activate zhl_lerobot
python tests/test_dataset_dryrun.py
```

该脚本用合成帧创建 2 个 episode 的 v6 数据集并回读校验
（data/videos/meta 结构、episode 数、任务表、视频帧格式）。

## 14. 数据格式说明

采集的数据存储为 **lerobot v6 (v3.0)** 格式：

```
CubeTransfer0804/
├── data/
│   └── chunk-000/
│       └── file-000.parquet     # 观测 + 动作数据（每帧一行）
├── videos/
│   └── observation.images.cam_top/
│   │   └── chunk-000/file-000.mp4
│   └── observation.images.cam_left/
│   └── observation.images.cam_right/
├── meta/
│   ├── info.json                # 数据集元信息（total_episodes/frames, features 等）
│   ├── tasks.parquet            # 任务定义（单一 task_name）
│   ├── stats.json               # 特征统计（训练归一化用）
│   └── episodes/                # 每集元数据（帧区间、视频时间戳等）
└── replay_output/               # data_replay_node 输出
```

**状态维度**：14 DoF（左臂 7 关节 + 右臂 7 关节，含夹爪，弧度）；单臂模式为 7 DoF
**帧率**：25 FPS
**视频编码**：H.264（NVENC 硬件加速，可配）
