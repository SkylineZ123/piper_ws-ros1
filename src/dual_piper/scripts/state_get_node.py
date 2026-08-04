#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""state_get_node.py — 发布单臂关节状态 (7 关节, 弧度, 含夹爪).

从 piper_sdk 读取物理关节角度与夹爪行程, 以 sensor_msgs/JointState 发布到
/{side}/master_joint_states 与 /{side}/slave_joint_states 两个话题
(保持原工作区话题契约; 遥操作模式下主/从状态一致, 采集节点默认读 slave).

用法 (由 teleoperation.launch 启动两个实例):
    rosrun dual_piper state_get_node.py _can_name:=can_left _side:=left
"""

import numpy as np
import rospy
from sensor_msgs.msg import JointState

# 单位换算 (与 data_replay_node_zhl.py 一致):
#   SDK 关节角度单位 0.001 度 -> 弧度
RAD_TO_DEG_X1000 = 180.0 / np.pi * 1000.0
#   SDK 夹爪行程单位 0.001 mm -> m
GRIPPER_SCALE = 1_000_000.0

JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["gripper"]


def read_state(piper) -> np.ndarray:
    """读取单臂当前位姿 -> [j1..j6 (rad), gripper (m)]."""
    js = piper.GetArmJointMsgs().joint_state
    gs = piper.GetArmGripperMsgs().gripper_state
    return np.array([
        js.joint_1 / RAD_TO_DEG_X1000,
        js.joint_2 / RAD_TO_DEG_X1000,
        js.joint_3 / RAD_TO_DEG_X1000,
        js.joint_4 / RAD_TO_DEG_X1000,
        js.joint_5 / RAD_TO_DEG_X1000,
        js.joint_6 / RAD_TO_DEG_X1000,
        gs.grippers_angle / GRIPPER_SCALE,
    ], dtype=np.float64)


def main():
    rospy.init_node("state_get_node", anonymous=True)
    from piper_sdk import C_PiperInterface  # conda 环境内导入

    can_name = rospy.get_param("~can_name", "can_left")
    side = rospy.get_param("~side", "left")          # left | right
    rate_hz = rospy.get_param("~rate", 100)
    enable_arm = rospy.get_param("~enable_arm", False)  # 仅读状态无需使能

    rospy.loginfo(f"state_get_node: side={side} can={can_name} rate={rate_hz}Hz")

    piper = C_PiperInterface(can_name=can_name)
    piper.ConnectPort()
    if enable_arm:
        piper.EnableArm(7)
        rospy.loginfo(f"[{side}] arm enabled")

    pub_master = rospy.Publisher(f"/{side}/master_joint_states", JointState, queue_size=10)
    pub_slave = rospy.Publisher(f"/{side}/slave_joint_states", JointState, queue_size=10)

    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        try:
            q = read_state(piper)
        except Exception as e:  # CAN 断线保护
            rospy.logwarn_throttle(5.0, f"[{side}] read state failed: {e}")
            rate.sleep()
            continue

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = JOINT_NAMES
        msg.position = q.tolist()
        msg.velocity = [0.0] * 7
        msg.effort = []
        pub_master.publish(msg)
        pub_slave.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
