#!/home/user/anaconda3/envs/lerobotv3/bin/python
# -*- coding: utf-8 -*-
"""
Dual-arm Piper State Publisher Node
Connects to two Piper robot arms via CAN bus, reads joint states
(position, velocity, effort) and gripper data at 200 Hz, and publishes
them as ``sensor_msgs/JointState`` messages on left/right master/slave
topics.
Usage::
    rosrun dual_piper state_get_node.py _left_can_port:=can_left _right_can_port:=can_right
"""
import sys
import os
# Ensure the scripts directory is on sys.path so that the utils package is
# importable regardless of the working directory used by rosrun / roslaunch.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import threading

import rospy
from sensor_msgs.msg import JointState
from piper_sdk import C_PiperInterface

from utils import (
    DEG_TO_RAD,
    JOINT_SCALE,
    GRIPPER_SCALE,
    EFFORT_SCALE,
    ENABLE_ALL_JOINTS,
    PUBLISH_RATE_HZ,
)

_JOINT_NAMES = [
    'joint1', 'joint2', 'joint3', 'joint4',
    'joint5', 'joint6', 'joint7', 'joint8',
]

_NUM_JOINTS = len(_JOINT_NAMES)


class DualPiperStateNode:
    """ROS node that publishes dual-arm Piper joint states."""
    def __init__(self):
        rospy.init_node('dual_piper_state_get_node', anonymous=True)

        # ── Parameters ───────────────────────────────────────────────────
        self._can_left = rospy.get_param('~left_can_port', 'can_left')
        self._can_right = rospy.get_param('~right_can_port', 'can_right')
        self._auto_enable = rospy.get_param('~auto_enable', True)

        rospy.loginfo("Left  CAN port : %s", self._can_left)
        rospy.loginfo("Right CAN port : %s", self._can_right)
        rospy.loginfo("Auto enable    : %s", self._auto_enable)


        # ── Connect to arms ──────────────────────────────────────────────
        self._piper_left = self._connect_arm('Left', self._can_left)
        self._piper_right = self._connect_arm('Right', self._can_right)


        # ── Publishers ───────────────────────────────────────────────────
        self._left_master_pub = rospy.Publisher(
            'left/master_joint_states', JointState, queue_size=10)
        self._left_slave_pub = rospy.Publisher(
            'left/slave_joint_states', JointState, queue_size=10)
        self._right_master_pub = rospy.Publisher(
            'right/master_joint_states', JointState, queue_size=10)
        self._right_slave_pub = rospy.Publisher(
            'right/slave_joint_states', JointState, queue_size=10)

        # ── Pre-allocated messages ───────────────────────────────────────
        self._left_master_msg = self._make_joint_msg()
        self._left_slave_msg = self._make_joint_msg()
        self._right_master_msg = self._make_joint_msg()
        self._right_slave_msg = self._make_joint_msg()

        # ── Enable & start publishing ────────────────────────────────────
        self._enabled_left = False
        self._enabled_right = False

        if self._auto_enable:
            self._enable_dual_arms()

        pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        pub_thread.start()

    # ── Helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _connect_arm(label, can_port):
        """Instantiate and connect a ``C_PiperInterface``."""
        rospy.loginfo("Connecting to %s arm on %s …", label, can_port)
        iface = C_PiperInterface(can_name=can_port)
        iface.ConnectPort()
        rospy.loginfo("%s arm connected.", label)
        return iface

    @staticmethod
    def _make_joint_msg():
        """Return a pre-filled ``JointState`` message."""
        msg = JointState()
        msg.name = list(_JOINT_NAMES)
        msg.position = [0.0] * _NUM_JOINTS
        msg.velocity = [0.0] * _NUM_JOINTS
        msg.effort = [0.0] * _NUM_JOINTS
        return msg

    @staticmethod
    def _check_arm_enabled(piper):
        """Return ``True`` if all six motors report *driver_enable_status*."""
        status = piper.GetArmLowSpdInfoMsgs()
        return all(
            getattr(status, f'motor_{i}').foc_status.driver_enable_status
            for i in range(1, 7)
        )

    # ── Enable ───────────────────────────────────────────────────────────
    def _enable_dual_arms(self):
        """Block until both arms are enabled (or timeout)."""
        rospy.loginfo("Enabling dual arms …")
        timeout = 10.0
        t0 = time.time()
        while not (self._enabled_left and self._enabled_right):
            if rospy.is_shutdown():
                return
            if time.time() - t0 > timeout:
                rospy.logerr("Timeout while enabling arms — exiting.")
                sys.exit(1)
            if not self._enabled_left:
                if self._check_arm_enabled(self._piper_left):
                    self._enabled_left = True
                    rospy.loginfo("Left arm enabled.")
                else:
                    self._piper_left.EnableArm(ENABLE_ALL_JOINTS)

            if not self._enabled_right:
                if self._check_arm_enabled(self._piper_right):
                    self._enabled_right = True
                    rospy.loginfo("Right arm enabled.")
                else:
                    self._piper_right.EnableArm(ENABLE_ALL_JOINTS)

            time.sleep(1)

    # ── Publishing ───────────────────────────────────────────────────────
    def _publish_loop(self):
        """Publish joint states for both arms at ``PUBLISH_RATE_HZ``."""
        rate = rospy.Rate(PUBLISH_RATE_HZ)
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            self._publish_arm(self._piper_left,
                              self._left_master_pub, self._left_slave_pub,
                              self._left_master_msg, self._left_slave_msg,
                              stamp)
            self._publish_arm(self._piper_right,
                              self._right_master_pub, self._right_slave_pub,
                              self._right_master_msg, self._right_slave_msg,
                              stamp)
            rate.sleep()

    @staticmethod
    def _publish_arm(piper, master_pub, slave_pub, master_msg, slave_msg, stamp):
        """Read SDK data for one arm and publish master + slave messages."""
        try:
            # ── Master (actual joint feedback) ───────────────────────────
            master_msg.header.stamp = stamp
            joint = piper.GetArmJointMsgs().joint_state
            gripper = piper.GetArmGripperCtrl().gripper_ctrl
            speed = piper.GetArmHighSpdInfoMsgs()

            master_msg.position = [
                joint.joint_1 / JOINT_SCALE * DEG_TO_RAD,
                joint.joint_2 / JOINT_SCALE * DEG_TO_RAD,
                joint.joint_3 / JOINT_SCALE * DEG_TO_RAD,
                joint.joint_4 / JOINT_SCALE * DEG_TO_RAD,
                joint.joint_5 / JOINT_SCALE * DEG_TO_RAD,
                joint.joint_6 / JOINT_SCALE * DEG_TO_RAD,
                gripper.grippers_angle / GRIPPER_SCALE,
                0.0,
                ]

            master_msg.velocity = [
                speed.motor_1.motor_speed / EFFORT_SCALE,
                speed.motor_2.motor_speed / EFFORT_SCALE,
                speed.motor_3.motor_speed / EFFORT_SCALE,
                speed.motor_4.motor_speed / EFFORT_SCALE,
                speed.motor_5.motor_speed / EFFORT_SCALE,
                speed.motor_6.motor_speed / EFFORT_SCALE,
                0.0, 0.0,
            ]

            master_msg.effort = [0.0] * _NUM_JOINTS
            master_msg.effort[6] = gripper.grippers_effort / EFFORT_SCALE
            master_pub.publish(master_msg)

            # ── Slave (commanded joint target) ───────────────────────────
            slave_msg.header.stamp = stamp
            s_joint = piper.GetArmJointMsgs().joint_state
            s_gripper = piper.GetArmGripperCtrl().gripper_ctrl

            slave_msg.position = [
                s_joint.joint_1 / JOINT_SCALE * DEG_TO_RAD,
                s_joint.joint_2 / JOINT_SCALE * DEG_TO_RAD,
                s_joint.joint_3 / JOINT_SCALE * DEG_TO_RAD,
                s_joint.joint_4 / JOINT_SCALE * DEG_TO_RAD,
                s_joint.joint_5 / JOINT_SCALE * DEG_TO_RAD,
                s_joint.joint_6 / JOINT_SCALE * DEG_TO_RAD,
                s_gripper.grippers_angle / GRIPPER_SCALE,
                0.0,
                ]
            slave_msg.velocity = list(master_msg.velocity)
            slave_msg.effort = list(master_msg.effort)
            slave_pub.publish(slave_msg)
        except Exception as e:
            rospy.logerr("Error reading/publishing arm state: %s", e)


def main():
    try:
        DualPiperStateNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down dual_piper state node.")


if __name__ == '__main__':
    main()