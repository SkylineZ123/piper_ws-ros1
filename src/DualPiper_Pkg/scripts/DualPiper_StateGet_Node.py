#coding=utf-8
import os
import argparse
import time
import copy
import math
import threading
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
from piper_msgs.msg import PiperStatusMsg, PosCmd
from piper_sdk import * # 确保这个库与ROS1兼容
from numpy import clip
import sys

class DualPiperRosNode():

    def __init__(self) -> None:
        rospy.init_node('dual_piper_state_get_node', anonymous=True)
        
        # --- 1. 获取左右臂的CAN端口参数 ---
        # 使用rospy.get_param获取参数，如果参数未设置，则使用默认值
        self.can_left_port = rospy.get_param('~left_can_port', 'can_left')
        self.can_right_port = rospy.get_param('~right_can_port', 'can_right')
        self.auto_enable = rospy.get_param('~auto_enable', True)

        rospy.loginfo(f"Left Arm CAN Port: {self.can_left_port}")
        rospy.loginfo(f"Right Arm CAN Port: {self.can_right_port}")
        rospy.loginfo(f"Auto Enable: {self.auto_enable}")

        # --- 2. 为左右臂分别实例化和连接Piper接口 ---
        rospy.loginfo("Connecting to Left Arm...")
        self.piper_left = C_PiperInterface(can_name=self.can_left_port)
        self.piper_left.ConnectPort()
        rospy.loginfo("Left Arm Connected.")
        
        rospy.loginfo("Connecting to Right Arm...")
        self.piper_right = C_PiperInterface(can_name=self.can_right_port)
        self.piper_right.ConnectPort()
        rospy.loginfo("Right Arm Connected.")
        
        # --- 3. 为左右臂分别创建话题发布者 ---
        # 左臂话题
        self.left_master_joint_pub = rospy.Publisher('left/master_joint_states', JointState, queue_size=10)
        self.left_slave_joint_pub = rospy.Publisher('left/slave_joint_states', JointState, queue_size=10)
        
        # 右臂话题
        self.right_master_joint_pub = rospy.Publisher('right/master_joint_states', JointState, queue_size=10)
        self.right_slave_joint_pub = rospy.Publisher('right/slave_joint_states', JointState, queue_size=10)

        # --- 4. 为左右臂分别创建JointState消息对象 ---
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
        
        # 左臂消息
        self.left_master_joint_states = JointState(name=joint_names, position=[0.0]*8, velocity=[0.0]*8, effort=[0.0]*8)
        self.left_slave_joint_states = JointState(name=joint_names, position=[0.0]*8, velocity=[0.0]*8, effort=[0.0]*8)
        
        # 右臂消息
        self.right_master_joint_states = JointState(name=joint_names, position=[0.0]*8, velocity=[0.0]*8, effort=[0.0]*8)
        self.right_slave_joint_states = JointState(name=joint_names, position=[0.0]*8, velocity=[0.0]*8, effort=[0.0]*8)


        # --- 5. 使能和线程启动 ---
        self.__enable_flag_left = False
        self.__enable_flag_right = False
        
        if self.auto_enable:
            self.enable_dual_arms()

        self.publisher_joint_thread = threading.Thread(target=self.publisher_joint_loop)
        self.publisher_joint_thread.daemon = True # 设置为守护线程
        self.publisher_joint_thread.start()


    def enable_dual_arms(self):
        """
        尝试使能左右两个机械臂，直到成功或超时。
        """
        rospy.loginfo("Attempting to enable dual arms...")
        timeout = 10
        start_time = time.time()
        while not (self.__enable_flag_left and self.__enable_flag_right) and not rospy.is_shutdown():
            if time.time() - start_time > timeout:
                rospy.logerr("Timeout reached while enabling arms. Exiting.")
                sys.exit(1)
            
            # --- 检查和使能左臂 ---
            if not self.__enable_flag_left:
                left_status = self.piper_left.GetArmLowSpdInfoMsgs()
                # 检查所有电机是否都已使能
                is_left_enabled = all([
                    left_status.motor_1.foc_status.driver_enable_status,
                    left_status.motor_2.foc_status.driver_enable_status,
                    left_status.motor_3.foc_status.driver_enable_status,
                    left_status.motor_4.foc_status.driver_enable_status,
                    left_status.motor_5.foc_status.driver_enable_status,
                    left_status.motor_6.foc_status.driver_enable_status
                ])
                if is_left_enabled:
                    self.__enable_flag_left = True
                    rospy.loginfo("Left Arm Enabled Successfully.")
                else:
                    self.piper_left.EnableArm(7) # 7代表所有关节
                    rospy.loginfo("Sent enable command to Left Arm.")

            # --- 检查和使能右臂 ---
            if not self.__enable_flag_right:
                right_status = self.piper_right.GetArmLowSpdInfoMsgs()
                is_right_enabled = all([
                    right_status.motor_1.foc_status.driver_enable_status,
                    right_status.motor_2.foc_status.driver_enable_status,
                    right_status.motor_3.foc_status.driver_enable_status,
                    right_status.motor_4.foc_status.driver_enable_status,
                    right_status.motor_5.foc_status.driver_enable_status,
                    right_status.motor_6.foc_status.driver_enable_status
                ])
                if is_right_enabled:
                    self.__enable_flag_right = True
                    rospy.loginfo("Right Arm Enabled Successfully.")
                else:
                    self.piper_right.EnableArm(7)
                    rospy.loginfo("Sent enable command to Right Arm.")
            
            time.sleep(1)
        
    def publisher_joint_loop(self):
        """
        以200Hz的频率循环发布左右两臂的关节状态。
        """
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            
            # 发布左臂关节信息
            self._publish_single_arm_joints(
                self.piper_left, 
                self.left_master_joint_pub, 
                self.left_slave_joint_pub,
                self.left_master_joint_states,
                self.left_slave_joint_states,
                current_time
            )
            
            # 发布右臂关节信息
            self._publish_single_arm_joints(
                self.piper_right, 
                self.right_master_joint_pub,
                self.right_slave_joint_pub,
                self.right_master_joint_states,
                self.right_slave_joint_states,
                current_time
            )
            
            rate.sleep()
            
    def _publish_single_arm_joints(self, piper_iface:C_PiperInterface, master_pub, slave_pub, master_msg, slave_msg, stamp):
        """
        辅助函数，用于获取并发布单个机械臂的主从关节状态。
        """
        try:
            # --- 发布主臂关节状态 ---
            master_msg.header.stamp = stamp

            master_joint_data = piper_iface.GetArmJointMsgs()
            # master_gripper_data = piper_iface.GetArmGripperMsgs()
            master_gripper_data = piper_iface.GetArmGripperCtrl()
            master_speed_data = piper_iface.GetArmHighSpdInfoMsgs()
            deg_to_rad = 0.01745329
            master_msg.position = [
                (master_joint_data.joint_state.joint_1 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_2 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_3 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_4 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_5 / 1000) * deg_to_rad,
                (master_joint_data.joint_state.joint_6 / 1000) * deg_to_rad,
                master_gripper_data.gripper_ctrl.grippers_angle / 1000000,
                0.0
            ]
            master_msg.velocity = [
                master_speed_data.motor_1.motor_speed / 1000,
                master_speed_data.motor_2.motor_speed / 1000,
                master_speed_data.motor_3.motor_speed / 1000,
                master_speed_data.motor_4.motor_speed / 1000,
                master_speed_data.motor_5.motor_speed / 1000,
                master_speed_data.motor_6.motor_speed / 1000,
                0.0, 0.0
            ]
            master_msg.effort = [0.0] * 8
            master_msg.effort[6] = master_gripper_data.gripper_ctrl.grippers_effort / 1000
            master_pub.publish(master_msg)
            
            # --- 发布从臂关节状态 ---
            slave_msg.header.stamp = stamp
            slave_joint_data = piper_iface.GetArmJointCtrl()
            slave_gripper_data = piper_iface.GetArmGripperCtrl()
            
            slave_msg.position = [
                (slave_joint_data.joint_ctrl.joint_1 / 1000) * deg_to_rad,
                (slave_joint_data.joint_ctrl.joint_2 / 1000) * deg_to_rad,
                (slave_joint_data.joint_ctrl.joint_3 / 1000) * deg_to_rad,
                (slave_joint_data.joint_ctrl.joint_4 / 1000) * deg_to_rad,
                (slave_joint_data.joint_ctrl.joint_5 / 1000) * deg_to_rad,
                (slave_joint_data.joint_ctrl.joint_6 / 1000) * deg_to_rad,
                master_gripper_data.gripper_ctrl.grippers_angle / 1000000,
                0.0
            ]
            # 从臂的速度和力矩信息通常与主臂相同或不直接提供，这里暂时复用主臂速度
            slave_msg.velocity = master_msg.velocity
            slave_msg.effort = master_msg.effort
            slave_pub.publish(slave_msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish joint states: {e}")

    def Sub_Left_JointThread(self):
        """
            机械臂关节订阅
        """
        rospy.Subscriber('slave_arm_topic', JointState, self.joint_callback, queue_size=1, tcp_nodelay=True)
        # rospy.Subscriber('/move_group/fake_controller_joint_states', JointState, self.joint_callback)
        rospy.spin()
def main():
    try:
        node = DualPiperRosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down dual piper node.")

if __name__ == '__main__':
    main()
