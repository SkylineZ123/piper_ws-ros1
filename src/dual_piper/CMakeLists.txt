cmake_minimum_required(VERSION 3.0.2)
project(dual_piper)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs geometry_msgs piper_msgs)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/data_to_lerobot3_node.py
    scripts/data_to_lerobot3_node_repeat.py
    scripts/data_replay_node.py
    scripts/state_get_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
