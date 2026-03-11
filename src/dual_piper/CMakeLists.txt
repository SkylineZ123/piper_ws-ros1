cmake_minimum_required(VERSION 3.0.2)
project(dual_piper)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  piper_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs piper_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# ── Install Python scripts ──
catkin_install_python(PROGRAMS
  scripts/state_get_node.py
  scripts/data_replay_node.py
  scripts/data_to_hdf5_node.py
  scripts/data_to_hdf5_highlevel_node.py
  scripts/data_to_lerobot3_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ── Install launch files ──
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# ── Install config files ──
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
