cmake_minimum_required(VERSION 3.0.2)
project(pika_piper)

find_package(catkin REQUIRED COMPONENTS
  roscpp rospy data_msgs std_msgs std_srvs sensor_msgs geometry_msgs tf)
find_package(Boost REQUIRED COMPONENTS system)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

catkin_package(CATKIN_DEPENDS roscpp rospy data_msgs std_msgs std_srvs sensor_msgs geometry_msgs tf)
include_directories(${catkin_INCLUDE_DIRS} ${JSONCPP_INCLUDE_DIRS})
add_executable(pika_serial src/pika_serial.cpp)
set_property(TARGET pika_serial PROPERTY CXX_STANDARD 14)
add_dependencies(pika_serial ${catkin_EXPORTED_TARGETS})
target_link_libraries(pika_serial ${catkin_LIBRARIES} ${JSONCPP_LIBRARIES} Boost::system pthread)
install(TARGETS pika_serial RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS
  scripts/bringup.py scripts/control_node.py scripts/record_session.py
  scripts/replay_worker.py scripts/replay.py
  scripts/doctor.py scripts/check_bringup_env.py scripts/measure_teleop.py
  scripts/check_ik_regression.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
# Keep Python utilities private to these executable entry points. In a devel
# space catkin wrappers execute the source script; in an install space utils
# sits beside the installed scripts, with the same imports in both cases.
install(DIRECTORY scripts/utils DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY launch config docs DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY environments DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES README.md LICENSE.pika_ros
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# RealSense source packages and runtime from the existing catkin workspace.
set(PIKA_REALSENSE_WORKSPACE "$ENV{HOME}/catkin_ws" CACHE PATH
  "Catkin workspace containing the built realsense-ros packages")
catkin_add_env_hooks(50.pika-local-ros SHELLS sh DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)
