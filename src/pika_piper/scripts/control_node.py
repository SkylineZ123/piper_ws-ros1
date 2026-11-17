#!/usr/bin/env python3
"""Run the Pika + Piper ROS controller."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rospy
from utils.controller import Controller


def main():
    rospy.init_node('pika_controller')
    Controller()
    rospy.spin()


if __name__ == '__main__':
    main()
