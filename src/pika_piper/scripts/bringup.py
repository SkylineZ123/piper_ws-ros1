#!/usr/bin/env python3
"""Translate the single YAML configuration into ROS launch arguments."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from pika_locator_runtime import supervise
from pika_piper.configuration import load_config, control_environment
from pika_piper.core import validate_control_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Defaults to pika_piper/config/pipeline.yaml')
    parser.add_argument('--control-python', default=os.environ.get('PIKA_CONTROL_PYTHON', sys.executable))
    cameras = parser.add_mutually_exclusive_group()
    cameras.add_argument('--start-cameras', action='store_true')
    cameras.add_argument('--no-cameras', action='store_true', help='Cameras are disabled by default')
    for name in ('control-hz', 'max-tcp-speed', 'max-tcp-angular-speed',
                 'max-joint-speed', 'max-gripper-speed', 'translation-scale'):
        parser.add_argument('--' + name, type=float)
    parser.add_argument('--motion-speed-percent', type=int)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print launch command without starting hardware')
    args = parser.parse_args()
    from pika_piper.configuration import package_file
    config_path = Path(args.config) if args.config else package_file('config/pipeline.yaml')
    names = ('control_hz', 'max_tcp_speed', 'max_tcp_angular_speed',
             'max_joint_speed', 'max_gripper_speed', 'translation_scale', 'motion_speed_percent')
    overrides = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    try:
        config = load_config(config_path, overrides)
        validate_control_config(config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    serials = config.get('camera_serials', {})
    command = [
        'roslaunch', 'pika_piper', 'teleop.launch',
        'config:=' + str(config_path.resolve()),
        'control_python:=' + args.control_python,
        'start_cameras:=' + str(args.start_cameras).lower(),
        'wrist_namespace:=' + ('cam_leftwrist' if config['arm'] == 'left' else 'cam_rightwrist')]
    command += [name + ':=' + str(value) for name, value in overrides.items()]
    command += [name + '_serial:=' + str(serials[key])
                for name, key in (('top', 'top'), ('wrist', 'wrist')) if key in serials]
    if args.dry_run:
        print(shlex.join(command))
        return 0
    # Refuse missing solver dependencies before roslaunch can enable the driver.
    try:
        subprocess.run([args.control_python, '-c',
                        'from pika_piper.kinematics import Kinematics'],
                       env=control_environment(args.control_python), check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error('Control Python cannot load official IK dependencies. '
                     'Activate pika-bringup and source devel/setup.bash: ' + str(error))
    return supervise(command, interrupt_timeout=20., terminate_timeout=3.)


if __name__ == '__main__':
    sys.exit(main())
