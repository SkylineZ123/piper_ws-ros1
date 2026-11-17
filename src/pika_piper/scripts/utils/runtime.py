"""Read-only control-environment preflight shared by bringup and replay."""

import json
import shutil
import subprocess
from pathlib import Path

from .configuration import control_environment


class ReplaySetupError(RuntimeError):
    """Actionable configuration/dependency failure before robot control."""


def check_control_environment(python):
    """Import the worker's dependencies without starting a ROS node or moving."""
    executable = shutil.which(str(python))
    if executable is None:
        raise ReplaySetupError(
            'Control Python is not executable: %s; set control_python in replay.yaml' % python
        )
    source = str(Path(__file__).resolve().parents[1])
    probe = (
        'import sys,json; sys.path.insert(0,' + repr(source) + '); '
        'import rospy,casadi,pinocchio; '
        'from pinocchio import casadi as cpin; '
        'from utils.kinematics import Kinematics; '
        'from utils.configuration import package_file; '
        'urdf=package_file("urdf/piper_description.urdf",package="piper_description"); '
        'srdf=package_file("config/piper_description.srdf",package="piper_with_gripper_moveit"); '
        'print(json.dumps(dict(python=sys.executable,casadi=casadi.__version__,'
        'pinocchio=pinocchio.__version__,urdf=str(urdf),srdf=str(srdf))))'
    )
    try:
        result = subprocess.run(
            [executable, '-c', probe],
            env=control_environment(executable),
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReplaySetupError('Cannot check control_python=%s: %s' % (executable, exc)) from exc
    if result.returncode:
        lines = (result.stderr or result.stdout).strip().splitlines()
        cause = next(
            (
                line
                for line in reversed(lines)
                if any(name in line for name in ('Error:', 'Exception:', 'ResourceNotFound:'))
            ),
            lines[-1] if lines else 'dependency import failed',
        )
        raise ReplaySetupError(
            'Control environment is incompatible: %s\n%s\n'
            'The physical worker requires casadi AND pinocchio.casadi. '
            'Its URDF and SRDF must also be available in the ROS/source workspace. '
            'Set control_python in replay.yaml to the validated controller environment '
            '(or use --control-python). No robot command was sent.' % (executable, cause)
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ReplaySetupError(
            'Unexpected control environment probe output: ' + result.stdout
        ) from exc
