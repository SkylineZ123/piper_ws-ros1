"""Load the same effective settings for direct Python and ROS node entry points."""
from pathlib import Path
import os
import shutil

import yaml

MOTION_KEYS = ('max_tcp_speed', 'max_tcp_angular_speed', 'max_joint_speed',
               'max_joint_acceleration', 'max_gripper_speed', 'translation_scale')


def package_file(relative):
    # rospkg also resolves the install space, where scripts and share data differ.
    import rospkg
    return Path(rospkg.RosPack().get_path('pika_piper')) / relative


def load_config(path=None, overrides=None):
    path = Path(path) if path else package_file('config/pipeline.yaml')
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Pipeline config must be a YAML mapping: ' + str(path))
    if overrides:
        config.update({key: value for key, value in overrides.items() if key != 'config'})
    return config


def control_environment(python):
    """Use the selected interpreter's C++ runtime in its child process only."""
    executable = shutil.which(python) or python
    library = str(Path(executable).absolute().parent.parent / 'lib')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = library + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env['PYTHONNOUSERSITE'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    return env
