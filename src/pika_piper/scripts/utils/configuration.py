"""Load the same effective settings for direct Python and ROS node entry points."""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

MOTION_KEYS = (
    'max_tcp_speed',
    'max_tcp_angular_speed',
    'max_joint_speed',
    'max_joint_acceleration',
    'max_gripper_speed',
    'translation_scale',
)


def package_file(relative, package='pika_piper'):
    # rospkg also resolves the install space, where scripts and share data differ.
    import rospkg

    try:
        return Path(rospkg.RosPack().get_path(package)) / relative
    except rospkg.ResourceNotFound as error:
        # Direct source-file execution (e.g. from an IDE) can find this Python
        # module before the workspace has been added to ROS_PACKAGE_PATH.
        own_root = Path(__file__).resolve().parents[2]
        root = own_root if package == 'pika_piper' else own_root.parent / package
        local = root / relative
        manifest = root / 'package.xml'
        if manifest.is_file() and local.is_file():
            if ET.parse(str(manifest)).getroot().findtext('name') == package:
                return local
        # A catkin source space may contain nested repositories, e.g.
        # src/piper_moveit/piper_with_gripper_moveit. Let rospkg index the
        # current source space instead of assuming every package is a sibling.
        source_space = own_root.parent
        try:
            indexed_root = Path(rospkg.RosPack(ros_paths=[str(source_space)]).get_path(package))
            indexed_file = indexed_root / relative
            if indexed_file.is_file():
                return indexed_file
        except rospkg.ResourceNotFound:
            pass
        raise rospkg.ResourceNotFound(
            "%s (also checked %s and indexed source space %s; source this workspace's devel/setup.bash)"
            % (package, local, source_space),
            ros_paths=error.ros_paths,
        ) from error


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
    env['LD_LIBRARY_PATH'] = library + (
        ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''
    )
    env['PYTHONNOUSERSITE'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    return env
