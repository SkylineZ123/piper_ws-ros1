"""Expand ROS launch files offline; never contact a master or start a node."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from utils.configuration import load_config, control_environment

PACKAGE = Path(__file__).resolve().parents[1]


def expand(tmp_path, package, launch, *args):
    env = dict(os.environ, ROS_PACKAGE_PATH=str(PACKAGE.parent) + ':/opt/ros/noetic/share',
               ROS_HOME=str(tmp_path), PYTHONPATH='/opt/ros/noetic/lib/python3/dist-packages')
    env.pop('LD_LIBRARY_PATH', None)
    code = '''
import json,sys,rospkg
from roslaunch.config import ROSLaunchConfig
from roslaunch.xmlloader import XmlLoader
from pathlib import Path
c=ROSLaunchConfig()
XmlLoader().load(str(Path(rospkg.RosPack().get_path(sys.argv[1]))/'launch'/sys.argv[2]),c,argv=sys.argv[3:],verbose=False)
print(json.dumps({'params':{k:v.value for k,v in c.params.items()},'nodes':[(n.package,n.type,n.name,n.namespace,n.launch_prefix) for n in c.nodes], 'env':{n.name:dict(n.env_args) for n in c.nodes}}))
'''
    result = subprocess.run(['/usr/bin/python3', '-c', code, package, launch, *args],
                            env=env, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def test_default_launch_keeps_yaml_quality_limits_and_cameras_off(tmp_path):
    report = expand(tmp_path, 'pika_piper', 'teleop.launch', 'start_cameras:=false')
    cfg = load_config(PACKAGE / 'config/pipeline.yaml')
    assert len(report['nodes']) == 4
    assert ('pika_piper', 'pika_serial', 'pika_serial', '/', None) in [tuple(n) for n in report['nodes']]
    for key in ('dist_limit', 'angle_limit', 'linear_limit', 'angular_limit'):
        assert report['params']['/pika_single_locator/' + key] == cfg[key]
    assert report['params']['/pika_piper/driver/motion_speed_percent'] == cfg['motion_speed_percent']
    assert report['params']['/pika_piper/driver/output_mode'] == cfg['output_mode']
    assert report['params']['/pika_piper/driver/sdk_protocol'] == cfg['sdk_protocol']
    assert 'PYTHONPATH' not in report['env']['driver']


@pytest.mark.parametrize('package,launch', [('pika_piper', 'teleop.launch'), ('dual_piper', 'pika_teleop.launch')])
def test_overrides_reach_correct_nodes(tmp_path, package, launch):
    report = expand(tmp_path, package, launch, 'control_hz:=60', 'max_tcp_speed:=0.2',
                    'motion_speed_percent:=65', 'max_joint_speed:=1.2', 'control_python:=/solver/python',
                    'start_cameras:=false')
    params = report['params']
    assert params['/pika_piper/controller/control_hz'] == 60.
    assert params['/pika_piper/controller/max_tcp_speed'] == .2
    assert params['/pika_piper/controller/max_joint_speed'] == 1.2
    assert params['/pika_piper/driver/motion_speed_percent'] == 65
    assert all(n[4] == '/solver/python' for n in report['nodes'] if n[3] == '/pika_piper/')
    assert report['env']['controller']['LD_LIBRARY_PATH'].startswith('/solver/../lib:')
    assert report['env']['driver']['LD_LIBRARY_PATH'].startswith('/solver/../lib:')
    assert 'LD_LIBRARY_PATH' not in report['env']['pika_single_locator']


def test_config_private_parameters_override_yaml(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('control_hz: 50\nmax_tcp_speed: 0.15\n')
    assert load_config(path, {'control_hz': 60, 'config': 'ignored'}) == {'control_hz': 60, 'max_tcp_speed': .15}


def test_control_cpp_runtime_is_scoped_to_child_environment(monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH','/ros/lib')
    env=control_environment('/environments/control/bin/python')
    assert env['LD_LIBRARY_PATH']=='/environments/control/lib:/ros/lib'
    assert os.environ['LD_LIBRARY_PATH']=='/ros/lib'
