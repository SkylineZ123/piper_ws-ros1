"""Exercise the real CAN-driver callback with fake ROS and SDK, without devices."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def driver(monkeypatch):
    class Msg:
        def __init__(self):
            self.header = types.SimpleNamespace(stamp=None)
            self.position = [0.] * 7
            self.velocity = []
            self.effort = []
    params = {'~motion_speed_percent': 50}
    fake = types.ModuleType('rospy')
    fake.get_param_cached = lambda name, default: params.get(name, default)
    fake.set_param = params.__setitem__
    fake.loginfo = fake.logerr_throttle = lambda *a: None
    monkeypatch.setitem(sys.modules, 'rospy', fake)
    monkeypatch.setitem(sys.modules, 'rosnode', types.ModuleType('rosnode'))
    messages = {
        'sensor_msgs.msg': ['JointState'], 'std_msgs.msg': ['Bool'],
        'std_srvs.srv': ['SetBool', 'SetBoolResponse', 'Trigger', 'TriggerResponse'],
        'piper_msgs.msg': ['PiperStatusMsg', 'PosCmd', 'PiperEulerPose'],
        'piper_msgs.srv': ['Enable', 'EnableResponse', 'Gripper', 'GripperResponse', 'GoZero', 'GoZeroResponse'],
        'geometry_msgs.msg': ['Pose', 'PoseStamped'],
        'tf.transformations': ['quaternion_from_euler'], 'piper_sdk': ['C_PiperInterface'],
    }
    for module, names in messages.items():
        if '.' in module:
            monkeypatch.setitem(sys.modules, module.split('.')[0], types.ModuleType(module.split('.')[0]))
        stub = types.ModuleType(module)
        for name in names:
            setattr(stub, name, Msg)
        monkeypatch.setitem(sys.modules, module, stub)
    path = Path(__file__).resolve().parents[2] / 'piper/scripts/piper_ctrl_single_node.py'
    spec = importlib.util.spec_from_file_location('driver_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.C_PiperRosNode.__new__(module.C_PiperRosNode)
    calls = []
    node.piper = types.SimpleNamespace(
        MotionCtrl_2=lambda *a: calls.append(('speed', a)),
        JointCtrl=lambda *a: calls.append(('joint', a)))
    node.motion_speed_percent = 50
    node._applied_motion_speed_percent = None
    node.block_ctrl_flag = False
    node.gripper_exist = False
    node.gripper_val_mutiple = 1
    node.GetEnableFlag = lambda: True
    node.command_sent_pub = types.SimpleNamespace(publish=lambda msg: calls.append(('echo', msg)))
    return module, node, params, calls, Msg


def test_default_and_runtime_percentage_reach_sdk(driver):
    _, node, params, calls, Msg = driver
    node.joint_callback(Msg())
    assert calls[0] == ('speed', (1, 1, 50, 0))
    assert params['~applied_motion_speed_percent'] == 50
    params['~motion_speed_percent'] = 65
    calls.clear()
    node.joint_callback(Msg())
    assert calls[0] == ('speed', (1, 1, 65, 0))
    assert [kind for kind, _ in calls] == ['speed', 'joint', 'echo']
    assert params['~applied_motion_speed_percent'] == 65


@pytest.mark.parametrize('value', [0, -1, 101, 50.5, '60', True, float('nan')])
def test_invalid_runtime_percentage_sends_no_can_or_echo(driver, value):
    _, node, params, calls, Msg = driver
    params['~motion_speed_percent'] = value
    node.joint_callback(Msg())
    assert calls == []


def test_legacy_explicit_velocity_percentage_preserved(driver):
    _, node, params, calls, Msg = driver
    msg = Msg()
    msg.velocity = [0.] * 6 + [75.]
    node.joint_callback(msg)
    assert calls[0] == ('speed', (1, 1, 75))
    assert params['~applied_motion_speed_percent'] == 75


def test_mit_follow_preserves_flag_and_gripper_echo(driver):
    _,node,params,calls,Msg=driver
    node.output_mode='mit_follow'
    msg=Msg();msg.velocity=[0.]*6+[75.]
    node.joint_callback(msg)
    assert calls[0]==('speed',(1,1,75,0xAD))
    assert [kind for kind,_ in calls]==['speed','joint','echo']
    assert params['~applied_output_mode']=='mit_follow'
    calls.clear();node.joint_callback(Msg())
    assert calls[0]==('speed',(1,1,50,0xAD))


def firmware_mock(node,calls,fail_joint=None):
    node.output_mode='mit_follow';node.sdk_protocol='v3'
    node.configure_firmware_speed=True;node.firmware_speed=1.5
    state={i:dict(min_angle_limit=-700,max_angle_limit=700,max_joint_spd=300,motor_num=i) for i in range(1,7)}
    wrapper=types.SimpleNamespace(time_stamp=0.,current_motor_angle_limit_max_vel=types.SimpleNamespace(motor_num=0))
    def query(joint,kind):
        assert kind==1
        wrapper.time_stamp+=1
        wrapper.current_motor_angle_limit_max_vel=types.SimpleNamespace(**state[joint])
    def set_speed(joint,speed):
        calls.append(('set_speed',(joint,speed)))
        if not (joint==fail_joint and speed==1500):state[joint]['max_joint_spd']=speed
    node.piper.GetPiperFirmwareVersion=lambda:'S-V1.9-1'
    node.piper.GetCurrentMotorAngleLimitMaxVel=lambda:wrapper
    node.piper.SearchMotorMaxAngleSpdAccLimit=query
    node.piper.MotorMaxSpdSet=set_speed
    return state


def test_firmware_speeds_are_read_back_before_ready_and_no_motion(driver):
    _,node,params,calls,_=driver;state=firmware_mock(node,calls)
    node._configure_follow_output()
    report=params['~applied_configuration']
    assert report['ready'] and report['joint_speed_limits_rad_s']==[1.5]*6
    assert report['sdk_protocol']=='v3' and not report['speed_percent_effective']
    assert [kind for kind,_ in calls]==['set_speed']*6
    assert all(row['min_angle_limit']==-700 and row['max_angle_limit']==700 for row in state.values())


def test_readback_failure_restores_previous_speeds_and_never_marks_ready(driver):
    _,node,params,calls,_=driver;state=firmware_mock(node,calls,fail_joint=3)
    with pytest.raises(RuntimeError,match='readback'):
        node._configure_follow_output()
    assert not params['~applied_configuration']['ready']
    assert all(params['~applied_configuration']['rollback_verified'])
    assert all(row['max_joint_spd']==300 for row in state.values())
    assert not any(kind in ('speed','joint','echo') for kind,_ in calls)


def test_protocol_mismatch_cannot_set_limits_or_send_motion(driver):
    _,node,params,calls,_=driver;firmware_mock(node,calls)
    node.sdk_protocol='v2'
    with pytest.raises(RuntimeError,match='requires SDK v3'):
        node._configure_follow_output()
    assert not calls and not params['~applied_configuration']['ready']


def test_legacy_position_start_does_not_write_firmware(driver):
    _,node,params,calls,_=driver
    node.output_mode='position';node.sdk_protocol='v2';node.configure_firmware_speed=False
    node._configure_follow_output()
    assert not calls
    assert params['~applied_configuration']['firmware']=='not_queried'
