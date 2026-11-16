"""Exercise mode ownership and watchdogs with fake ROS, without a ROS master/CAN."""
import importlib.util,sys,types,threading,time
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def controller(monkeypatch):
    class Msg:
        def __init__(self,*args):
            self.success=args[0] if args else False;self.message=args[1] if len(args)>1 else '';self.data=args[0] if args else ''
            self.header=types.SimpleNamespace(stamp=None,frame_id='')
            self.pose=types.SimpleNamespace(position=types.SimpleNamespace(x=0.,y=0.,z=0.),orientation=types.SimpleNamespace(x=0.,y=0.,z=0.,w=1.))
    class Stamp:
        def __init__(self,value=10.):self.value=value
        def __sub__(self,other):return Stamp(self.value-other.value)
        def to_sec(self):return self.value
    fake=types.ModuleType('rospy');fake.Time=types.SimpleNamespace(now=lambda:Stamp());fake.logerr=fake.logwarn=lambda *args:None
    fake.get_param=lambda name,default=None: ({'ready':True,'output_mode':c.cfg.get('output_mode','position'),
        'sdk_protocol':c.cfg.get('sdk_protocol','v2'),
        'joint_speed_limits_rad_s':[c.cfg.get('firmware_joint_speed_limit',.3)]*6}
        if name.endswith('/driver/applied_configuration') else default)
    monkeypatch.setitem(sys.modules,'rospy',fake)
    monkeypatch.setitem(sys.modules,'rospkg',types.ModuleType('rospkg'))
    for module,names in {'sensor_msgs.msg':['JointState'],'geometry_msgs.msg':['PoseStamped'],'std_msgs.msg':['String','Int32'],'std_srvs.srv':['SetBool','SetBoolRequest','SetBoolResponse','Trigger','TriggerResponse'],'data_msgs.msg':['Gripper','LocalizationStatus','TeleopStatus']}.items():
        parent=module.split('.')[0];monkeypatch.setitem(sys.modules,parent,types.ModuleType(parent));m=types.ModuleType(module)
        for name in names:setattr(m,name,Msg)
        monkeypatch.setitem(sys.modules,module,m)
    monkeypatch.setitem(sys.modules,'pika_piper.kinematics',types.SimpleNamespace(Kinematics=object))
    spec=importlib.util.spec_from_file_location('test_control',Path(__file__).parents[1]/'scripts/control_node.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    c=module.Controller.__new__(module.Controller);c.lock=threading.RLock();c.mode='idle';c.reason='';c.last_fault=None;c.cfg={'source_timeout':.2,'control_hz':50};c.hand=np.eye(4);c.feedback=np.zeros(7);c.sent=None;c.anchor=None;c.robot_anchor=None;c.accurate=True
    c.k=types.SimpleNamespace(fk=lambda q:np.eye(4),ik=lambda target,seed,width:np.r_[seed[:6],width],urdf_hash='test',validate_configuration=lambda *args:None,info={},last_ik=None);now=time.monotonic()
    c.k.validate_feedback=lambda *args:c.k.validate_configuration(*args)
    c.k.validate_teleop_start=lambda *args:c.k.validate_feedback(*args)
    c.k.start_target=lambda q:np.array(q,copy=True)
    class FakeTeleop:
        last=None
        def warmup(self,previous):pass
        def step(self,target,previous,width,dt,velocity,config):
            from pika_piper.core import limit_pose_step
            goal,limited=limit_pose_step(c.k.fk(previous),target,dt,config.get('max_tcp_speed',.15),config.get('max_tcp_angular_speed',1.))
            self.last={'tcp_limited':limited,'step_fraction':1.,'target_matrix':goal.tolist(),'state':'tracking'}
            return c.k.ik(goal,previous,width)
    c.teleop_solver=FakeTeleop();c.command_velocity=np.zeros(6);c.tracking=None
    c.startup_target=None;c.startup_elapsed=0.
    c.feasible_request=types.SimpleNamespace(publish=lambda msg:None)
    c.hand_at=c.grip_at=c.feedback_at=c.quality_at=c.replay_at=now;c.last_tick=now
    c.command_started=0.;c.echo_at=0.
    c.gesture_at=0.;c.gesture_command=None;c.last_echo=None;c.axes=np.eye(4);c.hand_tcp=np.eye(4)
    c.state=types.SimpleNamespace(publish=lambda msg:None)
    c.teleop_status=types.SimpleNamespace(publish=lambda msg:None)
    c.sense_relative=types.SimpleNamespace(publish=lambda msg:None)
    return c

def test_reanchor_and_exclusive_replay(controller):
    c=controller;on=types.SimpleNamespace(data=True);off=types.SimpleNamespace(data=False)
    assert c.set_teleop(on).success
    assert c.anchor is not c.hand
    assert not c.set_replay(on).success
    assert c.set_teleop(off).success
    assert c.set_replay(on).success
    assert not c.set_teleop(on).success
    assert c.set_replay(off).success


def test_motion_reload_is_atomic_and_requires_pause(controller,monkeypatch):
    from pika_piper.configuration import load_config
    c=controller;c.cfg=load_config(Path(__file__).parents[1]/'config/pipeline.yaml')
    c.cfg['max_tcp_speed']=.15
    params={'max_tcp_speed':.2,'max_joint_speed':1.2}
    monkeypatch.setattr(sys.modules['rospy'],'get_param',lambda *a:params,raising=False)
    c.mode='teleop'
    assert not c.reload_motion_params(None).success
    assert c.cfg['max_tcp_speed']==.15
    c.mode='idle'
    assert c.reload_motion_params(None).success
    assert c.cfg['max_tcp_speed']==.2
    assert c.cfg['max_joint_speed']==1.2
    params.update(max_tcp_speed=.3,max_joint_speed=-1)
    assert not c.reload_motion_params(None).success
    assert c.cfg['max_tcp_speed']==.2
    assert c.cfg['max_joint_speed']==1.2
    params.clear();params['control_hz']=100
    assert not c.reload_motion_params(None).success
    assert c.cfg['control_hz']==50


def test_effective_limits_and_joint_clipping_are_visible(controller):
    import json
    c=controller;c.cfg.update(max_joint_speed=.5,max_gripper_speed=.04)
    c.sent=np.zeros(7);c.cmd=types.SimpleNamespace(publish=lambda msg:None)
    c.send(np.ones(7)*.01,.02)
    assert c.joint_limited is True
    c.send(np.ones(7)*.01,.02)
    assert c.joint_limited is True  # Gripper is independently limited.
    states=[];c.state.publish=lambda msg:states.append(json.loads(msg.data))
    c.tick(None)
    assert states[-1]['motion_limits']['max_joint_speed']==.5
    assert states[-1]['motion_limits']['control_hz']==50

def test_tracking_timeout_latches_fault(controller):
    c=controller;c.mode='teleop';c.hand_at=0
    c.tick(None)
    assert c.mode=='fault'
    assert not c.set_teleop(types.SimpleNamespace(data=True)).success

def test_stale_feedback_prevents_acquisition(controller):
    c=controller;c.feedback_at=0
    assert not c.set_replay(types.SimpleNamespace(data=True)).success
    assert not c.set_teleop(types.SimpleNamespace(data=True)).success


@pytest.mark.parametrize('report',[None,{'ready':False},
    {'ready':True,'output_mode':'position','sdk_protocol':'v3','joint_speed_limits_rad_s':[1.5]*6},
    {'ready':True,'output_mode':'mit_follow','sdk_protocol':'v3','joint_speed_limits_rad_s':[.3]*6}])
def test_fast_follow_cannot_acquire_before_driver_readback(controller,monkeypatch,report):
    c=controller;c.cfg.update(output_mode='mit_follow',sdk_protocol='v3',
        configure_firmware_joint_speed=True,firmware_joint_speed_limit=1.5)
    monkeypatch.setattr(sys.modules['rospy'],'get_param',lambda *args:report)
    response=c.set_teleop(types.SimpleNamespace(data=True))
    assert not response.success and c.mode=='idle' and c.sent is None


def test_two_replays_cannot_own_control(controller):
    request=types.SimpleNamespace(data=True)
    assert controller.set_replay(request).success
    assert not controller.set_replay(request).success


def test_driver_echo_timeout(controller):
    c=controller;c.mode='replay';c.command_started=time.monotonic()-1;c.echo_at=0.
    c.tick(None)
    assert c.mode=='fault' and 'echo' in c.reason


def test_fault_remains_visible_after_pause(controller):
    c=controller
    published=[];c.state.publish=lambda msg:published.append(__import__('json').loads(msg.data))
    c.fault('IK position residual 4.617 mm exceeds 2.000 mm')
    assert c.set_teleop(types.SimpleNamespace(data=False)).success
    c.tick(None)
    assert published[-1]['mode']=='idle'
    assert published[-1]['reason']=='operator pause'
    assert published[-1]['last_fault']['reason']=='IK position residual 4.617 mm exceeds 2.000 mm'


@pytest.mark.parametrize('claim', ['set_teleop','set_replay'])
def test_invalid_feedback_is_rejected_before_acquiring_control(controller,claim):
    c=controller
    def reject(q,context):
        raise ValueError(context+': joint2=-2.017 deg outside [0.000, 179.909] deg')
    c.k.validate_configuration=reject
    result=getattr(c,claim)(types.SimpleNamespace(data=True))
    assert not result.success
    assert 'joint2=-2.017' in result.message
    assert c.mode=='fault'
    assert c.sent is None and c.anchor is None


def test_invalid_previous_command_is_not_clipped_into_a_jump(controller):
    c=controller;c.sent=np.r_[np.deg2rad([-.352,-2.017,1.825,0,19.273,3.437]),.03]
    writes=[];c.cmd=types.SimpleNamespace(publish=writes.append)
    def reject(q,context):raise ValueError(context+': joint2 outside bounds')
    c.k.validate_configuration=reject
    with pytest.raises(ValueError,match='previous command'):
        c.send(np.r_[np.zeros(6),.03],.02)
    assert not writes


def test_stationary_ik_failure_is_reported_before_teleop_starts(controller):
    c=controller
    def reject(*args):raise ValueError('IK orientation residual 3.585 deg exceeds 2.000 deg')
    c.k.ik=reject
    result=c.set_teleop(types.SimpleNamespace(data=True))
    assert not result.success and 'IK orientation residual' in result.message
    assert c.mode=='fault' and c.sent is None and c.anchor is None


def test_real_ik_small_motion_through_controller_tick(controller,monkeypatch):
    # Run the actual controller, model, solver and limiter with fake ROS IO.
    # The simulated feedback assumes perfect tracking; no device is opened.
    monkeypatch.delitem(sys.modules,'pika_piper.kinematics')
    real=importlib.import_module('pika_piper.kinematics')
    root=Path(__file__).resolve().parents[2]
    k=real.Kinematics(root/'piper_description/urdf/piper_description.urdf',root/'piper_description',[0,0,.1358,0,0,0],root/'piper_moveit/piper_with_gripper_moveit/config/piper_description.srdf')
    c=controller;c.k=k;c.feedback=np.r_[np.deg2rad([-.596,30.118,-23.212,-3.097,34.591,3.656]),.03];c.width=.03
    from pika_piper.teleop import TeleopIK
    c.teleop_solver=TeleopIK(k)
    c.cfg.update(axis_rpy=[0,0,0],translation_scale=1.,max_joint_speed=.5,max_gripper_speed=.04)
    commands=[];c.cmd=types.SimpleNamespace(publish=lambda msg:commands.append(np.array(msg.position)))
    c.request=types.SimpleNamespace(publish=lambda msg:None)
    c.hand_at=c.grip_at=c.quality_at=c.feedback_at=time.monotonic()
    assert c.set_teleop(types.SimpleNamespace(data=True)).success
    previous=c.feedback.copy()
    for i in range(20):
        c.hand[0,3]=(i+1)*.00005
        now=time.monotonic();c.hand_at=c.grip_at=c.quality_at=c.feedback_at=now;c.echo_at=now;c.last_tick=now-.02
        c.tick(None)
        assert c.mode=='teleop',c.reason
        k.validate_configuration(commands[-1])
        assert np.max(abs(commands[-1][:6]-previous[:6])) <= .020001
        previous=commands[-1];c.feedback=previous.copy()
    assert len(commands)==20


def test_delayed_driver_echo_does_not_rewind_issued_command(controller):
    c=controller;c.mode='teleop';c.sent=np.r_[np.full(6,.04),.03]
    issued=c.sent.copy();c.publish_eef=lambda *args:None;c.eef_cmd=object()
    stamp=type(c.__class__.on_sent.__globals__['rospy'].Time.now())()
    msg=types.SimpleNamespace(position=np.r_[np.full(6,.01),.03],header=types.SimpleNamespace(stamp=stamp))
    c.on_sent(msg)
    np.testing.assert_array_equal(c.sent,issued)
    np.testing.assert_array_equal(c.last_echo,msg.position)
    assert c.fresh(c.echo_at)


def test_current_audited_feedback_can_acquire_with_legal_first_command(controller,monkeypatch):
    import yaml
    monkeypatch.delitem(sys.modules,'pika_piper.kinematics')
    real=importlib.import_module('pika_piper.kinematics')
    from pika_piper.teleop import TeleopIK
    root=Path(__file__).resolve().parents[2]
    cfg=yaml.safe_load((root/'pika_piper/config/pipeline.yaml').read_text())
    c=controller;c.cfg=cfg
    c.k=real.Kinematics(root/'piper_description/urdf/piper_description.urdf',root/'piper_description',cfg['tcp_xyzrpy'],
        root/'piper_moveit/piper_with_gripper_moveit/config/piper_description.srdf',
        feedback_limits_deg=cfg['verified_feedback_limits_deg'],feedback_tolerance=cfg['feedback_tolerance_rad'])
    c.teleop_solver=TeleopIK(c.k)
    c.feedback=np.r_[np.deg2rad([-65.365,22.667,-32.257,72.466,69.912,-78.878]),0.]
    raw=c.feedback.copy();c.width=0.;commands=[]
    c.cmd=types.SimpleNamespace(publish=lambda msg:commands.append(np.array(msg.position)))
    c.request=types.SimpleNamespace(publish=lambda msg:None)
    c.hand_at=c.grip_at=c.quality_at=c.feedback_at=time.monotonic()
    result=c.set_teleop(types.SimpleNamespace(data=True))
    assert result.success,result.message
    now=time.monotonic();c.hand_at=c.grip_at=c.quality_at=c.feedback_at=c.echo_at=now;c.last_tick=now-.02
    c.tick(None)
    assert c.mode=='teleop',c.reason
    np.testing.assert_array_equal(c.feedback,raw)
    c.k.validate_configuration(commands[-1])
    assert commands[-1][4]<raw[4]
    assert c.tracking['feedback_recovery']
    # Strict replay cannot silently project the same out-of-model start.
    c.set_teleop(types.SimpleNamespace(data=False))
    result=c.set_replay(types.SimpleNamespace(data=True))
    assert not result.success and 'joint5' in result.message


@pytest.mark.parametrize('pause_while_waiting',[False,True])
def test_near_zero_start_waits_without_emitting_illegal_commands(controller,monkeypatch,pause_while_waiting):
    import yaml
    monkeypatch.delitem(sys.modules,'pika_piper.kinematics')
    real=importlib.import_module('pika_piper.kinematics')
    from pika_piper.teleop import TeleopIK
    root=Path(__file__).resolve().parents[2]
    c=controller;c.cfg=yaml.safe_load((root/'pika_piper/config/pipeline.yaml').read_text())
    c.cfg['max_joint_acceleration']=10.  # Exercise the two-tick waiting contract.
    c.k=real.Kinematics(root/'piper_description/urdf/piper_description.urdf',root/'piper_description',c.cfg['tcp_xyzrpy'],
        root/'piper_moveit/piper_with_gripper_moveit/config/piper_description.srdf',
        feedback_limits_deg=c.cfg['verified_feedback_limits_deg'],feedback_tolerance=c.cfg['feedback_tolerance_rad'],
        near_zero_start_tolerance_deg=c.cfg['near_zero_start_tolerance_deg'])
    c.teleop_solver=TeleopIK(c.k)
    c.feedback=np.r_[np.deg2rad([0,-.219,.290,0,0,0]),.03];c.width=.03
    raw=c.feedback.copy();commands=[]
    c.cmd=types.SimpleNamespace(publish=lambda msg:commands.append(np.array(msg.position)))
    c.request=types.SimpleNamespace(publish=lambda msg:None)
    c.hand_at=c.grip_at=c.quality_at=c.feedback_at=time.monotonic()
    assert c.set_teleop(types.SimpleNamespace(data=True)).success
    def tick():
        now=time.monotonic();c.hand_at=c.grip_at=c.quality_at=c.feedback_at=c.echo_at=now;c.last_tick=now-.02
        c.tick(None)
    tick()
    assert c.mode=='teleop',c.reason
    assert not commands
    assert c.tracking['state']=='recovering' and not c.tracking['command_valid']
    if pause_while_waiting:
        c.set_teleop(types.SimpleNamespace(data=False));tick()
        assert not commands and c.startup_target is None
        return
    c.hand[0,3]=.05
    tick()
    assert c.mode=='teleop',c.reason
    assert len(commands)==1
    first=commands[0];c.k.validate_configuration(first)
    assert first[1]==first[2]==0.
    np.testing.assert_array_equal(c.feedback,raw)
    assert np.max(abs(first[:6]-raw[:6]))/.04<=.3+1e-6
    assert np.max(abs(first[:6]-raw[:6]))/.04**2<=10.+1e-4
    assert c.startup_target is None
    np.testing.assert_array_equal(c.anchor,c.hand)
    # The hand's movement during startup is not chased on the next tick.
    c.feedback=first.copy();tick()
    assert c.mode=='teleop',c.reason
    assert np.max(abs(commands[-1][:6]-first[:6]))<1e-5


def test_stale_echo_does_not_refresh_watchdog(controller):
    c=controller;c.publish_eef=lambda *args:pytest.fail('stale echo published')
    Stamp=type(c.__class__.on_sent.__globals__['rospy'].Time.now())
    c.on_sent(types.SimpleNamespace(position=np.zeros(7),header=types.SimpleNamespace(stamp=Stamp(0.))))
    assert c.echo_at==0.


def test_sense_gesture_toggles_and_debounces_without_claiming_replay(controller):
    c=controller;c.cfg['sense_gesture_control']=True
    c.on_gesture(types.SimpleNamespace(data=1));assert c.mode=='teleop'
    c.gesture_at-=1
    c.on_gesture(types.SimpleNamespace(data=1));assert c.mode=='teleop'
    c.on_gesture(types.SimpleNamespace(data=2));assert c.mode=='idle'
    c.on_gesture(types.SimpleNamespace(data=3));assert c.mode=='idle' # debounce
    assert c.set_replay(types.SimpleNamespace(data=True)).success
    c.gesture_at-=1
    c.on_gesture(types.SimpleNamespace(data=4));assert c.mode=='replay'


def test_disabled_sense_gesture_cannot_start_motion(controller):
    controller.on_gesture(types.SimpleNamespace(data=1))
    assert controller.mode=='idle'


def test_large_hand_goal_is_limited_before_ik(controller):
    c=controller;c.sent=np.zeros(7);c.width=.03;targets=[]
    c.k.ik=lambda target,seed,width:targets.append(target.copy())
    goal=np.eye(4);goal[0,3]=.1
    c.teleop_ik(goal,.02)
    assert c.target_limited
    assert targets[-1][0,3]==pytest.approx(.003)


def test_resume_resets_sense_reference_after_repositioning(controller):
    c=controller;on=types.SimpleNamespace(data=True);off=types.SimpleNamespace(data=False)
    assert c.set_teleop(on).success
    assert c.set_teleop(off).success
    assert c.anchor is None
    from pika_piper.core import transform,relative_target
    c.hand=transform([.2,.3,.4,.2,-.3,.4])
    c.hand_at=c.feedback_at=c.grip_at=c.quality_at=time.monotonic()
    assert c.set_teleop(on).success
    np.testing.assert_allclose(np.linalg.inv(c.anchor)@c.hand,np.eye(4),atol=1e-12)
    np.testing.assert_allclose(relative_target(c.robot_anchor,c.anchor,c.hand,c.axes),c.robot_anchor,atol=1e-12)


def test_tick_keeps_publishing_status_when_tcp_limiting_activates(controller):
    # A large hand movement makes the limiter's fraction a NumPy scalar.
    # Exercise the complete tick, including command and JSON status output,
    # then a second tick to prove the callback can keep running.
    import json
    c=controller;c.width=.03
    c.cfg.update(translation_scale=1.,max_joint_speed=.5,max_gripper_speed=.04)
    commands=[];states=[]
    c.cmd=types.SimpleNamespace(publish=commands.append)
    c.request=types.SimpleNamespace(publish=lambda msg:None)
    c.state=types.SimpleNamespace(publish=lambda msg:states.append(json.loads(msg.data)))
    assert c.set_teleop(types.SimpleNamespace(data=True)).success
    for distance in [.1,0.]:
        c.hand[0,3]=distance
        now=time.monotonic()
        c.hand_at=c.feedback_at=c.grip_at=c.quality_at=c.echo_at=now
        c.last_tick=now-.02
        c.tick(None)
    assert len(commands)==len(states)==2
    assert [s['target_limited'] for s in states]==[True,False]
    assert all(s['mode']=='teleop' for s in states)
