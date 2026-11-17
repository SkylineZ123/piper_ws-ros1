"""Real model tests of bounded tracking, recovery and rejection contracts."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from utils.core import transform, effective_joint_speed
from utils.kinematics import Kinematics
from utils.teleop import TeleopIK, motion_bounds, pose_error


@pytest.fixture(scope='module')
def tracking():
    root = Path(__file__).resolve().parents[2]
    k = Kinematics(root/'piper_description/urdf/piper_description.urdf',
                   root/'piper_description', [0,0,.1358,0,0,0],
                   root/'piper_moveit/piper_with_gripper_moveit/config/piper_description.srdf')
    t = TeleopIK(k)
    config = yaml.safe_load((root/'pika_piper/config/pipeline.yaml').read_text())
    q = np.array([.1,.7,-1.2,.2,.5,.1,.03])
    t.warmup(q)
    return k, t, config


def recorded_faults():
    path = Path(__file__).resolve().parents[1]/'docs/ik_review_evidence_20261116.json'
    return json.loads(path.read_text())['bag']['ik_failure_events']


@pytest.mark.parametrize('event', recorded_faults(), ids=lambda e: e['status']['reason'])
def test_recorded_fault_has_bounded_command_without_branch_jump(tracking, event):
    k, t, config = tracking
    state = event['status']
    previous = np.r_[state['ik']['seed_rad'], .03]
    goal = np.array(state['ik']['target_matrix'])
    command = t.step(goal, previous, .03, .02, np.zeros(6), dict(config, **state['motion_limits']))
    k.validate_configuration(command)
    assert np.max(np.abs(command[:6] - previous[:6])) <= config['max_joint_acceleration'] * .02**2 + 1e-8
    assert t.last['command_valid']
    assert t.last['state'] in ('tracking', 'limited', 'saturated', 'braking')
    # A held/braking result must not masquerade as an accurate pose solution.
    if t.last['accepted']:
        assert t.last['position_error_m'] <= .002
        assert t.last['orientation_error_rad'] <= np.deg2rad(2)
    else:
        assert t.last['target_matrix'] is None
        assert t.last['position_error_m'] is None
    assert t.last['raw_position_error_m'] >= 0
    json.dumps(t.last, allow_nan=False)


def test_saturated_wrist_can_retreat_with_same_absolute_goal_reference(tracking):
    k, t, config = tracking
    previous = np.array([.1,.7,-1.2,.2,1.219,.1,.03])
    velocity = np.zeros(6)
    outward = previous.copy(); outward[4] = 1.45
    inward = previous.copy(); inward[4] = .85
    states = []
    for goal in [k.fk(outward)] * 35 + [k.fk(inward)] * 100:
        command = t.step(goal, previous, .03, .02, velocity, config)
        new_velocity = (command[:6] - previous[:6]) / .02
        assert np.max(np.abs(new_velocity)) <= effective_joint_speed(config) + 1e-7
        assert np.max(np.abs(new_velocity - velocity)) / .02 <= config['max_joint_acceleration'] + 1e-5
        k.validate_configuration(command)
        states.append(t.last['state'])
        previous, velocity = command, new_velocity
    assert 'saturated' in states[:35]
    assert previous[4] < 1.0
    position, orientation = pose_error(k.fk(previous), k.fk(inward))
    assert position < .002 and orientation < np.deg2rad(2)


def test_failed_solver_brakes_and_does_not_claim_goal_accuracy(tracking, monkeypatch):
    k, t, config = tracking
    config=dict(config,max_joint_acceleration=10.)
    previous = np.array([.1,.7,-1.2,.2,.5,.1,.03])
    monkeypatch.setattr(t, '_solve', lambda *args: (None, 'Maximum_CpuTime_Exceeded'))
    velocity = np.full(6, .3)
    command = t.step(k.fk(previous) @ transform([.1,0,0,0,0,0]), previous, .03,
                     .02, velocity, config)
    assert t.last['state'] == 'braking'
    assert not t.last['accepted'] and t.last['command_valid']
    np.testing.assert_allclose((command[:6] - previous[:6]) / .02, .1, atol=1e-8)
    assert t.last['position_error_m'] is None
    assert t.last['raw_position_error_m'] > .05


def test_collision_does_not_become_recoverable_saturation(tracking, monkeypatch):
    k, t, config = tracking
    previous = np.array([.1,.7,-1.2,.2,.5,.1,.03])
    original = k.validate_configuration
    def check(q, context, **kwargs):
        if context == 'teleop command segment':
            raise ValueError('self-collision')
        original(q, context, **kwargs)
    monkeypatch.setattr(k, 'validate_configuration', check)
    with pytest.raises(ValueError, match='self-collision'):
        t.step(k.fk(previous), previous, .03, .02, np.zeros(6), config)
    assert not t.last['command_valid']


def test_invalid_feedback_and_target_are_never_projected(tracking):
    k, t, config = tracking
    previous = np.array([.1,-.01,-1.2,.2,.5,.1,.03])
    with pytest.raises(ValueError, match='joint2'):
        t.step(k.fk(previous), previous, .03, .02, np.zeros(6), config)
    previous[1] = .7
    with pytest.raises(ValueError, match='rigid'):
        t.step(np.zeros((4,4)), previous, .03, .02, np.zeros(6), config)
    with pytest.raises(ValueError, match='gripper'):
        t.step(k.fk(previous), previous, .0888, .02, np.zeros(6), config)


def test_margin_and_stopping_distance_brake_before_hardware_boundary(tracking):
    k, _, _ = tracking
    q = np.array([.1,.7,-1.2,.2,1.17,.1])
    velocity = np.zeros(6)
    # Adversarially choose the furthest outward admissible command each step.
    for i in range(80):
        lower, upper, _ = motion_bounds(k.model, q, velocity, .02, 3., 10., .035)
        command = q.copy(); command[4] = upper[4]
        new_velocity = (command - q) / .02
        assert abs(new_velocity[4] - velocity[4]) <= .2 + 1e-7
        assert command[4] <= k.model.upperPositionLimit[4] - .035 + 1e-7
        q, velocity = command, new_velocity
    assert abs(velocity[4]) < 1e-7


def test_static_target_does_not_drift(tracking):
    k, t, config = tracking
    previous = np.array([.1,.7,-1.2,.2,.5,.1,.03])
    initial = previous.copy(); target = k.fk(previous); velocity = np.zeros(6)
    for _ in range(20):
        command = t.step(target, previous, .03, .02, velocity, config)
        velocity = (command[:6] - previous[:6]) / .02
        previous = command
    np.testing.assert_allclose(previous, initial, atol=1e-6)


def test_audited_feedback_recovers_inward_without_changing_command_bounds():
    root=Path(__file__).resolve().parents[2]
    cfg=yaml.safe_load((root/'pika_piper/config/pipeline.yaml').read_text())
    k=Kinematics(root/'piper_description/urdf/piper_description.urdf',root/'piper_description',
        cfg['tcp_xyzrpy'],root/'piper_moveit/piper_with_gripper_moveit/config/piper_description.srdf',
        feedback_limits_deg=cfg['verified_feedback_limits_deg'],feedback_tolerance=cfg['feedback_tolerance_rad'])
    previous=np.r_[np.deg2rad([-65.365,22.667,-32.257,72.466,69.912,-78.878]),0.]
    raw=previous.copy();k.validate_feedback(previous)
    with pytest.raises(ValueError,match='joint5'):
        k.validate_configuration(previous)
    t=TeleopIK(k);t.warmup(previous)
    command=t.step(k.fk(previous),previous,0.,.02,np.zeros(6),cfg)
    np.testing.assert_array_equal(previous,raw)
    k.validate_configuration(command)
    assert command[4]<previous[4]
    assert np.max(abs(command[:6]-previous[:6]))<=.004+1e-8
    assert t.last['feedback_recovery']
    assert t.last['effective_joint_speed_rad_s']==effective_joint_speed(cfg)
    # Firmware permits J6 ±180°, but no command/model range is expanded here.
    assert np.rad2deg(k.model.upperPositionLimit[5])<121
    outside=previous.copy();outside[4]=np.deg2rad(70.01)
    with pytest.raises(ValueError,match='joint5'):
        k.validate_feedback(outside)
    outside=previous.copy();outside[3]=np.deg2rad(-100.01)
    with pytest.raises(ValueError,match='joint4'):
        k.validate_feedback(outside)
    outside=previous.copy();outside[1]=np.deg2rad(-2.017)
    with pytest.raises(ValueError,match='joint2'):
        k.validate_feedback(outside)
