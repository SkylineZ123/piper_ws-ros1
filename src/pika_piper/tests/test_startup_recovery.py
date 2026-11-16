"""Regress the reported J2=-0.219/J3=+0.290 deg near-zero handover."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from pika_piper.kinematics import Kinematics
from pika_piper.teleop import TeleopIK
from pika_piper.core import effective_joint_speed


@pytest.fixture(scope='module')
def startup():
    root=Path(__file__).resolve().parents[2]
    cfg=yaml.safe_load((root/'pika_piper/config/pipeline.yaml').read_text())
    k=Kinematics(root/'piper_description/urdf/piper_description.urdf',root/'piper_description',
        cfg['tcp_xyzrpy'],root/'piper_moveit/piper_with_gripper_moveit/config/piper_description.srdf',
        feedback_limits_deg=cfg['verified_feedback_limits_deg'],
        feedback_tolerance=cfg['feedback_tolerance_rad'],
        near_zero_start_tolerance_deg=cfg['near_zero_start_tolerance_deg'])
    return k,TeleopIK(k),cfg


def measured(j2=-.219,j3=.290):
    return np.r_[np.deg2rad([0,j2,j3,0,0,0]),.03]


def test_reported_deviation_is_startup_only(startup):
    k,t,cfg=startup
    raw=measured();saved=raw.copy()
    k.validate_teleop_start(raw)
    with pytest.raises(ValueError,match='joint2'):
        k.validate_feedback(raw)
    with pytest.raises(ValueError,match='joint2'):
        k.validate_configuration(raw)
    with pytest.raises(ValueError,match='joint2'):
        t.step(k.fk(raw),raw,.03,.02,np.zeros(6),cfg)
    target=k.start_target(raw)
    assert target[1]==target[2]==0.
    np.testing.assert_array_equal(raw,saved)
    np.testing.assert_array_equal(target[[0,3,4,5,6]],raw[[0,3,4,5,6]])


@pytest.mark.parametrize('period',[.005,.01,.02])
@pytest.mark.parametrize('j2,j3',[(-.219,.290),(-.49,.49)])
def test_first_command_waits_and_is_inside_normal_limits(startup,period,j2,j3):
    k,t,cfg=startup
    raw=measured(j2,j3);target=k.start_target(raw)
    elapsed=0.;writes=[]
    for _ in range(200):
        elapsed+=period
        command=t.startup(raw,target,elapsed,cfg,k.fk(raw))
        if command is not None:
            writes.append(command);break
        assert not t.last['command_valid']
        assert not t.last['accepted'] and t.last['target_matrix'] is None
    assert len(writes)==1
    command=writes[0];k.validate_configuration(command)
    np.testing.assert_array_equal(command,target)
    assert np.max(abs(command[:6]-raw[:6]))/elapsed <= effective_joint_speed(cfg)+1e-9
    assert np.max(abs(command[:6]-raw[:6]))/elapsed**2 <= cfg['max_joint_acceleration']+1e-8
    assert t.last['recovery_complete']
    assert np.max(abs(command[:6]-raw[:6]))<=np.deg2rad(.5)


@pytest.mark.parametrize('j2,j3',[(-.501,0),(0,.501),(-2.017,1.825)])
def test_large_zero_deviation_is_still_rejected(startup,j2,j3):
    k,_,_=startup
    with pytest.raises(ValueError,match='joint[23]'):
        k.validate_teleop_start(measured(j2,j3))


def test_other_boundaries_and_gripper_are_not_relaxed(startup):
    k,_,_=startup
    for index,value in [(3,np.deg2rad(-100.01)),(4,np.deg2rad(70.01)),(6,.0888)]:
        q=measured();q[index]=value
        with pytest.raises(ValueError):k.validate_teleop_start(q)


def test_startup_cannot_send_an_arbitrary_near_zero_command(startup):
    k,t,cfg=startup
    raw=measured();target=k.start_target(raw);target[4]=.1
    with pytest.raises(ValueError,match='nearest legal boundary'):
        t.startup(raw,target,.1,cfg,k.fk(raw))


def test_warmup_budget_is_separate_from_motion_acceptance(startup,monkeypatch):
    k,_,_=startup;t=TeleopIK(k)
    monkeypatch.setattr(t,'_solve',lambda *args:(None,'Maximum_CpuTime_Exceeded'))
    t.warmup(k.start_target(measured()))
    assert t.warmup_status=='Maximum_CpuTime_Exceeded'
    assert t.last is None  # No authorized command is manufactured by warmup.
    monkeypatch.setattr(t,'_solve',lambda *args:(None,'Invalid_Number_Detected'))
    with pytest.raises(ValueError,match='Invalid_Number_Detected'):
        t.warmup(k.start_target(measured()))
