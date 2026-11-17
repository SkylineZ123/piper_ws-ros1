"""Upstream publication/seed behavior and measured feedback regressions."""
import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest
import yaml

from utils.kinematics import Kinematics
from utils.official import OfficialInterpolator, OfficialTeleopIK


@pytest.fixture(scope='module')
def model():
    root=Path(__file__).resolve().parents[2]
    cfg=yaml.safe_load((root/'pika_piper/config/pipeline.yaml').read_text())
    k=Kinematics(root/'piper_description/urdf/piper_description.urdf',root/'piper_description',cfg['tcp_xyzrpy'],
        root/'piper_moveit/piper_with_gripper_moveit/config/piper_description.srdf',
        feedback_limits_deg=cfg['verified_feedback_limits_deg'],feedback_tolerance=cfg['feedback_tolerance_rad'])
    import controller_harness as bench
    return k,cfg,bench


def test_default_acceptance_is_upstream_per_axis_without_orientation_rejection(model):
    k,_,_=model
    q=np.array([.1,.7,-1.2,.2,.5,.1,.03])
    k.last_ik={'position_axis_error_m':.299,'orientation_error_rad':2.5,'accepted':False}
    np.testing.assert_array_equal(k.accept_ik(q),q)
    assert k.last_ik['accepted']
    k.last_ik={'position_axis_error_m':.301,'orientation_error_rad':0.,'accepted':False}
    with pytest.raises(ValueError,match='300.000'):k.accept_ik(q)
    assert not k.last_ik['accepted']
    assert k.info['orientation_tolerance_rad'] is None
    assert np.rad2deg(k.model.upperPositionLimit[5])==pytest.approx(120.,abs=.001)


def test_full_target_is_not_shrunk_or_bound_by_command_acceleration(model):
    k,cfg,_=model
    t=OfficialTeleopIK(k)
    previous=np.array([.1,.7,-1.2,.2,.5,.1,.03])
    t.initial=previous[:6].copy();t.history=previous[:6].copy()
    goal=previous.copy();goal[0]+=.3
    command=t.step(k.fk(goal),previous,.03,.02,np.zeros(6),cfg)
    assert command is not None,t.last
    assert np.max(abs(command[:6]-previous[:6]))>.1
    assert not t.last['tcp_limited'] and t.last['step_fraction']==1.
    k.validate_configuration(command)


def test_official_seed_history_resets_next_seed_after_thirty_degrees():
    candidates=[np.array([.6,.7,-1.2,.2,.5,.1,.03]),np.array([.61,.7,-1.2,.2,.5,.1,.03])]
    seen=[]
    k=types.SimpleNamespace(last_ik=None,validate_feedback=lambda *a:None,fk=lambda q:np.eye(4))
    def solve(target,seed,width,posture_reference=None):
        seen.append(seed.copy());k.last_ik={'accepted':False};return candidates[len(seen)-1].copy()
    k.solve_ik=solve;k.accept_ik=lambda q:q
    t=OfficialTeleopIK(k)
    t.step(np.eye(4),np.r_[np.zeros(6),.03],.03,.02,np.zeros(6),{})
    assert t.last['seed_reset'];np.testing.assert_array_equal(t.initial,np.zeros(6))
    t.step(np.eye(4),candidates[0],.03,.02,np.zeros(6),{})
    assert not t.last['seed_reset'];np.testing.assert_array_equal(t.initial,candidates[1][:6])
    np.testing.assert_array_equal(seen,np.zeros((2,6)))


def test_small_changes_publish_directly_as_upstream():
    scheduler=OfficialInterpolator();initial=np.zeros(7);goal=initial.copy();goal[0]=np.deg2rad(20.)
    scheduler.offer(goal,10.)
    np.testing.assert_allclose(scheduler.sample(initial,10.),goal)
    assert scheduler.sample(goal,10.005) is None
    np.testing.assert_allclose(scheduler.sample(goal,10.02),goal)


def test_large_change_uses_upstream_one_degree_200hz_interpolation_and_cached_goal():
    scheduler=OfficialInterpolator();initial=np.zeros(7);goal=initial.copy();goal[0]=np.deg2rad(80.)
    scheduler.offer(goal,10.);steps=int(goal[0]/np.deg2rad(1.))
    commands=[];previous=initial.copy()
    for i in range(steps):
        command=scheduler.sample(previous,10.+i*.005)
        assert command is not None
        commands.append(command.copy());previous=command
        if i==5:scheduler.offer(-goal,10.+i*.005)
    np.testing.assert_allclose(np.array(commands)[:,0],np.linspace(0,goal[0],steps+1)[1:])
    np.testing.assert_allclose(commands[-1],goal)
    assert scheduler.sample(previous,10.+(steps-1)*.005+.005) is None
    next_command=scheduler.sample(previous,10.+(steps-1)*.005+.02)
    assert next_command[0]<goal[0]


def test_stale_cached_target_is_not_replayed():
    scheduler=OfficialInterpolator();initial=np.zeros(7);goal=np.ones(7)
    scheduler.offer(goal,10.);assert scheduler.sample(initial,10.) is not None
    assert scheduler.sample(initial,11.001) is None and scheduler.segment is None


def test_fresh_stationary_feedback_is_reported_then_stops_output(model, monkeypatch):
    k,cfg,bench=model;initial=np.array([.1,.7,-1.2,.2,.5,.1,.03])
    c,clock,result=bench.controller(k,cfg,initial,monkeypatch)
    assert isinstance(c.teleop_solver,OfficialTeleopIK)
    goal=initial.copy();goal[0]+=.3
    states=[]
    for _ in range(100):
        clock.now+=.02;c.hand=np.linalg.inv(c.robot_anchor)@k.fk(goal)
        c.hand_at=c.feedback_at=c.grip_at=c.quality_at=c.echo_at=clock.now
        c.feedback=initial.copy();c.tick(None)
        states.append(result.status['tracking']['state'])
        if c.mode=='fault':break
    assert 'lagging' in states
    assert c.mode=='fault' and 'execution stalled' in c.reason
    assert c.tracking['raw_position_error_m']>.05
    assert c.tracking['raw_orientation_error_rad']>.2
    assert c.command_schedule.target is None
    previous=result.command.copy();c.command_tick(None)
    np.testing.assert_array_equal(result.command,previous)


def test_runtime_invalid_feedback_is_rejected_before_output(model, monkeypatch):
    k,cfg,bench=model;initial=np.array([.1,.7,-1.2,.2,.5,.1,.03])
    c,clock,result=bench.controller(k,cfg,initial,monkeypatch)
    c.feedback=initial.copy();c.feedback[1]=-.04
    clock.now+=.02;c.hand_at=c.feedback_at=c.grip_at=c.quality_at=c.echo_at=clock.now
    c.tick(None)
    assert c.mode=='fault' and 'joint2' in c.reason
    assert result.command is None


def test_rejected_target_cache_expires_without_false_driver_fault_and_can_recover(model, monkeypatch):
    k,cfg,bench=model;initial=np.array([.1,.7,-1.2,.2,.5,.1,.03])
    c,clock,result=bench.controller(k,cfg,initial,monkeypatch)
    clock.now+=.02;c.hand_at=c.feedback_at=c.grip_at=c.quality_at=c.echo_at=clock.now
    c.tick(None);assert result.command is not None
    c.echo_at=clock.now
    invalid=k.fk(initial);invalid[0,3]+=2.
    for _ in range(85):
        clock.now+=.02;c.hand=np.linalg.inv(c.robot_anchor)@invalid
        c.hand_at=c.feedback_at=c.grip_at=c.quality_at=clock.now
        c.feedback=c.sent.copy();before=c.command_publish_at
        c.tick(None)
        # Exercise the independent publisher even when the new IK is rejected.
        c.command_tick(None)
        if c.command_publish_at>before:c.echo_at=clock.now
        assert c.mode=='teleop',c.reason
    assert c.command_schedule.target is None
    assert c.tracking['state']=='unreachable'
    clock.now+=.02;c.hand=np.eye(4)
    c.hand_at=c.feedback_at=c.grip_at=c.quality_at=clock.now
    c.tick(None)
    assert c.mode=='teleop' and c.command_schedule.target is not None
