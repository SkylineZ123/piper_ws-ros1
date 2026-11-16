#!/usr/bin/env python3
"""Offline FK/IK and complete Controller.tick benchmark; never starts ROS/CAN.

ROS messages, publishers and clocks are simulated. Commands and geometry use
the real production controller and solver. This is not hardware accuracy or
ROS transport validation. Run after sourcing this workspace's devel/setup.bash.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch

import numpy as np
import rospkg
import yaml

from pika_piper.core import limit_step, limit_pose_step, effective_joint_speed
from pika_piper.kinematics import Kinematics
from pika_piper.teleop import TeleopIK, pose_error
from pika_piper.official import make_teleop_solver


def distribution(values):
    if not values:
        return None
    return {name: float(value) for name, value in zip(
        ('p50', 'p95', 'p99', 'max'),
        (*np.percentile(values, [50, 95, 99]), max(values)))}


def controller(k, config, initial):
    """Real Controller methods, memory publishers and a deterministic 50 Hz clock."""
    import threading
    clock = types.SimpleNamespace(now=10.)
    class Stamp:
        def to_sec(self): return clock.now
    class Message:
        def __init__(self, *args):
            self.data = args[0] if args else ''
            self.success = args[0] if args else False
            self.message = args[1] if len(args) > 1 else ''
            self.header = types.SimpleNamespace(stamp=Stamp(), frame_id='')
            self.pose = types.SimpleNamespace(position=types.SimpleNamespace(x=0.,y=0.,z=0.),
                orientation=types.SimpleNamespace(x=0.,y=0.,z=0.,w=1.))
    fake_ros = types.ModuleType('rospy')
    fake_ros.Time = types.SimpleNamespace(now=Stamp)
    fake_ros.logerr = fake_ros.logwarn = lambda *args: None
    fake_ros.get_param = lambda name,default=None: {
        'ready':True,'output_mode':config.get('output_mode','position'),
        'sdk_protocol':config.get('sdk_protocol','v2'),
        'joint_speed_limits_rad_s':[effective_joint_speed(config)]*6}
    mocks = {'rospy': fake_ros}
    for module, names in {'sensor_msgs.msg':['JointState'], 'geometry_msgs.msg':['PoseStamped'],
            'std_msgs.msg':['String','Int32'], 'std_srvs.srv':['SetBool','SetBoolRequest','SetBoolResponse','Trigger','TriggerResponse'],
            'data_msgs.msg':['Gripper','LocalizationStatus','TeleopStatus']}.items():
        mocks[module.split('.')[0]] = types.ModuleType(module.split('.')[0])
        mocks[module] = types.ModuleType(module)
        for name in names: setattr(mocks[module], name, Message)
    path = Path(__file__).with_name('control_node.py')
    spec = importlib.util.spec_from_file_location('offline_controller_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, mocks):
        spec.loader.exec_module(module)
    module.time = types.SimpleNamespace(monotonic=lambda:clock.now, perf_counter=time.perf_counter)
    c = module.Controller.__new__(module.Controller)
    c.cfg = dict(config); c.k = k; c.teleop_solver = make_teleop_solver(k,config)
    c.lock = threading.RLock(); c.mode = 'idle'; c.reason = ''; c.last_fault = None
    c.hand = np.eye(4); c.axes = np.eye(4); c.hand_tcp = np.eye(4)
    c.feedback = initial.copy(); c.sent = None; c.width = float(initial[6])
    c.accurate = True; c.anchor = c.robot_anchor = None
    c.command_started = c.echo_at = 0.; c.command_velocity = np.zeros(6); c.tracking = None
    c.target_limited = c.joint_limited = False; c.timing = {}
    c.replay = None; c.gesture_at = 0.; c.gesture_command = None; c.last_echo = None
    c.hand_at = c.feedback_at = c.grip_at = c.quality_at = c.replay_at = c.last_tick = clock.now
    result = types.SimpleNamespace(status=None, command=None)
    c.cmd = types.SimpleNamespace(publish=lambda msg:setattr(result, 'command', np.array(msg.position)))
    c.state = types.SimpleNamespace(publish=lambda msg:setattr(result, 'status', json.loads(msg.data)))
    for name in ('request','feasible_request','sense_relative','teleop_status'):
        setattr(c, name, types.SimpleNamespace(publish=lambda msg:None))
    acquired = c.set_teleop(types.SimpleNamespace(data=True))
    if not acquired.success:
        raise ValueError('Offline acquisition failed: ' + acquired.message)
    return c, clock, result


def trajectory(k, config, name, references, lag=0., delay=0):
    initial = np.r_[references[0], .03]
    c, clock, receipt = controller(k, config, initial)
    previous = initial.copy(); velocity = np.zeros(6)
    old = initial.copy(); old_fault = None; old_first_fault = None
    timings = []; positions = []; orientations = []; goal_positions = []; goal_orientations = []
    measured_positions=[];measured_orientations=[];axis_errors=[]
    states = {}; maximum_speed = maximum_acceleration = maximum_jump = 0.
    maximum_feedback_gap = 0.; qualified = budget_exceeded = 0
    pending = [initial.copy()] * (delay*4)
    actual = initial.copy(); faults = []; final = None
    for i, reference in enumerate(references):
        target = k.fk(reference)
        if old_fault is None:
            try:
                # Direct-send reference using the CURRENT pose acceptance,
                # not a frozen before-change solver or precision baseline.
                stepped, _ = limit_pose_step(k.fk(old), target, .02,
                    config['max_tcp_speed'], config['max_tcp_angular_speed'])
                candidate = k.ik(stepped, old, .03)
                old = limit_step(old, candidate, .02, config['max_joint_speed'], config['max_gripper_speed'])
                k.validate_configuration(old)
            except ValueError as exc:
                old_fault = str(exc); old_first_fault = i
        c.hand = np.linalg.inv(c.robot_anchor) @ target
        for substep in range(4):
            clock.now+=.005
            c.hand_at=c.feedback_at=c.grip_at=c.quality_at=c.echo_at=clock.now
            c.feedback=actual.copy()
            if substep==0:
                started=time.perf_counter();c.tick(None)
                timings.append((time.perf_counter()-started)*1000)
            else:c.command_tick(None)
            if c.mode!='teleop':break
            pending.append(c.sent.copy());delayed=pending.pop(0)
            actual+=(delayed-actual)*(1. if lag<=0 else min(1.,.005/lag))
        if c.mode != 'teleop':
            faults.append(dict(frame=i, reason=c.reason)); break
        command = c.sent
        k.validate_configuration(command)
        new_velocity = (command[:6] - previous[:6]) / .02
        maximum_speed = max(maximum_speed, float(np.max(abs(new_velocity))))
        maximum_acceleration = max(maximum_acceleration, float(np.max(abs(new_velocity - velocity))) / .02)
        maximum_jump = max(maximum_jump, float(np.max(abs(command[:6] - previous[:6]))))
        maximum_feedback_gap = max(maximum_feedback_gap, float(np.max(abs(command[:6] - actual[:6]))))
        position, orientation = pose_error(k.fk(command), target)
        positions.append(position); orientations.append(orientation)
        measured_position,measured_orientation=pose_error(k.fk(actual),target)
        measured_positions.append(measured_position);measured_orientations.append(measured_orientation)
        tracking = receipt.status['tracking']; final = tracking
        states[tracking['state']] = states.get(tracking['state'], 0) + 1
        budget_exceeded += int(tracking['budget_exceeded'])
        if tracking['accepted']:
            qualified += 1
            goal_positions.append(tracking['position_error_m'])
            goal_orientations.append(tracking['orientation_error_rad'])
            if 'position_axis_error_m' in tracking:axis_errors.append(tracking['position_axis_error_m'])
        previous, velocity = command.copy(), new_velocity
    end_goal = k.fk(references[-1])
    old_position, old_orientation = pose_error(k.fk(old), end_goal)
    position, orientation = pose_error(k.fk(previous), end_goal)
    return dict(name=name,frames=len(positions),requested_frames=len(references),faults=faults,
        feedback_model=dict(time_constant_s=lag,delay_frames=delay),
        old_first_fault_frame=old_first_fault,old_fault=old_fault,
        old_final_position_error_m=old_position,old_final_orientation_error_rad=old_orientation,
        final_position_error_m=position,final_orientation_error_rad=orientation,
        tracking_states=states,qualified_step_count=qualified,budget_exceeded_frames=budget_exceeded,
        command_position_error_m=distribution(positions),command_orientation_error_rad=distribution(orientations),
        raw_position_error_m=distribution(measured_positions),raw_orientation_error_rad=distribution(measured_orientations),
        max_accepted_axis_error_m=max(axis_errors,default=0.),
        executable_goal_position_error_m=distribution(goal_positions),
        executable_goal_orientation_error_rad=distribution(goal_orientations),
        complete_tick_ms=distribution(timings),max_joint_speed_rad_s=maximum_speed,
        max_joint_acceleration_rad_s2=maximum_acceleration,max_joint_jump_rad=maximum_jump,
        max_command_feedback_gap_rad=maximum_feedback_gap,final_tracking=final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    package = Path(rospkg.RosPack().get_path('pika_piper'))
    description = Path(rospkg.RosPack().get_path('piper_description'))
    srdf = Path(rospkg.RosPack().get_path('piper_with_gripper_moveit'))/'config/piper_description.srdf'
    config_payload = (package/'config/pipeline.yaml').read_bytes()
    config = yaml.safe_load(config_payload)
    k = Kinematics(description/'urdf/piper_description.urdf', description,
                   config['tcp_xyzrpy'], srdf, posture_weight=config['ik_posture_weight'],
                   feedback_limits_deg=config.get('verified_feedback_limits_deg'),
                   feedback_tolerance=config.get('feedback_tolerance_rad',0.),
                   near_zero_start_tolerance_deg=config.get('near_zero_start_tolerance_deg',0.),
                   pose_acceptance=config.get('pose_acceptance','official'))
    t = make_teleop_solver(k,config)
    evidence = json.loads((package/'docs/ik_review_evidence_20261116.json').read_text())
    frames = []
    for event in evidence['bag']['ik_failure_events']:
        s = event['status']; previous = np.r_[s['ik']['seed_rad'], .03]
        target = np.array(s['ik']['target_matrix'])
        try:
            k.ik(target, previous, .03); old_accepted = True
        except ValueError:
            old_accepted = False
        t.warmup(previous)
        command = t.step(target, previous, .03, .02, np.zeros(6),dict(config, **s['motion_limits']))
        frames.append(dict(original_reason=s['reason'],old_accepted=old_accepted,new_tracking=t.last))
    print('Recorded failure frames:',len(frames),'valid commands:',sum(f['new_tracking']['command_valid'] for f in frames),flush=True)
    snapshot=json.loads((package/'docs/hardware_feedback_20261116.json').read_text())['sample_ranges']
    initial=np.r_[np.deg2rad([snapshot[str(i)]['last'] for i in range(6)]),snapshot['gripper_m']['last']]
    c,clock,receipt=controller(k,config,initial)
    clock.now+=.02
    c.hand_at=c.feedback_at=c.grip_at=c.quality_at=c.echo_at=clock.now
    c.tick(None)
    recovery=dict(mode=c.mode,reason=c.reason,raw_feedback=initial.tolist(),
        command=receipt.command.tolist() if receipt.command is not None else None,tracking=receipt.status['tracking'])
    if receipt.command is not None:k.validate_configuration(receipt.command)
    # Exact near-zero values reported by the user; startup timing depends on
    # the configured speed and acceleration, and every sent target is legal.
    zero_initial=np.r_[np.deg2rad([0,-.219,.290,0,0,0]),.03]
    zero,zero_clock,zero_receipt=controller(k,config,zero_initial)
    zero_frames=[]
    for index in range(3):
        zero_clock.now+=.02
        zero.hand_at=zero.feedback_at=zero.grip_at=zero.quality_at=zero.echo_at=zero_clock.now
        zero.tick(None)
        command=zero_receipt.command
        zero_frames.append(dict(frame=index+1,mode=zero.mode,reason=zero.reason,
            command=command.tolist() if command is not None else None,tracking=zero_receipt.status['tracking']))
        if command is not None:
            k.validate_configuration(command)
            zero.feedback=command.copy()
    # Repeat exactly the fixed-seed, collision-free FK sample distribution.
    rng = np.random.default_rng(20261116); fk = []
    lo, hi = k.model.lowerPositionLimit, k.model.upperPositionLimit
    for scale, count in [(.03,150),(.15,150),(.6,100)]:
        accepted = 0; precise = 0; done = 0
        while done < count:
            seed = rng.uniform(lo+.1,hi-.1); goal = seed+rng.normal(0,scale,6)
            if np.any(goal<lo+.001) or np.any(goal>hi-.001):continue
            if k.collision(seed,.03) or k.collision(goal,.03):continue
            try:
                k.ik(k.fk(goal),seed,.03);accepted+=1
                precise+=int(k.last_ik['position_error_m']<=.002 and
                             k.last_ik['orientation_error_rad']<=np.deg2rad(2.))
            except ValueError:pass
            done += 1
        fk.append(dict(scale_rad=scale,n=count,policy_ik_accepted=accepted,precision_2mm_2deg_count=precise))
    center = np.array([.1,.7,-1.2,.2,0.,.1])
    references = []
    for i in range(1000):
        phase = 2*np.pi*i/400
        references.append(center+np.array([.12*np.sin(phase),.08*np.sin(phase),
            .08*np.cos(phase),.3*np.sin(phase),.4*np.cos(phase),.3*np.cos(phase)]))
    boundary = []
    for angle in np.r_[np.linspace(1.16,1.42,250),np.full(150,1.42),np.linspace(1.42,.8,200),np.full(100,.8)]:
        q = center.copy();q[4] = angle;boundary.append(q)
    sequences = []
    for name, ref, lag, delay in [('smooth_and_j5_zero',references,0.,0),
            ('j5_boundary_out_and_back',boundary,0.,0),
            ('delayed_feedback',references[:400],.08,3)]:
        result = trajectory(k, config, name, ref, lag, delay)
        sequences.append(result)
        print(name,'frames',result['frames'],'faults',result['faults'],
              'tick P95 ms',result['complete_tick_ms']['p95'],flush=True)
    checks = dict(recorded_frames_no_invalid_command=all(f['new_tracking']['command_valid'] for f in frames),
        audited_current_feedback_recovery=recovery['mode']=='teleop' and recovery['command'] is not None and recovery['command'][4]<initial[4],
        reported_near_zero_startup=all(frame['mode']=='teleop' for frame in zero_frames) and
            any(frame['command'] is not None for frame in zero_frames) and
            next(frame['command'] for frame in zero_frames if frame['command'] is not None)[1:3]==[0.,0.],
        fk_regression=sum(row['policy_ik_accepted'] for row in fk)>=399 and
            sum(row['precision_2mm_2deg_count'] for row in fk)>=399,
        continuous_no_faults=all(not row['faults'] and row['frames']==row['requested_frames'] for row in sequences),
        joint_speed_respected=all(row['max_joint_speed_rad_s']<=effective_joint_speed(config)+1e-6 for row in sequences),
        joint_acceleration_respected=all(row['max_joint_acceleration_rad_s2']<=config['max_joint_acceleration']+1e-4 for row in sequences),
        qualified_goals_respect_tolerance=all(row['max_accepted_axis_error_m']<=.3 for row in sequences)
            if config.get('pose_acceptance','official')=='official' else
            all(row['executable_goal_position_error_m']['max']<=.002 and
                row['executable_goal_orientation_error_rad']['max']<=np.deg2rad(2) for row in sequences),
        boundary_can_retreat=sequences[1]['final_position_error_m']<.002 and sequences[1]['final_orientation_error_rad']<np.deg2rad(2))
    checks = {key: bool(value) for key, value in checks.items()}
    if config.get('teleop_policy','official')=='official':
        checks.pop('joint_speed_respected');checks.pop('joint_acceleration_respected')
    import pika_piper.teleop as teleop_module
    import pika_piper.kinematics as kinematics_module
    sources = [Path(teleop_module.__file__),Path(kinematics_module.__file__),
               Path(__file__),Path(__file__).with_name('control_node.py'),package/'config/pipeline.yaml']
    report = dict(hardware_started=False,scope=__doc__,urdf_sha256=k.urdf_hash,config=config,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        frame_comparison=frames,audited_start_recovery=recovery,
        near_zero_startup=dict(raw_feedback=zero_initial.tolist(),frames=zero_frames),
        fk_acceptance=fk,sequences=sequences,checks=checks,ok=all(checks.values()),
        timing_target_met=all(row['complete_tick_ms']['p95']<20 for row in sequences),
        limits=['No ROS transport or physical dynamics measured.',
                'Feedback band uses the audited hardware limits; gripper and commanded model ranges remain unchanged.',
                'Collision checking samples joint segments; it is not continuous collision detection.',
                'Official policy uses virtual 50 Hz IK and 200 Hz interpolation scheduling; no software speed/acceleration box is asserted.',
                'A qualified executable goal does not imply the raw hand target was reached.'])
    # The user can edit YAML during a long benchmark. Hash the configuration
    # actually loaded, rather than a newer file that was never exercised.
    report['source_sha256'][str(package/'config/pipeline.yaml')] = hashlib.sha256(config_payload).hexdigest()
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,timing_target_met=report['timing_target_met']),indent=2),flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
