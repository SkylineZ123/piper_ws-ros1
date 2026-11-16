#!/usr/bin/env python3
"""Send preset poses through the production controller into AgileX's Meshcat env.

Uses the original test_piper_sim.py Arm_IK environment, with its URDF and tool.
ROS IO/clocks are simulated. Motor lag/speed/delay are synthetic, not upstream
physics. No serial, locator, SDK or CAN driver is started.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import meshcat.animation
import numpy as np
import pinocchio as pin
import rospkg
import yaml

from pika_piper.core import transform, vector
from pika_piper.kinematics import Kinematics
from pika_piper.teleop import pose_error


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def load_environment(root):
    script=root/'piper/pika_remote_piper/scripts/test_piper_sim.py'
    description=root/'piper/piper_ros/piper_description'
    if not script.is_file() or not (description/'urdf/piper_description.urdf').is_file():
        raise ValueError('--upstream-root must contain a PikaAnyArm checkout, including Piper models')
    upstream=load_module('agilex_original_remote_sim',script)
    original_get_path=rospkg.RosPack.get_path
    original_build=pin.RobotWrapper.BuildFromURDF
    def get_path(self,name):
        return str(description) if name=='piper_description' else original_get_path(self,name)
    def build(path,**kwargs):
        # Upstream passes the package itself as a search directory. Provide
        # its parent for package:// resolution in this Pinocchio version.
        return original_build(path,package_dirs=[str(description.parent)])
    with patch.object(rospkg.RosPack,'get_path',get_path), \
            patch.object(pin.RobotWrapper,'BuildFromURDF',build), \
            patch('webbrowser.open',lambda *a,**kw:False):
        environment=upstream.Arm_IK()
    return environment,description,script


def stats(values):
    return {name:float(value) for name,value in zip(
        ('p50','p95','p99','max'),[*np.percentile(values,[50,95,99]),max(values)])} if len(values) else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root',type=Path,default=Path(os.environ.get('PIKA_ANYARM_PATH','/tmp/pika-anyarm-review-20261116')))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--html',type=Path)
    parser.add_argument('--case-seconds',type=float,default=3.)
    parser.add_argument('--no-render',action='store_true',help='Evaluate the same env model without recording an animation')
    args=parser.parse_args([x for x in sys.argv[1:] if ':=' not in x])
    if not 1 <= args.case_seconds <= 10:parser.error('--case-seconds must be in [1,10]')
    package=Path(rospkg.RosPack().get_path('pika_piper'))
    config=yaml.safe_load((package/'config/pipeline.yaml').read_text())
    env,description,source=load_environment(args.upstream_root.resolve())
    viewer=env.vis.viewer
    try:
        # Use the exact upstream environment geometry/tool for this experiment.
        # The real-arm pipeline YAML/TCP is not modified.
        base=env.reduced_robot.model.frames[env.reduced_robot.model.getFrameId('gripper_base')]
        tcp=base.placement.inverse()*pin.SE3(env.last_matrix)
        cfg=dict(config,tcp_xyzrpy=vector(tcp.homogeneous,0.)[:6].tolist())
        k=Kinematics(description/'urdf/piper_description.urdf',description,cfg['tcp_xyzrpy'],
            Path(rospkg.RosPack().get_path('piper_with_gripper_moveit'))/'config/piper_description.srdf',
            posture_weight=cfg['ik_posture_weight'],pose_acceptance=cfg['pose_acceptance'],
            feedback_limits_deg=cfg['verified_feedback_limits_deg'],feedback_tolerance=cfg['feedback_tolerance_rad'])
        bench=load_module('remote_sim_controller_harness',Path(__file__).with_name('benchmark_teleop.py'))
        model=env.reduced_robot.model;data=model.createData();frame=model.getFrameId('ee')
        def env_pose(q):
            pin.framesForwardKinematics(model,data,np.asarray(q)[:6])
            return data.oMf[frame].homogeneous.copy()
        initial=np.array([.1,.7,-1.2,.2,.5,.1,.03])
        k.validate_configuration(initial)
        witnesses=[('hold',initial.copy()),('arm_position_step',initial+np.array([.08,.06,-.05,0,0,0,0])),
            ('wrist_roll',initial+np.array([0,0,0,.35,0,0,0])),
            ('large_base_rotation',initial+np.array([.9,0,0,0,0,0,0]))]
        for name,index,value in [('j5_near_limit',4,1.18),('j6_positive_100deg',5,np.deg2rad(100.)),
                                 ('j6_negative_100deg',5,np.deg2rad(-100.))]:
            goal=initial.copy();goal[index]=value;witnesses.append((name,goal))
        cases=[]
        for name,goal in witnesses:
            k.validate_configuration(goal,'preset reachable witness')
            target=env_pose(goal)
            np.testing.assert_allclose(k.fk(goal),target,atol=1e-10)
            cases.append((name,target,goal))
        cases.append(('tcp_rotation_request',env_pose(initial)@transform([0,0,0,.25,0,0]),None))
        unreachable=env_pose(initial);unreachable[0,3]+=2.
        cases.append(('unreachable_translation',unreachable,None))
        animation=meshcat.animation.Animation(default_framerate=25)
        animation_frame=0;rows=[];all_valid=True
        baseline=[]
        for name,target,witness in cases:
            # Original env returns False unconditionally in ik_fun(). Inspect
            # returned joints and residuals rather than treating that flag as success.
            original_q,_,flag=env.ik_fun(target,initial[6])
            baseline_row={'name':name,'returned_flag':bool(flag),'returned_joint_solution':original_q is not None}
            if original_q is not None:
                p,r=pose_error(env_pose(original_q),target)
                baseline_row.update(position_error_mm=p*1000,orientation_error_deg=float(np.rad2deg(r)))
            baseline.append(baseline_row)
            c,clock,receipt=bench.controller(k,cfg,initial)
            actual=initial.copy();pending=[initial.copy() for _ in range(12)]
            positions=[];orientations=[];command_positions=[];command_orientations=[];compute=[]
            states=Counter();resets=0;published=[];settled_at=None
            original_publish=c.cmd.publish
            def publish(msg):
                published.append(np.asarray(msg.position).copy());original_publish(msg)
            c.cmd.publish=publish
            for i in range(int(args.case_seconds/.005)):
                clock.now+=.005
                c.hand_at=c.feedback_at=c.grip_at=c.quality_at=c.echo_at=clock.now
                c.feedback=actual.copy()
                if i%4==0:
                    c.hand=np.linalg.inv(c.robot_anchor)@target
                    started=time.perf_counter();c.tick(None);compute.append((time.perf_counter()-started)*1000)
                    if c.tracking:
                        states[c.tracking['state']]+=1
                        resets+=int(c.teleop_solver.last.get('seed_reset',False))
                else:c.command_tick(None)
                if c.mode=='fault':break
                pending.append(c.sent.copy());delayed=pending.pop(0)
                delta=(delayed-actual)*min(1.,.005/.08)
                delta[:6]=np.clip(delta[:6],-.005*cfg['firmware_joint_speed_limit'],.005*cfg['firmware_joint_speed_limit'])
                actual+=delta
                if i%8==0 and not args.no_render:
                    with animation.at_frame(viewer,animation_frame) as frame_viewer:
                        env.vis.viewer=frame_viewer
                        try:env.vis.display(actual[:6])
                        finally:env.vis.viewer=viewer
                    animation_frame+=1
                p,r=pose_error(env_pose(actual),target);positions.append(p);orientations.append(r)
                cp,cr=pose_error(env_pose(c.sent),target);command_positions.append(cp);command_orientations.append(cr)
                if settled_at is None and p<=.002 and r<=np.deg2rad(2.):settled_at=(i+1)*.005
            for command in published:
                try:k.validate_configuration(command)
                except ValueError:all_valid=False
            finalp,finalr=pose_error(env_pose(actual),target)
            row=dict(name=name,target_matrix=target.tolist(),target_xyzrpy=vector(target,initial[6])[:6].tolist(),
                reachable_witness_rad=witness.tolist() if witness is not None else None,
                mode=c.mode,reason=c.reason,published_commands=len(published),frames=i+1,seed_resets=resets,
                final_feedback_q_rad=actual.tolist(),final_command_q_rad=c.sent.tolist(),tracking_states=dict(states),
                final_feedback_position_error_mm=finalp*1000,final_feedback_orientation_error_deg=float(np.rad2deg(finalr)),
                final_tracking=c.tracking,time_to_2mm_2deg_s=settled_at,
                measured_position_error_mm=stats(np.array(positions)*1000),
                measured_orientation_error_deg=stats(np.rad2deg(orientations)),
                command_position_error_mm=stats(np.array(command_positions)*1000),
                command_orientation_error_deg=stats(np.rad2deg(command_orientations)),complete_tick_ms=stats(compute),
                response=dict(time_s=(np.arange(len(positions))*.005+.005)[::4].tolist(),
                    measured_position_error_mm=(np.array(positions)*1000)[::4].tolist(),
                    measured_orientation_error_deg=np.rad2deg(orientations)[::4].tolist(),
                    command_orientation_error_deg=np.rad2deg(command_orientations)[::4].tolist()))
            rows.append(row)
            print(name,'mode',c.mode,'final mm/deg',round(finalp*1000,4),round(float(np.rad2deg(finalr)),4),flush=True)
        if args.html and not args.no_render:
            viewer.set_animation(animation,play=True,repetitions=1)
            env.vis.display(initial[:6])
            args.html.parent.mkdir(parents=True,exist_ok=True)
            args.html.write_text(viewer.static_html())
        checks=dict(env_fk_matches_solver_tool=True,j6_120deg=abs(np.rad2deg(k.model.upperPositionLimit[5])-120.)<.001,
            all_published_commands_valid=all_valid,
            unreachable_target_has_no_command=rows[-1]['published_commands']==0,
            all_witness_targets_have_commands=all(r['published_commands']>0 for r in rows if r['reachable_witness_rad'] is not None),
            all_witness_targets_no_fault=all(r['mode']!='fault' for r in rows if r['reachable_witness_rad'] is not None))
        checks={key:bool(value) for key,value in checks.items()}
        feedback_regressions={}
        for scenario in ('stationary_feedback','invalid_feedback'):
            c,clock,receipt=bench.controller(k,cfg,initial)
            goal=initial.copy();goal[0]+=.3;target=env_pose(goal)
            states=[]
            for index in range(100):
                clock.now+=.02;c.hand=np.linalg.inv(c.robot_anchor)@target
                c.hand_at=c.feedback_at=c.grip_at=c.quality_at=c.echo_at=clock.now
                c.feedback=initial.copy()
                if scenario=='invalid_feedback':c.feedback[1]=-.04
                c.tick(None)
                if c.tracking:states.append(c.tracking['state'])
                if c.mode=='fault':break
            feedback_regressions[scenario]=dict(mode=c.mode,reason=c.reason,frames=index+1,
                states=states,tracking=c.tracking,command_published=receipt.command is not None,
                pending_target_cleared=c.command_schedule.target is None)
        checks['stationary_feedback_detected']=feedback_regressions['stationary_feedback']['mode']=='fault' and 'execution stalled' in feedback_regressions['stationary_feedback']['reason']
        checks['runtime_invalid_feedback_rejected']=feedback_regressions['invalid_feedback']['mode']=='fault' and not feedback_regressions['invalid_feedback']['command_published']
        report=dict(hardware_started=False,upstream_env_source=str(source),
            upstream_env_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),urdf_sha256=k.urdf_hash,
            test_tcp_xyzrpy=cfg['tcp_xyzrpy'],real_arm_config_unchanged_tcp=config['tcp_xyzrpy'],
            policy=cfg['teleop_policy'],kinematics=k.info,
            feedback_model=dict(time_constant_s=.08,transport_delay_s=.06,joint_speed_cap_rad_s=cfg['firmware_joint_speed_limit']),
            scope='Original AgileX Meshcat env/model, production controller, synthetic feedback and clock; no motor physics or ROS/CAN transport.',
            upstream_original_solver=baseline,preset_results=rows,feedback_regressions=feedback_regressions,checks=checks,ok=all(checks.values()),
            all_witness_targets_settle_2mm_2deg=all(r['final_feedback_position_error_mm']<=2. and
                r['final_feedback_orientation_error_deg']<=2. for r in rows if r['reachable_witness_rad'] is not None),
            html=str(args.html) if args.html else None,
            source_sha256={str(p.resolve().relative_to(package.resolve())):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [package/'scripts/control_node.py',package/'src/pika_piper/official.py',package/'src/pika_piper/tracking.py',
                 package/'src/pika_piper/kinematics.py',package/'config/pipeline.yaml',Path(__file__)]})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(checks,indent=2),flush=True)
        return 0 if report['ok'] else 1
    finally:
        # meshcat 0.3.2 exposes Visualizer.close(), but its ViewerWindow has
        # no close method. Explicitly close only this run's socket/server.
        viewer.window.zmq_socket.close(linger=0)
        process=viewer.window.server_proc
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5.)


if __name__=='__main__':raise SystemExit(main())
