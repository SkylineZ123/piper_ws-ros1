#!/usr/bin/env python3
"""Compare software follow limits offline; no ROS master, CAN or motor commands."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rospkg
import yaml

from pika_piper.core import effective_joint_speed
from pika_piper.kinematics import Kinematics
from pika_piper.teleop import TeleopIK, pose_error


def run(k,cfg,goals,settle=False):
    initial=goals[0][0]
    previous=np.r_[initial,.03];velocity=np.zeros(6)
    solver=TeleopIK(k);solver.warmup(previous)
    positions=[];orientations=[];times=[];counts={};streak=0
    max_speed=max_acceleration=0.
    for _,target in goals:
        command=solver.step(target,previous,.03,.02,velocity,cfg)
        new_velocity=(command[:6]-previous[:6])/.02
        k.validate_configuration(command)
        max_speed=max(max_speed,float(np.max(abs(new_velocity))))
        max_acceleration=max(max_acceleration,float(np.max(abs(new_velocity-velocity)))/.02)
        position,orientation=pose_error(k.fk(command),target)
        positions.append(position);orientations.append(orientation);times.append(solver.last['solve_ms'])
        state=solver.last['state'];counts[state]=counts.get(state,0)+1
        streak=streak+1 if position<=.002 and orientation<=np.deg2rad(2.) else 0
        previous,velocity=command,new_velocity
        if settle and streak>=4:break
    return dict(frames=len(positions),settled=streak>=4,
        time_to_goal_s=len(positions)*.02 if settle and streak>=4 else None,
        position_error_mm_p95=float(np.percentile(positions,95)*1000),
        orientation_error_deg_p95=float(np.rad2deg(np.percentile(orientations,95))),
        final_position_error_mm=positions[-1]*1000,final_orientation_error_deg=float(np.rad2deg(orientations[-1])),
        compute_ms_p95=float(np.percentile(times,95)),states=counts,
        max_joint_speed_rad_s=max_speed,max_joint_acceleration_rad_s2=max_acceleration,
        requested_joint_speed_rad_s=cfg['max_joint_speed'],effective_joint_speed_rad_s=effective_joint_speed(cfg))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();rp=rospkg.RosPack();package=Path(rp.get_path('pika_piper'))
    before_bytes=(package/'docs/follow_speed_before_20261116.yaml').read_bytes()
    after_bytes=(package/'config/pipeline.yaml').read_bytes()
    configs={'before':yaml.safe_load(before_bytes),'after':yaml.safe_load(after_bytes)}
    description=Path(rp.get_path('piper_description'))
    k=Kinematics(description/'urdf/piper_description.urdf',description,configs['after']['tcp_xyzrpy'],
        Path(rp.get_path('piper_with_gripper_moveit'))/'config/piper_description.srdf')
    initial=np.array([.1,.7,-1.2,.2,.5,.1]);goal=initial+np.array([.6,0,0,0,.2,.4])
    k.validate_configuration(np.r_[initial,.03]);k.validate_configuration(np.r_[goal,.03])
    step=[(initial,k.fk(goal))]*500
    smooth=[]
    for i in range(600):
        phase=2*np.pi*i/200
        q=np.array([.1,.7,-1.2,.2,.4,.1])+np.array([.1*np.sin(phase),.06*np.sin(phase),
            -.06*np.cos(phase),.2*np.sin(phase),.4*np.cos(phase),.5*np.sin(phase)])
        k.validate_configuration(np.r_[q,.03]);smooth.append((q,k.fk(q)))
    results={}
    for name,cfg in configs.items():
        results[name]=dict(step=run(k,cfg,step,True),smooth=run(k,cfg,smooth))
        print(name,json.dumps(results[name]),flush=True)
    checks={'all_step_targets_reached':all(row['step']['settled'] for row in results.values()),
        'new_step_faster':results['after']['step']['time_to_goal_s']<results['before']['step']['time_to_goal_s'],
        'new_smooth_position_error_lower':results['after']['smooth']['position_error_mm_p95']<results['before']['smooth']['position_error_mm_p95'],
        'new_smooth_orientation_error_lower':results['after']['smooth']['orientation_error_deg_p95']<results['before']['smooth']['orientation_error_deg_p95'],
        'speed_constraints_respected':all(row[task]['max_joint_speed_rad_s']<=effective_joint_speed(configs[name])+1e-6
            for name,row in results.items() for task in ('step','smooth')),
        'acceleration_constraints_respected':all(row[task]['max_joint_acceleration_rad_s2']<=configs[name]['max_joint_acceleration']+1e-4
            for name,row in results.items() for task in ('step','smooth'))}
    report=dict(hardware_started=False,scope='Software command tracking in the FK model; not physical MIT tracking performance.',
        configs=configs,config_sha256={'before':hashlib.sha256(before_bytes).hexdigest(),'after':hashlib.sha256(after_bytes).hexdigest()},
        urdf_sha256=k.urdf_hash,results=results,checks=checks,ok=all(checks.values()))
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
    return 0 if report['ok'] else 1


if __name__=='__main__':raise SystemExit(main())
