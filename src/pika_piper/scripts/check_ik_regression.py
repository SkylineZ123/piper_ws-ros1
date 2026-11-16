#!/usr/bin/env python3
"""Compare frozen prior solvers and control IK offline; never opens CAN/ROS."""
import rospkg
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import casadi as ca
import pinocchio as pin
from scipy.optimize import least_squares
import yaml

from pika_piper.core import limit_step, transform
from pika_piper.kinematics import Kinematics


def legacy_solve(k, target, seed):
    """Frozen pre-migration objective/bounds/budget, for regression only."""
    def residual(q):
        return pin.log6(pin.SE3(k.fk(q)).inverse() * pin.SE3(target)).vector
    lower = k.model.lowerPositionLimit + 1e-7
    upper = k.model.upperPositionLimit - 1e-7
    result = least_squares(residual, np.clip(seed, lower, upper),
                           bounds=(lower, upper), max_nfev=50,
                           ftol=1e-6, xtol=1e-6, gtol=1e-6)
    error = residual(result.x)
    return {'q_rad': result.x.tolist(),
            'log_position_error_mm': float(np.linalg.norm(error[:3]) * 1000),
            'orientation_error_deg': float(np.rad2deg(np.linalg.norm(error[3:]))),
            'optimizer_success': bool(result.success)}


def original_official_solve(k, target, seed):
    """Frozen upstream zero-centered log6 objective, for diagnosis only."""
    opti=ca.Opti();q=opti.variable(6)
    error=k.error(q,target)
    opti.minimize(20*(ca.sumsqr(error[:3])+.01*ca.sumsqr(error[3:]))+.01*ca.sumsqr(q))
    opti.subject_to(opti.bounded(k.model.lowerPositionLimit,q,k.model.upperPositionLimit))
    opti.solver('ipopt',{'ipopt':{'print_level':0,'max_iter':50,'tol':1e-4,'sb':'yes'},'print_time':False})
    opti.set_initial(q,seed)
    try:
        solution=opti.solve_limited();candidate=np.asarray(solution.value(q)).reshape(6)
        residual=np.asarray(k.error(candidate,target)).reshape(6)
        return {'solver_success':bool(opti.stats()['success']),
                'solver_status':str(opti.stats()['return_status']),
                'log_position_error_m':float(np.linalg.norm(residual[:3])),
                'orientation_error_rad':float(np.linalg.norm(residual[3:])),
                'q_rad':candidate.tolist()}
    except RuntimeError:
        return {'solver_success':False,'solver_status':str(opti.stats().get('return_status','exception'))}


def evaluate(k, name, seed, target, width, config):
    report = {'name': name, 'seed_deg': np.rad2deg(seed).tolist(),
              'target': target.tolist(), 'gripper_m': width,
              'legacy_scipy': legacy_solve(k, target, seed),
              'original_official_ipopt': original_official_solve(k,target,seed)}
    try:
        k.validate_configuration(np.r_[seed, width], 'feedback')
        report['feedback_valid'] = True
    except ValueError as error:
        report.update(feedback_valid=False, feedback_reason=str(error))
    try:
        k.ik(target, seed, width)
        report['control_ik_accepted'] = True
    except ValueError as error:
        report.update(control_ik_accepted=False, control_ik_reason=str(error))
    report['control_ipopt'] = k.last_ik
    if k.last_ik and 'q_rad' in k.last_ik:
        candidate = np.r_[k.last_ik['q_rad'], width]
        limited = limit_step(np.r_[seed, width], candidate, 1/config['control_hz'],
                             config['max_joint_speed'], config['max_gripper_speed'])
        report['first_limited_command_deg'] = np.rad2deg(limited[:6]).tolist()
        try:
            k.validate_configuration(limited, 'first limited command')
            report['first_limited_command_valid'] = True
        except ValueError as error:
            report.update(first_limited_command_valid=False, first_command_reason=str(error))
    # A fast limiter can reach a geometrically legal point from invalid feedback.
    # Production send() checks the previous configuration before limiting, so a
    # legal candidate alone must never be reported as permission to send it.
    report['first_command_allowed'] = bool(report['feedback_valid'] and
        report['control_ik_accepted'] and report.get('first_limited_command_valid', False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from pika_piper.configuration import package_file
    parser.add_argument('--config', default=str(package_file('config/pipeline.yaml')))
    parser.add_argument('--output', help='Optional JSON report path')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    description = Path(rospkg.RosPack().get_path('piper_description'))
    k = Kinematics(description/'urdf/piper_description.urdf',
                   description, config['tcp_xyzrpy'],
                   Path(rospkg.RosPack().get_path('piper_with_gripper_moveit'))/'config/piper_description.srdf',
                   posture_weight=config.get('ik_posture_weight',1e-4))
    cases = []
    width = .0688
    observed = np.deg2rad([-.352, -2.017, 1.825, 0, 19.273, 3.437])
    anchor = k.fk(observed)
    cases.append(evaluate(k, 'observed_start_no_motion', observed, anchor, width, config))
    for axis in range(3):
        for sign in (-1, 1):
            target = anchor.copy()
            target[axis, 3] += sign * .001
            cases.append(evaluate(k, 'observed_base_%s_%+dmm' % ('xyz'[axis], sign),
                                  observed, target, width, config))
    for axis in range(3):
        for sign in (-1, 1):
            delta = np.zeros(6)
            delta[axis + 3] = np.deg2rad(sign)
            cases.append(evaluate(k, 'observed_tcp_%s_%+ddeg' % ('rpy'[axis], sign),
                                  observed, anchor @ transform(delta), width, config))
    boundary = np.array([-.1559102621, .1253146403, 0., .5464451355, .1773254520, -.5023755719])
    target = k.fk(boundary)
    target[2, 3] -= .005
    cases.append(evaluate(k, 'previous_boundary_z_minus_5mm', boundary, target, width, config))
    for name, seed in [('reachable_small_angles', np.array([.1,.2,-.2,.1,.1,.1])),
                       ('interior_regularization_bias', np.array([0.,.5,-.7,0.,.4,0.])),
                       ('live_orientation_fault_3p464deg', np.deg2rad([-.596,30.118,-23.212,-3.097,34.591,3.656]))]:
        cases.append(evaluate(k, name, seed, k.fk(seed), .03, config))
    moving_seed=np.array([.1,.5,-.7,.1,.4,.1])
    moving_goal=np.array([.2100167168,.5453617363,-.9765622503,
                          -.3549964657,1.0239564615,-1.3078570230])
    cases.append(evaluate(k,'reachable_moving_goal',moving_seed,k.fk(moving_goal),.03,config))
    checks = {
        'old_fault_reproduced': abs(cases[0]['legacy_scipy']['log_position_error_mm']-8.782962)<.01,
        'out_of_bounds_start_rejected': not cases[0]['feedback_valid'],
        'no_unsafe_first_command': all(not c['first_command_allowed'] for c in cases[:13]),
        'previous_local_solve_failure_resolved': cases[13]['control_ik_accepted'],
        'valid_small_angles_pass': cases[14]['control_ik_accepted'],
        'interior_stationary_pose_fixed': cases[15]['control_ik_accepted'],
        'user_orientation_fault_reproduced': bool(abs(np.rad2deg(cases[16]['original_official_ipopt']['orientation_error_rad'])-3.464)<.002),
        'user_orientation_fault_fixed': cases[16]['control_ik_accepted'] and cases[16]['control_ipopt']['orientation_error_rad']<1e-5,
        'reachable_moving_goal_tracks_accurately': cases[17]['control_ik_accepted'] and cases[17]['control_ipopt']['orientation_error_rad']<np.deg2rad(.2),
    }
    checks = {name: bool(value) for name,value in checks.items()}
    import pika_piper.kinematics
    code = Path(pika_piper.kinematics.__file__)
    report = {'hardware_started': False, 'python': sys.executable,
              'urdf_sha256': k.urdf_hash, 'implementation_sha256': hashlib.sha256(code.read_bytes()).hexdigest(),
              'kinematics': k.info, 'checks': checks, 'ok': all(checks.values()), 'cases': cases,
              'scope': 'Offline model comparison, not hardware tracking validation. '
                       'Raw official candidates are evaluated even when feedback preflight fails.'}
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(payload)
    else:
        print(payload, end='')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
