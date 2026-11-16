"""Bounded, continuous teleoperation; strict trajectory IK remains separate.

Every candidate is constrained around the last issued command. A distant IK
branch is never a shortcut around a wrist limit. Saturation preserves the hand
anchor so the operator can retreat without restarting teleoperation.
"""
import time

import casadi as ca
import numpy as np
import pinocchio as pin

from .core import limit_pose_step, limit_step, effective_joint_speed


POSITION_TOLERANCE = .002
ORIENTATION_TOLERANCE = np.deg2rad(2.)


def pose_error(actual, target):
    error = pin.log6(pin.SE3(actual).inverse() * pin.SE3(target)).vector
    return (float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
            float(np.linalg.norm(error[3:])))


def motion_bounds(model, previous, velocity, dt, speed, acceleration, margin):
    """Intersect hardware, stopping-distance, speed and acceleration bounds.

    A state already inside the soft margin may hold or retreat. When incoming
    velocity cannot respect that margin, braking uses the remaining hardware
    envelope. An infeasible acceleration interval is an explicit fault.
    """
    q = np.asarray(previous)[:6]
    velocity = np.asarray(velocity, dtype=float)
    lower, upper = model.lowerPositionLimit, model.upperPositionLimit
    soft_lower = np.minimum(q, lower + margin)
    soft_upper = np.maximum(q, upper - margin)
    # Discrete stopping distance includes the next integration interval.
    def stopping_speed(distance):
        distance = np.maximum(distance, 0.)
        # Invert dt * sum(v - k*a*dt, k=0..ceil(v/(a*dt))-1).
        # A continuous sqrt(2*a*d) approximation can make the next discrete
        # acceleration interval infeasible even at a constant control period.
        steps = np.maximum(1., np.ceil((np.sqrt(1 + 8 * distance /
                                               (acceleration * dt**2)) - 1) / 2))
        return distance / (steps * dt) + acceleration * dt * (steps - 1) / 2
    v_lower = np.maximum(-speed, -stopping_speed(q - soft_lower))
    v_upper = np.minimum(speed, stopping_speed(soft_upper - q))
    a_lower = velocity - acceleration * dt
    a_upper = velocity + acceleration * dt
    brake_lower = v_lower > a_upper + 1e-9
    brake_upper = v_upper < a_lower - 1e-9
    margin_braking = bool(np.any(brake_lower) or np.any(brake_upper))
    if margin_braking:
        v_lower = np.maximum(-speed, -stopping_speed(q - lower))
        v_upper = np.minimum(speed, stopping_speed(upper - q))
        # If timing jitter invalidated the soft stopping envelope, use the
        # hardware reserve only to decelerate, never to accelerate farther out.
        v_lower = np.where(brake_lower, np.maximum(v_lower, a_upper), v_lower)
        v_upper = np.where(brake_upper, np.minimum(v_upper, a_lower), v_upper)
    v_lower = np.maximum(v_lower, a_lower)
    v_upper = np.minimum(v_upper, a_upper)
    if np.any(v_lower > v_upper + 1e-9):
        raise ValueError('Teleop cannot brake within joint limits; pause and inspect feedback')
    v_upper = np.maximum(v_upper, v_lower)
    q_lower = np.maximum(lower, q + dt * v_lower)
    q_upper = np.minimum(upper, q + dt * v_upper)
    if np.any(q_lower > q_upper + 1e-10):
        raise ValueError('Feedback recovery cannot reach model limits within this command interval')
    return q_lower, np.maximum(q_lower, q_upper), margin_braking


class TeleopIK:
    def __init__(self, kinematics):
        self.k = kinematics
        self.last = None
        self.warmup_status = None
        self.opti = ca.Opti()
        self.q = self.opti.variable(6)
        self.target = self.opti.parameter(4, 4)
        self.seed = self.opti.parameter(6)
        self.lower = self.opti.parameter(6)
        self.upper = self.opti.parameter(6)
        actual = self.k.pose(self.q)
        position = ca.sumsqr(actual[:3, 3] - self.target[:3, 3])
        rotation = ca.sumsqr(actual[:3, :3] - self.target[:3, :3]) / 2
        # At the acceptance tolerance each normalized pose cost equals one.
        # Matrix differences retain finite derivatives at a stationary hold.
        self.opti.minimize(position / POSITION_TOLERANCE**2 +
                           rotation / (4 * np.sin(ORIENTATION_TOLERANCE / 2)**2) +
                           self.k.posture_weight * ca.sumsqr(self.q - self.seed))
        self.opti.subject_to(self.opti.bounded(self.lower, self.q, self.upper))
        self.opti.solver('ipopt', {
            'ipopt': {'print_level': 0, 'max_iter': 40, 'tol': 1e-7,
                      'bound_relax_factor': 0., 'max_cpu_time': .010, 'sb': 'yes'},
            'print_time': False})

    def warmup(self, previous):
        # Load IPOPT at an interior FK target. A zero-cost target exactly on a
        # joint bound can exhaust the barrier solver's startup CPU budget.
        # This numerical warmup never changes feedback or publishes a command.
        lower, upper = self.k.model.lowerPositionLimit, self.k.model.upperPositionLimit
        warm = np.clip(np.asarray(previous)[:6], lower + .01, upper - .01)
        q, status = self._solve(self.k.fk(warm), warm, lower, upper)
        self.warmup_status = status
        # This is cache initialization, not a candidate for sending. An
        # initialization time/iteration limit must not veto an otherwise valid
        # handover. Every actual motion still passes its independent checks.
        if q is None and status not in ('Maximum_CpuTime_Exceeded', 'Maximum_Iterations_Exceeded'):
            raise ValueError('Teleop IK warmup failed: ' + status)

    def startup(self, previous, target, elapsed, config, raw_target):
        """Wait for an admissible first command interval, then move inward.

        Nothing is published while waiting. Even the first transmitted target
        stays inside model bounds; intermediate out-of-range goals are forbidden.
        Acceleration here is the same discrete command contract as normal steps,
        not a claim about the motor's continuous acceleration.
        """
        started = time.perf_counter()
        self.k.validate_teleop_start(previous)
        self.k.validate_configuration(target, 'teleop startup command')
        expected = self.k.start_target(previous)
        if not np.allclose(target, expected, atol=1e-12, rtol=0.):
            raise ValueError('Startup may only move to the nearest legal boundary')
        speed = effective_joint_speed(config)
        acceleration = config.get('max_joint_acceleration', 10.)
        delta = target[:6] - previous[:6]
        distance = float(np.max(np.abs(delta)))
        required = max(distance / speed, np.sqrt(distance / acceleration))
        ready = bool(elapsed + 1e-12 >= required)
        # Check the measured-to-legal segment with the startup allowance, while
        # validating the destination with the normal command bounds above.
        for fraction in np.linspace(0., 1., max(2, int(np.ceil(distance / .001)) + 1)):
            self.k.validate_teleop_start(previous + fraction * (target - previous),
                                         'teleop startup segment')
        actual = self.k.fk(target if ready else previous)
        raw_position, raw_orientation = pose_error(actual, raw_target)
        velocity = delta / max(elapsed, 1e-12) if ready else np.zeros(6)
        self.last = {
            'backend': 'bounded_inward_startup', 'state': 'recovering',
            'warmup_status': self.warmup_status,
            'reason': 'initial_feedback_outside_command_range',
            'accepted': ready, 'command_valid': ready, 'feedback_recovery': True,
            'recovery_complete': ready, 'startup_elapsed_s': float(elapsed),
            'startup_required_s': float(required), 'startup_remaining_s': max(0., required - elapsed),
            'effective_joint_speed_rad_s': speed, 'attempts': 0, 'trials': [],
            'tcp_limited': False, 'step_fraction': 0.,
            'position_error_m': 0. if ready else None,
            'orientation_error_rad': 0. if ready else None,
            'raw_position_error_m': raw_position, 'raw_orientation_error_rad': raw_orientation,
            'target_matrix': actual.tolist() if ready else None, 'raw_target_matrix': raw_target.tolist(),
            'seed_rad': previous[:6].tolist(), 'q_rad': target[:6].tolist() if ready else None,
            'joint_velocity_rad_s': velocity.tolist(),
            'joint_acceleration_rad_s2': (velocity / max(elapsed, 1e-12)).tolist(),
            'solve_ms': (time.perf_counter() - started) * 1000,
            'budget_exceeded': False,
        }
        self.k.last_ik = self.last
        return target.copy() if ready else None

    def _solve(self, target, previous, lower, upper):
        self.opti.set_value(self.target, target)
        self.opti.set_value(self.seed, previous[:6])
        self.opti.set_value(self.lower, lower)
        self.opti.set_value(self.upper, upper)
        self.opti.set_initial(self.q, np.clip(previous[:6], lower, upper))
        try:
            solution = self.opti.solve_limited()
        except RuntimeError:
            return None, str(self.opti.stats().get('return_status', 'exception'))
        status = self.opti.stats()
        if not status['success']:
            return None, str(status['return_status'])
        q = np.asarray(solution.value(self.q)).reshape(6)
        clipped = np.clip(q, lower, upper)
        if not np.isfinite(q).all() or np.max(np.abs(q - clipped)) > 1e-7:
            return None, 'invalid_motion_bounds'
        return clipped, str(status['return_status'])

    def _validate_path(self, previous, command):
        # Sample the short command segment as well as its endpoint. This is
        # discrete self-collision validation, not continuous collision proof.
        count = max(1, int(np.ceil(np.max(np.abs(command[:6] - previous[:6])) / .01)))
        self.k.validate_configuration(command, 'teleop command endpoint')
        recovering = bool(np.any(previous[:6] < self.k.model.lowerPositionLimit) or
                          np.any(previous[:6] > self.k.model.upperPositionLimit))
        for fraction in np.linspace(0., 1., count + 1)[1:]:
            validate = self.k.validate_feedback if recovering else self.k.validate_configuration
            validate(previous + fraction * (command - previous), 'teleop command segment')

    def step(self, target, previous, width, dt, velocity, config):
        started = time.perf_counter()
        self.last = {'accepted': False, 'command_valid': False,
                     'state': 'fault', 'reason': 'step_did_not_complete'}
        self.k.last_ik = self.last
        target = np.asarray(target, dtype=float)
        previous = np.asarray(previous, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        # Invalid input is never converted into a recoverable saturation.
        if (target.shape != (4, 4) or not np.isfinite(target).all() or
                not np.allclose(target[3], [0, 0, 0, 1], atol=1e-8) or
                not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6) or
                not np.isclose(np.linalg.det(target[:3, :3]), 1., atol=1e-6)):
            raise ValueError('Teleop target must be a finite rigid transform')
        if not np.isfinite(dt) or dt <= 0 or velocity.shape != (6,) or not np.isfinite(velocity).all():
            raise ValueError('Teleop requires positive dt and six finite command velocities')
        if not np.isfinite(width) or not 0 <= width <= .07:
            raise ValueError('Teleop gripper width outside [0, 0.07] m')
        self.k.validate_feedback(previous, 'teleop previous command')
        speed = effective_joint_speed(config)
        acceleration = config.get('max_joint_acceleration', 10.)
        margin = config.get('joint_limit_margin', .035)
        budget = config.get('teleop_ik_budget_ms', 16.)
        attempts_limit = config.get('teleop_ik_max_attempts', 2)
        lower, upper, margin_braking = motion_bounds(
            self.k.model, previous, velocity, dt, speed, acceleration, margin)
        origin = self.k.fk(previous)
        stepped, tcp_limited = limit_pose_step(origin, target, dt,
            config.get('max_tcp_speed', .15), config.get('max_tcp_angular_speed', 1.))
        command = previous.copy()
        command[6] = limit_step(previous, np.r_[previous[:6], width], dt,
                               speed, config['max_gripper_speed'])[6]
        # Default to a bounded braking command when no pose candidate qualifies.
        brake_velocity = velocity - np.clip(velocity, -acceleration * dt, acceleration * dt)
        command[:6] = np.clip(previous[:6] + brake_velocity * dt, lower, upper)
        trials = []
        accepted_target = None
        fraction = 1.
        solver_status = 'not_run'
        for attempt in range(attempts_limit):
            if (time.perf_counter() - started) * 1000 >= budget:
                break
            # Scale the same Cartesian segment; never reset the hand anchor.
            candidate_target = origin.copy()
            candidate_target[:3, 3] = origin[:3, 3] + fraction * (stepped[:3, 3] - origin[:3, 3])
            candidate_target[:3, :3] = origin[:3, :3] @ pin.exp3(
                pin.log3(origin[:3, :3].T @ stepped[:3, :3]) * fraction)
            q, solver_status = self._solve(candidate_target, previous, lower, upper)
            trial = {'fraction': fraction, 'solver_status': solver_status}
            reduction = .5
            if q is not None:
                position, orientation = pose_error(self.k.fk(q), candidate_target)
                trial.update(position_error_m=position, orientation_error_rad=orientation)
                if position <= POSITION_TOLERANCE and orientation <= ORIENTATION_TOLERANCE:
                    command[:6] = q
                    accepted_target = candidate_target
                else:
                    # A large residual needs more than a blind half step,
                    # particularly when accelerating from rest. Keep reserve
                    # under both tolerances without increasing the retry count.
                    reduction = min(.5, .75 / max(position / POSITION_TOLERANCE,
                                                  orientation / ORIENTATION_TOLERANCE))
            trials.append(trial)
            if accepted_target is not None:
                break
            fraction *= reduction
        self._validate_path(previous, command)
        actual = self.k.fk(command)
        raw_position, raw_orientation = pose_error(actual, target)
        goal_position, goal_orientation = (pose_error(actual, accepted_target)
            if accepted_target is not None else (None, None))
        margins = np.minimum(command[:6] - self.k.model.lowerPositionLimit,
                             self.k.model.upperPositionLimit - command[:6])
        active = ['joint%d' % (i + 1) for i in range(6) if margins[i] <= margin + 1e-5]
        raw_reached = raw_position <= POSITION_TOLERANCE and raw_orientation <= ORIENTATION_TOLERANCE
        if accepted_target is None:
            state, reason = ('braking' if np.linalg.norm(command[:6] - previous[:6]) > 1e-7
                             else 'saturated'), 'no_qualified_local_step'
        elif not raw_reached and (fraction < 1. or active):
            state, reason = 'saturated', 'local_motion_envelope'
        else:
            state, reason = ('tracking' if raw_reached else 'limited'), ''
        new_velocity = (command[:6] - previous[:6]) / dt
        self.last = {
            'backend': 'bounded_pinocchio_casadi_ipopt',
            'warmup_status': self.warmup_status,
            'objective_error': 'tolerance_normalized_cartesian_chordal',
            'accepted': accepted_target is not None, 'command_valid': True,
            'state': state, 'reason': reason, 'solver_status': solver_status,
            'attempts': len(trials), 'trials': trials,
            'step_fraction': fraction if accepted_target is not None else 0.,
            'tcp_limited': bool(tcp_limited), 'margin_braking': margin_braking,
            'effective_joint_speed_rad_s': speed,
            'feedback_recovery': bool(np.any(previous[:6] < self.k.model.lowerPositionLimit) or
                                      np.any(previous[:6] > self.k.model.upperPositionLimit)),
            'active_soft_limits': active, 'joint_margin_rad': margins.tolist(),
            'position_error_m': goal_position, 'orientation_error_rad': goal_orientation,
            'raw_position_error_m': raw_position, 'raw_orientation_error_rad': raw_orientation,
            'target_matrix': accepted_target.tolist() if accepted_target is not None else None,
            'raw_target_matrix': target.tolist(), 'seed_rad': previous[:6].tolist(),
            'q_rad': command[:6].tolist(), 'joint_velocity_rad_s': new_velocity.tolist(),
            'joint_acceleration_rad_s2': ((new_velocity - velocity) / dt).tolist(),
            'solve_ms': (time.perf_counter() - started) * 1000, 'budget_ms': budget,
        }
        self.last['budget_exceeded'] = self.last['solve_ms'] > budget
        self.k.last_ik = self.last
        return command
