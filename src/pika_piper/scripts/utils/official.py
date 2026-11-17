"""Upstream seed history and 50/200 Hz joint publication, without GUI or CAN.

Reference: PikaAnyArm d2b8b84, forward_inverse_kinematics.py / piper_IK.py.
The calibrated TCP, stable pose objective and SRDF validation stay local.
"""

import time

import numpy as np

from .teleop import TeleopIK, pose_error


class OfficialInterpolator:
    """Nonblocking equivalent of upstream's blocking interpolation publisher.

    A selected segment completes before consuming the latest cached target.
    Call sample() at 200 Hz; missed calls never cause a burst of old points.
    """

    rate_hz = 200.0
    idle_hz = 50.0
    threshold = np.deg2rad(30.0)
    angle_step = np.deg2rad(1.0)

    def __init__(self, timeout=1.0):
        self.timeout = timeout
        self.target = self.segment = None
        self.target_at = 0.0
        self.next_at = 0.0
        self.index = self.steps = 0

    def offer(self, command, now):
        command = np.asarray(command, dtype=float)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError('Official interpolation requires seven finite values')
        self.target = command.copy()
        self.target_at = now

    def cancel(self):
        self.target = self.segment = None
        self.index = self.steps = 0

    def sample(self, previous, now):
        if self.target is None:
            return None
        if not 0 <= now - self.target_at <= self.timeout:
            self.cancel()
            return None
        if now + 1e-10 < self.next_at:
            return None
        previous = np.asarray(previous, dtype=float)
        if self.segment is None:
            difference = float(np.max(np.abs(self.target[:6] - previous[:6])))
            if difference <= self.threshold:
                self.next_at = now + 1.0 / self.idle_hz
                return self.target.copy()
            self.steps = int(difference / self.angle_step)
            self.index = 0
            self.segment = (previous.copy(), self.target.copy())
        start, end = self.segment
        self.index += 1
        command = end.copy()
        command[:6] = start[:6] + (end[:6] - start[:6]) * self.index / self.steps
        self.next_at = now + 1.0 / self.rate_hz
        if self.index >= self.steps:
            self.segment = None
            self.next_at = now + 1.0 / self.idle_hz
        return command


class OfficialTeleopIK(TeleopIK):
    policy = 'official'

    def __init__(self, kinematics):
        self.k = kinematics
        self.last = None
        self.warmup_status = None
        self.initial = np.zeros(6)
        self.history = np.zeros(6)

    def warmup(self, previous):
        self.k.solve_ik(self.k.fk(previous), previous, previous[6])
        self.warmup_status = 'Solve_Succeeded'
        # Upstream starts from zero, independent of the measured pose. Warmup
        # must not accidentally replace that branch initialization.
        self.initial = np.zeros(6)
        self.history = np.zeros(6)

    def step(self, target, previous, width, dt, velocity, config):
        started = time.perf_counter()
        self.k.validate_feedback(previous, 'teleop previous command')
        initial = self.initial.copy()
        reset = False
        command = None
        reason = ''
        try:
            # Full target and hardware model bounds, without Cartesian shrinking
            # or velocity/acceleration boxes inside the optimization problem.
            command = self.k.solve_ik(target, initial, width, posture_reference=previous)
            difference = float(np.max(np.abs(self.history - command[:6])))
            self.initial = command[:6].copy()
            if difference > 30.0 / 180.0 * 3.1415:
                self.initial = np.zeros(6)
                reset = True
            self.history = command[:6].copy()
            self.k.accept_ik(command)
        except ValueError as error:
            command = None
            reason = str(error)
        info = dict(self.k.last_ik or {})
        position, orientation = pose_error(self.k.fk(previous), target)
        self.last = dict(
            info,
            backend='official_history_pinocchio_casadi_ipopt',
            policy=self.policy,
            state='target_accepted' if command is not None else 'unreachable',
            accepted=command is not None,
            command_valid=command is not None,
            reason=reason,
            tcp_limited=False,
            step_fraction=1.0 if command is not None else 0.0,
            attempts=1,
            seed_rad=initial.tolist(),
            next_seed_rad=self.initial.tolist(),
            seed_reset=reset,
            raw_position_error_m=position,
            raw_orientation_error_rad=orientation,
            target_matrix=self.k.fk(command).tolist() if command is not None else None,
            raw_target_matrix=np.asarray(target).tolist(),
            solve_ms=(time.perf_counter() - started) * 1000,
            budget_exceeded=False,
            feedback_recovery=False,
        )
        self.k.last_ik = self.last
        return command


def make_teleop_solver(kinematics, config):
    if config.get('teleop_policy', 'official') == 'official':
        return OfficialTeleopIK(kinematics)
    return TeleopIK(kinematics)
