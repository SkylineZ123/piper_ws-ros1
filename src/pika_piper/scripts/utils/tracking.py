"""Measured tracking quality, independent of the IK acceptance threshold."""

import numpy as np

from .teleop import pose_error


class TrackingMonitor:
    def __init__(self):
        self.lag_since = self.stall_since = None
        self.last_feedback = None
        self.last_at = None
        self.lagging = False

    def update(
        self,
        kinematics,
        requested,
        feasible,
        command,
        feedback,
        now,
        config,
        solver_state,
        interpolating=False,
    ):
        measured = kinematics.fk(feedback)
        issued = kinematics.fk(command)
        position, orientation = pose_error(measured, requested)
        command_position, command_orientation = pose_error(issued, requested)
        follow_position, follow_orientation = pose_error(measured, issued)
        gap = float(np.max(np.abs(np.asarray(command)[:6] - np.asarray(feedback)[:6])))
        speed = 0.0
        if self.last_feedback is not None and now > self.last_at:
            speed = float(
                np.max(np.abs(np.asarray(feedback)[:6] - self.last_feedback)) / (now - self.last_at)
            )
        threshold = config.get('tracking_joint_error_rad', 0.15)
        # Hysteresis prevents repeatedly flipping state on a noisy boundary.
        self.lagging = gap > threshold * (0.7 if self.lagging else 1.0)
        if self.lagging:
            if self.lag_since is None:
                self.lag_since = now
        else:
            self.lag_since = self.stall_since = None
        stationary = speed < config.get('tracking_stall_speed_rad_s', 0.02)
        if self.lagging and stationary:
            if self.stall_since is None:
                self.stall_since = now
        else:
            self.stall_since = None
        stall_duration = 0.0 if self.stall_since is None else max(0.0, now - self.stall_since)
        lag_duration = 0.0 if self.lag_since is None else max(0.0, now - self.lag_since)
        stalled = stall_duration >= config.get('tracking_stall_timeout_s', 0.75)
        reached = position <= config.get(
            'tracking_position_tolerance_m', 0.02
        ) and orientation <= config.get('tracking_orientation_tolerance_rad', np.deg2rad(10.0))
        if solver_state == 'recovering':
            state, reason = 'recovering', 'startup_feedback_recovery'
            stalled = False
        elif stalled:
            state, reason = 'stalled', 'measured_joints_not_following_command'
        elif self.lagging:
            state, reason = (
                (
                    'lagging'
                    if lag_duration >= config.get('tracking_lag_grace_s', 0.25)
                    else 'catching_up'
                ),
                'command_feedback_gap',
            )
        elif solver_state == 'unreachable':
            state, reason = 'unreachable', 'latest_ik_target_rejected'
        elif reached:
            state, reason = 'tracking', ''
        elif interpolating:
            state, reason = 'interpolating', 'official_joint_interpolation'
        else:
            state, reason = 'limited', 'measured_pose_outside_tracking_tolerance'
        feasible_error = pose_error(measured, feasible) if feasible is not None else (None, None)
        self.last_feedback = np.asarray(feedback)[:6].copy()
        self.last_at = now
        return dict(
            state=state,
            reason=reason,
            solver_state=solver_state,
            error_reference='measured_feedback_fk',
            measured_target_reached=bool(reached),
            raw_position_error_m=position,
            raw_orientation_error_rad=orientation,
            feedback_target_position_error_m=position,
            feedback_target_orientation_error_rad=orientation,
            command_target_position_error_m=command_position,
            command_target_orientation_error_rad=command_orientation,
            feedback_command_position_error_m=follow_position,
            feedback_command_orientation_error_rad=follow_orientation,
            feedback_feasible_position_error_m=feasible_error[0],
            feedback_feasible_orientation_error_rad=feasible_error[1],
            command_feedback_error_rad=gap,
            feedback_joint_speed_rad_s=speed,
            lag_duration_s=lag_duration,
            stall_duration_s=stall_duration,
            execution_stalled=bool(stalled),
        )
