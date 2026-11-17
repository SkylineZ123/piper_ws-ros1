"""ROS controller state machine, watchdogs and command publication."""

import json
import threading
import time

import numpy as np
import rospy
from data_msgs.msg import Gripper, LocalizationStatus, TeleopStatus
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, String
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse, Trigger, TriggerResponse

from .configuration import MOTION_KEYS, load_config, package_file
from .core import (
    effective_joint_speed,
    limit_step,
    pose_matrix,
    relative_target,
    transform,
    validate_control_config,
    vector,
)
from .kinematics import Kinematics
from .official import OfficialInterpolator, make_teleop_solver
from .tracking import TrackingMonitor


class Controller:
    def __init__(self):
        self.cfg = load_config(rospy.get_param('~config', None), rospy.get_param('~', {}))
        c = self.cfg
        validate_control_config(c)
        urdf = package_file('urdf/piper_description.urdf', package='piper_description')
        srdf = package_file('config/piper_description.srdf', package='piper_with_gripper_moveit')
        self.k = Kinematics(
            urdf,
            urdf.parent.parent,
            c['tcp_xyzrpy'],
            srdf,
            posture_weight=c.get('ik_posture_weight', 1e-4),
            feedback_limits_deg=c.get('verified_feedback_limits_deg'),
            feedback_tolerance=c.get('feedback_tolerance_rad', 0.0),
            near_zero_start_tolerance_deg=c.get('near_zero_start_tolerance_deg', 0.0),
            pose_acceptance=c.get('pose_acceptance', 'official'),
        )
        self.teleop_solver = make_teleop_solver(self.k, c)
        self.command_schedule = None
        self.follow_monitor = TrackingMonitor()
        self.requested_target = None
        self.feedback_recovering = False
        self.command_velocity = np.zeros(6)
        self.tracking = None
        self.startup_target = None
        self.startup_elapsed = 0.0
        self.driver_config_param = rospy.get_namespace() + 'driver/applied_configuration'
        self.driver_configuration = None
        self.lock = threading.RLock()
        self.mode = 'idle'
        self.reason = 'startup'
        self.last_fault = None
        self.hand = self.feedback = self.sent = self.anchor = self.robot_anchor = None
        self.hand_at = self.feedback_at = self.grip_at = self.quality_at = self.replay_at = 0.0
        self.width = 0.0
        self.accurate = False
        self.replay = None
        self.command_started = 0.0
        self.echo_at = 0.0
        self.gesture_at = 0.0
        self.gesture_command = None
        self.last_echo = None
        self.target_limited = False
        self.timing = {}
        self.joint_limited = False
        self.axes = transform([0, 0, 0] + c['axis_rpy'])
        self.hand_tcp = transform(c['pika_to_tcp_xyzrpy'])
        self.last_tick = time.monotonic()
        self.cmd = rospy.Publisher('command', JointState, queue_size=1, tcp_nodelay=True)
        self.state = rospy.Publisher('status', String, queue_size=10, latch=True)
        self.eef_state = rospy.Publisher('eef_state', JointState, queue_size=10)
        self.eef_cmd = rospy.Publisher('eef_command', JointState, queue_size=10)
        self.request = rospy.Publisher('eef_requested', PoseStamped, queue_size=1)
        self.feasible_request = rospy.Publisher('eef_feasible', PoseStamped, queue_size=1)
        self.sense_relative = rospy.Publisher(
            'sense_relative', PoseStamped, queue_size=1, tcp_nodelay=True
        )
        self.teleop_status = rospy.Publisher(
            '/teleop_status', TeleopStatus, queue_size=1, latch=True
        )
        rospy.Subscriber('/pika_pose', PoseStamped, self.on_hand, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/gripper/data', Gripper, self.on_grip, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            '/pika_localization_status',
            LocalizationStatus,
            self.on_quality,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.Subscriber('feedback', JointState, self.on_feedback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('command_sent', JointState, self.on_sent, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('replay_input', JointState, self.on_replay, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            '/pika/gesture_command', Int32, self.on_gesture, queue_size=1, tcp_nodelay=True
        )
        rospy.Service('set_teleop', SetBool, self.set_teleop)
        rospy.Service('toggle_teleop', Trigger, self.toggle_teleop)
        rospy.Service('set_replay', SetBool, self.set_replay)
        rospy.Service('reload_motion_params', Trigger, self.reload_motion_params)
        rospy.Timer(rospy.Duration(1 / c['control_hz']), self.tick)
        rospy.Timer(rospy.Duration(1 / OfficialInterpolator.rate_hz), self.command_tick)

    def fault(self, reason):
        if self.mode != 'fault' or self.reason != reason:
            self.last_fault = {'reason': reason, 'stamp': rospy.Time.now().to_sec()}
            rospy.logerr('Pika controller fault: %s', reason)
        self.mode = 'fault'
        self.reason = reason
        self.anchor = None
        self.startup_target = None
        if getattr(self, 'command_schedule', None) is not None:
            self.command_schedule.cancel()
        if self.tracking is not None:
            self.tracking.update(state='fault', reason=reason)

    def fresh(self, at):
        return 0 <= time.monotonic() - at < self.cfg['source_timeout']

    def on_hand(self, msg):
        with self.lock:
            try:
                h = pose_matrix(msg) @ self.hand_tcp
                if not np.isfinite(h).all():
                    raise ValueError('Non-finite pose')
                if self.mode == 'teleop' and self.hand is not None:
                    delta = np.linalg.inv(self.hand) @ h
                    if np.linalg.norm(delta[:3, 3]) > 0.1 or Rotation.from_matrix(
                        delta[:3, :3]
                    ).magnitude() > np.deg2rad(30):
                        self.fault('tracking jump')
                        return
                self.hand = h
                self.hand_at = time.monotonic()
                if abs((rospy.Time.now() - msg.header.stamp).to_sec()) > self.cfg['source_timeout']:
                    self.hand_at = 0.0
            except ValueError as e:
                self.fault(str(e))

    def on_grip(self, msg):
        with self.lock:
            self.width = float(
                np.clip(msg.distance / self.cfg['gripper_input_max'] * 0.07, 0, 0.07)
            )
            self.grip_at = time.monotonic()
            if not np.isfinite(msg.distance):
                self.grip_at = 0.0

    def on_quality(self, msg):
        with self.lock:
            self.accurate = msg.accurate
            self.quality_at = time.monotonic()

    def publish_eef(self, pub, msg):
        out = JointState()
        out.header = msg.header
        out.header.frame_id = 'base_link'
        out.name = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
        out.position = vector(self.k.fk(msg.position), msg.position[6])
        pub.publish(out)

    def on_feedback(self, msg):
        with self.lock:
            if len(msg.position) != 7 or not np.isfinite(msg.position).all():
                return
            self.feedback = np.array(msg.position)
            self.feedback_at = time.monotonic()
            if abs((rospy.Time.now() - msg.header.stamp).to_sec()) > self.cfg['source_timeout']:
                self.feedback_at = 0.0
            self.publish_eef(self.eef_state, msg)

    def on_sent(self, msg):
        with self.lock:
            if len(msg.position) != 7 or not np.isfinite(msg.position).all():
                return
            if abs((rospy.Time.now() - msg.header.stamp).to_sec()) > self.cfg['source_timeout']:
                return
            # Echo is acknowledgement, not the limiter's next starting point.
            # Delayed echoes must never rewind an already issued command.
            self.echo_at = time.monotonic()
            self.last_echo = np.array(msg.position)
            self.publish_eef(self.eef_cmd, msg)

    def toggle_teleop(self, req):
        with self.lock:
            response = self.set_teleop(SetBoolRequest(self.mode != 'teleop'))
            return TriggerResponse(response.success, response.message)

    def on_gesture(self, msg):
        # Sense firmware already recognizes double-clicks and changes Command.
        # The serial node emits changes only, never its startup baseline.
        with self.lock:
            if not self.cfg.get('sense_gesture_control', False):
                return
            now = time.monotonic()
            if msg.data == self.gesture_command:
                return
            self.gesture_command = msg.data
            if now - self.gesture_at < self.cfg.get('gesture_debounce', 0.4):
                return
            self.gesture_at = now
            response = self.toggle_teleop(None)
            if not response.success:
                rospy.logwarn('Sense teleop toggle rejected: %s', response.message)

    def on_replay(self, msg):
        with self.lock:
            if self.mode != 'replay':
                return
            if len(msg.position) != 7 or not msg.name or msg.name[0] not in ('joint', 'eef6d'):
                self.fault('invalid replay command')
                return
            if abs((rospy.Time.now() - msg.header.stamp).to_sec()) > self.cfg['source_timeout']:
                self.fault('stale replay command')
                return
            self.replay = (msg.name[0], np.asarray(msg.position))
            self.replay_at = time.monotonic()

    def set_teleop(self, req):
        with self.lock:
            if self.mode == 'replay':
                return SetBoolResponse(False, 'replay owns control')
            if not req.data:
                self.mode = 'idle'
                self.reason = 'operator pause'
                self.anchor = None
                self.robot_anchor = None
                self.startup_target = None
                self.startup_elapsed = 0.0
                if getattr(self, 'command_schedule', None) is not None:
                    self.command_schedule.cancel()
                return SetBoolResponse(True, 'idle')
            if (
                not all(
                    self.fresh(t)
                    for t in [self.hand_at, self.feedback_at, self.grip_at, self.quality_at]
                )
                or not self.accurate
            ):
                return SetBoolResponse(
                    False, 'waiting for fresh pose, gripper, localization and feedback'
                )
            driver_error = self.verify_driver_configuration()
            if driver_error:
                return SetBoolResponse(False, driver_error)
            try:
                self.k.validate_teleop_start(self.feedback, 'teleop feedback')
                startup_target = self.k.start_target(self.feedback)
                recovering = bool(np.any(startup_target != self.feedback))
                robot_anchor = self.k.fk(self.feedback)
                # A startup recovery has an explicit legal joint destination;
                # do not ask IK to hold the out-of-model measured pose.
                if not recovering:
                    self.k.ik(robot_anchor, self.feedback, self.feedback[6])
                self.teleop_solver.warmup(startup_target)
            except ValueError as error:
                self.fault(str(error))
                return SetBoolResponse(False, str(error))
            self.anchor = self.hand.copy()
            self.robot_anchor = robot_anchor
            self.mode = 'teleop'
            self.reason = ''
            self.sent = self.feedback.copy()
            self.command_started = 0.0
            self.command_velocity = np.zeros(6)
            self.tracking = None
            self.follow_monitor = TrackingMonitor()
            self.requested_target = robot_anchor.copy()
            self.command_schedule = (
                OfficialInterpolator()
                if getattr(self.teleop_solver, 'policy', None) == 'official'
                else None
            )
            self.command_publish_at = time.monotonic()
            self.feedback_recovering = recovering
            self.feedback_recovery_deadline = time.monotonic() + 1.0
            self.startup_target = startup_target if recovering else None
            self.startup_elapsed = 0.0
            self.last_tick = time.monotonic()
            return SetBoolResponse(True, 'teleop')

    def set_replay(self, req):
        with self.lock:
            if req.data:
                if self.mode in ('teleop', 'replay') or not self.fresh(self.feedback_at):
                    return SetBoolResponse(False, 'teleop active or no feedback')
                driver_error = self.verify_driver_configuration()
                if driver_error:
                    return SetBoolResponse(False, driver_error)
                try:
                    self.k.validate_configuration(self.feedback, 'replay feedback')
                except ValueError as error:
                    self.fault(str(error))
                    return SetBoolResponse(False, str(error))
                self.mode = 'replay'
                self.replay = None
                self.replay_at = time.monotonic()
                self.sent = self.feedback.copy()
                self.command_started = 0.0
                self.reason = ''
            elif self.mode == 'replay':
                self.mode = 'idle'
                self.reason = 'replay ended'
            return SetBoolResponse(True, self.mode)

    def verify_driver_configuration(self):
        # Read only at acquisition, not at command frequency. The planned fast
        # cap cannot be used on hardware before all six speeds are read back.
        if (
            not self.cfg.get('configure_firmware_joint_speed', False)
            and self.cfg.get('output_mode', 'position') == 'position'
        ):
            return None
        report = rospy.get_param(
            getattr(self, 'driver_config_param', '/pika_piper/driver/applied_configuration'), None
        )
        if not isinstance(report, dict) or report.get('ready') is not True:
            return 'waiting for driver firmware/protocol/speed configuration'
        if report.get('output_mode') != self.cfg.get('output_mode', 'position') or report.get(
            'sdk_protocol'
        ) != self.cfg.get('sdk_protocol', 'v2'):
            return 'driver output mode or SDK protocol does not match controller configuration'
        if self.cfg.get('configure_firmware_joint_speed', False):
            speeds = np.asarray(report.get('joint_speed_limits_rad_s', []), dtype=float)
            expected = self.cfg['firmware_joint_speed_limit']
            if (
                speeds.shape != (6,)
                or not np.isfinite(speeds).all()
                or np.any(np.abs(speeds - expected) > 0.00051)
            ):
                return 'driver has not verified the requested joint speed on all six motors'
        self.driver_configuration = report
        return None

    def reload_motion_params(self, req):
        """Atomically apply software speed limits only while control is paused."""
        with self.lock:
            if self.mode in ('teleop', 'replay'):
                return TriggerResponse(
                    False, 'Pause teleop/replay before reloading motion parameters'
                )
            updates = rospy.get_param('~', {})
            if 'control_hz' in updates and updates['control_hz'] != self.cfg['control_hz']:
                return TriggerResponse(False, 'control_hz requires a node restart')
            candidate = dict(self.cfg)
            candidate.update({key: updates[key] for key in MOTION_KEYS if key in updates})
            try:
                validate_control_config(candidate)
            except (ValueError, TypeError, KeyError) as error:
                return TriggerResponse(False, str(error))
            self.cfg = candidate
            return TriggerResponse(
                True, json.dumps({key: self.cfg[key] for key in MOTION_KEYS if key in self.cfg})
            )

    def tick(self, event):
        with self.lock:
            started = time.perf_counter()
            now = time.monotonic()
            period = now - self.last_tick
            dt = min(period, 2 / self.cfg['control_hz'])
            self.last_tick = now
            self.target_limited = False
            self.joint_limited = False
            try:
                if self.mode in ('teleop', 'replay') and not self.fresh(self.feedback_at):
                    self.fault('feedback timeout')
                if self.mode in ('teleop', 'replay'):
                    self.validate_runtime_feedback(now)
                if self.mode in ('teleop', 'replay') and self.command_echo_overdue():
                    self.fault('driver command echo timeout')
                if self.mode == 'teleop':
                    if (
                        not all(
                            self.fresh(t) for t in [self.hand_at, self.grip_at, self.quality_at]
                        )
                        or not self.accurate
                    ):
                        self.fault('tracking unavailable')
                    else:
                        relative = np.linalg.inv(self.anchor) @ self.hand
                        h = PoseStamped()
                        h.header.stamp = rospy.Time.now()
                        h.header.frame_id = 'sense_reference'
                        h.pose.position.x, h.pose.position.y, h.pose.position.z = relative[:3, 3]
                        (
                            h.pose.orientation.x,
                            h.pose.orientation.y,
                            h.pose.orientation.z,
                            h.pose.orientation.w,
                        ) = Rotation.from_matrix(relative[:3, :3]).as_quat()
                        self.sense_relative.publish(h)
                        target = relative_target(
                            self.robot_anchor,
                            self.anchor,
                            self.hand,
                            self.axes,
                            self.cfg['translation_scale'],
                        )
                        self.requested_target = target
                        p = PoseStamped()
                        p.header.stamp = rospy.Time.now()
                        p.header.frame_id = 'base_link'
                        p.pose.position.x, p.pose.position.y, p.pose.position.z = target[:3, 3]
                        (
                            p.pose.orientation.x,
                            p.pose.orientation.y,
                            p.pose.orientation.z,
                            p.pose.orientation.w,
                        ) = Rotation.from_matrix(target[:3, :3]).as_quat()
                        self.request.publish(p)
                        q = self.teleop_ik(target, dt)
                        if q is not None:
                            recovering = self.startup_target is not None
                            if not recovering and self.command_schedule is not None:
                                if self.command_schedule.target is None:
                                    # A new target after intentional cache expiry
                                    # starts a new acknowledgement grace window.
                                    self.command_started = 0.0
                                self.command_schedule.offer(q, now)
                                self.command_tick(None)
                            else:
                                self.send(
                                    q,
                                    self.startup_elapsed if recovering else dt,
                                    startup=recovering,
                                )
                            if recovering:
                                # Anchor the hand only once the legal first
                                # command was issued; ignore hand motion during
                                # the short initial recovery wait.
                                self.startup_target = None
                                self.anchor = self.hand.copy()
                                self.robot_anchor = self.k.fk(self.sent)
                if self.mode == 'replay':
                    if not self.fresh(self.replay_at):
                        self.fault('replay timeout')
                    elif self.replay is not None:
                        kind, a = self.replay
                        q = a if kind == 'joint' else self.k.ik(transform(a[:6]), self.sent, a[6])
                        self.send(q, dt)
                if self.mode == 'teleop':
                    self.refresh_tracking(now)
            except (ValueError, RuntimeError) as e:
                self.fault(str(e))
            self.timing = {
                'tick_ms': (time.perf_counter() - started) * 1000,
                'loop_hz': 1 / max(period, 1e-9),
                'hand_age_s': now - self.hand_at,
                'echo_age_s': now - self.echo_at if self.echo_at else None,
            }
            state = {
                'mode': self.mode,
                'reason': self.reason,
                'last_fault': self.last_fault,
                'stamp': rospy.Time.now().to_sec(),
                'urdf_sha256': self.k.urdf_hash,
                'kinematics': self.k.info,
                'ik': self.k.last_ik,
                'timing': self.timing,
                'target_limited': bool(self.target_limited),
                'axis_rpy': self.cfg.get('axis_rpy', [0, 0, 0]),
                'joint_limited': bool(self.joint_limited),
                'tracking': self.tracking,
                'teleop_policy': getattr(self.teleop_solver, 'policy', 'bounded'),
                'command_publication_policy': (
                    'replay_direct_targets'
                    if self.mode == 'replay'
                    else 'official_30deg_1deg_200hz'
                    if getattr(self.teleop_solver, 'policy', None) == 'official'
                    else 'bounded_joint_step'
                ),
                'command_feedback_error_rad': float(
                    np.max(np.abs(self.sent[:6] - self.feedback[:6]))
                )
                if self.sent is not None and self.feedback is not None
                else None,
                'motion_limits': {
                    key: self.cfg[key] for key in ('control_hz',) + MOTION_KEYS if key in self.cfg
                },
                'effective_joint_speed_rad_s': effective_joint_speed(self.cfg)
                if 'max_joint_speed' in self.cfg
                and self.mode != 'replay'
                and not (
                    self.mode == 'teleop'
                    and getattr(self.teleop_solver, 'policy', None) == 'official'
                )
                else None,
                'configured_firmware_joint_speed_rad_s': self.cfg.get('firmware_joint_speed_limit'),
                'output_mode_requested': self.cfg.get('output_mode', 'position'),
                'driver_configuration': getattr(self, 'driver_configuration', None),
                'sense_gesture_control': self.cfg.get('sense_gesture_control', False),
            }
            self.state.publish(String(json.dumps(state)))
            status = TeleopStatus()
            status.quit = self.mode != 'teleop'
            status.fail = self.mode == 'fault'
            self.teleop_status.publish(status)

    def teleop_ik(self, target, dt):
        self.tracking = None
        if self.startup_target is not None:
            self.startup_elapsed += dt
            q = self.teleop_solver.startup(
                self.sent, self.startup_target, self.startup_elapsed, self.cfg, target
            )
        else:
            q = self.teleop_solver.step(
                target, self.sent, self.width, dt, self.command_velocity, self.cfg
            )
        self.tracking = dict(self.teleop_solver.last)
        self.target_limited = bool(
            self.tracking['tcp_limited'] or self.tracking['step_fraction'] < 1.0
        )
        feasible = self.tracking.get('target_matrix')
        if feasible is not None:
            p = PoseStamped()
            p.header.stamp = rospy.Time.now()
            p.header.frame_id = 'base_link'
            pose = np.asarray(feasible)
            p.pose.position.x, p.pose.position.y, p.pose.position.z = pose[:3, 3]
            (
                p.pose.orientation.x,
                p.pose.orientation.y,
                p.pose.orientation.z,
                p.pose.orientation.w,
            ) = Rotation.from_matrix(pose[:3, :3]).as_quat()
            self.feasible_request.publish(p)
        return q

    def validate_runtime_feedback(self, now):
        if getattr(self, 'feedback_recovering', False):
            self.k.validate_teleop_start(self.feedback, 'runtime recovery feedback')
            try:
                self.k.validate_feedback(self.feedback, 'runtime feedback')
                self.feedback_recovering = False
            except ValueError:
                if now >= self.feedback_recovery_deadline:
                    raise ValueError('Startup feedback did not return inside the verified range')
        else:
            self.k.validate_feedback(self.feedback, 'runtime feedback')

    def command_echo_overdue(self):
        if not self.command_started or self.fresh(max(self.command_started, self.echo_at)):
            return False
        schedule = getattr(self, 'command_schedule', None)
        # Upstream intentionally stops publication when the last valid target
        # expires. An already acknowledged idle publisher is not a driver fault.
        if (
            self.mode == 'teleop'
            and schedule is not None
            and schedule.target is None
            and self.echo_at >= getattr(self, 'command_publish_at', self.command_started)
        ):
            return False
        return True

    def refresh_tracking(self, now):
        if self.sent is None or getattr(self, 'requested_target', None) is None:
            return
        if not hasattr(self, 'follow_monitor'):
            self.follow_monitor = TrackingMonitor()
        tracking = dict(self.tracking or {})
        feasible = tracking.get('target_matrix')
        schedule = getattr(self, 'command_schedule', None)
        measured = self.follow_monitor.update(
            self.k,
            self.requested_target,
            np.asarray(feasible) if feasible is not None else None,
            self.sent,
            self.feedback,
            now,
            self.cfg,
            tracking.get('state', 'unknown'),
            interpolating=schedule is not None and schedule.segment is not None,
        )
        tracking.update(measured)
        self.tracking = tracking
        if measured['execution_stalled']:
            self.fault('execution stalled: fresh measured joints do not follow issued commands')

    def command_tick(self, event):
        with self.lock:
            schedule = getattr(self, 'command_schedule', None)
            if self.mode != 'teleop' or self.startup_target is not None or schedule is None:
                return
            try:
                if (
                    not all(
                        self.fresh(t)
                        for t in (self.hand_at, self.grip_at, self.quality_at, self.feedback_at)
                    )
                    or not self.accurate
                ):
                    raise ValueError('tracking or feedback unavailable during interpolation')
                if self.command_echo_overdue():
                    raise ValueError('driver command echo timeout during interpolation')
                now = time.monotonic()
                self.validate_runtime_feedback(now)
                command = schedule.sample(self.sent, now)
                if command is not None:
                    self.send(
                        command,
                        max(now - self.command_publish_at, 1 / OfficialInterpolator.rate_hz),
                        official=True,
                    )
                    self.command_publish_at = now
            except (ValueError, RuntimeError) as error:
                self.fault(str(error))

    def send(self, q, dt, startup=False, official=False):
        if startup:
            self.k.validate_teleop_start(self.sent, 'previous startup feedback')
        else:
            self.k.validate_feedback(self.sent, 'previous command')
        self.k.validate_configuration(q, 'command target')
        target = np.asarray(q)
        if official:
            self.teleop_solver._validate_path(self.sent, target)
            q = target.copy()
        elif self.mode == 'replay':
            # Replay preserves target values and source timing. The worker
            # generates a smooth first-frame approach before recorded playback.
            q = target.copy()
        else:
            q = limit_step(
                self.sent,
                target,
                dt,
                effective_joint_speed(self.cfg),
                self.cfg['max_gripper_speed'],
            )
        self.joint_limited = bool(np.any(np.abs(target - q) > 1e-9))
        self.k.validate_configuration(q, 'limited command')
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'base_link'
        msg.name = ['joint%d' % i for i in range(1, 7)] + ['gripper']
        msg.position = q
        if not self.command_started:
            self.command_started = time.monotonic()
        self.cmd.publish(msg)
        self.command_publish_at = time.monotonic()
        self.command_velocity = (q[:6] - self.sent[:6]) / dt
        # Limit against issued command even while waiting for driver echo.
        self.sent = q.copy()
