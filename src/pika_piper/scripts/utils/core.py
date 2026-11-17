"""Pure numerical contracts shared by collection and replay. SI units throughout."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SCHEMA = 1
FIELDS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']


def effective_joint_speed(config):
    """Planned firmware cap; acquisition verifies the driver's actual readback."""
    cap = (
        config['firmware_joint_speed_limit']
        if config.get('configure_firmware_joint_speed', False)
        else config.get('verified_joint_speed_limit', config['max_joint_speed'])
    )
    return min(config['max_joint_speed'], cap)


def transform(xyzrpy):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', xyzrpy[3:6]).as_matrix()
    t[:3, 3] = xyzrpy[:3]
    return t


def vector(t, width):
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_euler('xyz'), width]


def pose_matrix(msg):
    p, q = msg.pose.position, msg.pose.orientation
    t = np.eye(4)
    t[:3, 3] = [p.x, p.y, p.z]
    t[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return t


def relative_target(robot_anchor, hand_anchor, hand, axes, scale=1.0):
    delta = np.linalg.inv(hand_anchor) @ hand
    delta[:3, 3] *= scale
    return robot_anchor @ axes @ delta @ np.linalg.inv(axes)


def limit_pose_step(previous, target, dt, linear_speed, angular_speed):
    """Approach an absolute TCP goal with bounded translation and rotation.

    Interpolate both with one fraction so a fast hand does not ask IK to
    solve a distant pose before the downstream joint limiter can act.
    """
    distance = np.linalg.norm(target[:3, 3] - previous[:3, 3])
    rotation = Rotation.from_matrix(previous[:3, :3].T @ target[:3, :3]).as_rotvec()
    angle = np.linalg.norm(rotation)
    fraction = min(
        1.0, linear_speed * dt / max(distance, 1e-12), angular_speed * dt / max(angle, 1e-12)
    )
    result = previous.copy()
    result[:3, 3] += fraction * (target[:3, 3] - previous[:3, 3])
    result[:3, :3] = previous[:3, :3] @ Rotation.from_rotvec(fraction * rotation).as_matrix()
    # A limited fraction comes from NumPy arithmetic; expose a native bool
    # so the control status remains JSON serializable when limiting starts.
    return result, bool(fraction < 1.0)


def limit_step(previous, target, dt, joint_speed=0.5, gripper_speed=0.04):
    previous, target = np.asarray(previous), np.asarray(target)
    if previous.shape != (7,) or target.shape != (7,) or not np.isfinite(target).all():
        raise ValueError('Expected finite seven-dimensional command')
    bound = np.r_[np.full(6, joint_speed), gripper_speed] * dt
    result = previous + np.clip(target - previous, -bound, bound)
    result[6] = np.clip(result[6], 0, 0.07)
    return result


def interpolate_eef(a, b, alpha):
    result = (1 - alpha) * np.asarray(a) + alpha * np.asarray(b)
    rotations = Rotation.from_euler('xyz', np.array([a[3:6], b[3:6]]))
    result[3:6] = Slerp([0, 1], rotations)([alpha]).as_euler('xyz')[0]
    return result


def align_series(series, t, kind, tolerance):
    """Return value and source time; never extrapolate states or use future commands."""
    stamps = series[0]
    values = series[1]
    i = int(np.searchsorted(stamps, t, side='right'))
    if kind == 'command':
        if i == 0 or t - stamps[i - 1] > tolerance:
            return None
        return values[i - 1], stamps[i - 1]
    if kind == 'image':
        candidates = [j for j in (i - 1, i) if 0 <= j < len(stamps)]
        if not candidates:
            return None
        j = min(candidates, key=lambda j: abs(stamps[j] - t))
        return (values[j], stamps[j]) if abs(stamps[j] - t) <= tolerance else None
    if i and abs(stamps[i - 1] - t) < 1e-8:
        return values[i - 1], stamps[i - 1]
    if i == 0 or i == len(stamps) or stamps[i] - stamps[i - 1] > tolerance:
        return None
    alpha = (t - stamps[i - 1]) / (stamps[i] - stamps[i - 1])
    if kind == 'eef':
        return interpolate_eef(values[i - 1], values[i], alpha), t
    return (1 - alpha) * values[i - 1] + alpha * values[i], t


def validate_metadata(meta):
    if meta.get('schema_version') != SCHEMA:
        raise ValueError('Unknown pipeline schema')
    if meta.get('action_type') not in ('joint', 'eef6d'):
        raise ValueError('Explicit action_type required')
    if meta.get('units') != {'length': 'm', 'angle': 'rad'}:
        raise ValueError('Unsupported units')
    if meta.get('rotation') != 'xyz_rpy' or meta.get('reference') != 'base_link_absolute':
        raise ValueError('Unsupported EEF convention')
    if meta.get('arm') not in ('left', 'right'):
        raise ValueError('Invalid arm')
    for key in ('urdf_sha256', 'tcp_xyzrpy'):
        if key not in meta:
            raise ValueError('Missing ' + key)


def read_trajectory(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    validate_metadata(meta)
    with np.load(path, allow_pickle=False) as data:
        actions = data['actions'].copy()
        stamps = data['timestamps'].copy()
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 2:
        raise ValueError('Trajectory requires at least two seven-dimensional frames')
    if (
        len(stamps) != len(actions)
        or not np.isfinite(actions).all()
        or not np.isfinite(stamps).all()
        or np.any(np.diff(stamps) <= 0)
    ):
        raise ValueError('Invalid trajectory values or timestamps')
    if np.any(actions[:, 6] < 0) or np.any(actions[:, 6] > 0.07):
        raise ValueError('Gripper out of range')
    return meta, actions, stamps - stamps[0]


def validate_control_config(config):
    if config.get('teleop_policy', 'official') not in ('official', 'bounded'):
        raise ValueError('teleop_policy must be official or bounded')
    if config.get('pose_acceptance', 'official') not in ('official', 'strict'):
        raise ValueError('pose_acceptance must be official or strict')
    for key in (
        'tracking_position_tolerance_m',
        'tracking_orientation_tolerance_rad',
        'tracking_joint_error_rad',
        'tracking_lag_grace_s',
        'tracking_stall_speed_rad_s',
        'tracking_stall_timeout_s',
    ):
        if key in config and (
            isinstance(config[key], bool) or not np.isfinite(config[key]) or config[key] <= 0
        ):
            raise ValueError(key + ' must be finite and positive')
    if config['arm'] not in ('left', 'right'):
        raise ValueError('arm must be left or right')
    percent = config.get('motion_speed_percent', 50)
    if isinstance(percent, bool) or not isinstance(percent, int) or not 1 <= percent <= 100:
        raise ValueError('motion_speed_percent must be an integer in [1, 100]')
    if config.get('output_mode', 'position') not in ('position', 'mit_follow'):
        raise ValueError('output_mode must be position or mit_follow')
    if config.get('sdk_protocol', 'v2') not in ('v2', 'v3'):
        raise ValueError('sdk_protocol must be v2 or v3')
    apply_speed = config.get('configure_firmware_joint_speed', False)
    if not isinstance(apply_speed, bool):
        raise ValueError('configure_firmware_joint_speed must be boolean')
    if apply_speed:
        limit = config.get('firmware_joint_speed_limit')
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not np.isfinite(limit)
            or not 0 < limit <= 3.0
        ):
            raise ValueError('firmware_joint_speed_limit must be in (0, 3] rad/s')
    for name in (
        'control_hz',
        'source_timeout',
        'max_joint_speed',
        'max_gripper_speed',
        'gripper_input_max',
    ):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(name + ' must be finite and positive')
    for name, default in [
        ('max_tcp_speed', 0.15),
        ('max_tcp_angular_speed', 1.0),
        ('ik_posture_weight', 1e-4),
        ('gesture_debounce', 0.4),
        ('max_joint_acceleration', 10.0),
        ('teleop_ik_budget_ms', 16.0),
    ]:
        if not np.isfinite(config.get(name, default)) or config.get(name, default) <= 0:
            raise ValueError(name + ' must be finite and positive')
    margin = config.get('joint_limit_margin', 0.035)
    if not np.isfinite(margin) or not 0 <= margin <= 0.25:
        raise ValueError('joint_limit_margin must be finite and in [0, 0.25] rad')
    attempts = config.get('teleop_ik_max_attempts', 2)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 4:
        raise ValueError('teleop_ik_max_attempts must be an integer in [1, 4]')
    firmware_speed = config.get('verified_joint_speed_limit', config['max_joint_speed'])
    if not np.isfinite(firmware_speed) or firmware_speed <= 0:
        raise ValueError('verified_joint_speed_limit must be finite and positive')
    tolerance = config.get('feedback_tolerance_rad', 0.0)
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 0.002:
        raise ValueError('feedback_tolerance_rad must be in [0, 0.002]')
    feedback_limits = config.get('verified_feedback_limits_deg')
    if tolerance and feedback_limits is None:
        raise ValueError('Feedback tolerance requires verified_feedback_limits_deg')
    startup = config.get('near_zero_start_tolerance_deg', 0.0)
    if not np.isfinite(startup) or not 0 <= startup <= 0.5:
        raise ValueError('near_zero_start_tolerance_deg must be in [0, 0.5]')
    if startup and feedback_limits is None:
        raise ValueError('Near-zero startup requires verified_feedback_limits_deg')
    if feedback_limits is not None:
        bounds = np.asarray(feedback_limits, dtype=float)
        if (
            bounds.shape != (6, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
        ):
            raise ValueError('verified_feedback_limits_deg requires six finite [min,max] pairs')
    for name in ('tcp_xyzrpy', 'pika_to_tcp_xyzrpy'):
        value = np.asarray(config[name])
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError('Invalid ' + name)
    if np.asarray(config['axis_rpy']).shape != (3,) or not np.isfinite(config['axis_rpy']).all():
        raise ValueError('Invalid axis_rpy')
    if not np.isfinite(config['translation_scale']) or config['translation_scale'] <= 0:
        raise ValueError('Invalid translation scale')
