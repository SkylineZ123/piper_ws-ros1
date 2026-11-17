"""Local LeRobot episode selection, continuity checks and quality measurements."""
import json
from pathlib import Path

import numpy as np


def latest_dataset(search_dir):
    candidates = []
    for path in Path(search_dir).rglob('meta/pika_pipeline.json'):
        info_path = path.with_name('info.json')
        try:
            info = json.loads(info_path.read_text())
            if info['total_episodes'] > 0 and info['total_frames'] > 0:
                candidates.append((info_path.stat().st_mtime_ns, str(path.parent.parent)))
        except (OSError, ValueError, KeyError):
            continue
    if not candidates:
        raise ValueError('No saved Pika LeRobot dataset under ' + str(search_dir))
    return Path(max(candidates)[1])


def select_episodes(selection, total):
    if selection == 'all':
        return list(range(total))
    values = selection if isinstance(selection, list) else [selection]
    if not values or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < total for v in values):
        raise ValueError('episodes must be all, an episode index, or a list of indices in [0, %d)' % total)
    if len(set(values)) != len(values):
        raise ValueError('Duplicate episode indices')
    return values


def episode_arrays(table, episode, start_frame=0, end_frame=None, segment_id=None):
    indices = np.flatnonzero(np.asarray(table['episode_index']).reshape(-1) == episode)
    if not len(indices):
        raise ValueError('No frames for episode %d' % episode)
    frame_indices = np.asarray(table['frame_index']).reshape(-1)[indices]
    selected = frame_indices >= start_frame
    if end_frame is not None:
        selected &= frame_indices < end_frame
    if segment_id is not None:
        if 'segment_id' not in table:
            raise ValueError('Dataset has no segment_id feature')
        selected &= np.asarray(table['segment_id']).reshape(-1)[indices] == segment_id
    indices = indices[selected]
    if not len(indices):
        raise ValueError('Selection contains no frames for episode %d' % episode)
    result = {}
    for key in ('action', 'observation.state', 'observation.eef_state', 'joint_command', 'eef_command',
                'timestamp', 'frame_index', 'source_timestamp', 'segment_id', 'subtask_index', 'back_event'):
        if key not in table:
            continue
        value = np.asarray([table[key][i] for i in indices])
        if key in ('action', 'observation.state', 'observation.eef_state', 'joint_command', 'eef_command'):
            value = value.astype(np.float32)
            if value.shape != (len(indices), 7) or not np.isfinite(value).all():
                raise ValueError('Expected finite (N, 7) feature: ' + key)
        else:
            value = value.reshape(-1)
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite ' + key)
        result[key] = value
    for key in ('action', 'observation.state', 'timestamp', 'frame_index'):
        if key not in result:
            raise ValueError('Missing required feature: ' + key)
    if np.any(np.diff(result['timestamp']) <= 0):
        raise ValueError('Episode timestamps must be strictly increasing')
    return result


def continuous_spans(arrays, fps):
    """Never interpolate across c/resume, source-time gaps, or removed frames."""
    count = len(arrays['action'])
    segments = arrays.get('segment_id', np.zeros(count, dtype=np.int64))
    source = arrays.get('source_timestamp', arrays['timestamp'])
    changes = np.flatnonzero(
        (np.diff(segments) != 0) | (np.diff(source) <= 0) |
        (np.diff(source) > 1.5 / fps) | (np.diff(arrays['frame_index']) != 1)) + 1
    bounds = [0] + changes.tolist() + [count]
    return [(start, end) for start, end in zip(bounds[:-1], bounds[1:])]


def replay_span(arrays, fps):
    spans = continuous_spans(arrays, fps)
    if len(spans) != 1:
        raise ValueError('Physical replay requires one continuous span; select segment_id '
                         'and/or start_frame/end_frame in replay.yaml')
    if len(arrays['action']) < 2:
        raise ValueError('Physical replay requires at least two frames')
    return arrays['action'], arrays.get('source_timestamp', arrays['timestamp'])


def speed_limit_from_peak(peak, joint_speed, gripper_speed):
    """Largest <=1 time scale that avoids changing targets via speed clipping."""
    return float(min(1., joint_speed / max(float(np.max(peak[:6])), 1e-12),
                     gripper_speed / max(float(peak[6]), 1e-12)))


def check_physical_speed(joints, stamps, speed, joint_speed, gripper_speed):
    peak = np.max(abs(np.diff(np.asarray(joints, dtype=float), axis=0) /
                      np.diff(stamps)[:, None]), axis=0)
    limit = speed_limit_from_peak(peak, joint_speed, gripper_speed)
    if speed > limit + 1e-6:
        raise ValueError('Replay would be clipped by controller speed limits; '
                         'set speed <= %.6f (current %.4f)' % (limit, speed))
    return limit


def error_statistics(command, state, eef=False):
    error = np.asarray(command) - np.asarray(state)
    if eef:
        error[:, 3:6] = (error[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    return dict(rmse=np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
                mae=np.mean(abs(error), axis=0).tolist(),
                max_abs=np.max(abs(error), axis=0).tolist())


def quality_report(arrays, meta, fps):
    count = len(arrays['action'])
    spans = continuous_spans(arrays, fps)
    segments = arrays.get('segment_id', np.zeros(count, dtype=np.int64))
    source = arrays.get('source_timestamp', arrays['timestamp'])
    report = dict(frames=count, duration_s=count / fps, fps=fps,
                  action_type=meta['action_type'],
                  source_duration_s=float(source[-1] - source[0]),
                  segment_ids=[int(v) for v in dict.fromkeys(segments.tolist())],
                  continuous_spans=[dict(start_frame=int(arrays['frame_index'][a]),
                                         end_frame=int(arrays['frame_index'][b-1]) + 1,
                                         segment_id=int(segments[a]), frames=b-a)
                                    for a, b in spans],
                  back_event_frames=int(np.count_nonzero(arrays.get('back_event', []))),
                  dataset_time_step_max_error_s=float(np.max(abs(np.diff(arrays['timestamp']) - 1/fps)))
                  if count > 1 else 0.)
    source_dt = np.diff(source)
    if len(source_dt):
        report['source_time_step_s'] = dict(min=float(source_dt.min()), median=float(np.median(source_dt)),
                                              max=float(source_dt.max()))
        report['source_gaps'] = int(np.count_nonzero(source_dt > 1.5/fps))
        report['source_clock_reversals'] = int(np.count_nonzero(source_dt <= 0))
    if 'joint_command' in arrays:
        report['joint_tracking'] = error_statistics(arrays['joint_command'], arrays['observation.state'])
    if 'eef_command' in arrays and 'observation.eef_state' in arrays:
        report['eef_tracking'] = error_statistics(arrays['eef_command'], arrays['observation.eef_state'], eef=True)
        from scipy.spatial.transform import Rotation
        command = arrays['eef_command']
        state = arrays['observation.eef_state']
        angles = (Rotation.from_euler('xyz', state[:, 3:6]).inv() *
                  Rotation.from_euler('xyz', command[:, 3:6])).magnitude()
        report['eef_tracking']['position_rmse_m'] = float(np.sqrt(np.mean(np.sum((command[:, :3]-state[:, :3])**2, axis=1))))
        report['eef_tracking']['orientation_rmse_rad'] = float(np.sqrt(np.mean(angles**2)))
    target = 'joint_command' if meta['action_type'] == 'joint' else 'eef_command'
    if target in arrays:
        report['action_command_max_difference'] = float(np.max(abs(arrays['action'] - arrays[target])))
    velocities = []
    for start, end in spans:
        if end - start < 2:
            continue
        delta = np.diff(arrays['action'][start:end], axis=0)
        if meta['action_type'] == 'eef6d':
            delta[:, 3:6] = (delta[:, 3:6] + np.pi) % (2*np.pi) - np.pi
        velocities.append(abs(delta / np.diff(source[start:end, None], axis=0)))
    if velocities:
        report['action_peak_speed_per_dimension'] = np.max(np.concatenate(velocities), axis=0).tolist()
    if 'joint_command' in arrays:
        joint_velocities = [abs(np.diff(arrays['joint_command'][a:b], axis=0) /
                                np.diff(source[a:b])[:, None]) for a, b in spans if b-a > 1]
        config = meta.get('config', {})
        if joint_velocities and 'max_joint_speed' in config and 'max_gripper_speed' in config:
            from .core import effective_joint_speed
            peak = np.max(np.concatenate(joint_velocities), axis=0)
            report['joint_command_peak_speed'] = peak.tolist()
            report['suggested_max_replay_speed'] = speed_limit_from_peak(
                peak, effective_joint_speed(config), config['max_gripper_speed'])
    return report


class CameraFrames:
    """Sequential PyAV read with v3 episode offsets, including shared shards."""

    def __init__(self, path, start_time, fps):
        import av
        self.path = Path(path)
        self.container = av.open(str(path))
        if start_time > 0:
            self.container.seek(int(start_time * av.time_base), backward=True)
        self.iterator = iter(self.container.decode(video=0))
        self.previous = self.current = None
        self.tolerance = .5/fps + .001
        self.matched_frames = 0

    def frame_at(self, stamp):
        while self.current is None or self.current.time < stamp:
            if self.current is not None:
                self.previous = self.current
            try:
                self.current = next(self.iterator)
            except StopIteration:
                self.current = None
                break
        candidates = [frame for frame in (self.previous, self.current) if frame is not None]
        if not candidates:
            raise ValueError('Empty camera video: ' + str(self.path))
        frame = min(candidates, key=lambda f: abs(f.time-stamp))
        if abs(frame.time-stamp) > self.tolerance:
            raise ValueError('Camera frame missing near %.6fs: %s' % (stamp, self.path))
        self.matched_frames += 1
        return frame.to_ndarray(format='rgb24')

    def close(self):
        self.container.close()
