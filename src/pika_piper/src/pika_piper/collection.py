"""ROS-independent live alignment and operator state machine (SI units)."""
import bisect
import threading
import time

import numpy as np

from .core import align_series

JOINT_NAMES = ['joint%d' % i for i in range(1, 7)] + ['gripper']
EEF_NAMES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
NUMERIC_TOPICS = {
    '/pika_piper/feedback': 'state',
    '/pika_piper/eef_state': 'eef_state',
    '/pika_piper/command_sent': 'joint_command',
    '/pika_piper/eef_command': 'eef_command',
}


class LiveStreams:
    """Bounded, sorted source-time buffers; callbacks never touch the dataset."""

    def __init__(self, pipeline, camera_capacity=80):
        self.cfg = pipeline
        self.lock = threading.RLock()
        self.series = {name: ([], []) for name in
                       list(NUMERIC_TOPICS.values()) + list(pipeline['cameras'])}
        self.capacity = {name: camera_capacity if name in pipeline['cameras'] else 1024
                         for name in self.series}
        self.controller = {'mode': 'unknown', 'reason': 'waiting for controller'}
        self.controller_at = 0.
        self.stop_sequence = 0

    def put(self, name, stamp, value):
        if not np.isfinite(stamp) or stamp <= 0:
            return
        with self.lock:
            stamps, values = self.series[name]
            i = bisect.bisect_left(stamps, stamp)
            if i < len(stamps) and stamps[i] == stamp:
                values[i] = value
            else:
                stamps.insert(i, stamp)
                values.insert(i, value)
            excess = len(stamps) - self.capacity[name]
            if excess > 0:
                del stamps[:excess]
                del values[:excess]

    def set_controller(self, status):
        with self.lock:
            if status.get('mode') != 'teleop' and self.controller.get('mode') == 'teleop':
                self.stop_sequence += 1
            self.controller = dict(status)
            self.controller_at = time.monotonic()

    def control_status(self):
        with self.lock:
            return dict(self.controller), self.controller_at, self.stop_sequence

    def preview(self):
        with self.lock:
            return {name: self.series[name][1][-1] for name in self.cfg['cameras']
                    if self.series[name][1]}

    def ready(self, now, timeout):
        with self.lock:
            for name in ['state', 'eef_state'] + list(self.cfg['cameras']):
                times = self.series[name][0]
                if not times or not -.1 <= now - times[-1] <= timeout:
                    return False, 'waiting for fresh ' + name
        return True, ''

    def align(self, stamp, start):
        result = {}
        with self.lock:
            for name, series in self.series.items():
                if name in self.cfg['cameras']:
                    kind, tolerance = 'image', self.cfg['camera_tolerance']
                elif name.endswith('command'):
                    kind, tolerance = 'command', self.cfg['command_age']
                else:
                    kind = 'eef' if name == 'eef_state' else 'state'
                    tolerance = self.cfg['state_gap']
                sample = align_series(series, stamp, kind, tolerance)
                if sample is None or (kind == 'command' and sample[1] < start):
                    return None, 'unaligned ' + name
                result[name] = sample[0]
        return result, ''

    def first_command_time(self, start):
        """Wait for THIS span's first sent command before anchoring its grid."""
        with self.lock:
            first = []
            for name in ('joint_command', 'eef_command'):
                stamps = self.series[name][0]
                index = bisect.bisect_left(stamps, start)
                if index == len(stamps):
                    return None
                first.append(stamps[index])
            return max(first)


class CollectionSession:
    """Single-owner state machine. Only the acquisition thread calls methods.

    A pause preserves the episode; the next c starts a new source-time span.
    LeRobot timestamps remain frame_index/fps, and source_timestamp + segment_id
    retain the physical discontinuity. No frame grid is skipped on missing data.
    """

    def __init__(self, streams, writer, config, teleop, now, log=print,
                 cancelled=lambda: False, notify=lambda _: None):
        self.streams, self.writer, self.cfg = streams, writer, config
        self.teleop, self.now, self.log = teleop, now, log
        self.cancelled, self.notify = cancelled, notify
        self.collecting = False
        self.quit = False
        self.back = False
        self.subtasks = {str(k): str(v) for k, v in config.get('subtasks', {}).items()}
        self.subtask = next(iter(self.subtasks), None)
        self.segment_id = -1
        self.frames = 0
        self.saved_episodes = 0
        self.saved_frames = 0
        self.message = 'Ready: c to start'
        self.measured_fps = 0.
        self._last_frame_wall = None
        self._last_clock = None
        self._last_source = None

    def _counts(self, reply):
        self.frames = reply['buffered_frames']
        self.saved_episodes = reply['saved_episodes']
        self.saved_frames = reply['saved_frames']

    def pause(self, reason='Paused'):
        self.collecting = False
        self.back = False
        try:
            success, message = self.teleop(False)
            if not success:
                reason += '; stop rejected: ' + message
        except Exception as exc:
            reason += '; stop service error: ' + str(exc)
        self.message = reason
        self.log(reason)
        self.notify(self.snapshot())

    def handle(self, key):
        if key == 'c' and not self.collecting:
            if self.cancelled():
                return
            now = self.now()
            if self._last_source is not None and now <= self._last_source:
                self.message = 'ROS clock moved backwards; restart collector'
                self.log(self.message)
                return
            ready, reason = self.streams.ready(now, self.cfg['source_timeout'])
            status, received, _ = self.streams.control_status()
            if not ready or time.monotonic() - received > self.cfg['status_timeout']:
                self.message = reason or 'waiting for fresh controller status'
                self.log(self.message)
                return
            if self.frames >= self.cfg['max_timesteps']:
                self.message = 'Frame limit reached: s to save or r to discard'
                return
            # Re-open the finalized dataset while still paused, so metadata
            # loading does not consume the first span's real-time budget.
            self._counts(self.writer.call('prepare'))
            if self.cancelled():
                return
            try:
                success, reason = self.teleop(True)
            except Exception as exc:
                success, reason = False, str(exc)
            if not success:
                self.message = 'Cannot start: ' + reason
                self.log(self.message)
                return
            if self.cancelled():
                self.pause('Start cancelled by operator')
                return
            self.start = self.now()
            self.next_stamp = self.start
            self.primed = False
            self.start_wall = time.monotonic()
            self._last_clock = self.start
            self.stop_sequence = self.streams.control_status()[2]
            self.segment_id += 1
            self.back = False
            self.collecting = True
            self._last_frame_wall = None
            self.message = 'Recording'
        elif key in (' ', 'x'):
            self.pause()
        elif key in ('s', 'r'):
            self.pause('Saving...' if key == 's' else 'Discarding...')
            if key == 's' and self.frames < 2:
                self.message = 'Need at least 2 frames; c to continue or r to discard'
                return
            self._counts(self.writer.call('save' if key == 's' else 'discard'))
            self._last_source = None
            self.subtask = next(iter(self.subtasks), None)
            self.message = 'Saved' if key == 's' else 'Discarded'
            self.log(self.message)
        elif key == 'e' and self.collecting and self.cfg['use_back_event']:
            self.back = not self.back
        elif key in self.subtasks and not self.collecting:
            self.subtask = key
            self.back = False
        elif key == 't' and not self.collecting:
            status = self.streams.control_status()[0]
            enabled = status.get('mode') != 'teleop'
            if enabled and self.cancelled():
                return
            try:
                success, reason = self.teleop(enabled)
            except Exception as exc:
                success, reason = False, str(exc)
            if enabled and success and self.cancelled():
                self.pause('Teleop start cancelled by operator')
                return
            self.message = ('Teleop: ' if success else 'Teleop rejected: ') + reason
            self.log(self.message)
        elif key == 'q':
            self.pause('Exiting; unsaved frames are discarded')
            self.quit = True

    def step(self):
        if not self.collecting:
            return
        now, wall = self.now(), time.monotonic()
        status, received, stop_sequence = self.streams.control_status()
        if now < self._last_clock - 1e-6:
            self.pause('ROS clock moved backwards; restart collector')
            self.quit = True
            return
        self._last_clock = now
        if wall - received > self.cfg['status_timeout'] or stop_sequence != self.stop_sequence:
            self.pause('Controller stopped or status timed out: ' + status.get('reason', ''))
            return
        if status.get('mode') != 'teleop':
            if wall - self.start_wall > self.cfg['alignment_timeout']:
                self.pause('Controller is not in teleop: ' + status.get('reason', ''))
            return
        if not self.primed:
            first = self.streams.first_command_time(self.start)
            if first is None:
                if wall - self.start_wall > self.cfg['alignment_timeout']:
                    self.pause('Waiting for new sent commands timed out')
                return
            self.next_stamp = first
        due = self.next_stamp + self.cfg['alignment_delay']
        if now < due:
            return
        frame, reason = self.streams.align(self.next_stamp, self.start)
        if not self.primed and frame is None:
            # Commands and cameras start at unrelated phases. Before the
            # span's FIRST frame only, choose a later fully aligned origin;
            # afterwards the grid is fixed and missing data must pause it.
            self.next_stamp = max(self.next_stamp, now - self.cfg['alignment_delay'])
            frame, reason = self.streams.align(self.next_stamp, self.start)
            if frame is None:
                if now - self.start > self.cfg['alignment_delay'] + self.cfg['alignment_timeout']:
                    self.pause(reason + '; could not align first frame')
                return
            due = self.next_stamp + self.cfg['alignment_delay']
        if now - due > self.cfg['alignment_timeout']:
            self.pause('Acquisition fell behind; buffered episode retained')
            return
        if frame is None:
            if now - due > self.cfg['alignment_timeout'] / 2:
                self.pause(reason + '; buffered episode retained')
            return
        frame.update(source_timestamp=self.next_stamp, segment_id=self.segment_id,
                     subtask_index=list(self.subtasks).index(self.subtask) if self.subtask else 0,
                     back_event=int(self.back))
        self._counts(self.writer.call('frame', frame))
        self.primed = True
        self._last_source = self.next_stamp
        if self._last_frame_wall is not None:
            instantaneous = 1 / max(1e-6, wall - self._last_frame_wall)
            self.measured_fps = .8 * self.measured_fps + .2 * instantaneous
        else:
            self.measured_fps = float(self.cfg['fps'])
        self._last_frame_wall = wall
        self.next_stamp += 1 / self.cfg['fps']
        if self.frames >= self.cfg['max_timesteps']:
            self.pause('Frame limit reached; s to save, r to discard')

    def snapshot(self):
        status = self.streams.control_status()[0]
        return dict(recording=self.collecting, back=self.back,
                    task=self.cfg['task'], skill_text=self.subtasks.get(self.subtask, ''),
                    episode_index=self.saved_episodes, buffered_frames=self.frames,
                    saved_episodes=self.saved_episodes, saved_frames=self.saved_frames,
                    fps=self.measured_fps, elapsed=self.frames / self.cfg['fps'],
                    mode=status.get('mode', 'unknown'), reason=status.get('reason', ''),
                    message=self.message)
