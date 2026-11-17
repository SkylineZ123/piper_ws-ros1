#!/usr/bin/env python3
"""Read-only ROS timing report; never claims control or publishes commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import collections
import json
import threading
import time

import numpy as np
import rospy
from data_msgs.msg import Gripper, LocalizationStatus
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--output', help='Optional JSON report path')
    args = parser.parse_args(rospy.myargv()[1:])
    if not 0 < args.duration <= 300:
        parser.error('duration must be in (0, 300] seconds')
    rospy.init_node('pika_timing_probe', anonymous=True, disable_rosout=True)
    records = collections.defaultdict(list)
    lock = threading.Lock()

    def receive(msg, topic):
        item = {'received': time.monotonic()}
        if hasattr(msg, 'header'):
            item['age_s'] = (rospy.Time.now() - msg.header.stamp).to_sec()
        if isinstance(msg, String):
            try:
                item['status'] = json.loads(msg.data)
            except ValueError:
                pass
        elif isinstance(msg, Int32):
            item['command'] = msg.data
        elif isinstance(msg, LocalizationStatus):
            item['accurate'] = bool(msg.accurate)
        elif isinstance(msg, JointState):
            item['position'] = list(msg.position)
        with lock:
            records[topic].append(item)

    topics = {
        '/pika_pose': PoseStamped,
        '/gripper/data': Gripper,
        '/pika_localization_status': LocalizationStatus,
        '/pika/gesture_command': Int32,
        '/pika_piper/feedback': JointState,
        '/pika_piper/eef_requested': PoseStamped,
        '/pika_piper/eef_feasible': PoseStamped,
        '/pika_piper/sense_relative': PoseStamped,
        '/pika_piper/command': JointState,
        '/pika_piper/command_sent': JointState,
        '/pika_piper/status': String,
    }
    subscribers = [
        rospy.Subscriber(
            topic, kind, receive, callback_args=topic, queue_size=100, tcp_nodelay=True
        )
        for topic, kind in topics.items()
    ]
    started = time.monotonic()
    while time.monotonic() - started < args.duration and not rospy.is_shutdown():
        time.sleep(0.02)
    for subscriber in subscribers:
        subscriber.unregister()

    def distribution(values):
        return {
            'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)),
            'max': float(max(values)),
        }

    report = {
        'read_only': True,
        'sample_seconds': time.monotonic() - started,
        'topics': {},
        'status': {},
    }
    with lock:
        for topic in topics:
            values = records[topic]
            result = {'count': len(values)}
            if len(values) > 1:
                stamps = np.array([v['received'] for v in values])
                result.update(
                    receive_hz=float((len(stamps) - 1) / (stamps[-1] - stamps[0])),
                    interval_ms=distribution(np.diff(stamps) * 1000),
                )
            ages = [v['age_s'] * 1000 for v in values if 'age_s' in v]
            if ages:
                result['stamp_age_ms'] = distribution(ages)
            if values and 'accurate' in values[-1]:
                result['accurate_fraction'] = float(np.mean([v['accurate'] for v in values]))
            if values and 'command' in values[-1]:
                result['events'] = [v['command'] for v in values]
            if values and 'position' in values[-1]:
                result['last_position'] = values[-1]['position']
            report['topics'][topic] = result
        states = [v['status'] for v in records['/pika_piper/status'] if 'status' in v]
        if states:
            report['status'] = {
                'last': states[-1],
                'modes': dict(collections.Counter(s['mode'] for s in states)),
                'joint_limited_fraction': float(
                    np.mean([s.get('joint_limited', False) for s in states])
                ),
                'target_limited_fraction': float(
                    np.mean([s.get('target_limited', False) for s in states])
                ),
            }
            for key in ('tick_ms', 'hand_age_s', 'echo_age_s'):
                values = [
                    s['timing'][key] for s in states if s.get('timing', {}).get(key) is not None
                ]
                if values:
                    report['status'][key] = distribution(values)
            active = [s for s in states if s['mode'] == 'teleop' and s.get('tracking')]
            report['status']['tracking_states'] = dict(
                collections.Counter(s['tracking']['state'] for s in active)
            )
            for key in (
                'raw_position_error_m',
                'raw_orientation_error_rad',
                'position_error_m',
                'orientation_error_rad',
                'solve_ms',
                'feedback_target_position_error_m',
                'feedback_target_orientation_error_rad',
                'command_target_position_error_m',
                'command_target_orientation_error_rad',
                'feedback_command_position_error_m',
                'feedback_command_orientation_error_rad',
                'command_feedback_error_rad',
                'lag_duration_s',
                'stall_duration_s',
            ):
                values = [s['tracking'][key] for s in active if s['tracking'].get(key) is not None]
                if values:
                    report['status']['tracking_' + key] = distribution(values)
            if active:
                report['status']['ik_budget_exceeded_fraction'] = float(
                    np.mean([s['tracking'].get('budget_exceeded', False) for s in active])
                )
    payload = json.dumps(report, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(payload)
    print(payload, end='')
    rospy.signal_shutdown('timing measurement complete')


if __name__ == '__main__':
    main()
