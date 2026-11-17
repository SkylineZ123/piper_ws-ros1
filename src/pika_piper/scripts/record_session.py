#!/usr/bin/env python3
"""Collect Pika + Piper episodes with keyboard controls and a live status view."""

import argparse
import hashlib
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rospy
from utils.collection_config import load_settings
from utils.configuration import load_config, package_file
from utils.recorder_ipc import WriterClient
from utils.recorder_node import Recorder
from utils.reproducibility import runtime_snapshot
from utils.status_view import StatusView, TerminalKeys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', help='Pika/Piper pipeline YAML, or a collection YAML with default pipeline'
    )
    parser.add_argument('--collection-config', help='LeRobot acquisition YAML')
    parser.add_argument(
        '--output', help='Exact dataset directory; default creates a unique dataset'
    )
    parser.add_argument('--repo-id')
    parser.add_argument('--action-type', choices=['joint', 'eef6d'])
    parser.add_argument('--task')
    parser.add_argument(
        '--resume', action='store_true', help='Append compatible episodes without overwriting'
    )
    parser.add_argument('--no-view', action='store_true')
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error(
            'Use your Python 3.12 environment containing LeRobot; current interpreter: '
            + sys.executable
        )
    args = parser.parse_args(rospy.myargv()[1:])
    cfg, root = load_settings(args)
    rospy.init_node('pika_recorder', disable_signals=True)
    controller = rospy.get_param('/pika_piper/controller', {})
    if controller:
        actual = load_config(controller.get('config', args.config), controller)
        for key in (
            'arm',
            'tcp_xyzrpy',
            'axis_rpy',
            'pika_to_tcp_xyzrpy',
            'control_hz',
            'translation_scale',
            'max_joint_speed',
            'max_gripper_speed',
        ):
            cfg[key] = actual[key]
    desc = package_file('urdf/piper_description.urdf', package='piper_description')
    contract_keys = (
        'fps',
        'cameras',
        'image_height',
        'image_width',
        'subtasks',
        'use_back_event',
        'video_codec',
        'task',
        'camera_tolerance',
        'state_gap',
        'command_age',
        'axis_rpy',
        'pika_to_tcp_xyzrpy',
        'translation_scale',
    )
    metadata = dict(
        schema_version=1,
        action_type=cfg['action_type'],
        runtime=runtime_snapshot(),
        units={'length': 'm', 'angle': 'rad'},
        rotation='xyz_rpy',
        reference='base_link_absolute',
        arm=cfg['arm'],
        tcp_xyzrpy=cfg['tcp_xyzrpy'],
        urdf_sha256=hashlib.sha256(desc.read_bytes()).hexdigest(),
        collection_contract={k: cfg[k] for k in contract_keys},
        config=cfg,
        timestamp_policy='frame_index/fps; source_timestamp retains pauses; '
        'segment_id changes on every c',
        alignment={
            'fps': cfg['fps'],
            'camera_tolerance': cfg['camera_tolerance'],
            'state_gap': cfg['state_gap'],
            'command_age': cfg['command_age'],
        },
    )
    writer, recorder, view = None, None, None
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    try:
        print('Collection and LeRobot Python: ' + sys.executable, flush=True)
        writer = WriterClient(root, cfg['repo_id'], cfg, metadata, args.resume)
        recorder = Recorder(cfg, writer)
        if not args.no_view and cfg['enable_view']:
            view = StatusView(cfg['cameras'], cfg['view_max_width']).start()
        print(
            'Dataset: %s\nc start/resume | SPACE/x pause | s save | r discard | '
            '1..9 subtask (paused) | e back-event | t teleop (paused) | q exit\n'
            'Keys work immediately in this terminal or the status window; no Enter.\n'
            'q/Ctrl-C discards the UNSAVED episode; saved episodes are retained.' % root,
            flush=True,
        )
        if (view is None or not view.enabled) and not sys.stdin.isatty():
            raise RuntimeError(
                'No keyboard input: use an interactive terminal, or enable StatusView'
            )
        with TerminalKeys() as keyboard:
            while (
                not stopping.is_set() and not rospy.is_shutdown() and not recorder.finish.is_set()
            ):
                keys = keyboard.poll()
                with recorder.state_lock:
                    status = dict(recorder.snapshot)
                if view is not None:
                    key = view.render_once(recorder.streams.preview(), status)
                    if key:
                        keys.append(key)
                for key in keys:
                    recorder.request(key)
                time.sleep(1 / 30)
        if recorder.error:
            raise RuntimeError(str(recorder.error))
    finally:
        try:
            if recorder is not None:
                recorder.stop()
            if writer is not None:
                try:
                    writer.call('close')
                except Exception as exc:
                    rospy.logerr('Writer shutdown: %s', exc)
        finally:
            if view is not None:
                view.stop()
            if writer is not None:
                writer.close()
            rospy.signal_shutdown('collector finished')


if __name__ == '__main__':
    main()
