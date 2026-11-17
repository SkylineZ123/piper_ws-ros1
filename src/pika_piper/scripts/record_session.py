#!/usr/bin/env python3
"""Pika + Piper single-key acquisition, live StatusView and LeRobot v3 output.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import yaml
import rospy
import rospkg
from pika_piper.collection import CollectionSession, LiveStreams, NUMERIC_TOPICS, JOINT_NAMES, EEF_NAMES
from pika_piper.configuration import load_config, package_file
from pika_piper.dataset import decode_image
from pika_piper.recorder_ipc import WriterClient
from pika_piper.status_view import StatusView, TerminalKeys


def load_settings(args):
    pipeline_path = args.config
    collection_path = args.collection_config
    supplied = load_config(Path(pipeline_path).expanduser()) if pipeline_path else None
    if (supplied is not None and 'arm' not in supplied and
            any(key in supplied for key in ('lerobot_python', 'action_type', 'image_height'))):
        if collection_path:
            raise ValueError('--config already specifies a collection YAML; use '
                             '--config pipeline.yaml --collection-config collection.yaml')
        # Accept the common single-config invocation, including IDE launches.
        collection_path, pipeline_path = pipeline_path, None
        supplied = None
    pipeline_path = Path(pipeline_path or package_file('config/pipeline.yaml')).expanduser().resolve()
    collection_path = Path(collection_path or package_file('config/collection.yaml')).expanduser().resolve()
    pipeline = supplied if supplied is not None else load_config(pipeline_path)
    settings = yaml.safe_load(collection_path.read_text())
    if not isinstance(settings, dict):
        raise ValueError('Collection config must be a YAML mapping')
    # Later controller-param merging must use the resolved PIPELINE path.
    args.config, args.collection_config = str(pipeline_path), str(collection_path)
    cfg = dict(pipeline, **settings)
    # Ignore the obsolete setting in older YAMLs; the launching interpreter
    # is the sole authority for both acquisition and encoding.
    cfg.pop('lerobot_python', None)
    for name in ('repo_id', 'action_type', 'task'):
        if getattr(args, name, None) is not None:
            cfg[name] = getattr(args, name)
    if cfg['action_type'] not in ('joint', 'eef6d'):
        raise ValueError('action_type must be joint or eef6d')
    for name in ('fps', 'image_height', 'image_width', 'max_timesteps',
                 'encoder_threads', 'encoder_queue_maxsize', 'view_max_width'):
        if isinstance(cfg[name], bool) or not isinstance(cfg[name], int) or cfg[name] <= 0:
            raise ValueError(name + ' must be a positive integer')
    if cfg['image_height'] % 2 or cfg['image_width'] % 2:
        raise ValueError('Video image dimensions must be even')
    for name in ('alignment_delay', 'alignment_timeout', 'status_timeout', 'source_timeout',
                 'camera_tolerance', 'state_gap', 'command_age'):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(name + ' must be finite and positive')
    if not cfg['cameras'] or len(set(cfg['cameras'].values())) != len(cfg['cameras']):
        raise ValueError('Configure at least one camera and distinct camera topics')
    cfg['subtasks'] = {str(k): str(v) for k, v in cfg.get('subtasks', {}).items()}
    if any(len(k) != 1 or k not in '123456789' for k in cfg['subtasks']):
        raise ValueError('Subtask keys must be single digits 1..9')
    if len(set(cfg['subtasks'].values())) != len(cfg['subtasks']):
        raise ValueError('Subtask names must be unique')
    if args.resume and not args.output:
        raise ValueError('--resume requires --output pointing to an existing dataset')
    if args.output:
        root = Path(args.output).expanduser().resolve()
    else:
        name = 'dataset_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        root = (Path(cfg['output_dir']).expanduser() / name).resolve()
    return cfg, root


class Recorder:
    def __init__(self, cfg, writer):
        import rospy
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import String
        from std_srvs.srv import SetBool
        self.rospy, self.cfg = rospy, cfg
        self.streams = LiveStreams(cfg)
        self.commands = queue.Queue(maxsize=64)
        self.finish = threading.Event()
        self.cancel = threading.Event()
        self.state_lock = threading.Lock()
        self.service_lock = threading.Lock()
        self.error = None
        self.teleop_proxy = rospy.ServiceProxy('/pika_piper/set_teleop', SetBool)
        self.session = CollectionSession(self.streams, writer, cfg, self.teleop,
                                         lambda: rospy.Time.now().to_sec(), rospy.loginfo,
                                         cancelled=self.start_cancelled, notify=self.publish_status)
        self.session._counts(writer.initial)
        self.snapshot = self.session.snapshot()
        self.subscribers = []
        for topic, name in NUMERIC_TOPICS.items():
            self.subscribers.append(rospy.Subscriber(topic, JointState, self.numeric,
                                                     callback_args=name, queue_size=100, tcp_nodelay=True))
        for name, topic in cfg['cameras'].items():
            self.subscribers.append(rospy.Subscriber(topic, Image, self.image,
                                                     callback_args=name, queue_size=2,
                                                     buff_size=8 * 1024 * 1024, tcp_nodelay=True))
        self.subscribers.append(rospy.Subscriber('/pika_piper/status', String, self.status, queue_size=10))
        self.thread = threading.Thread(target=self.acquire, name='pika-acquisition', daemon=True)
        self.thread.start()

    def publish_status(self, status):
        with self.state_lock:
            self.snapshot = status

    def start_cancelled(self):
        if self.cancel.is_set() or self.finish.is_set():
            return True
        with self.commands.mutex:
            return any(k in (' ', 'x', 's', 'r', 'q') for k in self.commands.queue)

    def teleop(self, enabled):
        with self.service_lock:
            self.rospy.wait_for_service('/pika_piper/set_teleop', timeout=1.)
            response = self.teleop_proxy(enabled)
            return response.success, response.message

    def numeric(self, msg, name):
        try:
            names = EEF_NAMES if name in ('eef_state', 'eef_command') else JOINT_NAMES
            if len(msg.name) != 7 or set(msg.name) != set(names) or len(msg.position) != 7:
                raise ValueError('Expected named seven dimensional ' + name)
            values = np.asarray([msg.position[msg.name.index(n)] for n in names], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite ' + name)
            self.streams.put(name, msg.header.stamp.to_sec(), values)
        except Exception as exc:
            self.rospy.logwarn_throttle(5., 'Numeric input rejected: %s' % exc)

    def image(self, msg, name):
        try:
            image = decode_image(msg)
            h, w = self.cfg['image_height'], self.cfg['image_width']
            if image.shape != (h, w, 3):
                import cv2
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            self.streams.put(name, msg.header.stamp.to_sec(), image)
        except Exception as exc:
            self.rospy.logwarn_throttle(5., 'Image input rejected: %s' % exc)

    def status(self, msg):
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict) or 'mode' not in value:
                raise ValueError('Expected controller status object')
            self.streams.set_controller(value)
        except (ValueError, TypeError) as exc:
            self.rospy.logwarn_throttle(5., 'Status input rejected: %s' % exc)

    def request(self, key):
        if key not in 'c xsretq' and key not in self.cfg['subtasks']:
            return
        # Stop motion even when the acquisition thread is busy encoding/saving.
        # Only that thread mutates the session and dataset.
        if key in (' ', 'x', 's', 'r', 'q'):
            self.cancel.set()
            try:
                self.teleop(False)
            except Exception as exc:
                self.rospy.logerr('Could not stop teleop: %s', exc)
        try:
            self.commands.put_nowait(key)
        except queue.Full:
            self.cancel.set()
            self.rospy.logerr('Command queue full; acquisition will pause')

    def acquire(self):
        try:
            while not self.finish.is_set() and not self.rospy.is_shutdown() and not self.session.quit:
                try:
                    key = self.commands.get(timeout=.005)
                except queue.Empty:
                    key = None
                if self.cancel.is_set():
                    self.session.pause('Paused by operator')
                    self.cancel.clear()
                if key is not None:
                    self.session.handle(key)
                if self.commands.empty() and not self.cancel.is_set():
                    self.session.step()
                self.publish_status(self.session.snapshot())
        except Exception as exc:
            self.error = exc
            self.rospy.logerr('Acquisition failed: %s', exc)
        finally:
            self.session.pause('Collector stopped')
            self.publish_status(self.session.snapshot())
            self.finish.set()

    def stop(self):
        self.finish.set()
        try:
            self.teleop(False)
        except Exception as exc:
            self.rospy.logwarn('Stop service unavailable: %s', exc)
        self.thread.join(timeout=190.)
        if self.thread.is_alive():
            raise RuntimeError('Acquisition thread did not stop; writer was unresponsive')
        for sub in self.subscribers:
            sub.unregister()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Pika/Piper pipeline YAML, or a collection YAML with default pipeline')
    parser.add_argument('--collection-config', help='LeRobot acquisition YAML')
    parser.add_argument('--output', help='Exact dataset directory; default creates a unique dataset')
    parser.add_argument('--repo-id')
    parser.add_argument('--action-type', choices=['joint', 'eef6d'])
    parser.add_argument('--task')
    parser.add_argument('--resume', action='store_true', help='Append compatible episodes without overwriting')
    parser.add_argument('--no-view', action='store_true')
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Use your Python 3.12 environment containing LeRobot; current interpreter: '
                     + sys.executable)
    args = parser.parse_args(rospy.myargv()[1:])
    cfg, root = load_settings(args)
    rospy.init_node('pika_recorder', disable_signals=True)
    controller = rospy.get_param('/pika_piper/controller', {})
    if controller:
        actual = load_config(controller.get('config', args.config), controller)
        for key in ('arm', 'tcp_xyzrpy', 'axis_rpy', 'pika_to_tcp_xyzrpy', 'control_hz',
                    'translation_scale', 'max_joint_speed', 'max_gripper_speed'):
            cfg[key] = actual[key]
    desc = package_file('urdf/piper_description.urdf', package='piper_description')
    contract_keys = ('fps', 'cameras', 'image_height', 'image_width', 'subtasks', 'use_back_event',
                     'video_codec', 'task', 'camera_tolerance', 'state_gap', 'command_age',
                     'axis_rpy', 'pika_to_tcp_xyzrpy', 'translation_scale')
    metadata = dict(schema_version=1, action_type=cfg['action_type'],
                    units={'length': 'm', 'angle': 'rad'}, rotation='xyz_rpy',
                    reference='base_link_absolute', arm=cfg['arm'], tcp_xyzrpy=cfg['tcp_xyzrpy'],
                    urdf_sha256=hashlib.sha256(desc.read_bytes()).hexdigest(),
                    collection_contract={k: cfg[k] for k in contract_keys}, config=cfg,
                    timestamp_policy='frame_index/fps; source_timestamp retains pauses; '
                                     'segment_id changes on every c',
                    alignment={'fps': cfg['fps'], 'camera_tolerance': cfg['camera_tolerance'],
                               'state_gap': cfg['state_gap'], 'command_age': cfg['command_age']})
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
        print('Dataset: %s\nc start/resume | SPACE/x pause | s save | r discard | '
              '1..9 subtask (paused) | e back-event | t teleop (paused) | q exit\n'
              'Keys work immediately in this terminal or the status window; no Enter.\n'
              'q/Ctrl-C discards the UNSAVED episode; saved episodes are retained.' % root, flush=True)
        if (view is None or not view.enabled) and not sys.stdin.isatty():
            raise RuntimeError('No keyboard input: use an interactive terminal, or enable StatusView')
        with TerminalKeys() as keyboard:
            while not stopping.is_set() and not rospy.is_shutdown() and not recorder.finish.is_set():
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
