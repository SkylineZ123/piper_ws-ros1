"""ROS subscriptions and acquisition worker; the session owns dataset state."""

import json
import queue
import threading

import numpy as np

from .collection import EEF_NAMES, JOINT_NAMES, NUMERIC_TOPICS, CollectionSession, LiveStreams
from .images import decode_image


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
        self.session = CollectionSession(
            self.streams,
            writer,
            cfg,
            self.teleop,
            lambda: rospy.Time.now().to_sec(),
            rospy.loginfo,
            cancelled=self.start_cancelled,
            notify=self.publish_status,
        )
        self.session._counts(writer.initial)
        self.snapshot = self.session.snapshot()
        self.subscribers = []
        for topic, name in NUMERIC_TOPICS.items():
            self.subscribers.append(
                rospy.Subscriber(
                    topic,
                    JointState,
                    self.numeric,
                    callback_args=name,
                    queue_size=100,
                    tcp_nodelay=True,
                )
            )
        for name, topic in cfg['cameras'].items():
            self.subscribers.append(
                rospy.Subscriber(
                    topic,
                    Image,
                    self.image,
                    callback_args=name,
                    queue_size=2,
                    buff_size=8 * 1024 * 1024,
                    tcp_nodelay=True,
                )
            )
        self.subscribers.append(
            rospy.Subscriber('/pika_piper/status', String, self.status, queue_size=10)
        )
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
            self.rospy.wait_for_service('/pika_piper/set_teleop', timeout=1.0)
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
            self.rospy.logwarn_throttle(5.0, 'Numeric input rejected: %s' % exc)

    def image(self, msg, name):
        try:
            image = decode_image(msg)
            h, w = self.cfg['image_height'], self.cfg['image_width']
            if image.shape != (h, w, 3):
                import cv2

                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            self.streams.put(name, msg.header.stamp.to_sec(), image)
        except Exception as exc:
            self.rospy.logwarn_throttle(5.0, 'Image input rejected: %s' % exc)

    def status(self, msg):
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict) or 'mode' not in value:
                raise ValueError('Expected controller status object')
            self.streams.set_controller(value)
        except (ValueError, TypeError) as exc:
            self.rospy.logwarn_throttle(5.0, 'Status input rejected: %s' % exc)

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
            while (
                not self.finish.is_set() and not self.rospy.is_shutdown() and not self.session.quit
            ):
                try:
                    key = self.commands.get(timeout=0.005)
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
        self.thread.join(timeout=190.0)
        if self.thread.is_alive():
            raise RuntimeError('Acquisition thread did not stop; writer was unresponsive')
        for sub in self.subscribers:
            sub.unregister()
