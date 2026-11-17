"""Legacy bag inspection helpers; the current recorder writes LeRobot directly."""

import json
import tempfile
from pathlib import Path

import numpy as np

from .core import align_series
from .images import decode_image


class CachedImage:
    def __init__(self, path, offset, shape):
        self.path = path
        self.offset = offset
        self.shape = shape

    def array(self):
        with open(self.path, 'rb') as f:
            f.seek(self.offset)
            return (
                np.frombuffer(f.read(int(np.prod(self.shape))), dtype=np.uint8)
                .reshape(self.shape)
                .copy()
            )


NUMERIC = {
    '/pika_piper/feedback': 'state',
    '/pika_piper/eef_state': 'eef_state',
    '/pika_piper/command_sent': 'joint_command',
    '/pika_piper/eef_command': 'eef_command',
}


def load_session(root):
    from rosbags.highlevel import AnyReader

    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Incomplete session; inspect raw bag before conversion')
    cache = tempfile.TemporaryDirectory(prefix='pika-images-')
    manifest['_cache'] = cache
    image_path = Path(cache.name) / 'images.rgb'
    cfg = manifest['config']
    images = {topic: name for name, topic in cfg['cameras'].items()}
    streams = {name: [] for name in list(NUMERIC.values()) + list(images.values())}
    with AnyReader([root / 'session.bag']) as reader, image_path.open('wb') as image_file:
        connections = [c for c in reader.connections if c.topic in NUMERIC or c.topic in images]
        for conn, arrival, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if stamp <= 0:
                continue
            name = NUMERIC.get(conn.topic, images.get(conn.topic))
            if conn.topic in NUMERIC:
                value = np.asarray(msg.position, dtype=np.float64)
                if value.shape != (7,) or not np.isfinite(value).all():
                    continue
            else:
                image = decode_image(msg)
                offset = image_file.tell()
                image_file.write(image.tobytes())
                value = CachedImage(image_path, offset, image.shape)
            streams[name].append((stamp, value))
    for name, pairs in streams.items():
        # Stable de-duplication, retaining the latest received sample at each stamp.
        values = dict(pairs)
        times = sorted(values)
        streams[name] = (np.array(times), [values[t] for t in times])
    return manifest, streams


def aligned_episodes(manifest, streams):
    cfg = manifest['config']
    dt = 1 / cfg['fps']
    for segment in manifest['segments']:
        if not segment['accepted'] or segment['end'] is None:
            continue
        frames = []
        count = max(0, int(np.ceil((segment['end'] - segment['start']) / dt - 1e-7)))
        for index in range(count):
            t = segment['start'] + index * dt
            frame = {'source_timestamp': float(t), 'segment_id': segment['id']}
            valid = True
            for name, series in streams.items():
                if name in cfg['cameras']:
                    kind = 'image'
                    tol = cfg.get('camera_tolerance', 0.02)
                elif name.endswith('command'):
                    kind = 'command'
                    tol = cfg.get('command_age', 0.1)
                else:
                    kind = 'eef' if name == 'eef_state' else 'state'
                    tol = cfg.get('state_gap', 0.1)
                sample = align_series(series, t, kind, tol)
                if sample is None:
                    valid = False
                    break
                value, source = sample
                # A previous segment's command must not seed this one.
                if kind == 'command' and source < segment['start']:
                    valid = False
                    break
                frame[name] = value
            if valid:
                frames.append(frame)
            else:
                if len(frames) >= 2:
                    yield frames
                frames = []
        if len(frames) >= 2:
            yield frames
