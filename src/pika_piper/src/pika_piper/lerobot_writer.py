"""LeRobot-only child process. No rospy, cv_bridge, CAN or device access."""
import argparse
import fcntl
import inspect
import json
from pathlib import Path
import socket
import sys
import tempfile
import traceback

import numpy as np

from .collection import JOINT_NAMES, EEF_NAMES
from .recorder_ipc import receive_packet, send_packet


def make_features(cfg):
    def numeric(names):
        return {'dtype': 'float32', 'shape': (7,), 'names': names}
    features = {
        'observation.state': numeric(JOINT_NAMES),
        'observation.eef_state': numeric(EEF_NAMES),
        'action': numeric(JOINT_NAMES if cfg['action_type'] == 'joint' else EEF_NAMES),
        'joint_command': numeric(JOINT_NAMES), 'eef_command': numeric(EEF_NAMES),
        'source_timestamp': {'dtype': 'float64', 'shape': (1,), 'names': ['seconds']},
        'segment_id': {'dtype': 'int64', 'shape': (1,), 'names': ['segment']},
    }
    for name in cfg['cameras']:
        features['observation.images.' + name] = {
            'dtype': 'video', 'shape': (cfg['image_height'], cfg['image_width'], 3),
            'names': ['height', 'width', 'channels']}
    if cfg['subtasks']:
        features['subtask_index'] = {'dtype': 'int64', 'shape': (1,), 'names': None}
    if cfg['use_back_event']:
        features['back_event'] = {'dtype': 'int64', 'shape': (1,), 'names': None}
    return features


class DatasetWriter:
    """Finalize saved episodes; support LeRobot v3 APIs in 0.5 and 0.6."""

    def __init__(self, root, repo_id, config, metadata, resume=False):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.api = LeRobotDataset
        self.root, self.repo_id, self.cfg = Path(root), repo_id, config
        self.ds = None
        self.frames = self.episodes = self.total_frames = 0
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.root.with_name('.' + self.root.name + '.record.lock').open('a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.options = dict(repo_id=repo_id, root=self.root,
                            streaming_encoding=config['streaming_encoding'],
                            encoder_threads=config['encoder_threads'],
                            encoder_queue_maxsize=config['encoder_queue_maxsize'])
        if 'rgb_encoder' in inspect.signature(LeRobotDataset.create).parameters:
            from lerobot.configs.video import RGBEncoderConfig
            self.options['rgb_encoder'] = RGBEncoderConfig(vcodec=config['video_codec'])
        else:
            self.options['vcodec'] = config['video_codec']
        self.features = make_features(config)
        meta_path = self.root / 'meta/pika_pipeline.json'
        if self.root.exists():
            if not resume:
                raise FileExistsError('Output already exists; choose a new path or use --resume')
            if not meta_path.is_file():
                raise ValueError('Existing dataset has no pika_pipeline metadata; cannot append')
            existing = json.loads(meta_path.read_text())
            # Changing these would make previously saved episodes ambiguous.
            for key in ('schema_version', 'action_type', 'arm', 'tcp_xyzrpy', 'urdf_sha256',
                        'units', 'rotation', 'reference', 'collection_contract'):
                if existing.get(key) != metadata.get(key):
                    raise ValueError('Resume metadata differs: ' + key)
            self.ds = self._resume()
            self._validate_features()
            if config['subtasks']:
                import pandas as pd
                table = pd.read_parquet(self.root / 'meta/subtasks.parquet')
                if (table.index.tolist() != list(config['subtasks'].values()) or
                        table['subtask_index'].tolist() != list(range(len(config['subtasks'])))):
                    raise ValueError('Resume subtask order differs')
        else:
            if resume:
                raise FileNotFoundError('--resume requires an existing dataset')
            self.ds = LeRobotDataset.create(
                fps=config['fps'], robot_type='single_piper_' + config['arm'],
                features=self.features, metadata_buffer_size=1, video_backend='pyav', **self.options)
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
            if config['subtasks']:
                import pandas as pd
                names = list(config['subtasks'].values())
                pd.DataFrame({'subtask_index': range(len(names))},
                             index=pd.Index(names, name='subtask')).to_parquet(
                                 self.root / 'meta/subtasks.parquet')
        self.episodes = self.ds.meta.total_episodes
        self.total_frames = self.ds.meta.total_frames

    def _resume(self):
        if hasattr(self.api, 'resume'):
            return self.api.resume(video_backend='pyav', **self.options)

        from lerobot.datasets.utils import load_info
        info = load_info(self.root)
        if info['total_episodes'] == 0 and info['total_frames'] == 0:
            # v0.5 cannot load a zero-episode dataset: metadata/data tables do
            # not exist yet and its constructor tries the Hub. Initialize an
            # empty writer via public create() in a temporary directory, then
            # attach it to the existing, validated metadata/root. The existing
            # directory and info.json are never recreated or overwritten.
            with tempfile.TemporaryDirectory(prefix='pika-empty-writer-') as temporary:
                options = dict(self.options, root=Path(temporary) / 'dataset')
                dataset = self.api.create(
                    fps=info['fps'], robot_type=info['robot_type'], features=self.features,
                    metadata_buffer_size=1, video_backend='pyav', **options)
                dataset.root = self.root
                dataset.meta.root = self.root
                dataset.meta.info = info
            return dataset

        # LeRobot 0.5 resumes saved local datasets via its constructor.
        dataset = self.api(video_backend='pyav', revision=info['codebase_version'], **self.options)
        dataset.episode_buffer = dataset.create_episode_buffer()
        return dataset

    def _validate_features(self):
        actual = self.ds.meta.features
        for key, expected in self.features.items():
            if key not in actual or any(actual[key].get(k) != expected.get(k)
                                        for k in ('dtype', 'names')):
                raise ValueError('Resume feature differs: ' + key)
            if tuple(actual[key]['shape']) != tuple(expected['shape']):
                raise ValueError('Resume shape differs: ' + key)
        if self.ds.fps != self.cfg['fps']:
            raise ValueError('Resume fps differs')
        extras = set(actual) - set(self.features) - {
            'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
        if extras:
            raise ValueError('Resume has unexpected features: ' + str(extras))

    def counts(self):
        return dict(buffered_frames=self.frames, saved_episodes=self.episodes,
                    saved_frames=self.total_frames)

    def append(self, frame):
        self.prepare()
        row = {'task': self.cfg['task']}
        for target, source in [('observation.state', 'state'),
                               ('observation.eef_state', 'eef_state'),
                               ('joint_command', 'joint_command'), ('eef_command', 'eef_command')]:
            array = np.asarray(frame[source], dtype=np.float32)
            if array.shape != (7,) or not np.isfinite(array).all():
                raise ValueError('Invalid seven dimensional ' + source)
            row[target] = array
        row['action'] = row['joint_command' if self.cfg['action_type'] == 'joint' else 'eef_command'].copy()
        row['source_timestamp'] = np.array([frame['source_timestamp']], np.float64)
        row['segment_id'] = np.array([frame['segment_id']], np.int64)
        for name in self.cfg['cameras']:
            row['observation.images.' + name] = np.asarray(frame[name], dtype=np.uint8)
        for name, enabled in [('subtask_index', bool(self.cfg['subtasks'])),
                              ('back_event', self.cfg['use_back_event'])]:
            if enabled:
                row[name] = np.array([frame[name]], np.int64)
        self.ds.add_frame(row)
        # LeRobot's streaming encoder may otherwise drop video but keep numeric
        # rows. Fail closed instead of accepting such a misaligned episode.
        encoder = (getattr(self.ds.writer, '_streaming_encoder', None) or
                   getattr(self.ds, '_streaming_encoder', None))
        if encoder is not None and any(getattr(encoder, '_dropped_frames', {}).values()):
            raise RuntimeError('Video encoder dropped frames; pending episode will be discarded. '
                               'Reduce resolution/fps or disable streaming_encoding.')
        self.frames += 1

    def prepare(self):
        if self.ds is None:
            self.ds = self._resume()

    def save(self):
        if self.frames < 2:
            raise ValueError('At least two frames are required')
        self.ds.save_episode()
        # Re-open a fresh writer for the next episode rather than reopening a
        # closed ParquetWriter on the same shard and truncating earlier rows.
        self.ds.finalize()
        self.episodes = self.ds.meta.total_episodes
        self.total_frames = self.ds.meta.total_frames
        self.ds = None
        self.frames = 0

    def discard(self):
        if self.ds is not None:
            if hasattr(self.ds, 'has_pending_frames'):
                pending = self.ds.has_pending_frames()
            else:
                pending = bool((getattr(self.ds, 'episode_buffer', None) or {}).get('size', 0))
            if pending:
                self.ds.clear_episode_buffer()
        self.frames = 0

    def close(self):
        try:
            self.discard()
            if self.ds is not None:
                self.ds.finalize()
                self.ds = None
        finally:
            self.lock_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    args = parser.parse_args()
    sock = socket.socket(fileno=args.fd)
    writer = None
    try:
        while True:
            message = receive_packet(sock)
            op = message.pop('op')
            if op == 'init':
                writer = DatasetWriter(**message)
            elif writer is None:
                raise RuntimeError('Writer is not initialized')
            elif op == 'frame':
                writer.append(message)
            elif op == 'prepare':
                writer.prepare()
            elif op == 'save':
                writer.save()
            elif op == 'discard':
                writer.discard()
            elif op == 'close':
                writer.close()
                writer = None
                send_packet(sock, {'ok': True})
                break
            else:
                raise ValueError('Unknown operation: ' + op)
            send_packet(sock, dict(ok=True, python_executable=sys.executable, **writer.counts()))
    except EOFError:
        pass
    except Exception as exc:
        traceback.print_exc()
        try:
            send_packet(sock, {'ok': False, 'error': str(exc)})
        except OSError:
            pass
    finally:
        if writer is not None:
            writer.close()
        sock.close()


if __name__ == '__main__':
    main()
