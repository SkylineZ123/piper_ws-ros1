"""Isolate encoding from ROS/GUI while using the launching Python environment."""

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _read(sock, count):
    chunks = bytearray(count)
    view = memoryview(chunks)
    offset = 0
    while offset < count:
        n = sock.recv_into(view[offset:])
        if not n:
            raise EOFError('LeRobot IPC disconnected')
        offset += n
    return chunks


def send_packet(sock, message):
    arrays, buffers, values = {}, [], {}
    for key, value in message.items():
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            if arr.dtype.hasobject:
                raise ValueError('Object arrays are not supported')
            arrays[key] = {'shape': arr.shape, 'dtype': arr.dtype.str, 'bytes': arr.nbytes}
            buffers.append(memoryview(arr).cast('B'))
        else:
            values[key] = value
    header = json.dumps({'values': values, 'arrays': arrays}, allow_nan=False).encode()
    sock.sendall(struct.pack('!I', len(header)))
    sock.sendall(header)
    for data in buffers:
        sock.sendall(data)


def receive_packet(sock):
    size = struct.unpack('!I', _read(sock, 4))[0]
    if size > 1024 * 1024:
        raise ValueError('Oversized IPC header')
    header = json.loads(_read(sock, size))
    result = header['values']
    for key, desc in header['arrays'].items():
        dtype = np.dtype(desc['dtype'])
        if dtype.hasobject or not 0 <= desc['bytes'] <= 512 * 1024 * 1024:
            raise ValueError('Invalid IPC array')
        result[key] = np.frombuffer(_read(sock, desc['bytes']), dtype=dtype).reshape(desc['shape'])
    return result


class WriterClient:
    """Acquisition thread owns RPC; main thread stays free to stop teleop/render."""

    def __init__(self, root, repo_id, config, metadata, resume=False):
        self.temporary = tempfile.TemporaryDirectory(prefix='pika-lerobot-')
        self.sock, child_sock = socket.socketpair()
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
        env['PYTHONNOUSERSITE'] = '1'
        env['HF_HUB_OFFLINE'] = '1'
        env['HF_DATASETS_OFFLINE'] = '1'
        env['HF_DATASETS_CACHE'] = str(Path(self.temporary.name) / 'datasets')
        env['LD_LIBRARY_PATH'] = str(Path(sys.executable).resolve().parent.parent / 'lib')
        env.pop('LD_PRELOAD', None)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = '1'
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'utils.lerobot_writer', '--fd', str(child_sock.fileno())],
                pass_fds=(child_sock.fileno(),),
                env=env,
                start_new_session=True,
            )
            child_sock.close()
            self.initial = self.call(
                'init',
                config=config,
                metadata=metadata,
                root=str(root),
                repo_id=repo_id,
                resume=resume,
            )
        except BaseException:
            child_sock.close()
            self.close()
            raise

    def call(self, operation, frame=None, **values):
        self.sock.settimeout(5.0 if operation == 'frame' else 180.0)
        packet = dict(frame or {}, op=operation, **values)
        send_packet(self.sock, packet)
        reply = receive_packet(self.sock)
        if not reply.get('ok'):
            raise RuntimeError('LeRobot writer: ' + reply.get('error', 'unknown error'))
        return reply

    def close(self):
        try:
            self.sock.close()
            process = getattr(self, 'process', None)
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        finally:
            self.temporary.cleanup()
