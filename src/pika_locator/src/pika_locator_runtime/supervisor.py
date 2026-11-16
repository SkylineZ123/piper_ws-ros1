"""Forward terminal/ROS shutdown signals and bound unresponsive native cleanup."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _group_alive(process):
    process.poll()  # Reap the direct child before checking its process group.
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def _stop(process, first_signal, interrupt_timeout, terminate_timeout, requests):
    stages = []
    if first_signal == signal.SIGINT:
        stages.append((signal.SIGINT, interrupt_timeout))
    stages.extend([(signal.SIGTERM, terminate_timeout), (signal.SIGKILL, 1.0)])
    for index, (signum, timeout) in enumerate(stages):
        if not _group_alive(process):
            break
        if index:
            print('[pika] Child has not exited; sending %s.' % signal.Signals(signum).name,
                  file=sys.stderr, flush=True)
        _signal_group(process, signum)
        deadline = time.monotonic() + timeout
        request_count = len(requests)
        while _group_alive(process) and time.monotonic() < deadline:
            if len(requests) > request_count:
                break  # Another Ctrl+C advances to the next shutdown stage.
            time.sleep(.02)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        print('[pika] Child remains in an uninterruptible OS state after SIGKILL.',
              file=sys.stderr, flush=True)


def supervise(command, cwd=None, env=None, interrupt_timeout=2.0, terminate_timeout=2.0):
    """Run one owned process group; Ctrl+C exits with 130 after bounded cleanup.

    The child has its own session, so terminal signals reach this supervisor once.
    SIGINT/SIGTERM are forwarded explicitly. No unrelated process is signalled.
    Call from the main thread, as required by Python's signal API.
    """
    requests = []
    process = None
    previous = {}

    def request_stop(signum, frame):
        requests.append(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
        while process.poll() is None and not requests:
            time.sleep(.02)
        if requests:
            print('[pika] Stopping child process; waiting for USB cleanup.',
                  file=sys.stderr, flush=True)
            _stop(process, requests[0], interrupt_timeout, terminate_timeout, requests)
            return 128 + requests[0]
        code = process.returncode
        if _group_alive(process):
            _stop(process, signal.SIGTERM, interrupt_timeout, terminate_timeout, requests)
        return code if code >= 0 else 128 - code
    finally:
        if process is not None and process.poll() is None:
            _signal_group(process, signal.SIGKILL)
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_locator_binary(binary, argv=None):
    import rospkg

    root = Path(rospkg.RosPack().get_path('pika_locator')) / 'runtime'
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = ':'.join((str(root / 'lib'), str(root / 'lib/plugins'),
                                     env.get('LD_LIBRARY_PATH', '')))
    arguments = sys.argv[1:] if argv is None else argv
    try:
        return supervise([str(root / 'bin' / binary)] + list(arguments),
                         cwd=str(root / 'lib'), env=env)
    except OSError as error:
        print('[pika] Cannot start %s: %s' % (binary, error), file=sys.stderr)
        return 1
