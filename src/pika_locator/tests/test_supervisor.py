"""Signal tests use fake native processes and file locks, never robot hardware."""

import fcntl
import os
from pathlib import Path
import pty
import signal
import subprocess
import sys
import time

import pytest

SOURCE = str(Path(__file__).parents[1] / 'src')
WORKER = '''
import fcntl, os, signal, subprocess, sys, time
from pathlib import Path
root=Path(sys.argv[1]); mode=sys.argv[2]
lock=(root/'device.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stop(signum,frame):
    (root/'received').write_text(str(signum))
    if mode=='graceful':
        (root/'cleaned').write_text('yes')
        raise SystemExit(0)
signal.signal(signal.SIGINT,stop)
signal.signal(signal.SIGTERM,stop)
if mode=='descendant':
    code="import signal,time; from pathlib import Path; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(%r).write_text('ready'); time.sleep(60)" % str(root/'grandchild_ready')
    child=subprocess.Popen([sys.executable,'-c',code])
    (root/'grandchild').write_text(str(child.pid))
    while not (root/'grandchild_ready').exists(): time.sleep(.005)
(root/'ready').write_text(str(os.getpid()))
while True: time.sleep(.02)
'''


def wait_for(path, timeout=4):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if path.exists():return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for '+str(path))


def process_running(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1][0]!='Z'
    except FileNotFoundError:
        return False


def harness(tmp_path, mode):
    worker=tmp_path/'worker.py';worker.write_text(WORKER)
    script=tmp_path/'supervisor.py'
    script.write_text(
        'import sys\n'
        'sys.path.insert(0,%r)\n' % SOURCE+
        'from pika_locator_runtime import supervise\n'
        'sys.exit(supervise([sys.executable,%r,%r,%r],interrupt_timeout=.15,terminate_timeout=.15))\n'
        % (str(worker),str(tmp_path),mode))
    return script


def assert_released(tmp_path):
    assert not process_running(int((tmp_path/'ready').read_text()))
    with (tmp_path/'device.lock').open('a') as resource:
        fcntl.flock(resource,fcntl.LOCK_EX|fcntl.LOCK_NB)


@pytest.mark.parametrize('mode,signum',[
    ('graceful',signal.SIGINT),
    ('graceful',signal.SIGTERM),
    ('ignore',signal.SIGINT),
    ('ignore',signal.SIGTERM),
    ('descendant',signal.SIGINT),
])
def test_shutdown_reaps_process_and_releases_resource(tmp_path,mode,signum):
    script=harness(tmp_path,mode)
    parent=subprocess.Popen([sys.executable,str(script)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        wait_for(tmp_path/'ready')
        parent.send_signal(signum)
        stdout,stderr=parent.communicate(timeout=5)
        assert parent.returncode==128+signum
        assert b'Traceback' not in stderr
        assert_released(tmp_path)
        if mode=='graceful':assert (tmp_path/'cleaned').exists()
        else:assert b'SIGKILL' in stderr
        if mode=='descendant':assert not process_running(int((tmp_path/'grandchild').read_text()))
    finally:
        if parent.poll() is None:
            parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=5)


def test_terminal_ctrl_c(tmp_path):
    script=harness(tmp_path,'ignore')
    pid,terminal=pty.fork()
    if pid==0:
        os.execv(sys.executable,[sys.executable,str(script)])
    reaped=False
    try:
        wait_for(tmp_path/'ready')
        os.write(terminal,b'\x03')
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            child,status=os.waitpid(pid,os.WNOHANG)
            if child:
                reaped=True
                assert os.WIFEXITED(status)
                assert os.WEXITSTATUS(status)==130
                break
            time.sleep(.02)
        assert reaped,'Terminal Ctrl+C failed to stop the supervisor'
        assert_released(tmp_path)
    finally:
        if not reaped:
            os.kill(pid,signal.SIGTERM)
            os.waitpid(pid,0)
        os.close(terminal)


def test_natural_exit_code_preserved():
    from pika_locator_runtime import supervise
    old_int=signal.getsignal(signal.SIGINT)
    old_term=signal.getsignal(signal.SIGTERM)
    assert supervise([sys.executable,'-c','raise SystemExit(7)'])==7
    assert signal.getsignal(signal.SIGINT)==old_int
    assert signal.getsignal(signal.SIGTERM)==old_term


def test_missing_executable_restores_handlers():
    from pika_locator_runtime import supervise
    old_int=signal.getsignal(signal.SIGINT)
    with pytest.raises(FileNotFoundError):supervise(['/missing/pika-test-executable'])
    assert signal.getsignal(signal.SIGINT)==old_int


def test_locator_environment_and_arguments(tmp_path,monkeypatch):
    import json
    import rospkg
    from pika_locator_runtime import run_locator_binary
    runtime=tmp_path/'runtime'
    (runtime/'bin').mkdir(parents=True)
    (runtime/'lib/plugins').mkdir(parents=True)
    executable=runtime/'bin/survive-cli'
    output=tmp_path/'child.json'
    executable.write_text('#!'+sys.executable+'\n'
        'import json,os,sys\n'
        'from pathlib import Path\n'
        'Path(%r).write_text(json.dumps([os.getcwd(),os.environ["LD_LIBRARY_PATH"],sys.argv[1:]]))\n'
        'raise SystemExit(7)\n' % str(output))
    executable.chmod(0o755)
    monkeypatch.setattr(rospkg.RosPack,'get_path',lambda self,name:str(tmp_path))
    monkeypatch.setenv('LD_LIBRARY_PATH','/test/existing')
    assert run_locator_binary('survive-cli',['--force-calibrate','__name:=test'])==7
    cwd,libraries,arguments=json.loads(output.read_text())
    assert cwd==str(runtime/'lib')
    assert libraries.split(':')==[str(runtime/'lib'),str(runtime/'lib/plugins'),'/test/existing']
    assert arguments==['--force-calibrate','__name:=test']
    assert os.environ['LD_LIBRARY_PATH']=='/test/existing'
