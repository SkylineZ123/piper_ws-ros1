"""Check dependency rejection before any hardware supervisor can be called."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def bringup():
    spec = importlib.util.spec_from_file_location(
        'test_bringup_cli', Path(__file__).parents[1]/'scripts/bringup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def argv(monkeypatch, *extra):
    config = Path(__file__).parents[1]/'config/pipeline.yaml'
    monkeypatch.setattr(sys, 'argv', ['bringup.py', '--config', str(config), *extra])


def test_dry_run_does_not_run_dependency_check_or_supervisor(bringup,monkeypatch,capsys):
    monkeypatch.delenv('PIKA_CONTROL_PYTHON',raising=False)
    argv(monkeypatch,'--dry-run')
    monkeypatch.setattr(bringup,'check_control_environment',lambda *a,**k:pytest.fail('preflight started'))
    monkeypatch.setattr(bringup,'supervise',lambda *a,**k:pytest.fail('supervisor started'))
    assert bringup.main()==0
    assert 'control_python:='+sys.executable in capsys.readouterr().out


def test_speed_options_forwarded_to_new_package(bringup,monkeypatch,capsys):
    argv(monkeypatch,'--dry-run','--motion-speed-percent','65','--max-tcp-speed','0.2')
    assert bringup.main()==0
    output=capsys.readouterr().out
    assert 'roslaunch pika_piper teleop.launch' in output
    assert 'motion_speed_percent:=65' in output
    assert 'max_tcp_speed:=0.2' in output
    assert 'start_cameras:=false' in output


def test_invalid_speed_cannot_launch_hardware(bringup,monkeypatch):
    argv(monkeypatch,'--motion-speed-percent','101')
    monkeypatch.setattr(bringup,'supervise',lambda *a,**k:pytest.fail('hardware launch reached'))
    with pytest.raises(SystemExit) as error:bringup.main()
    assert error.value.code==2


def test_missing_ik_dependencies_cannot_launch_driver(bringup,monkeypatch,capsys):
    argv(monkeypatch,'--control-python','/missing/solver/python')
    def missing(*args,**kwargs):
        raise bringup.ReplaySetupError("ModuleNotFoundError: No module named 'casadi'")
    monkeypatch.setattr(bringup,'check_control_environment',missing)
    monkeypatch.setattr(bringup,'supervise',lambda *a,**k:pytest.fail('hardware launch reached'))
    with pytest.raises(SystemExit) as error:bringup.main()
    assert error.value.code==2
    assert 'pika-bringup' in capsys.readouterr().err
