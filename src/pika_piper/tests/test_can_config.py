"""Exercise CAN selection with fake sysfs and ip; never change host interfaces."""
import os
from pathlib import Path
import subprocess
import pytest

@pytest.fixture
def run_can(tmp_path):
    net=tmp_path/'net';net.mkdir()
    tools=tmp_path/'bin';tools.mkdir()
    log=tmp_path/'commands.log'
    source=Path(__file__).resolve().parents[3]/'can_config.sh'
    text=source.read_text().replace('/sys/class/net',str(net))
    assert '/sys/class/net' not in text
    script=tmp_path/'can_config.sh';script.write_text(text)
    ip=tools/'ip';ip.write_text('''#!/usr/bin/python3
import os,sys
if 'show' in sys.argv:
 print('13: test: <UP> can bitrate '+os.environ.get('TEST_BITRATE','1000000'))
else:
 with open(os.environ['TEST_CAN_LOG'],'a') as f:f.write(' '.join(sys.argv[1:])+'\\n')
 if os.environ.get('TEST_FAIL_ON') in sys.argv[1:]:sys.exit(2)
''');ip.chmod(0o755)
    sudo=tools/'sudo';sudo.write_text('#!/bin/sh\nif [ "$1" = -v ]; then exit 0; fi\nexec "$@"\n');sudo.chmod(0o755)
    def add(name,bus,flags='0x80',kind=280):
        folder=net/name;folder.mkdir();(folder/'type').write_text(str(kind));(folder/'flags').write_text(flags)
        hardware=tmp_path/'hardware'/bus;hardware.mkdir(parents=True,exist_ok=True);(folder/'device').symlink_to(hardware)
    def run(*args,fail_on=''):
        env=os.environ.copy();env.update(PATH=str(tools)+':'+env['PATH'],TEST_CAN_LOG=str(log),TEST_FAIL_ON=fail_on)
        result=subprocess.run(['/bin/bash',str(script),*args],env=env,text=True,capture_output=True)
        return result,log.read_text().splitlines() if log.exists() else []
    return add,run

def test_only_usb_target_is_modified(run_can):
    add,run=run_can;add('can0','1-10.4.1:1.0');add('can1','1-6.1:1.0')
    result,commands=run()
    assert result.returncode==0,result.stderr
    assert commands==['link set dev can1 down','link set dev can1 type can bitrate 1000000','link set dev can1 name piper_left','link set dev piper_left up']
    assert '忽略 CAN 接口：can0' in result.stderr

def test_dry_run_is_read_only(run_can):
    add,run=run_can;add('can1','1-6.1:1.0')
    result,commands=run('--dry-run')
    assert result.returncode==0 and not commands
    assert 'piper_left' in result.stdout

def test_missing_target_changes_nothing(run_can):
    add,run=run_can;add('can0','1-10.4.1:1.0')
    result,commands=run()
    assert result.returncode!=0 and not commands

def test_duplicate_usb_match_changes_nothing(run_can):
    add,run=run_can;add('can1','1-6.1:1.0');add('can2','1-6.1:1.0')
    result,commands=run()
    assert result.returncode!=0 and not commands

def test_name_conflict_changes_nothing(run_can):
    add,run=run_can;add('can1','1-6.1:1.0');add('piper_left','other-device',kind=1)
    result,commands=run()
    assert result.returncode!=0 and not commands

def test_already_configured_does_not_interrupt_link(run_can):
    add,run=run_can;add('piper_left','1-6.1:1.0',flags='0x81')
    result,commands=run()
    assert result.returncode==0 and not commands

def test_configuration_failure_stops_before_rename(run_can):
    add,run=run_can;add('can1','1-6.1:1.0')
    result,commands=run(fail_on='type')
    assert result.returncode!=0
    assert len(commands)==2 and all(' name ' not in c and not c.endswith(' up') for c in commands)
