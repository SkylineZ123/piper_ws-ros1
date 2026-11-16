"""Run the shell script with fake USB/proc trees and a fake kill executable."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def tracker_script(tmp_path):
    usb=tmp_path/'sys_usb';usb.mkdir()
    hid=tmp_path/'sys_hid';hid.mkdir()
    dev=tmp_path/'dev_usb';dev.mkdir()
    proc=tmp_path/'fake_proc';proc.mkdir()
    tools=tmp_path/'tools';tools.mkdir()
    state=tmp_path/'owners.json';state.write_text('{}')
    log=tmp_path/'signals';source=Path(__file__).resolve().parents[3]/'stop_tracker.sh'
    fake_kill=tools/'kill'
    text=source.read_text()
    for old,new in [('/sys/bus/usb/devices',usb),('/sys/class/hidraw',hid),
                    ('/dev/bus/usb',dev),('/proc/',str(proc)+'/'),('/bin/kill',fake_kill)]:
        text=text.replace(old,str(new))
    assert '/bin/kill' not in text and '/proc/' not in text
    script=tmp_path/'stop_tracker.sh';script.write_text(text)
    fuser=tools/'fuser';fuser.write_text('''#!/usr/bin/python3
import json,os,sys
if '--' in sys.argv[1:]:
 print('No process specification given',file=sys.stderr);sys.exit(1)
if os.environ.get('TEST_FUSER_ERROR')=='1':
 print('fake fuser query error',file=sys.stderr);sys.exit(1)
d=json.load(open(os.environ['TEST_TRACKER_OWNERS']))
print(' '.join(str(pid) for name in sys.argv[1:] for pid in d.get(name,[])))
''');fuser.chmod(0o755)
    ps=tools/'ps';ps.write_text('''#!/usr/bin/python3
import os,sys
from pathlib import Path
pid=sys.argv[sys.argv.index('-p')+1]
print('PID PPID STAT COMMAND: '+pid+' fake_tracker')
if os.environ.get('TEST_REUSE_PID')=='1':
 p=Path(os.environ['TEST_TRACKER_PROC'])/pid/'stat'
 text=p.read_text().split();text[21]='999';p.write_text(' '.join(text))
''');ps.chmod(0o755)
    fake_kill.write_text('''#!/usr/bin/python3
import json,os,sys
from pathlib import Path
sig=sys.argv[sys.argv.index('--signal')+1];pid=int(sys.argv[-1])
with open(os.environ['TEST_TRACKER_LOG'],'a') as f:f.write(sig+' '+str(pid)+'\\n')
if sig==os.environ.get('TEST_RELEASE_ON','INT'):
 p=Path(os.environ['TEST_TRACKER_OWNERS']);d=json.loads(p.read_text())
 d={name:[n for n in values if n!=pid] for name,values in d.items()}
 if os.environ.get('TEST_NEW_OWNER')=='1':d[next(iter(d))]=[79999]
 p.write_text(json.dumps(d))
''');fake_kill.chmod(0o755)

    def add(name='1-2',pid=71234,vid='28de',product='2300',number=64):
        folder=usb/name;folder.mkdir()
        for field,value in [('idVendor',vid),('idProduct',product),('product','LHR'),
                            ('busnum','1'),('devnum',str(number))]:
            (folder/field).write_text(value)
        node=dev/'001'/('%03d'%number);node.parent.mkdir(exist_ok=True);node.touch()
        data=json.loads(state.read_text());data[str(node)]=[pid];state.write_text(json.dumps(data))
        process=proc/str(pid);process.mkdir(exist_ok=True)
        # proc fields 3..22, with starttime at field 22.
        (process/'stat').write_text('%d (fake) S '%pid+' '.join(['0']*18+['123']))

    def run(*args,answer='',release='INT',reuse=False,new_owner=False,query_error=False):
        env=os.environ.copy();env.update(PATH=str(tools)+':'+env['PATH'],
            TEST_TRACKER_OWNERS=str(state),TEST_TRACKER_LOG=str(log),TEST_TRACKER_PROC=str(proc),
            TEST_RELEASE_ON=release,TEST_REUSE_PID=str(int(reuse)),TEST_NEW_OWNER=str(int(new_owner)),
            TEST_FUSER_ERROR=str(int(query_error)))
        result=subprocess.run(['/bin/bash',str(script),'--timeout','0',*args],
                              input=answer,text=True,capture_output=True,env=env,timeout=5)
        return result,log.read_text().splitlines() if log.exists() else []
    return add,run


def test_no_tracker_changes_nothing(tracker_script):
    _,run=tracker_script;r,signals=run('--yes')
    assert r.returncode==0 and not signals


def test_dry_run_and_cancel_never_signal(tracker_script):
    add,run=tracker_script;add()
    r,signals=run('--dry-run');assert r.returncode==0 and not signals
    r,signals=run(answer='n\n');assert r.returncode==0 and not signals


def test_unrelated_usb_devices_are_ignored(tracker_script):
    add,run=tracker_script;add();add('2-1',pid=72222,vid='8086',product='0b5b',number=65)
    r,signals=run('--yes')
    assert r.returncode==0,r.stderr
    assert signals==['INT 71234']


def test_one_pid_opening_multiple_trackers_is_signalled_once(tracker_script):
    add,run=tracker_script;add();add('1-3',number=65)
    r,signals=run('--yes')
    assert r.returncode==0 and signals==['INT 71234']


def test_pid_reuse_is_not_signalled(tracker_script):
    add,run=tracker_script;add()
    r,signals=run('--yes',reuse=True)
    assert r.returncode==1 and not signals


def test_term_is_used_only_if_int_does_not_release_device(tracker_script):
    add,run=tracker_script;add()
    r,signals=run('--yes',release='TERM')
    assert r.returncode==0 and signals==['INT 71234','TERM 71234']


@pytest.mark.parametrize('force,expected_code',[(False,1),(True,0)])
def test_kill_requires_explicit_force(tracker_script,force,expected_code):
    add,run=tracker_script;add()
    r,signals=run('--yes',*(['--force'] if force else []),release='KILL')
    assert r.returncode==expected_code
    assert signals==['INT 71234','TERM 71234']+(['KILL 71234'] if force else [])


def test_new_owner_is_reported_without_being_killed(tracker_script):
    add,run=tracker_script;add()
    r,signals=run('--yes',new_owner=True)
    assert r.returncode==1 and signals==['INT 71234']
    assert '79999' in r.stdout


def test_fuser_query_error_is_not_reported_as_no_owners(tracker_script):
    add,run=tracker_script;add()
    r,signals=run('--yes',query_error=True)
    assert r.returncode==2 and not signals
    assert 'fake fuser query error' in r.stderr
    assert '未发现' not in r.stdout
