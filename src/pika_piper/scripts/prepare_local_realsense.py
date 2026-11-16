#!/usr/bin/env python3
"""Install the two official ROS runtime packages in this workspace without root."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path

PACKAGES={
    'ros-noetic-realsense2-camera':('2.3.2-1focal.20250521.010944','3adb639bca8382df217184f4c315fd89bda13cb90e9bcd93ce59a49a463cc95e'),
    'ros-noetic-realsense2-description':('2.3.2-1focal.20250519.234358','f5ce6d151945502f01e2e9ab67078ba7b27e4b979ceb229dbfb0c41b970440dc'),
}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace', type=Path, required=True)
    workspace=parser.parse_args().workspace.resolve()
    root=workspace/'.deps/ros-noetic';root.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with tempfile.TemporaryDirectory(prefix='pika-ros-debs-') as temporary:
        for name,(version,digest) in PACKAGES.items():
            subprocess.run(['apt-get','download',name+'='+version],cwd=temporary,check=True)
            matches=list(Path(temporary).glob(name+'_*.deb'))
            if len(matches)!=1:raise RuntimeError('Cannot identify downloaded package '+name)
            package=matches[0]
            if hashlib.sha256(package.read_bytes()).hexdigest()!=digest:raise RuntimeError('Checksum mismatch '+name)
            subprocess.run(['dpkg-deb','--extract',str(package),str(root)],check=True)
            manifest.append({'package':name,'file':package.name,'sha256':digest})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Local ROS runtime prepared:',root)
    print('Build pika_piper and source devel/setup.bash to enable it.')

if __name__=='__main__':main()
