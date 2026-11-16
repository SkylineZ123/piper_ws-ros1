#!/usr/bin/env python3
"""Read-only environment checks. Does not open CAN, cameras or tracking devices."""
import importlib,importlib.util,json,os,platform,subprocess,hashlib
from pathlib import Path
import rospkg

def main():
    errors=[]
    versions={}
    for module in ('rospy','rosbag','numpy','scipy','typing_extensions','can','piper_sdk','pinocchio','casadi','pinocchio.casadi','cv_bridge','tf.transformations'):
        try:
            loaded=importlib.import_module(module)
            versions[module]=getattr(loaded,'__version__','import OK')
        except (ImportError,OSError) as error:errors.append(module+': '+str(error))
    rp=rospkg.RosPack()
    for package in ('piper','piper_description','piper_with_gripper_moveit','data_msgs','pika_locator','realsense2_camera'):
        try: print(package+': '+rp.get_path(package))
        except rospkg.ResourceNotFound:errors.append('Missing ROS package: '+package)
    try:
        package=Path(rp.get_path('pika_locator'));root=package/'runtime'
        manifest=json.loads((package/'PROVENANCE.json').read_text())
        for name,expected in manifest['sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:errors.append('Locator checksum mismatch: '+name)
        env=os.environ.copy();env['LD_LIBRARY_PATH']=str(root/'lib')+':'+str(root/'lib/plugins')+':'+env.get('LD_LIBRARY_PATH','')
        for path in list((root/'bin').iterdir())+list((root/'lib/plugins').glob('*.so')):
            result=subprocess.run(['ldd',str(path)],env=env,text=True,capture_output=True)
            if result.returncode or 'not found' in result.stdout:errors.append(str(path)+': '+result.stdout+result.stderr)
        if platform.machine()!='x86_64':errors.append('Bundled locator requires x86_64')
    except rospkg.ResourceNotFound:pass
    print(json.dumps({'ok':not errors,'python':__import__('sys').executable,'versions':versions,'errors':errors},indent=2));raise SystemExit(bool(errors))
if __name__=='__main__':
    main()
