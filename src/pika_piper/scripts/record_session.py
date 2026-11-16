#!/usr/bin/env python3
"""Keyboard terminal recorder. Records all raw streams; saves only label decisions."""
import argparse, datetime, hashlib, json, threading, uuid
from pathlib import Path
import rospy, rosbag, rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState, Imu
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import SetBool
from data_msgs.msg import Gripper, LocalizationStatus
from pika_piper.configuration import load_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');a=p.parse_args(rospy.myargv()[1:])
    cfg=load_config(a.config);rospy.init_node('pika_recorder',disable_signals=True)
    # Preserve the effective startup limits when launch overrides the YAML.
    # Status messages in the bag retain any later paused tuning changes.
    controller_params=rospy.get_param('/pika_piper/controller',{})
    if controller_params:
        actual=load_config(controller_params.get('config',a.config),controller_params)
        for key in ('arm','tcp_xyzrpy','axis_rpy','pika_to_tcp_xyzrpy','control_hz',
                    'ik_posture_weight','max_joint_speed','max_gripper_speed',
                    'max_tcp_speed','max_tcp_angular_speed','translation_scale'):
            if key in actual:cfg[key]=actual[key]
    cfg['motion_speed_percent']=rospy.get_param('/pika_piper/driver/motion_speed_percent',cfg.get('motion_speed_percent',50))
    root=Path(cfg['output_dir']).expanduser()/('session_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
    desc=Path(rospkg.RosPack().get_path('piper_description'))/'urdf/piper_description.urdf'
    meta={'schema_version':1,'config':cfg,'urdf_sha256':hashlib.sha256(desc.read_bytes()).hexdigest(),'segments':[],'complete':False}
    lock=threading.RLock();bag=rosbag.Bag(str(root/'session.bag.active'),'w',compression='lz4')
    current=None;mode='idle';failure=[]
    teleop=rospy.ServiceProxy('/pika_piper/set_teleop',SetBool)
    def persist():
        temp=root/'manifest.json.tmp';temp.write_text(json.dumps(meta,indent=2));temp.replace(root/'manifest.json')
    def end():
        nonlocal current
        if current is not None:current['end']=rospy.Time.now().to_sec();current=None
    def callback(msg,topic):
        nonlocal mode
        with lock:
            if failure:return
            try:
                bag.write(topic,msg)
                if topic=='/pika_piper/status':
                    state=json.loads(msg.data);mode=state['mode']
                    if mode!='teleop':end()
            except Exception as e:
                failure.append(str(e));end();rospy.logerr('Recorder failed: %s',e)
                try:teleop(False)
                except rospy.ServiceException:pass
    subscriptions=[]
    topics={'/pika_pose':PoseStamped,'/gripper/data':Gripper,'/imu/data':Imu,'/pika_localization_status':LocalizationStatus,
            '/pika_piper/feedback':JointState,'/pika_piper/command_sent':JointState,'/pika_piper/eef_state':JointState,
            '/pika_piper/eef_command':JointState,'/pika_piper/eef_requested':PoseStamped,'/pika_piper/status':String}
    topics['/pika_piper/sense_relative']=PoseStamped
    topics['/pika_piper/eef_feasible']=PoseStamped
    topics.update({topic:Image for topic in cfg['cameras'].values()})
    for topic,typ in topics.items():subscriptions.append(rospy.Subscriber(topic,typ,callback,callback_args=topic,queue_size=100))
    persist();print('Session: '+str(root));print('t: toggle teleop | c: record | space/x: pause | s: save pending | r: discard pending | q: finish; press Enter after command')
    try:
        while not rospy.is_shutdown() and not failure:
            command=input('> ').strip()
            if command=='q':break
            if command=='t':
                response=teleop(mode!='teleop');print(response.message);continue
            if command in ('','x','s','r'):
                teleop(False)
                with lock:
                    end()
                    if command in ('s','r'):
                        for segment in meta['segments']:
                            if segment['accepted'] is None:segment['accepted']=command=='s'
            elif command=='c':
                with lock:
                    if mode!='teleop':print('Enable teleop first with t');continue
                    if current is None:
                        current={'start':rospy.Time.now().to_sec(),'end':None,'accepted':None,'id':len(meta['segments'])};meta['segments'].append(current)
            with lock:
                bag.write('/pika_piper/record_events',String(json.dumps({'key':command,'segments':meta['segments']})))
                persist()
    except (EOFError,KeyboardInterrupt):pass
    finally:
        try:teleop(False)
        except rospy.ServiceException:pass
        for sub in subscriptions:sub.unregister()
        with lock:
            end();bag.close();meta['complete']=not bool(failure);meta['errors']=failure;persist()
            if not failure:(root/'session.bag.active').rename(root/'session.bag')
        rospy.signal_shutdown('finished')
    if failure:raise RuntimeError(failure[0])
if __name__=='__main__':main()
