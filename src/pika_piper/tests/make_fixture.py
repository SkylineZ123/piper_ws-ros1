import json,sys
from pathlib import Path
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores,get_typestore
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
store=get_typestore(Stores.ROS1_NOETIC);T=store.types
Time=T['builtin_interfaces/msg/Time'];Header=T['std_msgs/msg/Header'];Joint=T['sensor_msgs/msg/JointState'];Image=T['sensor_msgs/msg/Image']
cfg={'fps':25,'arm':'left','task':'synthetic pick','tcp_xyzrpy':[0,0,.1358,0,0,0],'cameras':{'cam_top':'/test/top','cam_left':'/test/wrist'},'camera_tolerance':.02,'state_gap':.1,'command_age':.1}
manifest={'schema_version':1,'complete':True,'config':cfg,'urdf_sha256':'synthetic','segments':[{'id':0,'start':100.,'end':100.4,'accepted':True},{'id':1,'start':100.5,'end':100.9,'accepted':True}]}
(root/'manifest.json').write_text(json.dumps(manifest))
with Writer(root/'session.bag') as writer:
 topics={'/pika_piper/feedback':'sensor_msgs/msg/JointState','/pika_piper/eef_state':'sensor_msgs/msg/JointState','/pika_piper/command_sent':'sensor_msgs/msg/JointState','/pika_piper/eef_command':'sensor_msgs/msg/JointState','/test/top':'sensor_msgs/msg/Image','/test/wrist':'sensor_msgs/msg/Image'}
 conn={name:writer.add_connection(name,typ,typestore=store) for name,typ in topics.items()}
 for i in range(51):
  ns=100_000_000_000+i*20_000_000;h=Header(i,Time(ns//10**9,ns%10**9),'base_link')
  for name,typ in topics.items():
   if typ.endswith('JointState'):
    values=np.array([i*.001]*6+[.03]);values[:6]+=.1 if 'command' in name else 0
    msg=Joint(h,[],values,np.array([],dtype=np.float64),np.array([],dtype=np.float64))
   else:
    image=np.full((48,64,3),i,dtype=np.uint8);msg=Image(h,48,64,'rgb8',0,64*3,image.ravel())
   writer.write(conn[name],ns,store.serialize_ros1(msg,typ))
