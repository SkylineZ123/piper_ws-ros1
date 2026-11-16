#!/usr/bin/env python3
import argparse,json,time,threading
from pathlib import Path
import numpy as np
import rospy,rospkg,yaml
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool
from pika_piper.core import read_trajectory,interpolate_eef,limit_step
from pika_piper.kinematics import Kinematics

def main():
    p=argparse.ArgumentParser();p.add_argument('--trajectory',required=True);p.add_argument('--config',required=True);p.add_argument('--speed',type=float,default=1);a=p.parse_args(rospy.myargv()[1:])
    if not 0<a.speed<=1:raise ValueError('Invalid speed')
    meta,actions,stamps=read_trajectory(a.trajectory);c=yaml.safe_load(Path(a.config).read_text())
    rp=rospkg.RosPack();desc=Path(rp.get_path('piper_description'));k=Kinematics(desc/'urdf/piper_description.urdf',desc,c['tcp_xyzrpy'],Path(rp.get_path('piper_with_gripper_moveit'))/'config/piper_description.srdf',posture_weight=c.get('ik_posture_weight',1e-4),pose_acceptance=c.get('pose_acceptance','official'))
    if meta['arm']!=c['arm'] or meta['urdf_sha256']!=k.urdf_hash or not np.allclose(meta['tcp_xyzrpy'],c['tcp_xyzrpy']):raise ValueError('Robot model, arm or TCP mismatch')
    rospy.init_node('pika_replay');claim=rospy.ServiceProxy('/pika_piper/set_replay',SetBool);claim.wait_for_service(timeout=10)
    mode={'mode':'waiting'};rospy.Subscriber('/pika_piper/status',String,lambda m:mode.update(json.loads(m.data)))
    feedback=rospy.wait_for_message('/pika_piper/feedback',JointState,timeout=5)
    if abs((rospy.Time.now()-feedback.header.stamp).to_sec())>.2:raise ValueError('Stale feedback')
    # Validate every target before acquiring physical control.
    seed=np.asarray(feedback.position)
    for target in actions:
        q=target if meta['action_type']=='joint' else k.ik(__import__('pika_piper.core',fromlist=['transform']).transform(target[:6]),seed,target[6])
        if np.any(q[:6]<k.model.lowerPositionLimit) or np.any(q[:6]>k.model.upperPositionLimit) or k.collision(q,q[6]):raise ValueError('Invalid replay trajectory')
        seed=q
    pub=rospy.Publisher('/pika_piper/replay_input',JointState,queue_size=1)
    deadline=time.monotonic()+5
    while pub.get_num_connections()==0:
        if time.monotonic()>deadline:raise RuntimeError('Controller replay input unavailable')
        time.sleep(.01)
    response=claim(True)
    if not response.success:raise RuntimeError(response.message)
    mode['mode']='replay'
    def send(value,kind):
        if mode['mode']=='fault':raise RuntimeError(mode.get('reason','controller fault'))
        m=JointState();m.header.stamp=rospy.Time.now();m.name=[kind];m.position=value;pub.publish(m)
    try:
        from pika_piper.core import transform
        first=actions[0] if meta['action_type']=='joint' else k.ik(transform(actions[0,:6]),feedback.position,actions[0,6])
        start=np.asarray(feedback.position);duration=max(2.,np.max(abs(first[:6]-start[:6]))/c['max_joint_speed'],abs(first[6]-start[6])/c['max_gripper_speed'])
        hz=c['control_hz'];rate=rospy.Rate(hz)
        for alpha in np.linspace(0,1,int(duration*hz)+1):send(start+(first-start)*alpha,'joint');rate.sleep()
        # Verify arrival before starting recorded timing.
        actual=rospy.wait_for_message('/pika_piper/feedback',JointState,timeout=2)
        if np.max(abs(np.asarray(actual.position[:6])-first[:6]))>.05:raise RuntimeError('First-frame approach did not converge')
        origin=time.monotonic()
        while not rospy.is_shutdown():
            t=(time.monotonic()-origin)*a.speed
            if t>stamps[-1]:break
            j=min(int(np.searchsorted(stamps,t,side='right')),len(stamps)-1);i=max(0,j-1)
            alpha=0 if i==j else (t-stamps[i])/(stamps[j]-stamps[i])
            value=(1-alpha)*actions[i]+alpha*actions[j] if meta['action_type']=='joint' else interpolate_eef(actions[i],actions[j],alpha)
            send(value,meta['action_type']);rate.sleep()
        send(actions[-1],meta['action_type']);rate.sleep();rate.sleep()
    finally:claim(False)
if __name__=='__main__':main()
