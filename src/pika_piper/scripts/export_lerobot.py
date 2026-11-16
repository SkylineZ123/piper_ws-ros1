#!/usr/bin/env python3
"""Run with the LeRobot Python environment, not ROS Python."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from pika_piper.dataset import load_session,aligned_episodes
from pika_piper.core import FIELDS

def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--output',required=True);p.add_argument('--repo-id',default='local/pika_piper');p.add_argument('--action-type',choices=['joint','eef6d'],required=True);a=p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.video import RGBEncoderConfig
    manifest,streams=load_session(a.session);cfg=manifest['config'];joint=['joint%d'%i for i in range(1,7)]+['gripper']
    def numeric(names):return {'dtype':'float32','shape':(7,),'names':names}
    features={'observation.state':numeric(joint),'observation.eef_state':numeric(FIELDS),'action':numeric(joint if a.action_type=='joint' else FIELDS),
              'joint_command':numeric(joint),'eef_command':numeric(FIELDS),'source_timestamp':{'dtype':'float64','shape':(1,),'names':['seconds']},'segment_id':{'dtype':'int64','shape':(1,),'names':['segment']}}
    for camera in cfg['cameras']:
        if not streams[camera][1]:raise ValueError('No image data for '+camera)
        features['observation.images.'+camera]={'dtype':'video','shape':streams[camera][1][0].shape,'names':['height','width','channels']}
    ds=None;count=0
    try:
        for frames in aligned_episodes(manifest,streams):
            if ds is None:
                ds=LeRobotDataset.create(repo_id=a.repo_id,root=a.output,fps=cfg['fps'],robot_type='single_piper_'+cfg['arm'],features=features,rgb_encoder=RGBEncoderConfig(vcodec='h264'),streaming_encoding=True)
            for frame in frames:
                row={'task':cfg['task'],'observation.state':np.asarray(frame['state'],np.float32),'observation.eef_state':np.asarray(frame['eef_state'],np.float32),
                     'action':np.asarray(frame['joint_command' if a.action_type=='joint' else 'eef_command'],np.float32),
                     'joint_command':np.asarray(frame['joint_command'],np.float32),'eef_command':np.asarray(frame['eef_command'],np.float32),
                     'source_timestamp':np.array([frame['source_timestamp']],np.float64),'segment_id':np.array([frame['segment_id']],np.int64)}
                row.update({'observation.images.'+cam:frame[cam].array() for cam in cfg['cameras']});ds.add_frame(row)
            ds.save_episode();count+=1
    finally:
        if ds is not None:ds.finalize()
    if not count:raise ValueError('No valid accepted segments; no dataset produced')
    metadata={'schema_version':1,'action_type':a.action_type,'units':{'length':'m','angle':'rad'},'rotation':'xyz_rpy','reference':'base_link_absolute','arm':cfg['arm'],
              'tcp_xyzrpy':cfg['tcp_xyzrpy'],'urdf_sha256':manifest['urdf_sha256'],'source_session':str(Path(a.session).resolve()),'alignment':{'fps':cfg['fps'],'camera_tolerance':cfg['camera_tolerance'],'state_gap':cfg['state_gap'],'command_age':cfg['command_age']}}
    (Path(a.output)/'meta/pika_pipeline.json').write_text(json.dumps(metadata,indent=2)+'\n');print('Exported %d episodes'%count)
if __name__=='__main__':main()
