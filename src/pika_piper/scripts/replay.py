#!/usr/bin/env python3
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from pika_piper.core import validate_metadata
from pika_piper.configuration import control_environment

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--episode',type=int,default=0);p.add_argument('--execute',action='store_true');p.add_argument('--config');p.add_argument('--control-python',help='pika-bringup Python, required with --execute');p.add_argument('--speed',type=float,default=1.);a=p.parse_args()
    if a.execute and not a.control_python:p.error('--execute requires --control-python /path/to/pika-bringup/bin/python')
    if not 0<a.speed<=1:raise ValueError('Replay speed must be in (0,1]')
    root=Path(a.dataset);meta=json.loads((root/'meta/pika_pipeline.json').read_text());validate_metadata(meta)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset(repo_id='local/replay',root=root,episodes=[a.episode],video_backend='pyav')
    # Use the public reader: episode boundaries are not inferred from shard filenames.
    actions=[];stamps=[]
    import cv2
    video=None
    preview=root/"replay_preview";preview.mkdir(exist_ok=True)
    for i in range(len(ds)):
        row=ds[i]
        images=[]
        for key in sorted(k for k in row if k.startswith('observation.images.')):
            image=np.asarray(row[key]);image=np.moveaxis(image,0,-1) if image.shape[0]==3 else image
            if image.dtype!=np.uint8:image=np.clip(image*255,0,255).astype(np.uint8)
            images.append(image[:,:,::-1])
        if images:
            h=min(im.shape[0] for im in images);strip=np.concatenate([cv2.resize(im,(int(im.shape[1]*h/im.shape[0]),h)) for im in images],axis=1)
            if video is None:
                video=cv2.VideoWriter(str(preview/('episode_%d.mp4'%a.episode)),cv2.VideoWriter_fourcc(*'mp4v'),ds.fps,(strip.shape[1],h))
                if not video.isOpened():raise RuntimeError('Preview video encoder unavailable')
            video.write(strip)
        actions.append(np.asarray(row['action']));stamps.append(float(row['timestamp']))
    if video is not None:video.release()
    actions=np.asarray(actions);stamps=np.asarray(stamps)
    output=root/'replay_preview';output.mkdir(exist_ok=True)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,6));axes[0].plot(stamps,actions[:,:6]);axes[1].plot(stamps,actions[:,6]);axes[0].set_title(meta['action_type']);axes[1].set_xlabel('seconds');fig.savefig(output/('episode_%d.png'%a.episode));plt.close(fig)
    with tempfile.TemporaryDirectory(prefix='pika-replay-') as temporary:
        path=Path(temporary)/'trajectory.npz';np.savez(path,actions=actions,timestamps=stamps);path.with_suffix('.json').write_text(json.dumps(meta))
        from pika_piper.core import read_trajectory
        read_trajectory(path)
        print('Validated %d frames; preview: %s'%(len(actions),output))
        if a.execute:
            if not a.config:raise ValueError('--config required for physical replay')
            worker=Path(__file__).with_name('replay_worker.py')
            subprocess.run([a.control_python,str(worker),'--trajectory',str(path),'--config',a.config,'--speed',str(a.speed)],env=control_environment(a.control_python),check=True)
if __name__=='__main__':main()
