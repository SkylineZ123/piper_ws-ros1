"""Actual bag -> Parquet/MP4 -> LeRobot readback, optional export dependencies."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('rosbags.highlevel')
pytest.importorskip('lerobot')

@pytest.mark.parametrize('kind', ['joint','eef6d'])
def test_bag_to_lerobot(tmp_path,kind):
    scripts=Path(__file__).parents[1]/'scripts'
    session=tmp_path/'session';output=tmp_path/'dataset'
    subprocess.run([sys.executable,str(Path(__file__).with_name('make_fixture.py')),str(session)],check=True,capture_output=True)
    subprocess.run([sys.executable,str(scripts/'export_lerobot.py'),'--session',str(session),'--output',str(output),'--action-type',kind],check=True,capture_output=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset(repo_id='local/test',root=output,video_backend='pyav')
    assert ds.num_episodes==2 and len(ds)==20
    row=ds[0];assert row['action'].shape==(7,)
    np.testing.assert_allclose(row['action'],row['joint_command' if kind=='joint' else 'eef_command'])
    assert not np.allclose(row['action'][:6],row['observation.state'][:6])
    assert tuple(row['observation.images.cam_top'].shape)==(3,48,64)
    assert float(ds[9]['source_timestamp'])<100.4
    assert int(ds[10]['episode_index'])==1
    assert json.loads((output/'meta/pika_pipeline.json').read_text())['action_type']==kind
