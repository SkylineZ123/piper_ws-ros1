import json
from pathlib import Path
import numpy as np
import pytest
from utils.core import transform,relative_target,limit_step,align_series,interpolate_eef,read_trajectory
from utils.core import limit_pose_step
from utils.legacy_bag import aligned_episodes

def test_relative_mapping_and_reanchor():
    hand=transform([1,2,3,.1,.2,.3]);robot=transform([.2,.1,.3,0,.2,0]);axes=np.eye(4)
    np.testing.assert_allclose(relative_target(robot,hand,hand,axes),robot,atol=1e-12)
    delta=transform([.01,0,0,0,0,.1])
    np.testing.assert_allclose(relative_target(robot,hand,hand@delta,axes),robot@delta,atol=1e-12)

def test_speed_and_width_limits():
    result=limit_step(np.zeros(7),np.ones(7),.02)
    np.testing.assert_allclose(result[:6],.01);assert result[6]==pytest.approx(.0008)
    with pytest.raises(ValueError):limit_step(np.zeros(7),np.full(7,np.nan),.02)

def test_no_future_actions_or_state_extrapolation():
    series=(np.array([1.,1.1]),[np.zeros(7),np.ones(7)])
    assert align_series(series,.99,'command',.1) is None
    np.testing.assert_array_equal(align_series(series,1.05,'command',.1)[0],np.zeros(7))
    assert align_series(series,1.2,'command',.05) is None
    assert align_series(series,1.2,'state',.2) is None
    np.testing.assert_allclose(align_series(series,1.05,'state',.2)[0],.5)

def test_orientation_wrap():
    a=np.zeros(7);b=a.copy();a[5]=np.deg2rad(179);b[5]=np.deg2rad(-179)
    result=interpolate_eef(a,b,.5)
    assert abs(abs(result[5])-np.pi)<1e-6

def test_gap_splits_episodes():
    times=np.arange(0,1,.02)
    cfg={'fps':25,'cameras':{},'camera_tolerance':.02,'state_gap':.1,'command_age':.1}
    manifest={'config':cfg,'segments':[{'id':0,'start':0,'end':1,'accepted':True}]}
    times=times[(times<.4)|(times>.7)]
    streams={name:(times,[np.zeros(7) for t in times]) for name in ('state','eef_state','joint_command','eef_command')}
    episodes=list(aligned_episodes(manifest,streams))
    assert len(episodes)==2
    assert episodes[0][-1]['source_timestamp']<episodes[1][0]['source_timestamp']-.1

def test_replay_rejects_ambiguous_metadata(tmp_path):
    path=tmp_path/'trajectory.npz';np.savez(path,actions=np.zeros((3,7)),timestamps=np.arange(3))
    path.with_suffix('.json').write_text(json.dumps({'schema_version':1}))
    with pytest.raises(ValueError,match='action_type'):read_trajectory(path)


def test_invalid_control_config_rejected():
    from utils.core import validate_control_config
    import yaml
    config=yaml.safe_load((Path(__file__).parents[1]/'config/pipeline.yaml').read_text())
    validate_control_config(config)
    config['max_joint_speed']=-1
    with pytest.raises(ValueError,match='max_joint_speed'):validate_control_config(config)


def test_sense_forward_maps_to_piper_finger_direction_without_moving_anchor():
    # Official tool x-forward is stock Piper z-forward. This is a change of
    # axes, not an extra 135.8 mm/190 mm tool displacement.
    axes=transform([0,0,0,0,-np.pi/2,0])
    anchor=transform([.2,.1,.3,.1,.2,.3]);hand=transform([1,2,3,.2,.3,.4])
    np.testing.assert_allclose(relative_target(anchor,hand,hand,axes),anchor,atol=1e-12)
    moved=relative_target(anchor,hand,hand@transform([.01,0,0,0,0,0]),axes)
    np.testing.assert_allclose((np.linalg.inv(anchor)@moved)[:3,3],[0,0,.01],atol=1e-12)
    turned=relative_target(anchor,hand,hand@transform([0,0,0,.1,0,0]),axes)
    np.testing.assert_allclose(np.linalg.inv(anchor)@turned,transform([0,0,0,0,0,.1]),atol=1e-12)


def test_tcp_step_bounds_rotation_and_translation_with_one_fraction():
    from scipy.spatial.transform import Rotation
    previous=transform([.1,.2,.3,.2,.3,.4])
    target=previous@transform([.1,0,0,0,0,np.pi/2])
    step,limited=limit_pose_step(previous,target,.02,.1,.5)
    assert limited
    assert np.linalg.norm(step[:3,3]-previous[:3,3])<=.002+1e-12
    assert Rotation.from_matrix(previous[:3,:3].T@step[:3,:3]).magnitude()<=.01+1e-12
    np.testing.assert_allclose(limit_pose_step(previous,previous,.02,.1,.5)[0],previous,atol=1e-12)


@pytest.mark.parametrize('distance,expected',[(.001,False),(.1,True)])
def test_tcp_limit_flag_is_json_serializable_builtin_bool(distance,expected):
    target=np.eye(4);target[0,3]=distance
    _,limited=limit_pose_step(np.eye(4),target,.02,.15,1.)
    assert type(limited) is bool
    assert json.loads(json.dumps({'target_limited':limited}))['target_limited'] is expected
