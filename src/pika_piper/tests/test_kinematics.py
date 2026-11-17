from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
try:
    import pinocchio
except ImportError:
    pytest.skip('Pinocchio requires the separate control environment', allow_module_level=True)
from utils.kinematics import Kinematics


def test_regression_report_cannot_allow_send_from_invalid_feedback(monkeypatch):
    import importlib.util
    import types
    path=Path(__file__).parents[1]/'scripts/check_ik_regression.py'
    spec=importlib.util.spec_from_file_location('ik_report_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'legacy_solve',lambda *args:{})
    monkeypatch.setattr(module,'original_official_solve',lambda *args:{})
    def validate(q,*args):
        if q[1]<0:raise ValueError('joint2 outside limit')
    fake=types.SimpleNamespace(validate_configuration=validate,
        ik=lambda *args:None,last_ik={'q_rad':[0.]*6})
    seed=np.array([0.,-.01,0.,0.,0.,0.])
    report=module.evaluate(fake,'invalid_seed',seed,np.eye(4),.03,
        {'control_hz':50,'max_joint_speed':3,'max_gripper_speed':.2})
    assert report['first_limited_command_valid'] is True
    assert report['feedback_valid'] is False
    assert report['first_command_allowed'] is False

@pytest.fixture(scope='module')
def k():
    root=Path(__file__).resolve().parents[2]
    return Kinematics(root/'piper_description/urdf/piper_description.urdf',root/'piper_description',[0,0,.1358,0,0,0],root/'piper_moveit/piper_with_gripper_moveit/config/piper_description.srdf')

def test_reachable_pose_roundtrip_and_tcp(k):
    q=np.array([.1,.2,-.2,.1,.1,.1]);target=k.fk(q)
    solution=k.ik(target,q+.001,.03)
    actual=k.fk(solution)
    assert np.linalg.norm(actual[:3,3]-target[:3,3]) < .002
    assert Rotation.from_matrix(actual[:3,:3].T @ target[:3,:3]).magnitude() < np.deg2rad(2)
    assert k.last_ik['accepted']
    assert not k.collision(solution,.03)

def test_unreachable_target_is_rejected(k):
    target=k.fk(np.zeros(6));target[:3,3]=100.
    with pytest.raises(ValueError):k.ik(target,np.zeros(6),.03)


def test_previous_local_solver_failure_is_resolved(k):
    q=np.array([-.1559102621,.1253146403,0.,.5464451355,.1773254520,-.5023755719])
    target=k.fk(q);target[2,3]-=.005
    solution=k.ik(target,q,.0688)
    assert np.linalg.norm(k.fk(solution)[:3,3]-target[:3,3]) < .002
    k.validate_configuration(solution)


def test_observed_out_of_bounds_start_is_not_silently_repaired(k):
    q=np.deg2rad([-.352,-2.017,1.825,0,19.273,3.437])
    with pytest.raises(ValueError,match=r'joint2=-2.017.*joint3=1.825'):
        k.validate_configuration(np.r_[q,.03], 'teleop feedback')


def test_posture_cost_is_centered_on_previous_command(k):
    q=np.array([.1,.2,-.2,.1,.1,.1])
    solution=k.solve_ik(k.fk(q),q,.03)
    np.testing.assert_allclose(solution[:6],q,atol=1e-6)
    actual=k.fk(solution);target=k.fk(q)
    position_cost=np.sum((actual[:3,3]-target[:3,3])**2)
    orientation_cost=np.sum((actual[:3,:3]-target[:3,:3])**2)/2
    change=solution[:6]-q
    objective=20*(position_cost+.01*orientation_cost)+k.posture_weight*(change@change)
    assert k.last_ik['objective'] == pytest.approx(objective,abs=1e-10)
    assert k.info['backend']=='pinocchio_casadi_ipopt'
    assert k.info['posture_reference']=='seed_previous_command'


@pytest.mark.parametrize('q', [np.zeros(6),[.1,.2,-.2,.1,.1,.1],[0,.5,-.7,0,.4,0]])
def test_fk_matches_official_joint6_tool_composition(k,q):
    import pinocchio as pin
    data=k.model.createData()
    pin.forwardKinematics(k.model,data,np.asarray(q))
    expected=data.oMi[k.model.getJointId('joint6')].homogeneous @ k.tool.homogeneous
    np.testing.assert_allclose(k.fk(q),expected,atol=1e-12)
    # Preserve the pre-migration EEF dataset meaning independently of the new
    # tool frame: original full-model gripper_base followed by configured TCP.
    full_data=k.full.createData()
    pin.framesForwardKinematics(k.full,full_data,np.r_[q,0.,0.])
    previous_fk=full_data.oMf[k.full.getFrameId('gripper_base')].homogeneous @ k.tcp
    np.testing.assert_allclose(k.fk(q),previous_fk,atol=1e-12)


@pytest.mark.parametrize('q', [
    [0.,.5,-.7,0.,.4,0.],
    np.deg2rad([-.596,30.118,-23.212,-3.097,34.591,3.656]),
])
def test_stationary_pose_does_not_drift_toward_zero(k,q):
    # The second pose reproduces the user's 3.464 degree failure with the
    # former zero-centered objective. Repeated holds must not accumulate drift.
    q=np.asarray(q);seed=q.copy();target=k.fk(q)
    for _ in range(30):
        solution=k.ik(target,seed,.03)
        seed=solution[:6]
    np.testing.assert_allclose(seed,q,atol=1e-5)
    assert k.last_ik['position_error_m']<1e-6
    assert k.last_ik['orientation_error_rad']<1e-5


@pytest.mark.parametrize('axis',range(6))
@pytest.mark.parametrize('sign',[-1,1])
def test_small_translation_and_rotation_at_observed_pose(k,axis,sign):
    from utils.core import transform
    q=np.deg2rad([-.596,30.118,-23.212,-3.097,34.591,3.656])
    delta=np.zeros(6);delta[axis]=sign*(.001 if axis<3 else np.deg2rad(1))
    target=k.fk(q)@transform(delta)
    solution=k.ik(target,q,.03)
    k.validate_configuration(solution)
    assert k.last_ik['accepted']


def test_reachable_moving_goal_prioritizes_pose_over_posture(k):
    # A legal FK target formerly rejected with 2.712 mm / 4.520 degree errors
    # under the 0.01 previous-command posture weight.
    seed=np.array([.1,.5,-.7,.1,.4,.1])
    goal=np.array([.2100167168,.5453617363,-.9765622503,
                   -.3549964657,1.0239564615,-1.3078570230])
    k.validate_configuration(np.r_[goal,.03])
    k.ik(k.fk(goal),seed,.03)
    assert k.last_ik['position_error_m']<.0002
    assert k.last_ik['orientation_error_rad']<np.deg2rad(.2)


@pytest.mark.parametrize('target', [np.full((4,4),np.nan),np.zeros((4,4)),np.eye(3)])
def test_invalid_target_is_rejected_before_solver(k,target):
    with pytest.raises(ValueError,match='rigid'):
        k.ik(target,np.zeros(6),.03)
