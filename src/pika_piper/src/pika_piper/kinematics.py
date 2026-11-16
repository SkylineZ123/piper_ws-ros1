"""Headless Piper FK/IK based on AgileX PikaAnyArm, with send-side checks.

Reference: PikaAnyArm d2b8b84c756d8684cfdd3970741d32d89b248257,
piper/pika_remote_piper/scripts/forward_inverse_kinematics.py.
The joint bounds and IPOPT iteration budget follow upstream.
The posture cost is centered on the previous command to avoid zero-pose drift.
The objective uses smooth Cartesian/chordal errors; log6 is diagnostic only.
Default acceptance follows upstream's per-axis 0.3 m position gate without an
angular rejection. The former 2 mm / 2 deg gate is available as explicit strict mode.
The configured Piper TCP and SRDF collision exclusions describe our hardware.
"""
import hashlib
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

try:
    import casadi as ca
    from pinocchio import casadi as cpin
except ImportError as error:
    raise ImportError('Piper IK requires CasADi and pinocchio.casadi; activate pika-bringup') from error

from .core import transform


REFERENCE_COMMIT = 'd2b8b84c756d8684cfdd3970741d32d89b248257'
POSITION_TOLERANCE_M = .002
ORIENTATION_TOLERANCE_RAD = np.deg2rad(2.)


class Kinematics:
    def __init__(self, urdf, package_dir, tcp, srdf, posture_weight=1e-4,
                 feedback_limits_deg=None, feedback_tolerance=0.,
                 near_zero_start_tolerance_deg=0., pose_acceptance='official'):
        if pose_acceptance not in ('official', 'strict'):
            raise ValueError('pose_acceptance must be official or strict')
        self.pose_acceptance = pose_acceptance
        if not np.isfinite(posture_weight) or posture_weight <= 0:
            raise ValueError('IK posture weight must be finite and positive')
        self.posture_weight = float(posture_weight)
        self.urdf_hash = hashlib.sha256(Path(urdf).read_bytes()).hexdigest()
        self.tcp = transform(tcp)
        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(urdf), package_dirs=[str(Path(package_dir).resolve().parent)])
        self.full = self.robot.model
        self.model = pin.buildReducedModel(
            self.full, [self.full.getJointId(n) for n in ('joint7', 'joint8')],
            pin.neutral(self.full))
        if self.model.nq != 6 or self.full.nq != 8:
            raise ValueError('Expected a six-axis Piper with joint7/joint8 gripper')
        self.feedback_lower = self.model.lowerPositionLimit.copy()
        self.feedback_upper = self.model.upperPositionLimit.copy()
        if not np.isfinite(feedback_tolerance) or not 0 <= feedback_tolerance <= .002:
            raise ValueError('Feedback tolerance must be in [0, 0.002] rad')
        if feedback_limits_deg is None:
            if feedback_tolerance:
                raise ValueError('Feedback tolerance requires verified hardware limits')
        else:
            limits = np.asarray(feedback_limits_deg, dtype=float)
            if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
                raise ValueError('Verified feedback limits require six finite [min, max] degree pairs')
            limits = np.deg2rad(limits)
            self.feedback_lower = np.maximum(self.feedback_lower - feedback_tolerance, limits[:, 0])
            self.feedback_upper = np.minimum(self.feedback_upper + feedback_tolerance, limits[:, 1])
            if np.any(self.feedback_lower >= self.feedback_upper):
                raise ValueError('Verified hardware and model limits do not overlap')
        if (not np.isfinite(near_zero_start_tolerance_deg) or
                not 0 <= near_zero_start_tolerance_deg <= .5):
            raise ValueError('Near-zero startup tolerance must be in [0, 0.5] deg')
        self.start_feedback_lower = self.feedback_lower.copy()
        self.start_feedback_upper = self.feedback_upper.copy()
        if near_zero_start_tolerance_deg:
            if feedback_limits_deg is None:
                raise ValueError('Near-zero startup requires verified nominal hardware limits')
            if not (np.isclose(self.feedback_lower[1], 0.) and
                    np.isclose(self.feedback_upper[2], 0.)):
                raise ValueError('Near-zero startup requires joint2 lower/joint3 upper at zero')
            allowance = np.deg2rad(near_zero_start_tolerance_deg)
            self.start_feedback_lower[1] = -allowance
            self.start_feedback_upper[2] = allowance

        # As in upstream, place the tool frame on joint6. Our TCP is specified
        # relative to gripper_base, not the custom Ry(-1.57) @ Tx(0.19) tool.
        base = self.model.frames[self.model.getFrameId('gripper_base')]
        joint = self.model.getJointId('joint6')
        if base.parentJoint != joint:
            raise ValueError('gripper_base must be fixed to joint6')
        self.tool = base.placement * pin.SE3(self.tcp)
        self.frame = self.model.addFrame(pin.Frame(
            'pika_tcp', joint, self.tool, pin.FrameType.OP_FRAME))
        self.data = self.model.createData()

        self.geometry = self.robot.collision_model
        self.geometry.addAllCollisionPairs()
        excluded = {frozenset((e.attrib['link1'], e.attrib['link2']))
                    for e in ET.parse(srdf).getroot().findall('disable_collisions')}
        pairs = [(int(p.first), int(p.second)) for p in self.geometry.collisionPairs]
        for first, second in pairs:
            a, b = (self.geometry.geometryObjects[first],
                    self.geometry.geometryObjects[second])
            names = frozenset((self.full.frames[a.parentFrame].name,
                               self.full.frames[b.parentFrame].name))
            if names in excluded:
                self.geometry.removeCollisionPair(pin.CollisionPair(first, second))
        self.geometry_data = self.geometry.createData()
        self.last_ik = None
        self.info = {
            'backend': 'pinocchio_casadi_ipopt', 'reference_commit': REFERENCE_COMMIT,
            'pinocchio': pin.__version__, 'casadi': ca.__version__,
            'tcp_parent': 'gripper_base', 'tcp_xyzrpy': list(tcp),
            'pose_acceptance': pose_acceptance,
            'position_tolerance_m': .3 if pose_acceptance == 'official' else POSITION_TOLERANCE_M,
            'position_tolerance_metric': 'maximum_axis' if pose_acceptance == 'official' else 'se3_log_norm',
            'orientation_tolerance_rad': None if pose_acceptance == 'official' else float(ORIENTATION_TOLERANCE_RAD),
            'posture_reference': 'seed_previous_command', 'ipopt_tolerance': 1e-8,
            'objective_error': 'cartesian_position_chordal_rotation',
            'posture_weight': self.posture_weight,
            'feedback_tolerance_rad': float(feedback_tolerance),
            'feedback_lower_rad': self.feedback_lower.tolist(),
            'feedback_upper_rad': self.feedback_upper.tolist(),
            'near_zero_start_tolerance_deg': float(near_zero_start_tolerance_deg),
            'start_feedback_lower_rad': self.start_feedback_lower.tolist(),
            'start_feedback_upper_rad': self.start_feedback_upper.tolist(),
        }
        self._build_solver()

    def _build_solver(self):
        model = cpin.Model(self.model)
        data = model.createData()
        q = ca.SX.sym('q', self.model.nq)
        target = ca.SX.sym('target', 4, 4)
        cpin.framesForwardKinematics(model, data, q)
        self.error = ca.Function('piper_pose_error', [q, target], [
            cpin.log6(data.oMf[self.frame].inverse() * cpin.SE3(target)).vector])
        self.opti = ca.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.param_seed = self.opti.parameter(self.model.nq)
        # Differentiate FK and matrix differences, not log6: Pinocchio 3.2's
        # symbolic log can produce NaN derivatives near identity rotations.
        # ||R-R_target||_F^2 / 2 approximates squared rotation angle locally.
        self.pose = ca.Function('piper_pose', [q], [data.oMf[self.frame].homogeneous])
        actual = self.pose(self.var_q)
        position_cost = ca.sumsqr(actual[:3, 3] - self.param_tf[:3, 3])
        orientation_cost = ca.sumsqr(actual[:3, :3] - self.param_tf[:3, :3]) / 2
        self.opti.minimize(20 * (position_cost + .01 * orientation_cost)
                           + self.posture_weight * ca.sumsqr(self.var_q - self.param_seed))
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit, self.var_q, self.model.upperPositionLimit))
        self.opti.solver('ipopt', {
            'ipopt': {'print_level': 0, 'max_iter': 50, 'tol': 1e-8, 'sb': 'yes'},
            'print_time': False})

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        if q.ndim != 1 or len(q) < 6 or not np.isfinite(q[:6]).all():
            raise ValueError('FK requires six finite joint angles in radians')
        pin.framesForwardKinematics(self.model, self.data, q[:6])
        return self.data.oMf[self.frame].homogeneous.copy()

    def collision(self, q, width):
        fullq = np.r_[np.asarray(q)[:6], width / 2, -width / 2]
        pin.updateGeometryPlacements(self.full, self.robot.data, self.geometry,
                                     self.geometry_data, fullq)
        return pin.computeCollisions(self.geometry, self.geometry_data, False)

    def validate_configuration(self, q, context='configuration', _feedback=False, _startup=False):
        """Reject invalid measured/commanded states; never repair them silently."""
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError(context + ': expected six finite angles and gripper width')
        violations = []
        lower = self.feedback_lower if _feedback else self.model.lowerPositionLimit
        upper = self.feedback_upper if _feedback else self.model.upperPositionLimit
        if _startup:
            lower, upper = self.start_feedback_lower, self.start_feedback_upper
        for i, (value, lower, upper) in enumerate(zip(q[:6], lower, upper), 1):
            if not lower <= value <= upper:
                violations.append('joint%d=%.3f deg outside [%.3f, %.3f] deg' %
                                  (i, *np.rad2deg([value, lower, upper])))
        if not 0 <= q[6] <= .07:
            violations.append('gripper=%.6f m outside [0, 0.07] m' % q[6])
        if violations:
            raise ValueError(context + ': ' + '; '.join(violations))
        if self.collision(q, q[6]):
            raise ValueError(context + ': self-collision')

    def validate_feedback(self, q, context='feedback'):
        """Accept only the audited intersection of hardware and a small model band.

        Raw feedback is preserved. Command validation always uses the unchanged
        model limits, including the first command during inward recovery.
        """
        self.validate_configuration(q, context, _feedback=True)

    def validate_teleop_start(self, q, context='teleop start feedback'):
        """Permit only bounded near-zero measurement deviation at acquisition."""
        self.validate_configuration(q, context, _startup=True)

    def start_target(self, feedback):
        """Plan the nearest legal first target, preserving the measured input."""
        self.validate_teleop_start(feedback)
        target = np.asarray(feedback, dtype=float).copy()
        target[:6] = np.clip(target[:6], self.model.lowerPositionLimit, self.model.upperPositionLimit)
        self.validate_configuration(target, 'teleop startup target')
        return target

    def solve_ik(self, target, seed, width, posture_reference=None):
        """Return an optimization candidate, not send authorization.

        This entry point permits offline study of out-of-range seeds.
        Use ik() and validate_configuration() in the control path.
        """
        self.last_ik = None
        target = np.asarray(target, dtype=float)
        seed = np.asarray(seed, dtype=float)
        if (target.shape != (4, 4) or not np.isfinite(target).all()
                or not np.allclose(target[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(target[:3, :3]), 1., atol=1e-6)):
            raise ValueError('IK target must be a finite rigid 4x4 transform')
        if seed.ndim != 1 or len(seed) < 6 or not np.isfinite(seed[:6]).all():
            raise ValueError('IK seed requires six finite joint angles')
        posture_reference = seed if posture_reference is None else np.asarray(posture_reference, dtype=float)
        if (posture_reference.ndim != 1 or len(posture_reference) < 6 or
                not np.isfinite(posture_reference[:6]).all()):
            raise ValueError('IK posture reference requires six finite joint angles')
        if not np.isfinite(width) or not 0 <= width <= .07:
            raise ValueError('IK gripper width outside [0, 0.07] m')
        self.opti.set_initial(self.var_q, seed[:6])
        self.opti.set_value(self.param_seed, posture_reference[:6])
        self.opti.set_value(self.param_tf, target)
        started = time.perf_counter()
        try:
            solution = self.opti.solve_limited()
        except RuntimeError as error:
            self.last_ik = {'accepted': False, 'solver_success': False,
                            'solver_status': str(self.opti.stats().get('return_status', 'exception')),
                            'solve_ms': (time.perf_counter() - started) * 1000,
                            'reason': str(error).splitlines()[-1]}
            raise ValueError('IK IPOPT solve failed: ' + self.last_ik['reason']) from error
        stats = self.opti.stats()
        q = np.asarray(solution.value(self.var_q)).reshape(6)
        # Only remove IPOPT's numerical bound overshoot, never a real violation.
        clipped = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        bounds_ok = bool(np.isfinite(q).all() and np.max(abs(q - clipped)) <= 1e-7)
        if bounds_ok:
            q = clipped
        error = np.asarray(self.error(q, target)).reshape(6)
        actual = self.fk(q)
        position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        orientation = float(np.linalg.norm(error[3:]))
        self.last_ik = {
            'accepted': False, 'solver_status': str(stats['return_status']),
            'solver_success': bool(stats['success']), 'bounds_ok': bounds_ok,
            'iterations': int(stats['iter_count']),
            'solve_ms': (time.perf_counter() - started) * 1000,
            'log_position_error_m': float(np.linalg.norm(error[:3])),
            'position_error_m': position, 'orientation_error_rad': orientation,
            'position_axis_error_m': float(np.max(np.abs(actual[:3, 3] - target[:3, 3]))),
            'objective': float(solution.value(self.opti.f)), 'q_rad': q.tolist(),
            'seed_rad': seed[:6].tolist(), 'target_matrix': target.tolist(),
        }
        if not stats['success'] or not bounds_ok or not np.isfinite(error).all():
            raise ValueError('IK IPOPT candidate rejected: ' + str(stats['return_status']))
        return np.r_[q, width]

    def ik(self, target, seed, width, posture_reference=None):
        q = self.solve_ik(target, seed, width, posture_reference)
        return self.accept_ik(q)

    def accept_ik(self, q):
        """Apply the configured upstream/strict pose gate and geometry checks."""
        error = self.last_ik
        # Outer checks remain independent of the optimization objective.
        # Convergence alone does not imply accurate or safe tracking.
        if self.pose_acceptance == 'official':
            # Upstream piper_IK.py checks each Cartesian axis against 0.3 m;
            # its angular rejection is commented out. Keep errors observable.
            if error['position_axis_error_m'] > .3:
                raise ValueError('IK position axis residual %.3f mm exceeds 300.000 mm' %
                                 (error['position_axis_error_m'] * 1000))
        elif error['log_position_error_m'] > POSITION_TOLERANCE_M:
            raise ValueError('IK position residual %.3f mm exceeds 2.000 mm (IPOPT %s)' %
                             (error['log_position_error_m'] * 1000, error['solver_status']))
        if self.pose_acceptance == 'strict' and error['orientation_error_rad'] > ORIENTATION_TOLERANCE_RAD:
            raise ValueError('IK orientation residual %.3f deg exceeds 2.000 deg (IPOPT %s)' %
                             (np.rad2deg(error['orientation_error_rad']), error['solver_status']))
        self.validate_configuration(q, 'IK command')
        error['accepted'] = True
        return q
