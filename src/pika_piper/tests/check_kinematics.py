"""Run with the Pinocchio control interpreter; no pytest or hardware required."""
from pathlib import Path
import unittest
import numpy as np
from utils.kinematics import Kinematics

class KinematicChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        cls.k=Kinematics(root/'piper_description/urdf/piper_description.urdf',root/'piper_description',[0,0,.1358,0,0,0],root/'piper_moveit/piper_with_gripper_moveit/config/piper_description.srdf')
    def test_roundtrip(self):
        q=np.array([.1,.2,-.2,.1,.1,.1]);target=self.k.fk(q)
        solution=self.k.ik(target,q+.001,.03)
        self.assertLess(np.linalg.norm(self.k.fk(solution)[:3,3]-target[:3,3]),.002)
        self.assertLess(self.k.last_ik['orientation_error_rad'],np.deg2rad(2))
        self.assertFalse(self.k.collision(solution,.03))
    def test_unreachable(self):
        target=self.k.fk(np.zeros(6));target[:3,3]=100.
        with self.assertRaises(ValueError):self.k.ik(target,np.zeros(6),.03)
if __name__=='__main__':unittest.main()
