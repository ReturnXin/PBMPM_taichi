"""Matched circular materials and unconstrained rigid free fall."""
import unittest
import numpy as np
import taichi as ti
from rigid_coupling_2d import Solver
from rigid_coupling_2d.scene import RIGID
from rigid_coupling_2d.rigid_drop import (rigid_drop_config, rigid_drop_scene,
                                        BODY_CENTER, BODY_SIZE, MATERIAL_RADIUS)


class RigidDropChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_identical_circles_and_positive_release_gap(self):
        c=rigid_drop_config();base,b=rigid_drop_scene(c)
        for material in ('water','elastic','sand'):
            a,body=rigid_drop_scene(c,material)
            for key in ('x','mass','area','d','D','local'):
                np.testing.assert_array_equal(a[key],base[key])
            self.assertEqual(body['mass'],b['mass'])
            np.testing.assert_array_equal(body['velocity'],0)
            mask=a['material']!=RIGID
            self.assertEqual(int(mask.sum()),4064)
            self.assertEqual(int((~mask).sum()),4608)
            self.assertGreater(a['x'][~mask,1].min(),a['x'][mask,1].max())
            lo=c.origin+c.boundary*c.dx
            self.assertGreaterEqual(a['x'][mask].min(),lo)
            self.assertLessEqual(a['x'][mask,1].max(),lo+2*MATERIAL_RADIUS)
            self.assertAlmostEqual(a['x'][mask,0].mean(),.5,places=6)

    def test_dry_body_matches_discrete_freefall(self):
        c=rigid_drop_config();a,b=rigid_drop_scene(c,dry=True);s=Solver(c,a,b)
        n=100
        for _ in range(n):s.step()
        s.validate();state=s.body_state()
        expected=BODY_CENTER+np.array([0.,c.gravity*c.dt**2*n*(n+1)/2])
        np.testing.assert_allclose(state['center'],expected,atol=2e-6,rtol=0)
        np.testing.assert_allclose(state['velocity'],[0.,c.gravity*c.dt*n],atol=2e-5,rtol=0)
        self.assertLess(abs(state['angle']),1e-5)
        np.testing.assert_array_equal(s.wall_impulse[None].to_numpy(),0)
        np.testing.assert_array_equal(s.contact_impulse[None].to_numpy(),0)


if __name__=='__main__':unittest.main()
