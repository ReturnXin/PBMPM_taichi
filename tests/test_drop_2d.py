"""Independent gravity reference and shared pool initial-state checks."""
import unittest
import numpy as np
import taichi as ti
from rigid_coupling_2d import Solver
from rigid_coupling_2d.drop import DropSpec,drop_config,pool,drop_scene,BODY_SIZE
from rigid_coupling_2d.scene import FLUID


class DropChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu,cpu_max_num_threads=1,offline_cache=False,debug=True)

    @classmethod
    def tearDownClass(cls):ti.reset()

    def test_free_fall_before_contact(self):
        c=drop_config();spec=DropSpec()
        a,b=drop_scene(c,spec,dry=True);s=Solver(c,a,b)
        n=200
        for _ in range(n):s.step()
        s.validate();state=s.body_state()
        exact_y=spec.release_height+c.gravity*c.dt**2*n*(n+1)/2
        np.testing.assert_allclose(state['velocity'],[0,c.gravity*n*c.dt],atol=2e-5,rtol=0)
        np.testing.assert_allclose(state['center'],[.5,exact_y],atol=2e-6,rtol=0)
        self.assertLess(abs(state['angle']),1e-6)
        np.testing.assert_allclose(s.x.to_numpy()-state['center'],b['local'],atol=2e-7,rtol=0)

    def test_shared_water_state(self):
        c=drop_config();spec=DropSpec();water=pool(c,spec)
        # A nonzero water state must survive insertion without reinitialization.
        water['d'][:,0]=1e-5;water['D'][:,0,1]=2e-4;water['J'][:]=.99
        a,b=drop_scene(c,spec,water);f=a['material']==FLUID
        for key in water:np.testing.assert_array_equal(a[key][f],water[key])
        self.assertGreater(b['center'][1]-BODY_SIZE[1]/2-water['x'][:,1].max(),2*c.dx)
        self.assertEqual(int(f.sum()),23424)
        self.assertEqual(int((~f).sum()),1536)
        self.assertAlmostEqual(b['mass'],spec.body_density*np.prod(BODY_SIZE))


if __name__=='__main__':unittest.main()
