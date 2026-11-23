"""Unit solid occupancy is geometric; only real samples estimate water."""
import unittest
from unittest.mock import patch
import numpy as np
import taichi as ti

from rigid_coupling import cli
from rigid_coupling.config import Config
from rigid_coupling.container import roll_matrix
from rigid_coupling.scenes import Scene,lattice
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID,particle_arrays
from tests.test_boundary_density import gather


class SolidRoutingChecks(unittest.TestCase):
    def test_explicit_solid_mode(self):
        with patch("rigid_coupling.shake.run") as run:
            cli.main(["--scene","shake_water","--boundary-density","solid"])
            self.assertEqual(run.call_args.args[1][0].boundary_density,"solid")
        with self.assertRaises(SystemExit):
            cli.main(["--scene","shake_soft","--boundary-density","solid"])


class SolidDensityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu,cpu_max_num_threads=1,offline_cache=False)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_volume_partition_pose_cache_and_free_surface_without_impulse(self):
        c=Config(gravity=0,viscosity=0,fluid_relaxation=0,rigid_contact=True,
                 world_contact=True,grid_coupling="separate",boundary_density="solid")
        center=np.array([.5,.48,.5]); size=np.array([.5,.25,.46])
        local=lattice((0,-.27+.125,0),size,(32,16,30))
        p=particle_arrays(center+local,FLUID,float(np.prod(size))/len(local),
                          (0,0,0),c.dt,(.2,.5,.9))
        scene=Scene(p,[],"solid occupancy",container=dict(center=center.tolist(),half_size=[.25,.27,.23]))
        s=Solver(c,scene); density=s.boundary_density
        previous=None
        for angle,shift in ((0.,0.),(.18,.0),(-.18,.02)):
            with self.subTest(angle=angle):
                R=roll_matrix(angle); center[0]=.5+shift
                s.container.motion.center=center.copy(); s.container.motion.angle=angle
                s.container.publish()
                before=s.particles.snapshot(); before["x"][:]=center+local@R.T
                s.particles.data.from_numpy(before)
                before_mass=s.transfers.mass.to_numpy()
                density.build()
                self.assertEqual(density.rebuild[None],1)
                solid=density.solid_volume.to_numpy()
                self.assertTrue(np.all((solid>=0)&(solid<=1)))
                self.assertAlmostEqual(float((1-solid).sum()/c.grid**3),.5*.54*.46,delta=3e-5)
                if previous is not None:
                    self.assertGreater(float(np.linalg.norm(solid-previous)),1.)
                after=s.particles.snapshot()
                for key in ("x","d","D","mass","rest_volume","volume_ratio"):
                    np.testing.assert_array_equal(after[key],before[key])
                np.testing.assert_array_equal(s.transfers.mass.to_numpy(),before_mass)
                samples=center+np.array([[0,-.27,0],[-.25,-.27,0],[-.25,-.27,-.23],[0,-.255,0]])@R.T
                packing=s.transfers.liquid_volume.to_numpy()
                np.testing.assert_allclose(gather(packing,samples,c.grid),1,atol=.035)
                surface=center+np.array([[0,-.02,0],[0,.13,0]])@R.T
                q=gather(packing,surface,c.grid)
                self.assertLess(q[0],.65); self.assertLess(q[1],1e-6)
                density.build(); self.assertEqual(density.rebuild[None],0)
                np.testing.assert_array_equal(density.solid_volume.to_numpy(),solid)
                scaled=s.particles.snapshot(); scaled["rest_volume"]*=1.2
                s.particles.data.from_numpy(scaled); density.build()
                np.testing.assert_array_equal(density.solid_volume.to_numpy(),solid)
                np.testing.assert_allclose(s.transfers.liquid_volume.to_numpy()-solid,
                                           1.2*(packing-solid),atol=3e-6)
                s.particles.data.from_numpy(before); density.build()
                s.prepare_iteration(); s.transfers.p2g()
                np.testing.assert_allclose(s.transfers.liquid_volume.to_numpy(),packing,atol=1e-6)
                previous=solid
        s.reset(); np.testing.assert_array_equal(s.transfers.liquid_volume.to_numpy(),0)
        density.build()
        self.assertEqual(density.rebuild[None],1)
        with self.assertRaisesRegex(ValueError,"container"):
            Solver(c,Scene(s.initial,[],"no box"))


if __name__=="__main__":
    unittest.main()
