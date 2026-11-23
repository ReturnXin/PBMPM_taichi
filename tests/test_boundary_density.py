"""Reflected packing has no physical impulse and follows the actual box pose."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.container import roll_matrix
from rigid_coupling.scenes import Scene,lattice
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID,particle_arrays
from rigid_coupling import cli


def gather(field,x,grid):
    base = np.floor(x*grid-.5).astype(int); f = x*grid-base
    w = np.stack([.5*(1.5-f)**2,.75-(f-1)**2,.5*(f-.5)**2],axis=1)
    values = np.zeros(len(x))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                n = base+[a,b,c]
                values += w[:,a,0]*w[:,b,1]*w[:,c,2]*field[n[:,0],n[:,1],n[:,2]]
    return values


class RoutingChecks(unittest.TestCase):
    def test_water_default_is_raw_and_comparisons_are_available(self):
        for args,expected in (([],"raw"),(["--boundary-density","solid"],"solid"),(["--boundary-density","mirror"],"mirror")):
            with patch("rigid_coupling.shake.run") as run:
                cli.main(["--scene","shake_water",*args])
                self.assertEqual(run.call_args.args[1][0].boundary_density,expected)

    def test_soft_default_remains_raw(self):
        with patch("rigid_coupling.shake.run") as run:
            cli.main(["--scene","shake_soft"])
            self.assertEqual(run.call_args.args[1][0].boundary_density,"raw")


class BoundaryDensityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu,cpu_max_num_threads=1,offline_cache=False,
                offline_cache_file_path="results/shake_box_20261030/mirrored_boundary/test_cache")

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def fixture(self,angle=0.):
        c = Config(gravity=0,viscosity=0,fluid_relaxation=0,rigid_contact=True,
                   world_contact=True,grid_coupling="separate",boundary_density="mirror")
        center = np.array([.5,.48,.5]); size = np.array([.5,.25,.46])
        local = lattice((0,-.27+.125,0),size,(32,16,30))
        R = roll_matrix(angle)
        x = center+local@R.T
        p = particle_arrays(x,FLUID,float(np.prod(size))/len(x),(0,0,0),c.dt,(.2,.5,.9))
        scene = Scene(p,[],"uniform water packing",container=dict(center=center.tolist(),half_size=[.25,.27,.23]))
        s = Solver(c,scene)
        s.container.motion.angle = angle
        s.container.publish()
        return s,center,R

    def test_floor_corner_and_rotation_preserve_uniform_packing_without_impulse(self):
        for angle in (0.,.18):
            with self.subTest(angle=angle):
                s,center,R = self.fixture(angle)
                before = s.particles.snapshot()
                s.boundary_density.build()
                after = s.particles.snapshot()
                for key in ("x","d","D","mass","rest_volume"):
                    np.testing.assert_array_equal(before[key],after[key])
                np.testing.assert_array_equal(s.transfers.mass.to_numpy(),0)
                samples = center+np.array([[0,-.27,0],[-.25,-.27,0],[-.25,-.27,-.23],[0,-.27+.015,0]])@R.T
                density = gather(s.transfers.liquid_volume.to_numpy(),samples,s.config.grid)
                np.testing.assert_allclose(density,1,atol=.035)
                # There is no solid at the water's top: retain the lower
                # free-surface packing estimate rather than reflecting there.
                surface = center+np.array([[0,-.02,0]])@R.T
                self.assertLess(float(gather(s.transfers.liquid_volume.to_numpy(),surface,s.config.grid)[0]),.65)
                expected = s.transfers.liquid_volume.to_numpy()
                s.prepare_iteration(); s.transfers.p2g()
                np.testing.assert_array_equal(s.transfers.liquid_volume.to_numpy(),expected)
                s.reset()
                np.testing.assert_array_equal(s.transfers.liquid_volume.to_numpy(),0)

    def test_rebuild_uses_new_wall_pose_and_rejects_unsupported_scenes(self):
        s,center,R = self.fixture()
        s.boundary_density.build()
        original = s.transfers.liquid_volume.to_numpy()
        s.container.motion.center[0] += .04
        p = s.particles.snapshot(); p["x"][:,0] += .04
        s.particles.data.from_numpy(p)
        s.container.publish(); s.boundary_density.build()
        self.assertGreater(float(np.linalg.norm(s.transfers.liquid_volume.to_numpy()-original)),1)
        scene = Scene(s.initial,[],"no solid box")
        with self.assertRaisesRegex(ValueError,"container"):
            Solver(s.config,scene)


if __name__ == "__main__":
    unittest.main()
