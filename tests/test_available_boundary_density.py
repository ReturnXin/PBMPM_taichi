"""Available solid-union volume and ratio of matching particle gathers."""
import unittest
from unittest.mock import patch
import numpy as np
import taichi as ti

from rigid_coupling import cli
from rigid_coupling.config import Config
from rigid_coupling.scenes import Scene,lattice
from rigid_coupling.rigid import make_body
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID,RIGID,particle_arrays,combine_particles
from tests.test_boundary_density import gather


def box_integral(lo,hi,grid=32):
    def cdf(x):
        return np.select([x<=-1.5,x<-.5,x<.5,x<1.5],
                         [0,(x+1.5)**3/6,.5+.75*x-x**3/3,1-(1.5-x)**3/6],default=1)
    node=np.indices((grid,)*3)
    value=np.ones((grid,)*3)
    for a in range(3):
        value*=np.maximum(0,cdf(hi[a]*grid-node[a])-cdf(lo[a]*grid-node[a]))
    return value


class AvailableRoutingChecks(unittest.TestCase):
    def test_explicit_available_mode(self):
        with patch("rigid_coupling.shake.run") as run:
            cli.main(["--scene","shake_water","--boundary-density","available"])
            self.assertEqual(run.call_args.args[1][0].boundary_density,"available")
        with self.assertRaises(SystemExit):
            cli.main(["--scene","shake_soft","--boundary-density","available"])


class AvailableDensityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu,cpu_max_num_threads=1,offline_cache=False)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_union_clipping_cache_and_water_only_numerator(self):
        c=Config(gravity=0,viscosity=0,fluid_relaxation=0,rigid_contact=True,
                 world_contact=True,grid_coupling="separate",boundary_density="available")
        size=np.array([.12,.07,.1]); bodies=[]; groups=[]
        centers=[np.array([.48,.30,.5]),np.array([.51,.30,.5])]
        x=lattice((.5,.335,.5),(.5,.25,.46),(32,16,30))
        outside=np.ones(len(x),bool)
        for center in centers:outside &= ~np.all(np.abs(x-center)<size/2,axis=1)
        water=particle_arrays(x[outside],FLUID,.5*.25*.46/len(x),(0,0,0),c.dt,(0,0,1))
        groups.append(water)
        for i,center in enumerate(centers):
            body=make_body(lattice((0,0,0),size,4),center,1e-5); body.update(size=size,fixed=True)
            bodies.append(body)
            groups.append(particle_arrays(center+body["local"],RIGID,float(np.prod(size))/64,
                (0,0,0),c.dt,(0,1,0),body_id=i,local=body["local"],mass=body["particle_mass"]))
        s=Solver(c,Scene(combine_particles(groups),bodies,"overlapping geometry",
                         container=dict(center=[.5,.48,.5],half_size=[.25,.27,.23])))
        density=s.boundary_density; before=s.particles.snapshot()
        density.build()
        available=s.transfers.available_volume.to_numpy(); numerator=s.transfers.liquid_volume.to_numpy()
        exact=box_integral(np.array([.25,.21,.27]),np.array([.75,.75,.73]))
        exact-=box_integral(centers[0]-size/2,centers[0]+size/2)
        exact-=box_integral(centers[1]-size/2,centers[1]+size/2)
        exact+=box_integral(np.maximum(centers[0]-size/2,centers[1]-size/2),
                            np.minimum(centers[0]+size/2,centers[1]+size/2))
        np.testing.assert_allclose(available,exact,atol=2e-6)
        expected=.5*.54*.46-(.12+.03)*.07*.1
        self.assertAlmostEqual(float(available.sum()/32**3),expected,delta=3e-7)
        self.assertTrue(np.all((available>=0)&(available<=1)))
        for key in ("x","d","D","mass","rest_volume","volume_ratio"):
            np.testing.assert_array_equal(s.particles.snapshot()[key],before[key])
        # The complete wall and union term stays in the denominator. Changing
        # rigid sample rest volumes cannot affect either field.
        changed=s.particles.snapshot(); changed["rest_volume"][changed["material"]==RIGID]*=7
        s.particles.data.from_numpy(changed); density.build()
        np.testing.assert_array_equal(s.transfers.available_volume.to_numpy(),available)
        np.testing.assert_array_equal(s.transfers.liquid_volume.to_numpy(),numerator)
        self.assertEqual(density.rebuild_available[None],0)
        s.particles.data.from_numpy(before)
        sample=np.array([[.25,.21,.27],[.5,.21,.5],[.42,.30,.5],[.57,.30,.5]])
        corrected=gather(numerator,sample,32)/gather(available,sample,32)
        np.testing.assert_allclose(corrected[:2],1,atol=.02)
        # Carving whole discrete sample cells at a sharp body boundary does
        # not make exact uniform water. Verify the algorithm against the
        # independent geometric denominator, without assuming q=1 there.
        np.testing.assert_allclose(corrected,gather(numerator,sample,32)/gather(exact,sample,32),atol=2e-6)
        air=np.array([[.5,.6,.5]])
        self.assertLess(gather(numerator,air,32)[0]/gather(available,air,32)[0],1e-6)
        # Geometry touching/extending through a wall contributes only its
        # union inside the container, without double-counting outside space.
        b=s.bodies.data.to_numpy(); b["x"][0]=[.74,.3,.5]; b["x"][1]=[.9,.3,.5]
        s.bodies.data.from_numpy(b); s.bodies.reconstruct(); density.build()
        self.assertEqual(density.rebuild_available[None],1)
        expected=.5*.54*.46-(.75-(.74-.06))*.07*.1
        self.assertAlmostEqual(float(s.transfers.available_volume.to_numpy().sum()/32**3),expected,delta=3e-7)
        s.reset(); density.build()
        np.testing.assert_array_equal(s.transfers.available_volume.to_numpy(),available)
        s.prepare_iteration(); s.transfers.p2g()
        np.testing.assert_array_equal(s.transfers.liquid_volume.to_numpy(),numerator)

    def test_g2p_uses_ratio_of_gathers_and_skips_unresolved_denominator(self):
        c=Config(gravity=0,rigid_contact=True,grid_coupling="separate",boundary_density="available")
        p=particle_arrays([[.5003,.5007,.5011]],FLUID,1e-5,(0,0,0),c.dt,(0,0,1))
        s=Solver(c,Scene(p,[],"gather ratio",container=dict(center=[.5,.48,.5],half_size=[.25,.27,.23])))
        node=np.indices((32,)*3)
        a=(.15+.7*(node[0]%3)/2).astype(np.float32)
        n=(.5+a*.7).astype(np.float32)
        s.transfers.available_volume.from_numpy(a); s.transfers.liquid_volume.from_numpy(n)
        av=gather(a,p["x"],32)[0]; numerator=gather(n,p["x"],32)[0]
        s.transfers.g2p()
        actual=float(s.particles.snapshot()["volume_ratio"][0])
        self.assertAlmostEqual(actual,1+.1*av*(av/numerator-1),delta=2e-7)
        self.assertGreater(actual,.9+.1*av/numerator)
        wrong=float(gather(n/a,p["x"],32)[0])
        self.assertGreater(abs(wrong-numerator/av),.01)
        # Fully available support keeps the original strength, while a
        # correctly filled or underfilled support gets no density feedback.
        s.particles.data.from_numpy(p); s.transfers.available_volume.fill(1)
        s.transfers.liquid_volume.fill(1.2); s.transfers.g2p()
        self.assertAlmostEqual(float(s.particles.snapshot()["volume_ratio"][0]),.9+.1/1.2,delta=2e-7)
        p["volume_ratio"][:]=.8
        for numerator_value in (.5,.25):
            s.particles.data.from_numpy(p); s.transfers.available_volume.fill(.5)
            s.transfers.liquid_volume.fill(numerator_value); s.transfers.g2p()
            np.testing.assert_array_equal(s.particles.snapshot()["volume_ratio"],p["volume_ratio"])
        s.particles.data.from_numpy(p); s.transfers.available_volume.fill(0); s.transfers.g2p()
        np.testing.assert_array_equal(s.particles.snapshot()["volume_ratio"],p["volume_ratio"])


if __name__=="__main__":unittest.main()
