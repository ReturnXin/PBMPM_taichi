"""Floor-only support must not modify free-floating or already supported bodies."""
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np
import taichi as ti

from rigid_coupling.buoyancy import BuoyancyAssist
from rigid_coupling.cli import main
from rigid_coupling.config import Config
from rigid_coupling.container import MovingContainer, roll_matrix
from rigid_coupling.rigid import RigidBodies, make_body, rotation_numpy
from rigid_coupling.state import FLUID, RIGID, Particles, combine_particles, particle_arrays


def fixture(*, density=.5, gap=0., water=True, fixed=False, angle=0., box_angle=0., container=True):
    c=Config(buoyancy_assist=True, world_contact=True, rigid_contact=True)
    size=np.array([.10,.08,.10]); half=size/2
    q=np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)])
    R=rotation_numpy(q)
    box_R=roll_matrix(box_angle)
    floor=.21 if container else c.boundary/c.grid
    center=np.array([.5,floor+np.sum(np.abs(R[1])*half)+gap,.5])
    if container:
        extent=np.abs(box_R.T@R)@half
        center=np.array([.5,.48,.5])+box_R@np.array([0.,-.27+extent[1]+gap,0.])
    samples=np.array(list(product((-1,1),repeat=3)))*size/4
    body=make_body(samples,center,density*np.prod(size)/len(samples),quaternion=q)
    body.update(size=size.tolist(),fixed=fixed)
    rigid=particle_arrays(body['center']+body['local']@R.T,RIGID,np.prod(size)/len(samples),
                          (0.,0.,0.),c.dt,(1.,0.,0.),body_id=0,local=body['local'],mass=body['particle_mass'])
    groups=[rigid]
    if water:
        groups.append(particle_arrays([[.5,floor+.05,.5]],FLUID,.05,(0.,0.,0.),c.dt,(0.,0.,1.)))
    p=Particles(combine_particles(groups))
    b=RigidBodies(p,c,[body])
    wall=MovingContainer(p,c,dict(center=[.5,.48,.5],half_size=[.25,.27,.23]),b) if container else None
    if wall is not None:
        wall.orientation[None]=box_R.astype(np.float32)
    return c,p,b,BuoyancyAssist(p,b,c,wall)


class BottomBuoyancyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu,cpu_max_num_threads=2,offline_cache=False,debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_only_submerged_light_unfixed_floor_contacts_receive_support(self):
        for options in ({}, {'gap':.06}, {'gap':.5/32}, {'density':1.}, {'density':1.3}, {'water':False},
                        {'fixed':True}, {'angle':.45,'box_angle':.25}, {'container':False}):
            with self.subTest(options=options):
                c,p,b,a=fixture(**options)
                a.begin();b.predict()
                before=b.data.to_numpy()
                a.apply()
                after=b.data.to_numpy()
                expected=(options.get('gap',0.)<=1/c.grid and options.get('density',.5)<1
                          and options.get('water',True) and not options.get('fixed',False))
                if expected:
                    self.assertGreater(float(a.correction[0]),0.)
                    self.assertGreater(float(after['v'][0,1]),0.)
                else:
                    np.testing.assert_array_equal(after['v'],before['v'])
                    self.assertEqual(float(a.correction[0]),0.)
                np.testing.assert_array_equal(after['omega'],before['omega'])
                np.testing.assert_array_equal(after['v'][:,[0,2]],before['v'][:,[0,2]])

    def test_existing_upward_contact_response_is_never_cancelled_or_double_counted(self):
        c,p,b,a=fixture()
        a.begin()
        values=b.data.to_numpy();values['v'][0]=[.2,.8,-.3];b.data.from_numpy(values)
        a.apply()
        np.testing.assert_array_equal(b.data.to_numpy()['v'],values['v'])
        self.assertEqual(float(a.correction[0]),0.)

    def test_serial_displaced_level_matches_independent_horizontal_box_reference(self):
        c,p,b,a=fixture(gap=.015)
        # Prescribe a partial immersion where successive displacement updates differ.
        a.begin();a.pool_volume[None]=.008
        values=b.data.to_numpy();height=float(values['x'][0,1]);half=values['half_size'][0].astype(float)
        band=half.min()/4
        y=height+(2*(np.arange(4)+.5)/4-1)*half[1]
        floor=float(a.container.center[None][1])-float(a.container.half_size[1])
        area=4*float(a.container.half_size[0])*float(a.container.half_size[2])
        water=float(a.pool_volume[None]);volume=float(8*half.prod())
        level=floor+water/area
        first=None
        for _ in range(4):
            fraction=np.clip(.5+(level-y)/(2*band),0,1).mean()
            level=floor+(water+volume*fraction)/area
            if first is None:first=level
        self.assertGreater(abs(level-first),1e-6)
        a.apply()
        self.assertAlmostEqual(float(a.level[None]),level,places=6)

    def test_leaving_floor_and_reset_clear_previous_correction(self):
        c,p,b,a=fixture()
        a.begin();b.predict();a.apply()
        self.assertGreater(float(a.correction[0]),0.)
        values=b.data.to_numpy();values['x'][0,1]+=.08;b.data.from_numpy(values)
        a.begin();a.apply()
        self.assertEqual(float(a.correction[0]),0.)
        np.testing.assert_array_equal(b.data.to_numpy()['v'],values['v'])
        a.reset()
        np.testing.assert_array_equal(a.fraction.to_numpy(),0.)
        np.testing.assert_array_equal(a.correction.to_numpy(),0.)


class BuoyancyRoutingChecks(unittest.TestCase):
    def test_water_default_and_explicit_switches(self):
        for args,expected in (([],True),(['--no-buoyancy-assist'],False),
                              (['--buoyancy-assist'],True),(['--scene','shake_soft'],False)):
            with self.subTest(args=args),patch('rigid_coupling.shake.run') as run:
                main(args)
                self.assertEqual(run.call_args.args[1][0].buoyancy_assist,expected)


if __name__=='__main__':
    unittest.main()
