"""Floor-supported rigid body and gravity checks, independent of impact appearance."""
import unittest
import numpy as np
import taichi as ti
from rigid_coupling_2d import Solver
from rigid_coupling_2d.scene import RIGID,FLUID,make_body,particles,rotation_numpy
from rigid_coupling_2d.falling import falling_config,falling_scene,BODY_CENTER


class FallingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ti.init(arch=ti.cpu,cpu_max_num_threads=1,offline_cache=False,debug=True)
    @classmethod
    def tearDownClass(cls):ti.reset()

    def test_supported_body_stays_at_rest(self):
        c=falling_config()
        a,b=falling_scene(c,body_only=True);s=Solver(c,a,b)
        for _ in range(100):s.step()
        s.validate();state=s.body_state()
        np.testing.assert_allclose(state['center'],BODY_CENTER,atol=2e-6,rtol=0)
        np.testing.assert_allclose(state['velocity'],0,atol=2e-5)
        self.assertLess(abs(state['angle']),1e-5)
        self.assertAlmostEqual(s.wall_impulse[None][1],-b['mass']*c.gravity*100*c.dt,delta=2e-6)

    def test_rotated_body_floor_impulse(self):
        c=falling_config()
        a,b=falling_scene(c,body_only=True);s=Solver(c,a,b)
        s.angle[None]=.3;s.velocity[None]=(0,-1)
        lo=c.origin+c.boundary*c.dx
        extent=np.abs(rotation_numpy(.3))@np.array(s.half_size)
        s.center[None]=(.5,lo+extent[1])
        before=s.body_state();s.body_walls();after=s.body_state()
        self.assertGreaterEqual(after['center'][1]-extent[1],lo-2e-7)
        impulse=s.wall_impulse[None].to_numpy()
        np.testing.assert_allclose(b['mass']*(after['velocity']-before['velocity']),impulse,atol=1e-8)
        self.assertGreater(np.linalg.norm(impulse),0)
        self.assertGreater(abs(after['omega']),0)

    def test_materials_share_rest_initial_state(self):
        c=falling_config();base,_=falling_scene(c)
        for material in ('water','elastic','sand'):
            a,b=falling_scene(c,material)
            for key in ('x','d','mass','area','local'):np.testing.assert_array_equal(a[key],base[key])
            np.testing.assert_array_equal(a['d'],0)
            f=a['material']!=RIGID
            self.assertGreater(a['x'][f,1].min(),a['x'][~f,1].max())
            self.assertGreater(a['x'][f,0].mean(),b['center'][0])

    def test_box_impulse_balances_particle_momentum(self):
        c=falling_config(gravity=0.)
        lo=c.origin+c.boundary*c.dx
        a=particles([[lo+.0001,.4]],FLUID,1e-5,c.dt,velocity=(-1.,0.))
        s=Solver(c,a);before=a['mass'][0]*a['d'][0]/c.dt
        for _ in range(3):s.step()
        s.validate();after=a['mass'][0]*s.d.to_numpy()[0]/c.dt
        np.testing.assert_allclose(after-before,s.wall_impulse[None].to_numpy(),atol=2e-10)
        self.assertGreater(s.wall_impulse[None][0],0)
        self.assertGreaterEqual(s.x.to_numpy()[0,0],lo-1e-7)

    def test_shared_body_floor_face_uses_short_projection(self):
        c=falling_config(gravity=0.)
        _,body=falling_scene(c,body_only=True)
        lo=c.origin+c.boundary*c.dx
        a=particles([[.5,lo+1e-5]],FLUID,1e-5,c.dt)
        s=Solver(c,a,body);s.prepare();s.project_contact()
        delta=s.d.to_numpy()[0]
        self.assertLess(abs(delta[0]),1e-7)
        self.assertLess(abs(delta[1]),2e-5)


if __name__=='__main__':unittest.main()
