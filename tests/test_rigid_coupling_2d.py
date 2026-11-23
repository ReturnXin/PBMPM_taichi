"""Analytical and conservation checks for the planar implementation."""
import unittest
import numpy as np
import taichi as ti
from rigid_coupling_2d import Config, Solver, build_scene
from rigid_coupling_2d.scene import particles, make_body, rotation_numpy, FLUID, RIGID
from rigid_coupling_2d.metrics import measure, cross


class PlanarChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu,cpu_max_num_threads=1,offline_cache=False,debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_free_uniform_translation(self):
        c = Config(grid=64)
        # Sparse patch: occupancy < 1, isolates transfer from density feedback.
        x = np.array([[.3+i*.008,.4+j*.008] for i in range(10) for j in range(10)])
        a = particles(x,FLUID,1e-5,c.dt,velocity=(.37,-.12))
        s = Solver(c,a)
        e0 = measure(s)
        for _ in range(25): s.step()
        s.validate()
        out = s.snapshot()
        np.testing.assert_allclose(out['x'],a['x']+25*c.dt*np.array([.37,-.12]),atol=2e-6,rtol=0)
        np.testing.assert_allclose(out['d']/c.dt,np.tile([.37,-.12],(len(x),1)),atol=2e-5,rtol=0)
        self.assertLess(abs(measure(s)['kinetic']/e0['kinetic']-1),2e-4)

    def test_known_particle_impulses_and_reset(self):
        local = np.array([[x,y] for x in (-.04,.04) for y in (-.08,.08)])
        c = Config(grid=64)
        for angle in (0.,.63):
            b = make_body(local,(.6,.5),.002,angle=angle,size=(.12,.2))
            a = particles(b['center']+local@rotation_numpy(angle).T,RIGID,.002,c.dt,local=local)
            s = Solver(c,a,b)
            # Inject a known grid-return displacement into the actual feedback path.
            for eccentric in (False,True):
                s.reset()
                impulse = np.array([.0003,-.0001])
                pred = s.predicted_d.to_numpy()
                returned = pred.copy()
                if eccentric:
                    returned[0] += c.dt*impulse/a['mass'][0]
                    torque = cross(local[0]@rotation_numpy(angle).T,impulse)
                else:
                    returned += c.dt*impulse/b['mass']
                    torque = 0.
                s.d.from_numpy(returned)
                s.collect_feedback()
                s.apply_feedback()
                np.testing.assert_allclose(s.velocity[None].to_numpy(),impulse/b['mass'],atol=2e-7,rtol=5e-5)
                self.assertAlmostEqual(s.omega[None],torque/b['inertia'],delta=2e-6)
                s.integrate();s.rebuild()
                self.assertLess(measure(s)['shape_error'] if np.any(a['material']==FLUID) else
                                np.max(abs(s.x.to_numpy()-(s.center[None].to_numpy()+local@rotation_numpy(s.angle[None]).T))),2e-7)
            s.reset()
            np.testing.assert_allclose(s.x.to_numpy(),a['x'],atol=1e-7)
            np.testing.assert_array_equal(s.velocity[None].to_numpy(),0)

    def test_rotated_contact_pair_and_slip(self):
        c = Config(grid=64)
        angle = .37
        R = rotation_numpy(angle)
        b = make_body(np.array([[-.04,-.08],[-.04,.08],[.04,-.08],[.04,.08]]),(.6,.5),.002,angle=angle,size=(.12,.2))
        x = b['center']+R@np.array([-.0602,.025])
        a = particles([x],FLUID,1e-5,c.dt,velocity=R@np.array([.4,.13]))
        s = Solver(c,a,b)
        before = s.d.to_numpy().copy()
        s.prepare();s.p2g();s.project_contact()
        delta = s.d.to_numpy()[0]-before[0]
        reaction = s.feedback[None].to_numpy()
        np.testing.assert_allclose(a['mass'][0]*delta/c.dt+reaction,0,atol=1e-10)
        self.assertAlmostEqual(float(delta@R[:,1]),0,delta=1e-9)
        s.apply_feedback()
        center = s.center[None].to_numpy()+c.dt*s.velocity[None].to_numpy()
        rot = rotation_numpy(s.angle[None]+c.dt*s.omega[None])
        local = (s.x.to_numpy()[0]+s.d.to_numpy()[0]-center)@rot
        self.assertLessEqual(local[0],-.06+2e-7)
        projected = s.d.to_numpy().copy()
        s.scatter_contact();s.gather_contact()
        np.testing.assert_array_equal(s.d.to_numpy(),projected)
        np.testing.assert_allclose(s.D.to_numpy(),s.D.to_numpy().transpose(0,2,1),atol=1e-8)

    def test_matched_initial_states(self):
        for prefix in ('center','offset'):
            a,b = build_scene(Config(coupling='one_way'),prefix+'_one_way')
            c,d = build_scene(Config(),prefix+'_two_way')
            for k in a: np.testing.assert_array_equal(a[k],c[k])
            self.assertEqual(b['mass'],d['mass'])


if __name__ == '__main__': unittest.main()
