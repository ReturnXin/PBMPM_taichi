"""Analytical material checks and shared-contact checks for planar solids."""
import unittest
import numpy as np
import taichi as ti
from rigid_coupling_2d import Config, Solver, build_scene
from rigid_coupling_2d.scene import RIGID, ELASTIC, SAND, particles, make_body
from rigid_coupling_2d.materials import elastic_target, sand_return


@ti.kernel
def targets(F: ti.types.ndarray(), out: ti.types.ndarray(), beta: ti.f32):
    A=ti.Matrix([[F[0,0],F[0,1]],[F[1,0],F[1,1]]])
    T=elastic_target(A,ti.Matrix.zero(ti.f32,2,2),beta)
    for i,j in ti.static(ti.ndrange(2,2)):out[i,j]=T[i,j]


@ti.kernel
def plastic(F: ti.types.ndarray(), out: ti.types.ndarray(), info: ti.types.ndarray()):
    A=ti.Matrix([[F[0,0],F[0,1]],[F[1,0],F[1,1]]])
    B,h,p=sand_return(A,0.,.25,35.)
    for i,j in ti.static(ti.ndrange(2,2)):out[i,j]=B[i,j]
    info[0],info[1]=h,p


def history(a):
    n=len(a['x'])
    a.update(F=np.tile(np.eye(2,dtype=np.float32),(n,1,1)),
             F_total=np.tile(np.eye(2,dtype=np.float32),(n,1,1)),
             multiplier=np.zeros((n,2,2),np.float32),
             plastic_volume=np.zeros(n,np.float32),plastic_strain=np.zeros(n,np.float32))
    return a


class MaterialChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ti.init(arch=ti.cpu,cpu_max_num_threads=1,offline_cache=False,debug=True)

    @classmethod
    def tearDownClass(cls):ti.reset()

    def test_rotation_invariant_target(self):
        angle=.71
        F=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],np.float32)
        out=np.empty((2,2),np.float32)
        targets(F,out,.25)
        np.testing.assert_allclose(out,0,atol=3e-7)

    def test_xpbmpm_first_update_and_reset(self):
        for compliance in (0.,1e-6):
            c=Config(grid=32,compliance=compliance)
            a=history(particles([[.3,.4]],ELASTIC,1e-5,c.dt))
            a['F'][0]=np.diag([1.2,.8])
            s=Solver(c,a)
            s.prepare()
            F=a['F'][0].astype(float)
            target=((1-.25)*np.eye(2)+.25*F/np.sqrt(np.linalg.det(F)))@np.linalg.inv(F)-np.eye(2)
            expected=target/(1+compliance/c.dt**2)
            np.testing.assert_allclose(s.D.to_numpy()[0],expected,atol=3e-7)
            np.testing.assert_allclose(s.multiplier.to_numpy()[0],expected,atol=3e-7)
            s.predict()
            np.testing.assert_array_equal(s.multiplier.to_numpy(),0)
            s.step();s.reset()
            for key in ('F','F_total','multiplier','plastic_volume','plastic_strain'):
                np.testing.assert_array_equal(s.snapshot()[key],a[key])

    def test_sand_tension_release_and_yield_return(self):
        out=np.empty((2,2),np.float32);info=np.empty(2,np.float32)
        plastic(np.eye(2,dtype=np.float32)*1.1,out,info)
        np.testing.assert_allclose(out,np.eye(2),atol=2e-7)
        self.assertAlmostEqual(float(info[0]),np.log(1.1),delta=2e-7)
        self.assertGreater(info[1],0)
        plastic(np.diag(np.exp([.2,-.4])).astype(np.float32),out,info)
        e=np.log(np.linalg.svd(out,compute_uv=False))
        friction=np.sqrt(2/3)*2*np.sin(np.deg2rad(35))/(3-np.sin(np.deg2rad(35)))
        residual=np.linalg.norm(e-e.mean())+1.75*e.sum()*friction
        self.assertAlmostEqual(float(residual),0,delta=3e-7)
        self.assertEqual(info[0],0)
        self.assertGreater(info[1],0)

    def test_solid_contact_reaction(self):
        for material in (ELASTIC,SAND):
            c=Config(grid=32)
            b=make_body([[-.04,-.08],[-.04,.08],[.04,-.08],[.04,.08]],(.6,.5),.002,size=(.12,.2))
            a=history(particles([[.5398,.525]],material,1e-5,c.dt,velocity=(.4,.13)))
            s=Solver(c,a,b)
            before=s.d.to_numpy().copy()
            s.prepare();s.p2g();s.project_contact()
            delta=s.d.to_numpy()[0]-before[0]
            np.testing.assert_allclose(a['mass'][0]*delta/c.dt+s.feedback[None].to_numpy(),0,atol=1e-10)
            self.assertLess(delta[0],0)
            s.apply_feedback()
            self.assertGreater(s.velocity[None][0],0)
            self.assertLess(s.omega[None],0)

    def test_initial_materials_match(self):
        base,body=build_scene(Config())
        for material in ('elastic','sand'):
            a,b=build_scene(Config(),material=material)
            for key in ('x','d','D','mass','area','local'):
                np.testing.assert_array_equal(a[key],base[key])
            self.assertEqual(body['mass'],b['mass'])
            self.assertEqual(body['inertia'],b['inertia'])
            np.testing.assert_array_equal(a['material']==RIGID,base['material']==RIGID)


if __name__=='__main__':unittest.main()
