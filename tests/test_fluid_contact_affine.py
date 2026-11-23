"""Water contact strain, exact slip/nonpenetration, and impulse conservation."""
import unittest
import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.container import roll_matrix
from rigid_coupling.rigid import make_body, rotation_numpy
from rigid_coupling.scenes import Scene, lattice
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID, ELASTIC, RIGID, particle_arrays, combine_particles


def config(**kwargs):
    return Config(gravity=0, viscosity=0, rigid_contact=True, world_contact=True,
                  grid_coupling="separate", **kwargs)


def angular_momentum(p, grid, dt):
    # APIC angular momentum includes the antisymmetric affine contribution.
    f = p["material"] == FLUID
    m, x, v, D = p["mass"][f], p["x"][f], p["d"][f]/dt, p["D"][f]/dt
    spin = np.column_stack((D[:, 2, 1]-D[:, 1, 2], D[:, 0, 2]-D[:, 2, 0],
                            D[:, 1, 0]-D[:, 0, 1]))/(4*grid**2)
    return np.sum(m[:, None]*(np.cross(x, v)+spin), axis=0)


class FluidAffineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def wall_scene(self, c, angle=0., mixed=False):
        center, half = np.array([.5, .48, .5]), np.array([.25, .27, .23])
        R = roll_matrix(angle)
        local = lattice((0, -.239, 0), (.12, .075, .1), (7, 7, 6))
        velocity = R@np.array([.12, -.3, .07])
        water = particle_arrays(center+local@R.T, FLUID, 1e-6, velocity, c.dt, (.2, .5, .9))
        groups = [water]
        if mixed:
            groups.append(particle_arrays(center+local@R.T, ELASTIC, 1e-6,
                                          R@np.array([1.2, -.3, -.7]), c.dt, (.7, .4, .8)))
        scene = Scene(combine_particles(groups), [], "floor strain", container=dict(center=center, half_size=half))
        s = Solver(c, scene)
        s.container.upload_pose(*center, 0., 0., 0., angle, 0.)
        return s, R

    def test_floor_clip_updates_volume_without_undoing_slip_or_adding_impulse(self):
        for angle in (0., .17):
            with self.subTest(angle=angle):
                c = config()
                s, R = self.wall_scene(c, angle)
                before = s.particles.snapshot()
                s.container.project()
                projected = s.particles.snapshot()
                s.container.update_contact_affine()
                corrected = s.particles.snapshot()
                # The exact normal correction and its wall reaction stay intact.
                np.testing.assert_array_equal(corrected["d"], projected["d"])
                dp = np.sum(before["mass"][:, None]*(corrected["d"]-before["d"])/c.dt, axis=0)
                np.testing.assert_allclose(dp+s.container.wall_impulse[None].to_numpy(), 0, atol=2e-8)
                np.testing.assert_allclose(angular_momentum(corrected, c.grid, c.dt),
                                           angular_momentum(projected, c.grid, c.dt), atol=1e-9)
                normal_strain = np.einsum("i,nij,j->n", R[:, 1], corrected["D"], R[:, 1])
                self.assertLess(float(normal_strain.mean()), -1e-3)
                local = (corrected["x"]+corrected["d"]-s.container.motion.center)@R
                self.assertGreaterEqual(float(local[:, 1].min()), -.270001)
                delta = (corrected["d"]-before["d"])@R
                np.testing.assert_allclose(delta[:, [0, 2]], 0, atol=5e-8)
                s.materials.integrate()
                self.assertLess(float(s.particles.snapshot()["volume_ratio"].mean()), .999)

    def test_rotated_rigid_contact_updates_water_strain_and_preserves_reaction(self):
        c = config(coupling="one_way")
        angle = .43
        q = (np.cos(angle/2), 0, 0, np.sin(angle/2))
        size = np.array([.2, .2, .2])
        body = make_body(lattice((0, 0, 0), size, 4), (.5, .5, .5), .001, quaternion=q)
        body.update(size=size, fixed=True)
        R = rotation_numpy(body["quaternion"])
        local = lattice((.12, 0, 0), (.07, .1, .1), (7, 6, 6))
        water = particle_arrays(body["center"]+local@R.T, FLUID, 1e-6, R@[-.2, .12, .07], c.dt, (.2, .5, .9))
        rigid = particle_arrays(body["center"]+body["local"]@R.T, RIGID, 1e-6, (0, 0, 0),
                                c.dt, (.5, .8, .4), body_id=0, local=body["local"], mass=body["particle_mass"])
        s = Solver(c, Scene(combine_particles([water, rigid]), [body], "rigid face strain"))
        s.contacts.project()
        projected = s.particles.snapshot()
        before_b = s.bodies.data.to_numpy()
        s.contacts.update_contact_affine()
        corrected = s.particles.snapshot()
        np.testing.assert_array_equal(corrected["d"], projected["d"])
        np.testing.assert_array_equal(s.bodies.data.to_numpy()["v"], before_b["v"])
        f = corrected["material"] == FLUID
        strain = np.einsum("i,nij,j->n", R[:, 0], corrected["D"][f], R[:, 0])
        self.assertLess(float(strain.mean()), -1e-3)
        local_after = (corrected["x"][f]+corrected["d"][f]-body["center"])@R
        self.assertGreaterEqual(float(local_after[:, 0].min()), .099999)
        np.testing.assert_allclose(angular_momentum(corrected, c.grid, c.dt),
                                   angular_momentum(projected, c.grid, c.dt), atol=1e-9)

    def test_soft_friction_does_not_enter_water_affine_channel(self):
        c = config(soft_friction=.5)
        outputs = []
        for mixed in (False, True):
            s, _ = self.wall_scene(c, mixed=mixed)
            s.container.project()
            projected = s.particles.snapshot()
            s.container.update_contact_affine()
            after = s.particles.snapshot()
            f = after["material"] == FLUID
            np.testing.assert_array_equal(after["d"][f], projected["d"][f])
            outputs.append(after["D"][f])
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-7, rtol=1e-5)

    def test_zero_contact_and_reset_do_not_leave_stale_strain(self):
        c = config()
        s, _ = self.wall_scene(c)
        s.container.project()
        s.container.update_contact_affine()
        self.assertGreater(float(np.linalg.norm(s.particles.snapshot()["D"])), .001)
        # No new contacts must skip stale grid values without requiring reset.
        clear = s.particles.snapshot()
        clear["x"][:, 1] += .12
        clear["D"][:] = 0
        s.particles.data.from_numpy(clear)
        s.container.project()
        s.container.update_contact_affine()
        np.testing.assert_array_equal(s.particles.snapshot()["D"], 0)
        s.reset()
        for contact in (s.contacts, s.container):
            np.testing.assert_array_equal(contact.contact_delta.to_numpy(), 0)
            np.testing.assert_array_equal(contact.delta_mass.to_numpy(), 0)
            np.testing.assert_array_equal(contact.delta_grid.to_numpy(), 0)
        p = s.particles.snapshot()
        p["x"][:, 1] += .12
        s.particles.data.from_numpy(p)
        s.container.project()
        s.container.update_contact_affine()
        np.testing.assert_array_equal(s.particles.snapshot()["D"], 0)

    def test_reused_p2g_mass_matches_full_contact_transfer(self):
        outputs = []
        for reuse in (False, True):
            s, _ = self.wall_scene(config(), angle=.17)
            s.transfers.p2g()
            s.container.project()
            s.container.update_contact_affine(s.transfers if reuse else None)
            outputs.append(s.particles.snapshot())
        np.testing.assert_array_equal(outputs[0]["d"], outputs[1]["d"])
        np.testing.assert_allclose(outputs[0]["D"], outputs[1]["D"], atol=1e-7, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
