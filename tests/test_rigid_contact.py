"""Independent impulse and geometric checks for the pool contact extension."""

from dataclasses import replace
from itertools import product
import unittest

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.contact import Contacts
from rigid_coupling.rigid import RigidBodies, make_body, rotation_numpy
from rigid_coupling.state import FLUID, RIGID, Particles, combine_particles, particle_arrays


def contact_fixture(config, *, quaternion=(1, 0, 0, 0), with_fluid=True):
    size = np.array([.20, .16, .12])
    samples = np.array(list(product((-1, 1), repeat=3))) * size / 4
    body = make_body(samples, (.5, .5, .5), .25, quaternion=quaternion)
    body["size"] = size
    R = rotation_numpy(body["quaternion"])
    rigid = particle_arrays(body["center"] + body["local"] @ R.T, RIGID,
                            np.prod(size) / len(samples), (0, 0, 0), config.dt,
                            (.5, .8, .5), body_id=0, local=body["local"],
                            mass=body["particle_mass"])
    groups = [rigid]
    if with_fluid:
        # Unique closest face: +local x, with an eccentric arm for torque.
        point = body["center"] + R @ np.array([.095, .03, -.01])
        groups.append(particle_arrays([point], FLUID, .001, (0, 0, 0),
                                     config.dt, (.2, .5, .9), mass=.1))
    particles = Particles(combine_particles(groups))
    bodies = RigidBodies(particles, config, [body])
    return particles, bodies, Contacts(particles, bodies, config)


class ContactImpulseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_eccentric_contact_matches_effective_mass_and_conserves_momenta(self):
        config = Config(gravity=0, dt=.01, rigid_contact=True)
        particles, bodies, contacts = contact_fixture(config)
        before_p, before_b = particles.snapshot(), bodies.data.to_numpy()
        index = particles.n - 1
        mass = float(before_p["mass"][index])
        body_mass = float(before_b["mass"][0])
        arm = before_p["x"][index].astype(float) - before_b["x"][0]
        normal = np.array([1., 0, 0])
        depth = float(before_b["half_size"][0, 0]) - arm[0]
        inv_inertia = before_b["inverse_world"][0].astype(float)
        rn = np.cross(arm, normal)
        impulse = depth / config.dt / (1 / mass + 1 / body_mass + rn @ inv_inertia @ rn) * normal
        contacts.project()
        collected = bodies.data.to_numpy()
        np.testing.assert_allclose(collected["delta_momentum"][0], -impulse, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(collected["delta_angular"][0], -np.cross(arm, impulse),
                                   rtol=2e-6, atol=2e-9)
        bodies.apply_feedback()
        after_p, after_b = particles.snapshot(), bodies.data.to_numpy()
        fluid_dp = mass * (after_p["d"][index] - before_p["d"][index]) / config.dt
        body_dp = body_mass * (after_b["v"][0] - before_b["v"][0])
        angular_dp = np.linalg.solve(inv_inertia, after_b["omega"][0] - before_b["omega"][0])
        np.testing.assert_allclose(fluid_dp, impulse, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(fluid_dp + body_dp, 0, atol=2e-8)
        np.testing.assert_allclose(np.cross(arm, fluid_dp) + angular_dp, 0, atol=2e-9)
        # Application must consume the buffer before the next contact pass.
        np.testing.assert_array_equal(after_b["delta_momentum"], 0)
        np.testing.assert_array_equal(after_b["delta_angular"], 0)
        bodies.apply_feedback()
        np.testing.assert_array_equal(bodies.data.to_numpy()["v"], after_b["v"])
        np.testing.assert_array_equal(bodies.data.to_numpy()["omega"], after_b["omega"])

    def test_rotated_contact_uses_world_normal_and_equal_opposite_torque(self):
        angle = .7
        q = (np.cos(angle / 2), 0, 0, np.sin(angle / 2))
        config = Config(gravity=0, dt=.01, rigid_contact=True)
        particles, bodies, contacts = contact_fixture(config, quaternion=q)
        before_p, before_b = particles.snapshot(), bodies.data.to_numpy()
        index = particles.n - 1
        R = rotation_numpy(before_b["q"][0])
        normal = R[:, 0]
        contacts.project()
        after_p, after_b = particles.snapshot(), bodies.data.to_numpy()
        impulse = before_p["mass"][index] * (after_p["d"][index] - before_p["d"][index]) / config.dt
        self.assertGreater(impulse @ normal, 0)
        np.testing.assert_allclose(np.cross(impulse, normal), 0, atol=2e-8)
        np.testing.assert_allclose(after_b["delta_momentum"][0] + impulse, 0, atol=2e-8)
        arm = before_p["x"][index] - before_b["x"][0]
        np.testing.assert_allclose(after_b["delta_angular"][0] + np.cross(arm, impulse), 0, atol=2e-9)

    def test_one_way_projects_to_surface_without_changing_body(self):
        config = Config(gravity=0, dt=.01, coupling="one_way", rigid_contact=True)
        particles, bodies, contacts = contact_fixture(config)
        initial_body = bodies.data.to_numpy()
        contacts.project()
        after = particles.snapshot()
        point = after["x"][-1] + after["d"][-1]
        expected = initial_body["x"][0, 0] + initial_body["half_size"][0, 0]
        self.assertAlmostEqual(float(point[0]), float(expected), places=6)
        for name, value in bodies.data.to_numpy().items():
            np.testing.assert_array_equal(value, initial_body[name], err_msg=name)

    def test_contact_uses_predicted_translating_body_position(self):
        config = Config(gravity=0, dt=.01, coupling="one_way", rigid_contact=True)
        particles, bodies, contacts = contact_fixture(config)
        before = bodies.data.to_numpy()
        before["v"][0] = [.3, -.1, .2]
        bodies.data.from_numpy(before)
        contacts.project()
        after = particles.snapshot()
        point = after["x"][-1] + after["d"][-1]
        predicted_face = before["x"][0, 0] + config.dt * before["v"][0, 0] + before["half_size"][0, 0]
        self.assertAlmostEqual(float(point[0]), float(predicted_face), places=6)

    def test_world_contact_keeps_true_rotated_corners_inside_in_both_modes(self):
        base = Config(gravity=0, dt=.02, world_contact=True)
        initial_q = np.array([.8, .3, -.2, .4])
        signs = np.array(list(product((-1, 1), repeat=3)))
        outcomes = []
        for mode in ("one_way", "two_way"):
            config = replace(base, coupling=mode)
            particles, bodies, _ = contact_fixture(config, quaternion=initial_q, with_fluid=False)
            initial = bodies.data.to_numpy()
            mode_outcomes = []
            for axis in range(3):
                for side in (-1, 1):
                    values = {name: value.copy() for name, value in initial.items()}
                    values["v"][0, axis] = side * 25
                    values["omega"][0] = [.6, -.3, .4]
                    bodies.data.from_numpy(values)
                    bodies.integrate()
                    current = bodies.data.to_numpy()
                    R = rotation_numpy(current["q"][0])
                    corners = current["x"][0] + (signs * current["half_size"][0]) @ R.T
                    lo, hi = config.boundary / config.grid, 1 - config.boundary / config.grid
                    self.assertGreaterEqual(float(corners.min()), lo - 1e-7)
                    self.assertLessEqual(float(corners.max()), hi + 1e-7)
                    # Every sample still follows the same rigid pose.
                    p = particles.snapshot()
                    np.testing.assert_allclose(p["x"], current["x"][0] + p["local"] @ R.T, atol=1e-7)
                    mode_outcomes.append(current)
            outcomes.append(mode_outcomes)
        for one, two in zip(*outcomes):
            for name in one:
                np.testing.assert_array_equal(one[name], two[name], err_msg=name)


if __name__ == "__main__":
    unittest.main()
