"""Physical regressions for the optional separated fluid/rigid transfers."""
from dataclasses import replace
import unittest

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.rigid import make_body, rotation_numpy
from rigid_coupling.scenes import Scene, lattice
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID, RIGID, combine_particles, particle_arrays


def fixture(config, *, wall=True, fixed=True, angle=0., velocity=(0., .5, 0.),
            body_velocity=(0., 0., 0.), clearance=.0005):
    size = np.array([.2, .24, .2])
    q = (np.cos(angle / 2), 0., 0., np.sin(angle / 2))
    R = rotation_numpy(q)
    point = np.array([.5, .5, .5]) + R @ [-size[0] / 2 - clearance, .03, .01]
    groups = [particle_arrays([point], FLUID, 1.e-5, R @ velocity, config.dt,
                             (.2, .5, .9), mass=.001)]
    bodies = []
    if wall:
        local = lattice((0, 0, 0), size, 4)
        body = make_body(local, (.5, .5, .5), .002, quaternion=q,
                         velocity=R @ body_velocity)
        body.update(size=size, fixed=fixed)
        groups.append(particle_arrays(body['center'] + body['local'] @ R.T,
                                      RIGID, np.prod(size) / len(local),
                                      body['velocity'], config.dt, (.8, .5, .2),
                                      body_id=0, local=body['local'], mass=body['particle_mass']))
        bodies.append(body)
    return Scene(combine_particles(groups), bodies, 'slip regression'), R


def fluid_velocity(solver):
    p = solver.particles.snapshot()
    f = p['material'] == FLUID
    return np.average(p['d'][f] / solver.config.dt, weights=p['mass'][f], axis=0)


def momentum(solver):
    p = solver.particles.snapshot(); b = solver.bodies.data.to_numpy()
    f = p['material'] == FLUID
    return ((p['mass'][f, None] * p['d'][f] / solver.config.dt).sum(axis=0)
            + (b['mass'][:solver.bodies.count, None] * b['v'][:solver.bodies.count]).sum(axis=0))


class SlipChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def config(self, **overrides):
        return replace(Config(grid=32, dt=.01, iterations=4, gravity=0,
                              viscosity=0, fluid_relaxation=0, rigid_contact=True,
                              world_contact=True, grid_coupling='separate'), **overrides)

    def test_uniform_tangent_survives_near_wall_and_exposes_shared_damping(self):
        outputs = {}
        for mode,wall in [('shared', True), ('separate', True), ('separate', False)]:
            c = self.config(grid_coupling=mode)
            scene,_ = fixture(c, wall=wall)
            s = Solver(c, scene); s.advance(8)
            outputs[mode,wall] = fluid_velocity(s)
        np.testing.assert_allclose(outputs['separate',True], outputs['separate',False], atol=1.e-5)
        np.testing.assert_allclose(outputs['separate',True], [0, .5, 0], atol=1.e-5)
        self.assertLess(outputs['shared',True][1], .4)

    def test_rotated_fixed_wall_blocks_normal_motion_without_removing_tangent(self):
        c = self.config()
        scene,R = fixture(c, angle=.6, velocity=(.1, .5, 0.))
        s = Solver(c, scene); initial = momentum(s)
        s.advance(5)
        self.assertAlmostEqual(float(fluid_velocity(s) @ R[:,1]), .5, places=5)
        p = s.particles.snapshot(); b = s.bodies.data.to_numpy()
        local = R.T @ (p['x'][0] - b['x'][0])
        self.assertLessEqual(local[0], -.1 + 2.e-7)
        np.testing.assert_array_equal(b['v'][0], 0)
        np.testing.assert_allclose(momentum(s) + b['contact_impulse'][0] + b['grid_impulse'][0], initial, atol=2.e-8)
        self.assertGreater(np.linalg.norm(b['contact_impulse'][0]), 1.e-6)

    def test_dynamic_body_receives_normal_reaction_and_total_momentum_is_preserved(self):
        c = self.config()
        scene,_ = fixture(c, fixed=False, velocity=(.1, .5, 0.))
        s = Solver(c, scene); initial = momentum(s)
        s.step(); b = s.bodies.data.to_numpy()
        self.assertGreater(float(b['v'][0,0]), 0)
        self.assertGreater(np.linalg.norm(b['omega'][0]), 0)
        self.assertGreater(np.linalg.norm(b['contact_impulse'][0]), 1.e-6)
        np.testing.assert_allclose(momentum(s), initial, atol=2.e-8)

    def test_nearby_moving_body_does_not_drag_noncontacting_fluid(self):
        c = self.config()
        scene,_ = fixture(c, fixed=False, velocity=(0., .1, 0.),
                          body_velocity=(0., -.2, 0.), clearance=.004)
        s = Solver(c, scene); s.advance(4); b = s.bodies.data.to_numpy()
        np.testing.assert_allclose(fluid_velocity(s), [0., .1, 0.], atol=2.e-6)
        np.testing.assert_allclose(b['v'][0], [0., -.2, 0.], atol=2.e-6)
        np.testing.assert_array_equal(b['contact_impulse'][0], 0)

    def test_reset_clears_both_transfer_fields(self):
        c = self.config()
        scene,_ = fixture(c, velocity=(.1, .5, 0.))
        s = Solver(c, scene); s.advance(4)
        expected = s.particles.snapshot()
        s.reset()
        for _ in range(4): s.step()
        actual = s.particles.snapshot()
        for field in ['x', 'd', 'D', 'volume_ratio']:
            np.testing.assert_allclose(actual[field], expected[field], atol=1.e-7)


if __name__ == '__main__':
    unittest.main()
