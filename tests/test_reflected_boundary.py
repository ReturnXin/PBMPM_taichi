"""Affine wall consistency, real contact mass, and safe density feedback."""
import unittest
from unittest.mock import patch
import numpy as np
import taichi as ti

from rigid_coupling import cli
from rigid_coupling.config import Config
from rigid_coupling.container import roll_matrix
from rigid_coupling.scenes import Scene, lattice
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID, particle_arrays


def config(**options):
    return Config(gravity=0, viscosity=0, fluid_relaxation=0, rigid_contact=True,
                  world_contact=True, grid_coupling='separate', boundary_density='available', **options)


class ReflectedRoutingChecks(unittest.TestCase):
    def test_explicit_options_and_unchanged_defaults(self):
        with patch('rigid_coupling.shake.run') as run:
            cli.main(['--wall-transfer', 'reflected', '--boundary-density', 'available',
                      '--density-feedback', 'clamped'])
            c = run.call_args.args[1][0]
            self.assertEqual(c.wall_transfer, 'reflected')
            self.assertEqual(c.density_feedback, 'clamped')
        self.assertEqual(Config().wall_transfer, 'clamp')
        self.assertEqual(Config().density_feedback, 'legacy')
        with self.assertRaises(ValueError):
            config(wall_transfer='reflected', transfer_mode='incremental')
        with self.assertRaises(ValueError):
            Config(density_feedback='clamped')
        with self.assertRaises(SystemExit):
            cli.main(['--scene', 'shake_soft', '--wall-transfer', 'reflected'])


class ReflectedBoundaryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=4, offline_cache=False)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_affine_slip_at_translated_and_rotated_floor_and_momentum(self):
        c = config(wall_transfer='reflected')
        center = np.array([.5, .48, .5]); half = np.array([.25, .27, .23])
        local = lattice((0., -.23, 0.), (.16, .075, .14), (7, 6, 7))
        p = particle_arrays(center+local, FLUID, 1e-6, (0, 0, 0), c.dt, (.2, .5, .9))
        s = Solver(c, Scene(p, [], 'affine slip', container=dict(center=center, half_size=half)))
        for angle in (0., .17):
            R = roll_matrix(angle)
            gradient = R@np.diag([0., .02, 0.])@R.T
            x = center+local@R.T
            d = (x-(center-R[:, 1]*half[1]))@gradient.T+R@np.array([.001, 0., -.0005])
            p['x'][:] = x; p['d'][:] = d; p['D'][:] = gradient
            s.particles.data.from_numpy(p)
            s.container.upload_pose(*center, 0., 0., 0., angle, 0.)
            s.transfers.clear(); s.transfers.p2g(); s.grid_update(); s.transfers.g2p()
            after = s.particles.snapshot()
            np.testing.assert_allclose(after['d'], d, atol=5e-9)
            np.testing.assert_allclose(after['D'], np.broadcast_to(gradient, after['D'].shape), atol=1e-7)
            np.testing.assert_allclose(s.transfers.mass.to_numpy().sum(), p['mass'].sum(), rtol=3e-6)
        # Co-moving rigid translation and rotation are an exact affine field.
        angle, omega = .17, .3
        velocity = np.array([.1, -.05, .02])
        W = c.dt*np.array([[0., -omega, 0.], [omega, 0., 0.], [0., 0., 0.]])
        p['d'][:] = c.dt*velocity+(p['x']-center)@W.T
        p['D'][:] = W
        s.particles.data.from_numpy(p)
        s.container.upload_pose(*center, *velocity, angle, omega)
        s.transfers.clear(); s.transfers.p2g(); s.grid_update(); s.transfers.g2p()
        np.testing.assert_allclose(s.particles.snapshot()['d'], p['d'], atol=5e-9)
        np.testing.assert_allclose(s.particles.snapshot()['D'], p['D'], atol=1e-7)
        # A uniform outward velocity changes only its normal component. The
        # prescribed box receives the opposite total fluid momentum change.
        s.container.upload_pose(*center, 0., 0., 0., 0., 0.)
        s.container.wall_impulse.fill(0)
        p['x'][:] = center+local; p['d'][:] = [.0003, -.001, .0002]; p['D'][:] = 0
        s.particles.data.from_numpy(p)
        s.transfers.clear(); s.transfers.p2g(); s.grid_update(); s.transfers.g2p()
        after = s.particles.snapshot()
        delta = np.sum(p['mass'][:, None]*(after['d']-p['d'])/c.dt, axis=0)
        np.testing.assert_allclose(delta+s.container.wall_impulse[None].to_numpy(), 0., atol=2e-8)
        np.testing.assert_allclose(after['d'][:, [0, 2]], p['d'][:, [0, 2]], atol=2e-9)
        self.assertGreater(delta[1], 0.)
        s.reset()
        self.assertEqual(float(s.transfers.mass.to_numpy().sum()), 0.)
        self.assertEqual(float(s.transfers.physical_mass.to_numpy().sum()), 0.)

    def test_air_does_not_request_tension_and_unresolved_volume_is_skipped(self):
        c = config(density_feedback='clamped')
        p = particle_arrays([[.5, .5, .5]], FLUID, 1e-6, (0, 0, 0), c.dt, (.2, .5, .9))
        s = Solver(c, Scene(p, [], 'density', container=dict(center=[.5, .48, .5], half_size=[.25, .27, .23])))
        for J, a, n, expected in [(1., .5, .25, 1.), (.8, .5, .25, .81),
                                   (1., .5, .6, 1.+.05*(.5/.6-1.)), (.8, 0., 1., .8)]:
            p['volume_ratio'][:] = J
            s.particles.data.from_numpy(p)
            s.transfers.available_volume.fill(a); s.transfers.liquid_volume.fill(n)
            s.transfers.g2p()
            self.assertAlmostEqual(float(s.particles.snapshot()['volume_ratio'][0]), expected, delta=2e-7)

    def test_reaction_reduction_accumulates_once_and_skips_inactive_or_invalid(self):
        c = config(wall_transfer='reflected')
        center = np.array([.5, .48, .5])
        local = lattice((0., -.23, 0.), (.16, .075, .14), (7, 6, 7))
        p = particle_arrays(center+local, FLUID, 1e-6, (.15, -.5, .1),
                            c.dt, (.2, .5, .9))
        s = Solver(c, Scene(p, [], 'reaction reduction',
                            container=dict(center=center, half_size=[.25, .27, .23])))
        # 294 particles exercise several tiles and a partially occupied tile.
        for feedback in (False, True):
            s.reset()
            s.transfers.clear(); s.transfers.p2g(); s.grid_update()
            prior = np.array([.01, -.02, .03], dtype=np.float32)
            s.container.wall_impulse[None] = prior
            gather = (lambda: s.transfers.g2p_with_feedback(s.bodies)) if feedback else s.transfers.g2p
            gather()
            after = s.particles.snapshot()
            delta = np.sum(p['mass'][:, None]*(after['d']-p['d'])/c.dt, axis=0)
            np.testing.assert_allclose(s.container.wall_impulse[None].to_numpy(),
                                       prior-delta, atol=2e-8)
            # The same fixed grid gives the same d: no duplicate reaction.
            previous = s.container.wall_impulse[None].to_numpy()
            gather()
            np.testing.assert_allclose(s.container.wall_impulse[None].to_numpy(), previous, atol=2e-8)
            # A skipped entry must not contribute the previous gather's value.
            s.transfers.wall_reaction.fill(1.)
            after['x'][:] = 0.
            s.particles.data.from_numpy(after)
            gather()
            self.assertEqual(s.invalid[None], 1)
            np.testing.assert_array_equal(s.container.wall_impulse[None].to_numpy(), previous)
            s.active[None] = 0
            s.transfers.wall_reaction.fill(1.)
            gather()
            np.testing.assert_array_equal(s.container.wall_impulse[None].to_numpy(), previous)


if __name__ == '__main__':
    unittest.main()
