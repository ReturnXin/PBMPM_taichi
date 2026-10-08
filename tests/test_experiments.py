"""Numerical invariants and integration checks, runnable without a GPU or GUI."""
from dataclasses import replace
import unittest

import numpy as np
import taichi as ti

from experiments.cli import configurations
from experiments.metrics import CantileverContinuity, sample
from experiments.scenes import build
from pbmpm.config import Config
from pbmpm.materials import Materials
from pbmpm.reorder import MortonReorder
from pbmpm.solver import Solver
from pbmpm.state import ELASTIC, Particles, particle_arrays
from pbmpm.transfers import Transfers


def by_id(p):
    order = np.argsort(p["uid"])
    return {key: value[order] for key, value in p.items()}


class NumericalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_xpbd_local_balance_and_multiplier_lifetime(self):
        c = Config(grid=16, compliance=0.02, inverse_metric=2.5, dt=0.01, gravity=0, volume_weight=0)
        initial = particle_arrays([[0.5, 0.5, 0.5]], ELASTIC, 0.01, (0, 0, 0), c.dt, (1, 0, 0))
        initial["F"][0] = np.diag([1.2, 0.8, 1.1])
        initial["D"][0] = np.diag([0.01, 0.02, -0.01])
        initial["multiplier"][0] = np.eye(3) * 0.03
        particles = Particles(initial)
        material = Materials(particles, replace(c, elastic_solver="xpbmpm"))
        material.solve()
        p = particles.snapshot()
        target = np.linalg.inv(initial["F"][0]) - np.eye(3)
        residual = p["D"][0] - target + c.compliance / c.dt**2 * p["multiplier"][0]
        np.testing.assert_allclose(residual, 0, atol=2e-6)
        np.testing.assert_allclose(p["D"] - initial["D"],
                                   c.inverse_metric * (p["multiplier"] - initial["multiplier"]), atol=1e-6)
        material.predict()
        np.testing.assert_array_equal(particles.snapshot()["multiplier"], 0)

    def test_affine_transfer_mass_and_linear_momentum(self):
        c = Config(grid=16, gravity=0)
        x = np.array([[.42, .43, .48], [.46, .45, .52], [.48, .49, .50]], dtype=np.float32)
        p = particle_arrays(x, ELASTIC, .01, (0, 0, 0), c.dt, (1, 0, 0))
        affine = np.array([[.01, -.02, .0], [.02, .01, .0], [.0, .0, -.01]], dtype=np.float32)
        offset = np.array([.001, -.002, .003], dtype=np.float32)
        p["D"][:] = affine
        p["d"][:] = x @ affine.T + offset
        p["mass"][:] = [.01, .02, .04]
        particles = Particles(p)
        transfers = Transfers(particles, c)
        transfers.clear()
        transfers.p2g()
        np.testing.assert_allclose(transfers.mass.to_numpy().sum(), p["mass"].sum(), rtol=1e-6)
        np.testing.assert_allclose(transfers.displacement.to_numpy().sum(axis=(0, 1, 2)),
                                   (p["mass"][:, None] * p["d"]).sum(axis=0), atol=1e-8)
        transfers.boundary()
        transfers.g2p()
        final = particles.snapshot()
        np.testing.assert_allclose(final["d"], p["d"], atol=1e-7)
        np.testing.assert_allclose(final["D"], p["D"], atol=1e-6)

    def test_fixed_support_is_clamped_on_grid(self):
        c = Config(grid=32, gravity=0)
        x = np.array([[.3, .5, .5], [.33, .5, .5], [.7, .5, .5]], dtype=np.float32)
        p = particle_arrays(x, ELASTIC, .01, (0, 1, 0), c.dt, (1, 0, 0), [1, 0, 0])
        p["d"][0] = 0
        transfers = Transfers(Particles(p), c)
        transfers.clear()
        transfers.p2g()
        transfers.boundary()
        fixed = transfers.fixed.to_numpy().astype(bool)
        d = transfers.displacement.to_numpy()
        self.assertTrue(fixed.any())
        np.testing.assert_array_equal(d[fixed], 0)
        self.assertGreater(float(np.linalg.norm(d[~fixed], axis=-1).max()), 0)
        transfers.clear()
        np.testing.assert_array_equal(transfers.fixed.to_numpy().astype(bool), fixed)

    def test_cantilever_remains_connected_during_bending(self):
        # Exercise the coupled solver, not just the frozen single-particle formula.
        c = Config(grid=32, iterations=10, elastic_solver="xpbmpm", compliance=2e-6)
        scene = build("xpbmpm", c, resolution=6)
        solver = Solver(c, scene)
        continuity = CantileverContinuity(scene.particles)
        for _ in range(150):
            solver.step()
        p = solver.particles.snapshot()
        result = continuity.measure(p, c.grid)
        self.assertEqual(result["reference_neighbor_support_lost"], 0)
        self.assertLess(result["reference_neighbor_max_stretch"], 2.5)
        self.assertLess(sample(solver)["tip_deflection_y"], -0.005)
        pins = p["pinned"] != 0
        np.testing.assert_array_equal(p["x"][pins], p["x0"][pins])

    def test_morton_moves_every_field_and_preserves_particle_identity(self):
        c = Config(grid=16, reorder="full")
        scene = build("combined", c, resolution=3)
        rng = np.random.default_rng(10)
        scene.particles["multiplier"][:] = rng.normal(size=scene.particles["multiplier"].shape)
        scene.particles["plastic_volume"][:] = rng.uniform(size=len(scene.particles["x"]))
        particles = Particles(scene.particles)
        reorder = MortonReorder(particles, c)
        before = by_id(particles.snapshot())
        reorder.apply()
        keys = reorder.keys.to_numpy()
        self.assertTrue(np.all(keys[:-1] <= keys[1:]))
        ordered = particles.snapshot()
        cell = np.clip(np.floor(ordered["x"] * c.grid).astype(int), 0, c.grid - 1)
        expected = np.zeros(len(cell), dtype=np.int32)
        for bit in range(10):
            for axis in range(3):
                expected |= ((cell[:, axis] >> bit) & 1) << (3 * bit + axis)
        np.testing.assert_array_equal(keys, expected)
        after = by_id(ordered)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    def test_bidirectional_feedback_changes_both_participants(self):
        states, metrics = [], []
        for mode in ("one_way", "two_way"):
            c = Config(grid=16, gravity=0, iterations=2, coupling=mode)
            solver = Solver(c, build("rigid", c, resolution=4))
            for _ in range(6):
                solver.step()
            metrics.append(sample(solver))
            states.append(by_id(solver.particles.snapshot()))
        self.assertAlmostEqual(metrics[0]["body0_velocity_x"], 0, places=7)
        self.assertGreater(metrics[1]["body0_velocity_x"], 1e-4)
        self.assertGreater(metrics[1]["body0_angular_speed"], 1e-4)
        self.assertGreater(abs(metrics[0]["liquid_velocity_x"] - metrics[1]["liquid_velocity_x"]), 1e-5)
        self.assertLess(metrics[1]["body0_shape_radius_error"], 1e-6)

    def test_reorder_changes_storage_not_solution(self):
        states = []
        for mode in ("none", "full"):
            c = Config(grid=16, gravity=0, iterations=2, elastic_solver="xpbmpm", reorder=mode)
            solver = Solver(c, build("combined", c, resolution=3))
            for _ in range(3):
                solver.step()
            sample(solver)
            states.append(by_id(solver.particles.snapshot()))
        for key in states[0]:
            np.testing.assert_allclose(states[0][key], states[1][key], rtol=5e-4, atol=2e-5, err_msg=key)

    def test_four_scenes_and_reset(self):
        for name in ("xpbmpm", "rigid", "morton", "combined"):
            with self.subTest(scene=name):
                c = Config(grid=16, iterations=2, elastic_solver="xpbmpm")
                scene = build(name, c, resolution=3)
                solver = Solver(c, scene)
                for _ in range(20):
                    solver.step(diagnostics=True)
                metrics = sample(solver)
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                p = by_id(solver.particles.snapshot())
                pins = p["pinned"] > 0
                np.testing.assert_array_equal(p["x"][pins], p["x0"][pins])
                solver.reset()
                for key, value in solver.particles.snapshot().items():
                    np.testing.assert_array_equal(value, scene.particles[key])
                self.assertEqual(solver.step_index, 0)


class ExperimentChecks(unittest.TestCase):
    def test_cantilever_sampling_preserves_shape_mass_and_grid_density(self):
        for resolution, grid in ((3, 16), (12, 32), (16, 64)):
            scene = build("xpbmpm", Config(grid=grid), resolution)
            p = scene.particles
            record = scene.sampling[0]
            spacing = np.asarray(record["spacing"])
            self.assertLessEqual(spacing.max(), .5 / grid + 1e-7)
            self.assertLess(spacing.max() / spacing.min(), 1.15)
            self.assertAlmostEqual(float(p["mass"].sum()), .45 * .14 * .14, places=7)
            np.testing.assert_array_equal(p["pinned"] != 0, p["x0"][:, 0] < .35)
        self.assertEqual(build("xpbmpm", Config()).sampling[0]["counts"], [39, 12, 12])

    def test_continuity_detects_disconnected_layers_even_when_F_is_identity(self):
        c = Config()
        scene = build("xpbmpm", c)
        continuity = CantileverContinuity(scene.particles)
        self.assertEqual(continuity.measure(scene.particles, c.grid)["reference_neighbor_support_lost"], 0)
        # Translate half the beam without changing F: old determinant checks would pass.
        p = {k: v.copy() for k, v in scene.particles.items()}
        p["x"][p["x0"][:, 0] > .5, 1] -= .2
        self.assertGreater(continuity.measure(p, c.grid)["reference_neighbor_support_lost"], 0)
        order = np.random.default_rng(3).permutation(len(p["x"]))
        shuffled = {k: v[order] for k, v in p.items()}
        self.assertEqual(continuity.measure(p, c.grid), continuity.measure(shuffled, c.grid))

    def test_suites_keep_unrelated_features_constant(self):
        c = Config()
        xp = configurations("xpbmpm", c, True)
        self.assertEqual({x.elastic_solver for x in xp}, {"pbmpm", "xpbmpm"})
        self.assertEqual({x.reorder for x in xp}, {"none"})
        rigid = configurations("rigid", c, True)
        self.assertEqual(len(rigid), 2)
        self.assertEqual(len(configurations("combined", c, True)), 8)
        self.assertEqual({x.reorder for x in configurations("morton", c, True)}, {"none", "full", "periodic"})

    def test_invalid_configuration(self):
        for kw in (dict(dt=0), dict(dt=float("nan")), dict(reorder_interval=0), dict(volume_weight=2)):
            with self.assertRaises(ValueError):
                Config(**kw)


if __name__ == "__main__":
    unittest.main()
