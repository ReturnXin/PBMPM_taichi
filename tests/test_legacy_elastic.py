"""Compare the experiment adapter against the original elastic kernels."""
from dataclasses import replace
import unittest

import numpy as np
import taichi as ti

from experiments.cli import configurations
from experiments.metrics import CantileverContinuity, sample
from experiments.scenes import build
from mpm_pbd import MpmPBDSolver
from pbmpm.config import Config
from pbmpm.materials import Materials
from pbmpm.solver import Solver
from pbmpm.state import ELASTIC, Particles, particle_arrays
from pbmpm_types import MaterialParam


@ti.data_oriented
class LegacyReference:
    """Bind the actual old kernels to small fields; omit unused GUI allocations."""

    solve_constraint = MpmPBDSolver.solve_constraint
    compute_F = MpmPBDSolver.compute_F
    damp_lambdas = MpmPBDSolver.damp_lambdas

    def __init__(self, initial, config, stiffness):
        self.dim, self.dt, self.gravity = 3, config.dt, -config.gravity
        n = len(initial["x"])
        self.n_particles = ti.field(ti.i32, shape=())
        self.n_particles[None] = n
        self.average_height = ti.field(ti.f32, shape=())
        self.mat_params = MaterialParam.field(shape=3)
        self.mat_params[1].E = stiffness
        for name, source in (("x", "x"), ("dis", "d")):
            field = ti.Vector.field(3, ti.f32, shape=n)
            field.from_numpy(initial[source])
            setattr(self, name, field)
        for name, source in (("F", "F"), ("D", "D"), ("lambdas", "multiplier")):
            field = ti.Matrix.field(3, 3, ti.f32, shape=n)
            field.from_numpy(initial[source])
            setattr(self, name, field)
        self.material = ti.field(ti.i32, shape=n)
        self.material.from_numpy(initial["material"])
        self.L = ti.field(ti.f32, shape=n)
        self.log_JP = ti.field(ti.f32, shape=n)

    @ti.func
    def compute_water_color(self, p, color_option):
        # The reference only contains elastic particles; colors do not affect F.
        return 0

    @ti.kernel
    def integrate(self):
        for p in range(self.n_particles[None]):
            self.compute_F(p)
            self.x[p] += self.dis[p]
            self.dis[p][1] -= self.gravity * self.dt**2


class LegacyElasticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_repeated_updates_match_original_elastic_kernels(self):
        stiffness = 50000.0
        c = Config(dt=.008, elastic_solver="xpbmpm", elastic_model="legacy",
                   compliance=1.0 / (stiffness + 1e-6))
        initial = particle_arrays([[.5, .5, .5]] * 4, ELASTIC, .01,
                                  (.1, -.2, .05), c.dt, (1, 0, 0))
        initial["F"][:] = [np.eye(3), [[1.2, .2, .1], [0, .8, .05], [0, 0, 1.1]],
                            np.diag([1, 1, -1]), np.diag([.04, .9, 1.2])]
        initial["D"][:] = [[.01, .03, 0], [-.02, -.01, .01], [0, 0, .02]]
        initial["multiplier"][:] = np.eye(3) * .03
        particles = Particles(initial)
        materials = Materials(particles, c)
        old = LegacyReference(initial, c, stiffness)
        for _ in range(4):
            previous_multiplier = particles.snapshot()["multiplier"]
            materials.predict()
            old.damp_lambdas(.5)
            np.testing.assert_array_equal(particles.snapshot()["multiplier"], previous_multiplier * .5)
            # Prior SVD evaluations can differ slightly across the two field layouts.
            np.testing.assert_allclose(particles.snapshot()["multiplier"], old.lambdas.to_numpy(), rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(particles.snapshot()["d"], old.dis.to_numpy(), atol=1e-7)
            for _ in range(3):
                materials.solve()
                old.solve_constraint()
                p = particles.snapshot()
                np.testing.assert_allclose(p["D"], old.D.to_numpy(), rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(p["multiplier"], old.lambdas.to_numpy(), rtol=2e-5, atol=2e-6)
            materials.integrate()
            old.integrate()
            p = particles.snapshot()
            np.testing.assert_allclose(p["F"], old.F.to_numpy(), rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(p["x"], old.x.to_numpy(), atol=1e-7)
            np.testing.assert_allclose(p["d"], old.dis.to_numpy(), atol=1e-7)

    def test_legacy_gravity_clamp_and_diagnostics(self):
        c = Config(elastic_solver="xpbmpm", elastic_model="legacy", compliance=2e-6, iterations=10)
        scene = build("xpbmpm", c, resolution=6)
        solver = Solver(c, scene)
        solver.step(diagnostics=True)
        # Legacy applies gravity for the next step; initially stationary particles stay put.
        np.testing.assert_allclose(solver.particles.snapshot()["x"], scene.particles["x"], atol=1e-7)
        for _ in range(149):
            solver.step()
        solver.materials.measure_residuals()
        p = solver.particles.snapshot()
        pins = p["pinned"] != 0
        np.testing.assert_array_equal(p["x"][pins], p["x0"][pins])
        self.assertEqual(CantileverContinuity(scene.particles).measure(p, c.grid)["reference_neighbor_support_lost"], 0)
        self.assertLess(sample(solver)["tip_deflection_y"], -.005)
        # Re-evaluate the actual shape-only constraint independently for diagnostics.
        F, D = p["F"][~pins], p["D"][~pins]
        U, _, V = np.linalg.svd((np.eye(3) + D) @ F)
        u, s, v = np.linalg.svd(F)
        inverse = (v.transpose(0, 2, 1) * (1 / np.maximum(s, .1))[:, None, :]) @ u.transpose(0, 2, 1)
        residual = D - ((U @ V) @ inverse - np.eye(3))
        measured = solver.materials.residuals.to_numpy()
        self.assertAlmostEqual(float(measured[0] / measured[2]), float(np.linalg.norm(residual, axis=(1, 2)).mean()), places=5)
        solver.reset()
        np.testing.assert_array_equal(solver.particles.snapshot()["multiplier"], 0)


class LegacyConfigurationChecks(unittest.TestCase):
    def test_suite_keeps_selected_model(self):
        base = Config(elastic_model="legacy")
        self.assertEqual({c.elastic_model for c in configurations("xpbmpm", base, True)}, {"legacy"})
        with self.assertRaises(ValueError):
            replace(base, inverse_metric=2)
        with self.assertRaises(ValueError):
            replace(base, elastic_model="typo")


if __name__ == "__main__":
    unittest.main()
