"""XPBMPM integration: original material parity, history and shared execution."""
from dataclasses import replace
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import taichi as ti

from rigid_coupling import cli
from rigid_coupling.config import Config
from rigid_coupling.materials import Materials
from rigid_coupling.scenes import Scene
from rigid_coupling.shake import build_shake_scene
from rigid_coupling.solver import Solver
from rigid_coupling.state import ELASTIC, FLUID, Particles, particle_arrays


class UnifiedEntryChecks(unittest.TestCase):
    def test_main_routes_to_the_same_solver_and_is_safe_to_import(self):
        with patch('taichi.init') as initialize:
            entry = importlib.import_module('main')
            initialize.assert_not_called()
        for args, expected in [(['--scene', 'shake_soft', '--elastic-solver', 'xpbmpm'], 'legacy'),
                               (['--scene', 'shake_soft'], 'experiment')]:
            with patch('rigid_coupling.shake.run') as run:
                entry.main(args)
                c = run.call_args.args[1][0]
                self.assertEqual(c.elastic_model, expected)
                self.assertEqual(c.iterations, 5)
        with patch('rigid_coupling.shake.run') as run:
            entry.main(['--wall-transfer', 'reflected'])
            c = run.call_args.args[1][0]
            self.assertEqual((c.boundary_density, c.density_feedback), ('available', 'clamped'))

    def test_configuration_rejects_invalid_compliance_and_water_cli(self):
        for options in ({'compliance': -1.}, {'compliance': float('nan')},
                        {'inverse_metric': 0.}, {'elastic_solver': 'bad'},
                        {'elastic_model': 'bad'}, {'elastic_model': 'legacy', 'inverse_metric': 2.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config(**options)
        with self.assertRaises(SystemExit):
            cli.main(['--elastic-solver', 'xpbmpm'])


class XPBMPMCouplingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=2, offline_cache=False, debug=True)

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_material_updates_match_existing_chapter_implementation(self):
        from pbmpm.config import Config as ChapterConfig
        from pbmpm.materials import Materials as ChapterMaterials
        from pbmpm.state import Particles as ChapterParticles, particle_arrays as chapter_arrays
        from pbmpm.state import ELASTIC as CHAPTER_ELASTIC
        for model in ('legacy', 'experiment'):
            c = Config(elastic_solver='xpbmpm', elastic_model=model, compliance=1e-6)
            old_c = ChapterConfig(elastic_solver='xpbmpm', elastic_model=model, compliance=c.compliance)
            old_p = chapter_arrays([[.45, .5, .5], [.55, .5, .5]], CHAPTER_ELASTIC,
                                   .001, (.1, -.2, .05), c.dt, (1., 0., 0.))
            old_p['F'][:] = [[[1.2, .2, .1], [0., .8, .05], [0., 0., 1.1]], np.diag([.04, .9, 1.2])]
            old_p['D'][:] = [[.01, .03, 0.], [-.02, -.01, .01], [0., 0., .02]]
            old_p['multiplier'][:] = np.eye(3)*.03
            current = particle_arrays(old_p['x'], ELASTIC, .001, (.1, -.2, .05), c.dt, (1., 0., 0.))
            for name in ('F', 'D', 'multiplier'):
                current[name] = old_p[name].copy()
            new = Particles(current, xpbmpm=True)
            old = ChapterParticles(old_p)
            materials, reference = Materials(new, c), ChapterMaterials(old, old_c)
            for _ in range(3):
                materials.predict(); reference.predict()
                for _ in range(3):
                    materials.solve(); reference.solve()
                materials.integrate(); reference.integrate()
                actual, expected = new.snapshot(), old.snapshot()
                for name in ('x', 'd', 'D', 'F', 'multiplier'):
                    np.testing.assert_allclose(actual[name], expected[name], atol=2e-6, rtol=2e-5,
                                               err_msg=f'{model}: {name}')

    def test_frozen_diagonal_constraint_and_zero_compliance_limit(self):
        for compliance in (0., 1e-6):
            c = Config(gravity=0, elastic_solver='xpbmpm', elastic_model='legacy', compliance=compliance)
            p = particle_arrays([[.5, .5, .5]], ELASTIC, .001, (0., 0., 0.), c.dt, (1., 0., 0.))
            p['F'][0] = np.diag([1.2, .8, 1.1])
            p['multiplier'] = np.array([np.diag([.03, -.02, .01])], dtype=np.float32)
            particles = Particles(p, xpbmpm=True)
            material = Materials(particles, c)
            material.solve()
            result = particles.snapshot()
            target = np.diag([1/1.2-1, 1/.8-1, 1/1.1-1])
            # For this positive diagonal state, R=I independently of SVD code.
            residual = result['D'][0]-target+(compliance/c.dt**2)*result['multiplier'][0]
            np.testing.assert_allclose(residual, 0., atol=3e-6)
            if compliance == 0:
                np.testing.assert_allclose(result['D'][0], target, atol=3e-6)

    def test_multiplier_snapshot_restore_reset_and_water_allocation(self):
        c = Config(gravity=0, elastic_solver='xpbmpm', elastic_model='legacy')
        p = particle_arrays([[.5, .5, .5]], ELASTIC, .001, (0., 0., 0.), c.dt, (1., 0., 0.))
        s = Solver(c, Scene(p, [], 'history'))
        s.particles.multiplier.fill(.2)
        saved = s.particles.snapshot()
        s.particles.multiplier.fill(0.)
        s.particles.restore(saved)
        np.testing.assert_array_equal(s.particles.multiplier.to_numpy(), saved['multiplier'])
        s.reset()
        np.testing.assert_array_equal(s.particles.multiplier.to_numpy(), 0.)
        water = particle_arrays([[.5, .5, .5]], FLUID, .001, (0., 0., 0.), c.dt, (0., 0., 1.))
        for options in ({}, {'xpbmpm': True}):
            particles = Particles(water, **options)
            self.assertIsNone(particles.multiplier)
            self.assertNotIn('multiplier', particles.snapshot())

    def test_shared_soft_rigid_steps_and_invalid_history_recovery(self):
        c = Config(elastic_solver='xpbmpm', elastic_model='legacy', compliance=1e-6,
                   rigid_contact=True, world_contact=True, grid_coupling='separate')
        s = Solver(c, build_shake_scene(c, ELASTIC, resolution=6, seed=42))
        s.advance(4)
        self.assertEqual(s.step_index, 4)
        p = s.particles.snapshot()
        self.assertGreater(np.linalg.norm(p['multiplier']), 0.)
        np.testing.assert_array_equal(p['multiplier'][p['material'] != ELASTIC], 0.)
        p['multiplier'][np.flatnonzero(p['material'] == ELASTIC)[0], 0, 0] = np.nan
        s.particles.multiplier.from_numpy(p['multiplier'])
        with self.assertRaises(FloatingPointError):
            s.advance(1)
        s.reset()
        self.assertFalse(s.failed)
        s.advance(1)


if __name__ == '__main__':
    unittest.main()
