"""Incremental PBMPM experiment: remapping, contact impulses and reset."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import taichi as ti

from rigid_coupling import cli
from rigid_coupling.config import Config
from rigid_coupling.scenes import Scene, lattice
from rigid_coupling.shake import build_shake_scene
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID, ELASTIC, particle_arrays
from tests.test_rigid_slip import fixture, momentum, fluid_velocity


def config(**kwargs):
    return Config(gravity=0, viscosity=0, fluid_relaxation=0, rigid_contact=True,
                  world_contact=True, grid_coupling="separate", **kwargs)


def frozen_iterations(s):
    for i in range(s.config.iterations):
        if s.config.transfer_mode == "incremental" and i > 0:
            s.prepare_increment()
            s.transfers.p2g_increment()
            s.transfers.incremental_boundary()
        else:
            s.prepare_iteration()
            s.transfers.p2g()
            s.transfers.boundary()
        s.transfers.g2p()


def energy(p, c):
    return .5*np.sum(p["mass"]*(np.sum(p["d"].astype(float)**2, axis=1)
                    +np.sum(p["D"].astype(float)**2, axis=(1, 2))/(4*c.grid**2)))/c.dt**2


class RoutingChecks(unittest.TestCase):
    def test_cli_exposes_experiment_and_keeps_full_default(self):
        for mode in ("full", "incremental"):
            with patch("rigid_coupling.shake.run") as run:
                cli.main(["--scene", "shake_water", "--transfer-mode", mode])
                self.assertEqual(run.call_args.args[1][0].transfer_mode, mode)
        with self.assertRaises(ValueError):
            Config(transfer_mode="incremental")


class IncrementalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False,
                offline_cache_file_path="results/shake_box_20261030/incremental_transfer/test_cache")

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def test_one_iteration_matches_full_including_water_contacts(self):
        results = []
        c = config(iterations=1)
        for mode in ("full", "incremental"):
            selected = replace(c, transfer_mode=mode)
            scene = build_shake_scene(selected, FLUID, resolution=8)
            scene.particles["x"][0] = [.7504, .45, .5]
            s = Solver(selected, scene)
            s.advance(1)
            results.append(s.particles.snapshot())
        for key in ("x", "d", "D", "volume_ratio"):
            np.testing.assert_allclose(results[0][key], results[1][key], atol=1e-7, rtol=1e-6)

    def test_unconstrained_inner_iterations_do_not_repeat_remapping(self):
        x = lattice((.5, .5, .5), (.25, .15, .25), (14, 8, 14))
        c = config(iterations=5)
        wave = 2*np.pi*c.grid/4
        p = particle_arrays(x, FLUID, 1e-6, (0, 0, 0), c.dt, (.2, .5, .9))
        p["d"][:, 0] = c.dt*.3*np.sin(wave*x[:, 2])
        p["D"][:, 0, 2] = c.dt*.3*wave*np.cos(wave*x[:, 2])
        result = {}
        for mode, iterations in (("full", 5), ("incremental", 1), ("incremental", 5)):
            selected = replace(c, transfer_mode=mode, iterations=iterations)
            s = Solver(selected, Scene(p, [], "frozen shear"))
            frozen_iterations(s)
            result[mode, iterations] = s.particles.snapshot()
        np.testing.assert_allclose(result["incremental", 1]["d"], result["incremental", 5]["d"], atol=1e-8)
        np.testing.assert_allclose(result["incremental", 1]["D"], result["incremental", 5]["D"], atol=1e-7)
        self.assertGreater(energy(result["incremental", 5], c), 1.4*energy(result["full", 5], c))

    def test_post_g2p_contact_increment_reaches_retained_grid(self):
        c = config(transfer_mode="incremental", iterations=2)
        x = lattice((.5, .5, .5), (.1, .1, .1), 6)
        p = particle_arrays(x, FLUID, 1e-6, (0, 0, 0), c.dt, (.2, .5, .9))
        s = Solver(c, Scene(p, [], "increment persistence"))
        s.prepare_iteration(); s.transfers.p2g(); s.transfers.boundary(); s.transfers.g2p()
        changed = s.particles.snapshot()
        # A known uniform contact displacement has exactly known total impulse.
        changed["d"][:, 0] += .0003
        s.particles.data.from_numpy(changed)
        s.prepare_increment(); s.transfers.p2g_increment(); s.transfers.incremental_boundary(); s.transfers.g2p()
        actual = s.particles.snapshot()
        np.testing.assert_allclose(actual["d"][:, 0], .0003, atol=1e-8)
        np.testing.assert_allclose(actual["D"], 0, atol=2e-7)
        # An affine increment must retain its linear field, including boundary
        # stencils. Transferring only the translation increment cannot do this.
        strain = np.diag([.012, -.006, -.006]).astype(np.float32)
        expected_d = actual["d"]+(actual["x"]-.5)@strain.T
        actual["d"] = expected_d.copy()
        actual["D"] += strain
        mass = s.transfers.mass.to_numpy()
        occupancy = s.transfers.liquid_volume.to_numpy()
        s.particles.data.from_numpy(actual)
        s.prepare_increment(); s.transfers.p2g_increment(); s.transfers.incremental_boundary(); s.transfers.g2p()
        after = s.particles.snapshot()
        np.testing.assert_allclose(after["d"], expected_d, atol=1e-8)
        np.testing.assert_allclose(after["D"], np.broadcast_to(strain, after["D"].shape), atol=2e-7)
        np.testing.assert_array_equal(s.transfers.mass.to_numpy(), mass)
        np.testing.assert_array_equal(s.transfers.liquid_volume.to_numpy(), occupancy)

    def test_dynamic_rigid_reaction_and_fluid_tangent_survive(self):
        c = config(transfer_mode="incremental", dt=.01, iterations=4)
        scene, _ = fixture(c, fixed=False, velocity=(.1, .5, 0.))
        s = Solver(c, scene)
        initial = momentum(s)
        s.advance(3)
        np.testing.assert_allclose(momentum(s), initial, atol=1e-7)
        self.assertAlmostEqual(float(fluid_velocity(s)[1]), .5, places=5)
        self.assertGreater(np.linalg.norm(s.bodies.data.to_numpy()["contact_impulse"][0]), 1e-6)

    def test_reset_clears_references_and_repeats_step(self):
        c = config(transfer_mode="incremental")
        scene, _ = fixture(c, velocity=(.1, .5, 0.))
        s = Solver(c, scene)
        s.advance(3)
        expected = s.particles.snapshot()
        s.reset()
        for field in (s.transfers.increment, s.transfers.transported_d, s.transfers.transported_D):
            np.testing.assert_array_equal(field.to_numpy(), 0)
        s.advance(3)
        for key in ("x", "d", "D", "volume_ratio"):
            np.testing.assert_allclose(s.particles.snapshot()[key], expected[key], atol=1e-7)

    def test_elastic_scene_is_rejected_explicitly(self):
        c = config(transfer_mode="incremental")
        with self.assertRaisesRegex(ValueError, "water/rigid"):
            Solver(c, build_shake_scene(c, ELASTIC, resolution=8))


if __name__ == "__main__":
    unittest.main()
