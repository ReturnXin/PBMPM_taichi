"""Moving-wall response, FPS-independent mouse targets and bounded container motion."""
import unittest
from unittest.mock import patch

import numpy as np
import taichi as ti

from rigid_coupling import cli
from rigid_coupling.config import Config
from rigid_coupling.container import BoxMotion, roll_matrix
from rigid_coupling.scenes import Scene
from rigid_coupling.shake import MouseDrag
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID, ELASTIC, particle_arrays


class MotionChecks(unittest.TestCase):
    def test_mouse_drag_starts_without_jump_and_releases_anchor(self):
        motion = BoxMotion([.5, .48, .5], [.25, .27, .23], 32, 3)
        drag = MouseDrag()
        drag.update([.5, .5], True, motion)
        np.testing.assert_array_equal(motion.target, motion.origin)
        drag.update([.6, .55], True, motion)
        self.assertGreater(motion.target[0], motion.origin[0])
        self.assertLess(motion.target[2], motion.origin[2])
        self.assertGreater(motion.target_angle, 0)
        drag.update([.6, .55], False, motion)
        self.assertIsNone(drag.anchor)
        self.assertFalse(drag.update([.1, .1], True, motion))

    def test_extreme_mouse_targets_keep_entire_rotated_box_inside_grid(self):
        motion = BoxMotion([.5, .48, .5], [.25, .27, .23], 32, 3)
        for sign in (1, -1, 1):
            motion.set_target([sign*100, -sign*100], sign*100)
            for _ in range(400):
                motion.step(.002)
                extent = np.abs(roll_matrix(motion.angle))@motion.half_size
                self.assertTrue(np.all(motion.center-extent >= motion.lower-1e-7))
                self.assertTrue(np.all(motion.center+extent <= motion.upper+1e-7))
                self.assertLessEqual(abs(motion.angular_velocity), 1.8+1e-7)
        motion.reset()
        self.assertEqual(motion.time, 0)
        np.testing.assert_array_equal(motion.center, motion.origin)

    def test_recording_routes_both_materials_through_same_entry(self):
        for name in ("shake_water", "shake_soft"):
            with self.subTest(name=name), patch("rigid_coupling.shake.run") as run:
                cli.main(["--scene", name, "--gif-dir", "unused", "--duration", "2"])
                selected, configs, args = run.call_args.args
                self.assertEqual(selected, name)
                self.assertLess(configs[0].gravity, 0)
                self.assertEqual(configs[0].buoyancy_assist, name == "shake_water")
                self.assertEqual(args.duration, 2)
                self.assertEqual(configs[0].soft_friction, 0.)
                self.assertEqual(args.substeps, 2)

    def test_scenes_contain_three_floats_or_three_soft_plus_three_rigid_objects(self):
        from rigid_coupling.shake import build_shake_scene
        for material in (FLUID, ELASTIC):
            scene = build_shake_scene(Config(), material, resolution=16)
            self.assertEqual(len(scene.bodies), 3)
            self.assertEqual(scene.sampling["soft_object_count"], 3 if material == ELASTIC else 0)
            for body_id in range(3):
                self.assertEqual(np.count_nonzero(scene.particles["body_id"] == body_id), 64)


class MovingWallChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False,
                offline_cache_file_path="results/shake_box_20261030/test_cache")

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def make_solver(self, material, velocity=(0, .1, 0), friction=.45):
        config = Config(gravity=0, rigid_contact=True, world_contact=True,
                        grid_coupling="separate", soft_friction=friction)
        p = particle_arrays([[.7504, .45, .5]], material, .001, velocity, config.dt, (1, .5, 0))
        scene = Scene(p, [], "single wall contact", container=dict(center=[.5, .48, .5], half_size=[.25, .27, .23]))
        return Solver(config, scene)

    def test_fluid_wall_stops_penetration_without_tangential_glue(self):
        solver = self.make_solver(FLUID)
        before = solver.particles.snapshot()
        solver.container.project()
        after = solver.particles.snapshot()
        self.assertAlmostEqual(float((after["x"]+after["d"])[0, 0]), .75, places=6)
        self.assertAlmostEqual(float(after["d"][0, 1]), float(before["d"][0, 1]), places=8)
        delta = after["mass"][0]*(after["d"][0]-before["d"][0])/solver.config.dt
        np.testing.assert_allclose(delta+solver.container.wall_impulse[None].to_numpy(), 0, atol=1e-8)

    def test_zero_friction_keeps_soft_tangent_at_container_and_rigid_faces(self):
        solver = self.make_solver(ELASTIC, friction=0.)
        before = solver.particles.snapshot()
        solver.container.project()
        after = solver.particles.snapshot()
        self.assertAlmostEqual(float((after["x"]+after["d"])[0, 0]), .75, places=6)
        np.testing.assert_array_equal(after["d"][0, 1:], before["d"][0, 1:])

        from rigid_coupling.rigid import make_body
        from rigid_coupling.scenes import lattice
        from rigid_coupling.state import RIGID, combine_particles
        c = solver.config
        local = lattice((0, 0, 0), (.1, .1, .1), 4)
        body = make_body(local, (.5, .5, .5), .001)
        body.update(size=(.1, .1, .1), fixed=True)
        soft = particle_arrays([[.5496, .5, .5]], ELASTIC, .001,
                               (0, .1, .05), c.dt, (1, .5, 0))
        rigid = particle_arrays(body["center"]+body["local"], RIGID, .001/len(local),
                                (0, 0, 0), c.dt, (.5, .8, .4), body_id=0,
                                local=body["local"], mass=body["particle_mass"])
        contact = Solver(c, Scene(combine_particles([soft, rigid]), [body], "soft slip"))
        before = contact.particles.snapshot()
        contact.contacts.project()
        after = contact.particles.snapshot()
        self.assertAlmostEqual(float((after["x"]+after["d"])[0, 0]), .55, places=6)
        np.testing.assert_array_equal(after["d"][0, 1:], before["d"][0, 1:])
        np.testing.assert_array_equal(contact.contacts.friction_impulse.to_numpy(), 0)

    def test_grid_wall_contact_is_visible_to_affine_volume_update(self):
        solver = self.make_solver(FLUID, velocity=(.1, -1, 0))
        arrays = solver.particles.snapshot()
        arrays["x"][0] = [.5, .212, .5]
        solver.particles.data.from_numpy(arrays)
        solver.transfers.clear()
        solver.transfers.p2g()
        solver.transfers.boundary()
        solver.container.grid_boundary(solver.transfers)
        solver.transfers.g2p()
        corrected = solver.particles.snapshot()
        self.assertLess(corrected["D"][0, 1, 1], -1e-4)
        self.assertGreater(corrected["d"][0, 1], -solver.config.dt)
        self.assertAlmostEqual(float(corrected["d"][0, 0]), .1*solver.config.dt, places=7)

    def test_soft_wall_friction_and_compression_enter_deformation(self):
        solver = self.make_solver(ELASTIC)
        before = solver.particles.snapshot()
        solver.container.project()
        projected = solver.particles.snapshot()
        self.assertLess(projected["d"][0, 1], before["d"][0, 1])
        # Use a spatially extended block to test the shared deformation reconstruction.
        from rigid_coupling.shake import build_shake_scene
        scene = build_shake_scene(solver.config, ELASTIC, resolution=8)
        # Move the far-right soft object through the +x wall.
        scene.particles["x"][:, 0] += .13
        expanded = Solver(solver.config, scene)
        expanded.container.project()
        projected = expanded.particles.snapshot()
        expanded.container.update_contact_affine()
        corrected = expanded.particles.snapshot()
        self.assertLess(np.mean(corrected["D"][:, 0, 0]), -1e-5)
        np.testing.assert_allclose(np.sum(corrected["mass"][:, None]*corrected["d"], axis=0),
                                   np.sum(projected["mass"][:, None]*projected["d"], axis=0),
                                   atol=1e-9, rtol=3e-6)

    def test_rotated_moving_wall_and_reset_are_consistent(self):
        solver = self.make_solver(FLUID, velocity=(0, 0, 0))
        motion = solver.container.motion
        motion.set_target([-.08, 0], -.20)
        for _ in range(50):
            solver.container.begin_step()
        solver.container.project()
        p = solver.particles.snapshot()
        local = (p["x"]+p["d"]-motion.center)@roll_matrix(motion.angle)
        self.assertTrue(np.all(np.abs(local) <= np.array(solver.container.half_size)+1e-6))
        self.assertGreater(solver.container.contact_count[None], 0)
        solver.reset()
        self.assertEqual(solver.container.contact_count[None], 0)
        np.testing.assert_array_equal(solver.container.wall_impulse[None].to_numpy(), 0)
        self.assertEqual(motion.time, 0)

    def test_grouped_wall_metrics_match_particle_reaction_with_partial_last_tile(self):
        # More than two tiles, several corner contacts, both materials and a
        # shrinking active range expose stale totals and lost reactions.
        for friction in (0., .45):
            with self.subTest(friction=friction):
                n = 307
                c = Config(gravity=0, soft_friction=friction, rigid_contact=True,
                           grid_coupling="separate")
                angle = .17
                R = roll_matrix(angle)
                center = np.array([.5, .48, .5], np.float32)
                half = np.array([.25, .27, .23], np.float32)
                rng = np.random.default_rng(13)
                local = rng.uniform(-1, 1, (n, 3)).astype(np.float32)*half
                local[::2, 0] = half[0]+.0007
                local[::3, 1] = -half[1]-.0009
                local[::5, 2] = half[2]+.0006
                arrays = particle_arrays(center+local@R.T, ELASTIC, .001,
                                         (.08, .06, -.05), c.dt, (1, .5, 0),
                                         mass=np.linspace(.001, .003, n))
                arrays["material"][::2] = FLUID
                scene = Scene(arrays, [], "grouped wall metrics",
                              container=dict(center=center, half_size=half))
                solver = Solver(c, scene)
                solver.container.upload_pose(*center, .03, -.02, .01, angle, .2)
                # Fill all tile slots before repeating with a smaller range.
                solver.container.project()
                for active in (263, 11):
                    solver.particles.data.from_numpy(arrays)
                    solver.release_to(active)
                    solver.container.wall_impulse.fill(0)
                    solver.container.contact_count.fill(0)
                    expected = arrays["d"].copy()
                    hits = 0
                    for p in range(active):
                        for axis in range(3):
                            point = arrays["x"][p]+expected[p]
                            local_point = R.T@(point-center)
                            depth = abs(local_point[axis])-half[axis]
                            if depth > 0:
                                normal = (-1 if local_point[axis] >= 0 else 1)*R[:, axis]
                                expected[p] += depth*normal
                                if friction and arrays["material"][p] == ELASTIC:
                                    relative = expected[p]/c.dt-np.array([.03, -.02, .01])-np.cross([0, 0, .2], point-center)
                                    tangent = relative-normal*np.dot(relative, normal)
                                    speed = np.linalg.norm(tangent)
                                    if speed > 1e-8:
                                        expected[p] -= c.dt*min(speed, friction*depth/c.dt)*tangent/speed
                                hits += 1
                    reaction = -np.sum(arrays["mass"][:active, None]*(expected[:active]-arrays["d"][:active])/c.dt,
                                       axis=0, dtype=np.float64)
                    solver.container.project()
                    actual = solver.particles.snapshot()
                    np.testing.assert_allclose(actual["d"], expected, atol=2e-7, rtol=2e-5)
                    np.testing.assert_allclose(solver.container.wall_impulse[None].to_numpy(), reaction,
                                               atol=2e-5, rtol=2e-5)
                    self.assertEqual(solver.container.contact_count[None], hits)

    def test_small_rigid_boxes_use_whole_shape_contact_and_pair_momentum(self):
        from rigid_coupling.shake import build_shake_scene
        from rigid_coupling.rigid import rotation_numpy
        config = Config(gravity=0, rigid_contact=True, world_contact=True, grid_coupling="separate")
        solver = Solver(config, build_shake_scene(config, ELASTIC, resolution=8))
        b = solver.bodies.data.to_numpy()
        b["x"][0] = [.80, .5, .5]
        b["v"][0] = [.5, 0, 0]
        solver.bodies.data.from_numpy(b)
        solver.container.resolve_bodies()
        solved = solver.bodies.data.to_numpy()
        R = rotation_numpy(solved["q"][0])
        extent = np.abs(R)@solved["half_size"][0]
        self.assertLessEqual(solved["x"][0, 0]+extent[0], .750001)
        solver.bodies.reconstruct()
        p = solver.particles.snapshot()
        own = p["body_id"] == 0
        np.testing.assert_allclose(p["x"][own], solved["x"][0]+p["local"][own]@R.T, atol=1e-7)
        b = solver.bodies.data.to_numpy()
        b["x"][:3] = [[.45, .5, .5], [.52, .5, .5], [.65, .65, .65]]
        b["q"][:3] = [[1, 0, 0, 0]]*3
        b["omega"][:3] = 0
        b["v"][:3] = [[.2, 0, 0], [-.2, 0, 0], [0, 0, 0]]
        solver.bodies.data.from_numpy(b)
        before = np.sum(b["mass"][:3, None]*b["v"][:3], axis=0)
        solver.box_pairs.project()
        after = solver.bodies.data.to_numpy()
        np.testing.assert_allclose(np.sum(after["mass"][:3, None]*after["v"][:3], axis=0), before, atol=1e-9)
        self.assertGreaterEqual(after["x"][1, 0]-after["x"][0, 0], .094999)
        self.assertLessEqual(after["v"][0, 0]-after["v"][1, 0], 1e-6)

    def test_batched_interface_matches_separate_contact_passes(self):
        from rigid_coupling.shake import build_shake_scene
        from rigid_coupling.rigid import rotation_numpy
        c = Config(gravity=0, rigid_contact=True, world_contact=True,
                   grid_coupling="separate", soft_friction=0.)
        for material in (FLUID, ELASTIC):
            with self.subTest(material=material):
                scene = build_shake_scene(c, material, resolution=8)
                scene.particles["x"][0] = [.7504, .45, .5]
                body = scene.bodies[0]
                scene.particles["x"][1] = body["center"]+rotation_numpy(body["quaternion"])@(
                    np.array(body["size"])*[.5, 0, 0]-[.0004, 0, 0])
                scene.particles["d"][:2] = [0, .0002, .0001]
                reference = Solver(c, scene)
                reference.contacts.project()
                reference.bodies.apply_contact_feedback()
                reference.contacts.update_contact_affine()
                reference.container.project()
                reference.container.update_contact_affine()
                actual = Solver(c, scene)
                # Fused interfaces follow P2G/G2P and reuse its frozen-geometry
                # mass field; standalone correction builds its own equivalent.
                actual.transfers.p2g()
                actual.project_interfaces()
                expected_p, actual_p = reference.particles.snapshot(), actual.particles.snapshot()
                for key in ("x", "d", "D", "F"):
                    np.testing.assert_allclose(actual_p[key], expected_p[key], atol=2e-7, rtol=2e-6)
                expected_b, actual_b = reference.bodies.data.to_numpy(), actual.bodies.data.to_numpy()
                for key in ("v", "omega", "contact_impulse", "delta_momentum", "delta_angular"):
                    np.testing.assert_allclose(actual_b[key], expected_b[key], atol=2e-7, rtol=2e-6)
                np.testing.assert_allclose(actual.container.wall_impulse[None].to_numpy(),
                                           reference.container.wall_impulse[None].to_numpy(), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
