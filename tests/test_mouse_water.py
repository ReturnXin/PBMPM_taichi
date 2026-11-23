"""Drag increments set force direction; stationary clicks and releases add none."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.mouse_water import WaterFollow, horizontal_drag
from rigid_coupling.scenes import Scene
from rigid_coupling.solver import Solver
from rigid_coupling.shake import ShakeViewer
from rigid_coupling.state import FLUID, ELASTIC, particle_arrays


def camera():
    return SimpleNamespace(curr_position=[.5, 1.3, 2.],
                           curr_lookat=[.5, .3, .5], curr_up=[0, 1, 0])


class CursorChecks(unittest.TestCase):
    def test_left_button_movement_only_and_release(self):
        viewer = SimpleNamespace(window=Mock(), camera=camera(), water_control=True,
                                 container=SimpleNamespace(motion=SimpleNamespace(automatic=False)))
        cursor, buttons = [.5, .5], set()
        viewer.window.get_cursor_pos.side_effect = lambda: cursor
        viewer.window.is_pressed.side_effect = lambda key: key in buttons
        mouse = Mock()
        viewer.follow = WaterFollow(mouse)
        ShakeViewer.update_input(viewer)
        mouse.push.assert_not_called()
        buttons.add(ti.ui.LMB)
        ShakeViewer.update_input(viewer)
        ShakeViewer.update_input(viewer)
        mouse.push.assert_not_called()
        cursor[0] += .01
        ShakeViewer.update_input(viewer, steps=2)
        np.testing.assert_allclose(mouse.push.call_args.args[0], [.016, 0., 0.], atol=1e-12)
        self.assertEqual(mouse.push.call_args.args[1], 2)
        buttons.clear()
        ShakeViewer.update_input(viewer)
        mouse.release.assert_called_once()
        cursor[0] += .02
        ShakeViewer.update_input(viewer)
        mouse.push.assert_called_once()

    def test_direction_comes_from_delta_at_any_cursor_location(self):
        cam = camera()
        for start in ([.2, .4], [.7, .6]):
            mouse = Mock(); follow = WaterFollow(mouse)
            follow.update(start, True, cam)
            follow.update(np.asarray(start)+[.01, .02], True, cam)
            np.testing.assert_allclose(mouse.push.call_args.args[0], [.016, 0., -.02], atol=1e-12)
            # Both cursor positions can remain to the right of the water;
            # dragging left must still produce a leftward force.
            follow.update(np.asarray(start)+[.005, .02], True, cam)
            np.testing.assert_allclose(mouse.push.call_args.args[0], [-.008, 0., 0.], atol=1e-12)

    def test_delta_buffering_does_not_repeat_impulses(self):
        mouse = Mock(); follow = WaterFollow(mouse); cam = camera()
        follow.update([.5, .5], True, cam, steps=0)
        follow.update([.51, .5], True, cam, steps=0)
        mouse.push.assert_not_called()
        follow.update([.51, .5], True, cam, steps=2)
        mouse.push.assert_called_once()
        follow.update([.51, .5], True, cam, steps=2)
        mouse.push.assert_called_once()
        follow.update([.52, .5], True, cam, steps=0)
        follow.update([.52, .5], False, cam)
        np.testing.assert_array_equal(follow.pending, 0)
        follow.update([.52, .5], True, cam)
        mouse.push.assert_called_once()

    def test_camera_rotation_changes_only_horizontal_direction(self):
        cam = camera()
        np.testing.assert_allclose(horizontal_drag([.1, .1], cam), [.16, 0., -.1])
        cam.curr_position = [2., 1.3, .5]
        np.testing.assert_allclose(horizontal_drag([.1, .1], cam), [-.1, 0., -.16])


class MousePredictionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False,
                offline_cache_file_path="results/mouse_water_20261102/drag_delta/test_cache")

    @classmethod
    def tearDownClass(cls):
        ti.reset()

    def make_solver(self, gravity=0, dt=.002):
        c = Config(gravity=gravity, dt=dt, iterations=2, world_contact=True)
        arrays = particle_arrays([[.5, .3, .5], [.52, .3, .5], [.70, .3, .5], [.5, .32, .5]],
                                 FLUID, .001, (.1, .2, -.1), c.dt, (0, .6, 1))
        arrays["material"][-1] = ELASTIC
        return Solver(c, Scene(arrays, [], "delta force",
                      container=dict(center=[.5, .48, .5], half_size=[.25, .27, .23])), mouse_control=True)

    def test_horizontal_acceleration_integrates_for_scheduled_duration(self):
        s = self.make_solver(gravity=-9.8)
        initial = s.particles.snapshot()
        s.mouse.push([.001, 100., -.002], steps=2)
        for _ in range(2): s.materials.predict()
        after = s.particles.snapshot()
        expected = s.mouse.gain*np.array([.001, 0., -.002])*s.config.dt**2*2
        for axis in (0, 2):
            np.testing.assert_allclose(after["d"][:3, axis]-initial["d"][:3, axis], expected[axis], atol=1e-10)
        np.testing.assert_array_equal(after["d"][3, [0, 2]], initial["d"][3, [0, 2]])
        np.testing.assert_allclose(after["d"][:, 1], initial["d"][:, 1]+2*s.config.gravity*s.config.dt**2,
                                   atol=1e-10, rtol=1e-7)
        self.assertEqual(s.mouse.control[None][3], 0)
        s.materials.predict()
        np.testing.assert_array_equal(s.particles.snapshot()["d"][:, [0, 2]], after["d"][:, [0, 2]])
        np.testing.assert_array_equal(s.mouse.step_delta[None].to_numpy(), 0)
        s.mouse.push([-.001, 0., 0.])
        s.materials.predict()
        reversed_state = s.particles.snapshot()
        self.assertTrue(np.all(reversed_state["d"][:3, 0]-after["d"][:3, 0] < 0))
        s.mouse.push([1., 0., 0.]); s.mouse.release(); s.materials.predict()
        np.testing.assert_array_equal(s.particles.snapshot()["d"][:, [0, 2]], reversed_state["d"][:, [0, 2]])
        for key in ("x", "mass", "rest_volume", "D"):
            np.testing.assert_array_equal(s.particles.snapshot()[key], initial[key])
        s.reset()
        np.testing.assert_array_equal(s.mouse.control[None].to_numpy(), 0)
        np.testing.assert_array_equal(s.mouse.step_delta[None].to_numpy(), 0)

    def test_acceleration_is_independent_of_steps_and_large_force_is_bounded(self):
        s = self.make_solver()
        initial = s.particles.snapshot()
        delta = [.001, 0., .001]
        s.mouse.push(delta, steps=1)
        single_acceleration = s.mouse.control[None].to_numpy()[:3]
        s.materials.predict()
        single = s.particles.snapshot()["d"]-initial["d"]
        s.reset(); s.mouse.push(delta, steps=2)
        np.testing.assert_array_equal(s.mouse.control[None].to_numpy()[:3], single_acceleration)
        s.materials.predict(); s.materials.predict()
        np.testing.assert_allclose(s.particles.snapshot()["d"]-initial["d"], single*2, atol=1e-10)
        s.reset(); s.mouse.push([100., 0., -100.]); s.materials.predict()
        increment = s.particles.snapshot()["d"][0]-initial["d"][0]
        self.assertLessEqual(np.linalg.norm(increment)/s.config.dt,
                             s.mouse.max_acceleration*s.config.dt+1e-7)

    def test_fixed_acceleration_over_same_physical_time_preserves_velocity_change(self):
        changes = []
        for dt, steps in ((.002, 2), (.001, 4)):
            s = self.make_solver(dt=dt)
            before = s.particles.snapshot()
            s.mouse.push([.001, 0., -.001], steps=steps)
            np.testing.assert_allclose(s.mouse.control[None].to_numpy()[:3], [.35, 0., -.35], atol=1e-7)
            for _ in range(steps): s.materials.predict()
            changes.append((s.particles.snapshot()["d"][:3]-before["d"][:3])/dt)
        np.testing.assert_allclose(changes[0], changes[1], atol=2e-8)
        np.testing.assert_allclose(changes[0], np.tile([.0014, 0., -.0014], (3, 1)), atol=2e-8)


if __name__ == "__main__": unittest.main()
