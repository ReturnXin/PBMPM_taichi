"""Left-button drag increments push all water in the horizontal X/Z plane."""
import numpy as np
import taichi as ti

from .state import FLUID


def horizontal_drag(delta, camera, aspect=1.6):
    """Screen drag maps to camera-relative horizontal right and forward."""
    front = np.asarray(camera.curr_lookat, dtype=float)-np.asarray(camera.curr_position, dtype=float)
    right = np.cross(front, np.asarray(camera.curr_up, dtype=float))
    right[1] = 0.
    length = np.linalg.norm(right)
    right = right/length if length > 1e-8 else np.array([1., 0., 0.])
    forward = np.cross([0., 1., 0.], right)
    return aspect*delta[0]*right+delta[1]*forward


class WaterFollow:
    """Only movement since the last cursor sample generates mouse force."""
    def __init__(self, mouse):
        self.mouse = mouse
        self.previous = None
        self.pending = np.zeros(3)

    def reset(self):
        if self.previous is not None:
            self.mouse.release()
        self.previous = None
        self.pending.fill(0)

    def update(self, cursor, enabled, camera, steps=1):
        if not enabled or not (.04 < cursor[0] < .96 and .20 < cursor[1] < .86):
            self.reset()
            return False
        cursor = np.asarray(cursor, dtype=float)
        if self.previous is None:
            # Pressing at a new location never produces an initial impulse.
            self.previous = cursor.copy()
            return False
        self.pending += horizontal_drag(cursor-self.previous, camera)
        self.previous = cursor.copy()
        # Buffer movement when no physics step is scheduled. Each increment
        # is submitted once and distributed over this frame's fixed dt steps.
        if steps > 0 and np.linalg.norm(self.pending) > 1e-12:
            self.mouse.push(self.pending.copy(), steps)
            self.pending.fill(0)
            return True
        return False


@ti.data_oriented
class WaterMouse:
    """Uniform drag force, with no position attraction or velocity damping."""
    def __init__(self, particles, config, container):
        self.p, self.c = particles.data, config
        self.gain = 350.
        self.max_acceleration = 15.75
        # xyz = acceleration, w = remaining scheduled physics steps.
        self.control = ti.Vector.field(4, ti.f32, shape=())
        self.step_delta = ti.Vector.field(3, ti.f32, shape=())
        self.reset()

    def reset(self):
        self.control.fill(0)
        self.step_delta.fill(0)

    def push(self, delta, steps=1):
        if steps < 1:
            return
        delta = np.asarray(delta, dtype=float).copy()
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("mouse drag must be a finite three-component vector")
        delta[1] = 0.
        # Drag sets acceleration directly; its strength does not depend on
        # dt or the number of scheduled steps. Physics integrates it below.
        acceleration = self.gain*delta
        magnitude = np.linalg.norm(acceleration)
        if magnitude > self.max_acceleration:
            acceleration *= self.max_acceleration/magnitude
        self.set_force(*map(float, acceleration), steps)

    def release(self):
        self.set_force(0., 0., 0., 0)

    @ti.kernel
    def set_force(self, x: ti.f32, y: ti.f32, z: ti.f32, steps: ti.i32):
        self.control[None] = ti.Vector([x, y, z, ti.cast(steps, ti.f32)])
        self.step_delta[None] = ti.Vector.zero(ti.f32, 3)

    @ti.func
    def prepare_step(self):
        self.step_delta[None] = ti.Vector.zero(ti.f32, 3)
        if self.control[None][3] > 0:
            # PBMPM stores d = v*dt, so adding a*dt^2 is v += a*dt.
            self.step_delta[None] = self.control[None].xyz*self.c.dt**2
            self.control[None][3] -= 1.

    @ti.func
    def apply_particle(self, p):
        if self.p[p].material == FLUID:
            self.p[p].d += self.step_delta[None]
