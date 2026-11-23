"""Interactive 3-D comparison without changing the simulation coordinates."""

from itertools import product
from pathlib import Path

import numpy as np
import taichi as ti

from .rigid import rotation
from .state import FLUID, RIGID, ELASTIC


def box_edges(lower, upper):
    corners = np.array(list(product(*zip(lower, upper))), dtype=np.float32)
    return np.array([point for i in range(8) for bit in (1, 2, 4)
                     if i < (i ^ bit) for point in (corners[i], corners[i ^ bit])])


def box_faces(lower, upper):
    """Twelve outward-facing triangles with separate vertices for flat shading."""
    corners = np.array(list(product(*zip(lower, upper))), dtype=np.float32)
    triangles = ((0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
                 (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
                 (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3))
    return corners[np.asarray(triangles).ravel()]


@ti.data_oriented
class RenderState:
    """The display offset applies only to render fields, never to solver state."""

    def __init__(self, solver, offset):
        self.solver = solver
        self.p = solver.particles.data
        self.active = solver.particles.count
        self.b = solver.bodies.data
        self.n = solver.particles.n
        self.offset = tuple(offset)
        self.positions = ti.Vector.field(3, ti.f32, shape=self.n)
        lo = solver.config.boundary / solver.config.grid
        self.domain = ti.Vector.field(3, ti.f32, shape=24)
        self.domain.from_numpy(box_edges([lo] * 3, [1 - lo] * 3) + np.asarray(offset, np.float32))
        initial = solver.initial
        fluid = initial["material"] == FLUID
        self.fluid_count = int(np.count_nonzero(fluid))
        self.soft_count = int(np.count_nonzero(initial["material"] == ELASTIC))
        fluid_volume = initial["rest_volume"][fluid].sum(dtype=np.float64)
        self.realized_ppc = self.fluid_count / (fluid_volume * solver.config.grid ** 3) if fluid_volume else None
        self.body_count = solver.bodies.count
        body_edges, body_faces = [], []
        for b in range(solver.bodies.count):
            local = initial["local"][initial["body_id"] == b]
            size = solver.bodies.initial[b].get("size")
            if size is None:
                lower, upper = local.min(axis=0), local.max(axis=0)
            else:
                half_size = np.asarray(size, np.float32) / 2
                lower, upper = -half_size, half_size
            body_edges.extend(box_edges(lower, upper))
            body_faces.extend(box_faces(lower, upper))
        self.local_edges = ti.Vector.field(3, ti.f32, shape=max(1, 24 * self.body_count))
        self.local_faces = ti.Vector.field(3, ti.f32, shape=max(1, 36 * self.body_count))
        self.faces = ti.Vector.field(3, ti.f32, shape=max(1, 36 * self.body_count))
        if body_edges:
            self.local_edges.from_numpy(np.asarray(body_edges, np.float32))
            self.local_faces.from_numpy(np.asarray(body_faces, np.float32))
        self.lines = ti.Vector.field(3, ti.f32, shape=max(1, 30 * solver.bodies.count))
        self.line_colors = ti.Vector.field(3, ti.f32, shape=max(1, 30 * solver.bodies.count))
        colors = np.tile(np.array([(0.65, 0.95, 0.57)] * 24 +
                                 [(1., 0.45, 0.25)] * 2 + [(0.3, 0.95, 0.95)] * 2 +
                                 [(0.9, 0.5, 1.)] * 2, np.float32), (self.body_count, 1))
        if self.body_count:
            self.line_colors.from_numpy(colors)
        # Separate fluid/rigid sampling densities need separate display radii.
        self.radii = ti.field(ti.f32, shape=self.n)
        self.base_radii = ti.field(ti.f32, shape=self.n)
        self.cutaway = ti.field(ti.i32, shape=())
        self.cutaway[None] = int(solver.config.world_contact)
        radii = np.clip(np.cbrt(initial["rest_volume"]) * .45, .002, .012).astype(np.float32)
        # Halve the displayed water radius after clipping, preserving physics.
        radii[fluid] *= 0.5
        radii[initial["material"] == RIGID] = 0
        self.radii.from_numpy(radii)
        self.base_radii.from_numpy(radii)

    @ti.kernel
    def update(self):
        for p in range(self.n):
            self.positions[p] = self.p[p].x + ti.Vector(self.offset)
            self.radii[p] = self.base_radii[p] if p < self.active[None] else 0.0
            # Reveal the submerged body through the front half of the pool.
            # This fixed world-space section affects only drawing; every
            # fluid sample still participates in the full simulation.
            if self.cutaway[None] and self.p[p].material == FLUID and self.p[p].x[2] > 0.5:
                self.radii[p] = 0.0
        for b in range(self.body_count):
            center = self.b[b].x + ti.Vector(self.offset)
            R = rotation(self.b[b].q)
            for edge in range(24):
                self.lines[30 * b + edge] = center + R @ self.local_edges[24 * b + edge]
            for vertex in range(36):
                self.faces[36 * b + vertex] = center + R @ self.local_faces[36 * b + vertex]
            for axis in ti.static(range(3)):
                self.lines[30 * b + 24 + 2 * axis] = center
                self.lines[30 * b + 25 + 2 * axis] = center + .13 * R[:, axis]


@ti.data_oriented
class RenderBatch:
    """Pack all panels into one particle, mesh and line submission.

    Simulation fields remain separate. Static geometry and colours are copied
    once; each frame only reads positions and rigid poses on the device.
    """

    def __init__(self, states):
        self.states = tuple(states)
        self.particle_offsets = tuple(np.cumsum([0] + [state.n for state in states[:-1]]).tolist())
        self.body_offsets = tuple(np.cumsum([0] + [state.body_count for state in states[:-1]]).tolist())
        self.n = sum(state.n for state in states)
        self.body_count = sum(state.body_count for state in states)
        self.domain_vertices = 24 * len(states)
        self.positions = ti.Vector.field(3, ti.f32, shape=self.n)
        self.radii = ti.field(ti.f32, shape=self.n)
        self.colors = ti.Vector.field(3, ti.f32, shape=self.n)
        self.colors.from_numpy(np.concatenate([state.solver.initial["color"] for state in states]))
        self.faces = ti.Vector.field(3, ti.f32, shape=max(1, 36 * self.body_count))
        self.normals = ti.Vector.field(3, ti.f32, shape=max(1, 36 * self.body_count))
        self.local_normals = ti.Vector.field(3, ti.f32, shape=max(1, 36 * self.body_count))
        self.lines = ti.Vector.field(3, ti.f32, shape=self.domain_vertices + 30 * self.body_count)
        self.line_colors = ti.Vector.field(3, ti.f32, shape=self.domain_vertices + 30 * self.body_count)
        lines = np.zeros((self.lines.shape[0], 3), np.float32)
        lines[:self.domain_vertices] = np.concatenate([state.domain.to_numpy() for state in states])
        self.lines.from_numpy(lines)
        colors = np.zeros_like(lines)
        colors[:self.domain_vertices] = (0.22, 0.27, 0.34)
        if self.body_count:
            colors[self.domain_vertices:] = np.concatenate(
                [state.line_colors.to_numpy() for state in states if state.body_count])
            faces = np.concatenate([state.local_faces.to_numpy() for state in states if state.body_count])
            triangles = faces.reshape(-1, 3, 3)
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            self.local_normals.from_numpy(np.repeat(normals, 3, axis=0).astype(np.float32))
        self.line_colors.from_numpy(colors)

    @ti.kernel
    def update(self):
        for index in ti.static(range(len(self.states))):
            self.update_panel(self.states[index], self.particle_offsets[index], self.body_offsets[index])

    @ti.func
    def update_panel(self, state: ti.template(), start: ti.template(), body_start: ti.template()):
        for p in range(state.n):
            self.positions[start + p] = state.p[p].x + ti.Vector(state.offset)
            radius = state.base_radii[p] if p < state.active[None] else 0.0
            if state.cutaway[None] and state.p[p].material == FLUID and state.p[p].x[2] > 0.5:
                radius = 0.0
            self.radii[start + p] = radius
        for b in range(state.body_count):
            center = state.b[b].x + ti.Vector(state.offset)
            R = rotation(state.b[b].q)
            line_start = self.domain_vertices + 30 * (body_start + b)
            face_start = 36 * (body_start + b)
            for edge in range(24):
                self.lines[line_start + edge] = center + R @ state.local_edges[24 * b + edge]
            for vertex in range(36):
                self.faces[face_start + vertex] = center + R @ state.local_faces[36 * b + vertex]
                self.normals[face_start + vertex] = R @ self.local_normals[face_start + vertex]
            for axis in ti.static(range(3)):
                self.lines[line_start + 24 + 2 * axis] = center
                self.lines[line_start + 25 + 2 * axis] = center + .13 * R[:, axis]


class Viewer:
    """Shared rendering fields and window setup for the shaking materials."""
    def __init__(self, solvers, show_window=True, fps_limit=60):
        self.solvers = solvers
        self.interactive = show_window
        water = solvers[0].mouse is not None
        title = "Mouse water | Left drag: push in drag direction" if water else "Shake box | Left drag: shake"
        self.window = ti.ui.Window(title+" | SPACE: pause | R: reset",
                                   (1280, 800), vsync=False, fps_limit=fps_limit, show_window=show_window)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.render = [RenderState(solver, (0, 0, 0)) for solver in solvers]
        self.batch = RenderBatch(self.render)


class PlaybackClock:
    """Elapsed wall time schedules fixed dt steps, independently of display FPS.

    A bounded backlog lets an overloaded simulation slow down without enlarging
    dt or blocking the UI with an ever-growing catch-up loop.
    """

    def __init__(self, dt, speed=1.0, max_steps=8):
        self.dt, self.speed, self.max_steps = dt, speed, max_steps
        self.accumulator = 0.0

    def reset(self):
        self.accumulator = 0.0

    def steps(self, elapsed, paused=False):
        if paused:
            self.reset()
            return 0
        self.accumulator = min(self.accumulator + max(0., elapsed) * self.speed,
                               self.max_steps * self.dt)
        count = min(self.max_steps, int((self.accumulator + 1e-12) / self.dt))
        self.accumulator = max(0., self.accumulator - count * self.dt)
        return count


class PlaybackStats:
    """Measure completed display frames and physical seconds per wall second."""

    def __init__(self, dt, interval=.5):
        self.dt, self.interval = dt, interval
        self.reset()

    def reset(self):
        self.elapsed = self.frames = self.steps = 0
        self.fps = None
        self.speed = 0.

    def record(self, elapsed, completed_steps):
        if elapsed <= 0:
            return
        self.elapsed += elapsed
        self.frames += 1
        self.steps += completed_steps
        if self.elapsed >= self.interval:
            self.fps = self.frames/self.elapsed
            self.speed = self.steps*self.dt/self.elapsed
            self.elapsed = self.frames = self.steps = 0

