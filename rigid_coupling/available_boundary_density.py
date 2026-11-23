"""Estimate liquid packing per volume available outside the solid union.

The numerator contains only real water rest volumes. The denominator is the
same B-spline convolution of box interior minus the union of rigid boxes.
G2P gathers the two fields before dividing; solid nodes are never divided.
"""
import numpy as np
import taichi as ti

from .solid_boundary_density import SolidBoundaryDensity, spline_cdf, spline_value
from .rigid import rotation
from .state import FLUID
from .transfers import valid_position


@ti.data_oriented
class AvailableBoundaryDensity(SolidBoundaryDensity):
    def __init__(self, particles, transfers, container, config, bodies):
        super().__init__(particles, transfers, container, config)
        self.b, self.count = bodies.data, bodies.count
        n = max(1, self.count)
        self.cached_body_x = ti.Vector.field(3, ti.f32, shape=n)
        self.cached_body_q = ti.Vector.field(4, ti.f32, shape=n)
        self.axes = ti.Matrix.field(3, 3, ti.f32, shape=n)
        self.body_valid = ti.field(ti.i32, shape=())
        self.rebuild_available = ti.field(ti.i32, shape=())
        self.axis_aligned = ti.field(ti.i32, shape=())
        self.z_roll = ti.field(ti.i32, shape=())
        self.abscissa = ti.field(ti.f32, shape=6)
        self.weights = ti.field(ti.f32, shape=6)
        self.abscissa.from_numpy(np.array([-.9324695142, -.6612093865, -.2386191861,
                                           .2386191861, .6612093865, .9324695142], np.float32))
        self.weights.from_numpy(np.array([.1713244924, .3607615730, .4679139346,
                                          .4679139346, .3607615730, .1713244924], np.float32))

    @property
    def build_kernels(self):
        return (self.check_pose, self.check_bodies, self.build_grid)

    @ti.kernel
    def check_bodies(self):
        changed = self.body_valid[None] == 0 or self.rebuild[None] != 0
        axis = True
        roll = True
        for a in ti.static(range(3)):
            for c in ti.static(range(3)):
                if ti.static(a != c):
                    axis = axis and ti.abs(self.box.orientation[None][a, c]) < 1e-6
        # Scalar pose checks stay in their own kernel, avoiding mixed cache
        # matrix/grid offloads that failed on Taichi 1.7 CUDA in solid mode.
        for b in ti.static(range(self.count)):
            changed = changed or (self.b[b].x != self.cached_body_x[b]).any() or (self.b[b].q != self.cached_body_q[b]).any()
            R = rotation(self.b[b].q)
            for a in ti.static(range(3)):
                for c in ti.static(range(3)):
                    if ti.static(a != c):
                        axis = axis and ti.abs(R[a, c]) < 1e-6
            roll = roll and ti.abs(R[0, 2])+ti.abs(R[1, 2])+ti.abs(R[2, 0])+ti.abs(R[2, 1]) < 1e-6
            self.axes[b] = R
            self.cached_body_x[b] = self.b[b].x
            self.cached_body_q[b] = self.b[b].q
        self.axis_aligned[None] = ti.cast(axis, ti.i32)
        self.z_roll[None] = ti.cast(roll, ti.i32)
        self.rebuild_available[None] = ti.cast(changed, ti.i32)
        self.body_valid[None] = 1

    @ti.func
    def interval(self, x, y, center, R, half, direction: ti.template()):
        # Intersect a world-axis ray with the three local slabs of an OBB.
        lo, hi = -1e6, 1e6
        possible = True
        for a in ti.static(range(3)):
            offset, slope = 0., 0.
            if ti.static(direction == 2):
                offset = R[0, a]*(x-center[0])+R[1, a]*(y-center[1])
                slope = R[2, a]
            else:
                offset = R[0, a]*(x-center[0])
                slope = R[1, a]
            if ti.abs(slope) > 1e-7:
                p = center[direction]+(-half[a]-offset)/slope
                q = center[direction]+(half[a]-offset)/slope
                lo, hi = ti.max(lo, ti.min(p, q)), ti.min(hi, ti.max(p, q))
            elif ti.abs(offset) > half[a]:
                possible = False
        if not possible:
            hi = lo
        return lo, hi

    @ti.func
    def axis_union(self, node):
        # Exact integrals of the intersections give the union once, including
        # overlapping boxes and clipping of boxes against the container.
        total = 0.
        for mask in range(1, 1 << self.count):
            lo = self.box.center[None]-ti.Vector(self.box.half_size)
            hi = self.box.center[None]+ti.Vector(self.box.half_size)
            bits = 0
            for b in ti.static(range(self.count)):
                if mask & (1 << b):
                    half = ti.abs(self.axes[b])@self.b[b].half_size
                    lo = ti.max(lo, self.b[b].x-half)
                    hi = ti.min(hi, self.b[b].x+half)
                    bits += 1
            value = 1.
            for a in ti.static(range(3)):
                value *= ti.max(0., spline_cdf(hi[a]*self.c.grid-node[a])-spline_cdf(lo[a]*self.c.grid-node[a]))
            total += value if bits % 2 else -value
        return total

    @ti.func
    def roll_union(self, node):
        # At fixed X, Z-roll boxes intersect as axis-aligned Y/Z rectangles.
        total = 0.
        for sample in range(18):
            u = sample//6-1.+.5*self.abscissa[sample % 6]
            x = (node[0]+u)/self.c.grid
            wall_lo, wall_hi = self.interval(x, 0., self.box.center[None], self.box.orientation[None], ti.Vector(self.box.half_size), 1)
            value = 0.
            for mask in range(1, 1 << self.count):
                ylo, yhi = wall_lo, wall_hi
                zlo = self.box.center[None][2]-self.box.half_size[2]
                zhi = self.box.center[None][2]+self.box.half_size[2]
                bits = 0
                for b in ti.static(range(self.count)):
                    if mask & (1 << b):
                        lo, hi = self.interval(x, 0., self.b[b].x, self.axes[b], self.b[b].half_size, 1)
                        ylo, yhi = ti.max(ylo, lo), ti.min(yhi, hi)
                        zlo = ti.max(zlo, self.b[b].x[2]-self.b[b].half_size[2])
                        zhi = ti.min(zhi, self.b[b].x[2]+self.b[b].half_size[2])
                        bits += 1
                yz = ti.max(0., spline_cdf(yhi*self.c.grid-node[1])-spline_cdf(ylo*self.c.grid-node[1]))
                yz *= ti.max(0., spline_cdf(zhi*self.c.grid-node[2])-spline_cdf(zlo*self.c.grid-node[2]))
                value += yz if bits % 2 else -yz
            total += .5*self.weights[sample % 6]*spline_value(u)*value
        return total

    @ti.func
    def ray_union(self, node, x, y):
        # For arbitrary rotations, integrate each union of Z intervals exactly.
        lo = ti.Vector.zero(ti.f32, max(1, self.count))
        hi = ti.Vector.zero(ti.f32, max(1, self.count))
        local = self.box.orientation[None].transpose()@(ti.Vector([x, y, self.box.center[None][2]])-self.box.center[None])
        possible = ti.abs(local[0]) <= self.box.half_size[0] and ti.abs(local[1]) <= self.box.half_size[1]
        for b in ti.static(range(self.count)):
            a, z = self.interval(x, y, self.b[b].x, self.axes[b], self.b[b].half_size, 2)
            lo[b] = ti.max(a, self.box.center[None][2]-self.box.half_size[2])
            hi[b] = ti.min(z, self.box.center[None][2]+self.box.half_size[2])
            if not possible or hi[b] <= lo[b]:
                lo[b], hi[b] = 1e6, 1e6
        total = 0.
        start, end = 1e6, -1e6
        # Sorting by the next lower endpoint is O(n^2), without exponential
        # inclusion/exclusion or double-counting of overlapping solid rays.
        for _ in range(self.count):
            index = 0
            lower = 1e6
            for b in ti.static(range(self.count)):
                if lo[b] < lower:
                    lower, index = lo[b], b
            if lower < 1e6:
                if lower > end:
                    if end > start:
                        total += spline_cdf(end*self.c.grid-node[2])-spline_cdf(start*self.c.grid-node[2])
                    start, end = lower, hi[index]
                else:
                    end = ti.max(end, hi[index])
                lo[index] = 1e6
        if end > start:
            total += spline_cdf(end*self.c.grid-node[2])-spline_cdf(start*self.c.grid-node[2])
        return total

    @ti.func
    def rigid_union(self, node):
        point = ti.cast(node, ti.f32)/self.c.grid
        boundary = False
        full = False
        for b in ti.static(range(self.count)):
            local = self.axes[b].transpose()@(point-self.b[b].x)
            extent = 1.5/self.c.grid*(ti.abs(self.axes[b].transpose())@ti.Vector([1., 1., 1.]))
            gap = self.b[b].half_size-ti.abs(local)
            boundary = boundary or (gap > -extent).all()
            full = full or (gap >= extent).all()
        total = 0.
        if full:
            total = 1.-self.solid_volume[node]
        elif boundary:
            if ti.static(self.count <= 4):
                if self.axis_aligned[None]:
                    total = self.axis_union(node)
                elif self.z_roll[None]:
                    total = self.roll_union(node)
                else:
                    for a in range(18):
                        u = a//6-1.+.5*self.abscissa[a % 6]
                        for b in range(18):
                            v = b//6-1.+.5*self.abscissa[b % 6]
                            total += .25*self.weights[a % 6]*self.weights[b % 6]*spline_value(u)*spline_value(v)*self.ray_union(node, (node[0]+u)/self.c.grid, (node[1]+v)/self.c.grid)
            else:
                for a in range(18):
                    u = a//6-1.+.5*self.abscissa[a % 6]
                    for b in range(18):
                        v = b//6-1.+.5*self.abscissa[b % 6]
                        total += .25*self.weights[a % 6]*self.weights[b % 6]*spline_value(u)*spline_value(v)*self.ray_union(node, (node[0]+u)/self.c.grid, (node[1]+v)/self.c.grid)
        return total

    @ti.kernel
    def build_grid(self):
        for node in ti.grouped(self.t.liquid_volume):
            if self.rebuild[None]:
                self.solid_volume[node] = self.occupancy(node)
            if self.rebuild_available[None]:
                available = 1.-self.solid_volume[node]
                if ti.static(self.count > 0):
                    if available > 0:
                        available -= self.rigid_union(node)
                self.t.available_volume[node] = ti.max(0., ti.min(1., available))
            self.t.liquid_volume[node] = 0.
        for p in range(self.active[None]):
            if self.p[p].material == FLUID:
                if valid_position(self.p[p].x, self.c.grid):
                    self.scatter_volume(self.p[p].x, self.p[p].rest_volume)
                else:
                    ti.atomic_max(self.t.invalid[None], 1)
