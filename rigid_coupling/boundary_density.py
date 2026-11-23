"""Extend liquid packing across the prescribed solid box by reflected samples.

Only objective packing density receives virtual samples. Particle mass,
displacement, contact and rigid feedback continue to use physical samples.
Geometry is frozen within each PBMPM step, so this field is built once and
reused by all constraint iterations. Free surfaces receive no reflections.
"""
import taichi as ti

from .state import FLUID, RIGID
from .transfers import stencil, valid_position


@ti.data_oriented
class BoundaryDensity:
    def __init__(self, particles, transfers, container, config):
        self.p = particles.data
        self.active = particles.count
        self.t, self.box, self.c = transfers, container, config

    @property
    def build_kernels(self):
        return (self.build,)

    @ti.func
    def scatter_volume(self, x, volume):
        base, f, w = stencil(x, self.c.grid)
        for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
            node = base+o
            # Reflected points may be outside the world grid; only real
            # particles require a complete stencil. Drop unused ghost nodes.
            if (node >= 0).all() and (node < self.c.grid).all():
                weight = w[o[0], 0]*w[o[1], 1]*w[o[2], 2]
                self.t.liquid_volume[node] += weight*volume*self.c.grid**3

    @ti.kernel
    def build(self):
        for node in ti.grouped(self.t.liquid_volume):
            self.t.liquid_volume[node] = 0
        for p in range(self.active[None]):
            if self.p[p].material == FLUID or (ti.static(self.c.rigid_volume) and self.p[p].material == RIGID):
                x = self.p[p].x
                if valid_position(x, self.c.grid):
                    self.scatter_volume(x, self.p[p].rest_volume)
                    R, center = self.box.orientation[None], self.box.center[None]
                    local = R.transpose()@(x-center)
                    distance = ti.Vector.zero(ti.f32, 3)
                    side = ti.Vector.zero(ti.f32, 3)
                    cutoff = ti.Vector.zero(ti.f32, 3)
                    for a in ti.static(range(3)):
                        distance[a] = self.box.half_size[a]-ti.abs(local[a])
                        side[a] = 1.0 if local[a] >= 0 else -1.0
                        # P2G followed by G2P has twice the B-spline support.
                        # Project that support onto the actual wall normal.
                        for b in ti.static(range(3)):
                            cutoff[a] += 3.0*ti.abs(R[b, a])/self.c.grid
                    # Reflections across perpendicular box faces commute.
                    # Combined masks fill edge/corner support once each.
                    for mask in range(1, 8):
                        ghost = local
                        eligible = True
                        for a in ti.static(range(3)):
                            if mask & (1 << a):
                                eligible = eligible and distance[a] < cutoff[a]
                                ghost[a] = 2.0*side[a]*self.box.half_size[a]-local[a]
                        if eligible:
                            self.scatter_volume(center+R@ghost, self.p[p].rest_volume)
                else:
                    ti.atomic_max(self.t.invalid[None], 1)
