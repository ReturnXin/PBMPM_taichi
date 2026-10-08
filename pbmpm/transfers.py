import taichi as ti

from .state import FLUID


@ti.func
def stencil(x, grid):
    scaled = x * grid
    base = ti.cast(ti.floor(scaled - 0.5), ti.i32)
    f = scaled - base
    weights = ti.Matrix.rows([0.5 * (1.5 - f)**2, 0.75 - (f - 1.0)**2, 0.5 * (f - 0.5)**2])
    return base, f, weights


@ti.data_oriented
class Transfers:
    def __init__(self, particles, config):
        self.p, self.n, self.c = particles.data, particles.n, config
        shape = (config.grid,) * 3
        self.mass = ti.field(ti.f32, shape=shape)
        self.displacement = ti.Vector.field(3, ti.f32, shape=shape)
        self.liquid_volume = ti.field(ti.f32, shape=shape)
        self.fixed = ti.field(ti.i32, shape=shape)
        self.build_fixed_support()

    @ti.kernel
    def build_fixed_support(self):
        # Pinned particles are stationary; this mask also survives storage reordering.
        # The full interpolation support is clamped, giving a grid-scale transition.
        for i in ti.grouped(self.fixed):
            self.fixed[i] = 0
        for p in range(self.n):
            if self.p[p].pinned:
                base, f, w = stencil(self.p[p].x, self.c.grid)
                for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                    if w[o[0], 0] * w[o[1], 1] * w[o[2], 2] > 0:
                        ti.atomic_max(self.fixed[base + o], 1)

    @ti.kernel
    def clear(self):
        for i in ti.grouped(self.mass):
            self.mass[i] = 0
            self.displacement[i] = ti.Vector.zero(ti.f32, 3)
            self.liquid_volume[i] = 0

    @ti.kernel
    def p2g(self):
        for p in range(self.n):
            base, f, w = stencil(self.p[p].x, self.c.grid)
            for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                weight = w[o[0], 0] * w[o[1], 1] * w[o[2], 2]
                arm = (o - f) / self.c.grid
                mass = weight * self.p[p].mass
                self.mass[base + o] += mass
                self.displacement[base + o] += mass * (self.p[p].d + self.p[p].D @ arm)
                if self.p[p].material == FLUID:
                    self.liquid_volume[base + o] += weight * self.p[p].rest_volume * self.c.grid**3

    @ti.kernel
    def boundary(self):
        for i in ti.grouped(self.mass):
            if self.mass[i] > 1e-12:
                d = self.displacement[i] / self.mass[i]
                for a in ti.static(range(3)):
                    if i[a] <= self.c.boundary and d[a] < 0:
                        d[a] = 0
                    if i[a] >= self.c.grid - self.c.boundary and d[a] > 0:
                        d[a] = 0
                if self.fixed[i]:
                    d = ti.Vector.zero(ti.f32, 3)
                self.displacement[i] = d

    @ti.kernel
    def g2p(self):
        for p in range(self.n):
            base, f, w = stencil(self.p[p].x, self.c.grid)
            d = ti.Vector.zero(ti.f32, 3)
            D = ti.Matrix.zero(ti.f32, 3, 3)
            volume = 0.0
            for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                weight = w[o[0], 0] * w[o[1], 1] * w[o[2], 2]
                arm = (o - f) / self.c.grid
                gd = self.displacement[base + o]
                d += weight * gd
                D += 4 * self.c.grid**2 * weight * gd.outer_product(arm)
                if self.p[p].material == FLUID:
                    volume += weight * self.liquid_volume[base + o]
            if not self.p[p].pinned:
                self.p[p].d = d
                self.p[p].D = D
            if self.p[p].material == FLUID and volume > 1.0:
                self.p[p].volume_ratio = 0.9 * self.p[p].volume_ratio + 0.1 / volume
