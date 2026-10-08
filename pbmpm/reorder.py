import taichi as ti

from .state import Particle


@ti.func
def spread_bits(v):
    bits = ti.cast(v, ti.u32) & ti.u32(0x3ff)
    bits = (bits | (bits << 16)) & ti.u32(0x030000ff)
    bits = (bits | (bits << 8)) & ti.u32(0x0300f00f)
    bits = (bits | (bits << 4)) & ti.u32(0x030c30c3)
    return (bits | (bits << 2)) & ti.u32(0x09249249)


@ti.data_oriented
class MortonReorder:
    """Sort on the Taichi backend and move every particle field in one struct copy."""

    def __init__(self, particles, config):
        self.p, self.n, self.c = particles.data, particles.n, config
        self.keys = ti.field(ti.i32, shape=self.n)
        self.indices = ti.field(ti.i32, shape=self.n)
        self.scratch = Particle.field(shape=self.n, layout=ti.Layout.SOA)

    @ti.kernel
    def encode(self):
        for p in range(self.n):
            cell = ti.cast(ti.floor(self.p[p].x * self.c.grid), ti.i32)
            cell = ti.min(ti.max(cell, 0), self.c.grid - 1)
            key = spread_bits(cell[0]) | (spread_bits(cell[1]) << 1) | (spread_bits(cell[2]) << 2)
            self.keys[p] = ti.cast(key, ti.i32)
            self.indices[p] = p

    @ti.kernel
    def gather(self):
        for p in range(self.n):
            self.scratch[p] = self.p[self.indices[p]]

    @ti.kernel
    def commit(self):
        for p in range(self.n):
            self.p[p] = self.scratch[p]

    def apply(self):
        self.encode()
        ti.algorithms.parallel_sort(self.keys, self.indices)
        self.gather()
        self.commit()
