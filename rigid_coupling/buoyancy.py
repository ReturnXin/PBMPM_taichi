"""Optional upward support for submerged light boxes touching the floor.

The mean-water model only helps a box leave the bottom when resolved contact
cannot wet its underside. Free-floating bodies retain the coupled response.
This one-sided support is a demonstration approximation, not fluid pressure.
"""
import taichi as ti
from .state import FLUID
from .rigid import rotation


@ti.data_oriented
class BuoyancyAssist:
    def __init__(self, particles, bodies, config, container=None):
        self.p, self.active = particles.data, particles.count
        self.b, self.count, self.c = bodies.data, bodies.count, config
        self.container = container
        self.has_container = container is not None
        self.before_vy = ti.field(ti.f32, shape=max(1, self.count))
        self.fraction = ti.field(ti.f32, shape=max(1, self.count))
        self.correction = ti.field(ti.f32, shape=max(1, self.count))
        self.pool_volume = ti.field(ti.f32, shape=())
        self.level = ti.field(ti.f32, shape=())
        self.reset()

    def reset(self):
        self.before_vy.fill(0)
        self.fraction.fill(0)
        self.correction.fill(0)
        self.pool_volume.fill(0)
        self.level.fill(self.c.boundary / self.c.grid)

    @ti.kernel
    def begin(self):
        for b in range(self.count):
            self.before_vy[b] = self.b[b].v[1]
        self.pool_volume[None] = 0
        # Exclude the upper falling jet from the estimated pooled water.
        lo = self.c.boundary / self.c.grid
        for p in range(self.active[None]):
            if self.p[p].material == FLUID:
                if ti.static(self.has_container):
                    self.pool_volume[None] += self.p[p].rest_volume
                elif self.p[p].x[1] < lo + .5 * (1-2*lo):
                    self.pool_volume[None] += self.p[p].rest_volume

    @ti.func
    def submerged_fraction(self, b, level):
        R = rotation(self.b[b].q)
        half = self.b[b].half_size
        fraction = 0.0
        # Smooth 4^3 volume quadrature: all orientations use the same model.
        band = ti.max(half.min() / 4, 1e-5)
        for j in range(64):
            local = ti.Vector([(2*((j%4)+.5)/4-1)*half[0],
                               (2*(((j//4)%4)+.5)/4-1)*half[1],
                               (2*((j//16)+.5)/4-1)*half[2]])
            y = self.b[b].x[1] + (R @ local)[1]
            fraction += ti.min(1.0, ti.max(0.0, .5 + (level-y)/(2*band))) / 64
        return fraction

    @ti.func
    def near_floor(self, b):
        R = rotation(self.b[b].q)
        extent = ti.abs(R) @ self.b[b].half_size
        gap = self.b[b].x[1] - extent[1] - self.c.boundary/self.c.grid
        if ti.static(self.has_container):
            box_R = self.container.orientation[None]
            local = box_R.transpose() @ (self.b[b].x-self.container.center[None])
            extent = ti.abs(box_R.transpose() @ R) @ self.b[b].half_size
            gap = local[1] - extent[1] + self.container.half_size[1]
        # A grid-cell bottom band gives resolved water room to enter beneath
        # the box before support stops; a roundoff-only gap caused recontact.
        return ti.static(self.has_container or self.c.world_contact) and gap <= 1./self.c.grid

    @ti.kernel
    def apply(self):
        lo = self.c.boundary / self.c.grid
        area = (1-2*lo)**2
        if ti.static(self.has_container):
            # Mean hydrostatic level of the prescribed box; waves are not a pressure solve.
            cosine = ti.max(self.container.orientation[None][1, 1], .5)
            lo = self.container.center[None][1]-self.container.half_size[1]/cosine
            area = 4*self.container.half_size[0]*self.container.half_size[2]/cosine
        level = lo + self.pool_volume[None] / area
        # Account for the volume displaced by boxes in a constant-area tank.
        ti.loop_config(serialize=True)
        for _ in range(4):
            displacement = 0.0
            if self.pool_volume[None] > 0:
                for b in range(self.count):
                    half = self.b[b].half_size
                    displacement += 8*half[0]*half[1]*half[2]*self.submerged_fraction(b, level)
            level = lo + (self.pool_volume[None] + displacement) / area
        self.level[None] = level
        for b in range(self.count):
            wet = 0.0
            if self.pool_volume[None] > 0:
                wet = self.submerged_fraction(b, level)
            self.fraction[b] = wet
            half = self.b[b].half_size
            volume = 8*half[0]*half[1]*half[2]
            self.correction[b] = 0.
            if (self.b[b].fixed == 0 and self.c.gravity < 0 and wet > 0
                    and self.b[b].mass < (1-1e-6)*self.c.fluid_density*volume and self.near_floor(b)):
                acceleration = self.c.gravity + (-self.c.gravity)*self.c.fluid_density*wet*volume/self.b[b].mass
                target = self.before_vy[b] + self.c.dt*acceleration
                # Only supply missing upward support. Existing contact may
                # already provide it; never cancel that response or add it twice.
                delta = ti.max(0., target-self.b[b].v[1])
                self.b[b].v[1] += delta
                self.correction[b] = delta
