"""Quadratic B-spline transfers with shared or separate fluid/rigid motion."""

import taichi as ti

from .state import FLUID, RIGID


@ti.func
def stencil(x, grid):
    scaled = x * grid
    base = ti.cast(ti.floor(scaled - 0.5), ti.i32)
    f = scaled - base
    weights = ti.Matrix.rows([0.5 * (1.5 - f)**2,
                              0.75 - (f - 1.0)**2, 0.5 * (f - 0.5)**2])
    return base, f, weights


@ti.func
def valid_position(x, grid):
    valid = True
    for a in ti.static(range(3)):
        # Ordered comparisons also reject NaN/Inf, before floor/int conversion.
        # base=floor(x*grid-.5), offsets=0..2: require 0 <= base <= grid-3.
        valid = valid and x[a] >= 0.5 / grid and x[a] < 1.0 - 1.5 / grid
    return valid


@ti.data_oriented
class Transfers:
    def __init__(self, particles, config, invalid=None):
        self.p, self.n, self.c = particles.data, particles.n, config
        self.active = particles.count
        self.invalid = ti.field(ti.i32, shape=()) if invalid is None else invalid
        shape = (config.grid,) * 3
        self.mass = ti.field(ti.f32, shape=shape)
        self.displacement = ti.Vector.field(3, ti.f32, shape=shape)
        self.free_displacement = ti.Vector.field(3, ti.f32, shape=shape)
        self.liquid_volume = ti.field(ti.f32, shape=shape)
        if config.boundary_density == "available":
            self.available_volume = ti.field(ti.f32, shape=shape)
        if config.transfer_mode == "incremental":
            self.increment = ti.Vector.field(3, ti.f32, shape=shape)
            self.transported_d = ti.Vector.field(3, ti.f32, shape=self.n)
            self.transported_D = ti.Matrix.field(3, 3, ti.f32, shape=self.n)
        if config.grid_coupling == "separate":
            # Occupancy remains shared; displacement averaging does not mix
            # fluid tangents with rigid motion. Surface contact supplies the
            # normal impulse and its equal/opposite rigid reaction.
            self.rigid_mass = ti.field(ti.f32, shape=shape)
            self.rigid_displacement = ti.Vector.field(3, ti.f32, shape=shape)

    @ti.func
    def clear_node(self, i):
        self.mass[i] = 0
        self.displacement[i] = ti.Vector.zero(ti.f32, 3)
        if ti.static(self.c.boundary_density == "raw"):
            self.liquid_volume[i] = 0
        if ti.static(self.c.transfer_mode == "incremental"):
            self.increment[i] = ti.Vector.zero(ti.f32, 3)
        if ti.static(self.c.grid_coupling == "separate"):
            self.rigid_mass[i] = 0
            self.rigid_displacement[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def clear(self):
        for i in ti.grouped(self.mass):
            self.clear_node(i)
            self.liquid_volume[i] = 0
        if ti.static(self.c.transfer_mode == "incremental"):
            for p in range(self.n):
                self.transported_d[p] = ti.Vector.zero(ti.f32, 3)
                self.transported_D[p] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.func
    def clear_increment_node(self, i):
        # Geometry is frozen within a physical step: retain liquid mass,
        # occupancy, and the normalized candidate grid displacement.
        self.increment[i] = ti.Vector.zero(ti.f32, 3)
        self.rigid_mass[i] = 0
        self.rigid_displacement[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def p2g(self):
        for p in range(self.active[None]):
            if valid_position(self.p[p].x, self.c.grid):
                base, f, w = stencil(self.p[p].x, self.c.grid)
                for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                    weight = w[o[0], 0] * w[o[1], 1] * w[o[2], 2]
                    arm = (o - f) / self.c.grid
                    mass = weight * self.p[p].mass
                    weighted_d = mass * (self.p[p].d + self.p[p].D @ arm)
                    if ti.static(self.c.grid_coupling == "separate"):
                        if self.p[p].material != RIGID:
                            self.mass[base + o] += mass
                            self.displacement[base + o] += weighted_d
                        else:
                            self.rigid_mass[base + o] += mass
                            self.rigid_displacement[base + o] += weighted_d
                    else:
                        self.mass[base + o] += mass
                        self.displacement[base + o] += weighted_d
                    if ti.static(self.c.boundary_density == "raw"):
                        if self.p[p].material == FLUID or (ti.static(self.c.rigid_volume) and self.p[p].material == RIGID):
                            self.liquid_volume[base + o] += weight * self.p[p].rest_volume * self.c.grid**3
            else:
                ti.atomic_max(self.invalid[None], 1)

    @ti.kernel
    def p2g_increment(self):
        for p in range(self.active[None]):
            if valid_position(self.p[p].x, self.c.grid):
                base, f, w = stencil(self.p[p].x, self.c.grid)
                d, D = self.p[p].d, self.p[p].D
                if self.p[p].material != RIGID:
                    # Include material AND post-G2P contact changes. Updating
                    # this reference after contact would lose its normal impulse.
                    d -= self.transported_d[p]
                    D -= self.transported_D[p]
                for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                    weight = w[o[0], 0] * w[o[1], 1] * w[o[2], 2]
                    arm = (o-f)/self.c.grid
                    mass = weight*self.p[p].mass
                    weighted_d = mass*(d+D@arm)
                    if self.p[p].material == RIGID:
                        # Rigid motion is still re-broadcast and reconciled in
                        # its own grid; it cannot average away fluid tangents.
                        self.rigid_mass[base+o] += mass
                        self.rigid_displacement[base+o] += weighted_d
                    else:
                        self.increment[base+o] += weighted_d
            else:
                ti.atomic_max(self.invalid[None], 1)

    @ti.kernel
    def boundary(self):
        # Normalize the accumulated m_i*d_i before applying the box boundary.
        for i in ti.grouped(self.mass):
            self.boundary_node(i)

    @ti.kernel
    def incremental_boundary(self):
        for i in ti.grouped(self.mass):
            self.incremental_boundary_node(i)

    @ti.func
    def incremental_boundary_node(self, i):
        if self.mass[i] > 0:
            d = self.displacement[i]+self.increment[i]/self.mass[i]
            if ti.static(self.c.world_contact):
                self.free_displacement[i] = d
            for a in ti.static(range(3)):
                if i[a] <= self.c.boundary and d[a] < 0:
                    d[a] = 0
                if i[a] >= self.c.grid-self.c.boundary and d[a] > 0:
                    d[a] = 0
            self.displacement[i] = d
        if self.rigid_mass[i] > 0:
            rd = self.rigid_displacement[i]/self.rigid_mass[i]
            if ti.static(not self.c.world_contact):
                for a in ti.static(range(3)):
                    if i[a] <= self.c.boundary and rd[a] < 0:
                        rd[a] = 0
                    if i[a] >= self.c.grid-self.c.boundary and rd[a] > 0:
                        rd[a] = 0
            self.rigid_displacement[i] = rd

    @ti.func
    def boundary_node(self, i):
        # Per-node body of boundary(); exposed as a @ti.func so the solver can
        # fuse it with the moving container's wall pass into a single kernel.
        if self.mass[i] > 0:
            d = self.displacement[i] / self.mass[i]
            if ti.static(self.c.world_contact):
                self.free_displacement[i] = d
            for a in ti.static(range(3)):
                if i[a] <= self.c.boundary and d[a] < 0:
                    d[a] = 0
                if i[a] >= self.c.grid - self.c.boundary and d[a] > 0:
                    d[a] = 0
            self.displacement[i] = d
        if ti.static(self.c.grid_coupling == "separate"):
            if self.rigid_mass[i] > 0:
                rd = self.rigid_displacement[i] / self.rigid_mass[i]
                if ti.static(not self.c.world_contact):
                    for a in ti.static(range(3)):
                        if i[a] <= self.c.boundary and rd[a] < 0:
                            rd[a] = 0
                        if i[a] >= self.c.grid - self.c.boundary and rd[a] > 0:
                            rd[a] = 0
                self.rigid_displacement[i] = rd

    @ti.kernel
    def g2p(self):
        for p in range(self.active[None]):
            if valid_position(self.p[p].x, self.c.grid):
                self.g2p_particle(p)
            else:
                ti.atomic_max(self.invalid[None], 1)

    @ti.kernel
    def g2p_with_feedback(self, bodies: ti.template()):
        for p in range(self.active[None]):
            if valid_position(self.p[p].x, self.c.grid):
                self.g2p_particle(p)
                if ti.static(self.c.coupling == "two_way"):
                    b = self.p[p].body_id
                    if b >= 0:
                        bodies.collect_particle_feedback(p, b)
            else:
                ti.atomic_max(self.invalid[None], 1)

    @ti.func
    def g2p_particle(self, p):
        base, f, w = stencil(self.p[p].x, self.c.grid)
        d = ti.Vector.zero(ti.f32, 3)
        D = ti.Matrix.zero(ti.f32, 3, 3)
        volume = 0.0
        available = 0.0
        for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
            weight = w[o[0], 0] * w[o[1], 1] * w[o[2], 2]
            arm = (o - f) / self.c.grid
            gd = self.displacement[base + o]
            if ti.static(self.c.grid_coupling == "separate"):
                if self.p[p].body_id >= 0:
                    gd = self.rigid_displacement[base + o]
            elif ti.static(self.c.world_contact):
                # Rigid walls use the same geometric contact in both modes.
                # Exclude direct grid-wall impulses from rigid G2P feedback;
                # fluid still receives the grid boundary treatment.
                if self.p[p].body_id >= 0:
                    gd = self.free_displacement[base + o]
            d += weight * gd
            # For this stencil, sum(w*arm*arm^T) = I/(4*grid^2).
            D += 4 * self.c.grid**2 * weight * gd.outer_product(arm)
            if self.p[p].material == FLUID:
                volume += weight * self.liquid_volume[base + o]
                if ti.static(self.c.boundary_density == "available"):
                    available += weight * self.available_volume[base + o]
        self.p[p].d = d
        self.p[p].D = D
        if ti.static(self.c.transfer_mode == "incremental"):
            if self.p[p].material != RIGID:
                self.transported_d[p] = d
                self.transported_D[p] = D
        self.apply_density_feedback(p, volume, available)

    @ti.func
    def apply_density_feedback(self, p, volume, available):
        if ti.static(self.c.boundary_density == "available"):
            # Normalize after both gathers. A vanishing denominator carries
            # no resolvable water volume: omit its density feedback rather
            # than turning an epsilon denominator into a large fake pressure.
            if self.p[p].material == FLUID and available > 1e-4:
                if ti.static(self.c.density_feedback == "clamped"):
                    # Air/underfill returns the target to J=1, never above it.
                    # This relaxes stale compression without treating missing
                    # free-surface support as a request for tensile pressure.
                    target = available / ti.max(volume, available)
                    self.p[p].volume_ratio += .1 * available * (target - self.p[p].volume_ratio)
                elif volume > available:
                    # Keep the normalized target, but attenuate feedback where
                    # solids leave little volume and amplify sampling error.
                    blend = 0.1 * available
                    self.p[p].volume_ratio += blend * (available / volume - self.p[p].volume_ratio)
        else:
            if self.p[p].material == FLUID and volume > 1.0:
                self.p[p].volume_ratio = 0.9 * self.p[p].volume_ratio + 0.1 / volume

