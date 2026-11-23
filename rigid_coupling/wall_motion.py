"""Paired reflected APIC transfers for free-slip prescribed box walls.

Each particle splits its motion weight over its real and mirrored samples.
G2P inverse-reflects and averages every sample with those same weights. For a
fixed wall, reflection is an isometry in the APIC norm; the paired mapping
does not introduce kinetic energy in a frozen transfer. This does not imply
unconditional stability of the complete density/contact solver.

Only motion averaging uses virtual samples. Density and contact gradients
continue to use the full real particle masses and rest volumes.
"""
import taichi as ti
from .transfers import Transfers, stencil, valid_position
from .state import FLUID, RIGID


@ti.data_oriented
class ReflectedTransfers(Transfers):
    def __init__(self, particles, c, invalid, box):
        super().__init__(particles, c, invalid)
        self.box = box
        self.physical_mass = ti.field(ti.f32, shape=(c.grid,)*3)
        self.wall_reaction = ti.Vector.field(3, ti.f32, shape=self.n)
        self.reaction_tiles = (self.n+127)//128
        self.tile_reaction = ti.Vector.field(3, ti.f32, shape=self.reaction_tiles)

    @ti.func
    def images(self, x):
        R, center = self.box.orientation[None], self.box.center[None]
        local = R.transpose()@(x-center)
        eligible = ti.Vector([1, 0, 0, 0, 0, 0, 0, 0])
        count = 1
        for mask in range(1, 8):
            own = True
            ghost = local
            for a in ti.static(range(3)):
                if mask & (1 << a):
                    support = 3*(ti.abs(R[0, a])+ti.abs(R[1, a])+ti.abs(R[2, a]))/self.c.grid
                    own = own and self.box.half_size[a]-ti.abs(local[a]) < support
                    side = 1. if local[a] >= 0 else -1.
                    ghost[a] = 2*side*self.box.half_size[a]-local[a]
            own = own and valid_position(center+R@ghost, self.c.grid)
            eligible[mask] = ti.cast(own, ti.i32)
            count += ti.cast(own, ti.i32)
        return eligible, count

    @ti.func
    def image(self, x, mask):
        gx, S = x, ti.Matrix.identity(ti.f32, 3)
        if mask != 0:
            R, center = self.box.orientation[None], self.box.center[None]
            ghost = R.transpose()@(x-center)
            reflected = R
            for a in ti.static(range(3)):
                if mask & (1 << a):
                    side = 1. if ghost[a] >= 0 else -1.
                    ghost[a] = 2*side*self.box.half_size[a]-ghost[a]
                    for b in ti.static(range(3)):
                        reflected[b, a] = -reflected[b, a]
            gx, S = center+R@ghost, reflected@R.transpose()
        return gx, S

    @ti.func
    def wall_gradient(self):
        omega = self.box.omega[None]*self.c.dt
        return ti.Matrix([[0., -omega[2], omega[1]], [omega[2], 0., -omega[0]],
                          [-omega[1], omega[0], 0.]])

    @ti.func
    def clear_node(self, i):
        Transfers.clear_node(self, i)
        self.physical_mass[i] = 0.

    @ti.func
    def boundary_node(self, i):
        # The domain-edge guard belongs to real particles. Applying it to
        # virtual samples breaks a wall-compatible affine field whenever a
        # reflected stencil reaches that guard. Their stencils are already
        # validated; exact container/world projection still bounds real water.
        if self.mass[i] > 0:
            d = self.displacement[i]/self.mass[i]
            self.displacement[i] = d
            if ti.static(self.c.world_contact):
                self.free_displacement[i] = d
        if self.rigid_mass[i] > 0:
            d = self.rigid_displacement[i]/self.rigid_mass[i]
            if ti.static(not self.c.world_contact):
                for axis in ti.static(range(3)):
                    if i[axis] <= self.c.boundary and d[axis] < 0:
                        d[axis] = 0
                    if i[axis] >= self.c.grid-self.c.boundary and d[axis] > 0:
                        d[axis] = 0
            self.rigid_displacement[i] = d
        # Virtual weights normalize motion, but contacts scatter real masses.
        self.mass[i] = self.physical_mass[i]

    @ti.func
    def scatter(self, p, x, d, D, fraction, real):
        base, f, w = stencil(x, self.c.grid)
        for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
            node = base+o
            mass = self.p[p].mass*w[o[0], 0]*w[o[1], 1]*w[o[2], 2]
            if self.p[p].material == RIGID:
                self.rigid_mass[node] += mass
                self.rigid_displacement[node] += mass*(d+D@((o-f)/self.c.grid))
            else:
                self.mass[node] += mass*fraction
                self.displacement[node] += mass*fraction*(d+D@((o-f)/self.c.grid))
                if real:
                    self.physical_mass[node] += mass

    @ti.kernel
    def p2g(self):
        for p in range(self.active[None]):
            x, d, D = self.p[p].x, self.p[p].d, self.p[p].D
            if valid_position(x, self.c.grid):
                if self.p[p].material == RIGID:
                    self.scatter(p, x, d, D, 1., True)
                else:
                    eligible, count = self.images(x)
                    W = self.wall_gradient()
                    wall_d = self.c.dt*self.box.velocity[None]+W@(x-self.box.center[None])
                    for mask in range(8):
                        if eligible[mask]:
                            gx, S = self.image(x, mask)
                            gd = self.c.dt*self.box.velocity[None]+W@(gx-self.box.center[None])+S@(d-wall_d)
                            self.scatter(p, gx, gd, W+S@(D-W)@S, 1./count, mask == 0)
            else:
                ti.atomic_max(self.invalid[None], 1)

    @ti.func
    def gather(self, x, density):
        base, f, w = stencil(x, self.c.grid)
        d = ti.Vector.zero(ti.f32, 3); D = ti.Matrix.zero(ti.f32, 3, 3)
        n, a = 0., 0.
        for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
            weight = w[o[0], 0]*w[o[1], 1]*w[o[2], 2]
            gd = self.displacement[base+o]
            d += weight*gd
            D += 4*self.c.grid**2*weight*gd.outer_product((o-f)/self.c.grid)
            if density:
                n += weight*self.liquid_volume[base+o]
                a += weight*self.available_volume[base+o]
        return d, D, n, a

    @ti.func
    def g2p_particle(self, p):
        self.wall_reaction[p] = ti.Vector.zero(ti.f32, 3)
        if self.p[p].material == RIGID:
            Transfers.g2p_particle(self, p)
        else:
            x = self.p[p].x
            old_d = self.p[p].d
            eligible, count = self.images(x)
            W = self.wall_gradient()
            wall_d = self.c.dt*self.box.velocity[None]+W@(x-self.box.center[None])
            d = ti.Vector.zero(ti.f32, 3); D = ti.Matrix.zero(ti.f32, 3, 3)
            n, a = 0., 0.
            for mask in range(8):
                if eligible[mask]:
                    gx, S = self.image(x, mask)
                    gd, gD, gn, ga = self.gather(gx, mask == 0)
                    d += (wall_d+S@(gd-self.c.dt*self.box.velocity[None]-W@(gx-self.box.center[None])))/count
                    D += (W+S@(gD-W)@S)/count
                    if mask == 0:
                        n, a = gn, ga
            self.p[p].d = d; self.p[p].D = D
            self.wall_reaction[p] = -self.p[p].mass*(d-old_d)/self.c.dt
            self.apply_density_feedback(p, n, a)

    @ti.func
    def sum_reaction_tile(self, tile):
        reaction = ti.Vector.zero(ti.f32, 3)
        for offset in range(128):
            p = tile*128+offset
            if p < self.active[None]:
                reaction += self.wall_reaction[p]
        self.tile_reaction[tile] = reaction

    @ti.func
    def finish_wall_reaction(self):
        reaction = ti.Vector.zero(ti.f32, 3)
        ti.loop_config(serialize=True)
        for tile in range(self.reaction_tiles):
            reaction += self.tile_reaction[tile]
        # One final writer avoids contended vector atomics. A separate
        # parallel `wall_impulse += reaction[p]` loop remains contended on
        # Taichi 1.7 CUDA; it does not automatically become a fast reduction.
        self.box.wall_impulse[None] = self.box.wall_impulse[None]+reaction

    @ti.kernel
    def g2p(self):
        for p in range(self.active[None]):
            if valid_position(self.p[p].x, self.c.grid):
                self.g2p_particle(p)
            else:
                self.wall_reaction[p] = ti.Vector.zero(ti.f32, 3)
                ti.atomic_max(self.invalid[None], 1)
        for tile in range(self.reaction_tiles):
            self.sum_reaction_tile(tile)
        self.finish_wall_reaction()

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
                self.wall_reaction[p] = ti.Vector.zero(ti.f32, 3)
                ti.atomic_max(self.invalid[None], 1)
        for tile in range(self.reaction_tiles):
            self.sum_reaction_tile(tile)
        self.finish_wall_reaction()
