"""Reconcile contact displacement increments through the quadratic MPM grid.

PB-MPM (Lewin 2024, Algorithm 1 and section 2) reconciles candidate deformation
displacements through grid transfers. These additional particle contacts need
the same affine information before volume/deformation integration. Fluid keeps
the exact projected translation; only its strain gradient is added, so contact
is not applied twice and smoothing cannot undo nonpenetration or add spin.
"""
import taichi as ti

from .state import ELASTIC, RIGID
from .transfers import stencil, valid_position


@ti.data_oriented
class ContactCorrection:
    def __init__(self, particles, config):
        self.p, self.n, self.c = particles.data, particles.n, config
        self.active = particles.count
        self.has_elastic = particles.has_elastic
        self.has_fluid = particles.has_fluid
        self.has_continuum = self.has_fluid or self.has_elastic
        self.mixed_materials = self.has_fluid and self.has_elastic
        self.can_reuse_mass = config.grid_coupling == "separate" and not self.mixed_materials
        channels = 2 if self.mixed_materials else 1
        self.channels = channels
        self.contact_work = ti.field(ti.i32, shape=())
        self.contact_delta = ti.Vector.field(3, ti.f32, shape=self.n)
        # Separate material channels keep soft friction out of water strain.
        self.delta_mass = ti.field(ti.f32, shape=(channels,)+(config.grid,)*3)
        self.delta_grid = ti.Vector.field(3, ti.f32, shape=(channels,)+(config.grid,)*3)
        ContactCorrection.reset(self)

    def reset(self):
        self.contact_work.fill(0)
        self.contact_delta.fill(0)
        self.delta_mass.fill(0)
        self.delta_grid.fill(0)

    @ti.kernel
    def scatter_contact_correction(self):
        for i in ti.grouped(ti.ndrange(self.channels*self.contact_work[None], self.c.grid, self.c.grid, self.c.grid)):
            self.clear_delta_node(i)
        for p in range(self.active[None]*self.contact_work[None]):
            self.scatter_delta_particle(p)

    @ti.func
    def clear_delta_node(self, i):
        self.delta_mass[i] = 0
        self.delta_grid[i] = ti.Vector.zero(ti.f32, 3)

    @ti.func
    def clear_delta_grid_node(self, i):
        self.delta_grid[i] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def scatter_contact_increment(self):
        # Single-material separated scenes already computed this exact mass
        # field in the main P2G. Scatter only nonzero contact increments.
        for i in ti.grouped(ti.ndrange(self.channels*self.contact_work[None], self.c.grid, self.c.grid, self.c.grid)):
            self.clear_delta_grid_node(i)
        for p in range(self.active[None]*self.contact_work[None]):
            self.scatter_increment_particle(p)

    @ti.func
    def scatter_increment_particle(self, p):
        if self.p[p].material != RIGID and self.contact_delta[p].norm_sqr() > 0:
            if valid_position(self.p[p].x, self.c.grid):
                base, f, w = stencil(self.p[p].x, self.c.grid)
                for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                    mass = self.p[p].mass * w[o[0], 0] * w[o[1], 1] * w[o[2], 2]
                    node = base+o
                    self.delta_grid[0, node[0], node[1], node[2]] += mass * self.contact_delta[p]

    @ti.func
    def channel(self, p):
        result = 0
        if ti.static(self.mixed_materials):
            result = ti.cast(self.p[p].material == ELASTIC, ti.i32)
        return result

    @ti.func
    def scatter_delta_particle(self, p):
        if self.p[p].material != RIGID and valid_position(self.p[p].x, self.c.grid):
            base, f, w = stencil(self.p[p].x, self.c.grid)
            channel = self.channel(p)
            for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                mass = self.p[p].mass * w[o[0], 0] * w[o[1], 1] * w[o[2], 2]
                node = base+o
                self.delta_mass[channel, node[0], node[1], node[2]] += mass
                self.delta_grid[channel, node[0], node[1], node[2]] += mass * self.contact_delta[p]

    @ti.kernel
    def gather_contact_correction(self):
        for p in range(self.active[None]*self.contact_work[None]):
            self.gather_delta_particle(p)

    @ti.kernel
    def gather_contact_with_mass(self, mass: ti.template()):
        for p in range(self.active[None]*self.contact_work[None]):
            self.gather_reusing_mass(p, mass)

    @ti.func
    def gather_delta_particle(self, p):
        self.gather_particle(p, self.delta_mass, False)

    @ti.func
    def gather_reusing_mass(self, p, mass: ti.template()):
        self.gather_particle(p, mass, True)

    @ti.func
    def gather_particle(self, p, mass_field: ti.template(), reuse: ti.template()):
        if self.p[p].material != RIGID and valid_position(self.p[p].x, self.c.grid):
            base, f, w = stencil(self.p[p].x, self.c.grid)
            channel = self.channel(p)
            correction = ti.Matrix.zero(ti.f32, 3, 3)
            displacement = ti.Vector.zero(ti.f32, 3)
            for o in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                node = base+o
                mass = 0.0
                if ti.static(reuse):
                    mass = mass_field[node]
                else:
                    mass = mass_field[channel, node[0], node[1], node[2]]
                increment = self.delta_grid[channel, node[0], node[1], node[2]]
                if mass > 0 and increment.norm_sqr() > 0:
                    weight = w[o[0], 0] * w[o[1], 1] * w[o[2], 2]
                    delta = increment / mass
                    displacement += weight * delta
                    correction += 4*self.c.grid**2*weight*delta.outer_product((o-f)/self.c.grid)
            if self.p[p].material == ELASTIC:
                self.p[p].d += displacement - self.contact_delta[p]
                self.p[p].D += correction
            else:
                # Contact already applied the exact translational impulse and
                # its opposite body/wall reaction. Retain it. The symmetric
                # gradient has the full trace needed by the liquid constraint,
                # without adding APIC angular momentum from a second impulse.
                self.p[p].D += 0.5*(correction+correction.transpose())

    def update_contact_affine(self, transfers=None):
        if self.has_continuum:
            if self.can_reuse_mass and transfers is not None:
                self.scatter_contact_increment()
                self.gather_contact_with_mass(transfers.mass)
            else:
                self.scatter_contact_correction()
                self.gather_contact_correction()
