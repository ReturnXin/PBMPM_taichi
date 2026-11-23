"""One physical PBMPM step with geometry frozen during rigid feedback iterations."""

from copy import deepcopy

import taichi as ti

from .materials import Materials
from .contact import Contacts
from .buoyancy import BuoyancyAssist
from .container import MovingContainer
from .box_contact import RigidBoxPairs
from .rigid import RigidBodies
from .state import FLUID, ELASTIC, Particles
from .transfers import Transfers
from .boundary_density import BoundaryDensity
from .solid_boundary_density import SolidBoundaryDensity
from .available_boundary_density import AvailableBoundaryDensity


@ti.data_oriented
class Solver:
    def __init__(self, config, scene, *, mouse_control=False):
        self.config = config
        self.initial = deepcopy(scene.particles)
        self.invalid = ti.field(ti.i32, shape=())
        self.emitter = deepcopy(scene.emitter)
        self.emission_step = 0
        self.particles = Particles(self.initial, self.emitter.get("initial_count"),
                                   xpbmpm=config.elastic_solver == "xpbmpm")
        self.active = self.particles.count
        self.has_elastic = self.particles.has_elastic
        if config.boundary_density != "raw" and (self.has_elastic or not scene.container):
            raise ValueError(f"{config.boundary_density} boundary density requires a water/rigid container scene")
        if config.transfer_mode == "incremental" and self.has_elastic:
            raise ValueError("incremental transfer currently supports water/rigid scenes only")
        self.bodies = RigidBodies(self.particles, config, scene.bodies)
        self.contacts = Contacts(self.particles, self.bodies, config)
        self.container = MovingContainer(self.particles, config, scene.container, self.bodies) if scene.container else None
        if config.wall_transfer == "reflected":
            if self.container is None or self.has_elastic:
                raise ValueError("reflected wall transfers require a water/rigid container scene")
            if min(self.container.half_size) < 3 * 3**.5 / config.grid:
                raise ValueError("reflected wall transfers require box half-sizes >= 3*sqrt(3)/grid")
            from .wall_motion import ReflectedTransfers
            self.transfers = ReflectedTransfers(self.particles, config, self.invalid, self.container)
        else:
            self.transfers = Transfers(self.particles, config, self.invalid)
        self.mouse = None
        if mouse_control:
            if self.container is None or not (self.initial["material"] == FLUID).any():
                raise ValueError("mouse control requires water in a container")
            from .mouse_water import WaterMouse
            self.mouse = WaterMouse(self.particles, config, self.container)
        self.materials = Materials(self.particles, config, self.mouse)
        density_class = {"mirror": BoundaryDensity, "solid": SolidBoundaryDensity}.get(config.boundary_density)
        self.boundary_density = (AvailableBoundaryDensity(self.particles, self.transfers, self.container, config, self.bodies)
                                 if config.boundary_density == "available" else
                                 density_class(self.particles, self.transfers, self.container, config)
                                 if density_class is not None else None)
        # Fluid d stays exactly projected, and both corrections have the same
        # frozen P2G mass: their strain gradients add linearly. Reconcile once.
        self.combine_fluid_contacts = (self.container is not None and not self.has_elastic
                                       and config.grid_coupling == "separate"
                                       and self.bodies.count > 0 and config.rigid_contact)
        self.box_pairs = RigidBoxPairs(self.bodies) if self.container is not None and self.bodies.count > 1 else None
        self.buoyancy = BuoyancyAssist(self.particles, self.bodies, config, self.container)
        self.sampling = deepcopy(scene.sampling)
        self.p, self.n = self.particles.data, self.particles.n
        self.step_index = 0
        self.failed = False
        self.step_graph = None
        self.validate_state()

    def enable_cuda_graph(self):
        if ti.lang.impl.current_cfg().arch != ti.cuda:
            raise ValueError("native step graph requires the CUDA backend")
        if self.emitter:
            raise ValueError("native step graph does not support particle emitters")
        from .execution import CudaStepGraph
        self.step_graph = CudaStepGraph(self)

    def reset(self):
        self.particles.restore(self.initial)
        self.bodies.restore()
        self.buoyancy.reset()
        self.contacts.reset()
        if self.mouse is not None:
            self.mouse.reset()
        if self.container is not None:
            self.container.reset()
        self.transfers.clear()
        self.step_index = 0
        self.emission_step = 0
        self.failed = False
        self.validate_state()

    @ti.kernel
    def check_state(self):
        # Sticky across a batch: no invalid intermediate state can be hidden.
        for p in range(self.active[None]):
            for a in ti.static(range(3)):
                x = self.p[p].x[a]
                d = self.p[p].d[a]
                if ti.math.isnan(x) or ti.math.isinf(x) or x < 0.5 / self.config.grid or x >= 1 - 1.5 / self.config.grid:
                    ti.atomic_max(self.invalid[None], 1)
                if ti.math.isnan(d) or ti.math.isinf(d):
                    ti.atomic_max(self.invalid[None], 1)
                for b in ti.static(range(3)):
                    D = self.p[p].D[a, b]
                    if ti.math.isnan(D) or ti.math.isinf(D):
                        ti.atomic_max(self.invalid[None], 1)
                    if ti.static(self.has_elastic):
                        if self.p[p].material == ELASTIC:
                            F = self.p[p].F[a, b]
                            if ti.math.isnan(F) or ti.math.isinf(F):
                                ti.atomic_max(self.invalid[None], 1)
                            if ti.static(self.config.elastic_solver == "xpbmpm"):
                                multiplier = self.particles.multiplier[p][a, b]
                                if ti.math.isnan(multiplier) or ti.math.isinf(multiplier):
                                    ti.atomic_max(self.invalid[None], 1)
            volume = self.p[p].volume_ratio
            if ti.math.isnan(volume) or ti.math.isinf(volume) or volume <= 0:
                ti.atomic_max(self.invalid[None], 1)

    def validate_stencil(self):
        self.invalid.fill(0)
        self.check_state()

    def validate_state(self):
        # Fail before an out-of-domain particle can index outside the grid.
        self.validate_stencil()
        if self.invalid[None]:
            raise FloatingPointError("A particle left the valid grid stencil or has a non-finite/invalid state")

    @ti.kernel
    def prepare_iteration(self):
        # Three independent passes share one launch: per-particle constraint
        # preparation, grid reset, and body-feedback reset. p2g (next kernel)
        # depends on all three completing, so their relative order is free.
        for p in range(self.active[None]):
            b = self.p[p].body_id
            if b >= 0:
                self.bodies.broadcast_particle(p, b)
            elif self.p[p].material == FLUID:
                self.materials.solve_particle(p)
            if ti.static(self.has_elastic):
                if self.p[p].material == ELASTIC:
                    self.materials.solve_elastic_particle(p)
        for i in ti.grouped(self.transfers.mass):
            self.transfers.clear_node(i)
        for b in range(self.bodies.count):
            self.bodies.data[b].delta_momentum = ti.Vector.zero(ti.f32, 3)
            self.bodies.data[b].delta_angular = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def prepare_increment(self):
        for p in range(self.active[None]):
            b = self.p[p].body_id
            if b >= 0:
                self.bodies.broadcast_particle(p, b)
            elif self.p[p].material == FLUID:
                self.materials.solve_particle(p)
        for i in ti.grouped(self.transfers.mass):
            self.transfers.clear_increment_node(i)
        for b in range(self.bodies.count):
            self.bodies.data[b].delta_momentum = ti.Vector.zero(ti.f32, 3)
            self.bodies.data[b].delta_angular = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def clear_iteration(self):
        for i in ti.grouped(self.transfers.mass):
            self.transfers.clear_node(i)
        for b in range(self.bodies.count):
            self.bodies.data[b].delta_momentum = ti.Vector.zero(ti.f32, 3)
            self.bodies.data[b].delta_angular = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def release_to(self, count: ti.i32):
        self.active[None] = count

    @ti.kernel
    def project_interfaces(self):
        # Top-level parallel passes retain device barriers and avoid repeated
        # Python dispatch of each individual contact/correction kernel.
        # First apply the grid (G2P) feedback to the bodies so the contact and
        # container passes below see the updated body velocity. This is the
        # former standalone apply_feedback() kernel, folded in to save a launch.
        if ti.static(self.bodies.count > 0 and self.config.coupling == "two_way"):
            for b in range(self.bodies.count):
                self.bodies.apply_body_feedback(b, False)
        if ti.static(self.bodies.count > 0 and self.config.rigid_contact):
            self.contacts.contact_work[None] = 0
            for p in range(self.active[None]):
                self.contacts.project_particle(p)
            if ti.static(self.config.coupling == "two_way"):
                for b in range(self.bodies.count):
                    self.bodies.apply_body_feedback(b, True)
            if ti.static(self.contacts.has_continuum and not self.combine_fluid_contacts):
                if ti.static(self.contacts.can_reuse_mass):
                    for i in ti.grouped(ti.ndrange(self.contacts.channels*self.contacts.contact_work[None], self.config.grid, self.config.grid, self.config.grid)):
                        self.contacts.clear_delta_grid_node(i)
                    for p in range(self.active[None]*self.contacts.contact_work[None]):
                        self.contacts.scatter_increment_particle(p)
                    for p in range(self.active[None]*self.contacts.contact_work[None]):
                        self.contacts.gather_reusing_mass(p, self.transfers.mass)
                else:
                    for i in ti.grouped(ti.ndrange(self.contacts.channels*self.contacts.contact_work[None], self.config.grid, self.config.grid, self.config.grid)):
                        self.contacts.clear_delta_node(i)
                    for p in range(self.active[None]*self.contacts.contact_work[None]):
                        self.contacts.scatter_delta_particle(p)
                    for p in range(self.active[None]*self.contacts.contact_work[None]):
                        self.contacts.gather_delta_particle(p)
        for p in range(self.active[None]):
            if ti.static(self.combine_fluid_contacts):
                self.container.project_particle(p, self.contacts.contact_delta, True)
            else:
                self.container.project_particle(p)
        for tile in range(self.container.metric_tiles):
            self.container.accumulate_tile_metrics(tile)
        if ti.static(self.combine_fluid_contacts):
            self.container.finish_particle_metrics(self.contacts.contact_work, True)
        else:
            self.container.finish_particle_metrics()
        if ti.static(self.container.has_continuum):
            if ti.static(self.container.can_reuse_mass):
                for i in ti.grouped(ti.ndrange(self.container.channels*self.container.contact_work[None], self.config.grid, self.config.grid, self.config.grid)):
                    self.container.clear_delta_grid_node(i)
                for p in range(self.active[None]*self.container.contact_work[None]):
                    self.container.scatter_increment_particle(p)
                for p in range(self.active[None]*self.container.contact_work[None]):
                    self.container.gather_reusing_mass(p, self.transfers.mass)
            else:
                for i in ti.grouped(ti.ndrange(self.container.channels*self.container.contact_work[None], self.config.grid, self.config.grid, self.config.grid)):
                    self.container.clear_delta_node(i)
                for p in range(self.active[None]*self.container.contact_work[None]):
                    self.container.scatter_delta_particle(p)
                for p in range(self.active[None]*self.container.contact_work[None]):
                    self.container.gather_delta_particle(p)

    @ti.kernel
    def grid_update(self):
        # Fused normalize/box-boundary and moving-wall pass: two sequential
        # grid loops in one launch, replacing transfers.boundary() +
        # container.grid_boundary(). Only used when a container is present.
        for i in ti.grouped(self.transfers.mass):
            self.transfers.boundary_node(i)
        if ti.static(self.config.wall_transfer != "reflected"):
            for i in ti.grouped(self.transfers.mass):
                self.container.grid_boundary_node(i, self.transfers)

    @ti.kernel
    def grid_update_increment(self):
        for i in ti.grouped(self.transfers.mass):
            self.transfers.incremental_boundary_node(i)
        for i in ti.grouped(self.transfers.mass):
            self.container.grid_boundary_node(i, self.transfers)

    def _substep(self):
        if self.container is not None:
            self.container.begin_step()
        if self.emitter:
            self.emission_step += 1
            if self.emitted_count(self.emission_step - 1) < self.emitter["capacity"]:
                self.release_to(self.emitter["initial_count"] + self.emitted_count(self.emission_step))
        if self.config.buoyancy_assist and self.config.coupling == "two_way":
            self.buoyancy.begin()
        self.materials.predict()
        if self.bodies.count:
            self.bodies.predict()
        if self.boundary_density is not None:
            self.boundary_density.build()
        for iteration in range(self.config.iterations):
            incremental = self.config.transfer_mode == "incremental" and iteration > 0
            if incremental:
                self.prepare_increment()
                self.transfers.p2g_increment()
                if self.container is not None:
                    self.grid_update_increment()
                else:
                    self.transfers.incremental_boundary()
            else:
                self.prepare_iteration()
                self.transfers.p2g()
                if self.container is not None and not self.has_elastic:
                    self.grid_update()
                else:
                    self.transfers.boundary()
            self.transfers.g2p_with_feedback(self.bodies)
            if self.container is not None:
                # project_interfaces() applies the grid feedback to the bodies
                # as its first pass, so no separate apply_feedback() launch.
                self.project_interfaces()
            else:
                if self.bodies.count and self.config.coupling == "two_way":
                    # The complete particle reduction precedes this body update.
                    self.bodies.apply_feedback()
                if self.bodies.count and self.config.rigid_contact:
                    self.contacts.project()
                    if self.config.coupling == "two_way":
                        self.bodies.apply_contact_feedback()
                    self.contacts.update_contact_affine(self.transfers)
        if self.config.buoyancy_assist and self.config.coupling == "two_way":
            self.buoyancy.apply()
        self.materials.integrate()
        if self.bodies.count:
            self.bodies.integrate()
            if self.container is not None:
                for _ in range(2):
                    if self.box_pairs is not None:
                        self.box_pairs.project()
                    self.container.resolve_bodies()
                self.bodies.reconstruct()
        self.check_state()

    def emitted_count(self, step=None):
        """Integer release schedule depends only on physical time, never FPS."""
        if not self.emitter:
            return 0
        step = self.step_index if step is None else step
        duration_steps = max(1, round(self.emitter["duration"] / self.config.dt))
        return min(self.emitter["capacity"], step * self.emitter["capacity"] // duration_steps)

    def advance(self, steps):
        """Queue physical steps with one host error check for the whole batch.

        Transfers guard every grid address on-device. On a failed batch no time
        is committed; reset is required before using the partial state again.
        """
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if self.failed:
            raise FloatingPointError("Reset is required after a failed simulation batch")
        if steps == 0:
            return
        self.invalid.fill(0)
        self.check_state()
        for _ in range(steps):
            if self.step_graph is not None:
                self.step_graph.run()
            else:
                self._substep()
        if self.invalid[None]:
            self.failed = True
            raise FloatingPointError("A particle left the valid grid stencil or has a non-finite/invalid state; reset required")
        self.step_index += steps

    def step(self):
        """Advance dt exactly once, regardless of the number of inner iterations."""
        self.advance(1)

