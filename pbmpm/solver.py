from collections import defaultdict
from time import perf_counter

import taichi as ti

from .materials import Materials
from .reorder import MortonReorder
from .rigid import RigidBodies
from .state import Particles
from .transfers import Transfers


@ti.data_oriented
class Solver:
    def __init__(self, config, scene):
        self.config = config
        self.initial = scene.particles
        self.particles = Particles(scene.particles)
        self.materials = Materials(self.particles, config)
        self.transfers = Transfers(self.particles, config)
        self.bodies = RigidBodies(self.particles, config, scene.bodies)
        self.reorder = MortonReorder(self.particles, config) if config.reorder != "none" else None
        self.invalid = ti.field(ti.i32, shape=())
        self.p = self.particles.data
        self.n = self.particles.n
        self.step_index = 0

    def reset(self):
        self.particles.restore(self.initial)
        self.bodies.restore()
        self.step_index = 0

    @ti.kernel
    def validate_stencil(self):
        self.invalid[None] = 0
        for p in range(self.n):
            for a in ti.static(range(3)):
                x = self.p[p].x[a]
                if ti.math.isnan(x) or ti.math.isinf(x) or x < 1.5 / self.config.grid or x >= 1 - 1.5 / self.config.grid:
                    self.invalid[None] = 1

    def step(self, profile=False, diagnostics=False):
        """Advance one physical dt; positions/F stay fixed during all inner iterations."""
        times = defaultdict(float)

        def run(name, fn):
            if profile:
                ti.sync()
                start = perf_counter()
                fn()
                ti.sync()
                times[name] += perf_counter() - start
            else:
                fn()

        # Fail before a stencil can address outside the allocated grid.
        self.validate_stencil()
        if self.invalid[None]:
            raise FloatingPointError("A particle left the valid grid stencil or became non-finite")
        if self.reorder is not None and (self.config.reorder == "full" or self.step_index % self.config.reorder_interval == 0):
            run("reorder", self.reorder.apply)
        run("predict", self.materials.predict)
        if self.bodies.count:
            run("rigid_predict", self.bodies.predict)
        for _ in range(self.config.iterations):
            if self.bodies.count:
                run("rigid_broadcast", self.bodies.broadcast)
            run("constraints", self.materials.solve)
            run("grid_clear", self.transfers.clear)
            run("p2g", self.transfers.p2g)
            run("grid_boundary", self.transfers.boundary)
            run("g2p", self.transfers.g2p)
            if self.bodies.count:
                run("rigid_project", self.bodies.project)
        if diagnostics:
            self.materials.measure_residuals()
        run("integrate", self.materials.integrate)
        if self.bodies.count:
            run("rigid_integrate", self.bodies.integrate)
        self.step_index += 1
        return dict(times)
