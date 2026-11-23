"""Shared PBMPM water and PBMPM/XPBMPM elastic material updates."""

import taichi as ti

from .state import FLUID, ELASTIC, RIGID


@ti.func
def elastic_target(F, D, volume_weight):
    # Same rotation/volume target as pbmpm.materials, kept locally so the
    # standalone coupling path does not import the XPBMPM experiment solver.
    I = ti.Matrix.identity(ti.f32, 3)
    trial = (I + D) @ F
    U, singular, V = ti.svd(trial)
    shape = U @ V.transpose()
    if shape.determinant() < 0:
        for axis in ti.static(range(3)):
            U[axis, 2] = -U[axis, 2]
        shape = U @ V.transpose()
    volume = trial / ti.pow(ti.max(trial.determinant(), 1e-6), 1.0 / 3.0)
    target = (1.0 - volume_weight) * shape + volume_weight * volume
    U0, s0, V0 = ti.svd(F)
    inverse_s = ti.Matrix.zero(ti.f32, 3, 3)
    for axis in ti.static(range(3)):
        inverse_s[axis, axis] = 1.0 / ti.max(s0[axis, axis], 0.1)
    return target @ (V0 @ inverse_s @ U0.transpose()) - I


@ti.func
def legacy_elastic_target(F, D):
    # Match the chapter-three rotation target and regularized inverse.
    I = ti.Matrix.identity(ti.f32, 3)
    U, singular, V = ti.svd((I+D)@F)
    U0, s0, V0 = ti.svd(F)
    inverse_s = ti.Matrix.zero(ti.f32, 3, 3)
    for axis in ti.static(range(3)):
        inverse_s[axis, axis] = 1./ti.max(s0[axis, axis], .1)
    return (U@V.transpose())@(V0@inverse_s@U0.transpose())-I


@ti.data_oriented
class Materials:
    def __init__(self, particles, config, mouse=None):
        self.p, self.n, self.c = particles.data, particles.n, config
        self.active = particles.count
        self.has_elastic = particles.has_elastic
        self.mouse = mouse
        self.mouse_enabled = mouse is not None
        self.multiplier = particles.multiplier
        self.xpbmpm = self.has_elastic and self.c.elastic_solver == "xpbmpm"

    @ti.func
    def target(self, F, D):
        result = ti.Matrix.zero(ti.f32, 3, 3)
        if ti.static(self.c.elastic_model == "legacy"):
            result = legacy_elastic_target(F, D)
        else:
            result = elastic_target(F, D, self.c.elastic_volume_weight)
        return result

    @ti.kernel
    def predict(self):
        if ti.static(self.mouse_enabled):
            self.mouse.prepare_step()
        for p in range(self.active[None]):
            if ti.static(self.xpbmpm):
                if self.p[p].material == ELASTIC:
                    if ti.static(self.c.elastic_model == "legacy"):
                        self.multiplier[p] *= .5
                    else:
                        self.multiplier[p] = ti.Matrix.zero(ti.f32, 3, 3)
            if self.p[p].material != RIGID:
                if ti.static(self.c.elastic_model == "legacy"):
                    if self.p[p].material != ELASTIC:
                        self.p[p].d[1] += self.c.gravity * self.c.dt**2
                else:
                    self.p[p].d[1] += self.c.gravity * self.c.dt**2
                if ti.static(self.mouse_enabled):
                    self.mouse.apply_particle(p)

    @ti.kernel
    def solve(self):
        for p in range(self.active[None]):
            if self.p[p].material == FLUID:
                self.solve_particle(p)
            if ti.static(self.has_elastic):
                if self.p[p].material == ELASTIC:
                    self.solve_elastic_particle(p)

    @ti.func
    def solve_elastic_particle(self, p):
        D = self.p[p].D
        target = self.target(self.p[p].F, D)
        if ti.static(self.xpbmpm):
            alpha = self.c.compliance/self.c.dt**2
            delta = (target-D-alpha*self.multiplier[p])/(self.c.inverse_metric+alpha)
            self.p[p].D = D+self.c.inverse_metric*delta
            self.multiplier[p] += delta
        else:
            self.p[p].D = D + self.c.elastic_relaxation * (target - D)

    @ti.func
    def solve_particle(self, p):
        D = self.p[p].D
        D -= 0.5 * self.c.viscosity * (D + D.transpose())
        residual = 1.0 / ti.max(self.p[p].volume_ratio, 0.1) - 1.0 - D.trace()
        self.p[p].D = D + (self.c.fluid_relaxation / 3.0) * residual * ti.Matrix.identity(ti.f32, 3)

    @ti.kernel
    def integrate(self):
        for p in range(self.active[None]):
            if self.p[p].material != RIGID:
                if self.p[p].material == FLUID:
                    self.p[p].volume_ratio = ti.max(
                        self.p[p].volume_ratio * (1.0 + self.p[p].D.trace()), 0.2)
                if ti.static(self.has_elastic):
                    if self.p[p].material == ELASTIC:
                        F = (ti.Matrix.identity(ti.f32, 3) + self.p[p].D) @ self.p[p].F
                        if ti.static(self.c.elastic_model == "legacy"):
                            U, singular, V = ti.svd(F)
                            for axis in ti.static(range(3)):
                                singular[axis, axis] = ti.min(ti.max(singular[axis, axis], .1), 100000.)
                            F = U@singular@V.transpose()
                        self.p[p].F = F
                        self.p[p].volume_ratio = F.determinant()
                predicted = self.p[p].x + self.p[p].d
                if ti.static(self.c.world_contact):
                    # Contact runs after grid boundaries and may otherwise
                    # push a fluid point through the floor or a side wall.
                    lo, hi = self.c.boundary / self.c.grid, 1 - self.c.boundary / self.c.grid
                    corrected = ti.max(lo, ti.min(hi, predicted))
                    self.p[p].d += corrected - predicted
                    predicted = corrected
                self.p[p].x = predicted
                if ti.static(self.c.elastic_model == "legacy" and self.has_elastic):
                    if self.p[p].material == ELASTIC:
                        self.p[p].d[1] += self.c.gravity*self.c.dt**2

