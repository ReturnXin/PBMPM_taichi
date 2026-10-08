import taichi as ti

from .state import ELASTIC, FLUID, RIGID, SAND


@ti.func
def regularized_inverse(F):
    U, s, V = ti.svd(F)
    inverse = ti.Matrix.zero(ti.f32, 3, 3)
    for a in ti.static(range(3)):
        inverse[a, a] = 1.0 / ti.max(s[a, a], 0.1)
    return V @ inverse @ U.transpose()


@ti.func
def elastic_target(F, D, volume_weight):
    I = ti.Matrix.identity(ti.f32, 3)
    trial = (I + D) @ F
    U, s, V = ti.svd(trial)
    shape = U @ V.transpose()
    # Keep the rotation proper when regularization encounters an inversion.
    if shape.determinant() < 0:
        for a in ti.static(range(3)):
            U[a, 2] = -U[a, 2]
        shape = U @ V.transpose()
    volume = trial / ti.pow(ti.max(trial.determinant(), 1e-6), 1.0 / 3.0)
    target = (1.0 - volume_weight) * shape + volume_weight * volume
    return target @ regularized_inverse(F) - I


@ti.data_oriented
class Materials:
    def __init__(self, particles, config):
        self.p = particles.data
        self.n = particles.n
        self.c = config
        self.residuals = ti.field(ti.f32, shape=3)

    @ti.func
    def target(self, F, D):
        result = ti.Matrix.zero(ti.f32, 3, 3)
        if ti.static(self.c.elastic_model == "legacy"):
            I = ti.Matrix.identity(ti.f32, 3)
            U, s, V = ti.svd((I + D) @ F)
            # mpm_pbd.py uses U @ V.T directly, including its reflection behavior.
            # Its reconstructed/clamped trial F is unused by the elastic target.
            result = (U @ V.transpose()) @ regularized_inverse(F) - I
        else:
            result = elastic_target(F, D, self.c.volume_weight)
        return result

    @ti.kernel
    def measure_residuals(self):
        for i in range(3):
            self.residuals[i] = 0
        for p in range(self.n):
            if self.p[p].material == ELASTIC and not self.p[p].pinned:
                target = self.target(self.p[p].F, self.p[p].D)
                residual = self.p[p].D - target
                flexible = residual + self.c.compliance / self.c.dt**2 * self.p[p].multiplier
                self.residuals[0] += residual.norm()
                self.residuals[1] += flexible.norm()
                self.residuals[2] += 1

    @ti.kernel
    def predict(self):
        for p in range(self.n):
            if ti.static(self.c.elastic_model == "legacy" and self.c.elastic_solver == "xpbmpm"):
                self.p[p].multiplier *= 0.5
            else:
                self.p[p].multiplier = ti.Matrix.zero(ti.f32, 3, 3)
            if self.p[p].material != RIGID and not self.p[p].pinned:
                if ti.static(self.c.elastic_model == "legacy"):
                    # Legacy elastic particles receive next-step gravity in integrate().
                    if self.p[p].material != ELASTIC:
                        self.p[p].d[1] += self.c.gravity * self.c.dt**2
                else:
                    self.p[p].d[1] += self.c.gravity * self.c.dt**2

    @ti.kernel
    def solve(self):
        for p in range(self.n):
            I = ti.Matrix.identity(ti.f32, 3)
            D = self.p[p].D
            F = self.p[p].F
            material = self.p[p].material
            if self.p[p].pinned:
                self.p[p].d = ti.Vector.zero(ti.f32, 3)
                self.p[p].D = ti.Matrix.zero(ti.f32, 3, 3)
            elif material == FLUID:
                D -= 0.5 * self.c.viscosity * (D + D.transpose())
                residual = 1.0 / ti.max(self.p[p].volume_ratio, 0.1) - 1.0 - D.trace()
                self.p[p].D = D + (self.c.fluid_relaxation / 3.0) * residual * I
            elif material == ELASTIC:
                target = self.target(F, D)
                if ti.static(self.c.elastic_solver == "xpbmpm"):
                    alpha = self.c.compliance / self.c.dt**2
                    delta = (target - D - alpha * self.p[p].multiplier) / (self.c.inverse_metric + alpha)
                    self.p[p].D = D + self.c.inverse_metric * delta
                    self.p[p].multiplier += delta
                else:
                    self.p[p].D = D + self.c.relaxation * (target - D)
            elif material == SAND:
                trial = (I + D) @ F
                U, s, V = ti.svd(trial)
                for a in ti.static(range(3)):
                    if self.p[p].plastic_volume == 0:
                        s[a, a] = ti.min(ti.max(s[a, a], 1.0), 1000.0)
                target = (U @ s @ V.transpose()) @ regularized_inverse(F) - I
                D += self.c.sand_relaxation * (target - D)
                self.p[p].D = D - 0.05 * (D + D.transpose())

    @ti.func
    def sand_plasticity(self, p, F):
        U, s, V = ti.svd(F)
        e = ti.Vector.zero(ti.f32, 3)
        for a in ti.static(range(3)):
            e[a] = ti.log(ti.max(s[a, a], 1e-6))
        trace = e.sum() + self.p[p].plastic_volume
        deviator = e - trace / 3.0
        norm = deviator.norm()
        sin_phi = ti.sin(self.c.friction_angle * 3.141592653589793 / 180)
        friction = ti.sqrt(2.0 / 3.0) * 2 * sin_phi / (3 - sin_phi)
        if trace >= 0:
            e = ti.Vector.zero(ti.f32, 3)
            self.p[p].plastic_volume = 0.5 * trace
        else:
            self.p[p].plastic_volume = 0
            delta = norm + 2 * trace * friction
            if delta > 0 and norm > 1e-8:
                e -= delta * deviator / norm
        for a in ti.static(range(3)):
            s[a, a] = ti.exp(ti.min(ti.max(e[a], -9.0), 9.0))
        return U @ s @ V.transpose()

    @ti.kernel
    def integrate(self):
        for p in range(self.n):
            if self.p[p].material != RIGID and not self.p[p].pinned:
                if self.p[p].material == FLUID:
                    self.p[p].volume_ratio = ti.max(
                        self.p[p].volume_ratio * (1 + self.p[p].D.trace()), 0.2)
                else:
                    F = (ti.Matrix.identity(ti.f32, 3) + self.p[p].D) @ self.p[p].F
                    if self.p[p].material == SAND:
                        F = self.sand_plasticity(p, F)
                    if ti.static(self.c.elastic_model == "legacy"):
                        if self.p[p].material == ELASTIC:
                            # Match mpm_pbd.py::compute_F after elastic integration.
                            U, s, V = ti.svd(F)
                            for a in ti.static(range(3)):
                                s[a, a] = ti.min(ti.max(s[a, a], 0.1), 100000.0)
                            F = U @ s @ V.transpose()
                    self.p[p].F = F
                self.p[p].x += self.p[p].d
                if ti.static(self.c.elastic_model == "legacy"):
                    if self.p[p].material == ELASTIC:
                        self.p[p].d[1] += self.c.gravity * self.c.dt**2
