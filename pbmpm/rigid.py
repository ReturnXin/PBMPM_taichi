import numpy as np
import taichi as ti


@ti.dataclass
class Body:
    x: ti.types.vector(3, ti.f32)
    q: ti.types.vector(4, ti.f32)
    v: ti.types.vector(3, ti.f32)
    omega: ti.types.vector(3, ti.f32)
    mass: ti.f32
    inverse_local: ti.types.matrix(3, 3, ti.f32)
    inverse_world: ti.types.matrix(3, 3, ti.f32)
    delta_momentum: ti.types.vector(3, ti.f32)
    delta_angular: ti.types.vector(3, ti.f32)


@ti.func
def rotation(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ti.Matrix([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@ti.func
def skew(w):
    return ti.Matrix([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


@ti.data_oriented
class RigidBodies:
    """Rigid kinematics and the particle-displacement feedback used in chapter 4."""

    def __init__(self, particles, config, initial):
        self.p, self.n, self.c = particles.data, particles.n, config
        self.count = len(initial)
        self.data = Body.field(shape=max(1, self.count))
        self.initial = initial
        self.restore()

    def restore(self):
        for i, body in enumerate(self.initial):
            self.data[i] = Body(x=body["center"], q=[1., 0., 0., 0.], v=body["velocity"],
                                omega=body["omega"], mass=body["mass"],
                                inverse_local=body["inverse_inertia"], inverse_world=body["inverse_inertia"],
                                delta_momentum=[0., 0., 0.], delta_angular=[0., 0., 0.])

    @ti.kernel
    def predict(self):
        for b in range(self.count):
            self.data[b].v[1] += self.c.gravity * self.c.dt
            R = rotation(self.data[b].q)
            self.data[b].inverse_world = R @ self.data[b].inverse_local @ R.transpose()

    @ti.kernel
    def broadcast(self):
        for p in range(self.n):
            b = self.p[p].body_id
            if b >= 0:
                arm = rotation(self.data[b].q) @ self.p[p].local
                d = (self.data[b].v + self.data[b].omega.cross(arm)) * self.c.dt
                self.p[p].d = d
                self.p[p].predicted_d = d
                self.p[p].D = skew(self.data[b].omega) * self.c.dt

    @ti.kernel
    def clear_feedback(self):
        for b in range(self.count):
            self.data[b].delta_momentum = ti.Vector.zero(ti.f32, 3)
            self.data[b].delta_angular = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def collect_feedback(self):
        for p in range(self.n):
            b = self.p[p].body_id
            if b >= 0:
                impulse = self.p[p].mass * (self.p[p].d - self.p[p].predicted_d) / self.c.dt
                arm = rotation(self.data[b].q) @ self.p[p].local
                self.data[b].delta_momentum += impulse
                self.data[b].delta_angular += arm.cross(impulse)

    @ti.kernel
    def apply_feedback(self):
        for b in range(self.count):
            self.data[b].v += self.data[b].delta_momentum / self.data[b].mass
            self.data[b].omega += self.data[b].inverse_world @ self.data[b].delta_angular

    def project(self):
        if self.c.coupling == "two_way":
            self.clear_feedback()
            self.collect_feedback()
            self.apply_feedback()

    @ti.kernel
    def integrate(self):
        for b in range(self.count):
            self.data[b].x += self.c.dt * self.data[b].v
            q, w = self.data[b].q, self.data[b].omega
            xyz = ti.Vector([q[1], q[2], q[3]])
            v = q[0] * w + w.cross(xyz)
            dq = ti.Vector([-w.dot(xyz), v[0], v[1], v[2]])
            q += 0.5 * self.c.dt * dq
            self.data[b].q = q / ti.max(q.norm(), 1e-12)
        for p in range(self.n):
            b = self.p[p].body_id
            if b >= 0:
                arm = rotation(self.data[b].q) @ self.p[p].local
                self.p[p].x = self.data[b].x + arm
                self.p[p].d = (self.data[b].v + self.data[b].omega.cross(arm)) * self.c.dt
                self.p[p].D = skew(self.data[b].omega) * self.c.dt


def body_description(local, center, particle_mass, velocity=(0, 0, 0), omega=(0, 0, 0)):
    local = np.asarray(local, dtype=np.float64)
    inertia = particle_mass * (np.eye(3) * np.sum(local * local) - local.T @ local)
    return dict(center=list(center), mass=float(particle_mass * len(local)),
                inverse_inertia=np.linalg.inv(inertia).astype(np.float32),
                velocity=list(velocity), omega=list(omega))
