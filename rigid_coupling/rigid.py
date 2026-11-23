"""The rigid kinematics and mass-weighted feedback derived in chapter 4.

World-space angular velocity; scalar-first quaternion q=(w,x,y,z).
Only particle displacement feedback enters the six-DOF projection.
"""

from copy import deepcopy

import numpy as np
import taichi as ti


def _vector(value, size, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return array


def _unit_quaternion(value):
    q = _vector(value, 4, "quaternion")
    norm = np.linalg.norm(q)
    if norm < 1e-12 or not np.isfinite(norm):
        raise ValueError("quaternion must have a nonzero finite norm")
    return q / norm


def rotation_numpy(quaternion):
    """Local-to-world rotation; normalize an input scalar-first quaternion."""
    w, x, y, z = _unit_quaternion(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def make_body(local, center, particle_mass, velocity=(0, 0, 0), omega=(0, 0, 0),
              quaternion=(1, 0, 0, 0)):
    """Center samples by mass and compute the discrete local inertia tensor.

    ``center`` is the requested world center of mass. Returned ``local`` must
    be used when creating particles: input coordinates need not be centered.
    A scalar particle_mass gives uniform sampling; an (N,) array also works.
    A line/point sample has a singular 3-D inertia and is rejected.
    """
    local = np.array(local, dtype=np.float64, copy=True)
    if local.ndim != 2 or local.shape[1] != 3 or len(local) < 3 or not np.all(np.isfinite(local)):
        raise ValueError("local must be a finite (N, 3) array with at least three samples")
    masses = np.broadcast_to(np.asarray(particle_mass, dtype=np.float64), (len(local),)).copy()
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("particle_mass must contain finite positive values")
    total_mass = masses.sum()
    local -= np.sum(masses[:, None] * local, axis=0) / total_mass
    # I_local = sum m_p (|r_p|^2 I - r_p r_p^T).
    inertia = np.eye(3) * np.sum(masses * np.sum(local * local, axis=1)) - local.T @ (masses[:, None] * local)
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues[-1] <= 0 or eigenvalues[0] <= 1e-7 * eigenvalues[-1]:
        raise ValueError("rigid sampling has a singular or ill-conditioned inertia tensor")
    return dict(local=local, particle_mass=masses, mass=float(total_mass),
                inertia=inertia, inverse_inertia=np.linalg.inv(inertia),
                center=_vector(center, 3, "center"), velocity=_vector(velocity, 3, "velocity"),
                omega=_vector(omega, 3, "omega"), quaternion=_unit_quaternion(quaternion))


@ti.dataclass
class Body:
    x: ti.types.vector(3, ti.f32)
    q: ti.types.vector(4, ti.f32)
    v: ti.types.vector(3, ti.f32)
    omega: ti.types.vector(3, ti.f32)
    mass: ti.f32
    half_size: ti.types.vector(3, ti.f32)
    inverse_local: ti.types.matrix(3, 3, ti.f32)
    inverse_world: ti.types.matrix(3, 3, ti.f32)
    delta_momentum: ti.types.vector(3, ti.f32)
    delta_angular: ti.types.vector(3, ti.f32)
    fixed: ti.i32
    grid_impulse: ti.types.vector(3, ti.f32)
    contact_impulse: ti.types.vector(3, ti.f32)
    world_impulse: ti.types.vector(3, ti.f32)
    world_contacts: ti.i32


@ti.func
def rotation(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ti.Matrix([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@ti.func
def skew(w):
    return ti.Matrix([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


@ti.func
def predicted_quaternion(q, w, dt):
    xyz = ti.Vector([q[1], q[2], q[3]])
    v = q[0] * w + w.cross(xyz)
    q += 0.5 * dt * ti.Vector([-w.dot(xyz), v[0], v[1], v[2]])
    return q / ti.max(q.norm(), 1e-12)


@ti.data_oriented
class RigidBodies:
    def __init__(self, particles, config, initial):
        self.p, self.n, self.c = particles.data, particles.n, config
        self.active = particles.count
        self.initial = deepcopy(initial)
        self.count = len(initial)
        # Taichi requires a nonzero field shape; count=0 means the slot is unused.
        self.data = Body.field(shape=max(1, self.count))
        arrays = particles.snapshot()
        if np.any(arrays["body_id"] >= self.count):
            raise ValueError("particle body_id is outside the body list")
        for b, body in enumerate(self.initial):
            if config.rigid_contact or config.world_contact or config.buoyancy_assist:
                size = np.asarray(body.get("size", []), dtype=float)
                if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
                    raise ValueError("box contact requires a positive finite body size")
                if np.linalg.norm(size) >= 1 - 2 * config.boundary / config.grid:
                    raise ValueError("the rotating contact box must fit inside the world")
            mask = arrays["body_id"] == b
            if not np.any(mask):
                raise ValueError(f"body {b} has no sampling particles")
            local, masses = arrays["local"][mask].astype(float), arrays["mass"][mask].astype(float)
            centroid = np.sum(masses[:, None] * local, axis=0) / masses.sum()
            if np.linalg.norm(centroid) > 1e-6 * max(np.max(np.abs(local)), 1e-6):
                raise ValueError("use make_body's mass-centered local coordinates for particles")
            actual = make_body(local, body["center"], masses, body["velocity"], body["omega"], body["quaternion"])
            if not np.isclose(body["mass"], actual["mass"], rtol=2e-6, atol=0):
                raise ValueError("body mass must equal the sum of its particle masses")
            if not np.allclose(body["inverse_inertia"], actual["inverse_inertia"], rtol=2e-5,
                               atol=2e-6 * np.max(np.abs(actual["inverse_inertia"]))):
                raise ValueError("body inertia must be computed from its sampling particles")
            body["quaternion"] = actual["quaternion"]
        self.restore()

    def restore(self):
        for b, body in enumerate(self.initial):
            R = rotation_numpy(body["quaternion"])
            self.data[b] = Body(x=body["center"], q=body["quaternion"],
                                v=body["velocity"], omega=body["omega"], mass=body["mass"],
                                half_size=np.asarray(body.get("size", [0, 0, 0])) / 2,
                                inverse_local=body["inverse_inertia"],
                                inverse_world=R @ body["inverse_inertia"] @ R.T,
                                delta_momentum=[0., 0., 0.], delta_angular=[0., 0., 0.],
                                fixed=int(body.get("fixed", False)), grid_impulse=[0., 0., 0.],
                                contact_impulse=[0., 0., 0.], world_impulse=[0., 0., 0.], world_contacts=0)
        self.reconstruct()

    @ti.kernel
    def predict(self):
        # External acceleration is applied once per physical time step.
        for b in range(self.count):
            if self.data[b].fixed == 0:
                self.data[b].v[1] += self.c.gravity * self.c.dt
            R = rotation(self.data[b].q)
            self.data[b].inverse_world = R @ self.data[b].inverse_local @ R.transpose()

    @ti.kernel
    def broadcast(self):
        # d_pred = dt (v_b + omega_b x a_p), D_rig = dt [omega_b]_x.
        # No position or orientation changes during the inner iterations.
        for p in range(self.active[None]):
            b = self.p[p].body_id
            if b >= 0:
                self.broadcast_particle(p, b)

    @ti.func
    def broadcast_particle(self, p, b):
        arm = rotation(self.data[b].q) @ self.p[p].local
        d = self.c.dt * (self.data[b].v + self.data[b].omega.cross(arm))
        self.p[p].d = d
        self.p[p].predicted_d = d
        self.p[p].D = self.c.dt * skew(self.data[b].omega)

    @ti.kernel
    def clear_feedback(self):
        for b in range(self.count):
            self.data[b].delta_momentum = ti.Vector.zero(ti.f32, 3)
            self.data[b].delta_angular = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def collect_feedback(self):
        # delta_u_p = (d_return - d_pred) / dt, separately for each body_id.
        for p in range(self.active[None]):
            b = self.p[p].body_id
            if b >= 0:
                self.collect_particle_feedback(p, b)

    @ti.func
    def collect_particle_feedback(self, p, b):
        impulse = self.p[p].mass * (self.p[p].d - self.p[p].predicted_d) / self.c.dt
        arm = rotation(self.data[b].q) @ self.p[p].local
        self.data[b].delta_momentum += impulse
        self.data[b].delta_angular += arm.cross(impulse)

    @ti.kernel
    def apply_feedback(self):
        # Closed-form mass-weighted least squares: delta_v=sum(m delta_u)/M,
        # delta_omega=I_world^-1 sum(a x m delta_u).
        for b in range(self.count):
            self.apply_body_feedback(b, False)

    @ti.kernel
    def apply_contact_feedback(self):
        for b in range(self.count):
            self.apply_body_feedback(b, True)

    @ti.func
    def apply_body_feedback(self, b, contact):
            if contact:
                self.data[b].contact_impulse += self.data[b].delta_momentum
            else:
                self.data[b].grid_impulse += self.data[b].delta_momentum
            if self.data[b].fixed == 0:
                self.data[b].v += self.data[b].delta_momentum / self.data[b].mass
                self.data[b].omega += self.data[b].inverse_world @ self.data[b].delta_angular
            # Reuse the buffer for a separate, conservative surface-contact pass.
            if ti.static(self.c.rigid_contact):
                self.data[b].delta_momentum = ti.Vector.zero(ti.f32, 3)
                self.data[b].delta_angular = ti.Vector.zero(ti.f32, 3)

    def project(self):
        self.clear_feedback()
        if self.c.coupling == "two_way":
            self.collect_feedback()
            self.apply_feedback()

    @ti.kernel
    def advance_pose(self):
        for b in range(self.count):
          if self.data[b].fixed == 0:
            self.data[b].x += self.c.dt * self.data[b].v
            # q_new = normalize(q + dt/2 * (0,omega_world) * q).
            q = predicted_quaternion(self.data[b].q, self.data[b].omega, self.c.dt)
            self.data[b].q = q
            R = rotation(q)
            self.data[b].inverse_world = R @ self.data[b].inverse_local @ R.transpose()
            if ti.static(self.c.world_contact):
                self.resolve_world(b, R)

    @ti.func
    def resolve_world(self, b, R):
        # Static walls remain active in BOTH coupling modes. Clamp the common
        # pose (not individual samples), then cancel inward corner velocities
        # with frictionless, zero-restitution impulses.
        half = self.data[b].half_size
        extent = ti.abs(R) @ half
        lo = self.c.boundary / self.c.grid
        hi = 1.0 - lo
        for axis in range(3):
            for side_index in range(2):
                side = 2 * side_index - 1
                normal = ti.Vector.zero(ti.f32, 3)
                normal[axis] = -side
                penetration = lo - (self.data[b].x[axis] - extent[axis])
                if side == 1:
                    penetration = self.data[b].x[axis] + extent[axis] - hi
                if penetration > 0:
                    self.data[b].world_contacts += 1
                    self.data[b].x += penetration * normal
                    # Use the centroid of the supporting face/edge/vertex.
                    # A level box lands through its face center without the
                    # artificial torque of a single sequential corner sweep.
                    local = ti.Vector.zero(ti.f32, 3)
                    for a in ti.static(range(3)):
                        component = side * R[axis, a]
                        if ti.abs(component) > 1e-6:
                            local[a] = half[a] * ti.math.sign(component)
                    arm = R @ local
                    vn = (self.data[b].v + self.data[b].omega.cross(arm)).dot(normal)
                    if vn < 0:
                        rn = arm.cross(normal)
                        inv_mass = 1.0 / self.data[b].mass + rn.dot(self.data[b].inverse_world @ rn)
                        impulse = (-vn / inv_mass) * normal
                        self.data[b].world_impulse += impulse
                        self.data[b].v += impulse / self.data[b].mass
                        self.data[b].omega += self.data[b].inverse_world @ arm.cross(impulse)

    @ti.kernel
    def reconstruct(self):
        # Rigid positions are ALWAYS generated from one common pose.
        for p in range(self.active[None]):
            b = self.p[p].body_id
            if b >= 0:
                arm = rotation(self.data[b].q) @ self.p[p].local
                self.p[p].x = self.data[b].x + arm
                d = self.c.dt * (self.data[b].v + self.data[b].omega.cross(arm))
                self.p[p].d = d
                self.p[p].predicted_d = d
                self.p[p].D = self.c.dt * skew(self.data[b].omega)

    def integrate(self):
        self.advance_pose()
        self.reconstruct()

