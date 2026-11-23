"""Continuum/box contact with Coulomb friction for elastic material."""

import taichi as ti

from .rigid import predicted_quaternion, rotation
from .state import RIGID, ELASTIC
from .corrections import ContactCorrection


@ti.data_oriented
class Contacts(ContactCorrection):
    def __init__(self, particles, bodies, config):
        super().__init__(particles, config)
        self.bodies = bodies
        self.b, self.count = bodies.data, bodies.count
        self.normal_impulse = ti.Vector.field(3, ti.f32, shape=max(1, self.count))
        self.friction_impulse = ti.Vector.field(3, ti.f32, shape=max(1, self.count))
        self.reset()

    def reset(self):
        self.normal_impulse.fill(0)
        self.friction_impulse.fill(0)
        super().reset()

    @ti.func
    def inverse_effective_mass(self, p, b, arm, direction):
        value = 1.0 / self.p[p].mass
        if ti.static(self.c.coupling == "two_way"):
            if self.b[b].fixed == 0:
                r_cross = arm.cross(direction)
                value += 1.0 / self.b[b].mass + r_cross.dot(self.b[b].inverse_world @ r_cross)
        return value

    @ti.kernel
    def project(self):
        self.contact_work[None] = 0
        for p in range(self.active[None]):
            self.project_particle(p)

    @ti.func
    def project_particle(self, p):
        self.contact_delta[p] = ti.Vector.zero(ti.f32, 3)
        if self.p[p].material != RIGID:
            for b in range(self.count):
                lo, hi = self.c.boundary / self.c.grid, 1 - self.c.boundary / self.c.grid
                center = self.b[b].x + self.c.dt * self.b[b].v
                q = predicted_quaternion(self.b[b].q, self.b[b].omega, self.c.dt)
                R = rotation(q)
                if ti.static(self.c.world_contact):
                    extent = ti.abs(R) @ self.b[b].half_size
                    center = ti.max(lo + extent, ti.min(hi - extent, center))
                point = self.p[p].x + self.p[p].d
                local = R.transpose() @ (point - center)
                gap = self.b[b].half_size - ti.abs(local)
                if gap.min() > 0:
                    # Pick the nearest reachable face. A box touching the
                    # floor must not expel trapped water below that floor.
                    depth = 1e10
                    normal = ti.Vector.zero(ti.f32, 3)
                    for axis in ti.static(range(3)):
                        for side in ti.static((-1, 1)):
                            candidate = side * R[:, axis]
                            distance = self.b[b].half_size[axis] - side * local[axis]
                            reachable = True
                            if ti.static(self.c.world_contact):
                                surface = point + distance * candidate
                                for a in ti.static(range(3)):
                                    reachable = reachable and surface[a] >= lo + 1e-6 and surface[a] <= hi - 1e-6
                            if reachable and distance < depth:
                                depth, normal = distance, candidate
                    arm = point - center
                    inv_mass = self.inverse_effective_mass(p, b, arm, normal)
                    normal_size = 0.0
                    if depth < 1e9:
                        normal_size = ti.min(depth, 1.0) / (self.c.dt * inv_mass)
                    impulse = normal_size * normal
                    friction = ti.Vector.zero(ti.f32, 3)
                    if ti.static(self.c.soft_friction > 0):
                        if self.p[p].material == ELASTIC:
                            relative = self.p[p].d / self.c.dt - (
                                self.b[b].v + self.b[b].omega.cross(arm))
                            # Include this pair's normal response before
                            # solving its remaining tangential slip.
                            relative += impulse / self.p[p].mass
                            if ti.static(self.c.coupling == "two_way"):
                                if self.b[b].fixed == 0:
                                    relative += impulse / self.b[b].mass + (
                                        self.b[b].inverse_world @ arm.cross(impulse)).cross(arm)
                            tangent = relative - normal * relative.dot(normal)
                            speed = tangent.norm()
                            if speed > 1e-8:
                                direction = tangent / speed
                                tangent_mass = self.inverse_effective_mass(p, b, arm, direction)
                                # Stick if the stopping impulse lies inside
                                # the Coulomb limit; otherwise slide. Only
                                # newly generated normal impulse is budgeted
                                # here, so iterations do not reuse a load.
                                magnitude = ti.min(speed / tangent_mass,
                                                   self.c.soft_friction * normal_size)
                                friction = -magnitude * direction
                    impulse += friction
                    self.p[p].d += self.c.dt * impulse / self.p[p].mass
                    self.contact_delta[p] += self.c.dt * impulse / self.p[p].mass
                    if impulse.norm_sqr() > 0:
                        ti.atomic_max(self.contact_work[None], 1)
                    if ti.static(self.c.coupling == "two_way"):
                        self.b[b].delta_momentum -= impulse
                        self.b[b].delta_angular -= arm.cross(impulse)
                        self.normal_impulse[b] -= normal_size * normal
                        self.friction_impulse[b] -= friction

