"""Mouse-driven rigid container; shared moving-wall contact for water and soft bodies."""
import math

import numpy as np
import taichi as ti

from .corrections import ContactCorrection
from .box_contact import support_arm, inverse_contact_mass
from .rigid import rotation
from .state import ELASTIC, RIGID


def roll_matrix(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


class BoxMotion:
    """Bounded damped mouse targets, integrated in physical time, independent of FPS."""
    def __init__(self, center, half_size, grid, boundary):
        self.origin = np.asarray(center, dtype=float)
        self.half_size = np.asarray(half_size, dtype=float)
        self.lower, self.upper = boundary/grid + .015, 1-boundary/grid - .015
        self.reset()

    def reset(self):
        self.center = self.origin.copy()
        self.velocity = np.zeros(3)
        self.angle = self.angular_velocity = self.time = 0.
        self.target = self.origin.copy()
        self.target_angle = 0.
        self.automatic = False

    def set_target(self, horizontal, angle):
        horizontal = np.asarray(horizontal, dtype=float)
        if horizontal.shape != (2,) or not np.isfinite(horizontal).all() or not math.isfinite(angle):
            raise ValueError("mouse target must contain finite horizontal coordinates and angle")
        self.target[[0, 2]] = self.origin[[0, 2]] + np.clip(horizontal, -.085, .085)
        self.target_angle = float(np.clip(angle, -.20, .20))

    def step(self, dt):
        if self.automatic:
            t = max(0., self.time-.5)
            envelope = min(1., t/.45)
            self.set_target(envelope*np.array([.075*math.sin(2*math.pi*.9*t),
                                               .038*math.sin(2*math.pi*.6*t)]),
                            envelope*.18*math.sin(2*math.pi*.65*t))
        old_center, old_angle = self.center.copy(), self.angle
        acceleration = np.clip(140*(self.target-self.center)-24*self.velocity, -14, 14)
        self.velocity = np.clip(self.velocity+dt*acceleration, -.85, .85)
        self.center += dt*self.velocity
        alpha = np.clip(140*(self.target_angle-self.angle)-24*self.angular_velocity, -28, 28)
        self.angular_velocity = float(np.clip(self.angular_velocity+dt*alpha, -1.8, 1.8))
        self.angle = float(np.clip(self.angle+dt*self.angular_velocity, -.20, .20))
        extent = np.abs(roll_matrix(self.angle)) @ self.half_size
        self.center = np.clip(self.center, self.lower+extent, self.upper-extent)
        # Contact sees the displacement actually executed after bounds clipping.
        self.velocity = (self.center-old_center)/dt
        self.angular_velocity = (self.angle-old_angle)/dt
        self.time += dt


@ti.data_oriented
class MovingContainer(ContactCorrection):
    def __init__(self, particles, config, description, bodies):
        super().__init__(particles, config)
        self.b, self.body_count = bodies.data, bodies.count
        self.motion = BoxMotion(description["center"], description["half_size"], config.grid, config.boundary)
        self.half_size = tuple(description["half_size"])
        self.center = ti.Vector.field(3, ti.f32, shape=())
        self.orientation = ti.Matrix.field(3, 3, ti.f32, shape=())
        self.velocity = ti.Vector.field(3, ti.f32, shape=())
        self.omega = ti.Vector.field(3, ti.f32, shape=())
        self.wall_impulse = ti.Vector.field(3, ti.f32, shape=())
        self.contact_count = ti.field(ti.i32, shape=())
        self.particle_reaction = ti.Vector.field(3, ti.f32, shape=self.n)
        self.particle_hits = ti.field(ti.i32, shape=self.n)
        self.metric_tiles = (self.n+127)//128
        self.tile_reaction = ti.Vector.field(3, ti.f32, shape=self.metric_tiles)
        self.tile_hits = ti.field(ti.i32, shape=self.metric_tiles)
        self.reset()

    def publish(self):
        m = self.motion
        self.upload_pose(float(m.center[0]), float(m.center[1]), float(m.center[2]),
                         float(m.velocity[0]), float(m.velocity[1]), float(m.velocity[2]),
                         m.angle, m.angular_velocity)

    @ti.kernel
    def upload_pose(self, cx: ti.f32, cy: ti.f32, cz: ti.f32,
                    vx: ti.f32, vy: ti.f32, vz: ti.f32, angle: ti.f32, omega: ti.f32):
        # One device update replaces four separate scalar-field host accesses.
        self.center[None] = ti.Vector([cx, cy, cz])
        c, s = ti.cos(angle), ti.sin(angle)
        self.orientation[None] = ti.Matrix([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        self.velocity[None] = ti.Vector([vx, vy, vz])
        self.omega[None] = ti.Vector([0., 0., omega])

    def reset(self):
        super().reset()
        self.motion.reset()
        self.wall_impulse.fill(0)
        self.contact_count.fill(0)
        self.publish()

    def begin_step(self):
        self.motion.step(self.c.dt)
        self.publish()

    @ti.kernel
    def grid_boundary(self, transfers: ti.template()):
        # Apply wall-normal motion before G2P reconstructs d and D. Position
        # clipping alone hides compression from the local volume constraint.
        for node in ti.grouped(transfers.mass):
            self.grid_boundary_node(node, transfers)

    @ti.func
    def grid_boundary_node(self, node, transfers: ti.template()):
        # Per-node body of grid_boundary(); a @ti.func so the solver can fuse
        # it with transfers.boundary_node into one grid kernel.
        R, center = self.orientation[None], self.center[None]
        if transfers.mass[node] > 0:
            point = ti.cast(node, ti.f32)/self.c.grid
            local = R.transpose()@(point-center)
            wall_d = self.c.dt*(self.velocity[None]+self.omega[None].cross(point-center))
            before = transfers.displacement[node]
            relative = R.transpose()@(before-wall_d)
            for axis in ti.static(range(3)):
                if local[axis] <= -self.half_size[axis] and relative[axis] < 0:
                    relative[axis] = 0
                if local[axis] >= self.half_size[axis] and relative[axis] > 0:
                    relative[axis] = 0
            transfers.displacement[node] = wall_d+R@relative
            delta = transfers.displacement[node]-before
            if delta.norm_sqr() > 0:
                self.wall_impulse[None] -= transfers.mass[node]*delta/self.c.dt

    @ti.kernel
    def project(self):
        for p in range(self.active[None]):
            self.project_particle(p)
        for tile in range(self.metric_tiles):
            self.accumulate_tile_metrics(tile)
        self.finish_particle_metrics()

    @ti.func
    def project_particle(self, p, prior_delta: ti.template() = None, merge: ti.template() = False):
        R, center = self.orientation[None], self.center[None]
        self.particle_reaction[p] = ti.Vector.zero(ti.f32, 3)
        self.particle_hits[p] = 0
        hits = 0
        self.contact_delta[p] = ti.Vector.zero(ti.f32, 3)
        if self.p[p].material != RIGID:
            before = self.p[p].d
            displacement = before
            for axis in ti.static(range(3)):
                point = self.p[p].x+displacement
                local = R.transpose() @ (point-center)
                depth = ti.abs(local[axis])-self.half_size[axis]
                if depth > 0:
                    side = 1.0 if local[axis] >= 0 else -1.0
                    normal = -side*R[:, axis]
                    displacement += depth*normal
                    if ti.static(self.c.soft_friction > 0):
                        if self.p[p].material == ELASTIC:
                            relative = displacement/self.c.dt-self.velocity[None]-self.omega[None].cross(point-center)
                            tangent = relative-normal*relative.dot(normal)
                            speed = tangent.norm()
                            if speed > 1e-8:
                                displacement -= self.c.dt*ti.min(speed, self.c.soft_friction*depth/self.c.dt)*tangent/speed
                    hits += 1
            self.p[p].d = displacement
            delta = displacement-before
            self.particle_reaction[p] = -self.p[p].mass*delta/self.c.dt
            self.particle_hits[p] = hits
            self.contact_delta[p] = delta
            if ti.static(merge):
                self.contact_delta[p] += prior_delta[p]

    @ti.func
    def accumulate_tile_metrics(self, tile):
        reaction = ti.Vector.zero(ti.f32, 3)
        hits = 0
        for j in range(128):
            p = tile*128+j
            if p < self.active[None]:
                reaction += self.particle_reaction[p]
                hits += self.particle_hits[p]
        self.tile_reaction[tile] = reaction
        self.tile_hits[tile] = hits

    @ti.func
    def finish_particle_metrics(self, prior_work: ti.template() = None, merge: ti.template() = False):
        reaction = ti.Vector.zero(ti.f32, 3)
        hits = 0
        ti.loop_config(serialize=True)
        for tile in range(self.metric_tiles):
            reaction += self.tile_reaction[tile]
            hits += self.tile_hits[tile]
        # One thread writes each aggregate once; no contended atomic updates.
        self.wall_impulse[None] = self.wall_impulse[None]+reaction
        self.contact_count[None] = self.contact_count[None]+hits
        self.contact_work[None] = ti.cast(hits > 0, ti.i32)
        if ti.static(merge):
            self.contact_work[None] = ti.max(self.contact_work[None], prior_work[None])

    @ti.kernel
    def resolve_bodies(self):
        box_R, center = self.orientation[None], self.center[None]
        for b in range(self.body_count):
            if self.b[b].fixed == 0:
                R = rotation(self.b[b].q)
                extent = ti.abs(box_R.transpose()@R)@self.b[b].half_size
                for axis in ti.static(range(3)):
                    for side in ti.static((-1, 1)):
                        local = box_R.transpose()@(self.b[b].x-center)
                        depth = side*local[axis]+extent[axis]-self.half_size[axis]
                        if depth > 0:
                            normal = -side*box_R[:, axis]
                            self.b[b].x += depth*normal
                            arm = support_arm(R, self.b[b].half_size, -normal)
                            wall_v = self.velocity[None]+self.omega[None].cross(self.b[b].x+arm-center)
                            relative = self.b[b].v+self.b[b].omega.cross(arm)-wall_v
                            vn = relative.dot(normal)
                            if vn < 0:
                                amount = -vn/inverse_contact_mass(self.b[b], arm, normal)
                                impulse = amount*normal
                                self.b[b].v += impulse/self.b[b].mass
                                self.b[b].omega += self.b[b].inverse_world@arm.cross(impulse)
                                self.wall_impulse[None] -= impulse
                            self.contact_count[None] += 1

    def metrics(self, particles):
        continuum = particles["body_id"] < 0
        local = (particles["x"][continuum]-self.motion.center) @ roll_matrix(self.motion.angle)
        return dict(container_x=float(self.motion.center[0]), container_z=float(self.motion.center[2]),
                    container_roll=float(self.motion.angle),
                    container_max_penetration=float(max(0., (np.abs(local)-self.half_size).max())),
                    container_wall_impulse=self.wall_impulse[None].to_numpy().tolist(),
                    container_contacts=int(self.contact_count[None]))
