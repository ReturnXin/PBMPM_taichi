"""Box support points and separating-axis contacts for small rigid objects."""
import taichi as ti

from .rigid import rotation


@ti.func
def support_arm(R, half, direction):
    local = ti.Vector.zero(ti.f32, 3)
    for a in ti.static(range(3)):
        component = direction.dot(R[:, a])
        if ti.abs(component) > 1e-6:
            local[a] = half[a]*ti.math.sign(component)
    return R@local


@ti.func
def inverse_contact_mass(body, arm, normal):
    rn = arm.cross(normal)
    return 1/body.mass+rn.dot(body.inverse_world@rn)


@ti.data_oriented
class RigidBoxPairs:
    def __init__(self, bodies):
        self.b, self.count = bodies.data, bodies.count

    @ti.kernel
    def project(self):
        # Small demonstration populations: serial pair sweeps avoid conflicting pose writes.
        ti.loop_config(serialize=True)
        for a in range(self.count):
            for b in range(a+1, self.count):
                A, B = rotation(self.b[a].q), rotation(self.b[b].q)
                difference = self.b[b].x-self.b[a].x
                depth, normal, hit = 1e10, ti.Vector.zero(ti.f32, 3), True
                for axis_id in range(15):
                    axis = ti.Vector.zero(ti.f32, 3)
                    if axis_id < 3:
                        axis = A[:, axis_id]
                    elif axis_id < 6:
                        axis = B[:, axis_id-3]
                    else:
                        axis = A[:, (axis_id-6)//3].cross(B[:, (axis_id-6)%3])
                    length = axis.norm()
                    if length > 1e-6:
                        axis /= length
                        ra, rb = 0., 0.
                        for k in ti.static(range(3)):
                            ra += self.b[a].half_size[k]*ti.abs(axis.dot(A[:, k]))
                            rb += self.b[b].half_size[k]*ti.abs(axis.dot(B[:, k]))
                        overlap = ra+rb-ti.abs(difference.dot(axis))
                        if overlap <= 0:
                            hit = False
                        if overlap < depth:
                            depth = overlap
                            normal = axis if difference.dot(axis) >= 0 else -axis
                if hit:
                    ma = 0. if self.b[a].fixed else 1/self.b[a].mass
                    mb = 0. if self.b[b].fixed else 1/self.b[b].mass
                    if ma+mb > 0:
                        arm_a = support_arm(A, self.b[a].half_size, normal)
                        arm_b = support_arm(B, self.b[b].half_size, -normal)
                        relative = self.b[a].v+self.b[a].omega.cross(arm_a)-self.b[b].v-self.b[b].omega.cross(arm_b)
                        vn = relative.dot(normal)
                        if vn > 0:
                            inverse = 0.
                            if ma > 0:
                                inverse += inverse_contact_mass(self.b[a], arm_a, normal)
                            if mb > 0:
                                inverse += inverse_contact_mass(self.b[b], arm_b, normal)
                            impulse = vn/inverse*normal
                            if ma > 0:
                                self.b[a].v -= ma*impulse
                                self.b[a].omega -= self.b[a].inverse_world@arm_a.cross(impulse)
                            if mb > 0:
                                self.b[b].v += mb*impulse
                                self.b[b].omega += self.b[b].inverse_world@arm_b.cross(impulse)
                        self.b[a].x -= depth*ma/(ma+mb)*normal
                        self.b[b].x += depth*mb/(ma+mb)*normal
