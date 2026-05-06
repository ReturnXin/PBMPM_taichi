import taichi as ti

__all__ = ["sdf_box", "collide"]


@ti.func
def sdf_box(pos, center, size):
    p = pos - center
    half_size = size * 0.5
    q = ti.abs(p) - half_size
    return ti.Vector([ti.max(q[0], 0.0), ti.max(q[1], 0.0), ti.max(q[2], 0.0)]).norm() + ti.min(
        ti.max(q[0], ti.max(q[1], q[2])), 0.0
    )


@ti.func
def collide(pos, shape):
    is_collide = False
    internal_dist = 0.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    point_on_surface = pos

    center = shape.center

    if shape.type == 0:  # ball
        offset = pos - center
        dist = offset.norm()
        internal_dist = dist - shape.radius
        is_collide = dist < shape.radius
        if is_collide:
            if dist > 1e-6:
                normal = offset / dist
            else:
                normal = ti.Vector([0.0, 1.0, 0.0])
        point_on_surface = center + normal * shape.radius

    elif shape.type == 1:  # box
        half_size = shape.size / 2
        local_pos = pos - center
        q = ti.abs(local_pos) - half_size
        external_dist = ti.Vector([ti.max(q[0], 0.0), ti.max(q[1], 0.0), ti.max(q[2], 0.0)]).norm()
        internal_dist = ti.min(ti.max(q[0], ti.max(q[1], q[2])), 0.0)
        dist = external_dist + internal_dist
        is_collide = dist < 0
        normal = ti.Vector([0.0, 0.0, 0.0])
        point_on_surface = pos
        if is_collide:
            if q[0] > q[1] and q[0] > q[2]:
                normal[0] = 1.0 if local_pos[0] > 0 else -1.0
            elif q[1] > q[2]:
                normal[1] = 1.0 if local_pos[1] > 0 else -1.0
            else:
                normal[2] = 1.0 if local_pos[2] > 0 else -1.0
        point_on_surface = pos - dist * normal

    return is_collide, -internal_dist, -normal, point_on_surface
