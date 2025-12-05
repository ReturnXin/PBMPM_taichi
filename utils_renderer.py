import taichi as ti
import numpy as np


def get_box_wireframe(center, size):
    half_size = np.array(size) * 0.5
    center = np.array(center)
    signs = np.array(
        [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]
    )
    verts = center + signs * half_size
    indices = [0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7]
    lines_v = verts[indices]
    return lines_v


def get_unit_cube_mesh():
    # fmt: off
    vertices = np.array([
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5], # Front (Z+)
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5]  # Back (Z-)
    ], dtype=np.float32)
    indices = np.array([
        0, 1, 2, 2, 3, 0, # Front
        5, 4, 7, 7, 6, 5, # Back
        4, 0, 3, 3, 7, 4, # Left
        1, 5, 6, 6, 2, 1, # Right
        3, 2, 6, 6, 7, 3, # Top
        4, 5, 1, 1, 0, 4  # Bottom
    ], dtype=np.int32)
    # fmt: on
    return vertices, indices


def get_sphere_mesh(resolution=20):
    vertices = []
    indices = []

    pi = np.pi
    for i in range(resolution + 1):
        lat = pi * i / resolution
        for j in range(resolution + 1):
            lon = 2 * pi * j / resolution
            x = np.sin(lat) * np.cos(lon)
            y = np.cos(lat)
            z = np.sin(lat) * np.sin(lon)

            p = np.array([x, y, z])
            vertices.append(p)

    for i in range(resolution):
        for j in range(resolution):
            p1 = i * (resolution + 1) + j
            p2 = p1 + 1
            p3 = (i + 1) * (resolution + 1) + j
            p4 = p3 + 1
            indices.extend([p1, p3, p2, p2, p3, p4])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)


def render_obstacles(scene, solver):
    num_obs = solver.num_obstacles[None]
    if num_obs == 0:
        return
    all_lines = []
    all_colors = []

    for i in range(num_obs):
        center = solver.obstacles[i].center
        size = solver.obstacles[i].size
        color = solver.obstacles[i].color

        box_lines = get_box_wireframe(center, size)
        all_lines.append(box_lines)

        c = np.array([color[0], color[1], color[2]])
        colors = np.tile(c, (24, 1))
        all_colors.append(colors)

    if len(all_lines) > 0:
        lines_np = np.vstack(all_lines)
        colors_np = np.vstack(all_colors)

        lines_field = ti.Vector.field(3, dtype=ti.f32, shape=lines_np.shape[0])
        colors_field = ti.Vector.field(3, dtype=ti.f32, shape=colors_np.shape[0])

        lines_field.from_numpy(lines_np)
        colors_field.from_numpy(colors_np)

        scene.lines(lines_field, per_vertex_color=colors_field, width=3.0)
