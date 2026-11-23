"""Pack the shaking scene into one persistent host staging buffer per frame.

Taichi 1.7 CUDA/Windows renders through Vulkan and stages vertices on the host.
The ordinary scene API allocates and downloads a buffer for every draw call.
This adapter packs particles, float/stage meshes and box edges in one download.
"""
import numpy as np
import taichi as ti
from taichi.ui.utils import get_field_info

from .rigid import rotation


@ti.data_oriented
class ShakeRenderBuffer:
    def __init__(self, viewer):
        self.v = viewer
        self.state = viewer.render[0]
        self.p, self.b = self.state.p, self.state.b
        self.particle_count = self.state.n
        self.body_vertices = 36*self.state.body_count
        self.mesh_start = self.particle_count
        self.mesh_count = self.body_vertices+6
        self.line_start = self.mesh_start+self.mesh_count
        self.buffer = np.zeros((self.line_start+24, 12), np.float32)
        self.buffer[:,11] = 1
        self.buffer[:self.particle_count,8:11] = viewer.solvers[0].initial["color"]
        self.buffer[self.mesh_start:self.mesh_start+self.body_vertices,8:11] = (.45,.75,.39)
        start = self.mesh_start+self.body_vertices
        self.buffer[start:start+6,8:11] = (.07,.085,.12)
        self.particles_info = get_field_info(self.buffer[:self.mesh_start])
        self.mesh_info = get_field_info(self.buffer[self.mesh_start:self.line_start])
        self.lines_info = get_field_info(self.buffer[self.line_start:])
        self.empty_info = get_field_info(None)

    @ti.kernel
    def pack(self, out: ti.types.ndarray()):
        # Every loop stays at kernel top level; one external array is copied
        # back after all geometry has been reconstructed on the device.
        for p in range(self.particle_count):
            for a in ti.static(range(3)):
                out[p,a] = self.p[p].x[a]
            out[p,3] = self.state.base_radii[p] if p < self.state.active[None] else 0.
        for b in range(self.state.body_count):
            R = rotation(self.b[b].q)
            for j in range(36):
                index = self.mesh_start+36*b+j
                pos = self.b[b].x+R@self.state.local_faces[36*b+j]
                normal = R@self.v.batch.local_normals[36*b+j]
                for a in ti.static(range(3)):
                    out[index,a] = pos[a]
                    out[index,3+a] = normal[a]
        for j in range(6):
            index = self.mesh_start+self.body_vertices+j
            for a in ti.static(range(3)):
                out[index,a] = self.v.stage[j][a]
                out[index,3+a] = self.v.stage_normals[j][a]
        for j in range(24):
            pos = self.v.container.center[None]+self.v.container.orientation[None]@self.v.local_line_field[j]
            for a in ti.static(range(3)):
                out[self.line_start+j,a] = pos[a]

    def draw(self, scene, changed):
        if changed:
            self.pack(self.buffer)
        native = scene.scene
        native.mesh(self.mesh_info, True, self.empty_info, (1.,1.,1.), True,
                    self.mesh_count, 0, self.mesh_count, 0, False)
        native.particles(self.particles_info, True, True, (1.,1.,1.), 1., self.particle_count, 0)
        native.lines(self.lines_info, self.empty_info, False, (.64,.65,.84), 2., 24, 0, 24, 0)
