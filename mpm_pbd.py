import taichi as ti
import numpy as np
from utils_renderer import get_unit_cube_mesh, get_sphere_mesh
from morton_code import get_morton_code
from typing import Any

MaterialParam = ti.types.struct(
    rho=ti.f32,
    viscosity=ti.f32,
    stiffness=ti.f32,
    E=ti.f32,
    beta=ti.f32,
    elastic_relaxation=ti.f32,
    friction_angle=ti.f32,
    color=ti.types.vector(3, ti.f32),
)

Obstacle = ti.types.struct(
    type=ti.types.int32,
    center=ti.types.vector(3, ti.f32),
    radius=ti.types.f32,
    size=ti.types.vector(3, ti.f32),
    color=ti.types.vector(3, ti.f32),
    velocity=ti.types.vector(3, ti.f32),
)


@ti.func
def sdf_box(pos, center, size):
    p = pos - center
    half_size = size * 0.5
    q = ti.abs(p) - half_size
    return ti.Vector([ti.max(q[0], 0.0), ti.max(q[1], 0.0), ti.max(q[2], 0.0)]).norm() + ti.min(
        ti.max(q[0], ti.max(q[1], q[2])), 0.0
    )


@ti.data_oriented
class MpmPBDSolver:

    def __init__(self):

        self.dim = 3
        self.dt = 8e-3
        self.neighbour = (3,) * self.dim

        self.n_loop = ti.field(dtype=ti.i32, shape=())

        self.p_rho = 1
        self.p_vol = 1 / 2**3
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = 9.8
        self.bound = 10
        self.iteration = 5
        self.average_height = ti.field(dtype=ti.f32, shape=())
        self.fps_count = ti.field(dtype=ti.i32, shape=())
        self.interia_force = ti.Vector.field(self.dim, dtype=ti.f32, shape=())

        # ===Particle
        self.max_particles = 300000
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.n_particles[None] = 0
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_particles)  # Position
        self.dis = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_particles)  # Displacement
        self.D = ti.Matrix.field(
            self.dim, self.dim, dtype=ti.f32, shape=self.max_particles
        )  # Deformation Displacement
        self.L = ti.field(dtype=ti.f32, shape=self.max_particles)  # Density
        self.F = ti.Matrix.field(
            self.dim, self.dim, dtype=ti.f32, shape=self.max_particles
        )  # Deformation Gradient
        self.lambdas = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.max_particles)
        self.log_JP = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.color = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.radius = ti.field(dtype=ti.f32, shape=self.max_particles)

        # ===Material
        self.num_materials = 3
        self.mat_params = MaterialParam.field(shape=(self.num_materials,))
        self.material = ti.field(dtype=ti.int32, shape=self.max_particles)  # 0：fluid，1: jelly, 2: snow

        # ===Grid
        self.n_grid = 64
        self.dx = 1 / self.n_grid
        self.grid_v = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.n_grid,) * self.dim)
        self.grid_dis = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.n_grid,) * self.dim)
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid,) * self.dim)
        self.grid_vol = ti.field(dtype=ti.f32, shape=(self.n_grid,) * self.dim)

        # ===Obstacles
        self.max_num_obstacles = 10
        self.num_obstacles = ti.field(dtype=ti.i32, shape=())
        self.num_obstacles[None] = 0
        self.obstacles = Obstacle.field(shape=(self.max_num_obstacles,))
        unit_v, _ = get_unit_cube_mesh()
        self.unit_cube_verts = ti.Vector.field(3, dtype=ti.f32, shape=len(unit_v))
        self.unit_cube_verts.from_numpy(unit_v)
        self.mesh_local_pos = None
        self.mesh_owner_id = None
        self.mesh_vertics = None
        self.mesh_indices = None
        self.mesh_colors = None

        # ===Debug
        self.num_grid_lines = 3 * (2 * (self.n_grid + 1) + 1)
        self.grid_lines_vertex = ti.Vector.field(self.dim, dtype=ti.float32, shape=self.num_grid_lines * 2)

        # ===Morton Code
        self.use_morton_code = True
        self.shrink_factor = 3.8
        self.gaps = ti.field(dtype=ti.i32, shape=self.iteration)
        self.particle_sort_keys = ti.field(dtype=ti.i32, shape=self.max_particles)
        self.particle_sort_indices = ti.field(dtype=ti.i32, shape=self.max_particles)
        self.temp_x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_particles)
        self.temp_dis = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.max_particles)
        self.temp_D = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.max_particles)
        self.temp_F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.max_particles)
        self.temp_L = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.temp_log_JP = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.temp_color = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.temp_material = ti.field(dtype=ti.int32, shape=self.max_particles)
        self.temp_radius = ti.field(dtype=ti.f32, shape=self.max_particles)

        # ===Dynamic Bound
        self.use_dynamic_grid = False
        self.grid_min = ti.field(dtype=ti.i32, shape=self.dim)
        self.grid_max = ti.field(dtype=ti.i32, shape=self.dim)

        # ===Debug
        self.D_trace = ti.field(dtype=ti.f32, shape=())

    # region === Obstacles ===

    def add_ball_obstacles(self, center, radius, color):
        if self.num_obstacles[None] >= self.max_num_obstacles:
            return
        idx = self.num_obstacles[None]
        self.obstacles[idx].type = 0
        idx = self.num_obstacles[None]
        self.obstacles[idx].center = center
        self.obstacles[idx].radius = radius
        self.obstacles[idx].color = color
        self.num_obstacles[None] += 1

    def add_box_obstacles(self, center, size, color):
        if self.num_obstacles[None] >= self.max_num_obstacles:
            return
        idx = self.num_obstacles[None]
        self.obstacles[idx].type = 1
        self.obstacles[idx].center = center
        self.obstacles[idx].size = size
        self.obstacles[idx].color = color
        self.num_obstacles[None] += 1

    def move_obstacle(self, idx, offset):
        if idx < self.num_obstacles[None]:
            for d in range(self.dim):
                center = self.obstacles[idx].center[d]
                half_size = self.obstacles[idx].size[d] / 2
                if center + half_size + offset[d] >= (self.n_grid - self.bound) * self.dx:
                    offset[d] = (self.n_grid - self.bound) * self.dx - center - half_size
                elif center - half_size + offset[d] <= self.bound * self.dx:
                    offset[d] = self.bound * self.dx + half_size - center
                self.obstacles[idx].center[d] += offset[d]

    def prepare_render_data(self):
        num_obs = self.num_obstacles[None]
        if num_obs == 0:
            return

        all_local_vertices = []
        all_indices = []
        all_colors = []
        all_owner_ids = []

        index_offset = 0

        for i in range(num_obs):
            obs_type = self.obstacles[i].type
            center = self.obstacles[i].center.to_numpy()
            color = self.obstacles[i].color.to_numpy()

            verts = None
            inds = None
            if obs_type == 0:  # ball
                verts, inds = get_sphere_mesh()
            elif obs_type == 1:  # box
                verts, inds = get_unit_cube_mesh()

            if verts is not None:
                all_local_vertices.append(verts)
                all_indices.append(inds + index_offset)

                c = np.tile(color, (len(verts), 1))
                all_colors.append(c)

                ids = np.full(len(verts), i, dtype=np.int32)
                all_owner_ids.append(ids)

                index_offset += len(verts)

        if not all_local_vertices:
            return

        final_local_verts = np.vstack(all_local_vertices)
        final_inds = np.concatenate(all_indices)
        final_colors = np.vstack(all_colors)
        final_owner_ids = np.concatenate(all_owner_ids)

        # 初始化 Fields
        num_verts = len(final_local_verts)
        self.mesh_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)
        self.mesh_local_pos = ti.Vector.field(3, dtype=ti.f32, shape=num_verts)  # 新增
        self.mesh_owner_id = ti.field(dtype=ti.i32, shape=num_verts)  # 新增

        self.mesh_indices = ti.field(dtype=ti.i32, shape=len(final_inds))
        self.mesh_colors = ti.Vector.field(3, dtype=ti.f32, shape=len(final_colors))

        # 传输数据
        self.mesh_indices.from_numpy(final_inds)
        self.mesh_colors.from_numpy(final_colors)
        self.mesh_local_pos.from_numpy(final_local_verts)  # 传输局部坐标
        self.mesh_owner_id.from_numpy(final_owner_ids)  # 传输ID

        # 初始化一次世界坐标
        self.update_mesh_vertices()

    @ti.kernel
    def update_mesh_vertices(self):
        for i in range(self.mesh_vertices.shape[0]):
            oid = self.mesh_owner_id[i]

            type = self.obstacles[oid].type
            center = self.obstacles[oid].center
            local_pos = self.mesh_local_pos[i]

            if type == 0:  # Ball
                radius = self.obstacles[oid].radius
                self.mesh_vertices[i] = center + local_pos * radius

            elif type == 1:  # Box
                size = self.obstacles[oid].size
                self.mesh_vertices[i] = center + ti.Vector(
                    [local_pos[0] * size[0], local_pos[1] * size[1], local_pos[2] * size[2]]
                )

    @ti.func
    def collide(self, pos, shape):
        is_collide = False
        internal_dist = 0.0
        normal = ti.Vector([0.0, 0.0, 0.0])
        point_on_surface = pos

        center = shape.center

        if shape.type == 0:
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

        elif shape.type == 1:
            half_size = shape.size / 2
            local_pos = pos - center
            q = ti.abs(local_pos) - half_size
            external_dist = ti.Vector(
                [ti.max(q[0], 0.0), ti.max(q[1], 0.0), ti.max(q[2], 0.0)]
            ).norm()  # 找到外部的最近距离
            internal_dist = ti.min(ti.max(q[0], ti.max(q[1], q[2])), 0.0)  # 找到内部的最近距离
            dist = external_dist + internal_dist
            is_collide = dist < 0
            normal = ti.Vector([0.0, 0.0, 0.0])
            point_on_surface = pos
            if is_collide:
                # 找到由于哪个轴穿透最浅，离表面最近，法线朝外
                if q[0] > q[1] and q[0] > q[2]:
                    normal[0] = 1.0 if local_pos[0] > 0 else -1.0
                elif q[1] > q[2]:
                    normal[1] = 1.0 if local_pos[1] > 0 else -1.0
                else:
                    normal[2] = 1.0 if local_pos[2] > 0 else -1.0
            point_on_surface = pos - dist * normal
        return is_collide, -internal_dist, -normal, point_on_surface

    # endregion

    # region === Material ===
    def init_material_params(self):
        # 0: Fluid
        self.mat_params[0].rho = 1.0
        self.mat_params[0].viscosity = 0.0
        self.mat_params[0].stiffness = 0.8

        # 1: Elastic
        self.mat_params[1].E = 100000

        # 2. Sand
        self.mat_params[2].beta = 1.0
        self.mat_params[2].elastic_relaxation = 1.5
        self.mat_params[2].friction_angle = 35.0

    # endregion

    # region === Morton Code ===

    @ti.kernel
    def sort_init(self):
        for p in range(self.max_particles):
            if p < self.n_particles[None]:
                self.particle_sort_keys[p] = get_morton_code(self.x[p], self.dx, self.n_grid)
            else:
                self.particle_sort_keys[p] = 2147483647
            self.particle_sort_indices[p] = p

    @ti.func
    def copy_particle_to_temp(self, i):
        old_idx = self.particle_sort_indices[i]
        self.temp_x[i] = self.x[old_idx]
        self.temp_dis[i] = self.dis[old_idx]
        self.temp_D[i] = self.D[old_idx]
        self.temp_F[i] = self.F[old_idx]
        self.temp_L[i] = self.L[old_idx]
        self.temp_log_JP[i] = self.log_JP[old_idx]
        self.temp_color[i] = self.color[old_idx]
        self.temp_material[i] = self.material[old_idx]
        self.temp_radius[i] = self.radius[old_idx]

    @ti.func
    def copy_temp_to_particle(self, i):
        self.x[i] = self.temp_x[i]
        self.dis[i] = self.temp_dis[i]
        self.D[i] = self.temp_D[i]
        self.F[i] = self.temp_F[i]
        self.L[i] = self.temp_L[i]
        self.log_JP[i] = self.temp_log_JP[i]
        self.color[i] = self.temp_color[i]
        self.material[i] = self.temp_material[i]
        self.radius[i] = self.temp_radius[i]
        self.particle_sort_indices[i] = i

    @ti.kernel
    def copy_data(self):
        for i in range(self.n_particles[None]):
            self.copy_particle_to_temp(i)

        for i in range(self.n_particles[None]):
            self.copy_temp_to_particle(i)

    @ti.func
    def swap_particle_key(self, i, j):
        self.particle_sort_indices[i], self.particle_sort_indices[j] = (
            self.particle_sort_indices[j],
            self.particle_sort_indices[i],
        )
        self.particle_sort_keys[i], self.particle_sort_keys[j] = (
            self.particle_sort_keys[j],
            self.particle_sort_keys[i],
        )

    @ti.func
    def incremental_sort_step(self, p: int, i: int, gap: int):
        block_idx = p // gap
        phase = self.fps_count[None] % 2
        if block_idx % 2 == phase:
            q = p + gap
            if q < self.n_particles[None]:
                key_i = self.particle_sort_keys[p]
                key_j = self.particle_sort_keys[q]
                if key_i > key_j:
                    self.swap_particle_key(p, q)

    def generate_gaps(self):
        shrink_factor = self.shrink_factor
        temp_gaps = []
        current_gap = 1
        for _ in range(self.iteration):
            temp_gaps.append(int(current_gap))
            current_gap = max(current_gap * shrink_factor, current_gap + 1)
        temp_gaps = temp_gaps[::-1]
        for i in range(len(temp_gaps)):
            self.gaps[i] = temp_gaps[i]

    # endregion

    # region === init ===
    @ti.kernel
    def add_particles(
        self,
        new_particles_num: int,
        new_particles_positions: ti.types.ndarray(),
        new_particles_velocities: ti.types.ndarray(),
        new_particles_material: ti.types.ndarray(),
        new_particles_color: ti.types.ndarray(),
        new_particles_radius: ti.types.ndarray(),
    ):
        for p in range(self.n_particles[None], self.n_particles[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocities[p - self.n_particles[None], d]
                x[d] = new_particles_positions[p - self.n_particles[None], d]
            color = ti.Vector.zero(float, 3)
            for d in range(3):
                color[d] = new_particles_color[p - self.n_particles[None], d]
            self.x[p] = x
            self.dis[p] = v * self.dt
            self.color[p] = color
            self.material[p] = new_particles_material[p - self.n_particles[None]]
            self.radius[p] = new_particles_radius[p - self.n_particles[None]]
            self.D[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            if self.material[p] == 0:
                self.L[p] = 1.0
            elif self.material[p] == 1 or self.material[p] == 2:
                self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
        self.n_particles[None] += new_particles_num

    def add_cube(
        self,
        particle_num,
        center,
        cube_size,
        velocities=[0, 0, 0],
        color=[1.0, 1.0, 1.0],
        material=0,
        radius=0.01,
    ):
        assert self.n_particles[None] + particle_num < self.max_particles
        center = np.asarray(center, dtype=np.float32)
        cube_size = np.asarray(cube_size, dtype=np.float32)
        start = center - cube_size / 2
        end = center + cube_size / 2
        start = [max(self.bound * self.dx, start[d]) for d in range(self.dim)]
        end = [min((self.n_grid - self.bound) * self.dx, end[d]) for d in range(self.dim)]
        new_position = np.random.uniform(low=start, high=end, size=(particle_num, self.dim))
        new_velocity = np.tile(np.array(velocities, dtype=np.float32), (particle_num, 1))
        new_color = np.tile(np.array(color, dtype=np.float32), (particle_num, 1))
        new_material = np.full(particle_num, material, dtype=np.int32)
        new_radius = np.full(particle_num, radius, dtype=np.float32)
        self.add_particles(
            new_particles_num=particle_num,
            new_particles_positions=new_position,
            new_particles_velocities=new_velocity,
            new_particles_material=new_material,
            new_particles_color=new_color,
            new_particles_radius=new_radius,
        )

    def init(self, hide_obstacles):
        self.init_material_params()
        if not hide_obstacles:
            self.prepare_render_data()
        if self.use_morton_code:
            self.generate_gaps()
        print(self.gaps)

        print(self.n_particles[None])

    def add_continum_particle(self, material, center=[0.1, 0.9, 0.2], size=[0.1, 0.1, 0.1], num=8):
        if self.n_particles[None] + num >= self.max_particles:
            return
        self.add_cube(
            particle_num=num,
            center=center,
            cube_size=size,
            velocities=[0, -2.0, 0],  # 给一个向下的初速度
            color=[0.1, 0.4, 0.8],
            material=material,
            radius=0.006,
        )

    # endregion

    # region === MPM ===
    @ti.func
    def get_active_bounds(self):
        min_x = 0
        max_x = self.n_grid
        min_y = 0
        max_y = self.n_grid
        min_z = 0
        max_z = self.n_grid
        if self.use_dynamic_grid:
            padding = 3
            min_x = ti.max(0, self.grid_min[0] - padding)
            max_x = ti.min(self.n_grid, self.grid_max[0] + padding)
            min_y = ti.max(0, self.grid_min[1] - padding)
            max_y = ti.min(self.n_grid, self.grid_max[1] + padding)
            min_z = ti.max(0, self.grid_min[2] - padding)
            max_z = ti.min(self.n_grid, self.grid_max[2] + padding)
        return min_x, max_x, min_y, max_y, min_z, max_z

    @ti.kernel
    def solve_constraint(self):
        for p in range(self.n_particles[None]):
            if self.material[p] == 0:  # fluid
                # (A) 黏度约束
                deviatoric = -1.0 * (self.D[p] + self.D[p].transpose())
                self.D[p] += self.mat_params[0].viscosity * 0.5 * deviatoric

                # (B) 体积/压力约束
                current_trace = self.D[p].trace()
                safe_L = ti.max(self.L[p], 0.1)
                alpha = (1.0 / 3.0) * (1.0 / safe_L - current_trace - 1.0)
                self.D[p] += self.mat_params[0].stiffness * alpha * ti.Matrix.identity(ti.f32, self.dim)

            elif self.material[p] == 1:  # elastic
                I = ti.Matrix.identity(ti.f32, self.dim)
                F_star = (I + self.D[p]) @ self.F[p]
                U, sig, V = ti.svd(F_star)
                new_sig = ti.Matrix.identity(ti.f32, self.dim)
                for d in range(self.dim):
                    new_sig[d, d] = ti.max(0.1, ti.min(sig[d, d], 10000))
                F_star = U @ new_sig @ V.transpose()
                A_shape = U @ V.transpose()

                tgt = A_shape

                U_old, sig_old, V_old = ti.svd(self.F[p])
                inv_sig = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                for d in range(self.dim):
                    inv_sig[d, d] = 1.0 / ti.max(sig_old[d, d], 0.1)
                F_inv = V_old @ inv_sig @ U_old.transpose()

                D_target = tgt @ F_inv - I
                diff = D_target - self.D[p]

                # diff = (tgt @ F_inv - I) - self.D[p]
                # XPBD
                stiffness_E = self.mat_params[1].E
                alpha = 1.0 / (stiffness_E + 1e-6)
                tilde_alpha = alpha / (self.dt**2)
                delta_lambda = (diff - tilde_alpha * self.lambdas[p]) / (1.0 + tilde_alpha)

                self.D[p] += delta_lambda
                self.lambdas[p] += delta_lambda

            elif self.material[p] == 2:  # sand
                I = ti.Matrix.identity(ti.f32, self.dim)
                F_star = (I + self.D[p]) @ self.F[p]
                U, sig, V = ti.svd(F_star)
                for d in range(self.dim):
                    sig[d, d] = ti.max(1.0, ti.min(sig[d, d], 1000))
                A_shape = U @ sig @ V.transpose()

                det_F = F_star.determinant()
                det_F_clamped = ti.max(0.1, ti.min(det_F, 1.0))
                A_vol = F_star * 1.0 / ti.pow(det_F_clamped, 1.0 / 3.0)

                beta = self.mat_params[2].beta
                tgt = beta * A_shape + (1 - beta) * A_vol
                diff = (tgt @ self.F[p].inverse() - I) - self.D[p]
                elastic_relaxtion = self.mat_params[1].elastic_relaxation
                self.D[p] += elastic_relaxtion * diff

                alpha_visc = 0.1
                deviatoric = -1.0 * (self.D[p] + self.D[p].transpose())
                self.D[p] += alpha_visc * 0.5 * deviatoric

    @ti.func
    def P2G(self, p):
        n = self.n_particles[None]
        if self.use_morton_code:
            multiplier = 1000003
            p = (p * multiplier) % n
            p = (p + (p % 8) * (n // 8)) % n
        else:
            p = p
        Xp = self.x[p] / self.dx
        base = int(Xp - 0.5)  # 向下取整
        fx = Xp - base
        w = [
            0.5 * (1.5 - fx) ** 2,  # 对应网格节点i1
            0.75 - (fx - 1) ** 2,  # 对应网格节点i+1
            0.5 * (fx - 0.5) ** 2,  # 对应网格节点i+2
        ]

        for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
            weight = 1.0
            dpos = (offset - fx) * self.dx
            weight *= w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
            momentum = weight * (self.dis[p] + self.D[p] @ dpos)
            self.grid_dis[base + offset] += momentum
            self.grid_m[base + offset] += weight
            if self.material[p] == 0:
                self.grid_vol[base + offset] += weight * self.p_vol

    @ti.func
    def update_grid(self, I):
        if self.grid_m[I] > 1e-6:
            # 应用重力
            self.grid_dis[I] /= self.grid_m[I]
            # gravity_impulse = ti.Vector([0.0, -self.gravity, 0.0]) * self.dt * self.dt
            # self.grid_dis[I] += gravity_impulse

            # 处理障碍物碰撞
            grid_pos = ti.Vector([I[0], I[1], I[2]]) * self.dx
            grid_disp = self.grid_dis[I]
            for i in range(self.num_obstacles[None]):
                predict_pos = grid_disp + grid_pos
                is_collide, _, normal_in, _ = self.collide(predict_pos, self.obstacles[i])
                if is_collide:
                    v_proj = grid_disp.dot(normal_in)
                    if v_proj > 0:
                        grid_disp -= v_proj * normal_in
            self.grid_dis[I] = grid_disp

            # TODO: 为了沙子堆积添加了摩擦力，感觉需要把摩擦力移动到其他地方
            boundary_friction = 0.0
            damping = 1.0 - boundary_friction

            # 处理边界
            for d in ti.static(range(self.dim)):
                if I[d] < self.bound and self.grid_dis[I][d] < 0:
                    self.grid_dis[I][d] = 0
                    self.grid_dis[I] *= damping
                if I[d] > self.n_grid - self.bound and self.grid_dis[I][d] > 0:
                    self.grid_dis[I][d] = 0
                    self.grid_dis[I] *= damping
        else:
            self.grid_dis[I] = ti.Vector.zero(ti.f32, self.dim)

    @ti.func
    def G2P(self, p):
        Xp = self.x[p] / self.dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        new_dis = ti.zero(self.dis[p])
        new_D = ti.zero(self.D[p])

        gathered_vol = 0.0

        for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
            dpos = (offset - fx) * self.dx
            weight = 1.0
            for i in ti.static(range(self.dim)):
                weight *= w[offset[i]][i]
            g_dis = self.grid_dis[base + offset]
            new_dis += weight * g_dis
            new_D += weight * (4 * g_dis.outer_product(dpos) / self.dx**2)
            if self.material[p] == 0:
                gathered_vol += weight * self.grid_vol[base + offset]
        if self.material[p] == 0:
            J = 1.0 / gathered_vol
            if J < 1.0:
                self.L[p] = 0.9 * self.L[p] + 0.1 * J

        self.dis[p] = new_dis
        self.D[p] = new_D

    @ti.kernel
    def compute_average_height(self):
        self.average_height[None] = 0.0
        total_num = 0
        for p in range(self.n_particles[None]):
            if self.material[p] == 0:
                self.average_height[None] += self.x[p].y
                total_num += 1
        self.average_height[None] /= total_num

    @ti.func
    def compute_water_color(self, p, color_option=0):
        # Compute Color of The Water According to Speed
        if color_option == 0:
            speed = self.dis[p].norm() / self.dt
            color_deep = ti.Vector([0.05, 0.1, 0.35])
            color_surface = ti.Vector([0.3, 0.7, 0.9])
            color_foam = ti.Vector([1.0, 1.0, 1.0])
            pos_y = self.x[p].y
            bottom_y = self.bound * self.dx
            surface_y = self.average_height[None] * 1.5
            t_depth = ti.math.clamp((pos_y - bottom_y) / (surface_y - bottom_y), 0.0, 1.0)
            t_depth_smooth = ti.math.smoothstep(0.0, 1.0, t_depth)
            base_color = ti.math.mix(color_deep, color_surface, t_depth_smooth)
            foan_threshold = 1.0
            t_foam = ti.math.clamp((speed - foan_threshold) / 3.0, 0.0, 1.0)
            self.color[p] = ti.math.mix(base_color, color_foam, t_foam)
        elif color_option == 1:
            val = p / self.n_particles[None]
            self.color[p] = ti.Vector([val, 1.0 - val, 0.5 * ti.sin(val * 10)])

    @ti.func
    def compute_F(self, p):
        if self.material[p] == 0:  # fluid
            # Update Density
            self.L[p] *= self.D[p].trace() + 1
            self.L[p] = ti.max(self.L[p], 0.05)
            self.compute_water_color(p, 1)

        elif self.material[p] == 1:  # elastic
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.D[p]) @ self.F[p]
            U, sig, V = ti.svd(self.F[p])
            new_sig = ti.Matrix.identity(ti.f32, self.dim)
            for d in range(self.dim):
                new_sig[d, d] = ti.max(0.1, ti.min(sig[d, d], 100000))
            self.F[p] = U @ new_sig @ V.transpose()
        elif self.material[p] == 2:
            I = ti.Matrix.identity(ti.f32, self.dim)
            self.F[p] = (I + self.D[p]) @ self.F[p]
            U, sig, V = ti.svd(self.F[p])

            # === Drucker-Prager ===
            sin_phi = ti.sin(self.mat_params[2].friction_angle * 3.1415926 / 180.0)
            alpha = ti.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
            beta = 0.5

            epsilon = ti.Vector([0.0, 0.0, 0.0])
            for d in range(self.dim):
                epsilon[d] = ti.log(ti.max(ti.abs(sig[d, d]), 1e-6))
            trace_epsilon = epsilon.sum() + self.log_JP[p]
            epsilon_hat = epsilon - trace_epsilon / 3.0
            frob_norm = epsilon_hat.norm()
            epsilon_new = ti.Vector([0.0, 0.0, 0.0])
            if trace_epsilon >= 0:
                epsilon_new = ti.Vector([0.0, 0.0, 0.0])
                self.log_JP[p] = beta * trace_epsilon
            else:
                self.log_JP[p] = 0.0
                delta_gamma = frob_norm + (self.mat_params[2].beta + 1.0) * trace_epsilon * alpha
                if delta_gamma > 0:
                    ratio = delta_gamma / frob_norm
                    epsilon_new = epsilon - ratio * epsilon_hat
                else:
                    epsilon_new = epsilon

            sig_final = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            for d in range(self.dim):
                sig_final[d, d] = ti.exp(epsilon_new[d])
            self.F[p] = U @ sig_final @ V.transpose()

    @ti.func
    def update_position(self, p):
        self.compute_F(p)

        self.x[p] += self.dis[p]

        gravity_impulse = ti.Vector([0.0, -self.gravity, 0.0]) * self.dt * self.dt
        self.dis[p] += gravity_impulse

        # SDF碰撞检测
        for i in range(self.num_obstacles[None]):
            is_collide, peneration, normal, point_on_surface = self.collide(self.x[p], self.obstacles[i])
            if is_collide:
                self.dis[p] -= peneration * normal

        # 边界限制
        padding = (self.bound - 1) * self.dx - 1e-5
        for d in ti.static(range(self.dim)):
            if self.x[p][d] < padding:
                self.x[p][d] = padding
            if self.x[p][d] > 1:
                self.x[p][d] = 1

    @ti.kernel
    def mpm_solve(self):
        for I in ti.grouped(self.grid_m):
            self.grid_dis[I] = ti.zero(self.grid_dis[I])
            self.grid_m[I] = 0.0
            self.grid_vol[I] = 0.0

        ti.loop_config(parallelize=8, block_dim=128)
        for p in range(self.n_particles[None]):
            self.P2G(p)

        for I in ti.grouped(self.grid_m):
            self.update_grid(I)

        for p in range(self.n_particles[None]):
            self.G2P(p)

    @ti.kernel
    def update_particle(self):
        for p in range(self.n_particles[None]):
            self.update_position(p)

    @ti.kernel
    def damp_lambdas(self, factor: ti.f32):
        for p in range(self.n_particles[None]):
            self.lambdas[p] *= factor

    def substep(self):
        self.fps_count[None] += 1
        self.damp_lambdas(0.8)
        self.D.fill(0)
        for _ in range(self.iteration):
            self.solve_constraint()
            self.D_trace[None] = self.F[0][0, 1]
            self.mpm_solve()
        self.update_particle()

    # endregion

    # region === Utils ===
    # @ti.kernel
    def debug_probe(self, gui):

        p = 0
        F_ti = self.F[p]
        F_np = np.array([[F_ti[i, j] for j in range(3)] for i in range(3)])
        U, sig, Vh = np.linalg.svd(F_np)
        D = self.D[p]
        pos = self.x[p]
        gui.text(f"=== Debug P[0] ===")
        gui.text(f"  Pos Y: {pos.y:.4f}")
        gui.text(f"  F_yy : {F_ti[1, 1]:.4f} (如果接近0说明压扁了)")
        gui.text(f"  Sig  : {sig[0]:.3f}, {sig[1]:.3f}, {sig[2]:.3f} (奇异值)")
        gui.text(f"  D_yy : {D[1, 1]:.4f} (当前帧的形变位移)")

    # @ti.kernel
    def generate_lines_vertex(self):
        idx = 0
        n = self.n_grid
        for i in range(n + 1):
            self.grid_lines_vertex[2 * idx + 0] = ti.Vector([0, i, 0]) * self.dx
            self.grid_lines_vertex[2 * idx + 1] = ti.Vector([n, i, 0]) * self.dx
            idx += 1
            self.grid_lines_vertex[2 * idx + 0] = ti.Vector([0, i, 0]) * self.dx
            self.grid_lines_vertex[2 * idx + 1] = ti.Vector([0, i, n]) * self.dx
            idx += 1

            self.grid_lines_vertex[2 * idx + 0] = ti.Vector([i, 0, 0]) * self.dx
            self.grid_lines_vertex[2 * idx + 1] = ti.Vector([i, n, 0]) * self.dx
            idx += 1
            self.grid_lines_vertex[2 * idx + 0] = ti.Vector([i, 0, 0]) * self.dx
            self.grid_lines_vertex[2 * idx + 1] = ti.Vector([i, 0, n]) * self.dx
            idx += 1

            self.grid_lines_vertex[2 * idx + 0] = ti.Vector([0, 0, i]) * self.dx
            self.grid_lines_vertex[2 * idx + 1] = ti.Vector([0, n, i]) * self.dx
            idx += 1
            self.grid_lines_vertex[2 * idx + 0] = ti.Vector([0, 0, i]) * self.dx
            self.grid_lines_vertex[2 * idx + 1] = ti.Vector([n, 0, i]) * self.dx
            idx += 1

    @ti.kernel
    def apply_interia(self, interia_force: ti.types.vector(3, float)):
        self.interia_force[None] = interia_force

    def reset(self):
        self.n_particles[None] = 0
        self.fps_count[None] = 0
        self.average_height[None] = 0.0
        self.interia_force[None] = [0.0, 0.0, 0.0]

        self.num_obstacles[None] = 0

        self.grid_m.fill(0)
        self.grid_dis.fill(0)
        self.grid_vol.fill(0)

        self.x.fill(0)
        self.dis.fill(0)
        self.D.fill(0)
        self.L.fill(1.0)  # 密度默认为 1
        self.log_JP.fill(0)

        self.reset_F_to_identity()

    @ti.kernel
    def reset_F_to_identity(self):
        for i in range(self.max_particles):
            self.F[i] = ti.Matrix.identity(ti.f32, self.dim)

    # endregion
