import taichi as ti
import numpy as np
from utils_renderer import get_unit_cube_mesh, get_sphere_mesh
from morton_code import get_morton_code
from pbmpm_types import MaterialParam, Obstacle
from pbmpm_collision import sdf_box, collide as collide_sdf
from typing import Any

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
            self.obstacles[idx].velocity = ti.Vector(np.array(offset, dtype=np.float32) / self.dt)

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
        return collide_sdf(pos, shape)

    # endregion

    # region === Material ===
    def init_material_params(self):
        # 0: Fluid
        self.mat_params[0].rho = 1.0
        self.mat_params[0].viscosity = 0.0
        self.mat_params[0].stiffness = 0.8

        # 1: Elastic
        self.mat_params[1].E = 50000

        # 2. Sand
        self.mat_params[2].beta = 1.0
        self.mat_params[2].elastic_relaxation = 1.5
        self.mat_params[2].friction_angle = 35.0

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
                # self.D[p] += self.mat_params[0].stiffness * alpha * ti.Matrix.identity(ti.f32, self.dim)
                self.D[p] += 1.5 * alpha * ti.Matrix.identity(ti.f32, self.dim)

                # bulk_modulus = self.mat_params[0].stiffness
                # # compliance = 1.0 / (bulk_modulus * self.dt**2 + 1e-6)
                # compliance = 0.0
                # current_trace = self.D[p].trace()
                # safe_L = ti.max(self.L[p], 0.1)  # 密度比
                # C = (1.0 / safe_L - 1.0) - current_trace
                # w_i = 3.0
                # old_lambda = self.lambdas[p][0, 0]
                # delta_lambda = (C - compliance * old_lambda) / (w_i + compliance)
                # new_lambda = old_lambda + delta_lambda
                # if new_lambda < 0:
                #     new_lambda = 0.0
                #     delta_lambda = new_lambda - old_lambda
                # self.lambdas[p][0, 0] = new_lambda
                # self.D[p] += delta_lambda * ti.Matrix.identity(ti.f32, self.dim)

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

            # 处理障碍物碰撞
            grid_pos = ti.Vector([I[0], I[1], I[2]]) * self.dx
            grid_disp = self.grid_dis[I]
            for i in range(self.num_obstacles[None]):
                obs_vel = self.obstacles[i].velocity
                obs_disp = obs_vel * self.dt

                predict_pos = grid_disp + grid_pos
                is_collide, _, normal_in, _ = self.collide(predict_pos, self.obstacles[i])
                if is_collide:
                    rel_disp = grid_disp - obs_disp
                    v_proj = rel_disp.dot(normal_in)
                    if v_proj > 0:
                        grid_disp -= v_proj * normal_in
                        grid_disp *= 0.95
                        grid_disp = rel_disp + obs_disp
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
            beta = 0.9
            if J < 1.0:
                self.L[p] = beta * self.L[p] + (1 - beta) * J

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
            self.L[p] = ti.max(self.L[p], 0.2)
            self.compute_water_color(p, 0)

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
                self.dis[p] *= 0.95

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
        self.average_height[None] = 0.0
        for p in range(self.n_particles[None]):
            self.lambdas[p] *= factor
            # 计算平均高度
            self.average_height[None] += self.x[p][1]
        self.average_height[None] /= self.n_particles[None]

    def substep(self):
        self.fps_count[None] += 1
        self.damp_lambdas(0.5)
        # self.D.fill(0)
        for _ in range(self.iteration):
            self.solve_constraint()
            self.D_trace[None] = self.F[0][0, 1]
            self.mpm_solve()
        self.update_particle()

    # endregion

    # region === Utils ===
    def debug_probe(self, gui):
        gui.text(f"average heighjt : {self.average_height[None]: .4f}")
        gui.text(f"obstacles velocity : {self.obstacles[0].velocity}")
        pass

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
        self.lambdas.fill(0)

        self.reset_F_to_identity()

    @ti.kernel
    def reset_F_to_identity(self):
        for i in range(self.max_particles):
            self.F[i] = ti.Matrix.identity(ti.f32, self.dim)

    # endregion
