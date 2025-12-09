import taichi as ti
import numpy as np
from utils_renderer import get_unit_cube_mesh, get_sphere_mesh

MaterialParam = ti.types.struct(
    rho=ti.f32,
    viscosity=ti.f32,
    stiffness=ti.f32,
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
        self.n_loop[None] = 0

        self.p_rho = 1
        self.p_vol = 1 / 2**3
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = 9.8
        self.bound = 3
        self.iteration = 10
        self.average_velocity = ti.field(dtype=ti.f32, shape=())
        self.average_height = ti.field(dtype=ti.f32, shape=())
        self.average_density_list = ti.field(dtype=ti.f32, shape=self.iteration)

        # ===Particle
        self.max_particles = 400000
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
        self.log_JP = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.color = ti.Vector.field(3, ti.f32, shape=self.max_particles)
        self.radius = ti.field(dtype=ti.f32, shape=self.max_particles)

        # ===Material
        self.num_materials = 3
        self.mat_params = MaterialParam.field(shape=(self.num_materials,))
        self.material = ti.field(dtype=ti.int32, shape=self.max_particles)  # 0：fluid，1: jelly, 2: snow

        # ===Grid
        self.n_grid = 48
        self.dx = 1 / self.n_grid
        self.grid_v = ti.Vector.field(self.dim, dtype=ti.f32)
        self.grid_dis = ti.Vector.field(self.dim, dtype=ti.f32)
        self.grid_m = ti.field(dtype=ti.f32)
        self.grid_vol = ti.field(dtype=ti.f32)
        # Sparse Grid
        block_size = 8
        n_memory_size = self.n_grid * 2
        self.grid_snode = ti.root.pointer(ti.ijk, n_memory_size // block_size)
        self.pixel = self.grid_snode.dense(ti.ijk, block_size)
        self.pixel.place(self.grid_v, self.grid_dis, self.grid_m, self.grid_vol)

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
        self.mesh_vertices = None
        self.mesh_indices = None
        self.mesh_colors = None

        # ===Debug
        self.num_grid_lines = 3 * (2 * (self.n_grid + 1) + 1)
        self.grid_lines_vertex = ti.Vector.field(self.dim, dtype=ti.float32, shape=self.num_grid_lines * 2)

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
        self.mat_params[0].stiffness = 0.5

        # 1: Elastic
        self.mat_params[1].beta = 0.5
        self.mat_params[1].elastic_relaxation = 1.0

        # 2. Sand
        self.mat_params[2].beta = 1.0
        self.mat_params[2].elastic_relaxation = 1.5
        self.mat_params[2].friction_angle = 35.0

    # endregion

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

    # region === MPM ===
    @ti.func
    def solve_constraint(self, p):
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

            det_F = F_star.determinant()
            det_F_clamped = ti.max(0.1, ti.min(det_F, 1000))
            A_vol = F_star * 1.0 / ti.pow(det_F_clamped, 1.0 / 3.0)

            beta = self.mat_params[1].beta
            elastic_relaxation = self.mat_params[1].elastic_relaxation
            tgt = beta * A_shape + (1 - beta) * A_vol
            diff = (tgt @ self.F[p].inverse() - I) - self.D[p]
            self.D[p] += elastic_relaxation * diff

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
            for i in ti.static(range(self.dim)):
                weight *= w[offset[i]][i]
            momentum = weight * (self.dis[p] + self.D[p] @ dpos)
            self.grid_dis[base + offset] += momentum
            self.grid_m[base + offset] += weight
            if self.material[p] == 0:
                self.grid_vol[base + offset] += weight * self.p_vol

    @ti.func
    def update_grid(self, I):
        # 时间步进，网格更新
        if self.grid_m[I] > 1e-6:
            self.grid_dis[I] /= self.grid_m[I]
            grid_pos = ti.Vector([I[0], I[1], I[2]]) * self.dx
            grid_disp = self.grid_dis[I]
            for i in range(self.num_obstacles[None]):
                predict_pos = grid_disp + grid_pos
                is_collide, _, normal_in, point_on_surface = self.collide(predict_pos, self.obstacles[i])
                if is_collide:
                    v_proj = grid_disp.dot(normal_in)
                    if v_proj > 0:
                        grid_disp -= v_proj * normal_in
            self.grid_dis[I] = grid_disp

            # TODO: 为了沙子堆积添加了摩擦力，感觉需要把摩擦力移动到其他地方
            boundary_friction = 0.0
            damping = 1.0 - boundary_friction
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
            ratio = 0.9
            if J < 1.0:
                self.L[p] = ratio * self.L[p] + (1 - ratio) * J

        self.dis[p] = new_dis
        self.D[p] = new_D
        # self.solve_constraints(p)

    @ti.func
    def update_particles(self, p):
        if self.material[p] == 0:  # fluid
            # Update Density
            self.L[p] *= self.D[p].trace() + 1
            self.L[p] = ti.max(self.L[p], 0.05)

            # Compute Color of The Water According to Speed
            speed = self.dis[p].norm() / self.dt
            color_deep = ti.Vector([0.05, 0.1, 0.35])
            color_surface = ti.Vector([0.3, 0.7, 0.9])
            color_foam = ti.Vector([1.0, 1.0, 1.0])
            pos_y = self.x[p].y
            bottom_y = self.bound * self.dx
            surface_y = self.average_height[None] * 2
            t_depth = ti.math.clamp((pos_y - bottom_y) / (surface_y - bottom_y), 0.0, 1.0)
            t_depth_smooth = ti.math.smoothstep(0.0, 1.0, t_depth)
            base_color = ti.math.mix(color_deep, color_surface, t_depth_smooth)
            foan_threshold = 1.0
            t_foam = ti.math.clamp((speed - foan_threshold) / 3.0, 0.0, 1.0)
            self.color[p] = ti.math.mix(base_color, color_foam, t_foam)
        elif self.material[p] == 1:  # elastic
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.D[p]) @ self.F[p]
            U, sig, V = ti.svd(self.F[p])
            new_sig = ti.Matrix.identity(ti.f32, self.dim)
            for d in range(self.dim):
                new_sig[d, d] = ti.max(0.1, ti.min(sig[d, d], 10000))
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
    def solve_iteration(self):
        for p in range(self.n_particles[None]):
            self.P2G(p)

        for I in ti.grouped(self.grid_m):
            self.update_grid(I)

        for p in range(self.n_particles[None]):
            self.average_height[None] += self.x[p][1]
        self.average_height[None] /= self.n_particles[None]

        for p in range(self.n_particles[None]):
            self.G2P(p)
            if self.n_loop[None] == self.iteration - 1:
                self.update_particles(p)
            self.solve_constraint(p)

    def substep(self, gui):
        for _ in range(self.iteration):
            self.solve_iteration()
            self.compute_average_iteration()
            self.grid_snode.deactivate_all()
            self.n_loop[None] += 1
        self.compute_average_substep()
        self.print_substep(gui)

        self.n_loop[None] = 0

    # endregion

    # region === Utils ===
    @ti.kernel
    def compute_average_substep(self):
        self.average_velocity[None] = 0.0
        for p in range(self.n_particles[None]):
            self.average_velocity[None] += self.dis[p].norm()
        self.average_velocity[None] /= self.n_particles[None]

    @ti.kernel
    def compute_average_iteration(self):
        average_density = 0.0
        total_num = 0
        for p in range(self.n_particles[None]):
            if self.material[p] == 0:
                average_density += self.L[p]
                total_num += 1
        average_density /= total_num
        self.average_density_list[self.n_loop[None]] = self.D[0].trace()

    def print_substep(self, gui):
        gui.text(f"Particles: {self.n_particles[None]}")
        gui.text(f"Average Velocity: {self.average_velocity[None]}")
        for i in range(self.iteration):
            gui.text(f"{i}: {self.average_density_list[i]}")

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
        for p in range(self.n_particles[None]):
            if self.material[p] == 2:
                self.dis[p] += interia_force

    # endregion
