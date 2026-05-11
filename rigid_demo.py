"""
Rigid–MPM demo (v3): 参照 mpm_pbd.py 的 PBMPM 框架，加入迭代位置修正。

substep 主循环（对齐 mpm_pbd.py::substep）：

    predict_body()                       # 重力施加到 body_v (类比 update_position 末尾的 g*dt²)
    for _ in range(self.iteration):       # PBMPM 迭代位置修正
        regenerate_particles()           # 根据当前 (x_c, q, v_c, ω) 重生粒子 dis_p, D_p
        mpm_solve()                      # P2G → grid update → G2P
        project_to_body()                # delta = dis - dis_pred 聚合回 (v_c, ω)
    update_body()                        # 提交 x_c, q

迭代的物理意义：第 k 次迭代中，project_to_body 把网格地面的反推回收为 body_v 的修正。
下一次迭代时 regenerate 使用更新后的 body_v 重算 dis_pred，再让 MPM 处理残余穿透。
最终收敛到一个 body_v 既包含重力又不引起地面穿透的稳定状态——这正是 PBMPM "iterative
position correction" 的核心思想。
"""

import numpy as np
import taichi as ti

ti.init(arch=ti.cuda)


@ti.data_oriented
class RigidMpmSolver:
    def __init__(self, n_per_side: int = 16):
        # region === Sim params ===
        self.dim = 3
        self.dt = 2e-3
        self.n_grid = 64
        self.dx = 1.0 / self.n_grid
        self.inv_dx = float(self.n_grid)
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.bound = 5
        self.floor_y = self.bound * self.dx
        self.iteration = 5
        # endregion

        # region === Body DoF ===
        self.body_x = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.body_q = ti.Vector.field(4, dtype=ti.f32, shape=())  # (w, x, y, z)
        self.body_v = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.body_omega = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.body_mass = ti.field(dtype=ti.f32, shape=())
        self.body_I_inv_local = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        self.body_I_inv_world = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
        # endregion

        # region === Particles ===
        self.n_per_side = n_per_side
        self.n_particles = n_per_side**3
        self.r_local = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self.dis = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self.dis_pred = ti.Vector.field(3, dtype=ti.f32, shape=self.n_particles)
        self.D = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.n_particles)
        # endregion

        # region === Grid ===
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid,) * 3)
        self.grid_dis = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_grid,) * 3)
        self.body_min_y = ti.field(dtype=ti.f32, shape=())
        # endregion

    # region === Quaternion helpers ===
    @ti.func
    def quat_mul(self, a, b):
        return ti.Vector(
            [
                a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
            ]
        )

    @ti.func
    def quat_to_mat(self, q):
        w, xq, yq, zq = q[0], q[1], q[2], q[3]
        return ti.Matrix(
            [
                [1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * w), 2 * (xq * zq + yq * w)],
                [2 * (xq * yq + zq * w), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * w)],
                [2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)],
            ]
        )

    @ti.func
    def skew(self, v):
        return ti.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    # endregion

    # region === Init ===
    def sample_cube_regular(self, cube_size):
        coords = np.linspace(-0.5, 0.5, self.n_per_side + 2)[1:-1]
        gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
        pts = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=1) * cube_size
        rng = np.random.default_rng(0)
        jitter = (rng.random(pts.shape) - 0.5) * 0.05 * float(cube_size.min()) / self.n_per_side
        pts += jitter
        return pts.astype(np.float32)

    def compute_inertia_from_particles(self):
        """基于实际粒子分布算 I_local，避免随机采样的离散惯性与解析公式不一致"""
        r = self.r_local.to_numpy()
        m_p = self.body_mass[None] / self.n_particles
        r2 = (r * r).sum(axis=1)
        I = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                I[i, j] = (r2 * (i == j) - r[:, i] * r[:, j]).sum()
        I *= m_p
        self.body_I_inv_local.from_numpy(np.linalg.inv(I).astype(np.float32))

    @ti.kernel
    def init_body_kernel(self, center: ti.types.vector(3, ti.f32), omega0: ti.types.vector(3, ti.f32), mass: ti.f32):
        self.body_x[None] = center
        self.body_q[None] = ti.Vector([1.0, 0.0, 0.0, 0.0])
        self.body_v[None] = ti.Vector([0.0, 0.0, 0.0])
        self.body_omega[None] = omega0
        self.body_mass[None] = mass
        for p in range(self.n_particles):
            self.D[p] = ti.Matrix.zero(ti.f32, 3, 3)
            self.dis[p] = ti.Vector.zero(ti.f32, 3)
            self.dis_pred[p] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def update_world_inertia(self):
        R = self.quat_to_mat(self.body_q[None])
        self.body_I_inv_world[None] = R @ self.body_I_inv_local[None] @ R.transpose()

    def add_cube(self, center, cube_size, omega0=(0.0, 0.0, 0.0), mass=1.0):
        cs = np.asarray(cube_size, dtype=np.float32)
        pts = self.sample_cube_regular(cs)
        self.r_local.from_numpy(pts)
        self.init_body_kernel(
            ti.Vector(np.asarray(center, dtype=np.float32)),
            ti.Vector(np.asarray(omega0, dtype=np.float32)),
            float(mass),
        )
        self.compute_inertia_from_particles()
        self.update_world_inertia()

    # endregion

    # region === Substep stages ===
    @ti.kernel
    def predict_body(self):
        self.body_v[None] += self.gravity * self.dt
        # 无外扭矩，omega 不变；更新世界系下 I_inv
        R = self.quat_to_mat(self.body_q[None])
        self.body_I_inv_world[None] = R @ self.body_I_inv_local[None] @ R.transpose()

    @ti.kernel
    def regenerate_particles(self):
        R = self.quat_to_mat(self.body_q[None])
        xc = self.body_x[None]
        v = self.body_v[None]
        w = self.body_omega[None]
        # APIC 仿射项：刚体速度场梯度 = skew(omega)
        D_rigid = self.skew(w) * self.dt
        for p in range(self.n_particles):
            arm = R @ self.r_local[p]
            self.x[p] = xc + arm
            d = (v + w.cross(arm)) * self.dt
            self.dis[p] = d
            self.dis_pred[p] = d  # 保存预测值用于 delta 投影
            self.D[p] = D_rigid

    @ti.kernel
    def clear_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_dis[I] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def p2g(self):
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                self.grid_m[base + offset] += weight
                self.grid_dis[base + offset] += weight * (self.dis[p] + self.D[p] @ dpos)

    @ti.kernel
    def grid_update(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 1e-8:
                self.grid_dis[I] /= self.grid_m[I]
                # 地面 BC：包含 floor_y 这一层本身，避免 B-Spline 上半泄漏出下行位移
                if I[1] <= self.bound:
                    if self.grid_dis[I].y < 0:
                        self.grid_dis[I].y = 0.0
                    self.grid_dis[I].x *= 0.95
                    self.grid_dis[I].z *= 0.95
                # 侧/顶边界
                for d in ti.static(range(3)):
                    if I[d] < self.bound and self.grid_dis[I][d] < 0:
                        self.grid_dis[I][d] = 0
                    if I[d] > self.n_grid - self.bound and self.grid_dis[I][d] > 0:
                        self.grid_dis[I][d] = 0

    @ti.kernel
    def g2p(self):
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - ti.cast(base, ti.f32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_dis = ti.Vector.zero(ti.f32, 3)
            new_D = ti.Matrix.zero(ti.f32, 3, 3)
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = w[offset[0]][0] * w[offset[1]][1] * w[offset[2]][2]
                gd = self.grid_dis[base + offset]
                new_dis += weight * gd
                new_D += weight * (4.0 * self.inv_dx * self.inv_dx) * gd.outer_product(dpos)
            self.dis[p] = new_dis
            self.D[p] = new_D

    @ti.kernel
    def project_to_body(self):
        """delta 投影：只把碰撞引起的位移修正聚合回 (v_c, ω)

          Δ_p   = dis_p - dis_pred_p
          Δv    = (1/N) Σ Δ_p / dt
          ΔL    = (M/N) Σ arm_p × Δ_p / dt
          Δω    = I_world⁻¹ ΔL
        """
        R = self.quat_to_mat(self.body_q[None])
        sum_delta = ti.Vector.zero(ti.f32, 3)
        sum_torque = ti.Vector.zero(ti.f32, 3)
        for p in range(self.n_particles):
            delta = self.dis[p] - self.dis_pred[p]
            sum_delta += delta
            arm = R @ self.r_local[p]
            sum_torque += arm.cross(delta)

        delta_v = sum_delta / (self.n_particles * self.dt)
        delta_L = (self.body_mass[None] / self.n_particles) * (sum_torque / self.dt)
        delta_omega = self.body_I_inv_world[None] @ delta_L

        self.body_v[None] += delta_v
        self.body_omega[None] += delta_omega

    @ti.kernel
    def update_body(self):
        self.body_x[None] += self.body_v[None] * self.dt
        w = self.body_omega[None]
        q = self.body_q[None]
        dq = 0.5 * self.dt * self.quat_mul(ti.Vector([0.0, w[0], w[1], w[2]]), q)
        q_new = q + dq
        self.body_q[None] = q_new / q_new.norm()

    @ti.kernel
    def clamp_body_to_floor(self):
        """硬 AABB 兜底 + 弹性反射：扫所有粒子的最低点，穿透则顶回，并按恢复系数反弹。

        速度通道的"delta 平均"会被 N 稀释，导致 body 永远追不上重力造成的累积下沉，
        所以位置层再加一道直接约束。restitution=0 为完全非弹性，>0 才会弹。
        """
        restitution = 0.4
        R = self.quat_to_mat(self.body_q[None])
        self.body_min_y[None] = 1e10
        for p in range(self.n_particles):
            py = self.body_x[None][1] + (R @ self.r_local[p])[1]
            ti.atomic_min(self.body_min_y[None], py)
        if self.body_min_y[None] < self.floor_y:
            push = self.floor_y - self.body_min_y[None]
            self.body_x[None][1] += push
            if self.body_v[None][1] < 0.0:
                self.body_v[None][1] = -restitution * self.body_v[None][1]

    @ti.kernel
    def refresh_particles(self):
        """把最终 body_x / body_q 同步回粒子位置，否则渲染会用 regenerate 阶段的旧值"""
        R = self.quat_to_mat(self.body_q[None])
        xc = self.body_x[None]
        for p in range(self.n_particles):
            self.x[p] = xc + R @ self.r_local[p]

    def mpm_solve(self):
        """对应 mpm_pbd.py::mpm_solve"""
        self.clear_grid()
        self.p2g()
        self.grid_update()
        self.g2p()

    def substep(self):
        """对应 mpm_pbd.py::substep，使用 PBMPM 风格的迭代位置修正"""
        self.predict_body()
        for _ in range(self.iteration):
            self.regenerate_particles()
            self.mpm_solve()
            self.project_to_body()
        self.update_body()
        self.clamp_body_to_floor()
        self.refresh_particles()

    # endregion


# region === Render ===
def build_floor(y: float, lo: float = 0.0, hi: float = 1.0):
    verts = ti.Vector.field(3, dtype=ti.f32, shape=4)
    indices = ti.field(dtype=ti.i32, shape=6)
    verts.from_numpy(
        np.array(
            [[lo, y, lo], [hi, y, lo], [hi, y, hi], [lo, y, hi]],
            dtype=np.float32,
        )
    )
    indices.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))
    return verts, indices


def build_floor_grid(y: float, n: int = 11, lo: float = 0.0, hi: float = 1.0):
    """xz 方向的网格线，便于看出刚体的旋转/平移"""
    coords = np.linspace(lo, hi, n, dtype=np.float32)
    pts = []
    for c in coords:
        pts.append([lo, y, c]); pts.append([hi, y, c])
        pts.append([c, y, lo]); pts.append([c, y, hi])
    arr = np.asarray(pts, dtype=np.float32)
    f = ti.Vector.field(3, dtype=ti.f32, shape=arr.shape[0])
    f.from_numpy(arr)
    return f


def main():
    solver = RigidMpmSolver(n_per_side=16)
    solver.add_cube(
        center=[0.5, 0.7, 0.5],
        cube_size=[0.15, 0.15, 0.15],
        omega0=[3.0, 1.5, 1.0],
        mass=1.0,
    )

    floor_v, floor_i = build_floor(solver.floor_y)
    floor_lines = build_floor_grid(solver.floor_y + 1e-4)

    window = ti.ui.Window("Rigid-MPM Demo (v3, iterative)", (720, 720), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.1, 0.1, 0.12))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.5, 0.6, 1.6)
    camera.lookat(0.5, 0.3, 0.5)

    while window.running:
        for _ in range(4):
            solver.substep()

        scene.set_camera(camera)
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(1.0, 2.0, 1.5), color=(1, 1, 1))
        scene.mesh(floor_v, indices=floor_i, color=(0.25, 0.27, 0.32), two_sided=True)
        scene.lines(floor_lines, width=1.0, color=(0.45, 0.48, 0.55))
        scene.particles(solver.x, radius=0.004, color=(0.4, 0.7, 1.0))
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
# endregion
