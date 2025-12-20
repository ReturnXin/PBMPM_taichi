import taichi as ti


@ti.func
def expand_bits(v):
    v = ti.cast(v, ti.u32)
    v = (v * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
    v = (v * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
    v = (v * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
    v = (v * ti.u32(0x00000005)) & ti.u32(0x49249249)
    return v


@ti.func
def get_morton_code(xp, dx, n_grid):
    grid_idx = ti.cast(xp / dx + 1e-5, ti.i32)
    x = ti.max(0, ti.min(grid_idx[0], n_grid - 1))
    y = ti.max(0, ti.min(grid_idx[1], n_grid - 1))
    z = ti.max(0, ti.min(grid_idx[2], n_grid - 1))
    xx = expand_bits(x)
    yy = expand_bits(y)
    zz = expand_bits(z)
    result = (xx) | (yy << 1) | (zz << 2)
    return ti.cast(result, ti.i32)
