import numpy as np
import taichi as ti

FLUID, ELASTIC, SAND, RIGID = range(4)


@ti.dataclass
class Particle:
    x: ti.types.vector(3, ti.f32)
    x0: ti.types.vector(3, ti.f32)
    d: ti.types.vector(3, ti.f32)
    D: ti.types.matrix(3, 3, ti.f32)
    F: ti.types.matrix(3, 3, ti.f32)
    multiplier: ti.types.matrix(3, 3, ti.f32)
    volume_ratio: ti.f32
    plastic_volume: ti.f32
    mass: ti.f32
    rest_volume: ti.f32
    material: ti.i32
    body_id: ti.i32
    local: ti.types.vector(3, ti.f32)
    predicted_d: ti.types.vector(3, ti.f32)
    pinned: ti.i32
    uid: ti.i32
    color: ti.types.vector(3, ti.f32)


class Particles:
    """All per-particle fields travel together, including IDs and multipliers."""

    def __init__(self, initial):
        self.n = len(initial["x"])
        if self.n == 0:
            raise ValueError("The scene must contain particles")
        self.data = Particle.field(shape=self.n, layout=ti.Layout.SOA)
        self.restore(initial)

    def restore(self, arrays):
        self.data.from_numpy(arrays)

    def snapshot(self):
        return self.data.to_numpy()


def particle_arrays(x, material, volume, velocity, dt, color, pinned=None, body_id=-1, local=None):
    n = len(x)
    z = np.zeros((n, 3, 3), dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    return dict(
        x=x, x0=x.copy(), d=np.tile(np.asarray(velocity, dtype=np.float32) * dt, (n, 1)),
        D=z.copy(), F=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)), multiplier=z.copy(),
        volume_ratio=np.ones(n, dtype=np.float32), plastic_volume=np.zeros(n, dtype=np.float32),
        mass=np.full(n, volume, dtype=np.float32), rest_volume=np.full(n, volume, dtype=np.float32),
        material=np.full(n, material, dtype=np.int32), body_id=np.full(n, body_id, dtype=np.int32),
        local=np.zeros((n, 3), dtype=np.float32) if local is None else np.asarray(local, dtype=np.float32),
        predicted_d=np.zeros((n, 3), dtype=np.float32),
        pinned=np.zeros(n, dtype=np.int32) if pinned is None else np.asarray(pinned, dtype=np.int32),
        uid=np.zeros(n, dtype=np.int32), color=np.tile(np.asarray(color, dtype=np.float32), (n, 1)),
    )
