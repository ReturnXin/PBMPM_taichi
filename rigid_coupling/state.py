"""Particle storage for fluid, rigid and PBMPM elastic materials."""

import numpy as np
import taichi as ti

FLUID, RIGID, ELASTIC = range(3)


@ti.dataclass
class Particle:
    x: ti.types.vector(3, ti.f32)
    x0: ti.types.vector(3, ti.f32)
    d: ti.types.vector(3, ti.f32)
    D: ti.types.matrix(3, 3, ti.f32)
    F: ti.types.matrix(3, 3, ti.f32)
    volume_ratio: ti.f32
    mass: ti.f32
    rest_volume: ti.f32
    material: ti.i32
    body_id: ti.i32
    local: ti.types.vector(3, ti.f32)
    predicted_d: ti.types.vector(3, ti.f32)
    uid: ti.i32
    color: ti.types.vector(3, ti.f32)


class Particles:
    def __init__(self, initial, active_count=None, *, xpbmpm=False):
        self.n = len(initial["x"])
        if self.n == 0:
            raise ValueError("The scene must contain particles")
        self.data = Particle.field(shape=self.n, layout=ti.Layout.SOA)
        self.initial_count = self.n if active_count is None else active_count
        if not isinstance(self.initial_count, (int, np.integer)) or not 1 <= self.initial_count <= self.n:
            raise ValueError("active_count must be an integer in [1, particle capacity]")
        self.count = ti.field(ti.i32, shape=())
        self.has_fluid = bool(np.any(initial["material"] == FLUID))
        self.has_elastic = bool(np.any(initial["material"] == ELASTIC))
        # Water/PBMPM retain the existing particle layout and memory cost.
        self.multiplier = (ti.Matrix.field(3, 3, ti.f32, shape=self.n)
                           if xpbmpm and self.has_elastic else None)
        self.restore(initial)

    def restore(self, arrays):
        if len(arrays["x"]) != self.n:
            raise ValueError("reset cannot change the particle count")
        for name, value in arrays.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"particle {name} must be finite")
        if np.any(arrays["mass"] <= 0) or np.any(arrays["rest_volume"] <= 0):
            raise ValueError("particle mass and rest_volume must be positive")
        if not np.all(np.isin(arrays["material"], (FLUID, RIGID, ELASTIC))):
            raise ValueError("the isolated solver supports FLUID, RIGID and ELASTIC")
        rigid = arrays["material"] == RIGID
        if np.any(arrays["body_id"][rigid] < 0) or np.any(arrays["body_id"][~rigid] != -1):
            raise ValueError("rigid particles need a body_id; continuum particles need body_id=-1")
        # Old saved fluid/rigid states have no deformation gradient.
        arrays = dict(arrays)
        arrays.setdefault("F", np.tile(np.eye(3, dtype=np.float32), (self.n, 1, 1)))
        if arrays["F"].shape != (self.n, 3, 3) or not np.isfinite(arrays["F"]).all():
            raise ValueError("F must be a finite (N, 3, 3) array")
        elastic = arrays["material"] == ELASTIC
        if np.any(np.linalg.det(arrays["F"][elastic]) <= 0):
            raise ValueError("elastic deformation gradients must have positive determinant")
        multiplier = arrays.pop("multiplier", None)
        if self.multiplier is not None:
            if multiplier is None:
                self.multiplier.fill(0.)
            else:
                if multiplier.shape != (self.n, 3, 3):
                    raise ValueError("multiplier must have shape (N, 3, 3)")
                self.multiplier.from_numpy(multiplier)
        elif multiplier is not None and np.any(multiplier != 0):
            raise ValueError("nonzero multipliers require an XPBMPM elastic scene")
        self.data.from_numpy(arrays)
        self.count[None] = self.initial_count

    def snapshot(self):
        arrays = self.data.to_numpy()
        if self.multiplier is not None:
            arrays["multiplier"] = self.multiplier.to_numpy()
        return arrays


def _per_particle(value, n, name):
    array = np.broadcast_to(np.asarray(value, dtype=np.float32), (n,)).copy()
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def particle_arrays(x, material, volume, velocity, dt, color,
                    body_id=-1, local=None, mass=None):
    """Build one material group. volume/mass may be scalar or per-particle.

    ``volume`` is rest volume *per particle*, not the group's total volume.
    The default mass equals volume (unit reference density).
    """
    x = np.array(x, dtype=np.float32, copy=True)
    if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite (N, 3) array")
    n = len(x)
    volume = _per_particle(volume, n, "volume")
    masses = _per_particle(volume if mass is None else mass, n, "mass")
    local = np.zeros_like(x) if local is None else np.array(local, dtype=np.float32, copy=True)
    if local.shape != x.shape or not np.all(np.isfinite(local)):
        raise ValueError("local must have the same shape as x and be finite")
    velocity = np.broadcast_to(np.asarray(velocity, dtype=np.float32), (n, 3)).copy()
    color = np.broadcast_to(np.asarray(color, dtype=np.float32), (n, 3)).copy()
    if not np.isfinite(dt) or dt <= 0 or not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(color)):
        raise ValueError("dt must be positive; velocity and color must be finite")
    if material not in (FLUID, RIGID, ELASTIC):
        raise ValueError("the isolated solver supports FLUID, RIGID and ELASTIC")
    if not isinstance(body_id, (int, np.integer)):
        raise ValueError("body_id must be an integer")
    if (material == RIGID and body_id < 0) or (material != RIGID and body_id != -1):
        raise ValueError("body_id must be nonnegative for RIGID and -1 for continuum particles")
    return dict(x=x, x0=x.copy(), d=velocity * dt,
                D=np.zeros((n, 3, 3), dtype=np.float32),
                F=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
                volume_ratio=np.ones(n, dtype=np.float32), mass=masses,
                rest_volume=volume, material=np.full(n, material, dtype=np.int32),
                body_id=np.full(n, body_id, dtype=np.int32), local=local,
                predicted_d=np.zeros_like(x), uid=np.arange(n, dtype=np.int32), color=color)


def combine_particles(groups):
    groups = list(groups)
    if not groups:
        raise ValueError("at least one particle group is required")
    arrays = {name: np.concatenate([group[name] for group in groups]) for name in groups[0]}
    arrays["uid"] = np.arange(len(arrays["x"]), dtype=np.int32)
    return arrays

