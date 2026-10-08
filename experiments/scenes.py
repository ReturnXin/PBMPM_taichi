from dataclasses import dataclass

import numpy as np

from pbmpm.rigid import body_description
from pbmpm.state import ELASTIC, FLUID, RIGID, SAND, particle_arrays


@dataclass
class Scene:
    particles: dict
    bodies: list
    description: str
    sampling: list | None = None


def lattice(center, size, resolution):
    """Cell-centered regular samples; volume and mass track the chosen resolution."""
    size, center = np.asarray(size), np.asarray(center)
    counts = np.broadcast_to(np.asarray(resolution, dtype=int), (3,))
    axes = [(np.arange(counts[a]) + 0.5) / counts[a] * size[a] - size[a] / 2 + center[a] for a in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)


def cantilever_counts(size, resolution, grid):
    """Nearly isotropic samples, at least two per background cell along each axis."""
    spacing = min(min(size) / resolution, 0.5 / grid)
    return np.ceil(np.asarray(size) / spacing - 1e-10).astype(int)


def build(name, config, resolution=12, seed=42):
    if resolution < 3:
        raise ValueError("resolution must be at least 3")
    groups, bodies, sampling = [], [], []

    def block(material, center, size, velocity=(0, 0, 0), pin=False, omega=(0, 0, 0)):
        counts = cantilever_counts(size, resolution, config.grid) if pin else np.full(3, resolution)
        x = lattice(center, size, counts)
        volume = float(np.prod(size) / len(x))
        colors = [(0.2, 0.55, 0.95), (0.9, 0.3, 0.3), (0.85, 0.7, 0.4), (0.5, 0.8, 0.55)]
        local = x - np.asarray(center, dtype=np.float32)
        body_id = -1
        if material == RIGID:
            body_id = len(bodies)
            bodies.append(body_description(local, center, volume, velocity, omega))
        # Keep a physical clamp length when changing the sampling resolution.
        clamp_x = center[0] - size[0] / 2 + size[0] / 6
        pinned = x[:, 0] < clamp_x if pin else None
        group = particle_arrays(x, material, volume, velocity, config.dt, colors[material], pinned, body_id, local)
        if pin:
            group["color"][pinned] = (0.55, 0.60, 0.66)
        groups.append(group)
        sampling.append(dict(material=int(material), counts=counts.tolist(),
                             spacing=(np.asarray(size) / counts).tolist(), particles=len(x),
                             pinned=int(np.count_nonzero(pinned)) if pin else 0,
                             clamp_x=clamp_x if pin else None))

    if name == "xpbmpm":
        block(ELASTIC, (0.50, 0.60, 0.50), (0.45, 0.14, 0.14), pin=True)
        description = "Nearly isotropic cantilever; left one-sixth clamped through pinned particles and their grid support"
    elif name == "rigid":
        # Adjacent blocks share grid support from the first step without overlapping samples.
        block(FLUID, (0.36, 0.51, 0.5), (0.22, 0.20, 0.24), velocity=(1.0, 0, 0))
        block(RIGID, (0.55, 0.56, 0.5), (0.16, 0.20, 0.20))
        description = "Offset liquid impact on a sampled dynamic rigid box"
    elif name == "morton":
        block(FLUID, (0.5, 0.58, 0.5), (0.45, 0.35, 0.4))
        description = "Liquid block, identical randomized storage order for each variant"
    elif name == "combined":
        block(FLUID, (0.36, 0.49, 0.44), (0.22, 0.18, 0.24), velocity=(0.7, 0, 0))
        block(RIGID, (0.55, 0.54, 0.44), (0.16, 0.18, 0.20))
        block(ELASTIC, (0.5, 0.72, 0.5), (0.40, 0.10, 0.12), pin=True)
        block(SAND, (0.68, 0.3, 0.64), (0.13, 0.16, 0.13))
        description = "Liquid, elastic cantilever, granular block and rigid body on one grid"
    else:
        raise ValueError(f"Unknown scene: {name}")
    arrays = {key: np.concatenate([g[key] for g in groups]) for key in groups[0]}
    arrays["uid"] = np.arange(len(arrays["x"]), dtype=np.int32)
    order = np.random.default_rng(seed).permutation(len(arrays["x"]))
    arrays = {key: value[order] for key, value in arrays.items()}
    lo, hi = config.boundary / config.grid, 1 - config.boundary / config.grid
    if np.any(arrays["x"] < lo) or np.any(arrays["x"] > hi):
        raise ValueError("Scene does not fit the interior; increase grid or reduce boundary")
    return Scene(arrays, bodies, description, sampling)
