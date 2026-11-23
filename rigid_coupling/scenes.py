"""Scene storage and sampling helpers for the two shaking-box demonstrations."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Scene:
    particles: dict
    bodies: list
    description: str
    sampling: dict = field(default_factory=dict)
    emitter: dict = field(default_factory=dict)
    container: dict = field(default_factory=dict)


def lattice(center, size, resolution):
    center, size = np.asarray(center, dtype=float), np.asarray(size, dtype=float)
    counts = np.broadcast_to(np.asarray(resolution, dtype=int), (3,))
    axes = [center[a]+((np.arange(counts[a])+.5)/counts[a]-.5)*size[a] for a in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def build(name, config, resolution=24, *, fluid_resolution=None, soft_resolution=None, seed=42, **unused):
    from .shake import build_shake_scene
    from .state import FLUID, ELASTIC
    if name not in ("shake_water", "shake_soft"):
        raise ValueError("available scenes: shake_water, shake_soft")
    return build_shake_scene(config, FLUID if name == "shake_water" else ELASTIC,
                             resolution=resolution if soft_resolution is None else soft_resolution,
                             fluid_resolution=fluid_resolution, seed=seed)
