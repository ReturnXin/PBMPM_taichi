"""Parameters for the independent rigid/continuum coupling solver."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    grid: int = 32
    dt: float = 0.002
    iterations: int = 5
    gravity: float = -9.8
    boundary: int = 3
    fluid_relaxation: float = 1.5
    viscosity: float = 0.02
    coupling: str = "two_way"
    fluid_density: float = 1.0
    rigid_density: float = 1.0
    rigid_volume: bool = False
    rigid_contact: bool = False
    world_contact: bool = False
    buoyancy_assist: bool = False
    grid_coupling: str = "shared"
    soft_density: float = 1.0
    elastic_relaxation: float = 0.2
    elastic_volume_weight: float = 0.25
    soft_friction: float = 0.0
    elastic_solver: str = "pbmpm"
    elastic_model: str = "experiment"
    compliance: float = 1e-6
    inverse_metric: float = 1.0
    transfer_mode: str = "full"
    boundary_density: str = "raw"
    wall_transfer: str = "clamp"
    density_feedback: str = "legacy"

    def __post_init__(self):
        for name in ("grid", "iterations", "boundary"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be an integer")
        for name in ("dt", "gravity", "fluid_relaxation", "viscosity", "fluid_density", "rigid_density",
                     "soft_density", "elastic_relaxation", "elastic_volume_weight", "soft_friction",
                     "compliance", "inverse_metric"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 16 <= self.grid <= 1024 or self.dt <= 0 or self.iterations < 1:
            raise ValueError("grid must be 16..1024; dt and iterations must be positive")
        if not 2 <= self.boundary < self.grid // 2:
            raise ValueError("boundary must leave an interior and a complete B-spline stencil")
        if self.fluid_relaxation < 0 or not 0 <= self.viscosity <= 1:
            raise ValueError("fluid_relaxation must be nonnegative; viscosity must be in [0, 1]")
        if self.coupling not in ("one_way", "two_way"):
            raise ValueError("coupling must be one_way or two_way")
        if self.grid_coupling not in ("shared", "separate"):
            raise ValueError("grid_coupling must be shared or separate")
        if self.transfer_mode not in ("full", "incremental"):
            raise ValueError("transfer_mode must be full or incremental")
        if self.boundary_density not in ("raw", "mirror", "solid", "available"):
            raise ValueError("boundary_density must be raw, mirror, solid or available")
        if self.wall_transfer not in ("clamp", "reflected"):
            raise ValueError("wall_transfer must be clamp or reflected")
        if self.density_feedback not in ("legacy", "clamped"):
            raise ValueError("density_feedback must be legacy or clamped")
        if self.density_feedback == "clamped" and self.boundary_density != "available":
            raise ValueError("clamped density feedback requires available boundary density")
        if self.wall_transfer == "reflected":
            if self.transfer_mode != "full" or self.grid_coupling != "separate" or self.boundary_density != "available":
                raise ValueError("reflected wall transfers require full transfer, separate grids and available density")
            if self.fluid_relaxation > 1:
                raise ValueError("reflected wall transfers currently require fluid_relaxation <= 1")
        if self.transfer_mode == "incremental" and self.grid_coupling != "separate":
            raise ValueError("incremental transfer requires separate fluid/rigid grids")
        if self.grid_coupling == "separate" and not self.rigid_contact:
            raise ValueError("separate grid coupling requires rigid_contact for continuum/body interaction")
        if min(self.fluid_density, self.rigid_density) <= 0:
            raise ValueError("fluid_density and rigid_density must be positive")
        if self.soft_density <= 0 or not 0 < self.elastic_relaxation <= 1:
            raise ValueError("soft_density must be positive; elastic_relaxation must be in (0, 1]")
        if not 0 <= self.elastic_volume_weight < 1:
            raise ValueError("elastic_volume_weight must be in [0, 1) to retain shape restoration")
        if self.soft_friction < 0:
            raise ValueError("soft_friction must be nonnegative")
        if self.elastic_solver not in ("pbmpm", "xpbmpm"):
            raise ValueError("elastic_solver must be pbmpm or xpbmpm")
        if self.elastic_model not in ("experiment", "legacy"):
            raise ValueError("elastic_model must be experiment or legacy")
        if self.compliance < 0 or self.inverse_metric <= 0:
            raise ValueError("compliance >= 0 and inverse_metric > 0 are required")
        if self.elastic_model == "legacy" and self.inverse_metric != 1.0:
            raise ValueError("legacy elastic updates require inverse_metric=1")
        if self.soft_friction > 0 and (not self.rigid_contact or self.grid_coupling != "separate"):
            raise ValueError("soft friction requires rigid_contact and separate grid coupling")

    def as_dict(self):
        return asdict(self)

