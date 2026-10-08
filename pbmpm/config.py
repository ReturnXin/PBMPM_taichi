from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    grid: int = 32
    dt: float = 0.002
    iterations: int = 5
    gravity: float = -9.8
    boundary: int = 3
    elastic_solver: str = "pbmpm"
    elastic_model: str = "experiment"
    compliance: float = 2e-5
    inverse_metric: float = 1.0
    volume_weight: float = 0.25
    relaxation: float = 1.0
    fluid_relaxation: float = 1.5
    viscosity: float = 0.02
    sand_relaxation: float = 1.0
    friction_angle: float = 35.0
    coupling: str = "two_way"
    reorder: str = "none"
    reorder_interval: int = 10

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 16 <= self.grid <= 1024 or self.dt <= 0 or self.iterations < 1:
            raise ValueError("grid must be 16..1024; dt and iterations must be positive")
        if not 2 <= self.boundary < self.grid // 2:
            raise ValueError("boundary must leave an interior and a complete B-spline stencil")
        if self.compliance < 0 or self.inverse_metric <= 0:
            raise ValueError("compliance >= 0 and inverse_metric > 0 are required")
        if self.elastic_model == "legacy" and self.inverse_metric != 1.0:
            raise ValueError("The legacy elastic update requires inverse_metric=1")
        if not 0 <= self.volume_weight <= 1 or not 0 <= self.viscosity <= 1:
            raise ValueError("volume_weight and viscosity must be in [0, 1]")
        if not 0 <= self.friction_angle < 90:
            raise ValueError("friction_angle must be in [0, 90)")
        if min(self.relaxation, self.fluid_relaxation, self.sand_relaxation) < 0:
            raise ValueError("relaxation parameters must be nonnegative")
        for name, allowed in (("elastic_solver", ("pbmpm", "xpbmpm")),
                              ("elastic_model", ("experiment", "legacy")),
                              ("coupling", ("one_way", "two_way")),
                              ("reorder", ("none", "full", "periodic"))):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.reorder_interval < 1:
            raise ValueError("reorder_interval must be positive")
