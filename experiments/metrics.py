import numpy as np

from pbmpm.state import ELASTIC, FLUID


class CantileverContinuity:
    """Track reference neighbors by permanent ID, independently of particle storage."""

    def __init__(self, initial):
        x = initial["x0"]
        counts = tuple(len(np.unique(x[:, a])) for a in range(3))
        if np.prod(counts) != len(x):
            raise ValueError("Cantilever continuity requires a regular reference lattice")
        order = np.lexsort((x[:, 2], x[:, 1], x[:, 0]))
        indices = order.reshape(counts)
        pairs = []
        for axis in range(3):
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            pairs.append(np.column_stack((indices[tuple(a)].ravel(), indices[tuple(b)].ravel())))
        pairs = np.concatenate(pairs)
        pairs = pairs[~np.all(initial["pinned"][pairs] != 0, axis=1)]
        self.ids = initial["uid"][pairs]
        self.length = np.linalg.norm(x[pairs[:, 1]] - x[pairs[:, 0]], axis=1)

    def measure(self, particles, grid):
        x = particles["x"][np.argsort(particles["uid"])]
        a, b = x[self.ids[:, 0]], x[self.ids[:, 1]]
        lengths = np.linalg.norm(b - a, axis=1)
        # Positive quadratic B-spline support. Merely finite F cannot detect a gap.
        scaled = x * grid
        base = np.floor(scaled - 0.5).astype(int)
        f = scaled - base
        lo = base + (0.5 * (1.5 - f)**2 <= 0).astype(int)
        hi = base + 2 - (0.5 * (f - 0.5)**2 <= 0).astype(int)
        shared = np.all(np.maximum(lo[self.ids[:, 0]], lo[self.ids[:, 1]]) <=
                        np.minimum(hi[self.ids[:, 0]], hi[self.ids[:, 1]]), axis=1)
        return dict(reference_neighbor_max_distance=float(lengths.max()),
                    reference_neighbor_max_stretch=float(np.max(lengths / self.length)),
                    reference_neighbor_support_lost=int(np.count_nonzero(~shared)))


def sample(solver, continuity=None):
    p = solver.particles.snapshot()
    for key, value in p.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite particle field: {key}")
    x, mass = p["x"], p["mass"]
    c = solver.config
    result = dict(time=solver.step_index * c.dt, center_y=float(np.average(x[:, 1], weights=mass)),
                  min_y=float(x[:, 1].min()), max_y=float(x[:, 1].max()),
                  boundary_penetration=float(max(0, c.boundary/c.grid - x.min(), x.max() - (1-c.boundary/c.grid))))
    elastic = (p["material"] == ELASTIC) & (p["pinned"] == 0)
    if elastic.any():
        tip = elastic & (p["x0"][:, 0] >= p["x0"][elastic, 0].max() - 1e-5)
        result["tip_deflection_y"] = float(np.mean(p["x"][tip, 1] - p["x0"][tip, 1]))
        determinants = np.linalg.det(p["F"][elastic])
        result["mean_abs_detF_error"] = float(np.mean(np.abs(determinants - 1)))
        result["min_detF"] = float(determinants.min())
        residuals = solver.materials.residuals.to_numpy()
        if residuals[2] > 0:
            result["mean_target_residual"] = float(residuals[0] / residuals[2])
            if c.elastic_solver == "xpbmpm":
                result["mean_flexible_residual"] = float(residuals[1] / residuals[2])
    fluid = p["material"] == FLUID
    if fluid.any():
        result["mean_liquid_volume_ratio"] = float(np.mean(p["volume_ratio"][fluid]))
        result["liquid_velocity_x"] = float(np.average(p["d"][fluid, 0] / c.dt, weights=mass[fluid]))
    if solver.bodies.count:
        bodies = solver.bodies.data.to_numpy()
        for b in range(solver.bodies.count):
            q = bodies["q"][b]
            result[f"body{b}_x"] = float(bodies["x"][b, 0])
            result[f"body{b}_y"] = float(bodies["x"][b, 1])
            result[f"body{b}_velocity_x"] = float(bodies["v"][b, 0])
            result[f"body{b}_angular_speed"] = float(np.linalg.norm(bodies["omega"][b]))
            result[f"body{b}_orientation_angle"] = float(2*np.arccos(np.clip(abs(q[0]), 0, 1)))
            owned = p["body_id"] == b
            arms = p["x"][owned] - bodies["x"][b]
            result[f"body{b}_shape_radius_error"] = float(np.max(np.abs(
                np.linalg.norm(arms, axis=1) - np.linalg.norm(p["local"][owned], axis=1))))
    if continuity is not None:
        result.update(continuity.measure(p, c.grid))
    return result
