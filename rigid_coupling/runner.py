"""Headless runs and reproducible records for the isolated rigid coupling solver."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .rigid import rotation_numpy
from .state import FLUID, ELASTIC


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def source_hashes():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.py"))}


def sample(solver):
    p = solver.particles.snapshot()
    active_count = solver.particles.count[None] if hasattr(solver.particles, "count") else len(p["x"])
    p = {key: value[:active_count] for key, value in p.items()}
    b = solver.bodies.data.to_numpy()
    for prefix, arrays in (("particle", p), ("body", b)):
        for key, value in arrays.items():
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise FloatingPointError(f"non-finite {prefix} field: {key}")
    c = solver.config
    x, mass = p["x"].astype(float), p["mass"].astype(float)
    velocity = p["d"].astype(float) / c.dt
    momentum = np.sum(mass[:, None] * velocity, axis=0)
    result = dict(time=solver.step_index * c.dt,
                  center_y=float(np.average(x[:, 1], weights=mass)),
                  boundary_penetration=float(max(0, c.boundary / c.grid - x.min(),
                                                 x.max() - (1 - c.boundary / c.grid))))
    for axis, value in zip("xyz", momentum):
        result[f"particle_linear_momentum_{axis}"] = float(value)
    fluid = p["material"] == FLUID
    soft = p["material"] == ELASTIC
    if soft.any():
        stretch = np.linalg.svd(p["F"][soft].astype(float), compute_uv=False)
        result["soft_particle_count"] = int(soft.sum())
        result["soft_mean_volume_ratio"] = float(np.mean(p["volume_ratio"][soft]))
        result["soft_strain_rms"] = float(np.sqrt(np.mean((stretch - 1)**2)))
        result["soft_min_volume_ratio"] = float(np.min(p["volume_ratio"][soft]))
        for axis_id, axis in enumerate("xyz"):
            result[f"soft_center_{axis}"] = float(np.average(x[soft, axis_id], weights=mass[soft]))
            result[f"soft_velocity_{axis}"] = float(np.average(velocity[soft, axis_id], weights=mass[soft]))
            result[f"soft_extent_{axis}"] = float(np.ptp(x[soft, axis_id]))
    has_fluid = np.any(solver.initial["material"] == FLUID) if hasattr(solver, "initial") else fluid.any()
    if has_fluid:
        fluid_volume = float(p["rest_volume"][fluid].sum(dtype=float))
        fluid_mass = float(mass[fluid].sum())
        result["fluid_particle_count"] = int(fluid.sum())
        result["fluid_rest_volume"] = fluid_volume
        result["fluid_mass"] = fluid_mass
        result["fluid_density"] = c.fluid_density
        result["fluid_realized_ppc"] = int(fluid.sum()) / (fluid_volume * c.grid**3) if fluid_volume else 0.0
        result["mean_liquid_volume_ratio"] = float(np.mean(p["volume_ratio"][fluid])) if fluid.any() else 1.0
        result["liquid_velocity_x"] = float(np.average(velocity[fluid, 0], weights=mass[fluid])) if fluid.any() else 0.0
    for body_id in range(solver.bodies.count):
        if soft.any() and hasattr(solver, "contacts"):
            normal = solver.contacts.normal_impulse.to_numpy()[body_id]
            friction = solver.contacts.friction_impulse.to_numpy()[body_id]
            for a, axis in enumerate("xyz"):
                result[f"body{body_id}_normal_impulse_{axis}"] = float(normal[a])
                result[f"body{body_id}_friction_impulse_{axis}"] = float(friction[a])
            result[f"body{body_id}_weight"] = float(b["mass"][body_id] * abs(c.gravity))
        if c.buoyancy_assist and c.coupling == "two_way":
            result["assist_water_level"] = float(solver.buoyancy.level[None])
            result[f"body{body_id}_assist_submerged_fraction"] = float(solver.buoyancy.fraction[body_id])
            result[f"body{body_id}_assist_delta_vy"] = float(solver.buoyancy.correction[body_id])
        owned = p["body_id"] == body_id
        center, q = b["x"][body_id], b["q"][body_id]
        R = rotation_numpy(q)
        expected = center + p["local"][owned] @ R.T
        body = solver.bodies.initial[body_id]
        if "size" in body:
            half_size = np.asarray(body["size"], dtype=float) / 2
            world_half_size = np.abs(R) @ half_size
            geometric_min_y = float(center[1] - world_half_size[1])
            result[f"body{body_id}_surface_min_y"] = geometric_min_y
            result[f"body{body_id}_floor_gap"] = geometric_min_y - c.boundary / c.grid
            if soft.any():
                local_soft = (x[soft] - center) @ R
                depth = np.min(half_size - np.abs(local_soft), axis=1)
                result[f"body{body_id}_soft_inside_count"] = int(np.count_nonzero(depth > 1e-5))
                result[f"body{body_id}_soft_max_penetration"] = float(max(0., depth.max()))
                # Sample-based separation from the lower local face. This is
                # useful for an upright load/pad scene; sparse particles are
                # not a reconstructed continuous soft surface.
                below = ((np.abs(local_soft[:, 0]) < half_size[0]) &
                         (np.abs(local_soft[:, 2]) < half_size[2]) & (local_soft[:, 1] < 0))
                result[f"body{body_id}_soft_below_count"] = int(np.count_nonzero(below))
                result[f"body{body_id}_soft_below_gap"] = (
                    float(-half_size[1]-local_soft[below, 1].max()) if below.any() else None)
            if "fluid_particle_count" in result:
                local_fluid = (x[fluid] - center) @ R
                depth = np.min(half_size - np.abs(local_fluid), axis=1)
                # Numerical contact tolerances should not count a point on the surface as trapped water.
                result[f"body{body_id}_fluid_inside_count"] = int(np.count_nonzero(depth > 1e-5))
                result[f"body{body_id}_fluid_max_penetration"] = float(max(0.0, depth.max())) if fluid.any() else 0.0
                result[f"body{body_id}_density_ratio"] = float(body["density"] / result["fluid_density"])
        for axis_id, axis in enumerate("xyz"):
            result[f"body{body_id}_{axis}"] = float(center[axis_id])
            result[f"body{body_id}_velocity_{axis}"] = float(b["v"][body_id, axis_id])
            result[f"body{body_id}_omega_{axis}"] = float(b["omega"][body_id, axis_id])
        if soft.any():
            result[f"body{body_id}_world_contacts"] = int(b["world_contacts"][body_id])
        result[f"body{body_id}_angular_speed"] = float(np.linalg.norm(b["omega"][body_id]))
        result[f"body{body_id}_quaternion_norm_error"] = float(abs(np.linalg.norm(q) - 1))
        result[f"body{body_id}_shape_position_error"] = float(
            np.max(np.linalg.norm(x[owned] - expected, axis=1)))
        result[f"body{body_id}_shape_radius_error"] = float(np.max(np.abs(
            np.linalg.norm(x[owned] - center, axis=1) - np.linalg.norm(p["local"][owned], axis=1))))
    if getattr(solver, "emitter", {}):
        e = solver.emitter
        result["injected_volume_fraction"] = result["fluid_rest_volume"] / e["container_volume"]
        result["emission_finished"] = int(solver.emitted_count() == e["capacity"])
    return result


def save_state(solver, output, prefix):
    particles = solver.particles.snapshot()
    order = np.argsort(particles["uid"])
    np.savez_compressed(output / f"{prefix}_particles.npz",
                        active_count=np.array(solver.particles.count[None]),
                        **{key: value[order] for key, value in particles.items()})
    bodies = solver.bodies.data.to_numpy()
    np.savez_compressed(output / f"{prefix}_bodies.npz",
                        **{key: value[:solver.bodies.count] for key, value in bodies.items()})


