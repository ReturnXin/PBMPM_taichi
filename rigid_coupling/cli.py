"""Unified water and PBMPM/XPBMPM soft-body demonstrations."""
import argparse
import math
from pathlib import Path

from .config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Left-drag increments push all water horizontally, or shake soft/rigid objects")
    parser.add_argument("--scene", choices=("shake_water", "shake_soft"), default="shake_water")
    parser.add_argument("--gui", action="store_true", help="Open mouse control; default when not recording")
    parser.add_argument("--auto-shake", action="store_true", help="Start with the repeatable shaking motion")
    parser.add_argument("--gif-dir", type=Path, help="Save both material GIFs into the same chosen folder")
    parser.add_argument("--arch", choices=("auto", "cpu", "cuda", "vulkan"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frames-per-second", type=int, default=120)
    parser.add_argument("--playback-speed", type=float, default=1.)
    parser.add_argument("--substeps", type=int, default=2,
                        help="Maximum fixed physics steps per frame; default 2 balances motion and display response")
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--dt", type=float, default=.002)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--transfer-mode", choices=("full", "incremental"), default="full",
                        help="Experimental water-only incremental constraint transfers")
    parser.add_argument("--boundary-density", choices=("raw", "mirror", "solid", "available"), default=None,
                        help="Packing density: raw particle volumes (default), mirror/solid walls, or experimental available-volume normalization")
    parser.add_argument("--wall-transfer", choices=("clamp", "reflected"), default="clamp",
                        help="Experimental paired reflected APIC wall motion; defaults to available density, clamped feedback and relaxation 1")
    parser.add_argument("--density-feedback", choices=("legacy", "clamped"), default=None,
                        help="Available density: legacy overfill-only blend or clamped non-tensile packing target")
    parser.add_argument("--duration", type=float, default=None, help="GIF physical seconds; default 4")
    parser.add_argument("--fluid-resolution", type=int, default=24)
    parser.add_argument("--soft-resolution", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gravity", type=float, default=-9.8)
    parser.add_argument("--fluid-density", type=float, default=1.)
    parser.add_argument("--soft-density", type=float, default=1.)
    parser.add_argument("--rigid-density", type=float, default=None)
    parser.add_argument("--soft-friction", type=float, default=0., help="Soft contact friction; default 0")
    parser.add_argument("--elastic-relaxation", type=float, default=.04)
    parser.add_argument("--elastic-volume-weight", type=float, default=.25)
    parser.add_argument("--elastic-solver", choices=("pbmpm", "xpbmpm"), default="pbmpm",
                        help="Soft material solver; XPBMPM uses accumulated compliant multipliers")
    parser.add_argument("--elastic-model", choices=("experiment", "legacy"), default=None,
                        help="Default: legacy for XPBMPM, existing shape/volume model for PBMPM")
    parser.add_argument("--compliance", type=float, default=1e-6,
                        help="XPBMPM compliance alpha (scaled internally by dt squared)")
    parser.add_argument("--inverse-metric", type=float, default=1.)
    parser.add_argument("--fluid-relaxation", type=float, default=None)
    parser.add_argument("--viscosity", type=float, default=0.)
    parser.add_argument("--buoyancy-assist", action=argparse.BooleanOptionalAction, default=None,
                        help="Help submerged light boxes leave the floor; floating boxes use normal coupling (water default: on)")
    args = parser.parse_args(argv)
    water = args.scene == "shake_water"
    args.elastic_model = args.elastic_model or ("legacy" if args.elastic_solver == "xpbmpm" else "experiment")
    if water and args.elastic_solver == "xpbmpm":
        parser.error("XPBMPM is an elastic-material option; select --scene shake_soft")
    reflected = args.wall_transfer == "reflected"
    args.boundary_density = args.boundary_density or ("available" if reflected else "raw")
    args.density_feedback = args.density_feedback or ("clamped" if reflected else "legacy")
    if args.fluid_relaxation is None:
        args.fluid_relaxation = 1. if reflected else 1.5
    if reflected and not water:
        parser.error("reflected wall transfers currently support shake_water only")
    if not water and args.transfer_mode == "incremental":
        parser.error("incremental transfer currently supports shake_water only")
    if not water and args.boundary_density != "raw":
        parser.error("solid-box boundary density currently supports shake_water only")
    args.duration = (4. if args.gif_dir else 0.) if args.duration is None else args.duration
    if not math.isfinite(args.duration) or args.duration < 0 or (args.gif_dir and args.duration <= 0):
        parser.error("GIF duration must be positive; interactive duration can be 0")
    if min(args.threads, args.frames_per_second, args.substeps) < 1:
        parser.error("threads, frames-per-second and substeps must be positive")
    if min(args.fluid_resolution, args.soft_resolution) < 3 or args.seed < 0:
        parser.error("sampling resolution must be >= 3; seed must be nonnegative")
    if not math.isfinite(args.playback_speed) or args.playback_speed <= 0:
        parser.error("playback-speed must be finite and positive")
    try:
        config = Config(grid=args.grid, dt=args.dt, iterations=args.iterations, gravity=args.gravity,
                        fluid_relaxation=args.fluid_relaxation, viscosity=args.viscosity,
                        fluid_density=args.fluid_density, soft_density=args.soft_density,
                        rigid_density=(.35 if water else .7) if args.rigid_density is None else args.rigid_density,
                        rigid_volume=water, rigid_contact=True, world_contact=True,
                        buoyancy_assist=water if args.buoyancy_assist is None else args.buoyancy_assist,
                        grid_coupling="separate", elastic_relaxation=args.elastic_relaxation,
                        elastic_volume_weight=args.elastic_volume_weight,
                        elastic_solver=args.elastic_solver, elastic_model=args.elastic_model,
                        compliance=args.compliance, inverse_metric=args.inverse_metric,
                        soft_friction=args.soft_friction, transfer_mode=args.transfer_mode,
                        boundary_density=args.boundary_density, wall_transfer=args.wall_transfer,
                        density_feedback=args.density_feedback)
    except ValueError as exc:
        parser.error(str(exc))
    from .shake import run
    run(args.scene, [config], args)


if __name__ == "__main__":
    main()
