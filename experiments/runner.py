"""Run identical initial states for response, total timing, and optional stage timing."""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import taichi as ti

from pbmpm.solver import Solver
from .metrics import CantileverContinuity, sample
from .scenes import build


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("pbmpm", "experiments") for p in sorted((root / folder).glob("*.py"))}


def execute(name, config, args, output):
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(experiment=name, config=asdict(config), status="running", seed=args.seed,
                    resolution=args.resolution, duration=args.duration, warmup=args.warmup,
                    requested_backend=args.arch, python=sys.version, platform=platform.platform(),
                    taichi=".".join(map(str, ti.__version__)), numpy=np.__version__, source_hashes=source_hashes())
    try:
        ti.init(arch={"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch],
                enable_fallback=False, offline_cache=False, random_seed=args.seed,
                cpu_max_num_threads=args.threads)
        metadata["actual_backend"] = str(ti.lang.impl.current_cfg().arch)
        scene = build(name, config, args.resolution, args.seed)
        solver = Solver(config, scene)
        continuity = CantileverContinuity(scene.particles) if name == "xpbmpm" else None
        steps = max(1, int(round(args.duration / config.dt)))
        metadata.update(particles=solver.particles.n, bodies=solver.bodies.count,
                        scene=scene.description, sampling=scene.sampling,
                        steps=steps, actual_duration=steps * config.dt)
        write_json(output / "run.json", metadata)
        # Compile all active numerical/diagnostic kernels, then restore the exact initial state.
        for _ in range(args.warmup):
            solver.step(diagnostics=True)
        ti.sync()
        solver.reset()
        for i in range(steps):
            solver.step()
        ti.sync()
        # The preceding pass compiles any deferred sorting stages and primes backend resources.
        timings = []
        for _ in range(args.repeats):
            solver.reset()
            ti.sync()
            start = perf_counter()
            for i in range(steps):
                solver.step()
            ti.sync()
            timings.append((perf_counter() - start) * 1000 / steps)

        stages = defaultdict(float)
        if args.profile:
            solver.reset()
            for i in range(steps):
                for key, seconds in solver.step(profile=True).items():
                    stages[key] += seconds * 1000 / steps

        # Diagnostics and optional rendering are excluded from the total timing runs.
        solver.reset()
        rows = []
        viewer = None
        if args.gui:
            from .viewer import Viewer
            viewer = Viewer(name)
        for i in range(steps):
            record = (i + 1) % args.sample_every == 0 or i == steps - 1
            solver.step(diagnostics=record)
            if record:
                rows.append(sample(solver, continuity))
            if viewer is not None:
                viewer.draw(solver)
        import csv
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        p = solver.particles.snapshot()
        order = np.argsort(p["uid"])
        np.savez_compressed(output / "final_state.npz", **{k: v[order] for k, v in p.items()})
        if solver.bodies.count:
            np.savez_compressed(output / "bodies.npz", **solver.bodies.data.to_numpy())
        summary = dict(total_ms_per_step=timings, median_ms_per_step=float(np.median(timings)),
                       synchronized_stage_ms_per_step=dict(stages), final_metrics=rows[-1],
                       timing_note="Total includes reorder, stencil checks and Python dispatch; excludes diagnostics/rendering/I/O. Stage timings use a separate synchronized replay and are not additive estimates of the total run.")
        if continuity is not None:
            lost = max(r["reference_neighbor_support_lost"] for r in rows)
            summary["cantilever_continuity"] = dict(
                passed=lost == 0, maximum_lost_neighbor_pairs=lost,
                maximum_neighbor_stretch=max(r["reference_neighbor_max_stretch"] for r in rows),
                scope="Recorded states only; loss of shared grid support is an under-resolved deformation diagnostic, not a physical fracture model.")
            if lost:
                print(f"WARNING {output.name}: reference neighbors lost shared grid support; "
                      "do not treat this response as a connected cantilever.", flush=True)
        write_json(output / "summary.json", summary)
        metadata["status"] = "completed"
        write_json(output / "run.json", metadata)
        print(f"{output.name}: {summary['median_ms_per_step']:.3f} ms/step; {solver.particles.n} particles", flush=True)
        return summary
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", metadata)
        raise
    finally:
        ti.reset()
