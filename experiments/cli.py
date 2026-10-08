import argparse
from dataclasses import asdict, replace
from datetime import datetime
from itertools import product
from pathlib import Path

from pbmpm.config import Config


def configurations(name, base, suite):
    if not suite:
        return [base]
    if name == "xpbmpm":
        result = []
        for dt, iterations in product((base.dt, base.dt / 2), sorted({2, 5, base.iterations})):
            result.append(replace(base, dt=dt, iterations=iterations, elastic_solver="pbmpm"))
            for compliance in sorted({0.0, base.compliance, base.compliance * 4}):
                result.append(replace(base, dt=dt, iterations=iterations, elastic_solver="xpbmpm", compliance=compliance))
        return result
    if name == "rigid":
        return [replace(base, coupling=c) for c in ("one_way", "two_way")]
    if name == "morton":
        return [replace(base, reorder=r) for r in ("none", "full", "periodic")]
    return [replace(base, elastic_solver=s, coupling=c, reorder=r)
            for s, c, r in product(("pbmpm", "xpbmpm"), ("one_way", "two_way"), ("none", "periodic"))]


def main(name):
    parser = argparse.ArgumentParser(description=f"{name}: independent thesis experiment")
    parser.add_argument("--suite", action="store_true", help="Run the chapter comparison/ablation matrix")
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=12,
                        help="Samples across the shortest cantilever axis; other blocks use this count per axis")
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--iterations", type=int, default=10 if name == "xpbmpm" else 5)
    parser.add_argument("--duration", type=float, default=0.2, help="Simulated seconds, shared across dt variants")
    parser.add_argument("--gravity", type=float, default=-9.8)
    parser.add_argument("--solver", choices=("pbmpm", "xpbmpm"), default="xpbmpm" if name in ("xpbmpm", "combined") else "pbmpm")
    parser.add_argument("--elastic-model", choices=("legacy", "experiment"),
                        default="legacy" if name == "xpbmpm" else "experiment",
                        help="legacy follows mpm_pbd.py elastic math; experiment restores the earlier chapter model")
    parser.add_argument("--compliance", type=float, default=2e-6 if name == "xpbmpm" else 2e-5)
    parser.add_argument("--volume-weight", type=float, default=0.25,
                        help="Shape/volume target blend, used only with --elastic-model experiment")
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--coupling", choices=("one_way", "two_way"), default="two_way")
    parser.add_argument("--reorder", choices=("none", "full", "periodic"), default="periodic" if name == "combined" else "none")
    parser.add_argument("--reorder-interval", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="Separate synchronized replay for stage timings")
    parser.add_argument("--gui", action="store_true", help="Display the response replay")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    import math
    if not math.isfinite(args.duration) or args.duration <= 0 or min(args.warmup, args.repeats, args.sample_every, args.threads) < 1:
        parser.error("duration, warmup, repeats, sample-every and threads must be positive")
    if args.resolution < 3:
        parser.error("resolution must be >= 3")
    if args.suite and args.gui:
        parser.error("use --gui for one configuration at a time")
    try:
        config = Config(grid=args.grid, dt=args.dt, iterations=args.iterations, gravity=args.gravity,
                        elastic_solver=args.solver, elastic_model=args.elastic_model,
                        compliance=args.compliance, volume_weight=args.volume_weight,
                        relaxation=args.relaxation, coupling=args.coupling, reorder=args.reorder,
                        reorder_interval=args.reorder_interval)
    except ValueError as exc:
        parser.error(str(exc))
    configs = configurations(name, config, args.suite)
    from .runner import execute, write_json
    destination = args.output / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + name)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "configurations.json", [asdict(c) for c in configs])
    summaries = []
    failed = False
    for i, c in enumerate(configs):
        label = f"{i:02d}_{c.elastic_solver}_{c.coupling}_{c.reorder}_dt{c.dt:g}_it{c.iterations}"
        try:
            result = execute(name, c, args, destination / label)
            summaries.append(dict(directory=label, config=asdict(c), **result))
        except Exception as exc:
            failed = True
            print(f"FAILED {label}: {type(exc).__name__}: {exc}", flush=True)
            summaries.append(dict(directory=label, config=asdict(c), error=str(exc)))
    write_json(destination / "comparison.json", summaries)
    print(f"Results: {destination}", flush=True)
    if failed:
        raise SystemExit(1)
