"""Reproducible chapter-4 study; no changes to the simulation equations.

Use the project .venv (Taichi 1.7.4). All output paths are explicit. Completed
cases are reused only if their configuration and solver hashes match.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.rigid import make_body, rotation_numpy
from rigid_coupling.runner import sample, save_state, source_hashes, write_json
from rigid_coupling.scenes import Scene, build, lattice
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID, RIGID, particle_arrays, combine_particles


def plan():
    cases = []
    def add(name, scene="impact", **kw):
        case = dict(name=name, scene=scene, grid=32, dt=.002, iterations=5,
                    duration=.4, sample_interval=.01, coupling="two_way",
                    offset=-.05, gap=.01, density=1., sampling="lattice", seed=42,
                    contact=False, snapshots=[0., .1, .2, .4])
        case.update(kw)
        cases.append(case)
    for n in (5, 10, 20):
        add(f"free_n{n}", "free", iterations=n, duration=1., snapshots=[0., .5, 1.])
    add("free_dt001", "free", dt=.001, duration=1., snapshots=[0., .5, 1.])
    add("impact_one", coupling="one_way")
    add("impact_base")
    add("impact_center", offset=0.)
    add("impact_mirror", offset=.05)
    for n in (2, 10, 20):
        add(f"impact_n{n}", iterations=n)
    add("impact_dt001", dt=.001)
    for rho in (.5, 2.):
        add(f"impact_rho{rho:g}", density=rho)
    for seed in (42, 43, 44):
        add(f"impact_jitter{seed}", sampling="jittered", seed=seed)
    for grid in (32, 64):
        for multiple in (0., .5, 1., 1.5, 2., 3.):
            add(f"gap_g{grid}_{multiple:g}h", "gap", grid=grid, gap=multiple/grid,
                offset=0., iterations=1, duration=.002, sample_interval=.002,
                snapshots=[0., .002])
    for name, coupling, rho in (("pool_one", "one_way", .5),
                                ("pool_light", "two_way", .5),
                                ("pool_heavy", "two_way", 2.)):
        add(name, "pool", duration=3., sample_interval=.02, coupling=coupling,
            density=rho, sampling="random", contact=True, snapshots=[0., 1., 3.])
    return cases


def make_scene(case, c):
    if case["scene"] == "pool":
        return build("buoyancy_pool", c, rigid_resolution=8,
                     fluid_ppc=2, fluid_sampling="random", seed=case["seed"])
    free = case["scene"] == "free"
    size = np.array([.18]*3 if free else [.16, .20, .20])
    counts = np.maximum(3, np.ceil(2*c.grid*size-1e-10).astype(int))
    center = np.array([.45, .5, .45] if free else [.55, .5, .5])
    local = lattice([0, 0, 0], size, counts)
    volume = np.prod(size)/len(local)
    v = np.array([.1, 0., .05] if free else [0., 0., 0.])
    omega = np.array([.4, .6, 1.2] if free else [0., 0., 0.])
    body = make_body(local, center, volume*case["density"], velocity=v, omega=omega)
    body.update(size=size.tolist(), volume=float(np.prod(size)), density=case["density"])
    local = body["local"]
    rigid = particle_arrays(center+local, RIGID, volume, v+np.cross(omega, local),
                           c.dt, (.5, .8, .55), body_id=0, local=local,
                           mass=body["particle_mass"])
    wx, wy, wz = omega
    rigid["D"][:] = c.dt*np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])
    groups = []
    sampling = dict(bodies=[dict(count=len(local), counts=counts.tolist(),
                    center=center.tolist(), size=size.tolist(), mass=body["mass"])])
    if not free:
        fs = np.array([.22, .20, .24])
        fc = np.array([center[0]-size[0]/2-case["gap"]-fs[0]/2,
                       center[1]+case["offset"], .5])
        counts = np.maximum(3, np.ceil(2*c.grid*fs-1e-10).astype(int))
        points = lattice(fc, fs, counts)
        if case["sampling"] == "jittered":
            points += np.random.default_rng(case["seed"]).uniform(-.4,.4,points.shape)*fs/counts
        fvol = np.prod(fs)/len(points)
        groups.append(particle_arrays(points, FLUID, fvol, [1,0,0], c.dt, (.2,.55,.95)))
        sampling["fluid"] = dict(count=len(points), counts=counts.tolist(),
                                 center=fc.tolist(), size=fs.tolist(), volume=float(np.prod(fs)),
                                 mode=case["sampling"], seed=case["seed"])
    groups.append(rigid)
    return Scene(combine_particles(groups), [body],
                 "Uniform-inertia free cube" if free else "Controlled fluid-block impact",
                 sampling)


def measure(solver, case):
    row = sample(solver)
    b = solver.bodies.data.to_numpy()
    p = solver.particles.snapshot()
    q = b["q"][0].astype(float)
    q /= np.linalg.norm(q)
    for axis, value in zip("wxyz", q):
        row[f"body0_q{axis}"] = float(value)
    if case["scene"] == "free":
        t = row["time"]
        initial = solver.bodies.initial[0]
        v0, w0 = initial["velocity"], initial["omega"]
        angle = np.linalg.norm(w0)*t
        exact_q = np.r_[np.cos(angle/2), np.sin(angle/2)*w0/np.linalg.norm(w0)]
        row["free_center_error"] = float(np.linalg.norm(b["x"][0]-(initial["center"]+t*v0)))
        row["free_velocity_error"] = float(np.linalg.norm(b["v"][0]-v0))
        row["free_omega_error"] = float(np.linalg.norm(b["omega"][0]-w0))
        distance = min(np.linalg.norm(q-exact_q), np.linalg.norm(q+exact_q))
        row["free_orientation_error_rad"] = float(4*np.arcsin(min(1., distance/2)))
    owned = p["body_id"] == 0
    points, local = p["x"][owned].astype(float), p["local"][owned].astype(float)
    rng = np.random.default_rng(901)
    pairs = rng.integers(0, len(points), (1024, 2))
    ref = np.linalg.norm(local[pairs[:,0]]-local[pairs[:,1]], axis=1)
    actual = np.linalg.norm(points[pairs[:,0]]-points[pairs[:,1]], axis=1)
    row["pair_distance_max_error"] = float(np.max(np.abs(actual-ref)))
    row["min_sample_to_world_boundary"] = float(min(p["x"].min()-solver.config.boundary/solver.config.grid,
                                             1-solver.config.boundary/solver.config.grid-p["x"].max()))
    return row


def run_case(case, root, arch):
    out = root/"data"/case["name"]
    hashes = source_hashes()
    if (out/"run.json").exists():
        old = json.loads((out/"run.json").read_text(encoding="utf-8"))
        if old["case"] == case and old["source_hashes"] == hashes and old["status"] == "completed":
            print("REUSE", case["name"], flush=True)
            return
        raise RuntimeError(f"Refusing to overwrite incompatible/incomplete evidence: {out}")
    out.mkdir(parents=True, exist_ok=False)
    c = Config(grid=case["grid"], dt=case["dt"], iterations=case["iterations"],
               gravity=-9.8 if case["scene"] == "pool" else 0., viscosity=0.,
               coupling=case["coupling"], rigid_density=case["density"],
               rigid_volume=case["contact"], rigid_contact=case["contact"],
               world_contact=case["contact"], buoyancy_assist=False)
    record = dict(case=case, config=asdict(c), status="running", source_hashes=hashes,
                  python=sys.version, numpy=np.__version__, taichi=".".join(map(str,ti.__version__)),
                  platform=platform.platform(), requested_backend=arch,
                  date=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  study_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/"run.json", record)
    rows = []
    try:
        ti.init(arch={"cuda":ti.cuda,"cpu":ti.cpu}[arch], enable_fallback=False,
                offline_cache=False, random_seed=42, cpu_max_num_threads=4)
        record["actual_backend"] = str(ti.lang.impl.current_cfg().arch)
        scene = make_scene(case, c)
        solver = Solver(c, scene)
        record.update(sampling=scene.sampling, particles=solver.particles.n)
        steps = int(round(case["duration"]/c.dt))
        interval = int(round(case["sample_interval"]/c.dt))
        assert np.isclose(steps*c.dt, case["duration"]) and interval > 0
        record.update(steps=steps, sample_every=interval)
        save_state(solver, out, "initial")
        # Compile on the same configuration, then restore the exact initial state.
        solver.advance(min(2, steps))
        ti.sync()
        solver.reset()
        snapshot_steps = {round(t/c.dt) for t in case["snapshots"]}
        rows.append(measure(solver, case))
        (out/"snapshots").mkdir()
        if 0 in snapshot_steps:
            save_state(solver, out/"snapshots", "step000000")
        start = perf_counter()
        for end in range(interval, steps+interval, interval):
            count = min(end, steps)-solver.step_index
            if count <= 0:
                break
            solver.advance(count)
            rows.append(measure(solver, case))
            if solver.step_index in snapshot_steps:
                save_state(solver, out/"snapshots", f"step{solver.step_index:06d}")
        ti.sync()
        record["observed_wall_seconds_including_diagnostics"] = perf_counter()-start
        record["timing_note"] = "Includes diagnostics and snapshots; not a performance benchmark."
        save_state(solver, out, "final")
        summary = dict(final=rows[-1], samples=len(rows),
                       max_shape_error=max(r["body0_shape_position_error"] for r in rows),
                       max_pair_error=max(r["pair_distance_max_error"] for r in rows),
                       max_boundary_penetration=max(r["boundary_penetration"] for r in rows))
        write_json(out/"summary.json", summary)
        record["status"] = "completed"
        print("DONE",case["name"],"particles",solver.particles.n,
              "x",round(rows[-1]["body0_x"],6),"vx",round(rows[-1]["body0_velocity_x"],6),
              "wz",round(rows[-1]["body0_omega_z"],6),
              "seconds",round(record["observed_wall_seconds_including_diagnostics"],2),flush=True)
    except Exception as e:
        record.update(status="failed",error=f"{type(e).__name__}: {e}")
        print("FAILED",case["name"],record["error"],flush=True)
        raise
    finally:
        if rows:
            with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write_json(out/"run.json",record)
        ti.reset()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--arch",default="cuda",choices=["cuda","cpu"])
    parser.add_argument("--only",nargs="*")
    parser.add_argument("--plan-only",action="store_true")
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    cases=plan()
    protocol = dict(study="ch04-rigid-coupling-20261026", analysis_intent="descriptive_numerical_comparison",
                    cases=cases, sample_scope="Recorded instants; finite/stencil checks every physical step",
                    no_buoyancy_assist=True,
                    hypotheses=["Rigid affine mapping preserves a uniform-inertia free cube's motion and shape",
                                "Feedback transfers fluid impact into rigid translation and signed rotation",
                                "Response sensitivity must be measured, not assumed absent",
                                "Shared-grid coupling can act before geometric contact",
                                "Pool results include occupancy and surface contact and test qualitative support only"])
    pp=root/"protocol.json"
    if pp.exists() and json.loads(pp.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("The recorded protocol differs; use a new output directory")
    if not pp.exists():
        write_json(pp,protocol)
        snapshot=root/"source_snapshot"
        (snapshot/"rigid_coupling").mkdir(parents=True)
        for p in Path(__file__).parent.joinpath("rigid_coupling").glob("*.py"):
            shutil.copy2(p,snapshot/"rigid_coupling"/p.name)
        shutil.copy2(__file__,snapshot/Path(__file__).name)
        result=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],
                              capture_output=True,text=True,check=False)
        write_json(root/"hardware.json",dict(gpu_query=result.stdout.strip(),returncode=result.returncode))
    if args.plan_only:
        print("RECORDED",len(cases),"cases",flush=True)
        return
    if args.only:
        unknown=set(args.only)-{c["name"] for c in cases}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases=[c for c in cases if c["name"] in args.only]
    failures=[]
    for case in cases:
        try:
            run_case(case,root,args.arch)
        except Exception as e:
            failures.append(dict(case=case["name"],error=str(e)))
    if failures:
        write_json(root/"failures.json",failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
