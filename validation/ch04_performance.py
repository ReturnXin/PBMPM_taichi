"""Warmed, interleaved performance comparison for the production water solver."""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.runner import source_hashes, write_json
from rigid_coupling.scenes import Scene
from rigid_coupling.shake import build_shake_scene
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID

RESOLUTIONS = [24, 36, 48]
BATCHES, STEPS, CALL_STEPS, WARMUP = 7, 80, 2, 20


def config():
    return Config(grid=32, dt=.002, iterations=5, gravity=-9.8,
                  fluid_relaxation=1., viscosity=0., rigid_density=.35,
                  rigid_volume=True, rigid_contact=True, world_contact=True,
                  buoyancy_assist=False, grid_coupling='separate',
                  boundary_density='available', wall_transfer='reflected',
                  density_feedback='clamped')


def scene(resolution, bodies):
    full = build_shake_scene(config(), FLUID, fluid_resolution=resolution, seed=42)
    if bodies == 3:
        return full
    mask = full.particles['material']==FLUID
    return Scene({key:value[mask].copy() for key,value in full.particles.items()}, [],
                 'Same initial water particles with the three bodies removed',
                 container=deepcopy(full.container))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_state():
    fields = 'name,driver_version,memory.total,utilization.gpu,pstate,clocks.current.graphics,temperature.gpu,power.draw'
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu='+fields, '--format=csv'],
                                capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return {'unavailable':str(error)}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    snap = out/'source_snapshot'
    shutil.copytree(root/'rigid_coupling', snap/'rigid_coupling',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (snap/'validation').mkdir()
    (snap/'validation/__init__.py').write_text('', encoding='utf-8')
    shutil.copy2(__file__,snap/'validation'/Path(__file__).name)
    counts = {}
    for resolution in RESOLUTIONS:
        a, b = scene(resolution,0), scene(resolution,3)
        mask = b.particles['material']==FLUID
        assert all(np.array_equal(a.particles[key], b.particles[key][mask]) for key in a.particles)
        counts[resolution] = dict(fluid=len(a.particles['x']), total_with_bodies=len(b.particles['x']),
                                  rigid_samples=len(b.particles['x'])-len(a.particles['x']),
                                  identical_initial_water=True)
    protocol = dict(date='2026-11-11', status='planned_before_measurement',
        experiment='chapter 4 computational performance', analysis_intent='descriptive performance comparison',
        human_scientific_review='pending', config=config().as_dict(), resolutions=RESOLUTIONS,
        counts=counts, body_counts=[0,3], dispatch_modes=['ordinary','graph'], seed=42,
        scene='Production shake_water scene with fixed container. The control removes rigid samples and bodies, retaining exactly the same water particles, including the original excluded submerged regions.',
        geometry=dict(container_center=[.5,.48,.5],container_half_size=[.25,.27,.23],
                      water_size=[.43,.17,.38],body_size=[.09,.065,.09],body_samples_each=64),
        warmup_steps_per_mode=WARMUP, checkpoint_step=WARMUP,
        timed_steps=STEPS, steps_per_host_call=CALL_STEPS, repetitions=BATCHES,
        timed_physical_interval=[WARMUP*.002,(WARMUP+STEPS)*.002],
        timing='perf_counter around 40 advance(2) calls, with ti.sync before and after. Includes ordinary Python/native dispatch, prescribed box pose publication and production error checks; excludes compilation, warmup, state restoration, snapshots, disk I/O and rendering.',
        order='Within each resolution, rotate [0 ordinary,0 graph,3 ordinary,3 graph] by batch index modulo 4.',
        state='Warm ordinary once, save its full particle/body state at step 20; replay each mode from this checkpoint. Fixed container pose and time restored; derived grids recomputed by full transfers.',
        metrics=['median and range of ms/physical step', 'paired graph speedup',
                 'whole-step overhead of adding three bodies', 'ordinary/graph replay differences'],
        summary='Median of seven batch means per configuration; min/max describe timing variation, not statistical confidence intervals.',
        parity_steps=10, parity_tolerances=dict(particle_atol=4e-6,particle_rtol=4e-5,
                   body_pose_atol=3e-6,body_pose_rtol=3e-5,dt_body_v_error=1e-7,dt_body_omega_error=1e-6),
        failure_policy='Retain incomplete or failed runs. Record any amended protocol before additional measurements.',
        exclusions='No rendering, GUI frame cap, mouse forcing, buoyancy assistance, profiler or concurrent benchmark.',
        production_hashes=source_hashes(), experiment_sha256=digest(Path(__file__)),
        baseline_gpu=gpu_state())
    write_json(out/'protocol.json',protocol)
    print(json.dumps(counts,indent=2),flush=True)


def capture(s):
    return dict(particles=s.particles.snapshot(), bodies=s.bodies.data.to_numpy(),
                wall=s.container.wall_impulse[None].to_numpy(),
                motion=deepcopy(s.container.motion.__dict__), step=s.step_index)


def restore(s, checkpoint):
    s.reset()
    s.particles.restore(checkpoint['particles'])
    s.bodies.data.from_numpy(checkpoint['bodies'])
    s.container.motion.__dict__.update(deepcopy(checkpoint['motion']))
    s.container.publish()
    s.container.wall_impulse[None] = checkpoint['wall']
    s.step_index = checkpoint['step']


def save_capture(path, value):
    np.savez_compressed(path, **value['particles'],
                        **{'body_'+key:arr for key,arr in value['bodies'].items()},
                        wall_impulse=value['wall'])


def compare(a,b,c,count):
    errors = {}
    for key in ['x','d','D','volume_ratio']:
        left,right=a['particles'][key],b['particles'][key]
        errors[key]=float(abs(left.astype(float)-right).max())
        np.testing.assert_allclose(left,right,atol=4e-6,rtol=4e-5,err_msg=key)
    if count:
        for key in ['x','q','v','omega']:
            errors['body_'+key]=float(abs(a['bodies'][key].astype(float)-b['bodies'][key]).max())
        for key in ['x','q']:
            np.testing.assert_allclose(a['bodies'][key],b['bodies'][key],atol=3e-6,rtol=3e-5)
        assert c.dt*errors['body_v']<1e-7 and c.dt*errors['body_omega']<1e-6
    errors['wall_impulse']=float(abs(a['wall'].astype(float)-b['wall']).max())
    np.testing.assert_allclose(a['wall'],b['wall'],atol=3e-6,rtol=3e-4)
    return errors


def run(out,resolution):
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['production_hashes']==source_hashes()
    assert protocol['experiment_sha256']==digest(Path(__file__))
    folder=out/f'resolution_{resolution}';folder.mkdir(exist_ok=False)
    ti.init(arch=ti.cuda,enable_fallback=False,offline_cache=False,kernel_profiler=False,
            random_seed=42,cpu_max_num_threads=4)
    environment=dict(python=sys.version,numpy=np.__version__,taichi=ti.__version__,
                     platform=platform.platform(),machine=platform.machine(),
                     logical_processors=os.cpu_count(),arch=str(ti.lang.impl.current_cfg().arch),
                     profiler=False,gpu_before=gpu_state(),argv=sys.argv)
    write_json(folder/'environment.json',environment)
    solvers,graphs,checkpoints,parity,rows={},{},{},{},[]
    try:
        for count in [0,3]:
            s=Solver(config(),scene(resolution,count))
            print(f'resolution {resolution}, bodies {count}: compiling ordinary path',flush=True)
            start=perf_counter();s.advance(WARMUP);ti.sync()
            checkpoints[count]=capture(s)
            save_capture(folder/f'checkpoint_bodies{count}.npz',checkpoints[count])
            print(f'resolution {resolution}, bodies {count}: compiling graph path',flush=True)
            s.enable_cuda_graph();graphs[count]=s.step_graph
            restore(s,checkpoints[count]);s.advance(WARMUP);ti.sync()
            print(f'resolution {resolution}, bodies {count}: warmup complete ({perf_counter()-start:.1f}s)',flush=True)
            outcomes={}
            for mode in ['ordinary','graph']:
                s.step_graph=None if mode=='ordinary' else graphs[count]
                restore(s,checkpoints[count]);s.advance(10)
                outcomes[mode]=capture(s)
                save_capture(folder/f'parity_bodies{count}_{mode}.npz',outcomes[mode])
            parity[count]=compare(outcomes['ordinary'],outcomes['graph'],s.config,count)
            solvers[count]=s
        write_json(folder/'parity.json',dict(passed=True,steps=10,max_absolute_errors=parity))
        order=[(0,'ordinary'),(0,'graph'),(3,'ordinary'),(3,'graph')]
        with (folder/'timings.csv').open('w',encoding='utf-8',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=['batch','order','bodies','dispatch','elapsed_s','ms_per_step','steps','fluid_particles','total_particles'])
            writer.writeheader()
            for batch in range(BATCHES):
                shift=batch%4
                for ordinal,(count,mode) in enumerate(order[shift:]+order[:shift]):
                    s=solvers[count];s.step_graph=None if mode=='ordinary' else graphs[count]
                    restore(s,checkpoints[count]);ti.sync()
                    start=perf_counter()
                    for _ in range(STEPS//CALL_STEPS):
                        s.advance(CALL_STEPS)
                    ti.sync();elapsed=perf_counter()-start
                    row=dict(batch=batch,order=ordinal,bodies=count,dispatch=mode,elapsed_s=elapsed,
                             ms_per_step=elapsed*1000/STEPS,steps=STEPS,
                             fluid_particles=int((s.initial['material']==FLUID).sum()),total_particles=s.particles.n)
                    writer.writerow(row);file.flush();rows.append(row)
                    end=capture(s)
                    assert s.step_index==WARMUP+STEPS
                    for arrays in [end['particles'],end['bodies']]:
                        assert all(np.isfinite(value).all() for value in arrays.values())
                    if batch==BATCHES-1:
                        save_capture(folder/f'final_bodies{count}_{mode}.npz',end)
                print(f'resolution {resolution}: batch {batch+1}/{BATCHES}, '+', '.join(
                      f'{r["bodies"]} bodies/{r["dispatch"]} {r["ms_per_step"]:.3f} ms' for r in rows[-4:]),flush=True)
        summary={}
        for count,mode in order:
            values=[r['ms_per_step'] for r in rows if r['bodies']==count and r['dispatch']==mode]
            summary[f'bodies{count}_{mode}']=dict(median_ms=float(np.median(values)),
                  min_ms=min(values),max_ms=max(values),samples_ms=values)
        write_json(folder/'summary.json',dict(completed=True,summary=summary,parity_passed=True,
                   source_unchanged=protocol['production_hashes']==source_hashes(),gpu_after=gpu_state()))
        print(json.dumps(summary,indent=2),flush=True)
    except Exception:
        write_json(folder/'failure.json',dict(completed=False,timing_rows=len(rows),traceback=traceback.format_exc()))
        raise
    finally:
        ti.reset()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','run'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolution',type=int,choices=RESOLUTIONS,default=24)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output)
    else:run(args.output,args.resolution)


if __name__=='__main__':main()
