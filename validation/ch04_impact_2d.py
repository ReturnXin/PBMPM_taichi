"""Reproducible planar impact. Prepare freezes sources; cases never overwrite."""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
import traceback
from dataclasses import replace
from pathlib import Path
import numpy as np
import taichi as ti
from rigid_coupling_2d import Config, Solver, build_scene
from rigid_coupling_2d.scene import CASES, FLUID, RIGID, BODY_CENTER, BODY_SIZE, WATER_RADIUS, WATER_CENTER, SAMPLING_CELL, WATER_SEED, BODY_SEED, SPEED
from rigid_coupling_2d.metrics import measure

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CASES = ('offset_one_way','offset_two_way')
RUNS = {k: (k,1.,1) for k in (*PRIMARY_CASES,'mirror_two_way','fluid_control')}
RUNS.update(offset_half_dt=('offset_two_way',.5,1),offset_double_iterations=('offset_two_way',1.,2))


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def sources():
    paths = list((ROOT/'rigid_coupling_2d').glob('*.py'))+[Path(__file__),ROOT/'validation/ch04_impact_2d_analyze.py',ROOT/'validation/ch04_solid_impact_analyze.py',ROOT/'tests/test_rigid_coupling_2d.py',ROOT/'tests/test_materials_2d.py']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def prepare(out, c, duration, material='water'):
    out.mkdir(parents=True,exist_ok=False)
    a,b = build_scene(c,material=material)
    runs=RUNS if material=='water' else {k:RUNS[k] for k in (*PRIMARY_CASES,'fluid_control')}
    protocol = dict(experiment=f'two-dimensional circular {material} impact',material=material,status='planned_before_execution',
                    human_scientific_review='pending',units='consistent code length, area mass; configured seconds',
                    config=c.as_dict(),duration=duration,runs=runs,primary_cases=list(PRIMARY_CASES),
                    frame_interval=.01,snapshot_times=[0.,.16,.24,.34,.5],
                    water_center=WATER_CENTER.tolist(),water_radius=WATER_RADIUS,water_mass=float(a['mass'][a['material']!=RIGID].astype(float).sum()),
                    body_center=BODY_CENTER.tolist(),body_size=BODY_SIZE.tolist(),body_mass=b['mass'],body_inertia=b['inertia'],
                    water_particles=int((a['material']!=RIGID).sum()),rigid_particles=int((a['material']==RIGID).sum()),
                    sampling=dict(cell=SAMPLING_CELL,seeds=[WATER_SEED,BODY_SEED],method='stratified random samples reflected across two center axes'),
                    speed=SPEED,offset=.0625,repeats=1,warmup_steps=2,
                    method='2D area constraint; nine-node quadratic APIC; separate motion; shared rest-area occupancy; full transfer each iteration; paired normal contact; symmetric contact strain',
                    metrics='Every physical step; float64 reduction; kinetic energy excludes separately recorded APIC affine norm',
                    analysis_plan='Compare complete 0..duration trajectories, no time shifts. Sensitivity at matching times, excluding t=0; normalize position by .25 and speed by .75.',
                    checks=dict(finite=True,shape_error=2e-6,min_stencil_clearance=1,momentum_diagnostic=.01,penetration_diagnostic_dx=.05),
                    preflight_adjustment='Pilot domain [-.25,1.25]^2 at 128^2 touched truncated wall nodes in center_one_way. Expanded to [-.625,1.625]^2 at 192^2; dx=1.5/128 unchanged. Historical pilot outputs were removed at user request on 2026-11-17.',
                    exclusions=['gravity','viscosity','buoyancy assist','constant driving force'],sources=sources(),
                    environment=dict(python=platform.python_version(),taichi=ti.__version__,numpy=np.__version__,platform=platform.platform()))
    if material!='water':
        protocol.update(method='2D chapter-two shape/area target; elastic XPBMPM or sand logarithmic plastic return; nine-node quadratic APIC; separate continuum/rigid channels; paired normal contact; symmetric contact strain',
                        exclusions=['gravity','buoyancy assist','constant driving force'],
                        preflight_adjustment='Same geometry, particle realization, mass, initial velocity, dt and iterations as retained water experiment. Solid extension checked by analytical unit tests before execution.',
                        material_field_note='Historical water_* protocol keys and fluid_* CSV keys refer to the incoming continuum of the selected material.',
                        analysis_plan='Compare single/two-way impact and free-material control over full 0..duration trajectories. Report rigid translation/rotation, total linear momentum, true area det(F_total), recoverable log-strain and cumulative plastic correction. Kinetic energy excludes elastic stored energy; do not interpret it as total solid energy.')
    write_json(out/'protocol.json',protocol)
    for name in protocol['sources']:
        dest=out/'source_snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    print(json.dumps({k:protocol[k] for k in ('water_particles','rigid_particles','water_mass','body_mass')},indent=2),flush=True)


def run(out, name, arch):
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if sources()!=protocol['sources']: raise RuntimeError('Sources changed since prepare; use a new output directory')
    case,dt_factor,it_factor=protocol['runs'][name]
    c=Config(**protocol['config'])
    c=replace(c,coupling=CASES[case][1],dt=c.dt*dt_factor,iterations=c.iterations*it_factor)
    dest=out/'data'/name
    dest.mkdir(parents=True,exist_ok=False)
    material=protocol.get('material','water')
    write_json(dest/'config.json',dict(case=case,material=material,config=c.as_dict(),arch=arch))
    ti.init(arch=ti.cuda if arch=='cuda' else ti.cpu,offline_cache=False,**({'cpu_max_num_threads':1} if arch=='cpu' else {}))
    if arch=='cuda' and ti.lang.impl.current_cfg().arch!=ti.cuda: raise RuntimeError('CUDA unavailable')
    a,b=build_scene(c,case,material)
    s=Solver(c,a,b)
    np.savez_compressed(dest/'initial.npz',**s.snapshot())
    try:
        for _ in range(protocol['warmup_steps']): s.step()
        s.reset()
        steps=round(protocol['duration']/c.dt)
        if not np.isclose(steps*c.dt,protocol['duration']): raise ValueError('duration must be an integer number of steps')
        rows,frames,times=[],[],[]
        frame_steps=max(1,round(protocol['frame_interval']/c.dt))
        start=time.perf_counter()
        for step in range(steps+1):
            if step: s.step()
            s.validate()
            p=s.snapshot()
            rows.append(measure(s,p))
            if step%frame_steps==0 or step==steps:
                frames.append(p['x']);times.append(step*c.dt)
            if step and step%100==0: print(f'{name} {step}/{steps}',flush=True)
        with (dest/'trace.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        np.savez_compressed(dest/'frames.npz',x=np.array(frames),time=times,material=a['material'])
        np.savez_compressed(dest/'final.npz',**s.snapshot())
        write_json(dest/'completion.json',dict(status='complete',steps=steps,elapsed_seconds=time.perf_counter()-start,
                                             purpose='simulation and recording, not a performance benchmark',last=rows[-1]))
        print(json.dumps(rows[-1],indent=2),flush=True)
    except Exception:
        write_json(dest/'failure.json',dict(traceback=traceback.format_exc(),step=s.step_index))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--case',choices=list(RUNS),default='offset_two_way')
    p.add_argument('--arch',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--grid',type=int,default=192)
    p.add_argument('--material',choices=['water','elastic','sand'],default='water')
    p.add_argument('--compliance',type=float,default=Config().compliance)
    p.add_argument('--duration',type=float,default=.5)
    a=p.parse_args()
    if a.stage=='prepare': prepare(a.out,Config(grid=a.grid,compliance=a.compliance),a.duration,a.material)
    else: run(a.out,a.case,a.arch)


if __name__=='__main__': main()
