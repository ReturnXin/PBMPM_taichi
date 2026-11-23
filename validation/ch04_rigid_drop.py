"""Frozen reverse-impact experiment: a rigid block falls onto three circular materials."""
import argparse,csv,hashlib,json,platform,shutil,time,traceback
from pathlib import Path
import numpy as np
import taichi as ti
from rigid_coupling_2d import Solver,Config
from rigid_coupling_2d.scene import RIGID,rotation_numpy,SAMPLING_CELL
from rigid_coupling_2d.rigid_drop import (rigid_drop_config,rigid_drop_scene,
    BODY_CENTER,BODY_SIZE,BODY_DENSITY,MATERIAL_RADIUS,DURATION)
from rigid_coupling_2d.metrics import measure
from .ch04_impact_2d import write_json

ROOT=Path(__file__).resolve().parents[1]
RUNS=('water','elastic','sand','rigid_control')


def sources():
    paths=list((ROOT/'rigid_coupling_2d').glob('*.py'))+[Path(__file__),
        ROOT/'validation/ch04_rigid_drop_analyze.py',ROOT/'validation/ch04_falling_analyze.py',
        ROOT/'validation/ch04_falling.py',ROOT/'validation/ch04_impact_2d.py',
        ROOT/'validation/ch04_impact_2d_analyze.py',ROOT/'tests/test_rigid_drop_2d.py',ROOT/'tests/test_falling_2d.py',
        ROOT/'requirements-validation.txt']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    c=rigid_drop_config();a,b=rigid_drop_scene(c);f=a['material']!=RIGID
    lo=c.origin+c.boundary*c.dx
    p=dict(experiment='Rigid-body fall onto a circular material at the box floor',
        status='planned_before_execution',date='2026-11-18',runs=list(RUNS),
        config=c.as_dict(),duration=DURATION,frame_interval=.005,gif_fps=50,
        snapshot_times=[0,.10,.20,.30,.45,.65,.90],
        material_center=[.5,lo+MATERIAL_RADIUS],material_radius=MATERIAL_RADIUS,
        material_mass=float(a['mass'][f].astype(float).sum()),
        material_particles=int(f.sum()),rigid_particles=int((~f).sum()),
        body_center=BODY_CENTER.tolist(),body_size=BODY_SIZE.tolist(),
        body_mass=b['mass'],body_inertia=b['inertia'],body_density=BODY_DENSITY,
        initial_velocity=[0.,0.],offset=float(BODY_CENTER[0]-.5),
        initial_gap=float(BODY_CENTER[1]-BODY_SIZE[1]/2-lo-2*MATERIAL_RADIUS),
        support='A localized circular material touches the box floor; no material layer or pedestal.',
        initialization='Unstrained circle and rigid block both start at rest, with gravity active from t=0. No settling stage; initial material shape may change before rigid contact.',
        units='Consistent computational length L and area mass; seconds',repeats=1,warmup_steps=2,
        precision=dict(state='float32',cumulative_wall_impulse='float64',statistics='float64'),
        sampling='Shared mirrored stratified random points; cell L/256; seeds 20261117 and 20261118.',
        method='Common planar solver with separate material/rigid transfers, paired normal impulses and box walls; no buoyancy assist.',
        analysis_plan='Compare body height, vertical speed and angle; record material shape and deformation. Dry rigid control checks free fall and floor impact. Momentum budget includes gravity and box impulses.',
        limits=dict(shape=2e-6,penetration=.002,momentum_residual_fraction=.002,box_penetration=2e-6),
        elastic_integrity=dict(neighbor_radius=3*SAMPLING_CELL,max_components=1),
        environment=dict(python=platform.python_version(),taichi=ti.__version__,numpy=np.__version__,platform=platform.platform()),
        sources=sources(),human_scientific_review='pending')
    write_json(out/'protocol.json',p)
    for name in p['sources']:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    print('Prepared',out,flush=True)


def run(out,name):
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if sources()!=p['sources']:raise RuntimeError('Sources changed after prepare')
    dest=out/'data'/name;dest.mkdir(parents=True,exist_ok=False)
    ti.init(arch=ti.cuda,offline_cache=False,enable_fallback=False)
    c=Config(**p['config']);a,b=rigid_drop_scene(c,'water' if name=='rigid_control' else name,dry=name=='rigid_control')
    s=Solver(c,a,b);f=a['material']!=RIGID;m=a['mass'][f].astype(float);total_mass=float(a['mass'].astype(float).sum())
    write_json(dest/'config.json',dict(case=name,config=c.as_dict(),arch='cuda'))
    np.savez_compressed(dest/'initial.npz',**a)
    try:
        for _ in range(p['warmup_steps']):s.step()
        s.reset();rows=[];frames=[];times=[];steps=round(p['duration']/c.dt);interval=round(p['frame_interval']/c.dt)
        start=time.perf_counter()
        for i in range(steps+1):
            if i:s.step()
            s.validate();a=s.snapshot();r=measure(s,a);state=s.body_state();wall=s.wall_impulse[None].to_numpy()
            center=np.average(a['x'][f],weights=m,axis=0) if f.any() else np.zeros(2)
            r.update(material_cx=float(center[0]),material_cy=float(center[1]),
                contact_px=float(s.contact_impulse[None][0]),contact_py=float(s.contact_impulse[None][1]),
                wall_px=float(wall[0]),wall_py=float(wall[1]))
            lo=c.origin+c.boundary*c.dx;hi=c.origin+c.domain_size-c.boundary*c.dx
            extent=np.abs(rotation_numpy(state['angle']))@np.array(s.half_size)
            r['body_bottom']=float(state['center'][1]-extent[1])
            r['box_penetration']=float(max(0.,lo-a['x'].min(),a['x'].max()-hi,
                (lo-(state['center']-extent)).max(),((state['center']+extent)-hi).max()))
            r['momentum_residual']=float(np.hypot(r['pfx']+r['pbx']-wall[0],r['pfy']+r['pby']-total_mass*c.gravity*i*c.dt-wall[1]))
            rows.append(r)
            if i%interval==0 or i==steps:frames.append(a['x']);times.append(i*c.dt)
            if i and i%400==0:print(name,i,steps,flush=True)
        with (dest/'trace.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        np.savez_compressed(dest/'frames.npz',x=frames,time=times,material=a['material'])
        np.savez_compressed(dest/'final.npz',**a)
        write_json(dest/'completion.json',dict(status='complete',steps=steps,seconds=time.perf_counter()-start,last=rows[-1]))
        print(name,'complete',flush=True)
    except Exception:
        write_json(dest/'failure.json',dict(step=s.step_index,traceback=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','run'],required=True)
    parser.add_argument('--case',choices=RUNS,default='water')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.out)
    else:run(args.out,args.case)


if __name__=='__main__':main()
