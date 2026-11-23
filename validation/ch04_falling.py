"""Frozen experiment: three materials fall from rest onto a dynamic body on the box floor."""
import argparse,csv,hashlib,json,platform,shutil,time,traceback
from pathlib import Path
import numpy as np
import taichi as ti
from rigid_coupling_2d import Solver,Config
from rigid_coupling_2d.scene import RIGID,rotation_numpy,SAMPLING_CELL
from rigid_coupling_2d.falling import falling_config,falling_scene,BODY_CENTER,BODY_SIZE,MATERIAL_CENTER,MATERIAL_RADIUS,DURATION
from rigid_coupling_2d.metrics import measure
from .ch04_impact_2d import write_json

ROOT=Path(__file__).resolve().parents[1]
RUNS=('water','elastic','sand','rigid_control')


def sources():
    paths=list((ROOT/'rigid_coupling_2d').glob('*.py'))+[Path(__file__),ROOT/'validation/ch04_falling_analyze.py',ROOT/'validation/ch04_impact_2d.py',ROOT/'validation/ch04_impact_2d_analyze.py',ROOT/'tests/test_falling_2d.py',ROOT/'tests/test_materials_2d.py',ROOT/'requirements-validation.txt']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def prepare(out):
    out.mkdir(parents=True,exist_ok=False);c=falling_config();a,b=falling_scene(c)
    p=dict(experiment='Eccentric material fall onto a supported rigid body',status='planned_before_execution',date='2026-11-18',
        runs=list(RUNS),config=c.as_dict(),duration=DURATION,frame_interval=.005,snapshot_times=[0,.10,.16,.28,.45,.60,.90],
        material_center=MATERIAL_CENTER.tolist(),material_radius=MATERIAL_RADIUS,material_mass=float(a['mass'][a['material']!=RIGID].astype(float).sum()),
        body_center=BODY_CENTER.tolist(),body_size=BODY_SIZE.tolist(),body_mass=b['mass'],body_inertia=b['inertia'],
        material_particles=int((a['material']!=RIGID).sum()),rigid_particles=int((a['material']==RIGID).sum()),initial_velocity=[0,0],offset=.09375,
        support='Box floor only; no pedestal. Rigid body remains dynamic under gravity and box contact.',
        units='Consistent computational length L and area mass; seconds',repeats=1,warmup_steps=2,
        sampling='Same stratified random mirrored samples across materials; seeds 20261117 and 20261118',
        method='Shared planar APIC, material-specific local constraints, paired normal body contact; box nonpenetration and zero-restitution normal impulse; coupled two-point rigid face contact with a single-corner fallback. No buoyancy assist or constant drive.',
        analysis_plan='Compare full trajectories at physical time without shifts; control is the rigid body on the box floor without material. The original domain walls form a colliding box. Check precontact free fall and rigid rest; subtract gravity and box-wall impulses from total momentum.',
        limits=dict(shape=2e-6,penetration=.002,momentum_residual_fraction=.002),
        elastic_integrity=dict(neighbor_radius=3*SAMPLING_CELL,max_components=1,
            method='Connect particles within three sampling-cell widths at every saved frame; also inspect figures for local cracks.'),
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
    c=Config(**p['config']);a,b=falling_scene(c,'water' if name=='rigid_control' else name,name=='rigid_control')
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
            velocity=np.average(a['d'][f]/c.dt,weights=m,axis=0) if f.any() else np.zeros(2)
            r.update(material_cx=float(center[0]),material_cy=float(center[1]),material_vx=float(velocity[0]),material_vy=float(velocity[1]),
                contact_px=float(s.contact_impulse[None][0]),contact_py=float(s.contact_impulse[None][1]),
                wall_px=float(wall[0]),wall_py=float(wall[1]))
            lo=c.origin+c.boundary*c.dx;hi=c.origin+c.domain_size-c.boundary*c.dx
            extent=np.abs(rotation_numpy(state['angle']))@np.array(s.half_size)
            r['box_penetration']=float(max(0.,lo-a['x'].min(),a['x'].max()-hi,
                (lo-(state['center']-extent)).max(),((state['center']+extent)-hi).max()))
            r['momentum_residual']=float(np.hypot(r['pfx']+r['pbx']-wall[0],r['pfy']+r['pby']-total_mass*c.gravity*i*c.dt-wall[1]))
            rows.append(r)
            if i%interval==0 or i==steps:frames.append(a['x']);times.append(i*c.dt)
            if i and i%200==0:print(name,i,steps,flush=True)
        with (dest/'trace.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        np.savez_compressed(dest/'frames.npz',x=frames,time=times,material=a['material'])
        np.savez_compressed(dest/'final.npz',**a)
        write_json(dest/'completion.json',dict(status='complete',steps=steps,seconds=time.perf_counter()-start,last=rows[-1]))
        print(name,'complete',json.dumps(rows[-1]),flush=True)
    except Exception:
        write_json(dest/'failure.json',dict(step=s.step_index,traceback=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','run'],required=True);parser.add_argument('--case',choices=RUNS,default='water')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.out)
    else:run(args.out,args.case)


if __name__=='__main__':main()
