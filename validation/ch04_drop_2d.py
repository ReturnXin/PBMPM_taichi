"""Release a planar rigid body above settled water; save complete diagnostics."""
import argparse
import csv
import hashlib
import json
import shutil
import platform
import time
import traceback
from pathlib import Path
import numpy as np
import taichi as ti
from rigid_coupling_2d import Solver
from rigid_coupling_2d.drop import DropSpec,drop_config,pool,drop_scene,BODY_SIZE
from rigid_coupling_2d.scene import FLUID,RIGID,rotation_numpy

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT.parents[1]/'毕业论文/实验结果/第四章/刚体落水'


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(s):
    a=s.snapshot();f=a['material']==FLUID;b=s.body_state()
    x=a['x'][f].astype(float);v=a['d'][f].astype(float)/s.c.dt
    lo=s.c.origin+s.c.boundary*s.c.dx;hi=s.c.origin+(s.c.grid-s.c.boundary)*s.c.dx
    bottom,wall,shape,pen=0.,0.,0.,0.
    if s.has_body:
        R=rotation_numpy(b['angle']);half=np.asarray(s.half_size);extent=abs(R)@half
        bottom=float(b['center'][1]-extent[1])
        wall=float(np.min(np.r_[b['center']-extent-lo,hi-b['center']-extent]))
        r=a['material']==RIGID
        shape=float(np.max(np.linalg.norm(a['x'][r].astype(float)-(b['center']+a['local'][r]@R.T),axis=1)))
        if len(x):pen=float(np.maximum(np.min(half-abs((x-b['center'])@R),axis=1),0).max())
    return dict(time=s.step_index*s.c.dt,cx=float(b['center'][0]),cy=float(b['center'][1]),
                vx=float(b['velocity'][0]),vy=float(b['velocity'][1]),angle=b['angle'],omega=b['omega'],
                body_bottom=bottom,body_wall_clearance=wall,shape_error=shape,penetration=pen,
                contact_px=float(s.contact_impulse[None][0]),contact_py=float(s.contact_impulse[None][1]),
                fluid_rms_speed=float(np.sqrt(np.mean(np.sum(v*v,axis=1)))) if len(x) else 0.,
                water_max_y=float(x[:,1].max()) if len(x) else 0.,
                water_surface_p99=float(np.quantile(x[:,1],.99)) if len(x) else 0.,
                water_area=float(np.sum(a['area'][f].astype(float)*a['J'][f])) if len(x) else 0.,
                cfl=float(np.max(np.linalg.norm(a['d'],axis=1))/s.c.dx))


def write_trace(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def prepare(out,c,spec):
    for name,value in [('duration',spec.duration),('settle_time',spec.settle_time),('frame_interval',.01)]:
        if not np.isclose(round(value/c.dt)*c.dt,value,rtol=0,atol=1e-12):
            raise ValueError(f'{name} must be a multiple of dt')
    out.mkdir(parents=True,exist_ok=False)
    a,b=drop_scene(c,spec)
    files=list((ROOT/'rigid_coupling_2d').glob('*.py'))+[Path(__file__),ROOT/'validation/ch04_drop_2d_analyze.py',ROOT/'tests/test_drop_2d.py']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files if p.exists()}
    write_json(out/'protocol.json',dict(date='2026-11-18',stage='prepared_before_execution',config=c.as_dict(),spec=spec.as_dict(),
               body_size=BODY_SIZE.tolist(),water_particles=int((a['material']==FLUID).sum()),
               rigid_particles=int((a['material']==RIGID).sum()),body_mass=b['mass'],body_inertia=b['inertia'],
               initialization='Settle water alone, preserving x,d,D,J; then insert a motionless body and start t=0.',
               comparisons=['water_entry','pool_control','dry_fall'],frame_interval=.01,
               diagnostics='Every physical step; contact impulses, penetration, body shape, wall clearance, fluid RMS speed and CFL.',
               exclusions=['buoyancy assist','added water driving force','restitution bonus','prescribed body trajectory'],
               gravity_reference='Semi-implicit y_n=y0+g*dt^2*n*(n+1)/2, v_n=g*n*dt; compare dry body only before wall contact.',
               units='Consistent code units; mass = area times density',sources=hashes,human_scientific_review='pending'))
    for name in hashes:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)


def run(out,arch):
    start=time.perf_counter()
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    for name,h in protocol['sources'].items():
        if digest(ROOT/name)!=h:raise RuntimeError(f'Source changed after prepare: {name}')
    from rigid_coupling_2d.scene import Config
    c=Config(**protocol['config']);spec=DropSpec(**protocol['spec'])
    ti.init(arch=ti.cuda if arch=='cuda' else ti.cpu,offline_cache=False,enable_fallback=False)
    data=out/'data';data.mkdir(exist_ok=False)
    write_json(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
               numpy=np.__version__,taichi=ti.__version__,arch=arch,float_type='float32'))
    # Both observed water runs start from this exact settled state.
    settling=Solver(c,pool(c,spec));rows=[measure(settling)]
    for step in range(round(spec.settle_time/c.dt)):
        settling.step();settling.validate();rows.append(measure(settling))
        if (step+1)%200==0:print(f'settling {step+1}/{round(spec.settle_time/c.dt)}',flush=True)
    water=settling.snapshot();np.savez_compressed(data/'settled_pool.npz',**water)
    write_trace(data/'settling.csv',rows)
    steps=round(spec.duration/c.dt);stride=round(.01/c.dt)
    for case in ('water_entry','pool_control','dry_fall'):
        dest=data/case;dest.mkdir()
        if case=='pool_control':a,b=water,None
        else:a,b=drop_scene(c,spec,water,dry=case=='dry_fall')
        s=Solver(c,a,b);np.savez_compressed(dest/'initial.npz',**s.snapshot())
        rows=[];frames=[];times=[]
        for step in range(steps+1):
            if step:s.step()
            s.validate();rows.append(measure(s))
            if step%stride==0 or step==steps:
                frames.append(s.x.to_numpy());times.append(step*c.dt)
            if step and step%200==0:print(f'{case} {step}/{steps}',flush=True)
        write_trace(dest/'trace.csv',rows)
        np.savez_compressed(dest/'frames.npz',x=frames,time=times,material=a['material'])
        np.savez_compressed(dest/'final.npz',**s.snapshot())
        write_json(dest/'completion.json',dict(complete=True,steps=steps,finite=True,last=rows[-1]))
    print('All three drop cases completed.',flush=True)
    write_json(out/'completion.json',dict(complete=True,elapsed_wall_seconds=time.perf_counter()-start))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--arch',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--duration',type=float,default=.8)
    p.add_argument('--settle-time',type=float,default=.5)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.out,drop_config(),DropSpec(duration=a.duration,settle_time=a.settle_time))
    else:
        try:run(a.out,a.arch)
        except Exception:
            if a.out.is_dir():write_json(a.out/'failure.json',dict(traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()
