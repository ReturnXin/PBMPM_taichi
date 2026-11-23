"""Chapter 4 experiment 1: isolated water impact using the production Solver.

Run `--stage prepare` before any simulation. Outputs are append-only per case;
existing case folders are never overwritten. All quantities use code units.
"""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.rigid import RigidBodies, make_body, rotation_numpy
from rigid_coupling.runner import source_hashes, write_json
from rigid_coupling.scenes import Scene, lattice
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID, RIGID, Particles, combine_particles, particle_arrays

CASES = {
    'center_one_way': (0., 'one_way', True),
    'center_two_way': (0., 'two_way', True),
    'offset_one_way': (.0625, 'one_way', True),
    'offset_two_way': (.0625, 'two_way', True),
    'mirror_two_way': (-.0625, 'two_way', True),
    'fluid_control': (0., 'two_way', False),
}
BODY_CENTER = np.array([.625, .5, .5])
BODY_SIZE = np.array([.125, .25, .1875])
WATER_CENTER = np.array([.328125, .5, .5])
WATER_VOLUME = .21875 * .125 * .125
WATER_RADIUS = (3 * WATER_VOLUME / (4 * np.pi)) ** (1 / 3)
WATER_SIZE = np.full(3, 2 * WATER_RADIUS)
DURATION = .5
SPEED = .75
SAMPLING_CELL = 1 / 128
WATER_SEED = 20261117
BODY_SEED = 20261118


def random_box(size, seed):
    """One random point per cell, reflected across the three central planes.

    Reflection preserves the centered/mirrored impact controls and zero initial
    mass offset. The RNG seed fixes the same samples for every coupling variant.
    """
    size = np.asarray(size, dtype=float)
    counts = np.rint(size / SAMPLING_CELL).astype(int)
    assert np.all(counts % 2 == 0)
    assert np.allclose(counts * SAMPLING_CELL, size, rtol=0, atol=1e-14)
    cells = np.indices(counts // 2).reshape(3, -1).T
    rng = np.random.default_rng(seed)
    positive = (cells + rng.random(cells.shape)) * SAMPLING_CELL
    signs = np.array(list(product((-1., 1.), repeat=3)))
    return (positive[None, :, :] * signs[:, None, :]).reshape(-1, 3)


def random_sphere():
    """Crop stratified random samples to an equal-volume sphere."""
    half_cells = int(np.ceil(WATER_RADIUS / SAMPLING_CELL))
    local = random_box(np.full(3, 2 * half_cells * SAMPLING_CELL), WATER_SEED)
    return local[np.linalg.norm(local, axis=1) < WATER_RADIUS]


def config(mode='two_way', dt=.002, iterations=5):
    return Config(grid=32, boundary=2, dt=dt, iterations=iterations, gravity=0.,
                  fluid_relaxation=1., viscosity=0., coupling=mode,
                  fluid_density=1., rigid_density=1., rigid_volume=True,
                  rigid_contact=True, world_contact=True, buoyancy_assist=False,
                  grid_coupling='separate', boundary_density='raw',
                  wall_transfer='clamp', density_feedback='legacy')


def scene(c, offset=0., with_body=True):
    center = WATER_CENTER + [0., offset, 0.]
    water = center + random_sphere()
    volume = SAMPLING_CELL**3
    water_volume = WATER_VOLUME / len(water)
    groups = [particle_arrays(water, FLUID, water_volume, (SPEED, 0., 0.),
                              c.dt, (.13, .48, .82), mass=water_volume)]
    bodies = []
    if with_body:
        local = random_box(BODY_SIZE, BODY_SEED)
        body = make_body(local, BODY_CENTER, volume)
        body.update(size=BODY_SIZE.copy(), density=1.)
        groups.append(particle_arrays(body['center'] + body['local'], RIGID,
                                      volume, (0., 0., 0.), c.dt, (.9, .48, .12),
                                      body_id=0, local=body['local'], mass=body['particle_mass']))
        bodies.append(body)
    return Scene(combine_particles(groups), bodies, 'Isolated horizontal water impact')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    snapshot = out / 'source_snapshot'
    shutil.copytree(root / 'rigid_coupling', snapshot / 'rigid_coupling',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (snapshot / 'validation').mkdir()
    for name in ['ch04_impact.py', 'ch04_impact_analyze.py', 'ch04_impact_report.py']:
        shutil.copy2(root / 'validation' / name, snapshot / 'validation' / name)
    (snapshot / 'validation' / '__init__.py').write_text('', encoding='utf-8')
    c = config()
    s = scene(c)
    protocol = dict(
        experiment='ch04 experiment 1: spherical water impact',
        date=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),
        variant='sphere',
        status='planned_before_execution', analysis_intent='mechanism validation and descriptive comparison',
        human_scientific_review='pending', units='consistent code length, mass; time in configured seconds',
        baseline=c.as_dict(), duration=DURATION, steps=round(DURATION/c.dt),
        sample_every_physical_step=True, snapshot_times=[0., .16, .24, .34, .5],
        animation_every_steps=5, cases=CASES, primary_cases=list(CASES)[:4],
        controls=list(CASES)[4:], water_center=WATER_CENTER.tolist(), water_size=WATER_SIZE.tolist(),
        water_shape='sphere', water_radius=WATER_RADIUS, water_volume=WATER_VOLUME,
        initial_surface_gap=float(BODY_CENTER[0]-BODY_SIZE[0]/2-WATER_CENTER[0]-WATER_RADIUS),
        body_center=BODY_CENTER.tolist(), body_size=BODY_SIZE.tolist(), speed=SPEED,
        water_particle_count=int((s.particles['material']==FLUID).sum()),
        body_particle_count=int((s.particles['material']==RIGID).sum()),
        sampling=dict(method='one uniformly random point per cell; octant reflections; water cropped to sphere',
                      cell_size=SAMPLING_CELL, water_seed=WATER_SEED, body_seed=BODY_SEED,
                      generator='NumPy default_rng PCG64',
                      symmetry='reflected about x/y/z center planes',
                      water_particle_volume=WATER_VOLUME/int((s.particles['material']==FLUID).sum()),
                      rigid_particle_volume=SAMPLING_CELL**3),
        water_mass=float(s.particles['mass'][s.particles['material']==FLUID].astype(float).sum()),
        body_mass=s.bodies[0]['mass'],
        body_inertia=s.bodies[0]['inertia'].tolist(),
        exclusions='No gravity, buoyancy assist, viscosity, imposed body motion, or container motion.',
        scope='Local separate-grid coupling. The reflected container-wall path is not exercised.',
        one_way='Uses existing one_way switch: body feedback and inverse effective body mass are disabled; body starts at rest.',
        metrics=['body x/v/omega/q', 'fluid and body linear momentum', 'APIC-inclusive angular momentum',
                 'cumulative contact/grid impulses', 'post-step particle penetration', 'volume change',
                 'shape reconstruction error', 'boundary stencil clearance'],
        reference_origin=BODY_CENTER.tolist(), momentum_scale='initial fluid mass * 0.75',
        angular_scale='initial fluid mass * 0.75 * 0.25',
        angular_definition='sum((x-origin) cross m*v) + sum(m*h^2/(4*dt)*axial(D-D.T)) for fluid; body orbital + I_world*omega.',
        primary_windows='Full 0..0.5 s, identical physical times, no time shifting or selected tail window.',
        numerical_checks=dict(impulse_rtol=5e-5, impulse_atol=2e-7,
                              impulse_check_particles=192,
                              impulse_check_sampling='original 4x8x6 cell-centered analytical fixture',
                              shape_max=2e-6, allowed_boundary_stencil_nodes=0,
                              penetration_diagnostic_trigger_h=.05,
                              linear_momentum_diagnostic_trigger=.01),
        interpretation='Response magnitude is descriptive, not a physical accuracy claim. One-way momentum loss is an omitted external reaction.',
        sampling_note='Penetration is sampled after each committed step; no continuous collision or sub-iteration bound is claimed.',
        sensitivity='After the six cases, offset_two_way is additionally run at dt=.001/5 iterations and dt=.002/10 iterations. Changes are separate.',
        repeats='One full trajectory per configuration; no statistical confidence interval or performance claim.',
        failure_policy='Keep failed and incomplete cases. Record any parameter revision before a new run; do not overwrite.',
        production_hashes=source_hashes(), experiment_sha256=digest(__file__))
    write_json(out/'protocol.json', protocol)
    print(json.dumps({k:protocol[k] for k in ['water_particle_count','body_particle_count','body_mass','steps']}, indent=2), flush=True)


def verify_sources(out):
    p = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if p['production_hashes'] != source_hashes():
        raise RuntimeError('Production sources changed after protocol preparation')
    if p['experiment_sha256'] != digest(__file__):
        raise RuntimeError('Experiment runner changed after protocol preparation')
    return p


def impulse_checks(out):
    target = out/'impulse_checks'
    target.mkdir(exist_ok=False)
    results = []
    for angle in [0., .63]:
        c = config()
        # Keep the original small analytical fixture separate from scene sampling.
        # Its tolerance tests the feedback formula, not high-count float32 sums.
        local = lattice((0,0,0), BODY_SIZE, (4,8,6))
        b = make_body(local, BODY_CENTER, np.prod(BODY_SIZE)/len(local),
                      quaternion=(np.cos(angle/2), 0., 0., np.sin(angle/2)))
        b['size'] = BODY_SIZE
        R = rotation_numpy(b['quaternion'])
        arrays = particle_arrays(b['center']+b['local']@R.T, RIGID, np.prod(BODY_SIZE)/len(local),
                                 (0.,0.,0.), c.dt, (.9,.48,.12), body_id=0,
                                 local=b['local'], mass=b['particle_mass'])
        particles = Particles(arrays)
        bodies = RigidBodies(particles,c,[b])
        for kind in ['center_distributed','eccentric_sample']:
            particles.restore(arrays); bodies.restore(); bodies.broadcast()
            p = particles.snapshot()
            impulse = np.array([2.e-5, -1.e-5, 1.5e-5])
            if kind == 'center_distributed':
                p['d'] += (c.dt*impulse/b['mass']).astype(np.float32)
                angular = np.zeros(3)
            else:
                index = len(local)-1
                p['d'][index] += (c.dt*impulse/p['mass'][index]).astype(np.float32)
                angular = np.cross(R@p['local'][index].astype(float),impulse)
            particles.data.from_numpy(p)
            bodies.project()
            actual = bodies.data.to_numpy()
            expected_v=impulse/b['mass']
            expected_w=np.linalg.solve(R@b['inertia']@R.T,angular)
            dv=actual['v'][0].astype(float); dw=actual['omega'][0].astype(float)
            passed=bool(np.allclose(dv,expected_v,rtol=5e-5,atol=2e-7) and
                        np.allclose(dw,expected_w,rtol=5e-5,atol=2e-7))
            results.append(dict(angle=angle,kind=kind,passed=passed,
                                expected_v=expected_v.tolist(),actual_v=dv.tolist(),
                                expected_omega=expected_w.tolist(),actual_omega=dw.tolist(),
                                max_v_error=float(abs(dv-expected_v).max()),
                                max_omega_error=float(abs(dw-expected_w).max())))
    write_json(target/'results.json',dict(checks=results,passed=all(r['passed'] for r in results)))
    if not all(r['passed'] for r in results):
        raise AssertionError('Analytical impulse checks failed')
    print('Analytical impulse checks: 4/4 passed',flush=True)


def sample(s):
    p=s.particles.snapshot(); b=s.bodies.data.to_numpy(); c=s.config
    for arrays in [p,b]:
        for key,value in arrays.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite {key}')
    f=p['material']==FLUID; r=~f
    x=p['x'][f].astype(float); m=p['mass'][f].astype(float)
    v=p['d'][f].astype(float)/c.dt; D=p['D'][f].astype(float)
    pf=(m[:,None]*v).sum(axis=0)
    lf=np.cross(x-BODY_CENTER,m[:,None]*v).sum(axis=0)
    spin=np.stack([D[:,2,1]-D[:,1,2],D[:,0,2]-D[:,2,0],D[:,1,0]-D[:,0,1]],axis=1)
    spin=(m[:,None]*spin).sum(axis=0)/(4*c.grid**2*c.dt)
    base=np.floor(p['x'].astype(float)*c.grid-.5)
    clearance=float(min((base-c.boundary).min(),(c.grid-c.boundary-base-2).min()))
    row=dict(step=s.step_index,time=s.step_index*c.dt,fluid_mass=float(m.sum()),
             fluid_volume=float((p['rest_volume'][f].astype(float)*p['volume_ratio'][f]).sum()),
             fluid_j_min=float(p['volume_ratio'][f].min()),fluid_j_max=float(p['volume_ratio'][f].max()),
             boundary_stencil_clearance=clearance,penetration=0.,inside_count=0,
             shape_error=0.,quaternion_error=0.,world_contacts=0.)
    pb=np.zeros(3); lb=np.zeros(3)
    vectors={'fluid_p':pf,'fluid_l_orbital':lf,'fluid_l_affine':spin,'fluid_center':np.average(x,weights=m,axis=0)}
    if s.bodies.count:
        center=b['x'][0].astype(float); R=rotation_numpy(b['q'][0]); velocity=b['v'][0].astype(float)
        omega=b['omega'][0].astype(float); mass=float(b['mass'][0])
        pb=mass*velocity
        inertia=R@s.bodies.initial[0]['inertia']@R.T
        lb=np.cross(center-BODY_CENTER,pb)+inertia@omega
        depth=np.min(BODY_SIZE/2-np.abs((x-center)@R),axis=1)
        row.update(penetration=float(max(0.,depth.max())),inside_count=int((depth>1e-5).sum()),
                   shape_error=float(np.linalg.norm(p['x'][r]-center-p['local'][r]@R.T,axis=1).max()),
                   quaternion_error=float(abs(np.linalg.norm(b['q'][0].astype(float))-1)),
                   world_contacts=int(b['world_contacts'][0]))
        vectors.update(body_center=center,body_v=velocity,body_omega=omega,
                       contact_impulse=b['contact_impulse'][0],grid_impulse=b['grid_impulse'][0],
                       world_impulse=b['world_impulse'][0])
        for i,val in enumerate(b['q'][0]): row[f'body_q{i}']=float(val)
    vectors.update(body_p=pb,body_l=lb,total_p=pf+pb,total_l=lf+spin+lb)
    for name,vec in vectors.items():
        for axis,val in zip('xyz',vec): row[f'{name}_{axis}']=float(val)
    return row,p,b


def run_case(out,name,dt,iterations):
    offset,mode,with_body=CASES[name]
    suffix='' if dt==.002 and iterations==5 else f'_dt{dt:g}_it{iterations}'
    folder=out/(name+suffix); folder.mkdir(exist_ok=False)
    c=config(mode,dt,iterations)
    write_json(folder/'config.json',dict(config=c.as_dict(),offset=offset,with_body=with_body,
                                       stage='formal' if not suffix else 'sensitivity',
                                       production_hashes=source_hashes(),duration=DURATION))
    s=Solver(c,scene(c,offset,with_body))
    rows=[]; frames=[]; frame_times=[]; qs=[]; centers=[]
    try:
        print(f'{folder.name}: compiling/warming',flush=True)
        s.advance(2); s.reset()
        steps=round(DURATION/dt); frame_every=round(.01/dt)
        save_steps={round(t/dt) for t in [0.,.16,.24,.34,.5]}
        start=time.perf_counter()
        with (folder/'metrics.csv').open('w',newline='',encoding='utf-8') as file:
            writer=None
            for k in range(steps+1):
                if k: s.step()
                row,p,b=sample(s); rows.append(row)
                if writer is None:
                    writer=csv.DictWriter(file,fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
                if k%frame_every==0:
                    frames.append(p['x'].copy()); frame_times.append(k*dt)
                    qs.append(b['q'][:s.bodies.count].copy());centers.append(b['x'][:s.bodies.count].copy())
                if k in save_steps:
                    np.savez_compressed(folder/f'state_{k:04d}.npz',**p,
                                        **{'body_'+key:value[:s.bodies.count] for key,value in b.items()})
                if k and k%max(1,steps//5)==0:
                    print(f'{folder.name}: t={k*dt:.3f}, vx={row.get("body_v_x",0):.5f}, '
                          f'wz={row.get("body_omega_z",0):.5f}, penetration={row["penetration"]:.3g}',flush=True)
        np.savez_compressed(folder/'frames.npz',time=frame_times,x=np.array(frames),q=np.array(qs),
                            center=np.array(centers),material=p['material'],body_size=BODY_SIZE)
        P=np.array([[r[f'total_p_{a}'] for a in 'xyz'] for r in rows])
        L=np.array([[r[f'total_l_{a}'] for a in 'xyz'] for r in rows])
        scale=rows[0]['fluid_mass']*SPEED
        summary=dict(completed=True,steps=steps,samples=len(rows),physical_duration=DURATION,
                     wall_seconds_including_sampling_and_io=time.perf_counter()-start,
                     final=rows[-1],max_penetration=max(r['penetration'] for r in rows),
                     max_shape_error=max(r['shape_error'] for r in rows),
                     minimum_boundary_stencil_clearance=min(r['boundary_stencil_clearance'] for r in rows),
                     max_linear_momentum_drift=float(np.linalg.norm(P-P[0],axis=1).max()/scale),
                     max_angular_momentum_drift=float(np.linalg.norm(L-L[0],axis=1).max()/(scale*.25)),
                     max_relative_volume_change=max(abs(r['fluid_volume']/rows[0]['fluid_volume']-1) for r in rows),
                     source_unchanged=source_hashes()==json.loads((out/'protocol.json').read_text())['production_hashes'])
        summary['valid_isolated_run']=bool(summary['minimum_boundary_stencil_clearance']>0 and
                                           summary['max_shape_error']<2e-6 and rows[-1]['world_contacts']==0)
        summary['diagnostic_flags']=[]
        if summary['max_penetration']>.05/c.grid: summary['diagnostic_flags'].append('penetration_above_0.05h')
        if mode=='two_way' and summary['max_linear_momentum_drift']>.01:
            summary['diagnostic_flags'].append('linear_momentum_drift_above_1_percent')
        write_json(folder/'summary.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='final'}),flush=True)
    except Exception:
        write_json(folder/'failure.json',dict(completed=False,saved_samples=len(rows),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','impulse','run'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=CASES,default='offset_two_way')
    parser.add_argument('--dt',type=float,choices=[.002,.001],default=.002)
    parser.add_argument('--iterations',type=int,choices=[5,10],default=5)
    parser.add_argument('--arch',choices=['cuda','cpu','vulkan'],default='cuda')
    args=parser.parse_args()
    if args.stage=='prepare': prepare(args.output);return
    verify_sources(args.output)
    ti.init(arch=getattr(ti,args.arch),random_seed=42,offline_cache=False,
            enable_fallback=False,cpu_max_num_threads=4)
    env=args.output/f'environment_{args.arch}.json'
    if not env.exists():
        write_json(env,dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
                            taichi=ti.__version__,arch=str(ti.lang.impl.current_cfg().arch),
                            argv=sys.argv,performance_benchmark=False))
    if args.stage=='impulse': impulse_checks(args.output)
    else: run_case(args.output,args.case,args.dt,args.iterations)


if __name__=='__main__': main()
