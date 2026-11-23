"""Reproducible fixed-plate sliding experiment using the production Solver."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.rigid import make_body, rotation_numpy
from rigid_coupling.runner import source_hashes, write_json
from rigid_coupling.scenes import Scene, lattice
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID, RIGID, combine_particles, particle_arrays

CASES = {'fluid_control': (False, 'separate'),
         'plate_shared': (True, 'shared'), 'plate_separate': (True, 'separate')}
PLATE_CENTER = np.array([.5, .5, .5])
PLATE_SIZE = np.array([.0625, .5, .375])
WATER_CENTER = np.array([.4375, .375, .5])
WATER_SIZE = np.array([.0625, .125, .125])
SPACING, SPEED, DURATION = 1 / 64, .5, .4


def config(mode):
    return Config(grid=32, dt=.002, iterations=5, gravity=0.,
                  fluid_relaxation=1., viscosity=0., coupling='two_way',
                  fluid_density=1., rigid_density=1., rigid_volume=True,
                  rigid_contact=True, world_contact=True, buoyancy_assist=False,
                  grid_coupling=mode, boundary_density='raw',
                  wall_transfer='clamp', density_feedback='legacy')


def scene(c, with_plate):
    water = lattice(WATER_CENTER, WATER_SIZE, np.rint(WATER_SIZE / SPACING).astype(int))
    groups = [particle_arrays(water, FLUID, SPACING**3, (0., SPEED, 0.),
                              c.dt, (.13, .48, .82), mass=SPACING**3)]
    bodies = []
    if with_plate:
        local = lattice((0, 0, 0), PLATE_SIZE, np.rint(PLATE_SIZE / SPACING).astype(int))
        body = make_body(local, PLATE_CENTER, SPACING**3)
        body.update(size=PLATE_SIZE.copy(), fixed=True, density=1.)
        groups.append(particle_arrays(body['center'] + body['local'], RIGID,
                                      SPACING**3, (0., 0., 0.), c.dt, (.9, .48, .12),
                                      body_id=0, local=body['local'], mass=body['particle_mass']))
        bodies.append(body)
    return Scene(combine_particles(groups), bodies, 'Water sliding along a fixed frictionless plate')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    snap = out / 'source_snapshot'
    shutil.copytree(root / 'rigid_coupling', snap / 'rigid_coupling',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (snap / 'validation').mkdir()
    shutil.copy2(__file__, snap / 'validation' / Path(__file__).name)
    (snap / 'validation' / '__init__.py').write_text('', encoding='utf-8')
    shared, separate = config('shared'), config('separate')
    a, b = scene(shared, True), scene(separate, True)
    equal = {key: bool(np.array_equal(a.particles[key], b.particles[key])) for key in a.particles}
    differences = [key for key in shared.as_dict() if shared.as_dict()[key] != separate.as_dict()[key]]
    assert all(equal.values()) and differences == ['grid_coupling']
    write_json(out / 'initialization_check.json', dict(particle_fields_equal=equal,
               config_differences=differences, fixed=a.bodies[0]['fixed'], passed=True))
    protocol = dict(
        experiment='ch04 experiment 2: fixed plate sliding', date='2026-11-11',
        status='planned_before_execution', analysis_intent='mechanism validation and descriptive comparison',
        human_scientific_review='pending', baseline=separate.as_dict(), cases=CASES,
        units='consistent code length and mass; time in configured seconds',
        plate_center=PLATE_CENTER.tolist(), plate_size=PLATE_SIZE.tolist(),
        water_center=WATER_CENTER.tolist(), water_size=WATER_SIZE.tolist(),
        water_velocity=[0., SPEED, 0.], particle_spacing=SPACING,
        water_particle_count=int((a.particles['material'] == FLUID).sum()),
        plate_particle_count=int((a.particles['material'] == RIGID).sum()),
        fluid_mass=float(a.particles['mass'][a.particles['material'] == FLUID].sum(dtype=float)),
        plate_mass=a.bodies[0]['mass'], fixed_plate=True, duration=DURATION,
        steps=round(DURATION / separate.dt), sample_every_physical_step=True,
        frame_interval=.01, snapshot_times=[0., .1, .2, .4],
        initial_surface_gap=0., initial_particle_center_gap=SPACING / 2,
        normal_axis='x', tangent_axis='y', warmup_steps=2,
        forcing='No gravity, viscosity, buoyancy assist, prescribed motion, or container.',
        stage_diagnostic='After warmup/reset, record all five iterations of the first physical step: predict, material preparation, G2P, body grid feedback, contact, affine reconciliation. Reset again before formal recording.',
        measurements=['mass-weighted tangential speed', 'fluid center displacement',
                      'fluid tangential momentum', 'cumulative grid and contact impulses on fixed plate',
                      'momentum budget residual', 'post-step OBB particle penetration',
                      'plate pose and speed', 'boundary stencil clearance',
                      'plate tangential edge clearance', 'fluid volume change', 'finite state'],
        normalization='P0 = fluid mass * prescribed speed 0.5',
        control_difference='abs(P_plate(t) - P_control(t)) / P0 at each common physical time',
        impulse_budget='P_fluid(t) - P_fluid(0) + J_grid(t) + J_contact(t); the fixed plate support absorbs the impulse without changing body momentum.',
        penetration='Maximum positive OBB depth of particle centers after every committed step; inside count uses depth > 1e-5.',
        windows='Entire 0..0.4 s without tail selection or time shifting.',
        repeats='One formal trajectory per configuration, no statistical or performance comparison.',
        diagnostic_thresholds=dict(budget_fraction=.01, penetration_h=.05,
                                   min_boundary_clearance=0., min_plate_edge_clearance=0.),
        failure_policy='Retain failed and incomplete runs. Save protocol amendments before additional runs.',
        production_hashes=source_hashes(), experiment_sha256=digest(__file__))
    write_json(out / 'protocol.json', protocol)
    print(json.dumps({k: protocol[k] for k in ['water_particle_count', 'plate_particle_count', 'steps']}, indent=2), flush=True)


def verify_sources(out):
    p = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
    if p['production_hashes'] != source_hashes() or p['experiment_sha256'] != digest(__file__):
        raise RuntimeError('Production or experiment source changed after protocol preparation')
    return p


def sample(s):
    p = s.particles.snapshot()
    b = {key: value[:s.bodies.count] for key, value in s.bodies.data.to_numpy().items()}
    for prefix, arrays in [('particle', p), ('body', b)]:
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite {prefix} {key}')
    f = p['material'] == FLUID
    x, m = p['x'][f].astype(float), p['mass'][f].astype(float)
    v = p['d'][f].astype(float) / s.config.dt
    momentum = (m[:, None] * v).sum(axis=0)
    base = np.floor(p['x'].astype(float) * s.config.grid - .5)
    row = dict(step=s.step_index, time=s.step_index * s.config.dt,
               fluid_mass=float(m.sum()),
               fluid_volume=float((p['rest_volume'][f].astype(float) * p['volume_ratio'][f]).sum()),
               fluid_j_min=float(p['volume_ratio'][f].min()), fluid_j_max=float(p['volume_ratio'][f].max()),
               boundary_stencil_clearance=float(min((base-s.config.boundary).min(),
                    (s.config.grid-s.config.boundary-base-2).min())),
               penetration=0., inside_count=0, plate_pose_error=0., plate_speed=0.,
               plate_angular_speed=0., plate_edge_clearance=0., world_contacts=0)
    vectors = dict(fluid_p=momentum, fluid_v=momentum/m.sum(),
                   fluid_center=np.average(x, weights=m, axis=0),
                   grid_impulse=np.zeros(3), contact_impulse=np.zeros(3), world_impulse=np.zeros(3))
    if s.bodies.count:
        center, R = b['x'][0].astype(float), rotation_numpy(b['q'][0])
        local = (x-center) @ R
        depth = np.min(PLATE_SIZE/2-np.abs(local), axis=1)
        row.update(penetration=float(max(0., depth.max())), inside_count=int((depth>1e-5).sum()),
                   plate_pose_error=float(max(np.abs(center-PLATE_CENTER).max(),
                                             np.abs(b['q'][0]-[1.,0.,0.,0.]).max())),
                   plate_speed=float(np.linalg.norm(b['v'][0])),
                   plate_angular_speed=float(np.linalg.norm(b['omega'][0])),
                   plate_edge_clearance=float((PLATE_SIZE[1:]/2-np.abs(local[:,1:])).min()),
                   world_contacts=int(b['world_contacts'][0]))
        for name in ['grid_impulse', 'contact_impulse', 'world_impulse']:
            vectors[name] = b[name][0]
    for name, vec in vectors.items():
        for axis, value in zip('xyz', vec):
            row[f'{name}_{axis}'] = float(value)
    return row, p, b


def stage_diagnostic(s, folder):
    rows = []
    def record(iteration, stage):
        row, _, _ = sample(s)
        rows.append(dict(iteration=iteration, stage=stage, **row))
    record(0, 'initial')
    s.materials.predict()
    if s.bodies.count:
        s.bodies.predict()
    record(0, 'predict')
    for iteration in range(1, s.config.iterations+1):
        s.prepare_iteration()
        record(iteration, 'material')
        s.transfers.p2g()
        s.transfers.boundary()
        s.transfers.g2p_with_feedback(s.bodies)
        record(iteration, 'g2p')
        if s.bodies.count:
            s.bodies.apply_feedback()
            record(iteration, 'grid_feedback')
            s.contacts.project()
            s.bodies.apply_contact_feedback()
            record(iteration, 'contact')
            s.contacts.update_contact_affine(s.transfers)
            record(iteration, 'contact_affine')
    with (folder/'stage_diagnostic.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    s.reset()


def run_case(out, name):
    with_plate, mode = CASES[name]
    folder = out/name
    folder.mkdir(exist_ok=False)
    c = config(mode)
    write_json(folder/'config.json', dict(config=c.as_dict(), with_plate=with_plate,
                                         production_hashes=source_hashes(), duration=DURATION))
    s = Solver(c, scene(c, with_plate))
    rows, frames, times = [], [], []
    try:
        print(f'{name}: compiling/warming', flush=True)
        s.advance(2); s.reset()
        stage_diagnostic(s, folder)
        start = time.perf_counter()
        steps = round(DURATION/c.dt)
        snapshots = {round(t/c.dt) for t in [0., .1, .2, .4]}
        with (folder/'metrics.csv').open('w', newline='', encoding='utf-8') as file:
            writer = None
            for k in range(steps+1):
                if k:
                    s.step()
                row, p, b = sample(s)
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row)
                if k % round(.01/c.dt) == 0:
                    frames.append(p['x'].copy()); times.append(k*c.dt)
                if k in snapshots:
                    np.savez_compressed(folder/f'state_{k:04d}.npz', **p,
                                        **{'body_'+key: value for key, value in b.items()})
                if k and k % 50 == 0:
                    print(f'{name}: t={k*c.dt:.3f}, vy={row["fluid_v_y"]:.8f}, penetration={row["penetration"]:.3g}', flush=True)
        np.savez_compressed(folder/'frames.npz', time=times, x=np.array(frames),
                            material=p['material'], plate_center=PLATE_CENTER, plate_size=PLATE_SIZE)
        scale = rows[0]['fluid_mass']*SPEED
        budget = np.array([[r[f'fluid_p_{a}']-rows[0][f'fluid_p_{a}']+
                           r[f'grid_impulse_{a}']+r[f'contact_impulse_{a}'] for a in 'xyz'] for r in rows])
        summary = dict(completed=True, steps=steps, samples=len(rows), physical_duration=DURATION,
                       wall_seconds_including_sampling_and_io=time.perf_counter()-start,
                       final=rows[-1], max_penetration=max(r['penetration'] for r in rows),
                       max_inside_count=max(r['inside_count'] for r in rows),
                       max_plate_pose_error=max(r['plate_pose_error'] for r in rows),
                       max_plate_speed=max(r['plate_speed'] for r in rows),
                       max_plate_angular_speed=max(r['plate_angular_speed'] for r in rows),
                       min_boundary_stencil_clearance=min(r['boundary_stencil_clearance'] for r in rows),
                       min_plate_edge_clearance=min(r['plate_edge_clearance'] for r in rows) if with_plate else None,
                       max_momentum_budget_residual=float(np.linalg.norm(budget, axis=1).max()/scale),
                       max_relative_volume_change=max(abs(r['fluid_volume']/rows[0]['fluid_volume']-1) for r in rows),
                       source_unchanged=source_hashes()==json.loads((out/'protocol.json').read_text())['production_hashes'])
        summary['diagnostic_flags'] = []
        if summary['max_penetration'] > .05/c.grid:
            summary['diagnostic_flags'].append('penetration_above_0.05h')
        if summary['max_momentum_budget_residual'] > .01:
            summary['diagnostic_flags'].append('momentum_budget_above_1_percent')
        if summary['min_boundary_stencil_clearance'] <= 0:
            summary['diagnostic_flags'].append('outer_boundary_stencil')
        if with_plate and summary['min_plate_edge_clearance'] <= 0:
            summary['diagnostic_flags'].append('water_past_plate_edge')
        write_json(folder/'summary.json', summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='final'}), flush=True)
    except Exception:
        write_json(folder/'failure.json', dict(completed=False, saved_samples=len(rows),
                                             traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare','run'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, default='plate_separate')
    parser.add_argument('--arch', choices=['cuda','cpu','vulkan'], default='cuda')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output)
        return
    verify_sources(args.output)
    ti.init(arch=getattr(ti,args.arch), random_seed=42, offline_cache=True,
            enable_fallback=False, cpu_max_num_threads=4)
    env = args.output/f'environment_{args.arch}.json'
    if not env.exists():
        write_json(env, dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                             taichi=ti.__version__, arch=str(ti.lang.impl.current_cfg().arch),
                             argv=sys.argv, performance_benchmark=False))
    run_case(args.output, args.case)


if __name__ == '__main__':
    main()
