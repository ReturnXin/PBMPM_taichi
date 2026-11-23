"""Trace the loss of slip and summarize the three registered diagnostic runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID
from rigid_coupling.runner import source_hashes, write_json
from rigid_validation_run import WORKSPACE, create_scene

ROOT = WORKSPACE / '毕业论文/论文正文/assets/ch04-slip-investigation-20261028'


def trace(root, backend):
    out = root / 'trace.json'
    if out.exists(): raise FileExistsError(out)
    protocol = json.loads((root / 'planned_protocol.json').read_text(encoding='utf-8'))
    result = {'backend': backend, 'source_hashes': source_hashes(), 'cases': {}}
    for mode in ['shared', 'separate']:
        case = next(c for c in protocol['cases'] if c['case_id'] == f'S_{mode}_wall')
        c = Config(grid=case['grid'], boundary=case['grid_boundary_index'], dt=case['dt'],
                   iterations=case['iterations'], gravity=0, viscosity=0, rigid_volume=True,
                   rigid_contact=True, world_contact=True, grid_coupling=mode)
        ti.init(arch={'cpu':ti.cpu, 'vulkan':ti.vulkan}[backend], offline_cache=False,
                enable_fallback=False, random_seed=42)
        s = Solver(c, create_scene(case, c)); rows = []
        def record(stage):
            p = s.particles.snapshot(); b = s.bodies.data.to_numpy(); f = p['material'] == FLUID
            mom = (p['mass'][f, None].astype(float) * p['d'][f] / c.dt).sum(axis=0)
            rows.append(dict(stage=stage, fluid_velocity=(mom / p['mass'][f].sum()).tolist(),
                             fluid_momentum=mom.tolist(), body_grid_impulse=b['grid_impulse'][0].tolist(),
                             body_contact_impulse=b['contact_impulse'][0].tolist()))
        record('initial')
        s.materials.predict(); s.bodies.predict(); s.prepare_iteration(); record('after_material_projection')
        s.clear_iteration(); s.transfers.p2g(); s.transfers.boundary(); record('after_grid_normalization')
        s.transfers.g2p_with_feedback(s.bodies); record('after_g2p')
        s.bodies.apply_feedback(); record('after_body_feedback')
        s.contacts.project(); s.bodies.apply_contact_feedback(); record('after_surface_contact')
        result['cases'][mode] = rows
        ti.reset()
    write_json(out, result)
    print(json.dumps(result['cases'], indent=2))


def analyze(root):
    p = json.loads((root / 'planned_protocol.json').read_text(encoding='utf-8'))
    tables = {}; result = {'cases': {}, 'performance_claim': False}
    for c in p['cases']:
        d = root / 'data' / c['case_id']; meta = json.loads((d / 'run.json').read_text())
        if meta['status'] != 'completed': raise RuntimeError(c['case_id'])
        rows = list(csv.DictReader((d/'metrics.csv').open(encoding='utf-8')))
        if len(rows)!=41 or abs(float(rows[-1]['time'])-.2)>1.e-10: raise ValueError('incomplete run')
        tables[c['case_id']] = rows
        b = rows[-1]; init = rows[0]
        res = dict(final_velocity_y=float(b['fluid_velocity_y']), samples=len(rows),
                   max_penetration=float(b['max_penetration_so_far']), status=meta['status'])
        if 'body_grid_impulse1' in b:
            loss=float(init['fluid_momentum_y'])-float(b['fluid_momentum_y'])
            impulse=float(b['body_grid_impulse1'])+float(b['body_contact_impulse1'])
            res.update(grid_impulse_y=float(b['body_grid_impulse1']),contact_impulse_y=float(b['body_contact_impulse1']),
                       momentum_balance_relative=abs(loss-impulse)/abs(float(init['fluid_momentum_y'])),
                       fixed_pose_unchanged=all(all(r[f'body_{f}{i}']==init[f'body_{f}{i}'] for i in range(n)) for r in rows for f,n in [('x',3),('q',4)]))
        result['cases'][c['case_id']]=res
    control=tables['S_separate_control'][-1]
    for name in ['S_shared_wall','S_separate_wall']:
        row=tables[name][-1]
        error=abs(float(row['fluid_momentum_y'])-float(control['fluid_momentum_y']))/(float(row['fluid_mass'])*.5)
        result['cases'][name].update(extra_tangential_momentum_change=error,slip_target_met=error<=.01)
    write_json(root/'analysis.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for name,label,style in [('S_shared_wall','Shared grid + wall','-'),('S_separate_wall','Separate grids + wall','--'),('S_separate_control','No-wall control',':')]:
        rows=tables[name]
        ax.plot([float(r['time']) for r in rows],[float(r['fluid_velocity_y']) for r in rows],style,label=label,linewidth=2)
    ax.set(xlabel='Time (s)',ylabel='Mean velocity along the wall',ylim=(-.025,.55));ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'slip_comparison.png',dpi=160);fig.savefig(root/'slip_comparison.pdf');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['trace','analyze']);ap.add_argument('--output',type=Path,default=ROOT)
    ap.add_argument('--backend',choices=['cpu','vulkan'],default='vulkan');args=ap.parse_args()
    if args.action=='trace':trace(args.output,args.backend)
    else:analyze(args.output)
