"""Audit and plot the saved fixed-plate trajectories without rerunning physics."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

NAMES = ['fluid_control', 'plate_shared', 'plate_separate']
LABELS = ['无板', '共享网格固定板', '分开传递固定板']
COLORS = ['#666666', '#d97732', '#1268b5']
STYLES = ['--', '-', '-.']


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8') as file:
        return list(csv.DictReader(file))


def load(path):
    rows = read_csv(path)
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def analyze(out):
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    data = {n: load(out/n/'metrics.csv') for n in NAMES}
    scale = protocol['fluid_mass']*.5
    control = data['fluid_control']
    source_ok = all(digest(out/'source_snapshot'/'rigid_coupling'/name)==value
                    for name, value in protocol['production_hashes'].items())
    assert source_ok
    assert digest(out/'source_snapshot'/'validation'/'ch04_sliding.py')==protocol['experiment_sha256']
    initial_fields = {}
    initial = [np.load(out/n/'state_0000.npz') for n in NAMES]
    # Material ID is read from the stored control instead of assuming its integer value.
    material = initial[0]['material'][0]
    for key in ['x','d','D','mass','rest_volume','volume_ratio']:
        values = [a[key][a['material']==material] for a in initial]
        initial_fields[key] = all(np.array_equal(values[0], v) for v in values[1:])
    assert all(initial_fields.values())
    results = {}
    for name, d in data.items():
        summary = json.loads((out/name/'summary.json').read_text(encoding='utf-8'))
        assert summary['completed'] and summary['source_unchanged']
        assert len(d['time'])==201 and np.array_equal(d['time'], control['time'])
        assert not summary['diagnostic_flags'], (name, summary['diagnostic_flags'])
        assert all(np.isfinite(values).all() for values in d.values())
        stage = read_csv(out/name/'stage_diagnostic.csv')
        first = {r['stage']: r for r in stage if int(r['iteration'])==1}
        diagnostic = {key: float(r['fluid_v_y']) for key, r in first.items()}
        contact_changes = []
        for iteration in range(1,6):
            rows = {r['stage']: r for r in stage if int(r['iteration'])==iteration}
            if 'contact' in rows:
                contact_changes.append(abs(float(rows['contact']['fluid_v_y'])-float(rows['g2p']['fluid_v_y'])))
        error = abs(d['fluid_p_y']-control['fluid_p_y'])/scale
        budget = np.stack([d[f'fluid_p_{a}']-d[f'fluid_p_{a}'][0]+
                           d[f'grid_impulse_{a}']+d[f'contact_impulse_{a}'] for a in 'xyz'], axis=1)
        results[name] = dict(
            final_speed=d['fluid_v_y'][-1], speed_retention_percent=d['fluid_v_y'][-1]/.5*100,
            final_displacement=d['fluid_center_y'][-1]-d['fluid_center_y'][0],
            displacement_fraction_of_ideal=(d['fluid_center_y'][-1]-d['fluid_center_y'][0])/.2,
            final_control_difference_percent=error[-1]*100, max_control_difference_percent=error.max()*100,
            final_water_momentum_loss_percent=(d['fluid_p_y'][0]-d['fluid_p_y'][-1])/scale*100,
            final_grid_impulse_percent=d['grid_impulse_y'][-1]/scale*100,
            final_contact_impulse_percent=d['contact_impulse_y'][-1]/scale*100,
            max_contact_impulse_norm=float(np.linalg.norm(np.stack([d[f'contact_impulse_{a}'] for a in 'xyz'], axis=1),axis=1).max()),
            max_budget_residual_percent=float(np.linalg.norm(budget,axis=1).max()/scale*100),
            max_penetration=summary['max_penetration'], max_inside_count=summary['max_inside_count'],
            max_volume_change_percent=summary['max_relative_volume_change']*100,
            min_boundary_stencil_clearance=summary['min_boundary_stencil_clearance'],
            min_plate_edge_clearance=summary['min_plate_edge_clearance'],
            max_plate_pose_error=summary['max_plate_pose_error'], max_plate_speed=summary['max_plate_speed'],
            max_plate_angular_speed=summary['max_plate_angular_speed'],
            first_iteration_speed=diagnostic,
            max_first_step_contact_tangential_speed_change=max(contact_changes, default=0.),
            first_step_speed=d['fluid_v_y'][1])
    write_json(out/'analysis.json', dict(cases=results, passed=True, initial_fluid_fields_identical=initial_fields,
               archived_production_sources_match=True, human_scientific_review='pending',
               physical_time_samples=201, formal_trajectories=3,
               metrics_normalization='P0 = M_f * 0.5', analysis_intent='descriptive mechanism comparison'))
    return protocol, data, results


def figures(out, protocol, data):
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'font.size':10,
                         'axes.unicode_minus':False, 'axes.spines.top':False,
                         'axes.spines.right':False, 'savefig.facecolor':'white',
                         'figure.facecolor':'white', 'svg.fonttype':'none'})
    target = out/'figures'
    target.mkdir(exist_ok=True)
    def save(fig, name):
        for suffix in ['png','svg','pdf']:
            fig.savefig(target/f'{name}.{suffix}', dpi=200, bbox_inches='tight')
        plt.close(fig)
    fig, axs = plt.subplots(1,2, figsize=(10.5,3.9), layout='constrained')
    for name,label,color,style in zip(NAMES,LABELS,COLORS,STYLES):
        d = data[name]
        marker = 'o' if name=='plate_separate' else None
        axs[0].plot(d['time'], d['fluid_v_y'], ls=style, color=color, lw=1.9,
                    marker=marker, markevery=25, ms=4, mfc='white', label=label)
        axs[1].plot(d['time'], d['fluid_center_y']-d['fluid_center_y'][0], ls=style,
                    color=color, lw=1.9, marker=marker, markevery=25, ms=4, mfc='white', label=label)
    for ax,title,ylabel in zip(axs,['切向速度','滑移距离'],
                              ['平均速度（长度单位/s）','质心位移（长度单位）']):
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlabel('物理时间（s）'); ax.set_ylabel(ylabel)
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=9)
        ax.set_xlim(0,.4)
    axs[0].set_ylim(-.01,.53); axs[1].set_ylim(-.004,.215)
    save(fig, 'sliding_response')

    # Fixed initial central slice, no frame-dependent particle selection.
    fig, axs = plt.subplots(1,3, figsize=(8,5.3), sharex=True, sharey=True, layout='constrained')
    for ax,name,label in zip(axs,NAMES,LABELS):
        frames = np.load(out/name/'frames.npz')
        fluid = frames['material']==frames['material'][0]
        mask = fluid & (abs(frames['x'][0,:,2]-.5)<1/64)
        start, last = frames['x'][0,mask], frames['x'][-1,mask]
        if name != 'fluid_control':
            ax.add_patch(Rectangle((.46875,.25), .0625,.5, facecolor='#e5a965',
                                   edgecolor='#a96324', lw=1.1, alpha=.75))
        ax.scatter(start[:,0], start[:,1], s=7, facecolors='none', edgecolors='#9a9a9a', lw=.6)
        ax.scatter(last[:,0], last[:,1], s=10, c='#1268b5', edgecolors='none')
        ax.set_title(label, fontsize=11); ax.set_xlabel('x（长度单位）')
        ax.set_xlim(.382,.545); ax.set_ylim(.235,.765); ax.set_aspect('equal')
        ax.grid(alpha=.12)
    axs[0].set_ylabel('y（长度单位）')
    fig.suptitle('固定板附近的粒子分布：t = 0.4 s', fontsize=14)
    save(fig, 'sliding_particles')

    scale = protocol['fluid_mass']*.5
    fig, axs = plt.subplots(1,2, figsize=(10.5,3.9), layout='constrained')
    d = data['plate_shared']
    axs[0].plot(d['time'], (d['fluid_p_y'][0]-d['fluid_p_y'])/scale*100,
                color='#d97732', lw=2.1, label='水体损失的切向动量')
    axs[0].plot(d['time'], d['grid_impulse_y']/scale*100, '--', color='#1268b5', lw=1.6,
                marker='o',markevery=25,ms=4,mfc='white',label='固定板累计网格冲量')
    axs[0].plot(d['time'], d['contact_impulse_y']/scale*100, ':', color='#555555',
                lw=2, label='固定板累计接触冲量')
    for name,label,color,style in zip(NAMES,LABELS,COLORS,STYLES):
        d = data[name]
        budget = np.stack([d[f'fluid_p_{a}']-d[f'fluid_p_{a}'][0]+
                           d[f'grid_impulse_{a}']+d[f'contact_impulse_{a}'] for a in 'xyz'],axis=1)
        axs[1].plot(d['time'], np.linalg.norm(budget,axis=1)/scale*100,ls=style,color=color,label=label)
    for ax,title,ylabel in zip(axs,['共享网格的切向动量传递','动量收支残差'],
                              ['初始水流动量的比例（%）','归一化残差（%）']):
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlabel('物理时间（s）'); ax.set_ylabel(ylabel); ax.set_xlim(0,.4)
        ax.grid(alpha=.2); ax.legend(frameon=False,fontsize=9)
    save(fig, 'sliding_momentum')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol, data, results = analyze(args.output)
    figures(args.output, protocol, data)
    shutil.copy2(__file__, args.output/'source_snapshot'/'validation'/Path(__file__).name)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
