"""Diagnose a single impact without adding force or editing the thesis baseline."""
import argparse
import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import taichi as ti

from rigid_coupling.rigid import rotation_numpy
from rigid_coupling.runner import source_hashes, write_json
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID
from validation.ch04_impact import config, scene, sample
from validation.ch04_impact_analyze import load_csv


def energy(p, b, dt, grid, inertia):
    fluid = p['material'] == FLUID
    m = p['mass'][fluid].astype(float)
    v = p['d'][fluid].astype(float)/dt
    C = p['D'][fluid].astype(float)/dt
    ef = .5*np.sum(m[:, None]*v*v)
    ea = np.sum(m[:, None, None]*C*C)/(8*grid**2)
    eb = 0.
    if inertia is not None:
        R = rotation_numpy(b['q'][0])
        w = b['omega'][0].astype(float)
        eb = .5*b['mass'][0]*np.sum(b['v'][0].astype(float)**2) + .5*w@(R@inertia@R.T)@w
    return dict(fluid_kinetic=float(ef), rigid_kinetic=float(eb),
                affine_norm=float(ea), particle_rigid_kinetic=float(ef+eb),
                apic_rigid_norm=float(ef+ea+eb),
                fluid_mean_speed=float(np.average(np.linalg.norm(v,axis=1),weights=m)))


def run(out, variant):
    folder = out/variant
    folder.mkdir(exist_ok=False)
    c = replace(config(), grid=64 if variant=='full64' else 32,
                transfer_mode='incremental' if variant=='incremental32' else 'full')
    write_json(folder/'protocol.json',dict(config=c.as_dict(),source_hashes=source_hashes(),
        purpose='Exploratory inertia diagnosis; same particles/mass/initial velocity; no added force.',
        changed_parameter='grid' if variant=='full64' else 'transfer_mode',
        observation_duration=.5,case='center_two_way'))
    ti.init(arch=ti.cuda,offline_cache=False,enable_fallback=False,random_seed=42)
    initial=scene(c)
    s=Solver(c,initial)
    inertia=initial.bodies[0]['inertia']
    print(variant+': compiling/warming',flush=True)
    s.advance(2)
    s.reset()
    frames=[];times=[];centers=[];quaternions=[]
    records=[]
    for k in range(251):
        if k:s.step()
        row,p,b=sample(s)
        row.update(energy(p,b,c.dt,c.grid,inertia))
        records.append(row)
        if k%5==0:
            frames.append(p['x'].copy()); times.append(k*c.dt)
            centers.append(b['x'][:1].copy());quaternions.append(b['q'][:1].copy())
        if k in [0,80,120,170,250]:
            np.savez_compressed(folder/f'state_{k:04d}.npz',**p,
                                **{'body_'+key:value[:1] for key,value in b.items()})
        if k and k%50==0:
            print(f'{variant}: t={k*c.dt:.2f}, E/E0={row["particle_rigid_kinetic"]/records[0]["particle_rigid_kinetic"]:.5f}, '
                  f'vx={row["body_v_x"]:.5f}',flush=True)
    with (folder/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    np.savez_compressed(folder/'frames.npz',x=frames,time=times,center=centers,q=quaternions,material=p['material'])
    P=np.array([[r['total_p_'+a] for a in 'xyz'] for r in records])
    summary=dict(completed=True,final=records[-1],
        max_linear_momentum_drift_percent=float(np.linalg.norm(P-P[0],axis=1).max()/np.linalg.norm(P[0])*100),
        min_boundary_stencil_clearance=min(r['boundary_stencil_clearance'] for r in records),
        max_penetration=max(r['penetration'] for r in records),
        max_shape_error=max(r['shape_error'] for r in records),
        final_kinetic_percent=records[-1]['particle_rigid_kinetic']/records[0]['particle_rigid_kinetic']*100,
        final_apic_norm_percent=records[-1]['apic_rigid_norm']/records[0]['apic_rigid_norm']*100)
    write_json(folder/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='final'}),flush=True)


def analyze(out, baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    baseline_protocol=json.loads((baseline/'protocol.json').read_text())
    archived=[]
    for name in ['fluid_control','center_one_way','center_two_way','offset_one_way','offset_two_way']:
        e0=None
        for path in sorted((baseline/name).glob('state_*.npz')):
            saved=np.load(path)
            p={k:saved[k] for k in saved.files if not k.startswith('body_')}
            b={k[5:]:saved[k] for k in saved.files if k.startswith('body_')}
            inertia=np.array(baseline_protocol['body_inertia']) if len(b['mass']) else None
            row=energy(p,b,.002,32,inertia)
            if e0 is None:e0=row['particle_rigid_kinetic']
            mask=p['material']==FLUID
            mass=p['mass'][mask].astype(float)
            velocity=p['d'][mask].astype(float)/.002
            row.update(case=name,time=int(path.stem.split('_')[1])*.002,
                mean_v=np.average(velocity,weights=mass,axis=0).tolist(),
                water_extent=np.ptp(p['x'][mask].astype(float),axis=0).tolist(),
                kinetic_percent=row['particle_rigid_kinetic']/e0*100,
                apic_norm_percent=row['apic_rigid_norm']/e0*100)
            archived.append(row)
    write_json(out/'archived_energy.json',dict(source=str(baseline),samples=archived,
        energy_definition='Particle translation plus rigid translation/rotation. APIC affine norm is reported separately.'))
    names=['原配置：32³ 完整传递','32³ 增量传递','64³ 完整传递']
    folders=[baseline/'center_two_way',out/'incremental32',out/'full64']
    colors=['#555555','#dd7034','#1268b5']
    records=[]
    energy_curves=[]
    for index,folder in enumerate(folders):
        grid=64 if index==2 else 32
        metrics=load_csv(folder/'metrics.csv')
        values=[]
        for path in sorted(folder.glob('state_*.npz')):
            data=np.load(path)
            p={k:data[k] for k in data.files if not k.startswith('body_')}
            b={k[5:]:data[k] for k in data.files if k.startswith('body_')}
            values.append(dict(time=int(path.stem.split('_')[1])*.002,
                **energy(p,b,.002,grid,np.array(baseline_protocol['body_inertia']))))
        e0=values[0]['particle_rigid_kinetic']
        for row in values:
            row['kinetic_percent']=row['particle_rigid_kinetic']/e0*100
            row['apic_norm_percent']=row['apic_rigid_norm']/e0*100
        P=np.stack([metrics['total_p_'+a] for a in 'xyz'],axis=1)
        records.append(dict(label=names[index],energy_snapshots=values,
            final_vx=float(metrics['body_v_x'][-1]),
            max_linear_momentum_drift_percent=float(np.linalg.norm(P-P[0],axis=1).max()/np.linalg.norm(P[0])*100),
            max_penetration=float(metrics['penetration'].max()),
            min_boundary_stencil_clearance=float(metrics['boundary_stencil_clearance'].min())))
        energy_curves.append(values)
    write_json(out/'comparison.json',dict(cases=records,grid64_particles_unchanged=True,
        energy_definition='Particle translation plus rigid translation/rotation. APIC affine norm is reported separately.',
        human_scientific_review='pending',main_manuscript_changed=False))
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for label,values,color in zip(names,energy_curves,colors):
        for ax,key in zip(axs,['kinetic_percent','apic_norm_percent']):
            ax.plot([r['time'] for r in values],[r[key] for r in values],'-o',color=color,label=label)
            ax.set_xlabel('物理时间（s）');ax.grid(alpha=.2)
    axs[0].set_ylabel('水体及刚体动能 / 初始动能（%）')
    axs[1].set_ylabel('计入 APIC 仿射项的范数 / 初始值（%）')
    axs[0].legend(frameon=False,fontsize=9)
    fig.savefig(out/'energy_comparison.png',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(10,9),layout='constrained')
    for i,(label,folder) in enumerate(zip(names,folders)):
        f=np.load(folder/'frames.npz')
        layer=abs(f['x'][0,:,2]-.5)<1/64
        for j,t in enumerate([.24,.34,.5]):
            k=int(np.argmin(abs(f['time']-t)))
            ax=axs[i,j]
            for m,color in [(0,'#278ac4'),(1,'#d47a24')]:
                x=f['x'][k,layer&(f['material']==m)]
                ax.scatter(x[:,0],x[:,1],s=1.1,color=color,edgecolors='none')
            ax.set(xlim=(.1,.9),ylim=(.1,.9),aspect='equal')
            if i==0:ax.set_title(f't = {t:.2f} s')
            if j==0:ax.set_ylabel(label+'\ny')
            if i==2:ax.set_xlabel('x')
    fig.savefig(out/'inertia_frames.png',dpi=180)
    plt.close(fig)
    # Two views of the same final baseline state reveal sheet motion out of the XY view.
    f=np.load(baseline/'center_two_way/frames.npz')
    last=np.load(baseline/'center_two_way/state_0250.npz')
    initial_layer=abs(f['x'][0,:,2]-.5)<1/64
    fig,axs=plt.subplots(1,2,figsize=(9,4.5),layout='constrained')
    for ax,axes,title,keep in [(axs[0],(0,1),'当前侧视：初始中央薄层',initial_layer),
                               (axs[1],(2,1),'正视：全部粒子',np.ones(len(f['material']),dtype=bool))]:
        for m,color in [(1,'#d47a24'),(0,'#278ac4')]:
            x=f['x'][-1,keep&(f['material']==m)]
            ax.scatter(x[:,axes[0]],x[:,axes[1]],s=1.1,color=color,alpha=.7,edgecolors='none')
        selected=keep&(last['material']==FLUID)
        xy=last['x'][selected][:,axes].astype(float)
        v=last['d'][selected][:,axes].astype(float)/.002
        bins=np.floor(xy*16).astype(int)
        cells=np.unique(bins,axis=0)
        centers=[];velocities=[]
        for cell in cells:
            mask=np.all(bins==cell,axis=1)
            if mask.sum()>=8:
                centers.append(xy[mask].mean(axis=0))
                velocities.append(v[mask].mean(axis=0)*.08)
        centers=np.array(centers);velocities=np.array(velocities)
        ax.quiver(centers[:,0],centers[:,1],velocities[:,0],velocities[:,1],
                  angles='xy',scale_units='xy',scale=1,color='#34434e',width=.0035)
        ax.set(xlim=(.15,.9),ylim=(.15,.9),aspect='equal',title=title,
               xlabel='xyz'[axes[0]],ylabel='xyz'[axes[1]])
    fig.suptitle('相同状态 t = 0.50 s；箭头为局部平均速度 × 0.08 s',fontsize=11)
    fig.savefig(out/'baseline_two_views.png',dpi=180)
    plt.close(fig)
    print(json.dumps(records,ensure_ascii=False,indent=2))


def trace(out, baseline):
    """Attribute one full step from two saved impact states, with a replay check."""
    target=out/'stage_trace'
    target.mkdir(exist_ok=True)
    if (target/'results.json').exists():raise FileExistsError(target/'results.json')
    ti.init(arch=ti.cuda,offline_cache=False,enable_fallback=False,random_seed=42)
    c=config()
    initial=scene(c)
    s=Solver(c,initial)
    s.advance(2)
    inertia=initial.bodies[0]['inertia']
    results=[]
    for step in [120,170]:
        saved=np.load(baseline/'center_two_way'/f'state_{step:04d}.npz')
        p={k:saved[k].copy() for k in s.initial}
        b={k:saved['body_'+k].copy() for k in s.bodies.data.to_numpy()}
        def restore():
            s.reset()
            s.particles.restore(p)
            s.bodies.data.from_numpy(b)
            s.step_index=step
        def metric():
            return energy(s.particles.snapshot(),s.bodies.data.to_numpy(),c.dt,c.grid,inertia)
        restore()
        old=metric()
        stages=[]
        def record(name,iteration=0):
            nonlocal old
            new=metric()
            stages.append(dict(stage=name,iteration=iteration,
                delta_kinetic=new['particle_rigid_kinetic']-old['particle_rigid_kinetic'],
                delta_apic_norm=new['apic_rigid_norm']-old['apic_rigid_norm']))
            old=new
        s.materials.predict();s.bodies.predict()
        record('predict')
        for iteration in range(c.iterations):
            s.prepare_iteration()
            record('material_constraint',iteration+1)
            s.transfers.p2g();s.transfers.boundary()
            s.transfers.g2p_with_feedback(s.bodies);s.bodies.apply_feedback()
            record('grid_transfer',iteration+1)
            s.contacts.project();s.bodies.apply_contact_feedback()
            s.contacts.update_contact_affine(s.transfers)
            record('contact',iteration+1)
        s.materials.integrate();s.bodies.integrate()
        record('commit')
        traced_p=s.particles.snapshot()
        traced_b=s.bodies.data.to_numpy()
        restore()
        s.step()
        actual_p=s.particles.snapshot()
        actual_b=s.bodies.data.to_numpy()
        # Independent normal-step replay prevents the instrumentation defining a new solver.
        replay={}
        for key in ['x','d','D','volume_ratio']:
            error=float(np.max(abs(traced_p[key]-actual_p[key])))
            assert np.allclose(traced_p[key],actual_p[key],rtol=1e-4,atol=2e-6),(step,key,error)
            replay[key]=error
        assert np.allclose(traced_b['v'],actual_b['v'],rtol=1e-4,atol=2e-6)
        e0=.5*float(p['mass'][p['material']==FLUID].astype(float).sum())*.75**2
        totals={name:{key:sum(r[key] for r in stages if r['stage']==name)/e0*100
                      for key in ['delta_kinetic','delta_apic_norm']}
                for name in ['predict','material_constraint','grid_transfer','contact','commit']}
        results.append(dict(start_time=step*c.dt,end_time=(step+1)*c.dt,stages=stages,
                            percent_of_initial_energy=totals,replay_max_errors=replay))
        print(json.dumps(results[-1]['percent_of_initial_energy']),flush=True)
    write_json(target/'results.json',dict(results=results,normal_step_replay_passed=True,
        interpretation='Signed changes in recorded kinetic energy and APIC norm at two representative steps; not a full-trajectory energy-error decomposition.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['run','analyze','trace'],required=True)
    p.add_argument('--variant',choices=['incremental32','full64'])
    p.add_argument('--baseline',type=Path,default=Path('results/ch04_impact_20261117_sphere'))
    a=p.parse_args()
    if a.stage=='run':
        if a.variant is None:p.error('--variant is required for a run')
        a.output.mkdir(exist_ok=True)
        shutil.copy2(__file__,a.output/Path(__file__).name)
        run(a.output,a.variant)
    elif a.stage=='trace':trace(a.output,a.baseline)
    else:analyze(a.output,a.baseline)


if __name__=='__main__':main()
