"""Analyze saved ch04_impact trajectories; never reruns or adjusts the physics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from rigid_coupling.runner import write_json
from validation.ch04_impact import (CASES, BODY_CENTER, BODY_SIZE, WATER_CENTER, WATER_SIZE,
                                  WATER_RADIUS, WATER_VOLUME, SAMPLING_CELL, SPEED, verify_sources)

LABELS = {'center_one_way':'对心 · 单向', 'center_two_way':'对心 · 双向',
          'offset_one_way':'偏心 · 单向', 'offset_two_way':'偏心 · 双向',
          'mirror_two_way':'反向偏心 · 双向', 'fluid_control':'无刚体水流',
          'offset_two_way_dt0.001_it5':'偏心 · 步长减半',
          'offset_two_way_dt0.002_it10':'偏心 · 迭代加倍'}
PRIMARY=list(CASES)[:4]


def load_csv(path):
    with path.open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    return {key:np.array([float(r[key]) for r in rows]) for key in rows[0]}


def vec(data,name):
    return np.stack([data[f'{name}_{a}'] for a in 'xyz'],axis=1)


def compare(a,b):
    indices=np.rint(a['time']/(b['time'][1]-b['time'][0])).astype(int)
    assert np.allclose(a['time'],b['time'][indices],rtol=0,atol=1e-10)
    keep=np.arange(1,len(a['time']))
    dx=vec(a,'body_center')-vec(b,'body_center')[indices]
    dv=vec(a,'body_v')-vec(b,'body_v')[indices]
    qa=np.stack([a[f'body_q{i}'] for i in range(4)],axis=1)
    qb=np.stack([b[f'body_q{i}'] for i in range(4)],axis=1)[indices]
    qa/=np.linalg.norm(qa,axis=1)[:,None];qb/=np.linalg.norm(qb,axis=1)[:,None]
    theta=2*np.arccos(np.clip(abs((qa*qb).sum(axis=1)),0,1))*180/np.pi
    return dict(common_noninitial_samples=len(keep),position_rms_body_length=float(np.sqrt(np.mean((dx[keep]**2).sum(axis=1)))/.25),
                velocity_rms_incoming_speed=float(np.sqrt(np.mean((dv[keep]**2).sum(axis=1)))/SPEED),
                orientation_rms_deg=float(np.sqrt(np.mean(theta[keep]**2))))


def analyze(out):
    protocol=verify_sources(out)
    names=list(CASES)+['offset_two_way_dt0.001_it5','offset_two_way_dt0.002_it10']
    data={name:load_csv(out/name/'metrics.csv') for name in names}
    summaries={name:json.loads((out/name/'summary.json').read_text(encoding='utf-8')) for name in names}
    impulse=json.loads((out/'impulse_checks/results.json').read_text(encoding='utf-8'))
    assert impulse['passed']
    records=[]
    audit=[]
    for name in names:
        d=data[name]; s=summaries[name]
        assert s['completed'] and s['valid_isolated_run'] and s['source_unchanged'],name
        assert np.all(np.isfinite(np.concatenate(list(d.values())))),name
        assert len(d['time'])==s['samples'] and abs(d['time'][-1]-.5)<1e-12
        cfg=json.loads((out/name/'config.json').read_text(encoding='utf-8'))
        assert cfg['production_hashes']==protocol['production_hashes']
        assert cfg['config']['gravity']==cfg['config']['viscosity']==0
        assert cfg['config']['buoyancy_assist'] is False
        scale=d['fluid_mass'][0]*SPEED
        record=dict(case=name,label=LABELS[name],samples=len(d['time']),
                    max_penetration_h=float(d['penetration'].max()*32),
                    max_linear_momentum_drift_percent=s['max_linear_momentum_drift']*100,
                    max_angular_momentum_drift_percent=s['max_angular_momentum_drift']*100,
                    max_volume_change_percent=s['max_relative_volume_change']*100,
                    min_boundary_stencil_clearance=s['minimum_boundary_stencil_clearance'],
                    fluid_px_remaining_percent=float(d['fluid_p_x'][-1]/scale*100))
        if name!='fluid_control':
            ci=vec(d,'contact_impulse');nz=np.flatnonzero(np.linalg.norm(ci,axis=1)>1e-10)
            record.update(displacement_x=float(d['body_center_x'][-1]-d['body_center_x'][0]),
                          final_vx=float(d['body_v_x'][-1]),final_omega_z=float(d['body_omega_z'][-1]),
                          peak_angular_speed=float(np.linalg.norm(vec(d,'body_omega'),axis=1).max()),
                          signed_yaw_deg=float(np.arctan2(2*(d['body_q0'][-1]*d['body_q3'][-1]+d['body_q1'][-1]*d['body_q2'][-1]),
                                                1-2*(d['body_q2'][-1]**2+d['body_q3'][-1]**2))*180/np.pi),
                          body_px_received_percent=float(d['body_p_x'][-1]/scale*100),
                          first_recorded_contact_impulse_time=None if not len(nz) else float(d['time'][nz[0]]))
        # Independently recompute linear momentum at all five full-state snapshots.
        errors=[]
        for path in sorted((out/name).glob('state_*.npz')):
            p=np.load(path); f=p['material']==0
            momentum=(p['mass'][f,None].astype(float)*p['d'][f].astype(float)/cfg['config']['dt']).sum(axis=0)
            if len(p['body_mass']):momentum+=(p['body_mass'][:,None].astype(float)*p['body_v'].astype(float)).sum(axis=0)
            index=int(path.stem.split('_')[1])
            errors.append(float(np.linalg.norm(momentum-vec(d,'total_p')[index])))
        assert max(errors)<1e-12,name
        audit.append(dict(case=name,snapshot_momentum_max_difference=max(errors),passed=True))
        records.append(record)
    # The only difference within either primary pair is the coupling switch.
    for prefix in ['center','offset']:
        a=json.loads((out/f'{prefix}_one_way/config.json').read_text())['config']
        b=json.loads((out/f'{prefix}_two_way/config.json').read_text())['config']
        assert [k for k in a if a[k]!=b[k]]==['coupling']
        pa=np.load(out/f'{prefix}_one_way/state_0000.npz')
        pb=np.load(out/f'{prefix}_two_way/state_0000.npz')
        for k in pa.files:np.testing.assert_array_equal(pa[k],pb[k],err_msg=f'{prefix}/{k}')
    initial = np.load(out/'center_two_way/state_0000.npz')
    sampling = {}
    for material,size,center,key in [(0,WATER_SIZE,WATER_CENTER,'water_particle_count'),
                                     (1,BODY_SIZE,BODY_CENTER,'body_particle_count')]:
        mask = initial['material']==material
        local = initial['x'][mask].astype(float)-center
        cells = np.floor(local/SAMPLING_CELL).astype(int)
        count = len(local)
        assert count==protocol[key]
        assert np.all(abs(local)<size/2)
        if material==0:
            assert np.all(np.linalg.norm(local,axis=1)<WATER_RADIUS+1e-7)
        assert len(np.unique(cells,axis=0))==count
        assert len(np.unique(local,axis=0))==count
        mass = initial['mass'][mask].astype(float).sum()
        expected_mass = WATER_VOLUME if material==0 else np.prod(size)
        assert np.isclose(mass,expected_mass,rtol=2e-7,atol=0)
        centroid_error = float(np.linalg.norm(local.mean(axis=0)))
        assert centroid_error<1e-8
        sampling[str(material)] = dict(particles=count,unique_positions=count,
                                       occupied_cells=count,mass=float(mass),centroid_error=centroid_error)
    write_json(out/'sampling_audit.json',dict(passed=True,materials=sampling))
    off=data['offset_two_way'];mir=data['mirror_two_way']
    reflection=np.array([1.,-1.,1.])
    center_diff=vec(off,'body_center')-(BODY_CENTER+(vec(mir,'body_center')-BODY_CENTER)*reflection)
    velocity_diff=vec(off,'body_v')-vec(mir,'body_v')*reflection
    # Angular velocity is an axial vector: det(S)*S under a reflection.
    omega_diff=vec(off,'body_omega')+vec(mir,'body_omega')*reflection
    symmetry=dict(max_center_difference=float(np.linalg.norm(center_diff,axis=1).max()),
                  max_velocity_difference=float(np.linalg.norm(velocity_diff,axis=1).max()),
                  max_omega_difference=float(np.linalg.norm(omega_diff,axis=1).max()))
    result=dict(cases=records,symmetry=symmetry,
                sensitivity={name:compare(off,data[name]) for name in names[-2:]},
                analytical_checks=impulse,
                human_scientific_review='pending',performance_claim=False)
    write_json(out/'analysis.json',result)
    write_json(out/'audit.json',dict(passed=True,case_count=len(names),full_state_snapshot_count=5*len(names),
                                   primary_pairs_have_identical_initial_state=True,
                                   primary_pairs_only_change_coupling=True,
                                   production_hashes_match=True,independent_snapshot_checks=audit))
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(key for r in records for key in r))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    return result,data


def figures(out,data):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','figure.facecolor':'white'})
    target=out/'figures';target.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(target/f'{name}.png',dpi=240,bbox_inches='tight')
        fig.savefig(target/f'{name}.svg',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey='col',layout='constrained')
    for i,prefix in enumerate(['center','offset']):
        for mode,color,style in [('one_way','#666666','--'),('two_way','#1268b5','-')]:
            name=f'{prefix}_{mode}';d=data[name]
            for j,(key,label) in enumerate([('body_center_x','水平位移（长度单位）'),
                                           ('body_v_x','水平速度（长度单位/s）'),
                                           ('body_omega_z','绕 z 轴角速度（rad/s）')]):
                y=d[key]-(d[key][0] if j==0 else 0)
                axs[i,j].plot(d['time'],y,style,color=color,label=LABELS[name],lw=2)
                axs[i,j].set_ylabel(label);axs[i,j].grid(alpha=.2)
                if i==1:axs[i,j].set_xlabel('物理时间（s）')
        axs[i,0].legend(frameon=False)
        axs[i,0].set_title('对心冲击' if i==0 else '偏心冲击：水团上移 0.0625',loc='left',fontweight='bold')
    fig.suptitle('水流冲击的耦合响应',fontsize=16)
    save(fig,'response')
    fig,axs=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    for name,color in [('center_two_way','#1268b5'),('offset_two_way','#dd7034')]:
        d=data[name];scale=d['fluid_mass'][0]*SPEED
        axs[0].plot(d['time'],(d['fluid_p_x'][0]-d['fluid_p_x'])/scale*100,color=color,label=LABELS[name]+'：水损失')
        axs[0].plot(d['time'],d['body_p_x']/scale*100,'--',color=color,label=LABELS[name]+'：刚体获得')
        drift=np.linalg.norm(vec(d,'total_p')-vec(d,'total_p')[0],axis=1)/scale*100
        axs[1].plot(d['time'],drift,color=color,label=LABELS[name])
    for name,color in [('offset_two_way','#1268b5'),('mirror_two_way','#dd7034')]:
        d=data[name];axs[2].plot(d['time'],d['body_omega_z'],color=color,label=LABELS[name])
    for ax,label in zip(axs,['x 向动量 / 初始水流动量（%）','总线动量漂移（%）','绕 z 轴角速度（rad/s）']):
        ax.set_xlabel('物理时间（s）');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('冲击反馈检查',fontsize=15)
    save(fig,'momentum_and_mirror')
    fig,axs=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    variants=['offset_two_way','offset_two_way_dt0.001_it5','offset_two_way_dt0.002_it10']
    for name,style,color in zip(variants,['-','--',':'],['#1268b5','#dd7034','#339369']):
        d=data[name]
        for ax,key,label in zip(axs,['body_center_x','body_v_x','body_omega_z'],
                                ['质心 x（长度单位）','水平速度（长度单位/s）','绕 z 轴角速度（rad/s）']):
            ax.plot(d['time'],d[key],style,color=color,label=LABELS[name],lw=2);ax.set_ylabel(label)
    for ax in axs:ax.set_xlabel('物理时间（s）');ax.grid(alpha=.2)
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('偏心双向冲击的参数敏感性',fontsize=15)
    save(fig,'sensitivity')
    frames={name:np.load(out/name/'frames.npz') for name in PRIMARY}
    # Follow the same real particle IDs from the initial central slab.
    # Both materials use the same slab width; no display jitter or duplication.
    masks={name:{m:(f['material']==m)&(abs(f['x'][0,:,2]-.5)<1/64)
                 for m in [0,1]} for name,f in frames.items()}
    write_json(target/'particle_display.json',dict(
        projection='XY',selection='fixed particle IDs from initial |z-0.5| < 1/64',
        slab_thickness=1/32,water_color='#278ac4',rigid_color='#d47a24',
        marker='circle for both materials',display_jitter=False,duplicated_particles=False,
        selected_counts={name:{str(m):int(mask.sum()) for m,mask in mm.items()}
                         for name,mm in masks.items()}))
    fig,axs=plt.subplots(4,4,figsize=(12,10),layout='constrained')
    times=[.16,.24,.34,.5]
    for i,name in enumerate(PRIMARY):
        f=frames[name]
        for j,t in enumerate(times):
            k=int(np.argmin(abs(f['time']-t)));ax=axs[i,j]
            for m,color in [(0,'#278ac4'),(1,'#d47a24')]:
                x=f['x'][k,masks[name][m]]
                ax.scatter(x[:,0],x[:,1],s=1.05,color=color,alpha=.85,edgecolors='none')
            ax.set(xlim=(.15,.9),ylim=(.15,.9),aspect='equal')
            if i==0:ax.set_title(f't = {t:.2f} s')
            if j==0:ax.set_ylabel(LABELS[name]+'\ny')
            if i==3:ax.set_xlabel('x')
            ax.tick_params(labelsize=8)
    fig.suptitle('水流冲击过程\n蓝色：水粒子；橙色：刚体粒子（跟踪初始中央薄层）',fontsize=13)
    save(fig,'impact_frames')
    fig,axs=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    artists=[]
    for ax,name in zip(axs.flat,PRIMARY):
        f=frames[name];scatters=[]
        for m,color in [(0,'#278ac4'),(1,'#d47a24')]:
            x=f['x'][0,masks[name][m]]
            scatters.append(ax.scatter(x[:,0],x[:,1],s=2.2,color=color,alpha=.85,edgecolors='none'))
        ax.set(xlim=(.15,.9),ylim=(.15,.9),aspect='equal',title=LABELS[name],xlabel='x',ylabel='y')
        artists.append(scatters)
    title=fig.suptitle('水流冲击刚体 · 0.5 倍物理速度播放',fontsize=14)
    def update(k):
        for name,scatters in zip(PRIMARY,artists):
            f=frames[name]
            for m,scatter in enumerate(scatters):
                scatter.set_offsets(f['x'][k,masks[name][m],:2])
        title.set_text(f'水流冲击刚体 · t = {frames[PRIMARY[0]]["time"][k]:.2f} s · 0.5 倍速')
    animation=FuncAnimation(fig,update,frames=len(frames[PRIMARY[0]]['time']),interval=20)
    animation.save(target/'impact_comparison.gif',writer=PillowWriter(fps=50),dpi=110)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result,data=analyze(a.output);figures(a.output,data)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
