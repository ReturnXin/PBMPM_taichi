"""Complete-trajectory analysis and direct particle plots for the planar study."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from .ch04_impact_2d import RUNS,PRIMARY_CASES,write_json

LABELS={'offset_one_way':'偏心 · 单向',
        'offset_two_way':'偏心 · 双向','mirror_two_way':'反向偏心 · 双向','fluid_control':'自由水团',
        'offset_half_dt':'步长减半','offset_double_iterations':'迭代加倍'}
BLUE,ORANGE='#2084cc','#d87d32'


def setup():
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def load(out,name):
    data = out/'data' if (out/'data').is_dir() else out
    return np.genfromtxt(data/name/'trace.csv',delimiter=',',names=True,encoding='utf-8')


def summarize(out):
    data = out/'data' if (out/'data').is_dir() else out
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    result={}
    traces={name:load(out,name) for name in RUNS}
    scale=p['water_mass']*p['speed']
    for name,t in traces.items():
        momentum=np.column_stack((t['pfx']+t['pbx'],t['pfy']+t['pby']))
        result[name]=dict(displacement=float(t['cx'][-1]-t['cx'][0]),vx=float(t['vx'][-1]),omega=float(t['omega'][-1]),
                         angle_deg=float(np.rad2deg(t['angle'][-1])),body_momentum_percent=float(100*t['pbx'][-1]/scale),
                         momentum_drift_percent=float(100*np.max(np.linalg.norm(momentum-momentum[0],axis=1))/scale),
                         angular_drift_percent=float(100*np.max(abs(t['angular']-t['angular'][0]))/(scale*.25)),
                         kinetic_retained_percent=float(100*t['kinetic'][-1]/t['kinetic'][0]),
                         kinetic_affine_retained_percent=float(100*(t['kinetic'][-1]+t['affine_norm'][-1])/(t['kinetic'][0]+t['affine_norm'][0])),
                         max_area_change_percent=float(100*np.max(abs(t['area']/t['area'][0]-1))),
                         mean_speed=float(t['mean_speed'][-1]),max_penetration=float(t['penetration'].max()),
                         max_shape_error=float(t['shape_error'].max()),min_stencil_clearance=float(t['stencil_clearance'].min()),
                         finite=bool(all(np.isfinite(t[k]).all() for k in t.dtype.names)),records=len(t))
    baseline=traces['offset_two_way']
    sensitivity={}
    for name in ('offset_half_dt','offset_double_iterations'):
        t=traces[name]
        ids=np.searchsorted(t['time'],baseline['time'][1:])
        assert np.allclose(t['time'][ids],baseline['time'][1:])
        pos=np.column_stack([t[k][ids]-baseline[k][1:] for k in ('cx','cy')])
        vel=np.column_stack([t[k][ids]-baseline[k][1:] for k in ('vx','vy')])
        angle=t['angle'][ids]-baseline['angle'][1:]
        sensitivity[name]=dict(position_rms_percent=float(100*np.sqrt(np.mean(np.sum(pos*pos,axis=1)))/.25),
                               velocity_rms_percent=float(100*np.sqrt(np.mean(np.sum(vel*vel,axis=1)))/.75),
                               angle_rms_deg=float(np.rad2deg(np.sqrt(np.mean(angle*angle)))))
    mirror=traces['mirror_two_way']
    pair_checks={}
    for prefix in ('offset',):
        with np.load(data/(prefix+'_one_way')/'initial.npz') as a,np.load(data/(prefix+'_two_way')/'initial.npz') as b:
            pair_checks[prefix]=all(np.array_equal(a[k],b[k]) for k in a.files)
    audit=dict(initial_pairs_equal=pair_checks,
               mirror_max_omega_sum=float(np.max(abs(mirror['omega']+baseline['omega']))),
               mirror_max_position_difference=float(np.max(np.hypot(mirror['cx']-baseline['cx'],mirror['cy']+baseline['cy']-1))),
               trajectories_complete=all(json.loads((data/name/'completion.json').read_text())['status']=='complete' for name in RUNS),
               states_finite=all(v['finite'] for v in result.values()),
               no_boundary_stencil_overlap=all(v['min_stencil_clearance']>=1 for v in result.values()),
               rigid_shape_check=all(v['max_shape_error']<p['checks']['shape_error'] for v in result.values()),
               human_scientific_review='pending')
    summary=dict(cases=result,sensitivity=sensitivity,audit=audit)
    write_json(out/'summary.json',summary)
    return traces,summary


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


def figures(out,traces,animation=True):
    data_root = out/'data' if (out/'data').is_dir() else out
    setup()
    dest=out/'figures';dest.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),sharex=True,layout='constrained')
    for mode,style in (('two_way','-'),('one_way','--')):
        t=traces['offset_'+mode]
        for col,(key,label,scale) in enumerate((('cx','水平位移',1),('vx','水平速度',1),('angle','转角 / °',180/np.pi))):
            y=t[key]-(t[key][0] if key=='cx' else 0)
            axs[col].plot(t['time'],y*scale,style,color=BLUE if mode=='two_way' else ORANGE,label='双向' if mode=='two_way' else '单向')
            axs[col].set_ylabel(label);axs[col].grid(alpha=.2)
            axs[col].set_title('偏心冲击 · '+label.split(' /')[0])
            axs[col].set_xlabel('时间 / s')
    axs[0].legend(frameon=False)
    save(fig,dest/'response')
    names=list(PRIMARY_CASES)
    frames={name:np.load(data_root/name/'frames.npz') for name in names}
    # Identical view contains every saved particle in every main trajectory.
    allx=np.concatenate([frames[n]['x'].reshape(-1,2) for n in names])
    low=allx.min(axis=0)-.025;high=allx.max(axis=0)+.025
    write_json(dest/'view.json',dict(cases=names,low=low.tolist(),high=high.tolist(),all_particles=True,
                                   water_count=int((frames[names[0]]['material']==0).sum()),rigid_count=int((frames[names[0]]['material']==1).sum())))
    times=(0.,.16,.24,.34,.5)
    fig,axs=plt.subplots(2,5,figsize=(9.5,7),layout='constrained')
    for row,name in enumerate(names):
        data=frames[name];f=data['material']==0
        for col,time in enumerate(times):
            ax=axs[row,col];idx=int(np.argmin(abs(data['time']-time)));x=data['x'][idx]
            ax.scatter(x[f,0],x[f,1],s=.25,c=BLUE,edgecolors='none',rasterized=True)
            ax.scatter(x[~f,0],x[~f,1],s=.25,c=ORANGE,edgecolors='none',rasterized=True)
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal',xticks=[],yticks=[])
            if row==0:ax.set_title(f't = {data["time"][idx]:.2f} s')
            if col==0:ax.set_ylabel(LABELS[name])
            for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#dae0e6')
    save(fig,dest/'impact_frames')
    fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for name,color in (('offset_two_way',ORANGE),):
        t=traces[name];scale=traces[name]['pfx'][0];label=LABELS[name].split(' · ')[0]
        axs[0].plot(t['time'],(t['pfx'][0]-t['pfx'])/scale,color=color,label=label+'水体损失')
        axs[0].plot(t['time'],t['pbx']/scale,'--',color=color,label=label+'刚体获得')
        drift=np.hypot(t['pfx']+t['pbx']-t['pfx'][0],t['pfy']+t['pby']-t['pfy'][0])/scale*100
        axs[1].plot(t['time'],drift,color=color,label=label)
    for name,style in (('offset_two_way','-'),('mirror_two_way','--')):
        t=traces[name];axs[2].plot(t['time'],t['omega'],style,label='上偏心' if style=='-' else '下偏心')
    for ax,title,ylabel in zip(axs,['动量交换','总线动量漂移','偏心方向'],['归一化动量','相对漂移 / %','角速度 / (rad/s)']):
        ax.set(title=title,xlabel='时间 / s',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
    save(fig,dest/'momentum_and_mirror')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for name in ('fluid_control','offset_two_way'):
        t=traces[name]
        axs[0].plot(t['time'],100*t['kinetic']/t['kinetic'][0],label=LABELS[name])
        axs[1].plot(t['time'],100*t['affine_norm']/t['kinetic'][0],label=LABELS[name])
    for ax,title,ylabel in zip(axs,['粒子及刚体动能','APIC 仿射项'],['动能保留 / %','占初始动能 / %']):
        ax.set(title=title,xlabel='时间 / s',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(frameon=False)
    save(fig,dest/'energy')
    if animation:
        fig,axs=plt.subplots(1,2,figsize=(8,5.8),layout='constrained')
        artists=[]
        for ax,name in zip(axs.flat,names):
            d=frames[name];f=d['material']==0;x=d['x'][0]
            wf=ax.scatter(x[f,0],x[f,1],s=.5,c=BLUE,edgecolors='none')
            rb=ax.scatter(x[~f,0],x[~f,1],s=.5,c=ORANGE,edgecolors='none')
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal',xticks=[],yticks=[],title=LABELS[name])
            label=ax.text(.03,.96,'',transform=ax.transAxes,va='top')
            artists.append((wf,rb,label,f,d))
        def update(i):
            for wf,rb,label,f,d in artists:
                wf.set_offsets(d['x'][i][f]);rb.set_offsets(d['x'][i][~f]);label.set_text(f't = {d["time"][i]:.2f} s')
        ani=FuncAnimation(fig,update,frames=len(frames[names[0]]['time']),interval=40)
        ani.save(dest/'impact_comparison.gif',writer=PillowWriter(fps=25),dpi=100)
        plt.close(fig)
    for f in frames.values():f.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--no-animation',action='store_true')
    a=p.parse_args();traces,summary=summarize(a.out);figures(a.out,traces,not a.no_animation)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
