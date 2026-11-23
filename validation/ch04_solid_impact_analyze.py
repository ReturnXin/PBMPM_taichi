"""Full-trajectory diagnostics and particle figures for planar solid impacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from .ch04_impact_2d import write_json,PRIMARY_CASES
from .ch04_impact_2d_analyze import setup,save,load

COLORS={'elastic':'#289776','sand':'#a77829'}
NAMES={'elastic':'弹性体','sand':'沙体'}
ORANGE='#d87d32'


def summarize(out):
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    traces={n:load(out,n) for n in p['runs']}
    result={}
    for name,t in traces.items():
        momentum=np.column_stack((t['pfx']+t['pbx'],t['pfy']+t['pby']))
        scale=np.linalg.norm(momentum[0])
        peak=int(np.argmax(t['elastic_strain_rms']))
        result[name]=dict(displacement=float(t['cx'][-1]-t['cx'][0]),vx=float(t['vx'][-1]),
            omega=float(t['omega'][-1]),angle_deg=float(np.rad2deg(t['angle'][-1])),
            body_momentum_percent=float(100*t['pbx'][-1]/scale),
            momentum_drift_percent=float(100*np.max(np.linalg.norm(momentum-momentum[0],axis=1))/scale),
            kinetic_retained_percent=float(100*t['kinetic'][-1]/t['kinetic'][0]),
            mean_speed=float(t['mean_speed'][-1]),
            max_area_change_percent=float(100*np.max(abs(t['area']/t['area'][0]-1))),
            final_area_change_percent=float(100*(t['area'][-1]/t['area'][0]-1)),
            elastic_strain_peak=float(t['elastic_strain_rms'][peak]),elastic_strain_peak_time=float(t['time'][peak]),
            elastic_strain_final=float(t['elastic_strain_rms'][-1]),
            total_strain_final=float(t['total_strain_rms'][-1]),plastic_strain_final=float(t['plastic_strain_mean'][-1]),
            min_J=float(t['min_J'].min()),min_elastic_det=float(t['min_elastic_det'].min()),
            max_penetration=float(t['penetration'].max()),max_shape_error=float(t['shape_error'].max()),
            min_stencil_clearance=float(t['stencil_clearance'].min()),
            finite=bool(all(np.isfinite(t[k]).all() for k in t.dtype.names)),records=len(t))
    with np.load(out/'data/offset_one_way/initial.npz') as a,np.load(out/'data/offset_two_way/initial.npz') as b:
        same=all(np.array_equal(a[k],b[k]) for k in a.files)
    audit=dict(initial_pairs_equal=same,
        trajectories_complete=all(json.loads((out/'data'/n/'completion.json').read_text())['status']=='complete' for n in traces),
        states_finite=all(v['finite'] for v in result.values()),
        positive_deformation=all(min(v['min_J'],v['min_elastic_det'])>0 for v in result.values()),
        no_boundary_stencil_overlap=all(v['min_stencil_clearance']>=1 for v in result.values()),
        rigid_shape_check=all(v['max_shape_error']<p['checks']['shape_error'] for v in result.values()),
        penetration_check=all(v['max_penetration']<p['config']['domain_size']/p['config']['grid']*.05 for v in result.values()),
        momentum_check=all(result[n]['momentum_drift_percent']<.01 for n in ('offset_two_way','fluid_control')),
        human_scientific_review='pending')
    summary=dict(material=p['material'],cases=result,audit=audit,
        strain_definition='Mass-weighted RMS Frobenius norm of log singular values; F is recoverable, F_total is kinematic. plastic_strain_mean accumulates the norm of each logarithmic return correction; not a physical calibrated plastic strain.',
        energy_note='Kinetic energy excludes stored elastic energy and is not total solid energy.')
    write_json(out/'summary.json',summary)
    if not all(v for k,v in audit.items() if k!='human_scientific_review'):
        raise RuntimeError('Inspect failed simulation checks in summary.json before writing conclusions')
    return p,traces,summary


def figures(out,p,traces,animation=True):
    setup();dest=out/'figures';dest.mkdir(exist_ok=True)
    material=p['material'];color=COLORS[material];label=NAMES[material]
    frames={n:np.load(out/'data'/n/'frames.npz') for n in PRIMARY_CASES}
    allx=np.concatenate([f['x'].reshape(-1,2) for f in frames.values()])
    low=allx.min(axis=0)-.025;high=allx.max(axis=0)+.025
    write_json(dest/'view.json',dict(low=low.tolist(),high=high.tolist(),all_particles=True,
        continuum_count=p['water_particles'],rigid_count=p['rigid_particles'],frames=51,
        display='Raw saved particle centers, no interpolation or added points; common axes within this experiment.'))
    height=4*(high[1]-low[1])/(high[0]-low[0])+.75
    fig,axs=plt.subplots(2,5,figsize=(10,height),layout='constrained')
    for row,n in enumerate(PRIMARY_CASES):
        f=frames[n];mask=f['material']!=1
        for col,time in enumerate(p['snapshot_times']):
            ax=axs[row,col];idx=int(np.argmin(abs(f['time']-time)));x=f['x'][idx]
            ax.scatter(x[mask,0],x[mask,1],s=.4,c=color,edgecolors='none',rasterized=True)
            ax.scatter(x[~mask,0],x[~mask,1],s=.4,c=ORANGE,edgecolors='none',rasterized=True)
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal',xticks=[],yticks=[])
            if row==0:ax.set_title(f't = {f["time"][idx]:.2f} s')
            if col==0:ax.set_ylabel('单向耦合' if row==0 else '双向耦合')
            for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#dae0e6')
    save(fig,dest/'impact_frames')
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for n,style in zip(PRIMARY_CASES,('--','-')):
        t=traces[n]
        for ax,key,title,scale in zip(axs,('cx','vx','angle'),('水平位移 / L','水平速度 / (L/s)','转角 / °'),(1,1,180/np.pi)):
            y=t[key]-(t[key][0] if key=='cx' else 0)
            ax.plot(t['time'],y*scale,style,color=color if style=='-' else '#636c78',label='双向' if style=='-' else '单向')
            ax.set(xlabel='时间 / s',ylabel=title);ax.grid(alpha=.2)
    axs[0].legend(frameon=False);save(fig,dest/'response')
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    t=traces['offset_two_way'];scale=t['pfx'][0]
    axs[0].plot(t['time'],(scale-t['pfx'])/scale,color=color,label=label+'损失')
    axs[0].plot(t['time'],t['pbx']/scale,'--',color=ORANGE,label='刚体获得')
    axs[0].set(ylabel='水平动量 / 初始材料动量')
    if material=='elastic':
        axs[1].plot(t['time'],t['elastic_strain_rms'],color=color,label='可恢复形变')
        axs[1].plot(traces['fluid_control']['time'],traces['fluid_control']['elastic_strain_rms'],'--',color='#636c78',label='无障碍对照')
        axs[1].set(ylabel='对数主伸长的均方根范数')
    else:
        axs[1].plot(t['time'],t['total_strain_rms'],color=color,label='总形变')
        axs[1].plot(t['time'],t['elastic_strain_rms'],'--',color='#636c78',label='可恢复形变')
        axs[1].set(ylabel='对数主伸长的均方根范数')
    for ax in axs:ax.set_xlabel('时间 / s');ax.grid(alpha=.2);ax.legend(frameon=False)
    save(fig,dest/'diagnostics')
    if animation:
        fig,axs=plt.subplots(1,2,figsize=(8,4.8),layout='constrained')
        artists=[]
        for ax,n in zip(axs,PRIMARY_CASES):
            f=frames[n];mask=f['material']!=1;x=f['x'][0]
            a=ax.scatter(x[mask,0],x[mask,1],s=2,c=color,edgecolors='none')
            b=ax.scatter(x[~mask,0],x[~mask,1],s=2,c=ORANGE,edgecolors='none')
            title=ax.set_title('')
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal',xlabel='x / L',ylabel='y / L')
            artists.append((a,b,title,mask,n))
        def update(i):
            for a,b,title,mask,n in artists:
                f=frames[n];a.set_offsets(f['x'][i,mask]);b.set_offsets(f['x'][i,~mask])
                title.set_text(f'{label} · '+('单向' if n.endswith('one_way') else '双向')+f'   t={f["time"][i]:.2f} s')
        anim=FuncAnimation(fig,update,frames=len(frames[PRIMARY_CASES[0]]['time']),interval=40)
        anim.save(dest/'impact_comparison.gif',writer=PillowWriter(fps=25),dpi=110)
        plt.close(fig)
    for f in frames.values():f.close()
    plot_code='postprocessing/ch04_solid_impact_analyze.py' if (out/'postprocessing/ch04_solid_impact_analyze.py').exists() else 'source_snapshot/validation/ch04_solid_impact_analyze.py'
    write_json(dest/'provenance.json',dict(source='data/*/frames.npz and trace.csv',code=plot_code,
        transforms='Fixed equal axes; direct particle scatter; angle converted radians to degrees; momentum normalized by initial material momentum; no time shifts.',
        gif=dict(fps=25,physical_frame_interval=.01,playback_speed=.25),human_review='pending'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--no-animation',action='store_true')
    args=parser.parse_args();p,t,s=summarize(args.out);figures(args.out,p,t,not args.no_animation)
    print(json.dumps(s,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
