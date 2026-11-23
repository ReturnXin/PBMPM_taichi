"""Diagnostics and direct-particle figures for rigid-body reverse impacts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation,PillowWriter
from .ch04_impact_2d import write_json
from .ch04_impact_2d_analyze import setup,save
from .ch04_falling_analyze import particle_components,COLORS,LABELS
from .ch04_rigid_drop import RUNS


def summarize(out):
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    traces={n:np.genfromtxt(out/'data'/n/'trace.csv',delimiter=',',names=True) for n in RUNS}
    result={};c=p['config'];g=c['gravity'];dt=c['dt']
    for name,t in traces.items():
        contact=np.flatnonzero(np.hypot(t['contact_px'],t['contact_py'])>1e-10)
        # The dry control's analytic free fall ends at floor contact.
        events=contact if name!='rigid_control' else np.flatnonzero(abs(t['wall_py'])>1e-10)
        first=int(events[0]) if len(events) else len(t)
        n=np.rint(t['time'][:first]/dt)
        expected_y=p['body_center'][1]+g*dt**2*n*(n+1)/2
        scale=(p['body_mass']+(p['material_mass'] if name!='rigid_control' else 0))*abs(g)*p['duration']
        r=dict(records=len(t),first_contact_time=float(t['time'][contact[0]]) if len(contact) else None,
            final_x=float(t['cx'][-1]),final_y=float(t['cy'][-1]),min_y=float(t['cy'].min()),
            final_vx=float(t['vx'][-1]),final_vy=float(t['vy'][-1]),min_vy=float(t['vy'].min()),
            final_angle_deg=float(np.rad2deg(t['angle'][-1])),max_abs_angle_deg=float(np.rad2deg(abs(t['angle']).max())),
            max_shape_error=float(t['shape_error'].max()),max_penetration=float(t['penetration'].max()),
            max_box_penetration=float(t['box_penetration'].max()),
            max_momentum_residual_fraction=float(t['momentum_residual'].max()/scale),
            precontact_freefall_position_error=float(abs(t['cy'][:first]-expected_y).max()),
            precontact_freefall_velocity_error=float(abs(t['vy'][:first]-g*t['time'][:first]).max()),
            finite=bool(all(np.isfinite(t[k]).all() for k in t.dtype.names)))
        if name!='rigid_control':
            r['final_area_change_percent']=float(100*(t['area'][-1]/t['area'][0]-1))
        if name in ('elastic','sand'):
            r.update(elastic_strain_peak=float(t['elastic_strain_rms'].max()),
                elastic_strain_final=float(t['elastic_strain_rms'][-1]),
                total_strain_final=float(t['total_strain_rms'][-1]),
                min_J=float(t['min_J'].min()),min_elastic_det=float(t['min_elastic_det'].min()))
        result[name]=r
    radius=p['elastic_integrity']['neighbor_radius']
    with np.load(out/'data/elastic/frames.npz') as f:
        mask=f['material']!=1;connectivity=[particle_components(x[mask],radius) for x in f['x']]
    result['elastic'].update(max_connected_components=max(n for n,_ in connectivity),
        min_connected_fraction=min(f for _,f in connectivity),connectivity_radius=radius)
    limits=p['limits']
    audit=dict(all_completed=all(json.loads((out/'data'/n/'completion.json').read_text())['status']=='complete' for n in RUNS),
        finite=all(r['finite'] for r in result.values()),
        contact_in_all_materials=all(result[n]['first_contact_time'] is not None for n in COLORS),
        rigid_shape=all(r['max_shape_error']<limits['shape'] for r in result.values()),
        particle_body_contact=all(r['max_penetration']<limits['penetration'] for r in result.values()),
        box_contact=all(r['max_box_penetration']<limits['box_penetration'] for r in result.values()),
        external_momentum_budget=all(r['max_momentum_residual_fraction']<limits['momentum_residual_fraction'] for r in result.values()),
        freefall=all(r['precontact_freefall_position_error']<2e-5 and r['precontact_freefall_velocity_error']<2e-4 for r in result.values()),
        positive_deformation=all(result[n]['min_J']>0 and result[n]['min_elastic_det']>0 for n in ('elastic','sand')),
        elastic_integrity=result['elastic']['max_connected_components']<=p['elastic_integrity']['max_components'],
        human_scientific_review='pending')
    summary=dict(cases=result,audit=audit);write_json(out/'summary.json',summary)
    return p,traces,summary


def figures(out,p,traces,animation=True):
    setup();dest=out/'figures';dest.mkdir(exist_ok=True)
    c=p['config'];lo=c['origin']+c['boundary']*c['domain_size']/c['grid']
    hi=c['origin']+c['domain_size']-c['boundary']*c['domain_size']/c['grid']
    def decorate(ax):
        ax.add_patch(Rectangle((lo,lo),hi-lo,hi-lo,fill=False,edgecolor='#657182',linewidth=1.1))
        ax.set(xlim=(lo-.04,hi+.04),ylim=(lo-.04,hi+.04),aspect='equal')
    for name,color in COLORS.items():
        with np.load(out/'data'/name/'frames.npz') as f:
            x=f['x'];time=f['time'];mask=f['material']!=1
        fig,axs=plt.subplots(1,len(p['snapshot_times']),figsize=(2*len(p['snapshot_times']),2.55),layout='constrained')
        for ax,t in zip(axs,p['snapshot_times']):
            i=int(np.argmin(abs(time-t)));points=x[i]
            ax.scatter(points[mask,0],points[mask,1],s=.5,c=color,edgecolors='none',rasterized=True)
            ax.scatter(points[~mask,0],points[~mask,1],s=.5,c='#d87d32',edgecolors='none',rasterized=True)
            decorate(ax);ax.set(xticks=[],yticks=[],title=f't = {time[i]:.2f} s')
            for sp in ax.spines.values():sp.set_color('#dae0e6')
        save(fig,dest/(name+'_frames'))
        if animation:
            fig,ax=plt.subplots(figsize=(5.4,5.4),layout='constrained');decorate(ax)
            a=ax.scatter(x[0,mask,0],x[0,mask,1],s=2,c=color,edgecolors='none')
            b=ax.scatter(x[0,~mask,0],x[0,~mask,1],s=2,c='#d87d32',edgecolors='none')
            title=ax.set_title('');ax.set(xlabel='x / L',ylabel='y / L')
            def update(i):
                a.set_offsets(x[i,mask]);b.set_offsets(x[i,~mask]);title.set_text(f'刚体落向{LABELS[name]}   t = {time[i]:.3f} s')
            fps=p['gif_fps'];anim=FuncAnimation(fig,update,frames=len(x),interval=1000/fps)
            anim.save(dest/(name+'.gif'),writer=PillowWriter(fps=fps),dpi=110);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for name,t in traces.items():
        color=COLORS.get(name,'#657182');label=LABELS.get(name,'无材料对照');style='--' if name=='rigid_control' else '-'
        for ax,key,scale,ylabel in zip(axs,('cy','vy','angle'),(1,1,180/np.pi),('刚体质心高度 / L','竖直速度 / (L/s)','转角 / °')):
            ax.plot(t['time'],t[key]*scale,style,color=color,label=label)
            ax.set(xlabel='时间 / s',ylabel=ylabel);ax.grid(alpha=.2)
    axs[0].legend(frameon=False);save(fig,dest/'rigid_response')
    write_json(dest/'provenance.json',dict(source='data/*/frames.npz and trace.csv',
        plot_code='source_snapshot/validation/ch04_rigid_drop_analyze.py',
        direct_particle_positions=True,material_particles=p['material_particles'],rigid_particles=p['rigid_particles'],
        view=dict(low=lo-.04,high=hi+.04,box_low=lo,box_high=hi),
        gif=dict(fps=p['gif_fps'],physical_interval=p['frame_interval'],speed=p['gif_fps']*p['frame_interval'],frames=round(p['duration']/p['frame_interval'])+1),
        human_scientific_review='pending'))


def response_figures(out,p,traces):
    """Plot each material against the dry control using every saved time step."""
    setup();dest=out/'figures';dest.mkdir(exist_ok=True)
    quantities=(('cy',1,'刚体质心高度 / L','(a) 质心高度'),
                ('vy',1,'竖直速度 / (L/s)','(b) 竖直速度'),
                ('angle',180/np.pi,'转角 / °','(c) 转角'))
    for name,color in COLORS.items():
        fig,axs=plt.subplots(1,3,figsize=(11.4,3.6),sharex=True,layout='constrained')
        for ax,(key,scale,ylabel,title) in zip(axs,quantities):
            for case,style,line_color,label in (
                (name,'-',color,'刚体落向'+LABELS[name]),
                ('rigid_control','--','#657182','无材料对照')):
                t=traces[case]
                ax.plot(t['time'],t[key]*scale,style,color=line_color,
                        linewidth=1.7,label=label)
            ax.set(xlabel='时间 / s',ylabel=ylabel,title=title,xlim=(0,p['duration']))
            ax.set_xticks(np.arange(0,p['duration']+1e-9,.15))
            ax.grid(alpha=.2)
        axs[0].legend(frameon=False,fontsize=10)
        save(fig,dest/(name+'_response'))
    # Freeze this later plotting revision separately from the acquisition snapshot.
    snapshot=out/'postprocessing/validation/ch04_rigid_drop_analyze.py'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    helpers=('validation/ch04_impact_2d_analyze.py','validation/ch04_falling_analyze.py')
    helper_sources=[]
    for relative in helpers:
        active=Path(__file__).resolve().parents[1]/relative
        frozen=out/'source_snapshot'/relative
        if not frozen.exists() or active.read_bytes()!=frozen.read_bytes():
            frozen=out/'postprocessing'/relative
            frozen.parent.mkdir(parents=True,exist_ok=True)
            frozen.write_bytes(active.read_bytes())
        helper_sources.append(dict(path=frozen.relative_to(out).as_posix(),
            sha256=hashlib.sha256(frozen.read_bytes()).hexdigest()))
    provenance_path=dest/'provenance.json'
    provenance=json.loads(provenance_path.read_text(encoding='utf-8')) if provenance_path.exists() else {}
    provenance['per_material_response']=dict(
        source={n:f'data/{n}/trace.csv' for n in traces},
        source_sha256={n:hashlib.sha256((out/'data'/n/'trace.csv').read_bytes()).hexdigest() for n in traces},
        plot_code=snapshot.relative_to(out).as_posix(),
        plot_code_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        helper_sources=helper_sources,
        outputs=[f'figures/{n}_response.{ext}' for n in COLORS for ext in ('png','svg')],
        quantities=['cy','vy','angle'],angle_conversion='radians to degrees',
        records={n:len(t) for n,t in traces.items()},smoothing=False,resampling=False,
        control='rigid_control',human_scientific_review='pending')
    write_json(provenance_path,provenance)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--no-animation',action='store_true')
    mode.add_argument('--responses-only',action='store_true',help='Only plot per-material response curves from saved traces')
    a=parser.parse_args()
    if a.responses_only:
        p=json.loads((a.out/'protocol.json').read_text(encoding='utf-8'))
        t={n:np.genfromtxt(a.out/'data'/n/'trace.csv',delimiter=',',names=True) for n in RUNS}
        response_figures(a.out,p,t)
        print('Saved three per-material response figures from existing traces.')
        return
    p,t,s=summarize(a.out);figures(a.out,p,t,not a.no_animation);response_figures(a.out,p,t)
    print(json.dumps(s,indent=2))
    if not all(v for k,v in s['audit'].items() if k!='human_scientific_review'):
        raise RuntimeError('Inspect failed checks in summary.json')


if __name__=='__main__':main()
