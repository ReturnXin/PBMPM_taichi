"""Gravity-aware diagnostics and direct particle figures for the supported impact."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation,PillowWriter
from .ch04_impact_2d import write_json
from .ch04_impact_2d_analyze import setup,save
from .ch04_falling import RUNS

COLORS={'water':'#2084cc','elastic':'#289776','sand':'#a77829'}
LABELS={'water':'水体','elastic':'弹性体','sand':'沙体'}


def particle_components(x,radius):
    """Spatial connectivity is a separation diagnostic, not a fracture model."""
    pairs=cKDTree(x).query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs),bool),(pairs[:,0],pairs[:,1])),shape=(len(x),len(x)))
    count,labels=connected_components(graph,directed=False)
    return int(count),float(np.bincount(labels).max()/len(x))


def summarize(out):
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    traces={n:np.genfromtxt(out/'data'/n/'trace.csv',delimiter=',',names=True) for n in RUNS}
    result={};gravity=p['config']['gravity'];dt=p['config']['dt']
    for name,t in traces.items():
        contact=np.flatnonzero(np.hypot(t['contact_px'],t['contact_py'])>1e-10)
        first=int(contact[0]) if len(contact) else len(t)
        scale=(p['body_mass']+(p['material_mass'] if name!='rigid_control' else 0))*abs(gravity)*p['duration']
        r=dict(records=len(t),first_contact_time=float(t['time'][first]) if len(contact) else None,
            final_dx=float(t['cx'][-1]-t['cx'][0]),final_dy=float(t['cy'][-1]-t['cy'][0]),
            final_vx=float(t['vx'][-1]),final_vy=float(t['vy'][-1]),
            final_angle_deg=float(np.rad2deg(t['angle'][-1])),final_omega=float(t['omega'][-1]),
            max_abs_angle_deg=float(np.rad2deg(abs(t['angle']).max())),
            max_shape_error=float(t['shape_error'].max()),max_penetration=float(t['penetration'].max()),
            max_momentum_residual_fraction=float(t['momentum_residual'].max()/scale),
            max_box_penetration=float(t['box_penetration'].max()),final_wall_impulse=[float(t['wall_px'][-1]),float(t['wall_py'][-1])],
            min_stencil_clearance=float(t['stencil_clearance'].min()),finite=bool(all(np.isfinite(t[k]).all() for k in t.dtype.names)))
        if name!='rigid_control':
            time=t['time'][:first];n=np.rint(time/dt)
            expected_y=p['material_center'][1]+gravity*dt**2*n*(n+1)/2
            r.update(precontact_freefall_position_error=float(abs(t['material_cy'][:first]-expected_y).max()),
                precontact_freefall_velocity_error=float(abs(t['material_vy'][:first]-gravity*time).max()),
                precontact_body_angle_deg=float(np.rad2deg(abs(t['angle'][:first]).max())),
                material_velocity_before_contact=float(t['material_vy'][first-1]),
                final_area_change_percent=float(100*(t['area'][-1]/t['area'][0]-1)))
            if name!='water':
                r.update(elastic_strain_peak=float(t['elastic_strain_rms'].max()),elastic_strain_final=float(t['elastic_strain_rms'][-1]),
                    total_strain_final=float(t['total_strain_rms'][-1]),plastic_strain_final=float(t['plastic_strain_mean'][-1]),
                    min_J=float(t['min_J'].min()),min_elastic_det=float(t['min_elastic_det'].min()))
        result[name]=r
    radius=p['elastic_integrity']['neighbor_radius']
    with np.load(out/'data/elastic/frames.npz') as frames:
        mask=frames['material']!=1
        connectivity=[particle_components(x[mask],radius) for x in frames['x']]
    result['elastic'].update(connectivity_radius=radius,connectivity_samples=len(connectivity),
        max_connected_components=max(v[0] for v in connectivity),
        min_connected_fraction=min(v[1] for v in connectivity))
    limits=p['limits']
    audit=dict(all_completed=all(json.loads((out/'data'/n/'completion.json').read_text())['status']=='complete' for n in RUNS),
        finite=all(r['finite'] for r in result.values()),
        shape=all(r['max_shape_error']<limits['shape'] for r in result.values()),
        particle_body_contact=all(r['max_penetration']<limits['penetration'] for r in result.values()),
        momentum_with_external_impulses=all(r['max_momentum_residual_fraction']<limits['momentum_residual_fraction'] for r in result.values()),
        box_contact=all(r['max_box_penetration']<2e-6 for r in result.values()),
        freefall=all(result[n]['precontact_freefall_position_error']<2e-5 and result[n]['precontact_freefall_velocity_error']<2e-4 for n in COLORS),
        supported_control=result['rigid_control']['max_abs_angle_deg']<.02,
        positive_deformation=all(result[n]['min_J']>0 and result[n]['min_elastic_det']>0 for n in ('elastic','sand')),
        elastic_integrity=result['elastic']['max_connected_components']<=p['elastic_integrity']['max_components'],
        human_scientific_review='pending')
    summary=dict(cases=result,audit=audit)
    write_json(out/'summary.json',summary)
    if not all(v for k,v in audit.items() if k!='human_scientific_review'):raise RuntimeError('Inspect failed checks in summary.json')
    return p,traces,summary


def figures(out,p,traces,animation=True):
    setup();dest=out/'figures';dest.mkdir(exist_ok=True)
    views={}
    for name,color in COLORS.items():
        f=np.load(out/'data'/name/'frames.npz');x=f['x'];mask=f['material']!=1
        c=p['config'];dx=c['domain_size']/c['grid']
        box_low=c['origin']+c['boundary']*dx;box_high=c['origin']+c['domain_size']-c['boundary']*dx
        low=np.full(2,box_low-.04);high=np.full(2,box_high+.04)
        views[name]=dict(low=low.tolist(),high=high.tolist(),all_particles=True)
        def decorate(ax):
            ax.add_patch(Rectangle((box_low,box_low),box_high-box_low,box_high-box_low,fill=False,edgecolor='#657182',linewidth=1.1))
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal')
        height=2*(high[1]-low[1])/(high[0]-low[0])+.55
        fig,axs=plt.subplots(1,len(p['snapshot_times']),figsize=(2*len(p['snapshot_times']),height),layout='constrained')
        for ax,time in zip(axs,p['snapshot_times']):
            i=int(np.argmin(abs(f['time']-time)));points=x[i]
            ax.scatter(points[mask,0],points[mask,1],s=.5,c=color,edgecolors='none',rasterized=True)
            ax.scatter(points[~mask,0],points[~mask,1],s=.5,c='#d87d32',edgecolors='none',rasterized=True)
            decorate(ax);ax.set(xticks=[],yticks=[],title=f't = {f["time"][i]:.2f} s')
            for sp in ax.spines.values():sp.set_visible(True);sp.set_color('#dae0e6')
        save(fig,dest/(name+'_frames'))
        fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        for case,style,c,label in [(name,'-',color,LABELS[name]+'下落'),('rigid_control','--','#657182','无材料对照')]:
            t=traces[case]
            for ax,key,scale,title in zip(axs,('cx','vx'),(1,1),('水平位移 / L','水平速度 / (L/s)')):
                ax.plot(t['time'],(t[key]-t[key][0])*scale,style,color=c,label=label)
                ax.set(xlabel='时间 / s',ylabel=title);ax.grid(alpha=.2)
        axs[0].legend(frameon=False);save(fig,dest/(name+'_response'))
        if not animation:
            f.close()
            continue
        fig,ax=plt.subplots(figsize=(5.4,5.4),layout='constrained');decorate(ax)
        a=ax.scatter(x[0,mask,0],x[0,mask,1],s=2,c=color,edgecolors='none')
        b=ax.scatter(x[0,~mask,0],x[0,~mask,1],s=2,c='#d87d32',edgecolors='none')
        title=ax.set_title('');ax.set(xlabel='x / L',ylabel='y / L')
        def update(i):
            a.set_offsets(x[i,mask]);b.set_offsets(x[i,~mask]);title.set_text(f'{LABELS[name]}偏心下落   t = {f["time"][i]:.3f} s')
        anim=FuncAnimation(fig,update,frames=len(x),interval=20)
        anim.save(dest/(name+'.gif'),writer=PillowWriter(fps=50),dpi=110);plt.close(fig);f.close()
    plot_code='postprocessing/ch04_falling_analyze.py' if (out/'postprocessing/ch04_falling_analyze.py').exists() else 'source_snapshot/validation/ch04_falling_analyze.py'
    write_json(dest/'provenance.json',dict(views=views,source='data/*/frames.npz and trace.csv',plot_code=plot_code,
        direct_particle_positions=True,material_particles=p['material_particles'],rigid_particles=p['rigid_particles'],support='Box floor; no pedestal',
        gif=dict(fps=50,physical_interval=p['frame_interval'],speed=50*p['frame_interval'],frames=round(p['duration']/p['frame_interval'])+1),human_scientific_review='pending'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--no-animation',action='store_true')
    a=parser.parse_args();p,t,s=summarize(a.out);figures(a.out,p,t,not a.no_animation);print(json.dumps(s,indent=2))


if __name__=='__main__':main()
