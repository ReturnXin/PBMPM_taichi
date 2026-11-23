"""Analyze complete water-entry trajectories and render actual particle states."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from .ch04_drop_2d import DEFAULT_OUT,write_json,digest

BLUE,ORANGE='#2084cc','#d87d32'
CASES=('water_entry','pool_control','dry_fall')


def read_trace(path):return np.genfromtxt(path,delimiter=',',names=True,encoding='utf-8')


def summarize(out):
    p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    traces={name:read_trace(out/'data'/name/'trace.csv') for name in CASES}
    w,d=traces['water_entry'],traces['dry_fall']
    settling=read_trace(out/'data/settling.csv')
    dt=p['config']['dt'];g=p['config']['gravity'];y0=p['spec']['release_height']
    hit=np.flatnonzero(np.hypot(w['contact_px'],w['contact_py'])>1e-8)
    contact=int(hit[0]) if len(hit) else None
    # Stop the analytical check before the first body-wall interaction.
    wall=np.flatnonzero(d['body_wall_clearance']<=1e-6)
    stop=int(wall[0]) if len(wall) else len(d)
    free=d[:stop];t=free['time']
    exact_y=y0+.5*g*t*(t+dt)
    result={}
    for name,v in traces.items():
        has_water=name!='dry_fall'
        result[name]=dict(records=len(v),finite=bool(all(np.isfinite(v[k]).all() for k in v.dtype.names)),
            final_y=float(v['cy'][-1]),final_vy=float(v['vy'][-1]),
            min_y=float(v['cy'].min()),min_y_time=float(v['time'][v['cy'].argmin()]),
            min_vy=float(v['vy'].min()),min_vy_time=float(v['time'][v['vy'].argmin()]),
            max_shape_error=float(v['shape_error'].max()),max_penetration=float(v['penetration'].max()),
            min_body_wall_clearance=float(v['body_wall_clearance'].min()),max_cfl=float(v['cfl'].max()),
            max_fluid_rms_speed=float(v['fluid_rms_speed'].max()),final_fluid_rms_speed=float(v['fluid_rms_speed'][-1]),
            max_water_y=float(v['water_max_y'].max()),max_water_surface_p99=float(v['water_surface_p99'].max()),
            max_area_change_percent=float(100*np.max(abs(v['water_area']/v['water_area'][0]-1))) if has_water else None)
    equal={}
    with np.load(out/'data/settled_pool.npz') as a,np.load(out/'data/water_entry/initial.npz') as b,np.load(out/'data/pool_control/initial.npz') as c:
        for key in a.files:
            equal[key]=bool(np.array_equal(a[key],b[key][b['material']==0]) and np.array_equal(a[key],c[key]))
    reference=dict(last_time=float(t[-1]),max_y_error=float(np.max(abs(free['cy']-exact_y))),
                   max_vy_error=float(np.max(abs(free['vy']-g*t))))
    if contact is not None:
        before=w[:contact]
        reference['wet_precontact_max_y_error']=float(np.max(abs(before['cy']-(y0+.5*g*before['time']*(before['time']+dt)))))
        reference['wet_precontact_max_vy_error']=float(np.max(abs(before['vy']-g*before['time'])))
    # The body is externally accelerated by gravity. This is a body impulse
    # balance, not conservation of total momentum in a walled pool.
    residual=p['body_mass']*(w['vy']-w['vy'][0]-g*w['time'])-w['contact_py']
    audit=dict(trajectories_complete=all(json.loads((out/'data'/name/'completion.json').read_text())['complete'] for name in CASES),
        shared_water_initial_state=all(equal.values()),shared_water_fields=equal,
        states_finite=all(r['finite'] for r in result.values()),
        gravity_reference_pass=reference['max_y_error']<2e-5 and reference['max_vy_error']<2e-4,
        rigid_shape_pass=result['water_entry']['max_shape_error']<2e-6,
        wet_body_clear_of_walls=result['water_entry']['min_body_wall_clearance']>1e-5,
        entry_observed=contact is not None,human_scientific_review='pending')
    summary=dict(cases=result,first_contact=None if contact is None else dict(time=float(w['time'][contact]),
        cy=float(w['cy'][contact]),vy=float(w['vy'][contact])),free_fall_reference=reference,
        max_body_vertical_impulse_residual=float(np.max(abs(residual))),
        settling=dict(duration=p['spec']['settle_time'],final_rms_speed=float(settling['fluid_rms_speed'][-1]),
                      final_surface_p99=float(settling['water_surface_p99'][-1]),
                      area_change_percent=float(100*(settling['water_area'][-1]/settling['water_area'][0]-1))),
        audit=audit)
    write_json(out/'summary.json',summary)
    return p,traces,summary


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


def figures(out,p,traces,summary,animation=True):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':11,'savefig.facecolor':'white'})
    dest=out/'figures';dest.mkdir(exist_ok=True)
    frames=np.load(out/'data/water_entry/frames.npz');x=frames['x'];times=frames['time'];f=frames['material']==0
    c=p['config'];dx=c['domain_size']/c['grid'];lo=c['origin']+c['boundary']*dx;hi=c['origin']+c['domain_size']-c['boundary']*dx
    def pool_axes(ax):
        ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xticks=[],yticks=[])
        ax.plot([lo,lo,hi,hi],[hi,lo,lo,hi],color='#596878',lw=1.3)
        for sp in ax.spines.values():sp.set_visible(False)
    selected=np.array([0,.2,.26,.35,.5,.8])*times[-1]/.8
    fig,axs=plt.subplots(2,3,figsize=(10,6.8),layout='constrained')
    for ax,target in zip(axs.flat,selected):
        i=int(np.argmin(abs(times-target)));pool_axes(ax)
        ax.scatter(x[i,f,0],x[i,f,1],s=.32,c=BLUE,edgecolors='none',rasterized=True)
        ax.scatter(x[i,~f,0],x[i,~f,1],s=.32,c=ORANGE,edgecolors='none',rasterized=True)
        ax.set_title(f't = {times[i]:.2f} s')
    save(fig,dest/'drop_frames')
    w=traces['water_entry'];d=traces['dry_fall'];control=traces['pool_control']
    fig,axs=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    t=d['time'][d['time']<=summary['free_fall_reference']['last_time']]
    axs[0].plot(w['time'],w['cy'],c=ORANGE,label='刚体落水')
    axs[0].plot(t,p['spec']['release_height']+.5*c['gravity']*t*(t+c['dt']),'--',c='#596878',label='自由落体参考')
    axs[1].plot(w['time'],w['vy'],c=ORANGE,label='刚体落水')
    axs[1].plot(t,c['gravity']*t,'--',c='#596878',label='自由落体参考')
    axs[2].plot(w['time'],w['fluid_rms_speed'],c=BLUE,label='刚体落水')
    axs[2].plot(control['time'],control['fluid_rms_speed'],'--',c='#596878',label='空池对照')
    for ax,title,ylabel in zip(axs,['刚体高度','竖直速度','水体运动'],['质心高度 / L','竖直速度 / (L/s)','粒子均方根速度 / (L/s)']):
        ax.set(title=title,xlabel='释放后时间 / s',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
        if summary['first_contact']:ax.axvline(summary['first_contact']['time'],c='#999999',ls=':',lw=1)
    save(fig,dest/'drop_response')
    if animation:
        fig,ax=plt.subplots(figsize=(7.5,7.8),layout='constrained');pool_axes(ax)
        wf=ax.scatter(x[0,f,0],x[0,f,1],s=1.05,c=BLUE,edgecolors='none')
        rb=ax.scatter(x[0,~f,0],x[0,~f,1],s=1.05,c=ORANGE,edgecolors='none')
        ax.set_title('二维刚体落水',fontsize=17)
        label=ax.text(.045,.96,'',transform=ax.transAxes,va='top',fontsize=12)
        ax.text(.5,-.015,f'水粒子 {f.sum():,}  ·  刚体粒子 {(~f).sum():,}  ·  0.25 倍速',transform=ax.transAxes,ha='center',fontsize=10)
        def update(i):
            wf.set_offsets(x[i,f]);rb.set_offsets(x[i,~f]);label.set_text(f't = {times[i]:.2f} s')
        ani=FuncAnimation(fig,update,frames=len(times),interval=40)
        ani.save(dest/'drop.gif',writer=PillowWriter(fps=25),dpi=110);plt.close(fig)
    frames.close()
    # Both input and output hashes bind the rendered displays to this run.
    files=[out/'protocol.json',out/'summary.json',out/'data/water_entry/frames.npz']
    files+=list((out/'data').rglob('trace.csv'))+list(dest.glob('*'))
    write_json(out/'figure_manifest.json',dict(analysis_sha256=digest(Path(__file__)),
        files={path.relative_to(out).as_posix():digest(path) for path in files if path.is_file()},
        transforms='All recorded particles, fixed full-domain view, no interpolation or added particles.',
        units='L is the length unit of the simulation; time in seconds.',gif_fps=25,physical_frame_interval=.01,playback_speed=.25,
        human_scientific_review='pending'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--no-animation',action='store_true')
    a=parser.parse_args();p,traces,summary=summarize(a.out)
    figures(a.out,p,traces,summary,not a.no_animation)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
