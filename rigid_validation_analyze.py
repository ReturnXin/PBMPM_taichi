"""Analyze registered trajectories without treating failed references as truth."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from rigid_validation_run import DEFAULT_OUTPUT,SIGNS
from rigid_coupling.rigid import rotation_numpy

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'figure.dpi':130,'savefig.dpi':220})
COLORS=['#2369a8','#d75c29','#36855f','#8b5f9f','#656565']
def read_csv(path):
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    return {k:np.array([float(r[k]) if r.get(k) else np.nan for r in rows]) for k in rows[0]}
def vector(r,prefix,n=3):return np.stack([r[prefix+str(i)] for i in range(n)],axis=1)
def quat_distance(a,b):
    a=a/np.linalg.norm(a,axis=-1,keepdims=True);b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=-1)),0,1)))
def compare(a,b,end=1.6):
    times=a['time'][(a['time']>1e-10)&(a['time']<=end+1e-10)]
    result={'samples':len(times),'end':end,'initial_time_excluded':True}
    for prefix,scale,key in [('body_x',.2,'position_rms_L'),('body_v',1.4,'velocity_rms_U')]:
        aa=np.stack([np.interp(times,a['time'],a[prefix+str(i)]) for i in range(3)],axis=1)
        bb=np.stack([np.interp(times,b['time'],b[prefix+str(i)]) for i in range(3)],axis=1)
        d=np.linalg.norm(aa-bb,axis=1)/scale
        result[key]=float(np.sqrt(np.mean(d*d)));result[key.replace('rms','max')]=float(np.max(d))
    # Local runs have exactly matching times. Linear interpolation of components
    # is normalized below and is used only for a potential adaptive reference.
    aa=np.stack([np.interp(times,a['time'],a['body_q'+str(i)]) for i in range(4)],axis=1)
    bb=np.stack([np.interp(times,b['time'],b['body_q'+str(i)]) for i in range(4)],axis=1)
    angle=quat_distance(aa,bb)
    result.update(orientation_rms_deg=float(np.sqrt(np.mean(angle**2))),orientation_max_deg=float(np.max(angle)))
    return result
def crossing(t,y):
    ix=np.where((y[:-1]<0)&(y[1:]>=0))[0]
    if not len(ix):return None
    i=ix[0];return float(t[i]-y[i]*(t[i+1]-t[i])/(y[i+1]-y[i]))
def summarize(r,case,meta):
    result={'status':meta['status'],'last_time':float(r['time'][-1]),'sample_count':len(r['time']),
        'full_duration':bool(r['time'][-1]>=case['duration']-1e-9),
        'max_absolute_volume_error':float(np.max(np.abs(r['fluid_volume_error']))),
        'max_sampled_penetration':float(r['max_penetration_so_far'][-1])}
    if case['experiment']=='A_float':
        mask=r['time']>=4.5-1e-10
        if not np.any(mask):return result
        result.update(window_samples=int(mask.sum()),fluid_volume_error_final=float(r['fluid_volume_error'][-1]),
            fluid_vrms_window=float(np.sqrt(np.mean(r['fluid_vrms'][mask]**2))),
            surface_mean=float(np.nanmean(r['surface_y'][mask])),surface_fit_rms_max=float(np.nanmax(r['surface_rms'][mask])))
        if 'body_x1' in r:
            v=vector(r,'body_v');w=vector(r,'body_omega');chi=r['submerged_fraction'][mask]
            rho=case['geometry']['density_ratio']
            vrms=float(np.sqrt(np.mean(np.sum(v[mask]**2,axis=1)))/1.4)
            wrms=float(np.sqrt(np.mean(np.sum(w[mask]**2,axis=1)))*.2/1.4)
            yrange=float(np.ptp(r['body_x1'][mask]))
            result.update(density_ratio=rho,submerged_mean=float(np.nanmean(chi)),submerged_abs_error=float(abs(np.nanmean(chi)-rho)),
                submerged_min=float(np.nanmin(chi)),submerged_max=float(np.nanmax(chi)),center_y_mean=float(np.mean(r['body_x1'][mask])),
                center_y_range=yrange,velocity_rms_U=vrms,omega_rms_L_U=wrms,
                bottom_gap_min=float(np.min(r['bottom_gap'][mask])),world_contacts=int(r['body_world_contacts'][-1]),
                mechanical_steady_screen=bool(vrms<=.001 and wrms<=.001 and yrange<=.001 and np.min(r['bottom_gap'][mask])>=2/case['grid'] and r['body_world_contacts'][-1]==0 and np.max(abs(r['fluid_volume_error'][mask]))<=.01),
                strict_measurement_qualified=False)
    if case['experiment']=='B_slip':
        result.update(final_tangential_speed=float(r['fluid_velocity_y'][-1]),
            remaining_tangential_momentum_fraction=float(r['fluid_momentum_y'][-1]/(r['fluid_mass'][0]*.5)))
        if 'body_grid_impulse1' in r:
            p0=r['fluid_mass'][0]*.5
            result.update(wall_grid_impulse=float(r['body_grid_impulse1'][-1]),wall_contact_impulse=float(r['body_contact_impulse1'][-1]),
                fixed_body_max_displacement=float(np.max(np.linalg.norm(vector(r,'body_x')-vector(r,'body_x')[0],axis=1))),
                support_balance_relative=float(abs(r['fluid_momentum_y'][-1]-r['fluid_momentum_y'][0]+r['body_grid_impulse1'][-1]+r['body_contact_impulse1'][-1])/p0))
    if case['experiment']=='C_entry':
        q=vector(r,'body_q',4);angle=quat_distance(q,np.broadcast_to(q[0],q.shape))
        gap=np.array([r['body_x1'][i]-(np.abs(rotation_numpy(q[i]))@(np.array(case['geometry']['body_size'])/2))[1]-.6 for i in range(len(q))])
        touched=np.where(r['body_world_contacts']>0)[0]
        result.update(center_y_min=float(np.min(r['body_x1'])),center_y_final=float(r['body_x1'][-1]),
            center_y_min_time=float(r['time'][np.argmin(r['body_x1'])]),
            first_rebound_time=crossing(r['time'],r['body_v1']),
            first_below_initial_plane_time=crossing(r['time'],-gap),
            peak_linear_speed=float(np.max(np.linalg.norm(vector(r,'body_v'),axis=1))),
            max_rotation_from_initial_deg=float(np.max(angle)),first_world_contact_time=float(r['time'][touched[0]]) if len(touched) else None)
    return result
def savefig(fig,name,root):
    fig.tight_layout();fig.savefig(root/(name+'.png'),bbox_inches='tight');fig.savefig(root/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def plots(data,root):
    B=['B1_slip_control_g64','B1_slip_wall_g64','D_B_slip_g64_contact_off','D_B_slip_g32']
    if all(k in data for k in B):
        fig,ax=plt.subplots(1,2,figsize=(10,3.6))
        labels=['无板对照，64³','固定板，64³','关闭表面接触，64³','固定板，32³']
        for i,k in enumerate(B):
            r=data[k];ax[0].plot(r['time'],r['fluid_velocity_y'],color=COLORS[i],ls=['-','--',':','-.'][i],label=labels[i])
        ax[0].set(xlabel='时间 / s',ylabel='流体切向平均速度',ylim=(-.035,.535));ax[0].legend(fontsize=8)
        r=data[B[1]];p0=r['fluid_mass'][0]*.5
        ax[1].plot(r['time'],(p0-r['fluid_momentum_y'])/p0,label='流体切向动量损失',color=COLORS[0])
        ax[1].plot(r['time'],r['body_grid_impulse1']/p0,'--',label='板所受网格反馈冲量',color=COLORS[1])
        ax[1].plot(r['time'],r['body_contact_impulse1']/p0,':',label='表面接触冲量',color=COLORS[2])
        ax[1].set(xlabel='时间 / s',ylabel='冲量 / 初始切向动量');ax[1].legend(fontsize=8)
        for a in ax:a.grid(alpha=.2)
        savefig(fig,'fig4-2_slip',root)
    A=['A1_r0.25_g48','A1_r0.5_g48','A1_r0.75_g48','V2_A1_r0.5_g48_dt0.00025_n80']
    if all(k in data for k in A):
        fig,ax=plt.subplots(2,2,figsize=(10,6.3))
        for i,k in enumerate(A):
            r=data[k];label=['ρb/ρf=0.25','ρb/ρf=0.50','ρb/ρf=0.75','ρb/ρf=0.50，细步长 / 80 次'][i]
            style=['-','-.',':','--'][i]
            ax[0,0].plot(r['time'],r['body_x1'],color=COLORS[i],ls=style,label=label)
            ax[0,1].plot(r['time'],r['submerged_fraction'],color=COLORS[i],ls=style)
            ax[1,0].plot(r['time'],r['fluid_volume_error']*100,color=COLORS[i],ls=style)
            ax[1,1].plot(r['time'],r['body_v1'],color=COLORS[i],ls=style)
        for value in [.25,.5,.75]:ax[0,1].axhline(value,color='#aaaaaa',ls=':',lw=.8)
        ax[0,0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(.01,.90));ax[0,0].set(ylabel='刚体质心高度');ax[0,1].set(ylabel='测得的浸没体积比例')
        ax[1,0].set(ylabel='流体体积偏差 / %');ax[1,1].set(ylabel='刚体竖直速度')
        if 'A0_g48' in data:
            r=data['A0_g48'];ax[1,0].plot(r['time'],r['fluid_volume_error']*100,color='black',ls=':',label='纯流体对照');ax[1,0].legend(fontsize=8)
        for a in ax.flat:a.set_xlabel('时间 / s');a.grid(alpha=.2);a.axvspan(4.5,6,color='#cccccc',alpha=.15)
        savefig(fig,'fig4-1_float',root)
    C=['C_entry_a10_g48','T_C_entry_a10_g48_dt0.00025_n40','T_C_entry_a10_g48_dt0.00025_n80']
    if all(k in data for k in C):
        fig,ax=plt.subplots(2,2,figsize=(10,6.3))
        for i,k in enumerate(C):
            r=data[k];q=vector(r,'body_q',4)
            label=['Δt=0.0005，40 次','Δt=0.00025，40 次','Δt=0.00025，80 次'][i]
            for a,y in zip(ax.flat,[r['body_x1'],r['body_v1'],quat_distance(q,np.broadcast_to(q[0],q.shape)),r['bottom_gap']]):
                a.plot(r['time'],y,label=label,color=COLORS[i],ls=['-','--',':'][i])
        ballistic_t=np.linspace(0,np.sqrt(2*.03/9.8),80)
        ax[0,0].plot(ballistic_t,.7458455930679139-4.9*ballistic_t**2,color='#777777',ls='-.',lw=1,label='未耦合自由落体，理想接触前')
        ax[0,1].plot(ballistic_t,-9.8*ballistic_t,color='#777777',ls='-.',lw=1)
        for a,ylabel in zip(ax.flat,['刚体质心高度','刚体竖直速度','相对初始姿态的转角 / °','最低表面离底距离']):a.set(xlabel='时间 / s',ylabel=ylabel);a.grid(alpha=.2)
        ax[0,0].legend(fontsize=8);savefig(fig,'fig4-3_entry',root)
    if C[0] in data:
        out=DEFAULT_OUTPUT/'data'/C[0];initial=np.load(out/'initial.npz');idx=(initial['material']==0)&(abs(initial['x'][:,2]-.5)<1/96)
        fig,ax=plt.subplots(2,3,figsize=(10,6.8))
        for a,t in zip(ax.flat,[0,.1,.3,.6,1.,1.6]):
            fn=out/('initial.npz' if t==0 else f'frame_{round(t/.0005):06d}.npz')
            if not fn.exists():a.text(.5,.5,'未保存该时刻',ha='center');continue
            p=np.load(fn);x=p['x'][idx];a.scatter(x[:,0],x[:,1],s=.5,color='#307bad',alpha=.6,rasterized=True)
            vertices=p['body_x'][0]+(SIGNS*.1)@rotation_numpy(p['body_q'][0]).T
            polygon=vertices[ConvexHull(vertices[:,:2]).vertices,:2];a.add_patch(Polygon(polygon,color='#d88935',alpha=.75))
            a.plot([.125,.125,.875,.875],[.875,.125,.125,.875],color='#626262',lw=1)
            a.set(xlim=(.1,.9),ylim=(.1,.9),aspect='equal',title=f't = {t:.1f} s',xlabel='x',ylabel='y')
        savefig(fig,'fig4-4_entry_frames',root)
def main():
    p=json.loads((DEFAULT_OUTPUT/'planned_protocol.json').read_text());cases={c['case_id']:c for c in p['cases']}
    diag=DEFAULT_OUTPUT/'slip_diagnostic_protocol.json'
    if diag.exists():cases.update({c['case_id']:c for c in json.loads(diag.read_text())['cases']})
    data={};result={'cases':{},'comparisons':{},'reference_qualified':False}
    for name,case in cases.items():
        out=DEFAULT_OUTPUT/'data'/name
        if not (out/'run.json').exists():result['cases'][name]={'status':'not_started'};continue
        meta=json.loads((out/'run.json').read_text())
        if case['engine']=='SPlisHSPlasH':
            result['cases'][name]={'status':meta['status'],'exit_code':meta.get('exit_code'),'reason':(out/'failure.txt').read_text() if (out/'failure.txt').exists() else None};continue
        if not (out/'metrics.csv').exists():result['cases'][name]={'status':meta['status']};continue
        r=read_csv(out/'metrics.csv');data[name]=r;result['cases'][name]=summarize(r,case,meta)
    c='B1_slip_control_g64'
    if c in data:
        for name in ['B1_slip_wall_g64','D_B_slip_g64_contact_off']:
            if name in data:result['cases'][name]['extra_tangential_momentum_loss']=float(abs(data[name]['fluid_momentum_y'][-1]-data[c]['fluid_momentum_y'][-1])/(data[c]['fluid_mass'][0]*.5))
    pairs=[('C_entry_a10_g48','T_C_entry_a10_g48_dt0.00025_n40'),('T_C_entry_a10_g48_dt0.00025_n40','T_C_entry_a10_g48_dt0.00025_n80')]
    for a,b in pairs:
        if a in data and b in data:
            end=min(data[a]['time'][-1],data[b]['time'][-1]);comp=compare(data[a],data[b],end)
            contacts=[result['cases'][k]['first_world_contact_time'] for k in [a,b] if result['cases'][k].get('first_world_contact_time') is not None]
            comp['before_wall']=compare(data[a],data[b],min(contacts)-.005) if contacts else comp.copy()
            result['comparisons'][a+'__'+b]=comp
    (DEFAULT_OUTPUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    completed={k:r for k,r in data.items() if result['cases'][k].get('full_duration')}
    plots(completed,DEFAULT_OUTPUT)
    print(json.dumps({k:{q:v for q,v in r.items() if q in ['status','last_time','submerged_mean','submerged_abs_error','mechanical_steady_screen','extra_tangential_momentum_loss']} for k,r in result['cases'].items()},indent=2))
if __name__=='__main__':main()
