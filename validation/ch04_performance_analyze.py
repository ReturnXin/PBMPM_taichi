"""Audit saved performance batches and generate chapter tables and plots."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(out):
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    results=[];parity=[]
    for resolution in protocol['resolutions']:
        folder=out/f'resolution_{resolution}'
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        checks=json.loads((folder/'parity.json').read_text(encoding='utf-8'))
        assert summary['completed'] and summary['source_unchanged'] and checks['passed']
        with (folder/'timings.csv').open(encoding='utf-8') as file:
            rows=list(csv.DictReader(file))
        assert len(rows)==28
        samples={}
        for count in [0,3]:
            for mode in ['ordinary','graph']:
                selected=sorted([r for r in rows if int(r['bodies'])==count and r['dispatch']==mode],key=lambda r:int(r['batch']))
                assert [int(r['batch']) for r in selected]==list(range(7))
                assert all(int(r['steps'])==80 for r in selected)
                samples[f'bodies{count}_{mode}']=np.array([float(r['ms_per_step']) for r in selected])
        row=dict(resolution=resolution,fluid_particles=protocol['counts'][str(resolution)]['fluid'],
                 total_particles=protocol['counts'][str(resolution)]['total_with_bodies'])
        for key,values in samples.items():
            assert np.isfinite(values).all() and (values>0).all()
            row[key]=dict(median_ms=float(np.median(values)),min_ms=float(values.min()),max_ms=float(values.max()),samples_ms=values.tolist())
            assert abs(row[key]['median_ms']-summary['summary'][key]['median_ms'])<1e-10
        ordinary,graph=samples['bodies3_ordinary'],samples['bodies3_graph']
        control=samples['bodies0_graph']
        row['graph_speedup_ratio_of_medians']=float(np.median(ordinary)/np.median(graph))
        row['graph_time_reduction_percent']=float((1-np.median(graph)/np.median(ordinary))*100)
        row['paired_graph_speedup_median']=float(np.median(ordinary/graph))
        row['added_bodies_overhead_percent']=float((np.median(graph)/np.median(control)-1)*100)
        row['added_bodies_ms']=float(np.median(graph)-np.median(control))
        row['graph_physical_steps_per_second']=float(1000/np.median(graph))
        results.append(row)
        parity.append(dict(resolution=resolution,**checks))
    ratio_particles=results[-1]['fluid_particles']/results[0]['fluid_particles']
    ratio_time=results[-1]['bodies3_graph']['median_ms']/results[0]['bodies3_graph']['median_ms']
    max_errors={}
    for item in parity:
        for errors in item['max_absolute_errors'].values():
            for key,value in errors.items():
                max_errors[key]=max(max_errors.get(key,0.),value)
    analysis=dict(passed=True,resolutions=results,parity=parity,max_parity_errors=max_errors,
                  particle_count_ratio=ratio_particles,graph_time_ratio=ratio_time,
                  repetitions=7,timed_steps_per_batch=80,total_timed_batches=84,
                  total_timed_steps=6720,human_scientific_review='pending',
                  main_speedup_definition='ratio of ordinary median to graph median; paired ratios also retained',
                  source_snapshot_matches=all(digest(out/'source_snapshot/rigid_coupling'/key)==value for key,value in protocol['production_hashes'].items()))
    assert analysis['source_snapshot_matches']
    write_json(out/'analysis.json',analysis)
    return analysis


def figures(out,analysis):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none'})
    target=out/'figures';target.mkdir(exist_ok=True)
    results=analysis['resolutions'];x=np.arange(len(results))
    ticks=[f"{r['fluid_particles']:,}" for r in results]
    def save(fig,name):
        for extension in ['png','pdf','svg']:
            fig.savefig(target/f'{name}.{extension}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    variants=[('bodies0_graph','无刚体，CUDA 图','#777777',-.23),
              ('bodies3_graph','三刚体，CUDA 图','#1268b5',0),
              ('bodies3_ordinary','三刚体，普通执行','#d97732',.23)]
    for key,label,color,offset in variants:
        med=np.array([r[key]['median_ms'] for r in results])
        low=np.array([r[key]['min_ms'] for r in results]);high=np.array([r[key]['max_ms'] for r in results])
        axs[0].bar(x+offset,med,width=.21,color=color,label=label,alpha=.88)
        axs[0].errorbar(x+offset,med,yerr=[med-low,high-med],fmt='none',ecolor='#333333',capsize=3,lw=1)
    axs[0].set_title('整步求解耗时',loc='left',fontweight='bold')
    axs[0].set_ylabel('耗时（ms/物理步）');axs[0].set_xticks(x,ticks)
    axs[0].set_xlabel('水粒子数');axs[0].grid(axis='y',alpha=.2);axs[0].set_axisbelow(True)
    axs[0].legend(frameon=False,fontsize=8.5)
    for i,r in enumerate(results):
        for key,offset,marker,color in [('bodies3_ordinary',-.10,'s','#d97732'),('bodies3_graph',.10,'o','#1268b5')]:
            jitter=np.linspace(-.045,.045,7)
            axs[1].scatter(i+offset+jitter,r[key]['samples_ms'],s=26,marker=marker,facecolors='none',edgecolors=color,
                           label=('普通执行' if key.endswith('ordinary') else 'CUDA 图') if i==0 else None)
            axs[1].plot([i+offset-.08,i+offset+.08],[r[key]['median_ms']]*2,color=color,lw=2)
    axs[1].set_title('三刚体场景的重复计时',loc='left',fontweight='bold')
    axs[1].set_ylabel('耗时（ms/物理步）');axs[1].set_xlabel('水粒子数');axs[1].set_xticks(x,ticks)
    axs[1].grid(axis='y',alpha=.2);axs[1].legend(frameon=False,fontsize=9)
    save(fig,'performance_scaling')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    values=[r['graph_speedup_ratio_of_medians'] for r in results]
    bars=axs[0].bar(x,values,color='#1268b5',width=.5)
    axs[0].axhline(1,color='#888888',ls='--',lw=1)
    axs[0].bar_label(bars,labels=[f'{v:.2f}×' for v in values],padding=4)
    axs[0].set_ylim(0,max(values)*1.22);axs[0].set_ylabel('普通执行耗时 / 图调度耗时')
    axs[0].set_title('CUDA 图调度加速比',loc='left',fontweight='bold')
    values=[r['added_bodies_overhead_percent'] for r in results]
    bars=axs[1].bar(x,values,color='#d97732',width=.5)
    axs[1].bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4)
    axs[1].set_ylim(min(0,min(values)*1.25),max(values)*1.22)
    axs[1].set_ylabel('加入三刚体后的耗时增幅（%）')
    axs[1].set_title('刚体耦合的整步开销',loc='left',fontweight='bold')
    for ax in axs:
        ax.set_xticks(x,ticks);ax.set_xlabel('水粒子数');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,'performance_ratios')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analysis=analyze(args.output)
    figures(args.output,analysis)
    shutil.copy2(__file__,args.output/'source_snapshot/validation'/Path(__file__).name)
    print(json.dumps(analysis,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
