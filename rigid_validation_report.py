"""Render the report only after all registered local trajectories finish."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from rigid_validation_run import DEFAULT_OUTPUT,WORKSPACE

def marker(claim,evidence):return f'<!-- [claim:{claim}] [evidence:{evidence}] -->'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write-chapter',action='store_true');args=ap.parse_args()
    root=DEFAULT_OUTPUT;protocol=json.loads((root/'planned_protocol.json').read_text());analysis=json.loads((root/'analysis.json').read_text());s=analysis['cases']
    local=[c['case_id'] for c in protocol['cases'] if c['engine']=='PBMPM']
    incomplete=[k for k in local if s[k].get('status')!='completed' or not s[k].get('full_duration')]
    if incomplete:raise RuntimeError('Waiting for complete trajectories: '+', '.join(incomplete))
    A=['A1_r0.25_g48','A1_r0.5_g48','A1_r0.75_g48','V2_A1_r0.5_g48_dt0.00025_n80']
    a=[s[k] for k in A];a0=s['A0_g48'];c=s['C_entry_a10_g48']
    template=(root/'report_template.md').read_text(encoding='utf-8')
    floating=('从理论平衡位置释放后，三组浮体在评价窗口内均保持部分浸没，密度越大，浸在水下的体积越多。'
        f"以密度比 0.5 为例，理论上应有一半体积浸在水下，实际测得 {a[1]['submerged_mean']*100:.2f}%，比理论值高 {a[1]['submerged_abs_error']*100:.2f} 个百分点。"
        f"作为控制，纯流体组的末时刻体积偏差为 {a0['fluid_volume_error_final']*100:.4f}%，评价窗口内水面平均高度为 {a0['surface_mean']:.6f}，接近初始高度 0.6。"
        '图 4-1 展示完整过程，表 4-2 汇总固定末段窗口的测量结果。'+marker('C490','E480,E482,E484'))
    table='| 密度比 / 配置 | 理论浸没体积 / % | 测得浸没体积 / % | 绝对差 / 百分点 | 平均质心高度 | 达到静止判据 |\n| --- | ---: | ---: | ---: | ---: | --- |\n'
    for case_id,label,row in zip(A,['0.25','0.50','0.75','0.50，半步长 / 80 次'],a):
        theory=next(p['geometry']['density_ratio'] for p in protocol['cases'] if p['case_id']==case_id)
        table+=f"| {label} | {theory*100:.2f} | {row['submerged_mean']*100:.2f} | {row['submerged_abs_error']*100:.2f} | {row['center_y_mean']:.5f} | {'是' if row['mechanical_steady_screen'] else '否'} |\n"
    table+='\n注：浸没体积以刚体总体积的百分比表示；例如 51.51% 与 50% 相差 1.51 个百分点。“静止判据”指 4.2 节的基本静态条件，包含残余运动、体积变化和外部支撑检查；水面测量精度另行评价。'+marker('C492','E480,E482,E483,E484')
    diff=abs(a[1]['submerged_mean']-a[3]['submerged_mean'])
    over=sum(row['submerged_abs_error']>.02 for row in a[:3])
    steady=sum(row['mechanical_steady_screen'] for row in a)
    failed_conditions=[]
    if any(row['velocity_rms_U']>.001 for row in a):failed_conditions.append('残余线速度')
    if any(row['omega_rms_L_U']>.001 for row in a):failed_conditions.append('残余角速度')
    if any(row['center_y_range']>.001 for row in a):failed_conditions.append('质心高度变化')
    if any(row['world_contacts']>0 or row['bottom_gap_min']<2/48 for row in a):failed_conditions.append('池壁支撑或离底距离')
    steady_text=('四个配置均未满足基本静态条件。' if steady==0 else '四个配置均满足基本静态条件。' if steady==4 else f'四个配置中，有 {steady} 个满足基本静态条件。')
    if steady==1 and a[3]['mechanical_steady_screen']:steady_text='仅半步长、80 次迭代的密度比 0.5 配置满足基本静态条件。'
    if failed_conditions:steady_text+='其余配置的未达标项涉及'+'或'.join(failed_conditions)+'，对应末段均值描述的是仍有运动的状态。'
    over_labels='、'.join(label for label,row in zip(['0.25','0.50','0.75'],a[:3]) if row['submerged_abs_error']>.02)
    measured_text=('三种密度比的基准结果与理论值均相差不超过 2 个百分点。' if over==0 else f'密度比 {over_labels} 的基准结果超过了原定 2 个百分点的误差目标。')
    sensitivity_note=('这一变化超过原定 0.5 个百分点的检查触发值。' if diff>.005 else '')
    discussion=(''+steady_text+measured_text+
        '\n\n增加计算量后，浮体更接近静止，但浸没量未必更接近理论值。'
        f"密度比 0.5 采用半步长和 80 次迭代后，测得浸没比例为 {a[3]['submerged_mean']*100:.2f}%，较基准配置改变 {diff*100:.2f} 个百分点。"
        '由于步长和迭代次数同时改变，无法分别判断两者的影响。'+sensitivity_note+
        '\n\n此外，水面位置的测量也会影响浸没比例。校准中，仅改变水平面相对网格的位置，就产生了约 1.05 个百分点的浸没比例差，高于预设的 0.5 个百分点测量目标。'
        '水面重建误差限制了浸没比例的测量精度。'+marker('C493','E480,E482,E483,E484,E490'))
    entry=(f"刚体释放后先下沉，随后出现回升，同时发生转动。基准配置中，质心从高度 0.745846 下降，在 {c['first_rebound_time']:.4f} s 首次由下降转为回升，"
        f"随后仍有缓慢下移；1.6 s 时的质心高度为 {c['center_y_final']:.5f}。"
        f"全时段最大线速度为 {c['peak_linear_speed']:.5f}，相对初始姿态的最大转角为 {c['max_rotation_from_initial_deg']:.3f}°。"
        '三种配置均未发生刚体触壁。图 4-3 展示了下沉、回升及姿态变化，图 4-4 进一步比较这些运动对步长和迭代次数的依赖。'+marker('C498','E482,E484'))
    keys=['C_entry_a10_g48__T_C_entry_a10_g48_dt0.00025_n40','T_C_entry_a10_g48_dt0.00025_n40__T_C_entry_a10_g48_dt0.00025_n80']
    table_c='| 对照 | 位置差 / 体长（%） | 速度差 / 参考速度（%） | 姿态差 / ° |\n| --- | ---: | ---: | ---: |\n'
    for label,key in zip(['40 次迭代，步长减半','半步长下，40 → 80 次迭代'],keys):
        d=analysis['comparisons'][key];table_c+=f"| {label} | {d['position_rms_L']*100:.2f} | {d['velocity_rms_U']*100:.2f} | {d['orientation_rms_deg']:.4f} |\n"
    table_c+='\n注：各列均为 0.005–1.6 s 内同一时刻两条轨迹的均方根差，用于概括整段差异。体长为 $L=0.2$，参考速度为 $U=1.4$。位置差 5.38% 表示均方根差相当于体长的 5.38%。'+marker('C501','E482,E484')
    failures=[]
    for label,key in zip(['步长抽查','迭代抽查'],keys):
        d=analysis['comparisons'][key]
        dimensions=[name for name,field,limit in [('位置','position_rms_L',.01),('速度','velocity_rms_U',.01),('姿态','orientation_rms_deg',.5)] if d[field]>limit]
        if dimensions:failures.append(label+'中的'+'、'.join(dimensions)+'差超过预设目标')
    cd='表 4-3 表明，改变步长或迭代次数后，刚体的位置曲线仍有相当于体长约 5%–7% 的差异。'
    cd+=('；'.join(failures)+'。' if failures else '两次抽查均满足预设的三个敏感性目标。')
    cd+='其中迭代抽查的姿态差为 0.5017°，略高于 0.5° 的目标。以上差异反映了入水运动对数值配置的敏感性。'+marker('C508','E484,E486,E487,E490')
    conclusion=('三个主实验表明，共享网格配置能够产生部分浸没、入水回升和转动，但同时存在切向粘连。'
        '浮体的残余运动和入水曲线的配置差异说明响应仍受数值配置影响。'
        '静态结果还受水面重建误差影响，动态结果缺少有效的独立参考；实时性能尚未评测。'+marker('C504','E482,E483,E484,E487,E490,E491,E493'))
    replacement={'FLOAT_RESULTS':floating,'FLOAT_TABLE':table,'FLOAT_DISCUSSION':discussion,'ENTRY_RESULTS':entry,'ENTRY_TABLE':table_c,
        'ENTRY_DISCUSSION':cd,'CONCLUSION':conclusion,'LOCAL_RUN_COUNT':str(len(local)),
        'CHAPTER_END':'结果呈现了密度影响浸没程度、入水引起平移与转动的耦合行为，也揭示了共享网格界面的切向粘连及作用范围限制。动态精度和实时性能仍需补充评价。'}
    for key,value in replacement.items():template=template.replace('{{'+key+'}}',value)
    assert '{{' not in template
    standalone=template.replace('(assets/ch04-rigid-validation-20261028/','(')
    standalone=standalone.replace('(刚体耦合粘连问题排查.md)','(../../刚体耦合粘连问题排查.md)')
    (root/'experiment_report.md').write_text(standalone,encoding='utf-8')
    chapter=WORKSPACE/'毕业论文/论文正文/Ch04.刚体材质统一耦合方法.md'
    current=chapter.read_text(encoding='utf-8')
    parts=re.split(r'^## [34] 实验结果分析\s*$',current,maxsplit=1,flags=re.M)
    if len(parts)!=2:raise RuntimeError('Cannot locate experiment section; refusing to overwrite chapter')
    new=parts[0].rstrip()+'\n\n'+template
    if args.write_chapter:chapter.write_text(new,encoding='utf-8')
    evidence=WORKSPACE/'毕业论文/论文正文/.evidence/ch04-validation-20261028';evidence.mkdir(exist_ok=True)
    sources=[('E480','registry','预先保存的三个实验协议','../../assets/ch04-rigid-validation-20261028/planned_protocol.json'),
        ('E481','software','本次求解器、固定体、反力与观测实现','../../assets/ch04-rigid-validation-20261028/executed_source'),
        ('E482','dataset','本方法原始轨迹与快照','../../assets/ch04-rigid-validation-20261028/data'),
        ('E483','dataset','水面、几何交体积和固定壁面校准','../../assets/ch04-rigid-validation-20261028/measurement_calibration.json'),
        ('E484','dataset','完整轨迹统计与曲线差','../../assets/ch04-rigid-validation-20261028/analysis.json'),
        ('E485','dataset','三个实验的图表与绘图程序','../../assets/ch04-rigid-validation-20261028'),
        ('E486','registry','滑移失败触发的局部诊断协议','../../assets/ch04-rigid-validation-20261028/slip_diagnostic_protocol.json'),
        ('E487','software','外部参考构建、适配与失败记录','../../assets/ch04-rigid-validation-20261028/reference_adaptation_record.json'),
        ('E488','report','SIGGRAPH 原文定位及文献对照','../../刚体耦合实验文献对照与修订方案.md'),
        ('E489','dataset','既有自由运动、对称性与间隙扫描','../../assets/ch04-rigid-coupling-20261026'),
        ('E490','registry','解释范围修订','../../assets/ch04-rigid-validation-20261028/interpretation_amendments.json'),
        ('E491','dataset','入水早期的实际粒子与刚体间隙检查','../../assets/ch04-rigid-validation-20261028/entry_separation_probe/analysis.json'),
        ('E492','dataset','CUDA 与 Vulkan 完整入水轨迹一致性核查','../../assets/ch04-rigid-validation-20261028/backend_validation_result.json'),
        ('E494','report','刚体耦合实时评价实验设计：待实施方案','../../刚体耦合实时评价实验设计.md'),
        ('E495','report','可读性修订前的完整数值表与诊断说明','../../assets/ch04-rigid-validation-20261028/result_details.md')]
    manifest={'schema_version':'1.0','sources':[{'evidence_id':eid,'source_type':kind,'title':title,'authors':[],'year':2026,
        'identifiers':dict(doi='',pmid='',pmcid='',isbn='',url=''),'locator':loc,'confidentiality':'restricted',
        'verification':{'status':'unverified','source_opened':False,'verified_by':'','verified_on':''}} for eid,kind,title,loc in sources]}
    manifest['sources'].append({'evidence_id':'E493','source_type':'conference_paper',
        'title':'A Position Based Material Point Method','authors':['Chris Lewin'],'year':2024,
        'identifiers':dict(doi='10.1145/3641233.3664323',pmid='',pmcid='',isbn='',url='https://doi.org/10.1145/3641233.3664323'),
        'locator':'论文/Lewin - 2024 - A Position Based Material Point Method.pdf; SIGGRAPH Talks 2024, pp. 1–2, §1, §2.1, §3; local extraction and agent reading record: ../ch04-pbmpm-scope-20261028/',
        'confidentiality':'public','verification':{'status':'unverified','source_opened':False,'verified_by':'','verified_on':''}})
    interface_record=evidence.parent/'ch04-interface-support-20261028/source_manifest.json'
    interface_sources=json.loads(interface_record.read_text(encoding='utf-8'))['sources']
    for source in interface_sources:
        item=dict(source)
        item['locator']='../ch04-interface-support-20261028/'+source['locator']
        manifest['sources'].append(item)
    (evidence/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    claims=[];section='';last=0
    for match in re.finditer(r'<!-- \[claim:(C\d+)\] \[evidence:([^]]+)\] -->',template):
        text=template[last:match.start()].strip();headings=re.findall(r'^#{2,3} (.+)$',text,re.M)
        if headings:section=headings[-1]
        claims.append({'claim_id':match[1],'section':section,'claim_kind':'method' if match[1] in {'C481','C482','C483','C484','C485','C486','C487','C489'} else 'interpretive' if match[1] in {'C504','C507','C577','C578','C579'} else 'result','claim_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'evidence_ids':match[2].replace(',',';'),'verification_status':'unverified','uncertainty':'not_estimated','analysis_intent':'descriptive'})
        last=match.end()
    with (evidence/'claims.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(claims[0]));w.writeheader();w.writerows(claims)
    audit={'local_planned_completed':len(local),'diagnostic_completed':sum(s[k]['status']=='completed' for k in s if k.startswith('D_B')),
        'reference_failed':sum(s[c['case_id']]['status']=='failed' for c in protocol['cases'] if c['engine']=='SPlisHSPlasH'),
        'unresolved_template_tokens':False,'human_scientific_verification':False,'chapter_written':args.write_chapter,
        'figures_exist':all((root/(n+'.png')).exists() for n in ['fig4-1_float','fig4-2_slip','fig4-3_entry','fig4-4_entry_frames']),
        'contains_explicit_third_chapter_mention':'第三章' in new}
    assert audit['figures_exist'];assert not audit['contains_explicit_third_chapter_mention']
    (evidence/'report_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__),root/'executed_source'/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('rigid_validation_analyze.py'),root/'executed_source/rigid_validation_analyze.py')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
