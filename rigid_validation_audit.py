"""Cross-check raw run records, source versions, totals and figure availability."""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from rigid_validation_run import DEFAULT_OUTPUT,WORKSPACE

def consistency_record(root,evidence):
    analysis=json.loads((root/'analysis.json').read_text());facts=[]
    def fact(concept,value,unit,case,section='4.3 对比结果',eid='E484'):
        facts.append(dict(fact_id=f'N{4800+len(facts)}',concept=concept,section=section,
            value=value,unit=unit,numerator=None,denominator=None,sample_size=1,
            analysis_set=case,evidence_ids=[eid]))
    for name,row in analysis['cases'].items():
        if row['status']!='completed':continue
        for key,unit in [('center_y_mean','计算长度单位'),('submerged_mean','无量纲'),
                         ('submerged_abs_error','无量纲'),('velocity_rms_U','无量纲'),
                         ('omega_rms_L_U','无量纲'),('fluid_volume_error_final','相对偏差'),
                         ('extra_tangential_momentum_loss','相对初始切向动量'),
                         ('final_tangential_speed','计算长度单位/s'),('wall_grid_impulse','计算质量·长度单位/s'),
                         ('wall_contact_impulse','计算质量·长度单位/s'),('support_balance_relative','相对初始切向动量'),
                         ('center_y_final','计算长度单位'),('first_rebound_time','s'),
                         ('peak_linear_speed','计算长度单位/s'),('max_rotation_from_initial_deg','degree')]:
            if key in row and row[key] is not None:fact(key,row[key],unit,name)
    for pair,row in analysis['comparisons'].items():
        for key,unit in [('position_rms_L','无量纲'),('velocity_rms_U','无量纲'),('orientation_rms_deg','degree')]:
            fact(key,row[key],unit,pair)
    for name,row in analysis['cases'].items():
        if row['status']=='completed' and 'submerged_mean' in row:
            fact('submerged_mean_percent',100*row['submerged_mean'],'%',name)
            fact('submerged_abs_error_percentage_points',100*row['submerged_abs_error'],'百分点',name)
    for pair,row in analysis['comparisons'].items():
        fact('position_rms_percent_L',100*row['position_rms_L'],'% of L',pair)
        fact('velocity_rms_percent_U',100*row['velocity_rms_U'],'% of U',pair)
    methods=[
        dict(method_id='M480',name='静水浮体：固定末段窗口的四个配置描述',analysis_intent='descriptive',protocol_status='prespecified',outcome_ids=['O480','O481']),
        dict(method_id='M481',name='切向滑移：有板与无板比较',analysis_intent='descriptive',protocol_status='prespecified',outcome_ids=['O482']),
        dict(method_id='M482',name='倾斜入水：三个配置的共同时间比较',analysis_intent='descriptive',protocol_status='prespecified',outcome_ids=['O483']),
        dict(method_id='M483',name='入水早期间隙重放诊断',analysis_intent='exploratory',protocol_status='post_hoc',outcome_ids=['O484']),
        dict(method_id='M484',name='CUDA 与 Vulkan 的相同配置核查',analysis_intent='descriptive',protocol_status='amended_before_analysis',outcome_ids=['O485'])]
    results=[]
    for method,count,eids in zip(methods,[4,2,3,1,2],[['E482','E483','E484'],['E482','E484'],['E482','E484'],['E491'],['E492']]):
        for outcome in method['outcome_ids']:
            results.append(dict(result_id=f'R{480+len(results)}',method_id=method['method_id'],outcome_id=outcome,
                analysis_intent=method['analysis_intent'],sample_size=count,evidence_ids=eids,
                reported_sections=['4.1 实验设置' if method['method_id']=='M484' else '4.3 对比结果']))
    (evidence/'consistency_manifest.json').write_text(json.dumps(dict(schema_version='1.0',numeric_facts=facts,methods=methods,results=results),ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    root=DEFAULT_OUTPUT;protocol=json.loads((root/'planned_protocol.json').read_text())
    cases=protocol['cases']+json.loads((root/'slip_diagnostic_protocol.json').read_text())['cases']
    manifest=[];errors=[];sources={}
    for case in cases:
        name=case['case_id'];directory=root/'data'/name;meta=json.loads((directory/'run.json').read_text())
        record={'case_id':name,'experiment':case['experiment'],'engine':case['engine'],'status':meta['status'],
                'planned_duration':case['duration'],'actual_duration':meta.get('actual_duration'),
                'backend':meta.get('backend','cuda' if case['engine']=='PBMPM' else 'cpu'),
                'runner_sha256':meta.get('runner_sha256'),'directory':'data/'+name}
        if case['engine']=='PBMPM':
            rows=list(csv.DictReader((directory/'metrics.csv').open(encoding='utf-8-sig')))
            target_count=round(case['duration']/case['sample_interval'])+1
            t=np.array([float(r['time']) for r in rows]);record['samples']=len(t)
            if meta['status']!='completed' or len(t)!=target_count or abs(t[-1]-case['duration'])>1e-9:errors.append(name+': incomplete trajectory')
            if not np.allclose(np.diff(t),case['sample_interval'],rtol=0,atol=1e-9):errors.append(name+': unexpected sample times')
            expected_volume=case['geometry'].get('water_volume',np.prod(case['geometry'].get('fluid_size',[0])))
            actual_mass=float(rows[0]['fluid_mass']);relative=abs(actual_mass-expected_volume)/expected_volume
            record.update(expected_fluid_mass=float(expected_volume),actual_fluid_mass=actual_mass,initial_fluid_mass_relative_error=relative)
            if relative>1e-6:errors.append(name+': mismatched water mass')
            if abs(float(rows[-1]['fluid_mass'])-actual_mass)>1e-10:errors.append(name+': changed fluid mass')
            if meta['config']['buoyancy_assist'] or meta['config']['viscosity']!=0:errors.append(name+': forbidden helper enabled')
            for config_key,protocol_key in [('grid','grid'),('dt','dt'),('iterations','iterations'),('gravity','gravity_y'),('boundary','grid_boundary_index')]:
                if meta['config'][config_key]!=case[protocol_key]:errors.append(name+': config mismatch '+config_key)
            if not all(np.isfinite(float(value)) for row in rows for value in row.values() if value):errors.append(name+': non-finite recorded value')
            for file,digest in meta['source_hashes'].items():sources.setdefault(file,set()).add(digest)
            record['csv_sha256']=hashlib.sha256((directory/'metrics.csv').read_bytes()).hexdigest()
        else:
            record['failure_record']=(directory/'failure.txt').read_text() if (directory/'failure.txt').exists() else None
            record['qualified_reference']=False
        manifest.append(record)
    if any(len(v)>1 for v in sources.values()):errors.append('Core solver source changed between runs')
    for file,hashes in sources.items():
        archive=root/'executed_source/rigid_coupling'/file
        if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() not in hashes:errors.append('Missing or inconsistent source archive: '+file)
    figures=['fig4-1_float','fig4-2_slip','fig4-3_entry','fig4-4_entry_frames']
    if any(not (root/(name+ext)).exists() for name in figures for ext in ['.png','.pdf']):errors.append('Missing figure export')
    chapter=WORKSPACE/'毕业论文/论文正文/Ch04.刚体材质统一耦合方法.md'
    text=chapter.read_text(encoding='utf-8')
    if '{{' in text:errors.append('Unresolved template placeholders')
    if '第三章' in text:errors.append('Unwanted third-chapter wording')
    if any(ord(c)<32 and c not in '\n\r' for c in text):errors.append('Unexpected control character in manuscript')
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        if '://' not in link and not link.startswith('#') and not (chapter.parent/link).exists():errors.append('Broken manuscript link: '+link)
    evidence=WORKSPACE/'毕业论文/论文正文/.evidence/ch04-validation-20261028'
    consistency_record(root,evidence)
    (root/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'status':'passed' if not errors else 'failed','errors':errors,'core_solver_hashes':{k:sorted(v) for k,v in sources.items()},
            'valid_main_local_runs':sum(c['engine']=='PBMPM' for c in protocol['cases']),
            'full_slip_diagnostics':2,'backend_full_replays':1,'independent_reference_qualified':False,
            'human_scientific_verification':False}
    (evidence/'numeric_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
