"""Finish local analysis after the currently running registered jobs complete."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]/'毕业论文/论文正文/assets/ch04-rigid-validation-20261028'
def main():
    protocol=json.loads((ROOT/'planned_protocol.json').read_text());cases=[c for c in protocol['cases'] if c['engine']=='PBMPM']
    while True:
        state={}
        for c in cases:
            path=ROOT/'data'/c['case_id']/'run.json'
            try:
                state[c['case_id']]=json.loads(path.read_text())['status'] if path.exists() else 'not_started'
            except (json.JSONDecodeError,PermissionError):
                state[c['case_id']]='writing_metadata'
        completed=sum(v=='completed' for v in state.values())
        if any(v in ['failed','invalid_initialization'] for v in state.values()):
            (ROOT/'finish_status.json').write_text(json.dumps({'status':'needs_attention','cases':state},indent=2));return 2
        (ROOT/'finish_status.json').write_text(json.dumps({'status':'waiting','completed_local':completed,'required_local':len(cases),'cases':state},indent=2))
        if completed==len(cases):break
        print(f'Complete local trajectories: {completed}/{len(cases)}',flush=True)
        time.sleep(30)
    for script,args in [('rigid_validation_analyze.py',[]),('rigid_validation_report.py',['--write-chapter']),('rigid_validation_audit.py',[])]:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*args])
        if result.returncode:
            (ROOT/'finish_status.json').write_text(json.dumps({'status':'analysis_or_report_failed','script':script,'code':result.returncode},indent=2));return result.returncode
    (ROOT/'finish_status.json').write_text(json.dumps({'status':'report_written_pending_review','completed_local':len(cases),'manual_figure_and_text_review_required':True},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
