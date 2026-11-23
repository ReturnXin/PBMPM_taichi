"""Non-scientific backend feasibility/timing probe; never joins formal data."""
import json
from pathlib import Path
import shutil
import taichi as ti
import argparse
from rigid_validation_run import DEFAULT_OUTPUT,run

parser=argparse.ArgumentParser();parser.add_argument('--case',default='A0_g48');parser.add_argument('--duration',type=float,default=.01);parser.add_argument('--tag',default='vulkan_probe');args=parser.parse_args()
root=DEFAULT_OUTPUT/'dispatch_benchmarks'/args.tag
root.mkdir(parents=True,exist_ok=True)
if not (root/'planned_protocol.json').exists():shutil.copy2(DEFAULT_OUTPUT/'planned_protocol.json',root/'planned_protocol.json')
original_init=ti.init
def initialize(*a,**kw):
    kw['arch']=ti.vulkan
    return original_init(*a,**kw)
ti.init=initialize
protocol=json.loads((root/'planned_protocol.json').read_text())
case=next(c for c in protocol['cases'] if c['case_id']==args.case)
try:run(case,root,pilot=args.duration)
finally:
    p=root/'pilot'/args.case/'run.json'
    if p.exists():
        m=json.loads(p.read_text());m.update(backend_probe='vulkan',included_in_scientific_results=False);p.write_text(json.dumps(m,indent=2))
