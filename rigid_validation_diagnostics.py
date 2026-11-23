"""Run only the two slip diagnostics triggered by the registered 1% criterion."""
import copy
import json
from rigid_validation_run import DEFAULT_OUTPUT,run

def main():
    p=json.loads((DEFAULT_OUTPUT/'planned_protocol.json').read_text())
    base=next(c for c in p['cases'] if c['case_id']=='B1_slip_wall_g64')
    contact=copy.deepcopy(base);contact.update(case_id='D_B_slip_g64_contact_off',rigid_contact=False)
    coarse=copy.deepcopy(base);coarse.update(case_id='D_B_slip_g32',grid=32,grid_boundary_index=4,fluid_target_spacing=1/64)
    amendment={'id':'D_slip_20261028','trigger':'B1 measured extra tangential momentum loss 100.0128%, above 1% criterion',
               'purpose':'locate whether tangential damping persists without surface contact and at a second grid',
               'cases':[contact,coarse],'status':'registered_before_diagnostic_runs'}
    path=DEFAULT_OUTPUT/'slip_diagnostic_protocol.json'
    if not path.exists():path.write_text(json.dumps(amendment,indent=2))
    for c in [contact,coarse]:run(c,DEFAULT_OUTPUT)
if __name__=='__main__':main()
