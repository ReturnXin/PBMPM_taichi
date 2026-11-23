"""Independent measurement and completed fixed-boundary sanity checks."""
import csv
import json
import numpy as np
from rigid_validation_run import DEFAULT_OUTPUT,clipped_box_fraction
from rigid_coupling.rigid import rotation_numpy

def column(level,grid):
    n=int(np.ceil((level-.125)*grid*2-1e-10));dy=(level-.125)/n
    y=.125+(np.arange(n)+.5)*dy
    p=y*grid;base=np.floor(p-.5).astype(int);f=p-base
    w=np.array([.5*(1.5-f)**2,.75-(f-1)**2,.5*(f-.5)**2])
    phi=np.zeros(grid)
    for k in range(3):np.add.at(phi,base+k,w[k]*dy*grid)
    j=np.where((phi[:-1]>=.5)&(phi[1:]<.5))[0][-1]
    return (j+(phi[j]-.5)/(phi[j]-phi[j+1]))/grid

def main():
    clips=[]
    for deg in [0,10,35,73]:
        angle=np.deg2rad(deg)/2;R=rotation_numpy([np.cos(angle),0,0,np.sin(angle)])
        got=clipped_box_fraction([.5,.6,.5],[.3,.2,.3],R,[0,0,.6]);clips.append(abs(got-.5))
    for target in [0,.1,.25,.5,.75,.9,1]:
        got=clipped_box_fraction([.5,.6,.5],[.3,.2,.3],np.eye(3),[0,0,.5+target*.2]);clips.append(abs(got-target))
    phases={}
    for grid in [48,64]:
        bias=np.array([column(y,grid)-y for y in np.linspace(.57,.65,161)])
        phases[str(grid)]={'max_absolute_height_bias':float(np.max(abs(bias))),
            'max_between_phase_height_difference':float(np.ptp(bias)),
            'max_between_phase_fraction_difference_L0.2':float(np.ptp(bias)/.2)}
    control=list(csv.DictReader((DEFAULT_OUTPUT/'data/B1_slip_control_g64/metrics.csv').open()))
    wall=list(csv.DictReader((DEFAULT_OUTPUT/'data/B1_slip_wall_g64/metrics.csv').open()))
    W={k:np.array([float(r[k]) for r in wall]) for k in wall[0]}
    fixed_errors={name:float(np.max(np.abs(W[name]-W[name][0]))) for name in W if name.startswith(('body_x','body_v','body_omega','body_q'))}
    c=float(control[-1]['fluid_momentum_y']);w=W['fluid_momentum_y'][-1];m=W['fluid_mass'][0]
    results={'clipped_volume_max_abs_error':max(clips),'surface_horizontal_phase_scan':phases,
        'surface_scan_scope':'unperturbed horizontal uniform sampling only; not a bound on a disturbed or tilted water surface',
        'fixed_state_max_changes':fixed_errors,'slip_final_extra_momentum_loss':abs(w-c)/(m*.5),
        'slip_final_speed_control':float(control[-1]['fluid_velocity_y']),
        'slip_final_speed_wall':float(w/m),
        'slip_support_balance_relative':float(abs((w-W['fluid_momentum_y'][0])+W['body_grid_impulse1'][-1]+W['body_contact_impulse1'][-1])/(m*.5))}
    assert max(clips)<1e-10
    assert max(fixed_errors.values())==0
    (DEFAULT_OUTPUT/'measurement_calibration.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
