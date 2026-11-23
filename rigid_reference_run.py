"""Create and run the independently discretized DFSPH reference scenes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import numpy as np
from rigid_validation_run import DEFAULT_OUTPUT,box_samples,surface_plane,clipped_box_fraction
from rigid_coupling.rigid import rotation_numpy

SOURCE=Path(__file__).parent/'reference/SPlisHSPlasH'
def prepare(case,out,duration=None,density_scale=1.,cfl_method=1):
    out.mkdir(parents=True,exist_ok=False)
    g=case['geometry'];size=np.array(g['body_size']);center=np.array(g['body_center'])
    p,_=box_samples([.125]*3,[.875,g['initial_water_level'],.875],case['fluid_spacing'])
    if case['experiment']=='A_float':p=p[~np.all(np.abs(p-center)<size/2,axis=1)]
    with (out/'initial_points.bin').open('wb') as f:
        f.write(struct.pack('<Q',len(p)));f.write(np.asarray(p,dtype='<f8').tobytes())
    T=duration if duration is not None else case['duration']
    mass=float(np.prod(size)*g['density_ratio']*density_scale)
    (out/'study_parameters.txt').write_text(' '.join(map(str,[g['water_volume'],mass,*size,T])))
    shutil.copy2(SOURCE/'data/models/UnitBox.obj',out/'UnitBox.obj')
    common=dict(geometryFile='UnitBox.obj',rotationAxis=[0,0,1],restitution=0,friction=0,collisionObjectType=2,
                mapThickness=0.,mapResolution=[20]*3)
    wall=dict(common,id=1,translation=[.5]*3,scale=[.75]*3,collisionObjectScale=[.75]*3,
              rotationAngle=0,isDynamic=False,isWall=True,invertSDF=True,mapInvert=True)
    body=dict(common,id=2,translation=center.tolist(),scale=size.tolist(),collisionObjectScale=size.tolist(),
              rotationAngle=float(np.deg2rad(g.get('angle_degrees',0))),isDynamic=True,isWall=False,
              density=g['density_ratio']*density_scale,velocity=[0,0,0],angularVelocity=[0,0,0],invertSDF=False,mapInvert=False)
    scene={'Configuration':{'particleRadius':case['particle_radius'],'simulationMethod':4,'boundaryHandlingMethod':0,
        'gravitation':[0,-9.8,0],'cflMethod':cfl_method,'cflFactor':.5,'cflMinTimeStepSize':1e-6,
        'cflMaxTimeStepSize':case['dt_max'],'timeStepSize':case['dt_max'],
        'stopAt':T-1e-10,'pause':False,'dataExportFPS':200,
        'DFSPH':{'minIterations':2,'maxIterations':100,'maxError':.01,'maxIterationsV':100,'maxErrorV':.01,'enableDivergenceSolver':True}},
        'Simulation':{'timeStepSize':case['dt_max'],'gravity':[0,-9.8,0],'gravitation':[0,-9.8,0],'maxIterations':10,'maxIterationsV':5,'velocityUpdateMethod':0,
                      'contactTolerance':.0001,'contactStiffnessRigidBody':1.,'damping':0.},
        'Materials':[{'id':'Fluid','density0':density_scale,'viscosityMethod':0,'surfaceTensionMethod':0,
                     'vorticityMethod':0,'dragMethod':0,'xsph':0.,'xsphBoundary':0.}],
        'RigidBodies':[wall,body],
        'FluidBlocks':[{'id':'Fluid','denseMode':0,'start':[.3]*3,'end':[.32]*3}]}
    (out/'scene.json').write_text(json.dumps(scene,indent=2))
    info={'case':case,'status':'prepared','duration':T,'fluid_particles':len(p),'fluid_mass':g['water_volume']*density_scale,'density_unit_scale':density_scale,
          'body_mass':mass,'uniform_particle_volume':g['water_volume']/len(p),
          'sampling':'cell centroids of complete rectangular pool; body-interior points excluded; global equal volume normalized to prescribed water amount',
          'pilot':duration is not None,'exe_sha256':hashlib.sha256((SOURCE/'bin/SPHSimulator.exe').read_bytes()).hexdigest()}
    (out/'run.json').write_text(json.dumps(info,indent=2));return info

def process(out,case):
    g=case['geometry'];rows=list(csv.DictReader((out/'metrics_raw.csv').open()))
    output=[];bias=None
    for row in rows:
        if not row.get('phi_frame'):continue
        r={k:float(v) for k,v in row.items() if v and k is not None}
        center=np.array([r['body_x'+str(i)] for i in range(3)])
        q=np.array([r['body_q'+str(i)] for i in range(4)])
        body={'center':center,'R':rotation_numpy(q),'size':np.array(g['body_size'])}
        phi=np.fromfile(out/f"phi_{int(r['phi_frame'])}.bin",dtype='<f8').reshape(48,48,48)
        fitted=surface_plane(phi,48,body)
        r.update(surface_y=None,surface_raw_y=None,surface_rms=None,surface_columns=0,submerged_fraction=None)
        if fitted:
            coeff,rms,n=fitted
            raw=coeff@np.array([.5,.5,1])
            if bias is None:bias=float(raw-g['initial_water_level'])
            coeff[2]-=bias
            r.update(surface_y=float(raw-bias),surface_raw_y=float(raw),surface_rms=rms,surface_columns=n,
                submerged_fraction=clipped_box_fraction(center,body['size'],body['R'],coeff))
        output.append(r)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    (out/'measurement.json').write_text(json.dumps({'surface_calibration_offset':bias,'samples':len(output),
        'phi_uses':'SPH nominal particle volumes; not a proof of actual occupied fluid volume conservation'},indent=2))
    return output[-1]

def main():
    a=argparse.ArgumentParser();a.add_argument('--only',nargs='+',required=True);a.add_argument('--pilot',type=float);a.add_argument('--tag',default='');a.add_argument('--density-scale',type=float,default=1.);a.add_argument('--cfl-method',type=int,default=1);a.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);args=a.parse_args()
    root=args.output.resolve()
    cases={c['case_id']:c for c in json.loads((root/'planned_protocol.json').read_text())['cases']}
    for name in args.only:
        case=cases[name];out=root/('reference_pilot'+args.tag if args.pilot else 'data')/name
        info=prepare(case,out,args.pilot,args.density_scale,args.cfl_method);t=time.perf_counter();info['status']='running'
        (out/'run.json').write_text(json.dumps(info,indent=2))
        env=os.environ.copy();env.update(RIGID_STUDY='1',OMP_NUM_THREADS='8')
        with (out/'execution.log').open('w') as f:
            result=subprocess.run([str((SOURCE/'bin/SPHSimulator.exe').resolve()),str(out/'scene.json'),'--no-gui','--no-initial-pause','--output-dir','output'],cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT)
        info.update(exit_code=result.returncode,wall_seconds=time.perf_counter()-t,status='completed' if result.returncode==0 else 'failed')
        if result.returncode==0:
            info['final']=process(out,case)
            if info['final']['time']<info['duration']-case['dt_max']:info['status']='truncated'
        (out/'run.json').write_text(json.dumps(info,indent=2));print(name,info['status'],info['wall_seconds'],flush=True)
        if result.returncode!=0:print(f'Reference failure preserved: {out}',flush=True)
if __name__=='__main__':main()
