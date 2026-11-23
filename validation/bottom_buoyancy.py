"""CUDA checks for floor-only flotation: rest, water push/release and mixed densities."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.rigid import rotation, rotation_numpy
from rigid_coupling.runner import source_hashes, save_state
from rigid_coupling.shake import build_shake_scene
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID


def floor_scene(c):
    scene=build_shake_scene(c,FLUID,resolution=24,seed=42)
    p=scene.particles
    for i,(body,density) in enumerate(zip(scene.bodies,(.35,.8,1.2))):
        factor=density/c.rigid_density
        for key in ('particle_mass','mass'):
            body[key]*=factor
        body['inertia']*=factor
        body['inverse_inertia']/=factor
        body['density']=density
        R=rotation_numpy(body['quaternion'])
        extent=np.abs(R)@(np.array(body['size'])/2)
        body['center'][1]=.21+extent[1]
        owned=p['body_id']==i
        p['x'][owned]=body['center']+body['local']@R.T
        p['x0'][owned]=p['x'][owned]
        p['mass'][owned]*=factor
    keep=np.ones(len(p['x']),dtype=bool)
    for body in scene.bodies:
        local=(p['x']-body['center'])@rotation_numpy(body['quaternion'])
        keep &= ~((p['material']==FLUID)&np.all(np.abs(local)<np.array(body['size'])/2+1e-6,axis=1))
    scene.particles={k:v[keep] for k,v in p.items()}
    return scene


@ti.data_oriented
class Recorder:
    def __init__(self,s,capacity):
        self.s=s
        self.rows=ti.Vector.field(11,ti.f32,shape=(capacity,s.bodies.count))

    @ti.kernel
    def record(self,index:ti.i32):
        for b in range(self.s.bodies.count):
            body=self.s.bodies.data[b]
            R=self.s.container.orientation[None]
            extent=ti.abs(R.transpose()@rotation(body.q))@body.half_size
            local=R.transpose()@(body.x-self.s.container.center[None])
            self.rows[index,b]=ti.Vector([body.x[0],body.x[1],body.x[2],
                body.v[0],body.v[1],body.v[2],body.omega.norm(),
                local[1]-extent[1]+self.s.container.half_size[1],
                self.s.buoyancy.correction[b],self.s.buoyancy.fraction[b],
                body.mass/(8*body.half_size[0]*body.half_size[1]*body.half_size[2])])


def execute(s,recorder,mode,duration,out):
    s.reset()
    stride=5
    samples=round(duration/s.config.dt)//stride
    for i in range(samples):
        step=i*stride
        if mode=='push_release':
            if step==500:
                s.mouse.set_force(8.,0.,0.,100)
            if step==600:
                s.mouse.release()
        s.advance(stride)
        recorder.record(i)
        if (i+1)%100==0:
            print(mode, 'simulated seconds=',s.step_index*s.config.dt,flush=True)
    values=recorder.rows.to_numpy()[:samples]
    names=('x','y','z','vx','vy','vz','angular_speed','floor_gap','assist_delta_vy','wet_fraction','density')
    with (out/f'{mode}.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(('time','body',*names))
        for i in range(samples):
            for b in range(s.bodies.count):
                writer.writerow(((i+1)*stride*s.config.dt,b,*values[i,b].tolist()))
    tail=values[-min(samples,200):]
    metrics=[]
    for b in range(s.bodies.count):
        metrics.append(dict(body=b,density=float(values[0,b,10]),
            final_center_y=float(values[-1,b,1]),min_floor_gap=float(values[:,b,7].min()),
            max_floor_gap=float(values[:,b,7].max()),final_floor_gap=float(values[-1,b,7]),
            max_assist_delta_vy=float(values[:,b,8].max()),
            sampled_assist_count=int((values[:,b,8]>1e-7).sum()),
            last_two_seconds_vy_rms=float(np.sqrt(np.mean(tail[:,b,4]**2))),
            last_two_seconds_y_range=float(np.ptp(tail[:,b,1]))))
    assert np.isfinite(values).all()
    assert values[:,:,7].min()>-1e-5
    if mode=='static':
        assert np.max(np.abs(values[:,:,8]))==0.,'Floating scene must not invoke the assist'
        assert max(row['last_two_seconds_vy_rms'] for row in metrics)<.01,metrics
    if mode=='bottom':
        assert values[:,2,8].max()==0.,'Dense body must never receive the assist'
        assert values[:,:2,8].max()>0.,'At least one light bottom body should require support'
        assert all(row['max_floor_gap']>.01 for row in metrics[:2]),metrics
    if mode=='push_release':
        np.testing.assert_array_equal(s.mouse.control[None].to_numpy(),0.)
    save_state(s,out,mode)
    return dict(duration=duration,particle_count=s.particles.n,metrics=metrics,passed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('surface','bottom'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    c=Config(rigid_density=.35,rigid_volume=True,rigid_contact=True,world_contact=True,
             grid_coupling='separate',viscosity=0.,buoyancy_assist=True,
             fluid_relaxation=1.,boundary_density='available',wall_transfer='reflected',density_feedback='clamped')
    result=dict(config=c.as_dict(),source_hashes=source_hashes(),cases={})
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    ti.init(arch=ti.cuda,enable_fallback=False,offline_cache=False,random_seed=42)
    try:
        scene=floor_scene(c) if args.mode=='bottom' else build_shake_scene(c,FLUID,resolution=24,seed=42)
        s=Solver(c,scene,mouse_control=True)
        # Both schedulers must apply the same conditional correction.
        s.advance(4)
        ordinary=s.bodies.data.to_numpy()
        ordinary_correction=s.buoyancy.correction.to_numpy()
        s.reset();s.enable_cuda_graph();s.advance(4)
        graph=s.bodies.data.to_numpy()
        for name in ('x','q','v','omega'):
            np.testing.assert_allclose(ordinary[name],graph[name],atol=3e-5,rtol=3e-4,err_msg=name)
        np.testing.assert_allclose(ordinary_correction,s.buoyancy.correction.to_numpy(),atol=3e-6,rtol=3e-4)
        result['graph_parity_passed']=True
        recorder=Recorder(s,400)
        modes=(('bottom',2.),) if args.mode=='bottom' else (('static',4.),('push_release',4.))
        for mode,duration in modes:
            result['cases'][mode]=execute(s,recorder,mode,duration,args.output)
            (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(mode,result['cases'][mode],flush=True)
    finally:
        ti.reset()


if __name__=='__main__':
    main()
