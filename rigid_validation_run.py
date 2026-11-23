"""Run the registered three-scene rigid study without prescribed buoyancy.

The only solver extension is an actual fixed boundary and cumulative impulse
bookkeeping. Geometry and observations are independent of the coupling force.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import taichi as ti
from scipy.spatial import ConvexHull

from rigid_coupling.config import Config
from rigid_coupling.rigid import make_body, rotation_numpy, rotation
from rigid_coupling.scenes import Scene, lattice
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID, RIGID, particle_arrays, combine_particles
from rigid_coupling.transfers import stencil
from rigid_coupling.runner import source_hashes, write_json

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = WORKSPACE / "毕业论文/论文正文/assets/ch04-rigid-validation-20261028"
LOADED_RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def box_samples(lo, hi, spacing):
    lo, hi = np.asarray(lo), np.asarray(hi)
    counts = np.maximum(1, np.ceil((hi-lo)/spacing-1e-10).astype(int))
    x = lattice((lo+hi)/2, hi-lo, counts)
    vol = np.full(len(x), np.prod(hi-lo)/len(x))
    return x, vol


def pool_samples(level, spacing, body_geometry=None):
    # Split at all horizontal box faces, so cut cells have exact volumes and
    # centroids. C starts clear of the fluid and needs no rotated cut cells.
    axes = [[.125,.875], [.125,level], [.125,.875]]
    if body_geometry is not None:
        center = np.asarray(body_geometry["body_center"])
        size = np.asarray(body_geometry["body_size"])
        for a in range(3):
            lower, upper = axes[a][0], axes[a][1]
            for plane in (center[a]-size[a]/2, center[a]+size[a]/2):
                if lower < plane < upper: axes[a].append(float(plane))
            axes[a].sort()
    groups=[]
    for i,j,k in itertools.product(*(range(len(a)-1) for a in axes)):
        ix=(i,j,k)
        lo=np.array([axes[a][ix[a]] for a in range(3)])
        hi=np.array([axes[a][ix[a]+1] for a in range(3)])
        if body_geometry is not None and np.all(np.abs((lo+hi)/2-center)<size/2-1e-12):continue
        groups.append(box_samples(lo,hi,spacing))
    return np.concatenate([g[0] for g in groups]),np.concatenate([g[1] for g in groups])


def create_scene(case,c):
    g=case["geometry"]
    groups=[]; bodies=[]
    if case["experiment"] == "B_slip":
        fc=np.array(g["fluid_center"]);fs=np.array(g["fluid_size"])
        x,vol=box_samples(fc-fs/2,fc+fs/2,case["fluid_target_spacing"])
        fv=g["fluid_velocity"]
        present=g["wall_present"]
        center=g["wall_center"];size=g["wall_size"];q=[1,0,0,0];rho=1
    else:
        present="body_size" in g
        x,vol=pool_samples(g["initial_water_level"],case["fluid_target_spacing"],
                          g if present and case["experiment"]=="A_float" else None)
        fv=[0,0,0]
        if present:center=g["body_center"];size=g["body_size"];q=g["body_quaternion_wxyz"];rho=g["density_ratio"]
        if not np.isclose(vol.sum(),g["water_volume"],rtol=0,atol=1e-12):
            raise ValueError(f"Initial water volume mismatch: {vol.sum()} vs {g['water_volume']}")
        centroid=np.sum(x*vol[:,None],axis=0)/vol.sum()
        if not np.allclose(centroid[[0,2]],[.5,.5],rtol=0,atol=1e-10):
            raise ValueError(f"Initial water sampling is not centered: {centroid}")
    groups.append(particle_arrays(x,FLUID,vol,fv,c.dt,[.2,.55,.95]))
    if present:
        local=lattice([0,0,0],size,case["rigid_samples_per_axis"])
        rv=float(np.prod(size))/len(local)
        body=make_body(local,center,rv*rho,quaternion=q)
        body.update(size=list(size),volume=float(np.prod(size)),density=rho,
                    fixed=case["experiment"]=="B_slip")
        bodies.append(body)
        pos=body["center"]+body["local"]@rotation_numpy(q).T
        groups.append(particle_arrays(pos,RIGID,rv,[0,0,0],c.dt,[.85,.45,.12],body_id=0,
                                    local=body["local"],mass=body["particle_mass"]))
    return Scene(combine_particles(groups),bodies,case["case_id"],
                 {"fluid_count":len(x),"fluid_volume":float(vol.sum()),"body_count":len(bodies),
                  "initial_surface":g.get("initial_water_level"),"sampling":"exact split-box cell centroids"})


SIGNS=np.array(list(itertools.product((-1,1),repeat=3)),dtype=float)
EDGES=[(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(SIGNS[i]!=SIGNS[j])==1]


def clipped_box_fraction(center,size,R,plane):
    """Independent double-precision clipping by y=ax+bz+c."""
    vertices=np.asarray(center)+(SIGNS*np.asarray(size)/2)@R.T
    a,b,c=plane
    distance=vertices[:,1]-a*vertices[:,0]-b*vertices[:,2]-c
    if np.all(distance<=0):return 1.0
    if np.all(distance>=0):return 0.0
    points=list(vertices[distance<=0])
    for i,j in EDGES:
        if distance[i]*distance[j]<0:
            points.append(vertices[i]+distance[i]/(distance[i]-distance[j])*(vertices[j]-vertices[i]))
    return float(ConvexHull(np.array(points)).volume/np.prod(size))


@ti.data_oriented
class Observer:
    def __init__(self,solver):
        self.s=solver;self.p=solver.p;self.b=solver.bodies.data;self.c=solver.config
        self.n=solver.active;self.nb=solver.bodies.count
        self.phi=ti.field(ti.f64,shape=(self.c.grid,)*3)
        self.mass=ti.field(ti.f64,shape=());self.vol=ti.field(ti.f64,shape=())
        self.mom=ti.Vector.field(3,ti.f64,shape=());self.cent=ti.Vector.field(3,ti.f64,shape=())
        self.speed2=ti.field(ti.f64,shape=());self.initial_vol=ti.field(ti.f64,shape=())
        self.maxpenetration=ti.field(ti.f32,shape=());self.maxdensity=ti.field(ti.f32,shape=())
        self.reset()

    def reset(self):
        self.maxpenetration[None]=0;self.maxdensity[None]=0

    @ti.kernel
    def every_step(self):
        for p in range(self.n[None]):
            if self.p[p].material==FLUID:
                ti.atomic_max(self.maxdensity[None],ti.abs(self.p[p].volume_ratio-1.0))
                if ti.static(self.nb>0):
                    local=rotation(self.b[0].q).transpose()@(self.p[p].x-self.b[0].x)
                    gap=self.b[0].half_size-ti.abs(local)
                    if gap.min()>0:ti.atomic_max(self.maxpenetration[None],gap.min())

    @ti.kernel
    def measure(self):
        self.mass[None]=0;self.vol[None]=0;self.initial_vol[None]=0;self.speed2[None]=0
        self.mom[None]=ti.Vector.zero(ti.f64,3);self.cent[None]=ti.Vector.zero(ti.f64,3)
        for i in ti.grouped(self.phi):self.phi[i]=0
        for p in range(self.n[None]):
            if self.p[p].material==FLUID:
                m=ti.cast(self.p[p].mass,ti.f64);v=self.p[p].d/self.c.dt
                volume=ti.cast(self.p[p].rest_volume,ti.f64)*self.p[p].volume_ratio
                self.mass[None]+=m;self.vol[None]+=volume;self.initial_vol[None]+=self.p[p].rest_volume
                self.mom[None]+=m*v;self.cent[None]+=m*self.p[p].x;self.speed2[None]+=m*v.dot(v)
                base,f,w=stencil(self.p[p].x,self.c.grid)
                for o in ti.static(ti.grouped(ti.ndrange(3,3,3))):
                    weight=w[o[0],0]*w[o[1],1]*w[o[2],2]
                    self.phi[base+o]+=weight*volume*self.c.grid**3


def surface_plane(phi,grid,body=None):
    xx,zz=np.meshgrid(np.arange(grid)/grid,np.arange(grid)/grid,indexing="ij")
    valid=(xx>.125+2/grid)&(xx<.875-2/grid)&(zz>.125+2/grid)&(zz<.875-2/grid)
    if body is not None:
        extent=np.abs(body["R"])@(body["size"]/2)
        valid&=(np.abs(xx-body["center"][0])>extent[0]+2/grid)|(np.abs(zz-body["center"][2])>extent[2]+2/grid)
    heights=[];positions=[]
    for i,k in np.argwhere(valid):
        column=phi[i,:,k]
        crossing=np.where((column[:-1]>=.5)&(column[1:]<.5))[0]
        if not len(crossing):continue
        j=int(crossing[-1]);y=(j+(column[j]-.5)/(column[j]-column[j+1]))/grid
        if .2<y<.84:positions.append([i/grid,k/grid,1]);heights.append(y)
    if len(heights)<12:return None
    positions=np.array(positions);heights=np.array(heights)
    coeff=np.linalg.lstsq(positions,heights,rcond=None)[0]
    residual=float(np.sqrt(np.mean((positions@coeff-heights)**2)))
    return coeff,residual,len(heights)


def observe(solver,obs,case,bias):
    obs.measure()
    mass=float(obs.mass[None]);momentum=obs.mom[None].to_numpy();center=obs.cent[None].to_numpy()/mass
    row={"time":solver.step_index*solver.config.dt,"fluid_mass":mass,"fluid_volume":float(obs.vol[None]),
         "fluid_volume_error":float(obs.vol[None]/obs.initial_vol[None]-1),
         "fluid_vrms":math.sqrt(float(obs.speed2[None])/mass),
         "max_penetration_so_far":float(obs.maxpenetration[None]),"max_local_volume_error_so_far":float(obs.maxdensity[None])}
    for i,a in enumerate("xyz"):
        row["fluid_momentum_"+a]=float(momentum[i]);row["fluid_center_"+a]=float(center[i])
        row["fluid_velocity_"+a]=float(momentum[i]/mass)
    body=None
    if solver.bodies.count:
        b=solver.bodies.data.to_numpy();R=rotation_numpy(b["q"][0]);size=np.array(solver.bodies.initial[0]["size"])
        body={"center":b["x"][0].astype(float),"R":R,"size":size}
        for name in ("x","v","omega","q","grid_impulse","contact_impulse","world_impulse"):
            for i,value in enumerate(b[name][0]):row[f"body_{name}{i}"]=float(value)
        row["body_world_contacts"]=int(b["world_contacts"][0]);row["bottom_gap"]=float(b["x"][0,1]-(np.abs(R)@(size/2))[1]-.125)
    if case["experiment"]!="B_slip":
        result=surface_plane(obs.phi.to_numpy(),solver.config.grid,body)
        row.update(surface_y=None,surface_raw_y=None,surface_rms=None,surface_columns=0,submerged_fraction=None)
        if result is not None:
            plane,rms,count=result;raw=plane[0]*.5+plane[1]*.5+plane[2];plane[2]-=bias
            row.update(surface_y=float(raw-bias),surface_raw_y=float(raw),surface_rms=rms,surface_columns=count)
            if body:row["submerged_fraction"]=clipped_box_fraction(body["center"],body["size"],body["R"],plane)
    return row


def snapshot(solver,path):
    p=solver.particles.snapshot();b=solver.bodies.data.to_numpy()
    np.savez_compressed(path,x=p["x"],v=p["d"]/solver.config.dt,material=p["material"],
                        uid=p["uid"],mass=p["mass"],rest_volume=p["rest_volume"],J=p["volume_ratio"],
                        body_x=b["x"][:solver.bodies.count],body_q=b["q"][:solver.bodies.count])


def run(case,root,pilot=None,backend='cuda'):
    case=case.copy()
    if pilot is not None:case["duration"]=pilot
    out=root/("pilot" if pilot is not None else "data")/case["case_id"]
    if out.exists():raise RuntimeError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    c=Config(grid=case["grid"],boundary=case["grid_boundary_index"],dt=case["dt"],iterations=case["iterations"],
             gravity=case["gravity_y"],viscosity=0.,fluid_relaxation=1.5,rigid_volume=case["rigid_volume"],
             rigid_contact=case["rigid_contact"],world_contact=True,buoyancy_assist=False,
             grid_coupling=case.get("grid_coupling","shared"))
    metadata={"case":case,"config":c.as_dict(),"status":"running","source_hashes":source_hashes(),"backend":backend,
              "runner_sha256":LOADED_RUNNER_SHA256,"pilot":pilot is not None}
    write_json(out/"run.json",metadata)
    ti.init(arch={'cuda':ti.cuda,'vulkan':ti.vulkan}[backend],default_fp=ti.f32,enable_fallback=False,offline_cache=False,random_seed=42)
    try:
        scene=create_scene(case,c);solver=Solver(c,scene);obs=Observer(solver)
        metadata.update(sampling=scene.sampling,particles=solver.particles.n,
                        body=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in b.items() if k not in ("local","particle_mass")} for b in scene.bodies])
        solver.advance(1);obs.every_step();obs.measure();ti.sync();solver.reset();obs.reset()
        # Surface correction is measured on this known zero-time geometric
        # configuration, before any physical run. No density target is used.
        bias=0.
        if case["experiment"]!="B_slip":
            initial_body=None
            if scene.bodies:
                b=scene.bodies[0]
                initial_body={"center":np.asarray(b["center"]),"size":np.asarray(b["size"]),"R":rotation_numpy(b["quaternion"])}
            obs.measure();result=surface_plane(obs.phi.to_numpy(),c.grid,initial_body)
            if result:
                coeff,rms,n=result
                bias=float(coeff@np.array([.5,.5,1])-case["geometry"]["initial_water_level"])
        metadata["surface_calibration_offset"]=bias
        write_json(out/"run.json",metadata)
        steps=round(case["duration"]/c.dt);interval=round(case["sample_interval"]/c.dt)
        rows=[observe(solver,obs,case,bias)]
        snapshot(solver,out/"initial.npz")
        snapsteps={round(t/c.dt) for t in ([.1,.3,.6,1.,1.6] if case["experiment"]=="C_entry" else ([.1,.2] if case["experiment"]=="B_slip" else [1.,3.,6.]))}
        start=perf_counter();lastprint=start
        with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerow(rows[0]);f.flush()
            for _ in range(0,steps,interval):
                for j in range(min(interval,steps-solver.step_index)):
                    solver.advance(1);obs.every_step()
                row=observe(solver,obs,case,bias);writer.writerow(row);rows.append(row);f.flush()
                if solver.step_index in snapsteps:snapshot(solver,out/f"frame_{solver.step_index:06d}.npz")
                now=perf_counter()
                if now-lastprint>=30:
                    print(json.dumps({"case":case["case_id"],"t":row["time"],"elapsed":round(now-start),"y":row.get("body_x1"),"volume_error":row["fluid_volume_error"]}),flush=True);lastprint=now
        ti.sync();metadata.update(status="completed",wall_seconds=perf_counter()-start,samples=len(rows),actual_duration=steps*c.dt)
        snapshot(solver,out/"final.npz");write_json(out/"summary.json",rows[-1])
        print("COMPLETED",case["case_id"],metadata["wall_seconds"],flush=True)
    except Exception as e:
        metadata.update(status="failed",error=repr(e));raise
    finally:
        write_json(out/"run.json",metadata);ti.reset()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--only",nargs="+",required=True)
    parser.add_argument("--pilot",type=float)
    parser.add_argument("--backend",choices=['cuda','vulkan'],default='cuda')
    args=parser.parse_args();root=args.output.resolve()
    protocol=json.loads((root/"planned_protocol.json").read_text(encoding="utf-8"))
    cases={c["case_id"]:c for c in protocol["cases"]}
    for name in args.only:
        if name not in cases or cases[name]["engine"]!="PBMPM":parser.error(f"Unknown local case {name}")
        run(cases[name],root,args.pilot,args.backend)


if __name__=="__main__":main()
