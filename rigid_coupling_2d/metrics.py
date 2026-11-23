"""Double-precision diagnostics; translational/rotational KE and affine norm separate."""
import numpy as np
from .scene import FLUID, RIGID, BODY_CENTER, rotation_numpy


def cross(a, b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def measure(s, p=None):
    p = s.snapshot() if p is None else p
    f = p['material'] != RIGID
    x, v, m, D = p['x'][f].astype(float), p['d'][f].astype(float)/s.c.dt, p['mass'][f].astype(float), p['D'][f].astype(float)
    b = s.body_state()
    pf = np.sum(m[:,None]*v, axis=0)
    pb = s.body_mass*b['velocity'] if s.has_body else np.zeros(2)
    energy_f = .5*np.sum(m*np.sum(v*v,axis=1))
    energy_b = .5*(s.body_mass*np.dot(b['velocity'],b['velocity'])+s.inertia*b['omega']**2) if s.has_body else 0.
    affine = np.sum(m*s.c.dx**2/(8*s.c.dt**2)*np.sum(D*D,axis=(1,2)))
    angular = np.sum(cross(x-BODY_CENTER,m[:,None]*v)) + np.sum(m*s.c.dx**2/(4*s.c.dt)*(D[:,1,0]-D[:,0,1]))
    penetration, shape = 0., 0.
    if s.has_body:
        angular += cross(b['center']-BODY_CENTER,pb)+s.inertia*b['omega']
        R = rotation_numpy(b['angle'])
        local = (x-b['center'])@R
        penetration = float(np.maximum(np.min(np.array(s.half_size)-abs(local),axis=1),0).max(initial=0))
        r = p['material'] == RIGID
        shape = float(np.max(np.linalg.norm(p['x'][r]-(b['center']+p['local'][r]@R.T),axis=1)))
    base = np.floor((p['x'].astype(float)-s.c.origin)/s.c.dx-.5)
    clearance = min(float(base.min()-s.c.boundary),float(s.c.grid-s.c.boundary-(base+2).max()))
    result = dict(time=s.step_index*s.c.dt, cx=float(b['center'][0]),cy=float(b['center'][1]),
                vx=float(b['velocity'][0]),vy=float(b['velocity'][1]),angle=b['angle'],omega=b['omega'],
                pfx=pf[0],pfy=pf[1],pbx=pb[0],pby=pb[1],angular=float(angular),
                kinetic=float(energy_f+energy_b),fluid_kinetic=float(energy_f),body_kinetic=float(energy_b),
                affine_norm=float(affine),mean_speed=float(np.average(np.linalg.norm(v,axis=1),weights=m)) if len(m) else 0.,
                area=float(np.sum(p['area'][f].astype(float)*p['J'][f])),
                penetration=penetration,shape_error=shape,stencil_clearance=clearance)
    if 'F' in p and len(m):
        stretch=np.linalg.svd(p['F'][f].astype(float),compute_uv=False)
        total_stretch=np.linalg.svd(p['F_total'][f].astype(float),compute_uv=False)
        result.update(elastic_strain_rms=float(np.sqrt(np.average(np.sum(np.log(stretch)**2,axis=1),weights=m))),
                      total_strain_rms=float(np.sqrt(np.average(np.sum(np.log(total_stretch)**2,axis=1),weights=m))),
                      plastic_strain_mean=float(np.average(p['plastic_strain'][f],weights=m)),
                      plastic_volume_mean=float(np.average(p['plastic_volume'][f],weights=m)),
                      min_J=float(p['J'][f].min()),max_J=float(p['J'][f].max()),
                      min_elastic_det=float(np.linalg.det(p['F'][f]).min()))
    return result
