"""A rectangular body released above a planar pool, using the common Solver."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .scene import Config, FLUID, RIGID, particles, random_box, make_body


@dataclass(frozen=True)
class DropSpec:
    body_density: float = .8
    release_height: float = .75
    water_depth: float = .375
    settle_time: float = .5
    duration: float = .8
    cell: float = 1/256
    water_seed: int = 20261118
    body_seed: int = 20261119

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.body_density,self.release_height,self.water_depth,self.duration,self.cell)):
            raise ValueError('Drop dimensions, density, duration and spacing must be positive')
        if not math.isfinite(self.settle_time) or self.settle_time < 0:
            raise ValueError('settle_time must be nonnegative')

    def as_dict(self):
        return asdict(self)


BODY_SIZE = np.array([.1875,.125])


def drop_config():
    return Config(grid=128,domain_size=1.,origin=0.,boundary=3,dt=.0005,
                  iterations=5,gravity=-9.8,viscosity=.02)


def pool(c, spec):
    lo = c.origin+c.boundary*c.dx
    hi = c.origin+c.domain_size-c.boundary*c.dx
    size = np.array([hi-lo,spec.water_depth])
    center = np.array([(lo+hi)/2,lo+spec.water_depth/2])
    x = center+random_box(size,spec.water_seed,cell=spec.cell)
    return particles(x,FLUID,spec.cell**2,c.dt)


def drop_scene(c, spec, water=None, *, dry=False):
    local = random_box(BODY_SIZE,spec.body_seed,cell=spec.cell)
    center = np.array([c.origin+c.domain_size/2,spec.release_height])
    lo = c.origin+c.boundary*c.dx
    hi = c.origin+c.domain_size-c.boundary*c.dx
    if center[1]-BODY_SIZE[1]/2 <= lo+spec.water_depth+2*c.dx:
        raise ValueError('Release the body above the pool with a positive air gap')
    if center[1]+BODY_SIZE[1]/2 >= hi:
        raise ValueError('Body must start below the top wall')
    body = make_body(local,center,spec.body_density*spec.cell**2,size=BODY_SIZE)
    rigid = particles(body['center']+body['local'],RIGID,spec.cell**2,c.dt,
                      mass=body['particle_mass'],local=body['local'])
    if dry:
        arrays = rigid
    else:
        water = pool(c,spec) if water is None else water
        if np.any(water['material']!=FLUID):
            raise ValueError('The settled pool must contain only water')
        if np.max(water['x'][:,1]) >= center[1]-BODY_SIZE[1]/2-c.dx:
            raise ValueError('The settled pool intersects the release region')
        arrays = {key:np.concatenate((water[key],rigid[key])) for key in rigid}
    return arrays,body


def settled_pool(c,spec,progress=None):
    from .solver import Solver
    s=Solver(c,pool(c,spec))
    steps=round(spec.settle_time/c.dt)
    if not np.isclose(steps*c.dt,spec.settle_time):
        raise ValueError('settle_time must be a multiple of dt')
    for step in range(steps):
        s.step()
        if (step+1)%100==0:
            s.validate()
            if progress:progress(step+1,steps)
    s.validate()
    return s.snapshot()
