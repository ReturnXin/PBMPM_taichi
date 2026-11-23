"""Eccentric gravity-driven impact on a freely moving body resting on the box floor."""
from dataclasses import replace
import numpy as np
from .scene import (Config,make_body,particles,random_box,RIGID,MATERIALS,
                    WATER_SEED,BODY_SEED,SAMPLING_CELL)

BODY_SIZE=np.array([.1875,.375])
MATERIAL_RADIUS=.140625
BOX_FLOOR=3/128
BODY_CENTER=np.array([.4375,BOX_FLOOR+BODY_SIZE[1]/2])
MATERIAL_CENTER=np.array([.53125,BOX_FLOOR+BODY_SIZE[1]+.078125+MATERIAL_RADIUS])
DURATION=.9


def falling_config(**kwargs):
    return replace(Config(grid=128,domain_size=1.,origin=0.,gravity=-9.8,dt=.0005),**kwargs)


def falling_scene(c,material='water',body_only=False):
    if c.coupling!='two_way':raise ValueError('falling scene uses a dynamic supported rigid body')
    local=random_box((2*MATERIAL_RADIUS,2*MATERIAL_RADIUS),WATER_SEED)
    local=local[np.linalg.norm(local,axis=1)<MATERIAL_RADIUS]
    continuum=particles(MATERIAL_CENTER+local,MATERIALS[material],np.pi*MATERIAL_RADIUS**2/len(local),c.dt)
    local=random_box(BODY_SIZE,BODY_SEED)
    b=make_body(local,BODY_CENTER,SAMPLING_CELL**2,size=BODY_SIZE)
    rigid=particles(b['center']+b['local'],RIGID,SAMPLING_CELL**2,c.dt,local=b['local'],mass=b['particle_mass'])
    arrays=rigid if body_only else {k:np.concatenate((continuum[k],rigid[k])) for k in continuum}
    if material!='water':
        n=len(arrays['x'])
        arrays.update(F=np.tile(np.eye(2,dtype=np.float32),(n,1,1)),F_total=np.tile(np.eye(2,dtype=np.float32),(n,1,1)),
                      multiplier=np.zeros((n,2,2),np.float32),plastic_volume=np.zeros(n,np.float32),plastic_strain=np.zeros(n,np.float32))
    return arrays,b

