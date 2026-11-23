"""An eccentric rigid block released over a circular material resting on the floor."""
import numpy as np
from .falling import falling_config
from .scene import (MATERIALS, RIGID, SAMPLING_CELL, WATER_SEED, BODY_SEED,
                    particles, random_box, make_body)

MATERIAL_RADIUS = .140625
BODY_SIZE = np.array([.1875, .375])
BODY_CENTER = np.array([.59375, 3/128+2*MATERIAL_RADIUS+.078125+BODY_SIZE[1]/2])
BODY_DENSITY = 1.
DURATION = .9


def rigid_drop_config(**kwargs):
    return falling_config(**kwargs)


def rigid_drop_scene(c, material='water', *, dry=False):
    if c.coupling != 'two_way':
        raise ValueError('rigid_drop requires a dynamic rigid body')
    lo = c.origin+c.boundary*c.dx
    hi = c.origin+c.domain_size-c.boundary*c.dx
    if not (lo+2*MATERIAL_RADIUS < BODY_CENTER[1]-BODY_SIZE[1]/2 and
            BODY_CENTER[1]+BODY_SIZE[1]/2 < hi):
        raise ValueError('The body must start inside the box above the material')
    local = random_box(BODY_SIZE, BODY_SEED)
    body = make_body(local, BODY_CENTER, BODY_DENSITY*SAMPLING_CELL**2,
                     size=BODY_SIZE)
    rigid = particles(body['center']+body['local'], RIGID, SAMPLING_CELL**2,
                      c.dt, local=body['local'], mass=body['particle_mass'])
    arrays = rigid
    if not dry:
        center = np.array([(lo+hi)/2, lo+MATERIAL_RADIUS])
        local = random_box((2*MATERIAL_RADIUS,2*MATERIAL_RADIUS), WATER_SEED)
        local = local[np.linalg.norm(local,axis=1)<MATERIAL_RADIUS]
        continuum = particles(center+local, MATERIALS[material],
                              np.pi*MATERIAL_RADIUS**2/len(local), c.dt)
        arrays = {k:np.concatenate((continuum[k], rigid[k])) for k in rigid}
        if material != 'water':
            n = len(arrays['x'])
            arrays.update(F=np.tile(np.eye(2, dtype=np.float32), (n,1,1)),
                          F_total=np.tile(np.eye(2, dtype=np.float32), (n,1,1)),
                          multiplier=np.zeros((n,2,2), np.float32),
                          plastic_volume=np.zeros(n, np.float32),
                          plastic_strain=np.zeros(n, np.float32))
    return arrays, body
