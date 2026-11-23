"""Reproducible area samples and parameters; all coordinates are two-dimensional."""
from dataclasses import dataclass, asdict
from itertools import product
import math

import numpy as np

FLUID, RIGID, ELASTIC, SAND = 0, 1, 2, 3
MATERIALS = {'water':FLUID,'elastic':ELASTIC,'sand':SAND}
WATER_CENTER = np.array([.328125, .5])
WATER_RADIUS = .09375
BODY_CENTER = np.array([.625, .5])
BODY_SIZE = np.array([.125, .25])
SAMPLING_CELL = 1/256
WATER_SEED, BODY_SEED = 20261117, 20261118
SPEED = .75
CASES = {
    "center_one_way": (0., "one_way", True),
    "center_two_way": (0., "two_way", True),
    "offset_one_way": (.0625, "one_way", True),
    "offset_two_way": (.0625, "two_way", True),
    "mirror_two_way": (-.0625, "two_way", True),
    "fluid_control": (0., "two_way", False),
}


@dataclass(frozen=True)
class Config:
    grid: int = 192
    domain_size: float = 2.25
    origin: float = -.625
    boundary: int = 3
    dt: float = .001
    iterations: int = 5
    coupling: str = "two_way"
    relaxation: float = 1.
    viscosity: float = 0.
    gravity: float = 0.
    elastic_solver: str = "xpbmpm"
    compliance: float = 5e-7
    elastic_relaxation: float = .2
    volume_weight: float = .25
    sand_relaxation: float = .2
    sand_viscosity: float = .1
    friction_angle: float = 35.

    def __post_init__(self):
        if not isinstance(self.grid, int) or self.grid < 16:
            raise ValueError("grid must be an integer >= 16")
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(self.boundary, int) or not 2 <= self.boundary < self.grid//2:
            raise ValueError("boundary must leave complete particle stencils")
        if self.coupling not in ("one_way", "two_way"):
            raise ValueError("coupling must be one_way or two_way")
        if not all(math.isfinite(v) for v in
                   (self.domain_size, self.origin, self.dt, self.relaxation, self.viscosity, self.gravity)):
            raise ValueError("parameters must be finite")
        if min(self.domain_size, self.dt) <= 0 or self.relaxation < 0 or not 0 <= self.viscosity <= 1:
            raise ValueError("invalid time, domain, or material parameter")
        if self.elastic_solver not in ('pbmpm','xpbmpm'):
            raise ValueError("elastic_solver must be pbmpm or xpbmpm")
        if not all(math.isfinite(v) for v in (self.compliance,self.elastic_relaxation,
                   self.volume_weight,self.sand_relaxation,self.sand_viscosity,self.friction_angle)):
            raise ValueError("solid parameters must be finite")
        if self.compliance<0 or not 0<self.elastic_relaxation<=1 or not 0<=self.volume_weight<1:
            raise ValueError("invalid elastic parameters")
        if not 0<self.sand_relaxation<=1 or not 0<=self.sand_viscosity<=1 or not 0<=self.friction_angle<90:
            raise ValueError("invalid sand parameters")


    @property
    def dx(self):
        return self.domain_size/self.grid

    def as_dict(self):
        return asdict(self)


def random_box(size, seed, cell=SAMPLING_CELL):
    size = np.asarray(size, dtype=float)
    counts = np.rint(size/cell).astype(int)
    if np.any(counts % 2) or not np.allclose(counts*cell, size, atol=1e-14, rtol=0):
        raise ValueError("symmetric sampling requires even cell counts")
    cells = np.indices(counts//2).reshape(2, -1).T
    positive = (cells+np.random.default_rng(seed).random(cells.shape))*cell
    signs = np.array(list(product((-1., 1.), repeat=2)))
    return (positive[None]*signs[:, None]).reshape(-1, 2)


def rotation_numpy(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def make_body(local, center, mass, angle=0., velocity=(0., 0.), omega=0., size=BODY_SIZE):
    local = np.asarray(local, dtype=float).copy()
    masses = np.broadcast_to(np.asarray(mass, dtype=float), (len(local),)).copy()
    if local.ndim != 2 or local.shape[1] != 2 or len(local) < 3:
        raise ValueError("body samples must have shape (N, 2)")
    if not np.isfinite(local).all() or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("body samples and masses must be finite, with positive masses")
    local -= np.average(local, weights=masses, axis=0)
    inertia = float(np.sum(masses*np.sum(local*local, axis=1)))
    if inertia <= 1e-12:
        raise ValueError("body samples need positive planar inertia")
    return dict(local=local, particle_mass=masses, mass=float(masses.sum()), inertia=inertia,
                center=np.asarray(center, dtype=float), size=np.asarray(size, dtype=float),
                angle=float(angle), velocity=np.asarray(velocity, dtype=float), omega=float(omega))


def particles(x, material, area, dt, velocity=(0., 0.), local=None, mass=None):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    return dict(x=x, d=np.tile(np.asarray(velocity, dtype=np.float32)*dt, (n, 1)),
                D=np.zeros((n, 2, 2), np.float32), J=np.ones(n, np.float32),
                mass=np.broadcast_to(area if mass is None else mass, (n,)).astype(np.float32).copy(),
                area=np.broadcast_to(area, (n,)).astype(np.float32).copy(),
                material=np.full(n, material, np.int32),
                local=np.zeros_like(x) if local is None else np.asarray(local, np.float32),
                predicted_d=np.zeros_like(x))


def build_scene(c, case="offset_two_way", material="water"):
    offset, mode, with_body = CASES[case]
    if mode != c.coupling:
        raise ValueError("case and coupling mode disagree")
    local = random_box((2*WATER_RADIUS, 2*WATER_RADIUS), WATER_SEED)
    local = local[np.linalg.norm(local, axis=1) < WATER_RADIUS]
    water = WATER_CENTER+[0., offset]+local
    water_area = np.pi*WATER_RADIUS**2/len(water)
    groups = [particles(water, MATERIALS[material], water_area, c.dt, (SPEED, 0.))]
    body = None
    if with_body:
        local = random_box(BODY_SIZE, BODY_SEED)
        body = make_body(local, BODY_CENTER, SAMPLING_CELL**2)
        groups.append(particles(body["center"]+body["local"], RIGID, SAMPLING_CELL**2,
                                c.dt, local=body["local"], mass=body["particle_mass"]))
    arrays = {k: np.concatenate([g[k] for g in groups]) for k in groups[0]}
    if material!='water':
        n=len(arrays['x'])
        arrays.update(F=np.tile(np.eye(2,dtype=np.float32),(n,1,1)),
                      F_total=np.tile(np.eye(2,dtype=np.float32),(n,1,1)),
                      multiplier=np.zeros((n,2,2),np.float32),
                      plastic_volume=np.zeros(n,np.float32),
                      plastic_strain=np.zeros(n,np.float32))
    return arrays, body
