"""Planar PBMPM: area constraint, nine-node APIC, one three-DOF rigid body.

The displacement fields use length, and D stores dt times the velocity gradient.
Fluid/rigid motion channels are separate; rest-area occupancy is shared, as in
the production solver's separate/raw/full configuration.
"""
from copy import deepcopy
import numpy as np
import taichi as ti

from .scene import FLUID, RIGID, ELASTIC, SAND, rotation_numpy
from .materials import elastic_target,sand_target,sand_return


@ti.func
def cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


@ti.func
def perp(a):
    return ti.Vector([-a[1], a[0]])


@ti.func
def rotation(angle):
    c, s = ti.cos(angle), ti.sin(angle)
    return ti.Matrix([[c, -s], [s, c]])


@ti.data_oriented
class Solver:
    def __init__(self, config, arrays, body=None):
        self.c = config
        self.initial, self.initial_body = deepcopy(arrays), deepcopy(body)
        self.n, self.has_body = len(arrays['x']), body is not None
        self.has_solids = bool(np.any((arrays['material']==ELASTIC)|(arrays['material']==SAND)))
        if self.has_solids and not {'F','F_total','multiplier','plastic_volume','plastic_strain'}<=arrays.keys():
            raise ValueError('Solid particles need deformation and plastic history fields')
        self.dynamic = self.has_body and config.coupling == 'two_way'
        self.fields = {}
        for name, a in arrays.items():
            if a.ndim == 1:
                field = ti.field(ti.i32 if name == 'material' else ti.f32, shape=self.n)
            elif a.ndim == 2:
                field = ti.Vector.field(2, ti.f32, shape=self.n)
            else:
                field = ti.Matrix.field(2, 2, ti.f32, shape=self.n)
            self.fields[name] = field
            setattr(self, name, field)
        self.center = ti.Vector.field(2, ti.f32, shape=())
        self.velocity = ti.Vector.field(2, ti.f32, shape=())
        self.angle = ti.field(ti.f32, shape=())
        self.omega = ti.field(ti.f32, shape=())
        self.body_mass = body['mass'] if body else 1.
        self.inertia = body['inertia'] if body else 1.
        self.half_size = tuple(body['size']/2) if body else (0., 0.)
        self.feedback = ti.Vector.field(2, ti.f32, shape=())
        self.torque = ti.field(ti.f32, shape=())
        self.contact_impulse = ti.Vector.field(2, ti.f32, shape=())
        self.contact_angular = ti.field(ti.f32, shape=())
        # Cumulative diagnostics can lose tiny floor impulses after a long run.
        # This accumulator is read by diagnostics only, never by the dynamics.
        self.wall_impulse = ti.Vector.field(2, ti.f64, shape=())
        shape = (2, config.grid, config.grid)
        self.grid_mass = ti.field(ti.f32, shape=shape)
        self.grid_d = ti.Vector.field(2, ti.f32, shape=shape)
        self.occupancy = ti.field(ti.f32, shape=(config.grid,)*2)
        self.delta = ti.Vector.field(2, ti.f32, shape=self.n)
        self.delta_grid = ti.Vector.field(2, ti.f32, shape=(config.grid,)*2)
        self.invalid = ti.field(ti.i32, shape=())
        self.reset()

    def reset(self):
        for name, a in self.initial.items():
            self.fields[name].from_numpy(a)
        b = self.initial_body
        self.center[None] = b['center'] if b else (0., 0.)
        self.velocity[None] = b['velocity'] if b and self.dynamic else (0., 0.)
        self.angle[None] = b['angle'] if b else 0.
        self.omega[None] = b['omega'] if b and self.dynamic else 0.
        self.contact_impulse.fill(0)
        self.contact_angular.fill(0)
        self.wall_impulse.fill(0)
        self.delta.fill(0)
        self.delta_grid.fill(0)
        self.invalid.fill(0)
        self.step_index = 0
        self.rebuild()
        self.validate()

    def snapshot(self):
        return {name: f.to_numpy() for name, f in self.fields.items()}

    def body_state(self):
        return dict(center=self.center[None].to_numpy(), velocity=self.velocity[None].to_numpy(),
                    angle=float(self.angle[None]), omega=float(self.omega[None]))

    @ti.func
    def stencil(self, x):
        scaled = (x-self.c.origin)/self.c.dx
        base = ti.cast(ti.floor(scaled-.5), ti.i32)
        f = scaled-base
        w = ti.Matrix.rows([.5*(1.5-f)**2, .75-(f-1.)**2, .5*(f-.5)**2])
        return base, f, w

    @ti.func
    def valid(self, x):
        lo = self.c.origin+.5*self.c.dx
        hi = self.c.origin+(self.c.grid-1.5)*self.c.dx
        return x[0] >= lo and x[0] < hi and x[1] >= lo and x[1] < hi

    @ti.kernel
    def check(self):
        for p in range(self.n):
            if not self.valid(self.x[p]) or not self.J[p] > 0:
                self.invalid[None] = 1
            for a in ti.static(range(2)):
                if ti.math.isnan(self.d[p][a]) or ti.math.isinf(self.d[p][a]):
                    self.invalid[None] = 1
                for b in ti.static(range(2)):
                    if ti.math.isnan(self.D[p][a,b]) or ti.math.isinf(self.D[p][a,b]):
                        self.invalid[None] = 1
            if ti.math.isinf(self.J[p]):
                self.invalid[None] = 1
            if ti.static(self.has_solids):
                if self.material[p]==ELASTIC or self.material[p]==SAND:
                    if not self.F[p].determinant()>0 or not self.F_total[p].determinant()>0:
                        self.invalid[None]=1
                    for a,b in ti.static(ti.ndrange(2,2)):
                        if ti.math.isnan(self.F[p][a,b]) or ti.math.isinf(self.F[p][a,b]):
                            self.invalid[None]=1
                        if ti.math.isnan(self.F_total[p][a,b]) or ti.math.isinf(self.F_total[p][a,b]):
                            self.invalid[None]=1
                        if ti.math.isnan(self.multiplier[p][a,b]) or ti.math.isinf(self.multiplier[p][a,b]):
                            self.invalid[None]=1
                    if ti.math.isnan(self.plastic_volume[p]) or ti.math.isinf(self.plastic_volume[p]):
                        self.invalid[None]=1
                    if ti.math.isnan(self.plastic_strain[p]) or ti.math.isinf(self.plastic_strain[p]):
                        self.invalid[None]=1

    def validate(self):
        self.check()
        if self.invalid[None]:
            raise FloatingPointError('Invalid planar state or incomplete transfer stencil')
        b = self.body_state()
        if not all(np.isfinite(v).all() for v in b.values()):
            raise FloatingPointError('Non-finite rigid state')

    @ti.kernel
    def predict(self):
        for p in range(self.n):
            if self.material[p] != RIGID:
                self.d[p][1] += self.c.gravity*self.c.dt**2
            if ti.static(self.has_solids):
                self.multiplier[p]=ti.Matrix.zero(ti.f32,2,2)
        if ti.static(self.dynamic):
            self.velocity[None][1] += self.c.gravity*self.c.dt

    @ti.func
    def broadcast(self, p):
        arm = rotation(self.angle[None])@self.local[p]
        self.d[p] = self.c.dt*(self.velocity[None]+self.omega[None]*perp(arm))
        self.predicted_d[p] = self.d[p]
        z = self.c.dt*self.omega[None]
        self.D[p] = ti.Matrix([[0., -z], [z, 0.]])

    @ti.kernel
    def prepare(self):
        for p in range(self.n):
            if self.material[p] == RIGID:
                self.broadcast(p)
            elif self.material[p]==FLUID:
                D = self.D[p]
                D -= .5*self.c.viscosity*(D+D.transpose())
                residual = 1./ti.max(self.J[p], .1)-1.-D.trace()
                self.D[p] = D+.5*self.c.relaxation*residual*ti.Matrix.identity(ti.f32, 2)
            if ti.static(self.has_solids):
                if self.material[p]==ELASTIC:
                    D=self.D[p]
                    target=elastic_target(self.F[p],D,self.c.volume_weight)
                    if ti.static(self.c.elastic_solver=='xpbmpm'):
                        alpha=self.c.compliance/self.c.dt**2
                        delta=(target-D-alpha*self.multiplier[p])/(1.+alpha)
                        self.D[p]=D+delta
                        self.multiplier[p]+=delta
                    else:
                        self.D[p]=D+self.c.elastic_relaxation*(target-D)
                elif self.material[p]==SAND:
                    D=self.D[p]
                    target=sand_target(self.F[p],D,self.plastic_volume[p],self.c.volume_weight)
                    D+=self.c.sand_relaxation*(target-D)
                    self.D[p]=D-.5*self.c.sand_viscosity*(D+D.transpose())
        for i in ti.grouped(self.grid_mass):
            self.grid_mass[i] = 0.
            self.grid_d[i] = ti.Vector.zero(ti.f32, 2)
        for i in ti.grouped(self.occupancy):
            self.occupancy[i] = 0.
            self.delta_grid[i] = ti.Vector.zero(ti.f32, 2)
        self.feedback[None] = ti.Vector.zero(ti.f32, 2)
        self.torque[None] = 0.

    @ti.kernel
    def p2g(self):
        for p in range(self.n):
            if self.valid(self.x[p]):
                base, f, w = self.stencil(self.x[p])
                channel = ti.cast(self.material[p]==RIGID,ti.i32)
                for o in ti.static(ti.grouped(ti.ndrange(3, 3))):
                    weight = w[o[0],0]*w[o[1],1]
                    node = base+o
                    arm = (o-f)*self.c.dx
                    m = weight*self.mass[p]
                    self.grid_mass[channel,node[0],node[1]] += m
                    self.grid_d[channel,node[0],node[1]] += m*(self.d[p]+self.D[p]@arm)
                    self.occupancy[node] += weight*self.area[p]/self.c.dx**2
            else:
                self.invalid[None] = 1

    @ti.kernel
    def normalize(self):
        for ch,i,j in self.grid_mass:
            if self.grid_mass[ch,i,j] > 0:
                d = self.grid_d[ch,i,j]/self.grid_mass[ch,i,j]
                original_d=d
                if ch == FLUID:
                    node = ti.Vector([i,j])
                    for a in ti.static(range(2)):
                        if (node[a] <= self.c.boundary and d[a] < 0) or (node[a] >= self.c.grid-self.c.boundary and d[a] > 0):
                            d[a] = 0.
                self.grid_d[ch,i,j] = d
                if (d-original_d).norm_sqr()>0:
                    self.wall_impulse[None]+=self.grid_mass[ch,i,j]*(d-original_d)/self.c.dt

    @ti.kernel
    def g2p(self):
        for p in range(self.n):
            if self.valid(self.x[p]):
                base,f,w = self.stencil(self.x[p])
                d, D, density = ti.Vector.zero(ti.f32,2), ti.Matrix.zero(ti.f32,2,2), 0.
                ch = ti.cast(self.material[p]==RIGID,ti.i32)
                for o in ti.static(ti.grouped(ti.ndrange(3,3))):
                    weight = w[o[0],0]*w[o[1],1]
                    node = base+o
                    gd = self.grid_d[ch,node[0],node[1]]
                    d += weight*gd
                    D += 4./self.c.dx**2*weight*gd.outer_product((o-f)*self.c.dx)
                    density += weight*self.occupancy[node]
                self.d[p], self.D[p] = d, D
                if self.material[p] == FLUID and density > 1.:
                    self.J[p] = .9*self.J[p]+.1/density
                if ti.static(self.dynamic):
                    if ch == RIGID:
                        self.collect_particle(p)

    @ti.func
    def collect_particle(self, p):
        impulse = self.mass[p]*(self.d[p]-self.predicted_d[p])/self.c.dt
        self.feedback[None] += impulse
        self.torque[None] += cross(self.x[p]-self.center[None], impulse)

    @ti.kernel
    def collect_feedback(self):
        for p in range(self.n):
            if self.material[p] == RIGID:
                self.collect_particle(p)

    @ti.kernel
    def apply_feedback(self):
        if ti.static(self.dynamic):
            self.velocity[None] += self.feedback[None]/self.body_mass
            self.omega[None] += self.torque[None]/self.inertia
        self.feedback[None] = ti.Vector.zero(ti.f32,2)
        self.torque[None] = 0.

    @ti.kernel
    def project_contact(self):
        for p in range(self.n):
            self.delta[p] = ti.Vector.zero(ti.f32,2)
            if ti.static(self.has_body):
                if self.material[p] != RIGID:
                    R = rotation(self.angle[None]+self.c.dt*self.omega[None])
                    half = ti.Vector(self.half_size)
                    lo = self.c.origin+self.c.boundary*self.c.dx
                    hi = self.c.origin+(self.c.grid-self.c.boundary)*self.c.dx
                    extent = ti.abs(R)@half
                    center = ti.max(lo+extent, ti.min(hi-extent, self.center[None]+self.c.dt*self.velocity[None]))
                    point = self.x[p]+self.d[p]
                    local = R.transpose()@(point-center)
                    if (half-ti.abs(local)).min() > 0:
                        depth, normal = 1e10, ti.Vector.zero(ti.f32,2)
                        for a in ti.static(range(2)):
                            for side in ti.static((-1,1)):
                                candidate = side*R[:,a]
                                distance = half[a]-side*local[a]
                                surface = point+distance*candidate
                                if surface.min() >= lo-1e-6 and surface.max() <= hi+1e-6 and distance < depth:
                                    depth, normal = distance, candidate
                        if depth < 1e9:
                            arm = point-center
                            inverse = 1./self.mass[p]
                            if ti.static(self.dynamic):
                                inverse += 1./self.body_mass+cross(arm,normal)**2/self.inertia
                            impulse = ti.min(depth,1.)/(self.c.dt*inverse)*normal
                            self.delta[p] = self.c.dt*impulse/self.mass[p]
                            self.d[p] += self.delta[p]
                            if ti.static(self.dynamic):
                                self.feedback[None] -= impulse
                                self.torque[None] -= cross(arm,impulse)
                            self.contact_impulse[None] -= impulse
                            self.contact_angular[None] -= cross(arm,impulse)

    @ti.kernel
    def scatter_contact(self):
        for p in range(self.n):
            if self.material[p] != RIGID and self.valid(self.x[p]):
                base,f,w = self.stencil(self.x[p])
                for o in ti.static(ti.grouped(ti.ndrange(3,3))):
                    weight = w[o[0],0]*w[o[1],1]
                    self.delta_grid[base+o] += weight*self.mass[p]*self.delta[p]

    @ti.kernel
    def gather_contact(self):
        for p in range(self.n):
            if self.material[p] != RIGID and self.valid(self.x[p]):
                base,f,w = self.stencil(self.x[p])
                D = ti.Matrix.zero(ti.f32,2,2)
                for o in ti.static(ti.grouped(ti.ndrange(3,3))):
                    weight = w[o[0],0]*w[o[1],1]
                    node = base+o
                    m = self.grid_mass[FLUID,node[0],node[1]]
                    if m > 0:
                        D += 4./self.c.dx**2*weight*(self.delta_grid[node]/m).outer_product((o-f)*self.c.dx)
                # Symmetric strain changes area but preserves APIC spin.
                self.D[p] += .5*(D+D.transpose())

    @ti.kernel
    def integrate(self):
        lo = self.c.origin+self.c.boundary*self.c.dx
        hi = self.c.origin+(self.c.grid-self.c.boundary)*self.c.dx
        for p in range(self.n):
            if self.material[p] != RIGID:
                trial = self.x[p]+self.d[p]
                clipped = ti.max(lo,ti.min(hi,trial))
                for a in ti.static(range(2)):
                    if trial[a] != clipped[a]:
                        self.wall_impulse[None][a]-=self.mass[p]*self.d[p][a]/self.c.dt
                        self.d[p][a] = 0.
                self.x[p] = clipped
                if self.material[p]==FLUID:
                    self.J[p] = ti.max(self.J[p]*(1.+self.D[p].trace()), .2)
                if ti.static(self.has_solids):
                    if self.material[p]==ELASTIC or self.material[p]==SAND:
                        increment=ti.Matrix.identity(ti.f32,2)+self.D[p]
                        self.F_total[p]=increment@self.F_total[p]
                        F=increment@self.F[p]
                        if self.material[p]==SAND:
                            F,history,plastic=sand_return(F,self.plastic_volume[p],self.c.volume_weight,self.c.friction_angle)
                            self.plastic_volume[p]=history
                            self.plastic_strain[p]+=plastic
                        self.F[p]=F
                        self.J[p]=self.F_total[p].determinant()
        if ti.static(self.dynamic):
            self.center[None] += self.c.dt*self.velocity[None]
            self.angle[None] += self.c.dt*self.omega[None]
            R = rotation(self.angle[None])
            extent = ti.abs(R)@ti.Vector(self.half_size)
            self.center[None] = ti.max(lo+extent,ti.min(hi-extent,self.center[None]))

    @ti.kernel
    def body_walls(self):
        if ti.static(self.dynamic):
            lo = self.c.origin+self.c.boundary*self.c.dx
            hi = self.c.origin+(self.c.grid-self.c.boundary)*self.c.dx
            R = rotation(self.angle[None])
            for axis in ti.static(range(2)):
                for side in ti.static((-1,1)):
                    n=ti.Vector.zero(ti.f32,2)
                    n[axis]=-side
                    count=0
                    min_arm,max_arm=1e10,-1e10
                    for sx in ti.static((-1,1)):
                        for sy in ti.static((-1,1)):
                            arm=R@ti.Vector([sx*self.half_size[0],sy*self.half_size[1]])
                            point=self.center[None]+arm
                            touching=(side<0 and point[axis]<=lo+1e-6) or (side>0 and point[axis]>=hi-1e-6)
                            if touching:
                                a=cross(arm,n)
                                min_arm=ti.min(min_arm,a)
                                max_arm=ti.max(max_arm,a)
                                count+=1
                    solved=False
                    vn=n.dot(self.velocity[None])
                    if count>=2 and max_arm-min_arm>1e-8:
                        total=-self.body_mass*vn
                        first=(-self.inertia*self.omega[None]-max_arm*total)/(min_arm-max_arm)
                        second=total-first
                        if first>=0 and second>=0:
                            impulse=total*n
                            self.velocity[None]+=impulse/self.body_mass
                            self.omega[None]+=(min_arm*first+max_arm*second)/self.inertia
                            self.wall_impulse[None]+=impulse
                            solved=True
                    if count>0 and not solved:
                        # An inclined body normally has one active corner.
                        a=min_arm
                        if vn+self.omega[None]*max_arm<vn+self.omega[None]*min_arm:a=max_arm
                        speed=vn+self.omega[None]*a
                        if speed<0:
                            magnitude=-speed/(1./self.body_mass+a*a/self.inertia)
                            impulse=magnitude*n
                            self.velocity[None]+=impulse/self.body_mass
                            self.omega[None]+=a*magnitude/self.inertia
                            self.wall_impulse[None]+=impulse

    @ti.kernel
    def rebuild(self):
        for p in range(self.n):
            if self.material[p] == RIGID:
                self.x[p] = self.center[None]+rotation(self.angle[None])@self.local[p]
                self.broadcast(p)

    def step(self):
        self.predict()
        for _ in range(self.c.iterations):
            self.prepare()
            self.p2g()
            self.normalize()
            self.g2p()
            self.apply_feedback()
            if self.has_body:
                self.project_contact()
                self.apply_feedback()
                self.scatter_contact()
                self.gather_contact()
        self.integrate()
        self.body_walls()
        self.rebuild()
        self.check()
        self.step_index += 1
