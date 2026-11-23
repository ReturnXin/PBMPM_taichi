"""Fill the physical box solid with unit occupancy, independently of water.

At grid node i the added occupancy is integral_outside_box N_i(x) dx / h^3.
Real water/rigid sample volumes supply the inside contribution. The box's
geometric term is cached by pose and does not participate in momentum.
"""
import taichi as ti

from .boundary_density import BoundaryDensity
from .state import FLUID, RIGID
from .transfers import valid_position


@ti.func
def spline_cdf(x):
    result = 0.
    if x >= 1.5:
        result = 1.
    elif x > .5:
        result = 1.-(1.5-x)**3/6.
    elif x > -.5:
        result = .5+.75*x-x**3/3.
    elif x > -1.5:
        result = (x+1.5)**3/6.
    return result


@ti.func
def spline_value(x):
    a = ti.abs(x)
    value = 0.
    if a < .5:
        value = .75-a*a
    elif a < 1.5:
        value = .5*(1.5-a)**2
    return value


@ti.data_oriented
class SolidBoundaryDensity(BoundaryDensity):
    def __init__(self, particles, transfers, container, config):
        super().__init__(particles, transfers, container, config)
        self.solid_volume = ti.field(ti.f32, shape=(config.grid,)*3)
        self.cached_center = ti.Vector.field(3, ti.f32, shape=())
        self.cached_orientation = ti.Matrix.field(3, 3, ti.f32, shape=())
        self.valid = ti.field(ti.i32, shape=())
        self.rebuild = ti.field(ti.i32, shape=())

    @ti.func
    def inside_y(self, world_x, node_y, R, center):
        # The container orientation is a roll about world Z. Integrate Y
        # analytically after intersecting the two rotated rectangle slabs.
        dx = world_x-center[0]
        lo, hi = -1e6, 1e6
        possible = True
        for a in ti.static(range(2)):
            slope = R[1, a]
            offset = R[0, a]*dx
            if ti.abs(slope)>1e-7:
                p = center[1]+(-self.box.half_size[a]-offset)/slope
                q = center[1]+(self.box.half_size[a]-offset)/slope
                lo, hi = ti.max(lo,ti.min(p,q)), ti.min(hi,ti.max(p,q))
            elif ti.abs(offset)>self.box.half_size[a]:
                possible = False
        fraction = 0.
        if possible and hi>lo:
            fraction = spline_cdf(hi*self.c.grid-node_y)-spline_cdf(lo*self.c.grid-node_y)
        return fraction

    @ti.func
    def occupancy(self, node):
        R, center = self.box.orientation[None],self.box.center[None]
        point = ti.cast(node,ti.f32)/self.c.grid
        local = R.transpose()@(point-center)
        extent = 1.5/self.c.grid*(ti.abs(R.transpose())@ti.Vector([1.,1.,1.]))
        gap = ti.Vector(self.box.half_size)-ti.abs(local)
        inside = 0.
        if (gap>=extent).all():
            inside = 1.
        elif (gap>-extent).all():
            if ti.abs(R[0,1])+ti.abs(R[1,0])<1e-7:
                inside = 1.
                for a in ti.static(range(3)):
                    lo = (center[a]-self.box.half_size[a])*self.c.grid-node[a]
                    hi = (center[a]+self.box.half_size[a])*self.c.grid-node[a]
                    inside *= spline_cdf(hi)-spline_cdf(lo)
            else:
                zlo = (center[2]-self.box.half_size[2])*self.c.grid-node[2]
                zhi = (center[2]+self.box.half_size[2])*self.c.grid-node[2]
                z = spline_cdf(zhi)-spline_cdf(zlo)
                # Six-point Gauss-Legendre per polynomial B-spline segment.
                # Only nodes whose support crosses the box boundary reach it.
                abscissa = ti.Vector([-.9324695142,-.6612093865,-.2386191861,
                                      .2386191861,.6612093865,.9324695142])
                weights = ti.Vector([.1713244924,.3607615730,.4679139346,
                                     .4679139346,.3607615730,.1713244924])
                xy = 0.
                for segment in ti.static(range(3)):
                    for j in ti.static(range(6)):
                        u = segment-1.+.5*abscissa[j]
                        world_x = (node[0]+u)/self.c.grid
                        xy += .5*weights[j]*spline_value(u)*self.inside_y(world_x,node[1],R,center)
                inside = xy*z
        return ti.max(0.,ti.min(1.,1.-inside))

    @property
    def build_kernels(self):
        return (self.check_pose, self.build_grid)

    def build(self):
        for kernel in self.build_kernels:
            kernel()

    @ti.kernel
    def check_pose(self):
        # A separate scalar pass avoids Taichi 1.7 CUDA misaligned accesses
        # observed when cache matrix reads/writes share the grid offload.
        self.rebuild[None] = ti.cast(self.valid[None]==0
            or (self.box.center[None]!=self.cached_center[None]).any()
            or (self.box.orientation[None]!=self.cached_orientation[None]).any(),ti.i32)
        self.cached_center[None] = self.box.center[None]
        self.cached_orientation[None] = self.box.orientation[None]
        self.valid[None] = 1

    @ti.kernel
    def build_grid(self):
        for node in ti.grouped(self.t.liquid_volume):
            if self.rebuild[None]:
                self.solid_volume[node] = self.occupancy(node)
            self.t.liquid_volume[node] = self.solid_volume[node]
        for p in range(self.active[None]):
            if self.p[p].material==FLUID or (ti.static(self.c.rigid_volume) and self.p[p].material==RIGID):
                if valid_position(self.p[p].x,self.c.grid):
                    self.scatter_volume(self.p[p].x,self.p[p].rest_volume)
                else:
                    ti.atomic_max(self.t.invalid[None],1)
