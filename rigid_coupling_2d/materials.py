"""Planar shape/area targets and the chapter-two sand plasticity update."""
import taichi as ti


@ti.func
def regularized_inverse(F):
    U,s,V=ti.svd(F)
    inv=ti.Matrix.zero(ti.f32,2,2)
    for a in ti.static(range(2)):inv[a,a]=1./ti.max(s[a,a],.1)
    return V@inv@U.transpose()


@ti.func
def elastic_target(F,D,beta):
    I=ti.Matrix.identity(ti.f32,2)
    trial=(I+D)@F
    U,s,V=ti.svd(trial)
    shape=U@V.transpose()
    area=trial/ti.sqrt(ti.max(trial.determinant(),1e-6))
    return ((1-beta)*shape+beta*area)@regularized_inverse(F)-I


@ti.func
def sand_target(F,D,plastic_volume,beta):
    I=ti.Matrix.identity(ti.f32,2)
    trial=(I+D)@F
    U,s,V=ti.svd(trial)
    if plastic_volume==0:
        for a in ti.static(range(2)):s[a,a]=ti.min(ti.max(s[a,a],1.),1000.)
    shape=U@s@V.transpose()
    area=trial/ti.sqrt(ti.min(ti.max(trial.determinant(),.1),1.))
    return ((1-beta)*shape+beta*area)@regularized_inverse(F)-I


@ti.func
def sand_return(F,plastic_volume,beta,friction_angle):
    U,s,V=ti.svd(F)
    e=ti.Vector([ti.log(ti.max(s[0,0],1e-6)),ti.log(ti.max(s[1,1],1e-6))])
    original=e
    trace=e.sum()+plastic_volume
    deviator=e-.5*trace
    norm=deviator.norm()
    sin_phi=ti.sin(friction_angle*3.141592653589793/180.)
    friction=ti.sqrt(2./3.)*2.*sin_phi/(3.-sin_phi)
    history=0.
    if trace>=0:
        e=ti.Vector.zero(ti.f32,2)
        history=.5*trace
    else:
        delta=norm+(2.-beta)*trace*friction
        if delta>0 and norm>1e-8:e-=delta*deviator/norm
    for a in ti.static(range(2)):s[a,a]=ti.exp(ti.min(ti.max(e[a],-9.),9.))
    return U@s@V.transpose(),history,(e-original).norm()
