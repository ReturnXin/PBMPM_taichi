import taichi as ti

__all__ = ["MaterialParam", "Obstacle"]

MaterialParam = ti.types.struct(
    rho=ti.f32,
    viscosity=ti.f32,
    stiffness=ti.f32,
    E=ti.f32,
    beta=ti.f32,
    elastic_relaxation=ti.f32,
    friction_angle=ti.f32,
    color=ti.types.vector(3, ti.f32),
)

Obstacle = ti.types.struct(
    type=ti.types.int32,
    center=ti.types.vector(3, ti.f32),
    radius=ti.types.f32,
    size=ti.types.vector(3, ti.f32),
    color=ti.types.vector(3, ti.f32),
    velocity=ti.types.vector(3, ti.f32),
)
