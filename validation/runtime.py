"""Check CUDA graph parity on the production water/PBMPM/XPBMPM scenes.

Run: python -m validation.runtime --output results/a_new_validation_directory
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.runner import source_hashes, save_state
from rigid_coupling.shake import build_shake_scene
from rigid_coupling.solver import Solver
from rigid_coupling.state import ELASTIC, FLUID


def capture(s):
    return s.particles.snapshot(), s.bodies.data.to_numpy(), s.container.wall_impulse[None].to_numpy()


def exercise(s, steps=8):
    s.container.motion.set_target([.012, -.007], .025)
    if s.mouse is not None:
        s.mouse.set_force(.7, 0., -.3, 5)
    s.advance(steps)
    if s.mouse is not None:
        s.mouse.release()
        np.testing.assert_array_equal(s.mouse.control[None].to_numpy(), 0.)
    return capture(s)


def compare(a, b, c, *, accumulated=False):
    errors = {}
    for name in ('x', 'd', 'D', 'F', 'volume_ratio', 'multiplier'):
        if name in a[0]:
            errors[name] = float(np.max(np.abs(a[0][name]-b[0][name])))
            scale = c.compliance/c.dt**2 if name == 'multiplier' and accumulated else 1.
            # Over multiple SVD/contact iterations compare the actual term
            # alpha/dt^2 * Lambda in the compliant equation. Single-step
            # replay below still checks Lambda itself with the strict bound.
            np.testing.assert_allclose(scale*a[0][name], scale*b[0][name],
                                       atol=4e-6, rtol=4e-5, err_msg=name)
    for name in ('x', 'q', 'v', 'omega'):
        errors['body_'+name] = float(np.max(np.abs(a[1][name]-b[1][name])))
    np.testing.assert_allclose(a[1]['x'], b[1]['x'], atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(a[1]['q'], b[1]['q'], atol=3e-6, rtol=3e-5)
    assert c.dt*errors['body_v'] < 1e-7, errors
    assert c.dt*errors['body_omega'] < 1e-6, errors
    errors['wall_impulse'] = float(np.max(np.abs(a[2]-b[2])))
    np.testing.assert_allclose(a[2], b[2], atol=3e-6, rtol=3e-4)
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    ti.init(arch=ti.cuda, enable_fallback=False, offline_cache=False, kernel_profiler=False)
    common = dict(viscosity=0., rigid_contact=True, world_contact=True, grid_coupling='separate')
    cases = [
        ('xpbmpm', ELASTIC, Config(**common, elastic_solver='xpbmpm', elastic_model='legacy',
                                  compliance=1e-6, iterations=10, rigid_density=.7)),
        ('water', FLUID, Config(**common, boundary_density='available', wall_transfer='reflected',
                                density_feedback='clamped', fluid_relaxation=1., rigid_density=.35,
                                rigid_volume=True, buoyancy_assist=True)),
        ('pbmpm_soft', ELASTIC, Config(**common, elastic_relaxation=.04, rigid_density=.7)),
    ]
    result = dict(source_hashes=source_hashes(), cases={})
    try:
        for name, material, c in cases:
            print('validating', name, flush=True)
            scene = build_shake_scene(c, material, resolution=12, fluid_resolution=24, seed=42)
            s = Solver(c, scene, mouse_control=material == FLUID)
            initial = s.particles.snapshot()
            ordinary = exercise(s)
            s.reset()
            for key in initial:
                np.testing.assert_array_equal(s.particles.snapshot()[key], initial[key])
            s.enable_cuda_graph()
            graphed = exercise(s)
            errors = compare(ordinary, graphed, c, accumulated=True)
            checkpoint = capture(s)
            motion, index = deepcopy(s.container.motion.__dict__), s.step_index
            s.advance(1)
            next_graph = capture(s)
            s.particles.restore(checkpoint[0])
            s.bodies.data.from_numpy(checkpoint[1])
            s.container.wall_impulse[None] = checkpoint[2]
            s.container.motion.__dict__.update(motion)
            s.container.publish()
            s.step_index = index
            graph, s.step_graph = s.step_graph, None
            s.advance(1)
            single_step = compare(next_graph, capture(s), c)
            s.step_graph = graph
            # Exercise sustained moving contact after the short parity comparison.
            s.container.motion.automatic = True
            s.container.motion.time = 1.
            s.advance(100)
            if material == FLUID:
                assert s.particles.multiplier is None
            s.reset()
            if s.particles.multiplier is not None:
                np.testing.assert_array_equal(s.particles.multiplier.to_numpy(), 0.)
            # Two steps per host call, warmed and without diagnostic snapshots.
            s.advance(20)
            ti.sync()
            start = perf_counter()
            for _ in range(20):
                s.advance(2)
            ti.sync()
            elapsed = (perf_counter()-start)*1000/40
            result['cases'][name] = dict(config=c.as_dict(), particle_count=s.particles.n,
                                        parity_max_abs=errors, warm_ms_per_step=elapsed,
                                        single_step_max_abs=single_step,
                                        multiplier_allocated=s.particles.multiplier is not None,
                                        sustained_steps=100, passed=True)
            save_state(s, args.output, name)
            (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
            print(name, result['cases'][name], flush=True)
    finally:
        ti.reset()


if __name__ == '__main__':
    main()
