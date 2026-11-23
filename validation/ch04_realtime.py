"""Measure rendered real-time capacity at four particle counts."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import taichi as ti

from rigid_coupling.config import Config
from rigid_coupling.runner import source_hashes, write_json
from rigid_coupling.shake import ShakeViewer, build_shake_scene
from rigid_coupling.solver import Solver
from rigid_coupling.state import FLUID

RESOLUTIONS = [24, 36, 48, 60]
WARM_SECONDS, MEASURE_SECONDS, REPEATS = 5., 20., 3
SIMULATION_FPS = 30


def config():
    return Config(grid=32, dt=.002, iterations=5, gravity=-9.8,
                  fluid_relaxation=1., viscosity=0., rigid_density=.35,
                  rigid_volume=True, rigid_contact=True, world_contact=True,
                  buoyancy_assist=True, grid_coupling='separate',
                  boundary_density='available', wall_transfer='reflected',
                  density_feedback='clamped')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_state():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,utilization.gpu,pstate,temperature.gpu,power.draw',
                               '--format=csv'], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)


def make_scene(resolution):
    return build_shake_scene(config(), FLUID, fluid_resolution=resolution, seed=42)


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    snap=out/'source_snapshot'
    shutil.copytree(root/'rigid_coupling',snap/'rigid_coupling',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (snap/'validation').mkdir()
    (snap/'validation/__init__.py').write_text('',encoding='utf-8')
    shutil.copy2(__file__,snap/'validation'/Path(__file__).name)
    counts={}
    for resolution in RESOLUTIONS:
        s=make_scene(resolution)
        counts[resolution]=dict(water=int((s.particles['material']==FLUID).sum()),
                                rigid_samples=int((s.particles['material']!=FLUID).sum()),
                                total=len(s.particles['x']))
    write_json(out/'protocol.json',dict(
        experiment='chapter 4 rendered real-time capacity',date='2026-11-11',
        status='planned_before_measurement',analysis_intent='descriptive capacity measurement',
        human_scientific_review='pending',resolutions=RESOLUTIONS,counts=counts,
        config=config().as_dict(),rigid_body_count=3,seed=42,
        production_scene='shake_water',container='fixed',camera='unchanged ShakeViewer default',
        window_size=[1280,800],vsync=False,display_fps_limit=1000,
        dispatch='production CUDA step graph',mouse_control_allocated=True,mouse_force='zero',
        rendering='production ShakeViewer.draw and Window.show, visible window, no image readback in timed loop',
        warmup_wall_seconds=WARM_SECONDS,measure_wall_seconds=MEASURE_SECONDS,repeats=REPEATS,
        simulation_seconds_per_display_frame=1/SIMULATION_FPS,
        step_schedule='round(frame_index/(30*dt)), giving 17,16,17 physical steps per 3 displayed frames; dt and iterations unchanged',
        realtime_definition='FPS >= 30, corresponding to a simulated/real time ratio >= 1 with this fixed simulation-time frame schedule',
        trial_summary='FPS = completed displayed frames / elapsed wall time; speed = committed physics steps * dt / elapsed wall time',
        aggregate='Arithmetic mean of the three trial FPS and speed values; min/max retained. All three trials must reach 30 FPS to mark sustained real-time.',
        repeats_start='Reset to the original scene before each trial, warm using displayed frames for at least 5 wall seconds, then measure at least 20 wall seconds',
        timing='Starts before the first measured frame, ends after Window.show of the final frame. Includes simulation, render staging, drawing, presentation, event checks, and small host bookkeeping. Compilation, reset, post-trial state snapshots and file writing excluded.',
        excluded_work='No optimization ablations or solver-only timing in this experiment.',
        failure_policy='Keep failures and partial records. Do not classify a stopped trial as a completed measurement.',
        supersedes='Earlier solver-only performance draft was replaced at the user request; its measured values are not used as displayed FPS.',
        production_hashes=source_hashes(),runner_sha256=digest(Path(__file__)),baseline_gpu=gpu_state()))
    print(json.dumps(counts,indent=2),flush=True)


def frame(s,viewer,index,fps=None,speed=None):
    if not viewer.window.running:
        raise InterruptedError('Benchmark window closed by user')
    for event in viewer.window.get_events(ti.ui.PRESS):
        if event.key==ti.ui.ESCAPE:
            raise InterruptedError('Benchmark stopped by user')
    before=round(index/(SIMULATION_FPS*s.config.dt))
    after=round((index+1)/(SIMULATION_FPS*s.config.dt))
    steps=after-before
    s.advance(steps)
    viewer.draw(False,False,fps=fps,state_changed=True,speed=speed)
    viewer.window.show()
    return steps


def run(out,resolution):
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert protocol['production_hashes']==source_hashes()
    assert protocol['runner_sha256']==digest(Path(__file__))
    folder=out/f'resolution_{resolution}';folder.mkdir(exist_ok=False)
    write_json(folder/'environment.json',dict(python=sys.version,numpy=np.__version__,taichi=ti.__version__,
               platform=platform.platform(),logical_processors=os.cpu_count(),gpu_before=gpu_state()))
    ti.init(arch=ti.cuda,enable_fallback=False,offline_cache=False,kernel_profiler=False,
            random_seed=42,cpu_max_num_threads=4,
            offline_cache_file_path=str(out.resolve()/'unused_cache'))
    viewer=None;results=[];partial=[]
    try:
        print(f'resolution {resolution}: compiling production solver',flush=True)
        s=Solver(config(),make_scene(resolution),mouse_control=True)
        s.enable_cuda_graph();s.advance(2);s.reset();ti.sync()
        viewer=ShakeViewer([s],show_window=True,fps_limit=1000)
        frame(s,viewer,0)
        for repeat in range(REPEATS):
            s.reset();index=0
            print(f'resolution {resolution}, trial {repeat+1}: warming visible frames',flush=True)
            start=perf_counter()
            while perf_counter()-start<WARM_SECONDS:
                frame(s,viewer,index);index+=1
            # Start measured physical-frame scheduling at a fresh triplet boundary.
            index=0;frames=0;steps=0;partial=[];frame_start_step=s.step_index
            ti.sync();start=perf_counter()
            while perf_counter()-start<MEASURE_SECONDS:
                begin=perf_counter()
                count=frame(s,viewer,index)
                end=perf_counter()
                partial.append(dict(frame=index,wall_start_s=begin-start,wall_end_s=end-start,
                                    wall_duration_s=end-begin,physical_steps=count))
                index+=1;frames+=1;steps+=count
            elapsed=perf_counter()-start
            assert s.step_index-frame_start_step==steps
            particles=s.particles.snapshot()
            bodies=s.bodies.data.to_numpy()
            finite=all(np.isfinite(v).all() for group in [particles,bodies] for v in group.values())
            assert finite
            row=dict(trial=repeat+1,frames=frames,physical_steps=steps,wall_seconds=elapsed,
                     simulation_seconds=steps*s.config.dt,fps=frames/elapsed,
                     simulation_speed=steps*s.config.dt/elapsed,finite=finite,
                     realtime=frames/elapsed>=SIMULATION_FPS)
            results.append(row)
            with (folder/f'frames_trial{repeat+1}.csv').open('w',newline='',encoding='utf-8') as file:
                writer=csv.DictWriter(file,fieldnames=list(partial[0]));writer.writeheader();writer.writerows(partial)
            write_json(folder/'trials.json',results)
            np.savez_compressed(folder/f'final_trial{repeat+1}.npz',**particles,
                                **{'body_'+key:val for key,val in bodies.items()})
            print(f'resolution {resolution}, trial {repeat+1}: {row["fps"]:.2f} FPS, {row["simulation_speed"]:.3f}x',flush=True)
        # Save the rendered view outside all timed intervals.
        from PIL import Image
        viewer.draw(False,False,state_changed=True)
        pixels=viewer.window.get_image_buffer_as_numpy()
        rgb=np.transpose(np.clip(pixels[:,:,:3]*255,0,255).astype(np.uint8),(1,0,2))[::-1]
        Image.fromarray(rgb).save(folder/'view.png')
        summary=dict(completed=True,resolution=resolution,counts=protocol['counts'][str(resolution)],
                     trials=results,mean_fps=float(np.mean([r['fps'] for r in results])),
                     min_fps=min(r['fps'] for r in results),max_fps=max(r['fps'] for r in results),
                     mean_simulation_speed=float(np.mean([r['simulation_speed'] for r in results])),
                     realtime_all_trials=all(r['realtime'] for r in results),
                     source_unchanged=protocol['production_hashes']==source_hashes(),gpu_after=gpu_state())
        write_json(folder/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    except Exception:
        write_json(folder/'failure.json',dict(completed=False,completed_trials=results,
                   partial_measured_frames=partial,traceback=traceback.format_exc()))
        raise
    finally:
        if viewer is not None:
            viewer.window.destroy()
        ti.reset()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','run'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolution',type=int,choices=RESOLUTIONS,default=24)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output)
    else:run(args.output,args.resolution)


if __name__=='__main__':main()
