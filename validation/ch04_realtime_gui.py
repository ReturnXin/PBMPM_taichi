"""Measure displayed FPS using the unmodified production frame scheduler."""

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
from rigid_coupling.viewer import PlaybackClock, PlaybackStats

RESOLUTIONS = [24, 36, 48, 60]
WARM_SECONDS, MEASURE_SECONDS, REPEATS = 5., 20., 3
MIN_DISPLAY_FPS = 30


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
        experiment='chapter 4 production display frame rate',date='2026-11-11',
        status='planned_before_measurement',analysis_intent='descriptive capacity measurement',
        human_scientific_review='pending',resolutions=RESOLUTIONS,counts=counts,
        config=config().as_dict(),rigid_body_count=3,seed=42,
        production_scene='shake_water',container='fixed',camera='unchanged ShakeViewer default',
        window_size=[1280,800],vsync=False,display_fps_limit=120,
        dispatch='production CUDA step graph',mouse_control_allocated=True,mouse_force='zero',
        rendering='production ShakeViewer.draw and Window.show, visible window, no image readback in timed loop',
        warmup_wall_seconds=WARM_SECONDS,measure_wall_seconds=MEASURE_SECONDS,repeats=REPEATS,
        maximum_physical_steps_per_display_frame=2,playback_target_speed=1.,
        step_schedule='Production PlaybackClock(dt,1.0,2) schedules 0..2 fixed physical steps using real elapsed time, with the same bounded backlog as the main GUI',
        realtime_definition='Display FPS >= 30. Actual simulated/real time ratio is reported separately, as explicitly requested by the user',
        trial_summary='FPS = completed displayed frames / elapsed wall time; speed = committed physics steps * dt / elapsed wall time',
        aggregate='Arithmetic mean of three trial FPS and speed values; min/max retained. All three trials must reach 30 displayed FPS to mark real-time display.',
        repeats_start='Reset to the original scene before each trial, warm using displayed frames for at least 5 wall seconds, then measure at least 20 wall seconds',
        timing='Starts before the first measured frame, ends after Window.show of the final frame. Includes simulation, render staging, drawing, presentation, event checks, and small host bookkeeping. Compilation, reset, post-trial state snapshots and file writing excluded.',
        excluded_work='No optimization ablations or solver-only timing in this experiment.',
        failure_policy='Keep failures and partial records. Do not classify a stopped trial as a completed measurement.',
        supersedes='The user rejected a 16/17-steps-per-frame test and specified the production display FPS criterion. Use the existing main-loop PlaybackClock with max_steps=2, not a fixed simulated duration per displayed frame.',
        production_hashes=source_hashes(),runner_sha256=digest(Path(__file__)),baseline_gpu=gpu_state()))
    print(json.dumps(counts,indent=2),flush=True)


def frame(s,viewer,clock,stats,last):
    now=perf_counter()
    elapsed=now-last
    if not viewer.window.running:
        raise InterruptedError('Benchmark window closed by user')
    for event in viewer.window.get_events(ti.ui.PRESS):
        if event.key==ti.ui.ESCAPE:
            raise InterruptedError('Benchmark stopped by user')
    count=clock.steps(elapsed,False)
    viewer.update_input(enabled=False,steps=count)
    if count:
        s.advance(count)
    viewer.draw(False,False,fps=stats.fps,state_changed=bool(count),speed=stats.speed)
    viewer.window.show()
    stats.record(perf_counter()-now,count)
    return count,now


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
        viewer=ShakeViewer([s],show_window=True,fps_limit=120)
        frame(s,viewer,PlaybackClock(s.config.dt,1.,2),PlaybackStats(s.config.dt),perf_counter())
        for repeat in range(REPEATS):
            s.reset();index=0
            clock=PlaybackClock(s.config.dt,1.,2);stats=PlaybackStats(s.config.dt);last=perf_counter()
            print(f'resolution {resolution}, trial {repeat+1}: warming visible frames',flush=True)
            start=perf_counter()
            while perf_counter()-start<WARM_SECONDS:
                _,last=frame(s,viewer,clock,stats,last);index+=1
            # Preserve the production elapsed-time clock across the warmup boundary.
            index=0;frames=0;steps=0;partial=[];frame_start_step=s.step_index
            ti.sync();start=perf_counter()
            while perf_counter()-start<MEASURE_SECONDS:
                begin=perf_counter()
                count,last=frame(s,viewer,clock,stats,last)
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
                     realtime=frames/elapsed>=MIN_DISPLAY_FPS)
            results.append(row)
            with (folder/f'frames_trial{repeat+1}.csv').open('w',newline='',encoding='utf-8') as file:
                writer=csv.DictWriter(file,fieldnames=list(partial[0]));writer.writeheader();writer.writerows(partial)
            write_json(folder/'trials.json',results)
            np.savez_compressed(folder/f'final_trial{repeat+1}.npz',**particles,
                                **{'body_'+key:val for key,val in bodies.items()})
            print(f'resolution {resolution}, trial {repeat+1}: {row["fps"]:.2f} FPS, {row["simulation_speed"]:.3f}x',flush=True)
        # Save the rendered view outside all timed intervals.
        from PIL import Image
        viewer.draw(False,False,fps=results[-1]['fps'],state_changed=True,speed=results[-1]['simulation_speed'])
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
