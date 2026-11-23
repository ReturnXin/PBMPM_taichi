"""One container demo/recorder, with water and elastic material as scene choices."""
from dataclasses import asdict
from pathlib import Path
import csv
import json

import numpy as np
import taichi as ti

from .scenes import Scene, lattice
from .state import FLUID, ELASTIC, RIGID, particle_arrays, combine_particles
from .rigid import make_body, rotation_numpy
from .viewer import Viewer, PlaybackClock, PlaybackStats, box_edges, box_faces
from .mouse_water import WaterFollow


def build_shake_scene(config, material=FLUID, resolution=24, fluid_resolution=None, seed=42):
    if material not in (FLUID, ELASTIC):
        raise ValueError("shaking supports water or elastic material")
    resolution = resolution if material == ELASTIC or fluid_resolution is None else fluid_resolution
    if not isinstance(resolution, (int, np.integer)) or resolution < 3 or seed < 0:
        raise ValueError("resolution must be an integer >= 3 and seed nonnegative")
    center, half_size = np.array([.5, .48, .5]), np.array([.25, .27, .23])
    # Water occupies a shallow layer; the soft scene uses separate small objects.
    size = np.array([.43, .17, .38])
    fill_center = center.copy()
    fill_center[1] = center[1]-half_size[1]+size[1]/2+.008
    counts = np.maximum(3, np.ceil(resolution*size/size.max()).astype(int))
    x = lattice(fill_center, size, counts)
    x += np.random.default_rng(seed).uniform(-.35, .35, x.shape)*(size/counts)
    volume = float(np.prod(size))/len(x)
    density = config.fluid_density if material == FLUID else config.soft_density
    p = particle_arrays(x, material, volume, (0, 0, 0), config.dt,
                        (.16, .61, .93) if material == FLUID else (.67, .36, .87), mass=density*volume)
    groups, bodies = [], []
    if material == FLUID:
        p["color"][:, 1] += .12*(x[:, 1]-x[:, 1].min())/size[1]
        groups.append(p)
    else:
        rng = np.random.default_rng(seed)
        for location, soft_size, color in (
                ((.34, .30, .34), (.13, .14, .14), (.74, .38, .91)),
                ((.49, .30, .58), (.15, .13, .13), (.96, .57, .28)),
                ((.65, .30, .35), (.13, .14, .15), (.24, .72, .67))):
            soft_size = np.array(soft_size)
            soft_counts = np.maximum(4, np.ceil(resolution*soft_size/.43).astype(int))
            samples = lattice(location, soft_size, soft_counts)
            samples += rng.uniform(-.35, .35, samples.shape)*(soft_size/soft_counts)
            v0 = float(np.prod(soft_size))/len(samples)
            group = particle_arrays(samples, ELASTIC, v0, (0, 0, 0), config.dt, color,
                                    mass=config.soft_density*v0)
            band = np.floor((samples[:, 1]-location[1]+soft_size[1]/2)/soft_size[1]*5).astype(int)
            group["color"][band % 2 == 0] = np.minimum(np.array(color)+.14, 1)
            groups.append(group)
    surface = fill_center[1]+size[1]/2
    body_locations = ((.35, surface+.01, .36), (.51, surface+.01, .58), (.65, surface+.01, .37)) if material == FLUID else (
        (.34, .49, .35), (.50, .48, .57), (.65, .50, .35))
    for body_id, location in enumerate(body_locations):
        body_size = np.array([.09, .065, .09] if material == FLUID else [.095, .095, .085])
        local = lattice((0, 0, 0), body_size, 4)
        body_volume = float(np.prod(body_size))
        angle = (body_id-1)*.18
        quaternion = (np.cos(angle/2), 0, 0, np.sin(angle/2))
        body = make_body(local, location, config.rigid_density*body_volume/len(local), quaternion=quaternion)
        body.update(size=body_size.tolist(), volume=body_volume, density=config.rigid_density)
        groups.append(particle_arrays(body["center"]+body["local"]@rotation_numpy(body["quaternion"]).T,
                                      RIGID, body_volume/len(local), (0, 0, 0), config.dt, (.51, .81, .43),
                                      body_id=body_id, local=body["local"], mass=body["particle_mass"]))
        bodies.append(body)
    if material == FLUID:
        # Initial water does not occupy the submerged parts of the small floats.
        keep = np.ones(len(p["x"]), dtype=bool)
        for body in bodies:
            local_fluid = (p["x"]-body["center"])@rotation_numpy(body["quaternion"])
            keep &= ~np.all(np.abs(local_fluid) < np.array(body["size"])/2, axis=1)
        groups[0] = {key: value[keep] for key, value in p.items()}
    particles = combine_particles(groups)
    sampling = dict(material="water" if material == FLUID else "soft", count=len(particles["x"]),
                    continuum_count=int(np.count_nonzero(particles["body_id"] < 0)),
                    rigid_count=len(bodies), soft_object_count=3 if material == ELASTIC else 0,
                    seed=int(seed), mode="jittered")
    container = dict(center=center.tolist(), half_size=half_size.tolist(), prescribed=True)
    return Scene(particles, bodies, "Mouse-driven rigid box containing water or small elastic objects plus small rigid boxes",
                 sampling=sampling, container=container)


class MouseDrag:
    """Relative dragging avoids a jump when the user first clicks the viewport."""
    def __init__(self):
        self.anchor = None

    def update(self, cursor, pressed, motion):
        cursor = np.asarray(cursor, dtype=float)
        if not pressed:
            self.anchor = None
            return False
        if self.anchor is None:
            if not .04 < cursor[0] < .96 or not .20 < cursor[1] < .86:
                return False
            self.anchor = cursor.copy(), motion.target[[0, 2]]-motion.origin[[0, 2]], motion.target_angle
        start, horizontal, angle = self.anchor
        delta = cursor-start
        motion.automatic = False
        motion.set_target(horizontal+np.array([.65*delta[0], -.5*delta[1]]), angle+.7*delta[0])
        return True


@ti.data_oriented
class ShakeViewer(Viewer):
    def __init__(self, solvers, show_window=True, fps_limit=60):
        super().__init__(solvers, show_window, fps_limit)
        self.container = solvers[0].container
        self.water_control = solvers[0].mouse is not None
        self.follow = WaterFollow(solvers[0].mouse) if self.water_control else None
        self.drag = MouseDrag()
        self.camera.position(1.32, 1.05, 1.70)
        self.camera.lookat(.5, .41, .5)
        self.camera.fov(40)
        for state in self.render:
            state.cutaway[None] = 0
            material = state.solver.initial["material"]
            radii = np.clip(np.cbrt(state.solver.initial["rest_volume"])*.57, .003, .016).astype(np.float32)
            radii[material == FLUID] *= .55
            radii[material == RIGID] = 0
            state.base_radii.from_numpy(radii)
        half = np.asarray(self.container.half_size)
        self.local_line_field = ti.Vector.field(3, ti.f32, shape=24)
        self.local_line_field.from_numpy(box_edges(-half, half))
        self.stage = ti.Vector.field(3, ti.f32, shape=6)
        stage = box_faces([.03, .10, .03], [.97, .105, .97])[18:24]
        self.stage.from_numpy(stage)
        self.stage_normals = ti.Vector.field(3, ti.f32, shape=6)
        normal = np.cross(stage[1]-stage[0], stage[2]-stage[0])
        normal /= np.linalg.norm(normal)
        self.stage_normals.from_numpy(np.repeat(normal[None, :], 6, axis=0).astype(np.float32))
        from .rendering import ShakeRenderBuffer
        self.staging = ShakeRenderBuffer(self)
        self.geometry_ready = False

    def update_input(self, enabled=True, steps=1):
        cursor = self.window.get_cursor_pos()
        right = self.window.is_pressed(ti.ui.RMB)
        if right:
            self.camera.track_user_inputs(self.window, movement_speed=.008, hold_key=ti.ui.RMB)
        else:
            self.camera.last_mouse_x, self.camera.last_mouse_y = cursor
            self.camera.last_time = None
        pressed = enabled and self.window.is_pressed(ti.ui.LMB) and not right
        if self.water_control:
            self.follow.update(cursor, pressed and not self.container.motion.automatic, self.camera, steps=steps)
        else:
            self.drag.update(cursor, pressed, self.container.motion)

    def draw(self, paused, finished, fps=None, error=None, *, state_changed=True, speed=None):
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((.68, .68, .72))
        self.scene.point_light(pos=(.4, 1.8, 1.5), color=(.9, .86, 1.))
        self.scene.point_light(pos=(-1, .8, .4), color=(.28, .35, .5))
        self.staging.draw(self.scene, state_changed or not self.geometry_ready)
        self.geometry_ready = True
        self.canvas.set_background_color((.025, .033, .052))
        self.canvas.scene(self.scene)
        solver = self.solvers[0]
        material = "WATER + FLOATS" if solver.initial["material"][0] == FLUID else "SOFT + RIGID OBJECTS"
        gui = self.window.get_gui()
        title = " | MOUSE WATER" if self.water_control else " | SHAKE BOX"
        with gui.sub_window(material+title, .025, .025, .46, .15 if self.interactive else .12):
            gui.text(f"Time: {solver.step_index*solver.config.dt:.2f} s | Small rigid boxes: {solver.bodies.count}")
            mode = ("Auto shake" if self.container.motion.automatic else
                    "Left drag: force follows movement (X/Z)" if self.water_control else "Mouse control")
            gui.text("Paused" if paused else mode)
            if self.interactive:
                fps_text = "--" if fps is None else f"{fps:.1f}"
                gui.text(f"FPS: {fps_text} | Actual playback: {0. if speed is None else speed:.2f}x")
        if self.interactive:
            with gui.sub_window("Controls", .025, .89, .95, .09):
                control = "Left drag: push all water in drag direction" if self.water_control else "Left drag: shake + tilt"
                gui.text(f"{control} | Right drag: camera | A: auto shake | SPACE: pause | R: reset | ESC: close")
                if error:
                    gui.text("Stopped: invalid simulation state; R resets")
        return []


def run(name, configs, args):
    """The same view and fixed-step solver loop serve mouse playback and GIF capture."""
    from .solver import Solver
    from .runner import sample, source_hashes, save_state
    config = configs[0]
    recording = getattr(args, "gif_dir", None) is not None
    output = Path(args.gif_dir) if recording else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        gif_path = output/("water.gif" if name == "shake_water" else "soft.gif")
        if gif_path.exists():
            raise FileExistsError(f"GIF already exists: {gif_path}")
    backend = args.arch
    if backend == "auto":
        backend = "cuda" if ti._lib.core.with_cuda() else "vulkan"
    ti.set_logging_level(ti.WARN)
    ti.init(arch={"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[backend],
            enable_fallback=False, offline_cache=False, random_seed=args.seed,
            offline_cache_file_path=str(Path(__file__).resolve().parents[1]/"results"/"shake_box_cache"),
            cpu_max_num_threads=args.threads)
    viewer = None
    try:
        scene = build_shake_scene(config, FLUID if name == "shake_water" else ELASTIC,
                                  resolution=args.soft_resolution, fluid_resolution=args.fluid_resolution, seed=args.seed)
        solver = Solver(config, scene, mouse_control=name == "shake_water" and not recording)
        if backend == "cuda":
            solver.enable_cuda_graph()
        solver.step()
        solver.reset()
        viewer = ShakeViewer([solver], show_window=not recording, fps_limit=args.frames_per_second)
        motion = solver.container.motion
        motion.automatic = bool(getattr(args, "auto_shake", False) or recording)
        if recording:
            from PIL import Image
            rows, frames = [], []
            duration = args.duration if args.duration > 0 else 4.
            gif_fps = 24
            frame_count = int(round(duration*gif_fps))
            for frame in range(frame_count+1):
                target = int(round(frame/gif_fps/config.dt))
                solver.advance(target-solver.step_index)
                viewer.draw(False, False)
                rgba = np.clip(viewer.window.get_image_buffer_as_numpy()*255, 0, 255).astype(np.uint8)
                # Taichi's image buffer is W,H,RGBA with a lower-left origin.
                rgb = np.transpose(rgba, (1, 0, 2))[::-1, :, :3]
                frames.append(Image.fromarray(rgb).resize((960, 600), Image.Resampling.LANCZOS))
                p = solver.particles.snapshot()
                rows.append(dict(**sample(solver), **solver.container.metrics(p)))
                if frame % gif_fps == 0:
                    print(f"{name}: {frame/gif_fps:.1f}/{duration:g} s captured", flush=True)
            palette = frames[0].quantize(colors=256)
            frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                           duration=int(round(1000/gif_fps)), loop=0, optimize=False)
            prefix = "water" if name == "shake_water" else "soft"
            with (output/(prefix+"_metrics.csv")).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            save_state(solver, output, prefix+"_final")
            record = dict(scene=name, config=asdict(config), sampling=scene.sampling,
                          source_hashes=source_hashes(), backend=backend, physical_duration=duration,
                          frame_count=len(frames), fps=gif_fps, image_size=[960, 600],
                          motion="same bounded, damped translation and roll for both materials",
                          wall_motion_prescribed=True,
                          approximate_buoyancy_assist=config.buoyancy_assist,
                          max_sampled_wall_penetration=max(row["container_max_penetration"] for row in rows),
                          min_sampled_soft_volume=min(row.get("soft_min_volume_ratio", 1.) for row in rows))
            (output/(prefix+"_record.json")).write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            print(f"Saved {gif_path}", flush=True)
            return
        control = "left drag pushes all water in the drag direction (X/Z); no movement or release means no new mouse force" if viewer.water_control else "left drag shakes/tilts the box"
        print(f"Ready: {control}, right drag moves the camera, A toggles auto shake, SPACE pauses, R resets.", flush=True)
        # This hands-on demo starts playing so a mouse drag responds immediately.
        paused, error = False, None
        from time import perf_counter
        clock = PlaybackClock(config.dt, args.playback_speed, args.substeps)
        stats = PlaybackStats(config.dt)
        last = perf_counter()
        while viewer.window.running:
            now = perf_counter()
            elapsed, last = now-last, now
            starting_step = solver.step_index
            for event in viewer.window.get_events(ti.ui.PRESS):
                if event.key == ti.ui.ESCAPE:
                    viewer.window.running = False
                elif event.key == ti.ui.SPACE:
                    paused = not paused
                    clock.reset()
                    stats.reset()
                elif event.key == "r":
                    solver.reset()
                    viewer.drag.anchor = None
                    if viewer.water_control:
                        viewer.follow.reset()
                    paused, error = False, None
                    clock.reset()
                    stats.reset()
                    starting_step = solver.step_index
                elif event.key == "a":
                    if viewer.water_control:
                        viewer.follow.reset()
                    motion.automatic = not motion.automatic
                    if not motion.automatic:
                        motion.target = motion.center.copy()
                        motion.target_angle = motion.angle
            if not viewer.window.running:
                break
            count = clock.steps(elapsed, paused or error is not None)
            viewer.update_input(enabled=not paused and error is None, steps=count)
            if count:
                try:
                    solver.advance(count)
                except FloatingPointError as exc:
                    paused, error = True, str(exc)
                    print(error, flush=True)
            viewer.draw(paused, False, stats.fps, error, state_changed=bool(count),
                        speed=0. if paused or error is not None else stats.speed)
            viewer.window.show()
            stats.record(perf_counter()-now, solver.step_index-starting_step)
    finally:
        if viewer is not None:
            viewer.window.destroy()
        ti.reset()
