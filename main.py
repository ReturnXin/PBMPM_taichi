import numpy as np
import time
import argparse
import taichi as ti

from mpm_pbd import MpmPBDSolver

ti.init(arch=ti.cuda, kernel_profiler=True)

n_grid, steps, dt = 32, 25, 4e-4
# region === Parameters ===
shake_strength = 0.0
is_add_fluid = False
move_speed = 0.01
hide_obstacles = True
# endregion

window = ti.ui.Window("MPM3D", (720, 720), vsync=True)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((0.1, 0.1, 0.1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()


def setup_scene(mpm):
    mpm.reset()
    # mpm.add_ball_obstacles(center=[0.5, 0.125, 0.5], radius=0.15, color=[0.6, 0.4, 0.8])
    # mpm.add_cube(
    #     particle_num=2**16,
    #     center=[0.4, 0.5, 0.4],
    #     cube_size=[0.5, 0.5, 0.5],
    #     color=[0.1, 0.4, 0.8],
    #     material=0,
    #     radius=0.005,
    # )
    mpm.add_cube(
        particle_num=2**16,
        center=[0.4, 0.5, 0.4],
        cube_size=[0.3, 0.3, 0.3],
        color=[0.95, 0.4, 0.4],
        material=1,
        radius=0.008,
    )
    # mpm.add_cube(
    #     particle_num=2**13,
    #     center=[0.4, 0.25, 0.4],
    #     cube_size=[0.5, 0.5, 0.5],
    #     color=[0.85, 0.75, 0.55],
    #     material=2,
    #     radius=0.006,
    # )
    mpm.init(hide_obstacles)
    time.sleep(1)


# region === process input function ===
last_mouse_x = 0.0
last_mouse_y = 0.0


def process_mouse_action(mpm, shake_strength):
    global last_mouse_x
    global last_mouse_y
    mouse_x, mouse_y = window.get_cursor_pos()
    if window.is_pressed(ti.ui.LMB):
        dx = mouse_x - last_mouse_x
        dy = mouse_y - last_mouse_y
        interia = ti.Vector([0.0, 0.0, 0.0])
        if abs(dx) > 0.001 and abs(dy) > 0.001:
            interia = ti.Vector([dx * shake_strength, 0.0, -dy * shake_strength])
        mpm.apply_interia(interia)
        last_mouse_x = mouse_x
        last_mouse_y = mouse_y


def process_key_action(mpm, move_speed, hide_obstacles):
    # if window.is_pressed(ti.ui.SPACE):
    #     is_add_fluid = not is_add_fluid

    dx, dy, dz = 0.0, 0.0, 0.0
    if window.is_pressed("i"):
        dz -= move_speed
    if window.is_pressed("k"):
        dz += move_speed
    if window.is_pressed("j"):
        dx -= move_speed
    if window.is_pressed("l"):
        dx += move_speed
    if window.is_pressed("u"):
        dy += move_speed
    if window.is_pressed("o"):
        dy -= move_speed
    if dx != 0 or dy != 0 or dz != 0:
        mpm.move_obstacle(0, [dx, dy, dz])  # 移动索引为0的障碍物
        if not hide_obstacles:
            mpm.update_mesh_vertices()  # 更新 Mesh


# endregion


def main(args):
    mpm = MpmPBDSolver()
    # mpm.generate_lines_vertex()

    camera.position(0.5, 1.0, 2.0)
    camera.lookat(0.5, 0.3, 0.5)

    # region === scene setting ===

    scene.ambient_light((0.5, 0.5, 0.5))
    # mpm.add_box_obstacles(center=[0.5, 0.125, 0.5], size=[0.25, 0.25, 0.25], color=[0.6, 0.4, 0.8])
    scene.set_camera(camera)
    # endregion
    start_frame = 50
    end_frame = 200

    use_optimization = args.opt
    if use_optimization:
        mpm.shrink_factor = 3.1
        mpm.use_morton_code = True
        mpm.use_dynamic_grid = False
    else:
        mpm.use_morton_code = False
        mpm.use_dynamic_grid = False
    setup_scene(mpm)

    r_key_pressed = False
    test_count = 0
    # mpm.substep()
    while window.running:
        # camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)

        # region === process input ===
        process_mouse_action(mpm, shake_strength)
        process_key_action(mpm, move_speed, hide_obstacles)
        if window.is_pressed("r"):
            if not r_key_pressed:
                setup_scene(mpm)
                r_key_pressed = True
        else:
            r_key_pressed = False
        # endregion
        # region === Render Scene ===
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        if is_add_fluid:
            mpm.add_continum_particle(material=0)

        if not hide_obstacles:
            if mpm.mesh_vertices is not None:
                scene.mesh(mpm.mesh_vertices, indices=mpm.mesh_indices, per_vertex_color=mpm.mesh_colors)

        scene.particles(mpm.x, radius=0.005, per_vertex_radius=mpm.radius, per_vertex_color=mpm.color)
        # endregion

        # region === Print Information ===
        gui.text(f"min:({mpm.grid_min[0]},{mpm.grid_min[1]},{mpm.grid_min[2]})")
        gui.text(f"max:({mpm.grid_max[0]},{mpm.grid_max[1]},{mpm.grid_max[2]})")
        # endregion

        # scene.lines(mpm.grid_lines_vertex, width=1.0, color=(0.3, 0.3, 0.3))

        canvas.scene(scene)
        window.show()
        ti.profiler.clear_kernel_profiler_info()

        # region === Profiler ===
        current_f = mpm.fps_count[None]

        if current_f == start_frame:
            ti.sync()
            start_time = time.time()
            print(f">>> Start profiling from frame {start_frame}...")

        ti.profiler.clear_kernel_profiler_info()
        mpm.substep()

        if current_f == end_frame:
            test_count += 1
            ti.sync()
            end_time = time.time()

            total_time = end_time - start_time
            num_frames = end_frame - start_frame + 1
            avg_fps = num_frames / total_time

            print(f"===== Profiling Report =====")
            ti.profiler.print_kernel_profiler_info(mode="trace")
            print(f"Total time for {num_frames} frames: {total_time:.4f} s")
            print(f"Average FPS: {avg_fps:.2f}")
            # if test_count == 1:
            #     setup_scene(mpm, is_test=True)

            # break
        # endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", action="store_true")
    args = parser.parse_args()
    main(args)
