import taichi as ti


class Viewer:
    def __init__(self, title, show_window=True):
        self.interactive = show_window
        self.window = ti.ui.Window(title, (1000, 720), vsync=True, show_window=show_window)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        if title == "xpbmpm":
            self.camera.position(0.95, 0.85, 1.45)
            self.camera.lookat(0.5, 0.52, 0.5)
        else:
            self.camera.position(1.6, 1.2, 1.8)
            self.camera.lookat(0.5, 0.5, 0.5)

    def draw(self, solver, capture_path=None):
        if not self.window.running:
            return
        if self.interactive:
            self.camera.track_user_inputs(self.window, movement_speed=0.01, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.65, 0.65, 0.65))
        self.scene.point_light(pos=(1, 2, 1), color=(1, 1, 1))
        self.scene.particles(solver.particles.data.x, radius=0.004, per_vertex_color=solver.particles.data.color)
        self.canvas.set_background_color((0.08, 0.09, 0.12))
        self.canvas.scene(self.scene)
        if capture_path is not None:
            self.window.save_image(str(capture_path))
        else:
            self.window.show()
