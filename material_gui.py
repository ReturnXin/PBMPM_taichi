"""
材质切换 + 参数调试面板 (Taichi GGUI)

交互:
  - 上方按钮: 切换当前编辑/查看的材质 (会立即 setup_scene 加载该材质的粒子)
  - 中间滑动条: 仅改本地暂存值, 不会立刻影响仿真
  - 底部 [Apply & Reset]: 把滑动条值写回 mpm.mat_params, 然后 setup_scene 重置

注: setup_scene 内部会调用 init_material_params, 它会把 mat_params 覆盖为硬编码默认值,
所以 Apply 顺序必须是 "先 setup_scene, 后写 pending 值"。

新增材质类型: 在 MATERIALS 和 PARAM_SCHEMA 里加项即可。
"""

MATERIALS: list[tuple[int, str]] = [
    (0, "Fluid 流体"),
    (1, "Elastic 弹性体"),
    (2, "Sand 沙土"),
]

# 每种材质暴露给用户调整的参数: (key, label, vmin, vmax)
PARAM_SCHEMA: dict[int, list[tuple[str, str, float, float]]] = {
    0: [
        ("rho",       "rho",       0.1, 5.0),
        ("viscosity", "viscosity", 0.0, 1.0),
        ("stiffness", "stiffness", 0.0, 5.0),
    ],
    1: [
        ("rho", "rho",       0.1, 5.0),
        ("E",   "Young's E", 1000.0, 2000000.0),
    ],
    2: [
        ("rho",                "rho",            0.1, 5.0),
        ("beta",               "beta",           0.0, 5.0),
        ("elastic_relaxation", "elastic relax",  0.0, 5.0),
        ("friction_angle",     "friction angle", 0.0, 60.0),
    ],
}


class MaterialPanel:
    def __init__(self, setup_scene_fn, mpm, current: int = 0):
        self.setup_scene_fn = setup_scene_fn
        self.mpm = mpm
        self.current = current
        self.pending: dict[str, float] = {}
        self._switch_to: int | None = None
        self._apply_now: bool = False
        self._restore_defaults: bool = False
        self._reload_pending_from_mpm()

    def _reload_pending_from_mpm(self):
        """从 mpm.mat_params[current] 把当前值拷回 pending (滑动条显示值)。"""
        self.pending = {
            key: float(getattr(self.mpm.mat_params[self.current], key))
            for key, *_ in PARAM_SCHEMA.get(self.current, [])
        }

    def draw(self, gui):
        with gui.sub_window("Material Tuner", 0.02, 0.02, 0.28, 0.55) as w:
            # --- 材质切换按钮 ---
            w.text("Material:")
            for idx, name in MATERIALS:
                marker = "[x] " if idx == self.current else "[ ] "
                if w.button(marker + name):
                    if idx != self.current:
                        self._switch_to = idx

            w.text("")
            w.text(f"Params ({self._name_of(self.current)}):")

            # --- 参数滑动条 (只改 pending) ---
            for key, label, vmin, vmax in PARAM_SCHEMA.get(self.current, []):
                cur = self.pending.get(key, 0.0)
                self.pending[key] = w.slider_float(label, cur, vmin, vmax)

            # --- 全局 (所有材质共用) ---
            w.text("")
            w.text("Global:")
            new_iter = w.slider_int("iterations", int(self.mpm.iteration), 3, 20)
            if new_iter != self.mpm.iteration:
                self.mpm.iteration = int(new_iter)

            # --- 应用 / 撤销 ---
            w.text("")
            dirty = self._is_dirty()
            apply_label = "Apply & Reset *" if dirty else "Apply & Reset"
            if w.button(apply_label):
                self._apply_now = True
            if w.button("Revert (discard edits)"):
                self._reload_pending_from_mpm()
            if w.button("Restore defaults"):
                self._restore_defaults = True

        # --- 在 sub_window 渲染后处理副作用, 避免 setup_scene 的 sleep/reset 打断 GUI ---
        if self._switch_to is not None:
            new_mat = self._switch_to
            self._switch_to = None
            self.setup_scene_fn(self.mpm, new_mat)
            self.current = new_mat
            self._reload_pending_from_mpm()

        if self._apply_now:
            self._apply_now = False
            applied = dict(self.pending)
            # 先 reset, 再写参数 (init_material_params 会覆盖默认值)
            self.setup_scene_fn(self.mpm, self.current)
            for key, val in applied.items():
                setattr(self.mpm.mat_params[self.current], key, val)
            self.pending = applied  # 保持滑动条值不变

        if self._restore_defaults:
            self._restore_defaults = False
            # setup_scene -> mpm.init -> init_material_params 会写入硬编码默认值
            self.setup_scene_fn(self.mpm, self.current)
            self._reload_pending_from_mpm()

    def _is_dirty(self) -> bool:
        for key, val in self.pending.items():
            cur = float(getattr(self.mpm.mat_params[self.current], key))
            if abs(cur - val) > 1e-6:
                return True
        return False

    @staticmethod
    def _name_of(idx: int) -> str:
        for i, name in MATERIALS:
            if i == idx:
                return name
        return f"#{idx}"
