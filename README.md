# PBMPM 水体、软体与刚体

当前主代码为 `rigid_coupling/`，统一入口为 `main.py`。水体边界改进、接触仿射修正、缓存、CUDA 图、反力分块归约和 XPBMPM 软体材料都进入同一个 Solver；`python -m rigid_coupling` 是同一入口的兼容写法。

## 运行

环境使用项目 `.venv`，依赖见 `requirements.txt`，Taichi 版本为 1.7.4。

```powershell
# 已验证的水面边界组合与性能优化
.\.venv\Scripts\python.exe main.py --arch cuda --wall-transfer reflected

# XPBMPM 软体与小刚体：顺应度不随 dt 手动缩放
.\.venv\Scripts\python.exe main.py --scene shake_soft --arch cuda --elastic-solver xpbmpm --compliance 0.000001 --iterations 10

# 原 PBMPM 软体对照
.\.venv\Scripts\python.exe main.py --scene shake_soft --arch cuda --elastic-solver pbmpm

# 全部选项
.\.venv\Scripts\python.exe main.py --help
```

水体左键拖动施加水平加速度；软体左键拖动摇晃盒子。右键调相机，`Space` 暂停，`R` 复位，`A` 切换自动摇晃。默认每帧最多 2 个物理步；显示帧率与物理播放速度分别统计。

默认参数继续保留 raw/clamp 水体及 PBMPM 软体，避免隐式改变已有对照。`--wall-transfer reflected` 自动选择 available、clamped 与 fluid_relaxation=1；`--elastic-solver xpbmpm` 默认选择前期章节验证的 legacy 材料模型。使用 `--elastic-model experiment` 可选形状/体积混合目标。两种材料求解器的 dt 和迭代数都由命令行明确指定，比较时应使用相同模型和场景。

水体演示的浮力辅助现在只帮助底部的轻刚体脱困：刚体下沿位于盒底一层网格范围内、实际密度小于水且有浸没时，才补足缺失的向上速度增量。离开底部区域或漂在水面时使用正常耦合结果。仍可用 `--no-buoyancy-assist` 完全关闭这个展示近似，详见 [底部浮力辅助](../../毕业论文/实验优化/13_底部触发浮力辅助.md)。

## 二维多材质下落实验

圆形材料团从刚体右上方静止下落，刚体直接放在盒底；两者均受重力，方框四壁参与碰撞。下落材料 4,064 粒子，刚体 4,608 采样粒子。

~~~powershell
.\.venv\Scripts\python.exe -m rigid_coupling_2d --scene falling --material water
.\.venv\Scripts\python.exe -m rigid_coupling_2d --scene falling --material elastic --compliance 5e-07
.\.venv\Scripts\python.exe -m rigid_coupling_2d --scene falling --material sand
~~~

默认二维场景为 falling，0.9 s 后暂停。Space 暂停，R 重放，Esc 关闭。数据、三种 GIF、参数和复测命令见[多材质下落记录](../../毕业论文/实验结果/第四章/多材质下落/README.md)。

## 刚体下落实验

圆形水体、弹性体或沙体局部放在盒底，矩形刚体从右上方静止落下，两者均受重力。

~~~powershell
.\.venv\Scripts\python.exe -m rigid_coupling_2d --scene rigid_drop --material water
.\.venv\Scripts\python.exe -m rigid_coupling_2d --scene rigid_drop --material elastic
.\.venv\Scripts\python.exe -m rigid_coupling_2d --scene rigid_drop --material sand
~~~

0.9 s 后暂停，Space 暂停、R 重放、Esc 关闭。[实验记录、数据及 GIF](../../毕业论文/实验结果/第四章/刚体下落/README.md)。

## 保留的悬臂对比实验

悬臂场景和 PBMPM/XPBMPM 对比通过第三章入口运行。在本项目目录分别执行：

```powershell
.\.venv\Scripts\python.exe -m experiments.ch03_xpbmpm --arch cuda --gui --elastic-model legacy --solver pbmpm --relaxation 0.192 --dt 0.002 --iterations 10 --duration 3 --repeats 1
.\.venv\Scripts\python.exe -m experiments.ch03_xpbmpm --arch cuda --gui --elastic-model legacy --solver xpbmpm --compliance 0.000001 --dt 0.002 --iterations 10 --duration 3 --repeats 1
```

窗口在预热、预跑及计时后打开；去掉 `--gui` 可仅保存数据。输出默认写入“毕业论文/实验结果/第三章”下的时间戳目录。[原三维悬臂的 18 组曲线、统计及六张形态/响应图](../../毕业论文/实验结果/第三章/悬臂对比/README.md)。

## 代码与验证

- `rigid_coupling/`：主数值实现、配置、当前演示与绘制。
- `tests/`：单元和回归测试；`test_xpbmpm_coupling.py` 比较迁移后的材料核与原章节实现。
- `validation/runtime.py`：直接调用主求解器，检查水体、PBMPM/XPBMPM 软体的普通执行/CUDA 图一致性、移动接触和复位。
- `pbmpm/`、`experiments/`：保留论文章节和 Morton 对照，不是主程序运行依赖，详见 [章节实验](EXPERIMENTS.md)。
- [毕业论文/实验结果](../../毕业论文/实验结果/README.md)：按章节保存正文采用的原始数据、图表及复现说明。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_xpbmpm_coupling.py -v
.\.venv\Scripts\python.exe -m validation.runtime --output results/new_runtime_check
```

第二条命令需要 CUDA，并要求输出目录不存在。它是正确性冒烟检查，输出的单批耗时不代替严谨的性能对照。

分项公式、实验条件、数据来源、限制及清理清单见 [毕业论文／实验优化](../../毕业论文/实验优化/00_优化总览与使用方法.md)。正文性能数据见[第四章实时性能](../../毕业论文/实验结果/第四章/实时性能/README.md)。

原 `main.py` 的独立 GUI 已从主入口移除。旧 `mpm_pbd.py` 仍由历史一致性测试使用。
