# PBMPM Taichi 仓库指南

**项目简介**: Position-Based Material Point Method (PBMPM) 流体仿真引擎，使用Taichi高性能GPU计算框架实现。支持流体、弹性体和沙子三种材料的物理仿真及刚体碰撞。

## 架构概览

### 核心组件

- **MpmPBDSolver** (mpm_pbd.py): 主仿真求解器类
  - 粒子系统: 最多30万个粒子，存储位置、位移、变形梯度等物理量
  - 网格系统: 3D均匀网格用于邻域搜索和力计算 (64x64x64)
  - 材料系统: 3种材料类型（流体:0、果冻:1、沙子:2），每种有独立参数
  - 障碍物系统: 支持球体和立方体障碍物，支持动态移动和碰撞检测

- **utils_renderer.py**: 渲染辅助工具
  - 几何体生成: 单位立方体网格、球面网格、线框体
  - 用于Taichi UI可视化

- **morton_code.py**: 空间索引
  - Morton码计算用于高效的3D网格哈希和邻域搜索

### 数据流

`
初始化场景 → 粒子初始化 → 仿真循环:
  [读取输入] → [P2G传输] → [网格更新] → [G2P传输] → [约束求解] → [碰撞检测] → [渲染]
`

## 关键开发工作流

### 运行仿真

`ash
# 交互式3D演示 (main.py/test.py/demo.py 类似)
python main.py

# 性能基准测试 (比较优化/未优化版本)
python bench_marks.py
`

### GPU初始化

所有脚本必须先初始化Taichi:
`python
ti.init(arch=ti.cuda, kernel_profiler=True)  # 使用CUDA + 内核分析
`

### 场景设置模式

三个演示脚本在setup_scene()中展示不同场景:
- **类型0**: 流体(水)与球体碰撞
- **类型1**: 果冻(弹性体)
- **类型2**: 沙子(粒状材料)

## 项目特定的模式和约定

### 1. 材料参数定义

所有材料通过MaterialParam结构体定义:
- 
ho: 密度 (通常=1.0)
- iscosity: 粘度 (流体特有)
- stiffness: 刚度 (弹性体特有)
- riction_angle: 摩擦角 (沙子特有)
- 在MpmPBDSolver.__init__()中初始化3种材料的参数

### 2. 粒子生成

使用dd_cube()方法添加粒子:
`python
mpm.add_cube(
    particle_num=2**16,          # 粒子数 (通常2的幂)
    center=[0.4, 0.5, 0.4],      # 中心位置
    cube_size=[0.5, 0.5, 0.5],   # 立方体尺寸
    color=[0.1, 0.4, 0.8],       # RGB颜色 (0-1范围)
    material=0,                  # 材料类型 (0/1/2)
    radius=0.005                 # 粒子显示半径
)
`

### 3. 碰撞系统

- **障碍物类型**: type=0为球体，type=1为立方体
- **SDF碰撞**: 使用sdf_box()和内置球体检测
- **约束求解**: 每帧迭代次数(self.iteration)与精度成正相关

### 4. 优化策略

仓库包含两个版本分支:
- **未优化** (没有优化/): 基础算法实现
- **优化** (优化/): 性能改进版本

基准测试脚本自动测试两个版本，使用--opt标志区分。

## 常见编辑点

1. **调整仿真参数**:
   - self.dt: 时间步 (当前8e-3，较大值=更快但不稳定)
   - self.iteration: 约束求解迭代次数
   - self.gravity: 重力加速度

2. **修改材料行为**:
   - 编辑__init__()中的mat_params设置
   - 流体: 调整iscosity
   - 弹性体: 调整E(杨氏模量)和elastic_relaxation
   - 沙子: 调整riction_angle

3. **场景扩展**:
   - 在setup_scene()添加更多dd_cube() / dd_ball_obstacles()调用
   - 或创建新的场景函数

## 依赖和环境

- **Python**: 3.10.11+
- **Taichi**: 1.7.4+
- **其他**: NumPy (用于数据处理)

## 代码风格

- 使用@ti.kernel和@ti.func装饰器分别定义GPU并行化核心和辅助函数
- @ti.data_oriented装饰类以支持Taichi字段访问
- 命名约定: 
  - Taichi字段用	i.field()，变量名通常短 (如x, , F)
  - Python变量用下划线分隔 (如
_particles, mat_params)

## 参考论文源码 (Reference Implementation)

**目录位置**: `[demo/pbmpm/]`

**说明**: 该目录包含了原论文的原始实现代码（通常为javasript）。**这些文件仅作为算法逻辑和数学公式推导的“绝对真理”参考（Ground Truth）。请勿修改此目录下的任何文件。** 
### 核心对照映射表 (File Mapping)
当需要实现新功能或修复物理模拟 Bug 时，请优先参考原版代码的数学推导，并将其转换为符合 Taichi 规范的代码。对应关系如下：

| 本项目 Taichi 文件 | 参考源码对应文件 | 参考重点说明 |
|-------------------|----------------|-------------|
| `mpm_pbd.py`      | `(g2p2g.wgsl)` | 核心 P2G/G2P 传输逻辑，以及流体/弹性体/沙子的 PBD 约束求解 |


3. **数学一致性**: 遇到 Taichi 仿真爆炸（NaN）或结果不符合预期时，必须逐行对比参考代码中的矩阵求逆、SVD 分解或梯度计算步骤，确保数学顺序绝对一致。