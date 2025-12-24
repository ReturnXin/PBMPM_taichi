import taichi as ti
import time
import numpy as np

# 初始化 Taichi，开启性能分析器
ti.init(arch=ti.cuda, kernel_profiler=True)

from mpm_pbd import MpmPBDSolver


def run_benchmark(use_optimization=True, num_steps=100, start_frame=50):
    mpm = MpmPBDSolver()
    if not use_optimization:
        mpm.use_morton_code = False
        mpm.use_dynamic_grid = False
    else:
        mpm.use_morton_code = True
        mpm.use_dynamic_grid = False

    mpm.add_cube(
        particle_num=2**16,
        center=[0.5, 0.5, 0.5],
        cube_size=[0.4, 0.4, 0.4],
    )
    mpm.init(hide_obstacles=True)

    for _ in range(10):  # 预热 GPU (Warm up)，排除初次编译时间
        mpm.substep()
    ti.sync()

    print(f"开始测试 {'优化版' if use_optimization else '原始版'}...")
    ti.profiler.clear_kernel_profiler_info()
    start_t = 0
    for i in range(num_steps):
        mpm.substep()
        if i == start_frame:
            start_t = time.time()
        if (i + 1) % 20 == 0:
            print(f"已完成 {i+1}/{num_steps-start_frame} 步...")

    ti.sync()
    end_t = time.time()

    total_time = end_t - start_t
    avg_step_time = (total_time / num_steps) * 1000

    return avg_step_time, total_time


def main():
    steps = 100
    opt_avg_ms, opt_total = run_benchmark(use_optimization=True, num_steps=steps)
    ti.sync()
    time.sleep(20)
    raw_avg_ms, raw_total = run_benchmark(use_optimization=False, num_steps=steps)

    # --- 计算结果 ---
    speedup = raw_total / opt_total
    improvement = (1 - opt_total / raw_total) * 100

    print("\n" + "=" * 30)
    print("      性能对比报告")
    print("=" * 30)
    print(f"测试步数:     {steps}")
    print(f"原始版平均耗时: {raw_avg_ms:.2f} ms/step")
    print(f"优化版平均耗时: {opt_avg_ms:.2f} ms/step")
    print("-" * 30)
    print(f"加速比 (Speedup):  {speedup:.2f}x")
    print(f"性能提升 (Improvement): {improvement:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    main()
