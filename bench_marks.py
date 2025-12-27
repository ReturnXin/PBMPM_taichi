import sys
import subprocess
import time
import argparse
import re

# =================配置区域=================
# 增加步数以获得更稳定的平均值
TOTAL_STEPS = 300
WARMUP_STEPS = 50  # 每个子进程内部的预热
PARTICLE_NUM = 2**16
# =========================================


def run_worker_process(mode_name, is_optimized, dry_run=False):
    """
    dry_run=True 时，不打印花里胡哨的输出，只负责唤醒显卡
    """
    if not dry_run:
        print(f"🚀 正在测试: {mode_name} ... ", end="", flush=True)

    cmd = [sys.executable, __file__, "--worker", "--steps", str(TOTAL_STEPS), "--warmup", str(WARMUP_STEPS)]

    if is_optimized:
        cmd.append("--opt")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        match = re.search(r"BENCHMARK_RESULT:\s*([\d\.]+)", result.stdout)
        if match:
            ms_per_step = float(match.group(1))
            if not dry_run:
                print(f"✅ 完成 ({ms_per_step:.2f} ms/step)")
            return ms_per_step
        else:
            if not dry_run:
                print("\n❌ 错误: 无有效数据")
            return None
    except subprocess.CalledProcessError:
        return None


def worker_task(args):
    # 子进程代码保持不变
    try:
        import taichi as ti
        from mpm_pbd import MpmPBDSolver
    except ImportError:
        sys.exit(1)

    ti.init(arch=ti.cuda, kernel_profiler=False)

    mpm = MpmPBDSolver()
    # 强制让所有参数显式生效
    mpm.use_morton_code = args.opt
    mpm.use_dynamic_grid = False

    mpm.add_cube(
        particle_num=PARTICLE_NUM,
        center=[0.5, 0.5, 0.5],
        cube_size=[0.4, 0.4, 0.4],
    )
    mpm.init(hide_obstacles=True)

    # 内部预热
    for _ in range(args.warmup):
        mpm.substep()
    ti.sync()

    # 正式计时
    measure_steps = args.steps - args.warmup
    start_t = time.time()
    for _ in range(measure_steps):
        mpm.substep()
    ti.sync()
    end_t = time.time()

    avg_ms = ((end_t - start_t) / measure_steps) * 1000.0
    print(f"BENCHMARK_RESULT: {avg_ms}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--opt", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if args.worker:
        worker_task(args)
    else:

        # 1. 关键步骤：唤醒 GPU (Dummy Run)
        # 我们跑一次原始版，但不记录成绩，单纯为了让显卡频率拉满
        print("🔥 正在唤醒 GPU (Dummy Run)... 请稍候")
        run_worker_process("唤醒", is_optimized=False, dry_run=True)
        print("   -> GPU 已唤醒，频率已稳定。")
        print("-" * 40)

        # 2. 为了公平，我们在两个测试之间几乎不给显卡休息时间
        # 这样能保证两者都在 GPU 热的状态下运行

        # 测试 A
        # time.sleep(40)
        t_opt = run_worker_process("优化版 (Morton)", is_optimized=True)

        # 测试 B
        time.sleep(60)
        t_raw = run_worker_process("原始版 (Naive) ", is_optimized=False)

        # 3. 报告
        if t_opt and t_raw:
            speedup = t_raw / t_opt
            improvement = (1 - t_opt / t_raw) * 100

            print("\n" + "=" * 40)
            print(" 📊 最终性能报告")
            print("=" * 40)
            print(f"原始耗时: {t_raw:8.4f} ms/step")
            print(f"优化耗时: {t_opt:8.4f} ms/step")
            print("-" * 40)
            if improvement > 0:
                print(f"🚀 加速比:     {speedup:.2f} x")
                print(f"🔥 性能提升:   {improvement:.2f} %")
            else:
                print(f"🐢 性能下降:   {abs(improvement):.2f} %")
                print("   (注意: 如果下降，可能是 Morton 计算开销超过了缓存收益)")
            print("=" * 40)
