"""
Measure offline scheduling computation time vs job count scaling.
For each of 3 topologies: job count doubles from 16 to 1024.
Each data point averaged over 5 runs with different seeds.
One subplot per topology.
"""

import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 获取当前脚本所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将其插入到 sys.path 的最前面
sys.path.insert(0, current_dir)
from utopia_scheduler import (
    make_topo, WorkloadGenerator, UtopiaScheduler, logger
)

# Job counts从64开始,是64倍数,一直到1024
JOB_COUNTS = list(range(64, 1025, 64))

TOPOS = ["ocs", "dragonfly", "spine-leaf"]
COLORS = {"ocs": "steelblue", "dragonfly": "darkorange", "spine-leaf": "seagreen"}
NUM_REPEATS = 1  # average over 5 seeds per data point


def measure_one(topo_type: str, num_jobs: int, seed: int):
    """Generate jobs, measure scheduling time only."""
    topo = make_topo(topo_type)
    wg = WorkloadGenerator(topo.num_gpus, rng_seed=seed)
    jobs = wg.generate_jobs(num_jobs)

    t0 = time.perf_counter()
    scheduler = UtopiaScheduler(topo)
    scheduler.schedule_all_jobs(jobs)
    elapsed = time.perf_counter() - t0
    return elapsed


def main():
    # results[topo] = { num_jobs: [times...] }
    results = {t: {} for t in TOPOS}

    for topo_type in TOPOS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Topology: {topo_type}")
        logger.info(f"{'='*60}")

        for nj in JOB_COUNTS:
            times = []
            for r in range(NUM_REPEATS):
                seed = 1000 + r
                elapsed = measure_one(topo_type, nj, seed)
                times.append(elapsed)
                logger.info(f"  [{topo_type}] jobs={nj}, run {r+1}/{NUM_REPEATS}: {elapsed:.3f}s")
            results[topo_type][nj] = times
            logger.info(f"  [{topo_type}] jobs={nj} => mean={np.mean(times):.3f}s, "
                        f"std={np.std(times):.3f}s")

    # Plot: one subplot per topology
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, topo_type in enumerate(TOPOS):
        ax = axes[idx]
        # 计算均值并转换为纳秒 (ns)
        means_ns = [np.mean(results[topo_type][nj]) * 1e9 for nj in JOB_COUNTS]
        stds_ns = [np.std(results[topo_type][nj]) * 1e9 for nj in JOB_COUNTS]

        ax.errorbar(means_ns, JOB_COUNTS, xerr=stds_ns, fmt='o-',
                    color=COLORS[topo_type], linewidth=2.5, markersize=8,
                    capsize=5, capthick=2, label=topo_type)

        # 用虚线标注 0.999999 分位值
        # 对所有 job count 的时间取 0.999999 分位
        all_times_ns = []
        for nj in JOB_COUNTS:
            for t in results[topo_type][nj]:
                all_times_ns.append(t * 1e9)
        p999999 = np.percentile(all_times_ns, 99.9999)

        ax.axvline(x=p999999, color=COLORS[topo_type], linestyle='--',
                   linewidth=2, alpha=0.7,
                   label=f'p99.9999 = {p999999:.0f} ns')

        # Annotate each point
        for nj, m in zip(JOB_COUNTS, means_ns):
            ax.annotate(f'{m:.0f} ns', xy=(m, nj),
                        xytext=(12, 0), textcoords='offset points',
                        fontsize=9, ha='left', color=COLORS[topo_type])

        ax.set_xlabel('Scheduling Time (ns)', fontsize=12)
        ax.set_ylabel('Number of Jobs', fontsize=12)
        ax.set_yticks(JOB_COUNTS)
        ax.set_yticklabels([str(n) for n in JOB_COUNTS])
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)

    # 不需要 title
    plt.tight_layout()
    out = 'scheduling_time_scaling.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to {out}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Jobs':>6}", end="")
    for t in TOPOS:
        print(f"  {t:>16s}", end="")
    print()
    print(f"{'-'*70}")
    for nj in JOB_COUNTS:
        print(f"{nj:>6}", end="")
        for t in TOPOS:
            m = np.mean(results[t][nj]) * 1e9
            s = np.std(results[t][nj]) * 1e9
            print(f"  {m:>10.0f}±{s:<4.0f} ns", end="")
        print()


if __name__ == "__main__":
    main()