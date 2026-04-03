import random
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

# =============================================================================
# Clock Parameters
# =============================================================================
BASE_FREQ = 700e6
EXEC_TIME_MS = 700

MU = 0
SIGMA_DRIFT = 3.33

THETA = 0.005

N_SIM = 1_000_000
BATCH_SIZE = 10_000

BOUNDS = [125, 150, 175, 200]
COLORS = ['steelblue', 'darkorange', 'green', 'crimson']

CACHE_FILE = 'sim_cache.pkl'


def truncated_normal_batch(n):
    """向量化生成截断正态分布 [0, 50]"""
    results = np.empty(n)
    remaining = n
    idx = 0
    while remaining > 0:
        samples = np.random.normal(MU, SIGMA_DRIFT, size=int(remaining * 1.1) + 100)
        valid = samples[(samples >= 0) & (samples <= 50)]
        take = min(len(valid), remaining)
        results[idx:idx + take] = valid[:take]
        idx += take
        remaining -= take
    return results


def simulate_batch(exec_times, theta):
    """向量化模拟一批 OU 过程"""
    decay = math.exp(-theta)
    batch_n = len(exec_times)
    max_t = int(np.max(exec_times))
    e = np.zeros(batch_n)
    for t in range(max_t):
        active = t < exec_times
        n_active = int(np.sum(active))
        if n_active == 0:
            break
        drift = truncated_normal_batch(n_active)
        e[active] = e[active] * decay + drift
    return e


# =============================================================================
# 模拟（带缓存）
# =============================================================================
need_sim = True

if os.path.exists(CACHE_FILE):
    print("从缓存加载模拟结果...")
    with open(CACHE_FILE, 'rb') as f:
        cached = pickle.load(f)
    if cached.get('n_sim') == N_SIM and cached.get('bounds') == BOUNDS:
        all_results = cached['results']
        print("缓存加载成功!")
        need_sim = False
    else:
        print("缓存参数不匹配，重新模拟...")

if need_sim:
    print(f"模拟 {N_SIM:,} 次（分批 {BATCH_SIZE:,}）...")
    exec_times = np.random.poisson(EXEC_TIME_MS, size=N_SIM)
    clock_errs = np.empty(N_SIM)

    n_batches = (N_SIM + BATCH_SIZE - 1) // BATCH_SIZE
    for b in tqdm(range(n_batches), desc="模拟进度", ncols=80):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, N_SIM)
        clock_errs[start:end] = simulate_batch(exec_times[start:end], THETA)

    all_results = {}
    for bound in BOUNDS:
        total = np.sort(clock_errs + bound)
        all_results[bound] = total

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'n_sim': N_SIM, 'bounds': BOUNDS, 'results': all_results}, f)
    print(f"缓存已保存到 {CACHE_FILE}")

# =============================================================================
# Academic Style
# =============================================================================
ACAD_COLORS = ['#0C4C8A', '#CE5C00', '#1D8E3E', '#75507B']

plt.rcParams.update({
    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "patch.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
})

# =============================================================================
# 画图 - 断轴 (broken axis)：省略 850~7000 之间的横坐标
# =============================================================================
fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, figsize=(4, 3),
    gridspec_kw={'width_ratios': [6, 1.5], 'wspace': 0.08}
)

cdf = np.arange(1, N_SIM + 1) / N_SIM

for bound, color in zip(BOUNDS, ACAD_COLORS):
    data = all_results[bound]
    # 左图：主数据区 (0 ~ 850)
    ax1.plot(data, cdf, linewidth=2.5, color=color, label=f'bound = {bound} ns')
    # 右图：尾部区域 (6800 ~ 7200)
    ax2.plot(data, cdf, linewidth=2.5, color=color)

# 设置左右图的 x 范围
ax1.set_xlim(400, 850)
ax2.set_xlim(6800, 7200)

# 在 x=7000 处画一条不同颜色的虚线（只在右图可见）
ax2.axvline(x=7000, color='red', linestyle='--', linewidth=1.5, alpha=0.85)
ax2.text(6950, 0.5, '99.999%\nPredictability', color='red', fontsize=7,
         rotation=0, va='center', ha='right')
# 在 ax1 上添加一条不可见的线，用于在图例中显示红色虚线
ax1.plot([], [], color='red', linestyle='--', linewidth=1.5, label='0.001% JCT')

# ---------- 断轴样式 ----------
# 隐藏断轴处相邻的 spines
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.tick_params(left=False)

# 断轴斜线标记
d = 0.012  # 斜线长度
kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False, linewidth=1.2)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)        # 右下角
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右上角

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d * 6, +d * 6), (-d, +d), **kwargs)        # 左下角
ax2.plot((-d * 6, +d * 6), (1 - d, 1 + d), **kwargs)  # 左上角

# ---------- 标签与图例 ----------
ax1.set_ylabel('CDF')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax2.grid(True, alpha=0.3)

# 右图 x 轴只显示 7000
ax2.set_xticks([7000])

# 用 fig.text 让 xlabel 居中于整个图的底部
fig.text(0.5, -0.02, 'Accumulative clock-error of one job (ns)', ha='center', fontsize=12)

# ---------- 边框颜色 ----------
for spine in ['top', 'bottom']:
    ax1.spines[spine].set_visible(True)
    ax1.spines[spine].set_edgecolor('black')
    ax2.spines[spine].set_visible(True)
    ax2.spines[spine].set_edgecolor('black')
ax1.spines['left'].set_visible(True)
ax1.spines['left'].set_edgecolor('black')
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_edgecolor('black')

# 右图右侧 tick（不显示标签，只保持对称）
ax2.yaxis.tick_right()
ax2.tick_params(right=True, labelright=False)

plt.tight_layout()
plt.savefig('clock_sim_result.png', dpi=300, bbox_inches='tight')
plt.savefig('clock_sim_result.pdf', dpi=300, bbox_inches='tight')
print("Done. Saved to clock_sim_result.png / .pdf")
