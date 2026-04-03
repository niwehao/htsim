#!/usr/bin/env python3
"""
Standalone CDF plotter for failure simulation results.
Reads cached scenario data and plots Node / Link failure on a single figure.
Only plots: All jobs + Directly affected (per scenario).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Academic style (adapted from user's AcademicStyleManager)
# ──────────────────────────────────────────────────────────────────────
COLORS = ['#0C4C8A', '#CE5C00', '#1D8E3E', '#75507B', '#555753']

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


def finalize_axes(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')


# ──────────────────────────────────────────────────────────────────────
# Ratio re-computation from sim_results
# ──────────────────────────────────────────────────────────────────────
def recompute_ratios(baseline_stats, sim_results):
    """Recompute ratios from raw sim_results.
    post_delayed: JCT = new_finish - new_start
    directly affected: JCT = new_finish - original_submit
    denominator = baseline JCT
    """
    ratios = []
    for r in baseline_stats['job_details']:
        jid = r['job_id']
        bjct = r['jct']
        if jid in sim_results:
            sr = sim_results[jid]
            cat = sr['category']
            if cat == 'post_delayed':
                njct = sr['new_finish'] - sr['new_start']
            else:
                njct = sr['new_jct']
            ratio = njct / bjct if bjct > 0 else 1.0
        else:
            ratio = 1.0
            cat = 'fixed'
        ratios.append((jid, ratio, cat))
    return ratios


# ──────────────────────────────────────────────────────────────────────
# CDF helper
# ──────────────────────────────────────────────────────────────────────
def _cdf_xy(values):
    """Return (x, y) arrays for a step-CDF starting at y=0."""
    vs = np.sort(values)
    cdf = np.arange(1, len(vs) + 1) / len(vs)
    x = np.concatenate([[vs[0]], vs])
    y = np.concatenate([[0.0], cdf])
    return x, y


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Config (must match failure_simulator.py tags) ---
    num_jobs = 1024
    seed = 42
    num_gpus = 16384
    frate = 0.01
    ftime = 0.0

    # --- Load baseline ---
    bl_tag = f"online_bl_j{num_jobs}_s{seed}_g{num_gpus}"
    bl_path = os.path.join(base_dir, f".cache_{bl_tag}.pkl")
    if not os.path.exists(bl_path):
        raise FileNotFoundError(f"Baseline cache not found: {bl_path}")
    with open(bl_path, 'rb') as f:
        bl_data = pickle.load(f)
    # baseline is (jobs, stats) tuple
    if isinstance(bl_data, tuple):
        baseline_stats = bl_data[1]
    else:
        baseline_stats = bl_data

    # --- Load scenarios ---
    scenarios = {}
    for ftype, label in [("node", "Switch failure"), ("link", "Link failure")]:
        sc_tag = f"online_sc_{ftype}_j{num_jobs}_s{seed}_g{num_gpus}_fr{frate}_ft{ftime}"
        path = os.path.join(base_dir, f".cache_{sc_tag}.pkl")
        if not os.path.exists(path):
            print(f"WARNING: No cache found for {ftype}: {path}")
            continue
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Re-compute ratios if sim_results available
        if isinstance(data, dict) and 'sim_results' in data:
            ratios = recompute_ratios(baseline_stats, data['sim_results'])
        elif isinstance(data, dict) and 'ratios' in data:
            ratios = data['ratios']
        else:
            print(f"WARNING: Unknown cache format for {ftype}")
            continue

        n_direct = data.get('n_direct_affected', 0) if isinstance(data, dict) else 0
        scenarios[ftype] = {
            'label': label,
            'ratios': ratios,
            'n_direct': n_direct,
        }

    if not scenarios:
        raise RuntimeError("No scenario data found")

    n_total = len(baseline_stats['job_details'])

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    color_map = {
        'node': {'all': COLORS[0], 'direct': COLORS[0]},
        'link': {'all': COLORS[2], 'direct': COLORS[2]},
    }

    for ftype, sc in scenarios.items():
        ratios = sc['ratios']
        lbl = sc['label']
        c = color_map[ftype]

        # All jobs
        vals_all = sorted([r for _, r, _ in ratios])
        x, y = _cdf_xy(vals_all)
        ax.plot(x, y, color=c['all'], lw=1.8,
                label=f'{lbl} — All jobs')

        # Directly affected (pre_affected + post_affected)
        vals_direct = sorted([r for _, r, cat in ratios
                              if cat in ('pre_affected', 'post_affected')])
        if vals_direct:
            x, y = _cdf_xy(vals_direct)
            ax.plot(x, y, color=c['direct'], lw=1.4, ls='--',
                    label=f'{lbl} — Failed jobs')

    ax.axvline(x=1.0, color='grey', ls=':', lw=0.8, alpha=0.6)

    finalize_axes(ax,
                  xlabel='JCT ratio (failure / baseline)',
                  ylabel='CDF')
    ax.legend(loc='lower right', fontsize=8, handlelength=2,
            labelspacing=0.3, borderpad=0.4)
    ax.grid(True, alpha=0.3)

    out = os.path.join(base_dir, "failure_cdf_combined.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    #保存为pdf
    fig.savefig(out.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
