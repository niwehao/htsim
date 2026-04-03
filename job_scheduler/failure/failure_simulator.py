"""
Online Failure Simulator for Spine-Leaf Topology (16K GPUs)

Key model:
  - Baseline: static optimal scheduling (all jobs known upfront)
  - Failure scenario: online scheduling model
    - Jobs arrive via Poisson process
    - At failure_time: affected in-progress jobs restart from scratch
    - Rescheduled jobs only see GPU reservations from already-scheduled jobs
    - Future arriving jobs (even "unaffected") must schedule online on degraded topo
    - Ripple effect: more jobs delayed than just directly-affected ones

Two failure scenarios (each 1% rate, once, no recovery):
  1. Node (leaf switch) failure → GPUs under failed leaves unavailable
  2. Link failure → affected paths unavailable

Usage:
  python failure_simulator.py [num_jobs] [failure_rate] [failure_time]
  python failure_simulator.py --plot-only
"""

import numpy as np
import time
import os
import pickle
import logging
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utopia_scheduler import (
    Flow, Collective, Job, SpineLeafTopology,
    UtopiaScheduler, compute_delay, JOB_PROFILES
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Job profiles: (num_gpus, duration) ──
# Distribution: 40% / 30% / 20% / 10%
ONLINE_PROFILES = [
    (16,   8),
    (32,  14),
    (64,  25),
    (128, 46),
]
PROFILE_WEIGHTS = [0.40, 0.30, 0.20, 0.10]


# =============================================================================
# Spine-Leaf Topology with Failures
# =============================================================================

class FailedSpineLeafTopology(SpineLeafTopology):
    """SpineLeafTopology with failed leaf switches and/or failed links."""

    def __init__(self, failed_leaves: Set[int] = None,
                 failed_links: Set[Tuple[int, int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.failed_leaves = failed_leaves or set()
        self.failed_links = failed_links or set()
        self._failed_link_set = set()
        for (a, b) in self.failed_links:
            self._failed_link_set.add((a, b))
            self._failed_link_set.add((b, a))

        self.unavailable_gpus: Set[int] = set()
        for leaf_idx in self.failed_leaves:
            start_gpu = leaf_idx * self.gpus_per_leaf
            for g in range(start_gpu, start_gpu + self.gpus_per_leaf):
                self.unavailable_gpus.add(g)
        for (a, b) in self.failed_links:
            if a < self.leaf_offset and self.leaf_offset <= b < self.spine_offset:
                self.unavailable_gpus.add(a)
            elif b < self.leaf_offset and self.leaf_offset <= a < self.spine_offset:
                self.unavailable_gpus.add(b)

        logger.info(f"[FailedTopo] {len(self.failed_leaves)} leaves failed, "
                    f"{len(self.failed_links)} links failed, "
                    f"{len(self.unavailable_gpus)} GPUs unavailable")

    def _is_link_failed(self, link: Tuple[int, int]) -> bool:
        return link in self._failed_link_set

    def _is_leaf_failed_by_id(self, leaf_node_id: int) -> bool:
        return (leaf_node_id - self.leaf_offset) in self.failed_leaves

    def _get_all_paths(self, src: int, dst: int) -> List[List[Tuple[int, int]]]:
        if src == dst:
            return [[]]
        src_leaf = self._gpu_to_leaf(src)
        dst_leaf = self._gpu_to_leaf(dst)
        if self._is_leaf_failed_by_id(src_leaf) or self._is_leaf_failed_by_id(dst_leaf):
            return []
        if src_leaf == dst_leaf:
            path = [(src, src_leaf), (src_leaf, dst)]
            if any(self._is_link_failed(lk) for lk in path):
                return []
            return [path]
        paths = []
        for s in range(self.num_spines):
            spine = self.spine_offset + s
            path = [(src, src_leaf), (src_leaf, spine),
                    (spine, dst_leaf), (dst_leaf, dst)]
            if any(self._is_link_failed(lk) for lk in path):
                continue
            paths.append(path)
        return paths

    def get_residual_capacity(self, link: Tuple[int, int], t: float) -> float:
        if self._is_link_failed(link):
            return 0.0
        return super().get_residual_capacity(link, t)


# =============================================================================
# Online Failure Simulator
# =============================================================================

class OnlineFailureSimulator:
    """
    Compares: baseline (static optimal, no failure) vs
              online scheduling on degraded topology after failure.
    """

    def __init__(self, num_jobs: int = 1024, seed: int = 42,
                 num_gpus: int = 16384, target_utilization: float = 0.75):
        self.num_jobs = num_jobs
        self.seed = seed
        self.num_gpus = num_gpus
        self.target_util = target_utilization

    # ─── helpers ───

    @staticmethod
    def _alltoall_flows(gpus: List[int], data_size_mb: float) -> List[Flow]:
        n = len(gpus)
        per_pair = data_size_mb / (n * n)
        flows, fid = [], 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    flows.append(Flow(flow_id=fid, src=gpus[i],
                                      dst=gpus[j], data_size=per_pair))
                    fid += 1
        return flows

    def _cache_file(self, tag: str) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f".cache_{tag}.pkl")

    # ─── Job generation (Poisson arrival, new distribution) ───

    def _generate_jobs(self) -> List[Job]:
        rng = np.random.RandomState(self.seed)

        # counts per profile
        counts = [max(1, int(round(w * self.num_jobs)))
                  for w in PROFILE_WEIGHTS]
        counts[0] += self.num_jobs - sum(counts)

        # arrival window for target utilization
        total_gpu_time = sum(counts[i] * ONLINE_PROFILES[i][0] * ONLINE_PROFILES[i][1]
                             for i in range(len(ONLINE_PROFILES)))
        arrival_window = total_gpu_time / (self.num_gpus * self.target_util)
        mean_inter = arrival_window / self.num_jobs

        # build (profile_index) list, shuffle
        specs = []
        for pi in range(len(ONLINE_PROFILES)):
            specs.extend([pi] * counts[pi])
        rng.shuffle(specs)

        # Poisson arrival times
        arrivals = np.cumsum(rng.exponential(mean_inter, size=self.num_jobs))

        jobs = []
        for idx, pi in enumerate(specs):
            ngpu, dur = ONLINE_PROFILES[pi]
            max_start = max(0, self.num_gpus - ngpu)
            sg = rng.randint(0, max_start + 1)
            gpus = list(range(sg, sg + ngpu))

            ncoll = max(1, dur // 10)
            colls = []
            for c_id in range(ncoll):
                ds = rng.uniform(50, 200)
                ct = dur / ncoll
                colls.append(Collective(collective_id=c_id,
                                        flows=self._alltoall_flows(gpus, ds),
                                        compute_time=ct))
            jobs.append(Job(job_id=idx, collectives=colls, gpu_nodes=gpus,
                            priority=rng.randint(0, 3),
                            submit_time=float(arrivals[idx]),
                            duration=dur))

        logger.info(
            f"Generated {self.num_jobs} jobs: "
            + ", ".join(f"{ONLINE_PROFILES[i][0]}GPU={counts[i]}"
                        for i in range(len(ONLINE_PROFILES)))
            + f"  arrival∈[{arrivals[0]:.2f}, {arrivals[-1]:.2f}]")
        return jobs

    # ─── Baseline (static optimal, cached) ───

    def run_baseline(self, use_cache: bool = True):
        tag = f"online_bl_j{self.num_jobs}_s{self.seed}_g{self.num_gpus}"
        cf = self._cache_file(tag)

        if use_cache and os.path.exists(cf):
            logger.info(f"Loading baseline cache: {cf}")
            with open(cf, 'rb') as f:
                jobs, stats = pickle.load(f)
            logger.info(f"  {len(jobs)} jobs, avg_jct={stats['avg_jct']:.2f}")
            return jobs, stats

        logger.info("Running baseline (static optimal)…")
        jobs = self._generate_jobs()
        topo = SpineLeafTopology(num_gpus=self.num_gpus)
        sched = UtopiaScheduler(topo)
        t0 = time.time()
        stats = sched.schedule_all_jobs(jobs)
        logger.info(f"Baseline done in {time.time()-t0:.1f}s — "
                    f"avg_jct={stats['avg_jct']:.2f}")

        if use_cache:
            with open(cf, 'wb') as f:
                pickle.dump((jobs, stats), f)
            logger.info(f"Cached → {cf}")
        return jobs, stats

    # ─── Failure injection ───

    def _enumerate_all_links(self) -> List[Tuple[int, int]]:
        topo = SpineLeafTopology(num_gpus=self.num_gpus)
        links = []
        for g in range(topo.num_gpus):
            links.append((g, topo._gpu_to_leaf(g)))
        for li in range(topo.num_leaves):
            lid = topo.leaf_offset + li
            for si in range(topo.num_spines):
                links.append((lid, topo.spine_offset + si))
        return links

    def inject_node_failure(self, rng, rate=0.01) -> Set[int]:
        topo = SpineLeafTopology(num_gpus=self.num_gpus)
        n = max(1, int(topo.num_leaves * rate))
        failed = set(rng.choice(topo.num_leaves, n, replace=False).tolist())
        logger.info(f"Node failure: {n} leaves → "
                    f"{n * topo.gpus_per_leaf} GPUs lost")
        return failed

    def inject_link_failure(self, rng, rate=0.01) -> Set[Tuple[int, int]]:
        all_lk = self._enumerate_all_links()
        n = max(1, int(len(all_lk) * rate))
        idx = rng.choice(len(all_lk), n, replace=False)
        failed = set(all_lk[i] for i in idx)
        logger.info(f"Link failure: {n}/{len(all_lk)} links failed")
        return failed

    # ─── Affected-job detection (for pre-failure jobs) ───

    def _is_job_affected(self, job: Job,
                         failed_leaves: Set[int],
                         failed_links: Set[Tuple[int, int]]) -> bool:
        topo_ref = SpineLeafTopology(num_gpus=self.num_gpus)
        failed_leaf_ids = set(topo_ref.leaf_offset + idx for idx in failed_leaves)

        # GPUs under failed leaves
        for li in failed_leaves:
            lo = li * topo_ref.gpus_per_leaf
            hi = lo + topo_ref.gpus_per_leaf
            for g in job.gpu_nodes:
                if lo <= g < hi:
                    return True

        # Normalise failed links for bidirectional check
        fls = set()
        for (a, b) in failed_links:
            fls.add((a, b)); fls.add((b, a))

        # GPU isolated by gpu-leaf link failure
        for (a, b) in failed_links:
            if a < topo_ref.leaf_offset and topo_ref.leaf_offset <= b < topo_ref.spine_offset:
                if a in job.gpu_nodes:
                    return True
            elif b < topo_ref.leaf_offset and topo_ref.leaf_offset <= a < topo_ref.spine_offset:
                if b in job.gpu_nodes:
                    return True

        # Flow paths
        for coll in job.collectives:
            for flow in coll.flows:
                for lk in flow.path:
                    if lk in fls:
                        return True
                    if lk[0] in failed_leaf_ids or lk[1] in failed_leaf_ids:
                        return True
        return False

    # ─── Partition jobs into 5 categories ───

    def _partition_jobs(self, jobs, baseline_stats,
                        failed_leaves, failed_links, failure_time):
        bmap = {r['job_id']: r for r in baseline_stats['job_details']}

        completed, pre_unaff, pre_aff = [], [], []
        post_aff, post_unaff = [], []
        for job in jobs:
            br = bmap[job.job_id]
            if br['finish_time'] <= failure_time:
                completed.append(job)
            elif job.submit_time <= failure_time:
                if self._is_job_affected(job, failed_leaves, failed_links):
                    pre_aff.append(job)
                else:
                    pre_unaff.append(job)
            else:
                # Post-failure arrival: check if baseline assignment uses failed resources
                if self._is_job_affected(job, failed_leaves, failed_links):
                    post_aff.append(job)
                else:
                    post_unaff.append(job)

        logger.info(f"Partition: completed={len(completed)}, "
                    f"pre_unaffected={len(pre_unaff)}, "
                    f"pre_affected={len(pre_aff)}, "
                    f"post_affected={len(post_aff)}, "
                    f"post_unaffected={len(post_unaff)}")
        return completed, pre_unaff, pre_aff, post_aff, post_unaff

    # ─── GPU block finder ───

    def _find_gpu_block(self, num_needed: int,
                        gpu_free: Dict[int, float],
                        unavailable: Set[int]):
        available = [g for g in range(self.num_gpus) if g not in unavailable]
        best_block, best_start = None, float('inf')
        i = 0
        while i <= len(available) - num_needed:
            blk = available[i:i + num_needed]
            if blk[-1] - blk[0] == num_needed - 1:      # contiguous
                st = max(gpu_free.get(g, 0.0) for g in blk)
                if st < best_start:
                    best_start = st
                    best_block = blk
            i += 1
        if best_block is None:
            logger.warning(f"No contiguous block of {num_needed}; "
                          f"falling back to non-contiguous")
            sa = sorted(available, key=lambda g: gpu_free.get(g, 0.0))
            best_block = sa[:num_needed]
            best_start = max(gpu_free.get(g, 0.0) for g in best_block)
        return best_block, best_start

    # ─── Rebuild job on new GPUs ───

    def _rebuild_job(self, orig: Job, new_gpus: List[int],
                     new_submit: float) -> Job:
        n = len(new_gpus)
        new_colls = []
        for coll in orig.collectives:
            pp = coll.flows[0].data_size if coll.flows else 0.0
            flows, fid = [], 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        flows.append(Flow(flow_id=fid, src=new_gpus[i],
                                          dst=new_gpus[j], data_size=pp))
                        fid += 1
            new_colls.append(Collective(collective_id=coll.collective_id,
                                        flows=flows,
                                        compute_time=coll.compute_time))
        return Job(job_id=orig.job_id, collectives=new_colls,
                   gpu_nodes=new_gpus, priority=-1,
                   submit_time=new_submit, duration=orig.duration)

    # ─── Core: online failure simulation ───

    def _simulate_online(self, jobs, baseline_stats,
                         failed_leaves, failed_links, failure_time):
        """
        Returns: (results_dict, n_direct_affected, n_post_delayed)
        Categories:
          - fixed: completed before failure OR pre-failure unaffected → ratio=1.0
          - pre_affected: running at failure time, uses failed resources → reschedule
          - post_affected: arrives after failure, baseline uses failed resources → reschedule
          - post_delayed: arrives after failure, baseline OK, but GPU contention delays it
          - post_unaffected: arrives after failure, baseline OK, no delay → ratio=1.0
        """
        bmap = {r['job_id']: r for r in baseline_stats['job_details']}

        completed, pre_unaff, pre_aff, post_aff, post_unaff = \
            self._partition_jobs(
                jobs, baseline_stats, failed_leaves, failed_links, failure_time)

        # Build degraded topology
        ftopo = FailedSpineLeafTopology(
            failed_leaves=failed_leaves,
            failed_links=failed_links,
            num_gpus=self.num_gpus)
        unavail = ftopo.unavailable_gpus

        # Seed topo with reservations from fixed jobs (completed + pre_unaff)
        fixed = completed + pre_unaff
        for job in fixed:
            for coll in job.collectives:
                for fl in coll.flows:
                    if fl.path and fl.rate > 0:
                        dur = fl.finish_time - fl.start_time
                        if dur > 0:
                            ftopo.reserve_bandwidth(
                                fl.path, fl.start_time, dur, fl.rate)

        gpu_free: Dict[int, float] = defaultdict(float)
        for job in fixed:
            ft = bmap[job.job_id]['finish_time']
            for g in job.gpu_nodes:
                gpu_free[g] = max(gpu_free[g], ft)

        sched = UtopiaScheduler(ftopo)
        results: Dict[int, Dict] = {}

        def _schedule_one(job, category):
            nn = len(job.gpu_nodes)
            blk, es = self._find_gpu_block(nn, gpu_free, unavail)
            es = max(es, job.submit_time)            # can't start before arrival
            nj = self._rebuild_job(job, blk, es)
            nf = sched.schedule_job(nj)
            for g in blk:
                gpu_free[g] = max(gpu_free[g], nf)
            results[job.job_id] = {
                'job_id': job.job_id,
                'original_submit': job.submit_time,
                'baseline_start': bmap[job.job_id]['submit_time'],
                'new_start': es,
                'new_finish': nf,
                'new_jct': nf - job.submit_time,
                'category': category,
            }

        # Phase 1: reschedule directly affected jobs (pre + post affected)
        all_affected = sorted(pre_aff + post_aff, key=lambda j: j.submit_time)
        for job in tqdm(all_affected, desc="Reschedule affected"):
            cat = 'pre_affected' if job.submit_time <= failure_time else 'post_affected'
            _schedule_one(job, cat)

        # Phase 2: process post-failure unaffected jobs in arrival order
        # These jobs keep their baseline GPU assignment, but check if
        # rescheduled jobs have occupied their GPUs → delay
        n_post_delayed = 0
        for job in tqdm(sorted(post_unaff, key=lambda j: j.submit_time),
                        desc="Check post-unaffected"):
            br = bmap[job.job_id]
            baseline_gpus = job.gpu_nodes
            # Check if any of baseline GPUs are now busy past job's baseline start
            baseline_start = br['submit_time']
            gpu_conflict = max((gpu_free.get(g, 0.0) for g in baseline_gpus),
                               default=0.0)

            if gpu_conflict <= baseline_start + 1e-9:
                # No conflict: job runs as baseline, register its GPU usage
                for g in baseline_gpus:
                    gpu_free[g] = max(gpu_free[g], br['finish_time'])
                # Also reserve bandwidth on degraded topo (same flows as baseline)
                for coll in job.collectives:
                    for fl in coll.flows:
                        if fl.path and fl.rate > 0:
                            dur = fl.finish_time - fl.start_time
                            if dur > 0:
                                ftopo.reserve_bandwidth(
                                    fl.path, fl.start_time, dur, fl.rate)
                results[job.job_id] = {
                    'job_id': job.job_id,
                    'original_submit': job.submit_time,
                    'baseline_start': baseline_start,
                    'new_finish': br['finish_time'],
                    'new_jct': br['jct'],
                    'category': 'fixed',
                }
            else:
                # GPU conflict: must reschedule this job too
                n_post_delayed += 1
                _schedule_one(job, 'post_delayed')

        n_direct = len(pre_aff) + len(post_aff)
        return results, n_direct, n_post_delayed

    # ─── Ratio computation ───

    def _compute_ratios(self, baseline_stats, sim_results):
        """Returns [(job_id, ratio, category), ...]
        For post_delayed jobs: JCT = new_finish - new_start (actual reschedule time)
        For directly affected: JCT = new_finish - original_submit
        Denominator is always baseline JCT.
        """
        ratios = []
        for r in baseline_stats['job_details']:
            jid = r['job_id']
            bjct = r['jct']
            if jid in sim_results:
                sr = sim_results[jid]
                cat = sr['category']
                if cat == 'post_delayed':
                    # Ripple-delayed: use actual reschedule start as reference
                    njct = sr['new_finish'] - sr['new_start']
                else:
                    # Directly affected: use original submit_time
                    njct = sr['new_jct']
                ratio = njct / bjct if bjct > 0 else 1.0
                cat = sr['category']
            else:
                ratio = 1.0
                cat = 'fixed'
            ratios.append((jid, ratio, cat))
        return ratios

    # ─── Run one scenario (with cache) ───

    def _scenario_tag(self, ftype, frate, ftime):
        return (f"online_sc_{ftype}_j{self.num_jobs}_s{self.seed}"
                f"_g{self.num_gpus}_fr{frate}_ft{ftime}")

    def run_scenario(self, ftype, jobs, baseline_stats,
                     frate=0.01, ftime=0.0, rng_seed=0,
                     use_cache=True):
        cf = self._cache_file(self._scenario_tag(ftype, frate, ftime))

        if use_cache and os.path.exists(cf):
            logger.info(f"Loading {ftype} scenario from cache")
            with open(cf, 'rb') as f:
                data = pickle.load(f)
            # Re-compute ratios from sim_results (in case formula changed)
            if 'sim_results' in data:
                data['ratios'] = self._compute_ratios(baseline_stats, data['sim_results'])
            return data

        rng = np.random.RandomState(rng_seed)
        fl_leaves: Set[int] = set()
        fl_links: Set[Tuple[int, int]] = set()

        if ftype == "node":
            fl_leaves = self.inject_node_failure(rng, frate)
        else:
            fl_links = self.inject_link_failure(rng, frate)

        sim_res, n_direct, n_delayed = self._simulate_online(
            jobs, baseline_stats, fl_leaves, fl_links, ftime)

        ratios = self._compute_ratios(baseline_stats, sim_res)

        data = {
            'sim_results': sim_res,
            'ratios': ratios,
            'n_direct_affected': n_direct,
            'n_post_delayed': n_delayed,
            'n_total_rescheduled': n_direct + n_delayed,
        }
        if use_cache:
            with open(cf, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Scenario cached → {cf}")
        return data

    # ─── Apply baseline paths to fresh jobs ───

    def _apply_baseline_paths(self, fresh, baseline):
        bmap = {j.job_id: j for j in baseline}
        for job in fresh:
            bj = bmap[job.job_id]
            for co, bco in zip(job.collectives, bj.collectives):
                co.release_time = bco.release_time
                co.finish_time = bco.finish_time
                for fl, bfl in zip(co.flows, bco.flows):
                    fl.path = bfl.path
                    fl.start_time = bfl.start_time
                    fl.finish_time = bfl.finish_time
                    fl.rate = bfl.rate

    # ─── Plot CDF ───

    def _plot_cdf(self, d_node, d_link, n_total):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Online Failure Impact on JCT — Spine-Leaf 16K GPU, "
            f"{self.num_jobs} Jobs, 1% Failure",
            fontsize=13, fontweight='bold')

        scenarios = [
            (axes[0], d_node, "Node (Leaf Switch) Failure", 'tab:red'),
            (axes[1], d_link, "Link Failure",               'tab:blue'),
        ]

        for ax, data, title, color in scenarios:
            ratios = data['ratios']
            n_direct = data['n_direct_affected']
            n_delayed = data['n_post_delayed']
            n_re = data['n_total_rescheduled']

            # (1) All jobs
            vals = sorted([r for _, r, _ in ratios])
            cdf = np.arange(1, len(vals)+1) / len(vals)
            ax.plot(np.concatenate([[vals[0]], vals]),
                    np.concatenate([[0.0], cdf]),
                    color=color, lw=2, label=f'All jobs (n={n_total})')

            # (2) Directly affected (pre_affected + post_affected)
            dv = sorted([r for _, r, c in ratios
                         if c in ('pre_affected', 'post_affected')])
            if dv:
                dc = np.arange(1, len(dv)+1) / len(dv)
                dcol = 'darkred' if 'Node' in title else 'darkblue'
                ax.plot(np.concatenate([[dv[0]], dv]),
                        np.concatenate([[0.0], dc]),
                        color=dcol, lw=1.5, ls='--', alpha=.7,
                        label=f'Directly affected (n={n_direct})')

            # (3) Ripple-delayed (post_delayed)
            rv = sorted([r for _, r, c in ratios if c == 'post_delayed'])
            if rv:
                rc = np.arange(1, len(rv)+1) / len(rv)
                ax.plot(np.concatenate([[rv[0]], rv]),
                        np.concatenate([[0.0], rc]),
                        color='green', lw=1.2, ls='-.', alpha=.6,
                        label=f'Ripple-delayed (n={n_delayed})')

            ax.axvline(x=1.0, color='gray', ls=':', alpha=.5,
                       label='ratio = 1')
            ax.set_xlabel('JCT Ratio (failure / baseline)')
            ax.set_ylabel('CDF')
            ax.set_title(f"{title}\n{n_direct} direct + {n_delayed} ripple"
                        f" = {n_re} affected / {n_total}")
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=.3)
            ax.set_xlim(left=0.95)

        plt.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "failure_simulation_online_cdf.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved → {out}")

    # ─── Main entry ───

    def run_all(self, failure_rate=0.01, failure_time=0.0,
                plot_only=False):
        logger.info("=" * 70)
        logger.info(f"ONLINE FAILURE SIMULATOR — {self.num_gpus} GPUs — "
                    f"{self.num_jobs} jobs")
        logger.info(f"Failure rate={failure_rate*100:.1f}%, "
                    f"failure_time={failure_time}")
        if plot_only:
            logger.info("*** PLOT-ONLY MODE ***")
        logger.info("=" * 70)

        jobs_bl, bl_stats = self.run_baseline()

        # ── Node failure ──
        logger.info("\n" + "="*50)
        logger.info(">>> SCENARIO 1: Node (Leaf Switch) Failure <<<")
        logger.info("="*50)
        if not plot_only:
            jobs_n = self._generate_jobs()
            self._apply_baseline_paths(jobs_n, jobs_bl)
        else:
            jobs_n = jobs_bl
        d_node = self.run_scenario("node", jobs_n, bl_stats,
                                   failure_rate, failure_time,
                                   rng_seed=self.seed+1000)

        # ── Link failure ──
        logger.info("\n" + "="*50)
        logger.info(">>> SCENARIO 2: Link Failure <<<")
        logger.info("="*50)
        if not plot_only:
            jobs_l = self._generate_jobs()
            self._apply_baseline_paths(jobs_l, jobs_bl)
        else:
            jobs_l = jobs_bl
        d_link = self.run_scenario("link", jobs_l, bl_stats,
                                   failure_rate, failure_time,
                                   rng_seed=self.seed+2000)

        # ── Plot ──
        self._plot_cdf(d_node, d_link, len(jobs_bl))

        # ── Summary ──
        logger.info("\n" + "="*70)
        logger.info("SUMMARY")
        logger.info("="*70)
        for label, d in [("Node failure", d_node), ("Link failure", d_link)]:
            vals = [r for _, r, _ in d['ratios']]
            dv = [r for _, r, _ in d['ratios'] if r > 1.0]
            avg = f"{np.mean(dv):.2f}" if dv else "N/A"
            logger.info(
                f"  {label}: {d['n_total_rescheduled']} rescheduled "
                f"({d['n_direct_affected']} direct + {d['n_post_delayed']} ripple), "
                f"max ratio={max(vals):.2f}, avg ratio (delayed)={avg}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    plot_only = "--plot-only" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--plot-only"]

    num_jobs     = int(args[0])   if len(args) > 0 else 1024
    failure_rate = float(args[1]) if len(args) > 1 else 0.01
    failure_time = float(args[2]) if len(args) > 2 else 0.0

    sim = OnlineFailureSimulator(num_jobs=num_jobs, seed=42)
    sim.run_all(failure_rate=failure_rate, failure_time=failure_time,
                plot_only=plot_only)
