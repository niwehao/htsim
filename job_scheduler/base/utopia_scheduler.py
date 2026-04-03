"""
Utopia Job Scheduling Algorithm (Algorithm 1) - 16K GPU Cluster Simulator

Strictly follows Algorithm 1:
  - Only resource: per-link residual capacity C̃_e(t)
  - Delay model: g(h_p, D_j, s_j, r_j) = transmission + propagation
  - Scheduling: iterate collectives → flows (concurrent within collective)

Key design:
  - Hash-based ECMP: each flow gets a deterministic path via hash(src, dst, flow_id)
  - Concurrent flows within a collective share link bandwidth fairly
  - Job sizes follow Poisson distribution (small jobs most frequent)

Topologies: Spine-Leaf, Dragonfly, OCS (selectable)
Job profiles: 128/64/32/16 GPUs with durations 46/25/14/8
"""

import numpy as np
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Flow:
    """Individual flow f_j within a collective."""
    flow_id: int
    src: int
    dst: int
    data_size: float   # MB
    path: List[Tuple[int, int]] = field(default_factory=list)
    start_time: float = 0.0
    rate: float = 0.0
    finish_time: float = 0.0


@dataclass
class Collective:
    """One collective F_i (AllToAll). Contains n*(n-1) flows."""
    collective_id: int
    flows: List[Flow]
    compute_time: float   # T_i: computation before this collective (time units)
    release_time: float = 0.0
    finish_time: float = 0.0


@dataclass
class Job:
    """A training job with m collectives {F_1, ..., F_m}."""
    job_id: int
    collectives: List[Collective]
    gpu_nodes: List[int]
    priority: int = 0
    submit_time: float = 0.0
    duration: float = 0.0


# =============================================================================
# Topology Abstract Base
# =============================================================================

class Topology(ABC):
    """
    Abstract topology.
    Only models what Algorithm 1 needs: C̃_e(t) per link.
    Routing uses hash-based ECMP (one deterministic path per flow).
    """

    def __init__(self, link_bw: float):
        self.link_bw = link_bw
        # C̃_e(t): link -> [(start, end, reserved_bw), ...]
        self.link_reservations: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = defaultdict(list)
        self.num_gpus = 0

    @abstractmethod
    def _get_all_paths(self, src: int, dst: int) -> List[List[Tuple[int, int]]]:
        """Return all ECMP candidate paths between src and dst."""
        pass

    def hash_flow_to_path(self, src: int, dst: int, flow_id: int) -> List[Tuple[int, int]]:
        """
        Hash-based ECMP: deterministic path selection using hash(src, dst, flow_id).
        Simulates switch ECMP hashing on (src_ip, dst_ip, src_port, dst_port).
        """
        paths = self._get_all_paths(src, dst)
        if len(paths) <= 1:
            return paths[0] if paths else []
        h = int(hashlib.md5(f"{src}:{dst}:{flow_id}".encode()).hexdigest(), 16)
        return paths[h % len(paths)]

    def get_residual_capacity(self, link: Tuple[int, int], t: float) -> float:
        """Algo1 核心状态: C̃_e(t) — 链路 e 在时刻 t 的剩余带宽"""
        entries = self.link_reservations[link]
        reserved = 0.0
        # Compact: remove expired entries (end <= t) during scan
        alive = []
        for (s, e, bw) in entries:
            if e <= t:
                continue  # expired, skip
            alive.append((s, e, bw))
            if s <= t:
                reserved += bw
        if len(alive) < len(entries):
            self.link_reservations[link] = alive
        return max(0.0, self.link_bw - reserved)

    def get_min_residual_on_path(self, path_links: List[Tuple[int, int]], t: float) -> float:
        """Algo1 Line 7: r_j = min{C̃_e(S_i + t_j)}, ∀e ∈ p — bottleneck rate on path"""
        if not path_links:
            return self.link_bw
        return min(self.get_residual_capacity(lk, t) for lk in path_links)

    def reserve_bandwidth(self, path_links: List[Tuple[int, int]],
                          start: float, duration: float, bw: float):
        """Algo1 Lines 14-16: for each e ∈ p*, deduct r* from C̃_e(t) at S_i+t* for duration δ*-t*"""
        end = start + duration
        for lk in path_links:
            self.link_reservations[lk].append((start, end, bw))

    def get_next_capacity_change(self, path_links: List[Tuple[int, int]],
                                  after: float) -> Optional[float]:
        """Next time when capacity changes on any link in path."""
        nxt = None
        for lk in path_links:
            for (s, e, bw) in self.link_reservations[lk]:
                if e > after:
                    if nxt is None or e < nxt:
                        nxt = e
        return nxt


# =============================================================================
# Spine-Leaf Topology (ref: spine-leaf-n/constants.h)
# =============================================================================

class SpineLeafTopology(Topology):
    """
    2-layer Spine-Leaf for 16K GPUs.
    ports_per_switch=64: gpus_per_leaf=32, num_leaves=512, num_spines=32
    Path: same leaf → 2 hops; different leaf → 4 hops (GPU-Leaf-Spine-Leaf-GPU)

    Hardware config (由代码全连通假设反推):
      - Leaf 交换机: 64 端口 (32 下行→GPU + 32 上行→Spine), 512 台
      - Spine 交换机: 512 端口 (512 下行→Leaf), 32 台  ← 非对称, Spine 远大于 Leaf
      - 链路带宽: 400 Gbps
      - 代码假设任意 Leaf↔Spine 全连通 (每个 Spine 连接全部 512 个 Leaf)
    """

    def __init__(self, num_gpus: int = 16384, ports_per_switch: int = 64,
                 link_bw: float = 400.0):
        super().__init__(link_bw)
        self.num_gpus = num_gpus
        self.ports_per_switch = ports_per_switch
        self.gpus_per_leaf = ports_per_switch // 2
        self.num_leaves = num_gpus // self.gpus_per_leaf
        self.num_spines = ports_per_switch // 2
        self.leaf_offset = num_gpus
        self.spine_offset = num_gpus + self.num_leaves

        logger.info(f"[Spine-Leaf] {num_gpus} GPUs, {self.num_leaves} Leaves, "
                     f"{self.num_spines} Spines, {link_bw} Gbps/link")

    def _gpu_to_leaf(self, gpu: int) -> int:
        return self.leaf_offset + gpu // self.gpus_per_leaf

    def _get_all_paths(self, src: int, dst: int) -> List[List[Tuple[int, int]]]:
        if src == dst:
            return [[]]
        src_leaf = self._gpu_to_leaf(src)
        dst_leaf = self._gpu_to_leaf(dst)

        if src_leaf == dst_leaf:
            return [[(src, src_leaf), (src_leaf, dst)]]

        # All ECMP paths: one per spine
        paths = []
        for s in range(self.num_spines):
            spine = self.spine_offset + s
            paths.append([
                (src, src_leaf), (src_leaf, spine),
                (spine, dst_leaf), (dst_leaf, dst)
            ])
        return paths


# =============================================================================
# Dragonfly Topology (ref: dragonfly_n/constants.h)
# =============================================================================

class DragonflyTopology(Topology):
    """
    Dragonfly with balanced (p, a, h, g) auto-derived:
      k = p + (a-1) + h, N = p*a*g, h >= g-1, minimize |a - 2h|

    Hardware config (ports_per_switch=128 时自动推导):
      - 交换机: 128 端口, 512 台, 均匀配置
        p=32 (32 端口→GPU) + a-1=63 (63 组内链路) + h=33 (33 全局链路) = 128 端口  ✅ 自洽
      - 8 个 group, 每 group 64 台交换机, 每 group 2048 GPU
      - 链路带宽: 400 Gbps
    """

    def __init__(self, num_gpus: int = 16384, ports_per_switch: int = 32,
                 link_bw: float = 400.0):
        super().__init__(link_bw)
        self.num_gpus = num_gpus
        self.ports_per_switch = ports_per_switch
        self.p, self.a, self.h, self.g = self._auto_derive(num_gpus, ports_per_switch)
        self.num_switches = num_gpus // self.p
        self.gpus_per_group = self.a * self.p
        self.sw_offset = num_gpus

        logger.info(f"[Dragonfly] {num_gpus} GPUs, p={self.p} a={self.a} "
                     f"h={self.h} g={self.g}, {self.num_switches} switches, "
                     f"{link_bw} Gbps/link")

    @staticmethod
    def _auto_derive(N: int, k: int) -> Tuple[int, int, int, int]:
        best = (0, 0, 0, 0)
        best_score = 99999
        for a in range(2, k):
            for h in range(1, k):
                if (a - 1) + h >= k:
                    break
                p = k - (a - 1) - h
                if p < 1 or N % p != 0:
                    continue
                num_sw = N // p
                if num_sw % a != 0:
                    continue
                g = num_sw // a
                if g < 2 or h < g - 1:
                    continue
                score = abs(a - 2 * h)
                if score < best_score or (score == best_score and p > best[0]):
                    best = (p, a, h, g)
                    best_score = score
        assert best[1] >= 2, f"Cannot derive Dragonfly for N={N}, k={k}"
        return best

    def _gpu_to_switch(self, gpu: int) -> int:
        return self.sw_offset + gpu // self.p

    def _switch_to_group(self, sw: int) -> int:
        return (sw - self.sw_offset) // self.a

    def _get_all_paths(self, src: int, dst: int) -> List[List[Tuple[int, int]]]:
        if src == dst:
            return [[]]

        src_sw = self._gpu_to_switch(src)
        dst_sw = self._gpu_to_switch(dst)
        src_grp = self._switch_to_group(src_sw)
        dst_grp = self._switch_to_group(dst_sw)

        if src_sw == dst_sw:
            return [[(src, src_sw), (src_sw, dst)]]

        if src_grp == dst_grp:
            return [[(src, src_sw), (src_sw, dst_sw), (dst_sw, dst)]]

        # Different groups: enumerate paths via different intermediate switches
        paths = []
        for i in range(self.a):
            interm_sw = self.sw_offset + src_grp * self.a + i
            for j in range(self.a):
                land_sw = self.sw_offset + dst_grp * self.a + j
                if interm_sw == src_sw and land_sw == dst_sw:
                    paths.append([
                        (src, src_sw), (src_sw, dst_sw), (dst_sw, dst)
                    ])
                elif interm_sw == src_sw:
                    paths.append([
                        (src, src_sw), (src_sw, land_sw),
                        (land_sw, dst_sw), (dst_sw, dst)
                    ])
                elif land_sw == dst_sw:
                    paths.append([
                        (src, src_sw), (src_sw, interm_sw),
                        (interm_sw, dst_sw), (dst_sw, dst)
                    ])
                else:
                    paths.append([
                        (src, src_sw), (src_sw, interm_sw),
                        (interm_sw, land_sw), (land_sw, dst_sw), (dst_sw, dst)
                    ])
        return paths if paths else [[(src, src_sw), (src_sw, dst_sw), (dst_sw, dst)]]


# =============================================================================
# OCS Topology (ref: ocs_syn/constants.h, hoho_routing.h, hohoo_scheduler.h)
# =============================================================================

class OCSTopology(Topology):
    """
    Optical Circuit Switch topology for 16K GPUs.

    Reference: ocs_syn/ — Circle Method (round-robin tournament) scheduling,
    128 GPUs per OCS pod, 200 Gbps per port.

    Structure:
      - 128 pods × 128 GPUs/pod = 16384 GPUs
      - Intra-pod: single OCS switch, non-blocking circuit for each pair
        Path: GPU → OCS_pod → GPU (2 links, 1 hop through OCS)
        No ECMP (OCS provides a single dedicated circuit per pair)
      - Inter-pod: 32 electrical spine switches for cross-pod traffic
        Path: GPU → OCS_src → Spine → OCS_dst → GPU (4 links)
        ECMP across 32 spines

    Hardware config (由代码全连通假设反推):
      - OCS 交换机: 160 端口 (128 下行→GPU + 32 上行→Spine), 128 台
        ← 超过参考代码的 128 端口, 代码假设 OCS↔Spine 全连通
      - Spine 交换机: 128 端口 (128 下行→OCS), 32 台  ✅ 刚好 128 端口
      - 链路带宽: 200 Gbps

    Bandwidth model for Algorithm 1:
      - Each GPU port: 200 Gbps uplink to OCS
      - OCS is non-blocking internally (no internal contention)
      - Fair-share on GPU uplink naturally models OCS time-division:
        if GPU has k concurrent flows, each gets 200/k Gbps
        (equivalent to each flow getting full 200 Gbps for 1/k of cycle)
      - Inter-pod spine links: 200 Gbps each
    """

    def __init__(self, num_gpus: int = 16384, gpus_per_pod: int = 128,
                 num_spines: int = 32, link_bw: float = 200.0):
        super().__init__(link_bw)
        self.num_gpus = num_gpus
        self.gpus_per_pod = gpus_per_pod
        self.num_pods = num_gpus // gpus_per_pod
        self.num_spines = num_spines

        # Node ID offsets: GPUs [0, num_gpus), OCS [ocs_offset, ...), Spines [spine_offset, ...)
        self.ocs_offset = num_gpus
        self.spine_offset = num_gpus + self.num_pods

        assert num_gpus % gpus_per_pod == 0, \
            f"num_gpus={num_gpus} not divisible by gpus_per_pod={gpus_per_pod}"

        logger.info(f"[OCS] {num_gpus} GPUs, {self.num_pods} pods × {gpus_per_pod} GPUs/pod, "
                     f"{num_spines} spines, {link_bw} Gbps/link")

    def _gpu_to_pod(self, gpu: int) -> int:
        return gpu // self.gpus_per_pod

    def _gpu_to_ocs(self, gpu: int) -> int:
        return self.ocs_offset + self._gpu_to_pod(gpu)

    def _get_all_paths(self, src: int, dst: int) -> List[List[Tuple[int, int]]]:
        if src == dst:
            return [[]]

        src_ocs = self._gpu_to_ocs(src)
        dst_ocs = self._gpu_to_ocs(dst)

        if src_ocs == dst_ocs:
            # Intra-pod: GPU → OCS → GPU (non-blocking circuit, single path)
            return [[(src, src_ocs), (src_ocs, dst)]]

        # Inter-pod: GPU → OCS_src → Spine → OCS_dst → GPU
        # ECMP across spine switches
        paths = []
        for s in range(self.num_spines):
            spine = self.spine_offset + s
            paths.append([
                (src, src_ocs), (src_ocs, spine),
                (spine, dst_ocs), (dst_ocs, dst)
            ])
        return paths


# =============================================================================
# Delay Model: g(h_p, D_j, s_j, r_j)  — Algorithm 1, Line 8
# =============================================================================

def compute_delay(num_hops: int, data_size_mb: float, rate_gbps: float,
                  prop_delay_per_hop: float = 0.5) -> float:
    """
    Algo1 Line 8: g(h_p, D_j, s_j, r_j) = transmission_time + propagation_delay.
      - h_p = num_hops (path length)
      - D_j = data_size_mb (flow data volume)
      - r_j = rate_gbps (allocated bandwidth)
    Returns: delay in μs (not including wait time t_j, that's added by caller).
    """
    if rate_gbps <= 0:
        return float('inf')
    transmission = (data_size_mb * 8.0 * 1000.0 / rate_gbps)  # μs
    propagation = num_hops * prop_delay_per_hop
    return transmission + propagation


# =============================================================================
# Utopia Scheduler — Algorithm 1, with concurrent fair-share
# =============================================================================

class UtopiaScheduler:
    """
    Strict Algorithm 1 implementation:
      - Line 3: flows scheduled SEQUENTIALLY (not concurrently)
      - Line 4: enumerate ALL candidate paths per flow
      - Line 7: try different wait times t_j >= 0 (wait optimization)
      - Line 14-15: update C̃_e(t) after each flow
      - Line 17: S_i accumulates per flow
    """

    def __init__(self, topo: Topology, max_wait_probes: int = 10):
        self.topo = topo
        self.max_wait_probes = max_wait_probes

    def _find_wait_times(self, path_links: List[Tuple[int, int]],
                         S_i: float) -> List[float]:
        """
        Algo1 Line 7 helper: enumerate candidate t_j values where C̃_e changes.
        Each t_j corresponds to a reservation expiring on some link in the path,
        which may increase residual capacity → higher r_j → lower total delay.
        Returns sorted list of t_j ≥ 0 to try (always includes t_j=0).
        """
        wait_times = [0.0]
        t = S_i
        for _ in range(self.max_wait_probes):
            nxt = self.topo.get_next_capacity_change(path_links, t)
            if nxt is None:
                break
            t_j = nxt - S_i
            if t_j > 0:
                wait_times.append(t_j)
            t = nxt
        return wait_times

    def schedule_collective(self, coll: Collective, S_i: float) -> float:
        """
        Algorithm 1, Lines 3-19: schedule all flows in one collective.
        Flows are scheduled SEQUENTIALLY — each flow sees updated C̃_e(t)
        from previously scheduled flows. S_i advances after each flow.
        """
        flows = coll.flows
        if not flows:
            coll.finish_time = S_i
            return S_i

        # ── Algo1 Line 3: for j = 1 to n do ──
        for f in flows:
            if f.src == f.dst:
                f.start_time = S_i
                f.finish_time = S_i
                f.rate = 0
                continue

            # ── Algo1 Line 4: P_j ← PATHS(src_j, dst_j) ──
            all_paths = self.topo._get_all_paths(f.src, f.dst)

            # ── Algo1 Line 5: δ*←+∞; p*←null; r*←0; t*←0 ──
            best_delta = float('inf')   # δ*
            best_path = all_paths[0] if all_paths else []  # p*
            best_rate = 0.001           # r*
            best_t = 0.0               # t*

            # ── Algo1 Line 6: for each path p ∈ P_j do ──
            for p in all_paths:
                if not p:
                    continue
                # ── Algo1 Line 7: for each r_j ← min{C̃_e(S_i + t_j)}, t_j ≥ 0 ──
                # Try different wait times where residual capacity changes
                wait_times = self._find_wait_times(p, S_i)
                for t_j in wait_times:
                    r_j = self.topo.get_min_residual_on_path(p, S_i + t_j)
                    if r_j <= 0:
                        continue
                    # ── Algo1 Line 8: δ_{i,p} ← g(h_p, D_j, s_j, r_j) + t_j ──
                    hops = len(p)
                    delta = compute_delay(hops, f.data_size, r_j) + t_j
                    # ── Algo1 Lines 9-11: if δ_{i,p} < δ* then update best ──
                    if delta < best_delta:
                        best_delta = delta   # δ* ← δ_{i,p}
                        best_path = p        # p* ← p
                        best_rate = r_j      # r* ← r_j
                        best_t = t_j         # t* ← t_j
            # ── Algo1 Lines 12-13: end for (paths × wait times) ──

            # ── Algo1 Line 18: Schedule f_j to start at t* with speed r* over path p* ──
            f.path = best_path
            f.start_time = S_i + best_t          # actual start = S_i + t*
            f.rate = max(best_rate, 0.001)
            duration = best_delta - best_t if best_delta < float('inf') else 0  # δ* - t*
            f.finish_time = S_i + best_delta if best_delta < float('inf') else S_i

            # ── Algo1 Lines 14-16: Deduct r* from C̃_e(t) at S_i+t* for duration δ*-t* ──
            if duration > 0 and best_path:
                self.topo.reserve_bandwidth(best_path, S_i + best_t, duration, f.rate)

            # ── Algo1 Line 17: S_i ← S_i + δ*  (advance time cursor for next flow) ──
            S_i = S_i + best_delta if best_delta < float('inf') else S_i

        # ── Algo1 Line 19: end for (flows) ──
        coll.finish_time = S_i
        return S_i

    def schedule_job(self, job: Job) -> float:
        """Algorithm 1, Lines 1-20: schedule one job's m collectives."""
        # ── Algo1 Line 1: for i = 1 to m do ──
        for i, coll in enumerate(job.collectives):
            # ── Algo1 Line 2: S_i ← 0 if i==1; else S_i ← S_{i-1} + T_i ──
            if i == 0:
                S_i = job.submit_time       # first collective: release at submit time
            else:
                S_i = job.collectives[i - 1].finish_time + coll.compute_time  # S_{i-1} + T_i
            coll.release_time = S_i

            # ── Algo1 Lines 3-19: schedule all n flows in collective F_i ──
            self.schedule_collective(coll, S_i)
        # ── Algo1 Line 20: end for (collectives) ──

        return job.collectives[-1].finish_time

    def schedule_all_jobs(self, jobs: List[Job]) -> Dict:
        sorted_jobs = sorted(jobs, key=lambda j: (-j.priority, j.submit_time))
        results = []
        for job in tqdm(sorted_jobs, desc="Utopia"):
            finish = self.schedule_job(job)
            jct = finish - job.submit_time
            results.append({
                'job_id': job.job_id,
                'num_gpus': len(job.gpu_nodes),
                'duration': job.duration,
                'jct': jct,
                'submit_time': job.submit_time,
                'finish_time': finish,
                'num_flows': sum(len(c.flows) for c in job.collectives),
            })
        jcts = [r['jct'] for r in results]
        return {
            'num_jobs': len(jobs),
            'total_flows': sum(r['num_flows'] for r in results),
            'avg_jct': np.mean(jcts), 'median_jct': np.median(jcts),
            'p95_jct': np.percentile(jcts, 95), 'p99_jct': np.percentile(jcts, 99),
            'max_jct': np.max(jcts), 'min_jct': np.min(jcts),
            'job_details': results,
        }


# =============================================================================
# FIFO Baseline: FIFO order, hash-ECMP routing, concurrent fair-share
# =============================================================================

class FIFOScheduler:
    """
    FIFO baseline: same concurrent fair-share model, but:
    - Jobs scheduled in FIFO (submit_time) order, no priority
    - No wait optimization (always start immediately)
    """

    def __init__(self, topo: Topology):
        self.topo = topo

    def schedule_collective(self, coll: Collective, S_i: float):
        flows = coll.flows
        if not flows:
            coll.finish_time = S_i
            return

        for f in flows:
            f.path = self.topo.hash_flow_to_path(f.src, f.dst, f.flow_id)

        link_flow_count: Dict[Tuple[int, int], int] = defaultdict(int)
        for f in flows:
            for lk in f.path:
                link_flow_count[lk] += 1

        # Batch query residual + pre-compute fair share
        link_fair_share: Dict[Tuple[int, int], float] = {}
        for lk in link_flow_count:
            residual = self.topo.get_residual_capacity(lk, S_i)
            link_fair_share[lk] = residual / link_flow_count[lk]

        max_finish = S_i
        reservations = []

        for f in flows:
            if not f.path:
                f.start_time = S_i
                f.finish_time = S_i
                f.rate = 0
                continue

            rate = min(link_fair_share[lk] for lk in f.path)
            rate = max(rate, 0.001)
            delta = compute_delay(len(f.path), f.data_size, rate)
            f.start_time = S_i
            f.rate = rate
            f.finish_time = S_i + delta
            if delta > 0:
                reservations.append((f.path, S_i, delta, rate))
            max_finish = max(max_finish, f.finish_time)

        for path, start, dur, bw in reservations:
            self.topo.reserve_bandwidth(path, start, dur, bw)

        coll.finish_time = max_finish

    def schedule_job(self, job: Job) -> float:
        for i, coll in enumerate(job.collectives):
            S_i = job.submit_time if i == 0 else \
                job.collectives[i - 1].finish_time + coll.compute_time
            coll.release_time = S_i
            self.schedule_collective(coll, S_i)
        return job.collectives[-1].finish_time

    def schedule_all_jobs(self, jobs: List[Job]) -> Dict:
        sorted_jobs = sorted(jobs, key=lambda j: j.submit_time)
        results = []
        for job in tqdm(sorted_jobs, desc="FIFO"):
            finish = self.schedule_job(job)
            jct = finish - job.submit_time
            results.append({
                'job_id': job.job_id,
                'num_gpus': len(job.gpu_nodes),
                'duration': job.duration,
                'jct': jct,
                'submit_time': job.submit_time,
                'finish_time': finish,
                'num_flows': sum(len(c.flows) for c in job.collectives),
            })
        jcts = [r['jct'] for r in results]
        return {
            'num_jobs': len(jobs),
            'total_flows': sum(r['num_flows'] for r in results),
            'avg_jct': np.mean(jcts), 'median_jct': np.median(jcts),
            'p95_jct': np.percentile(jcts, 95), 'p99_jct': np.percentile(jcts, 99),
            'max_jct': np.max(jcts), 'min_jct': np.min(jcts),
            'job_details': results,
        }


# =============================================================================
# Job Generator — Poisson-distributed job counts
# =============================================================================

# (num_gpus, duration_time_units)
JOB_PROFILES = [
    (16,   8),
    (32,  14),
    (64,  25),
    (128, 46),
]

# Poisson lambda for each profile: small jobs most frequent, large jobs least
# Expected counts (for 200 total): ~80, ~60, ~40, ~20
POISSON_LAMBDAS = [4.0, 3.0, 2.0, 1.0]


class WorkloadGenerator:
    """Generate AllToAll jobs with Poisson-distributed profile counts."""

    def __init__(self, num_gpus: int, rng_seed: int = 42):
        self.num_gpus = num_gpus
        self.rng = np.random.RandomState(rng_seed)

    def _alltoall_flows(self, gpus: List[int], data_size_mb: float) -> List[Flow]:
        """
        AllToAll: every GPU sends to every other GPU.
        Per-pair data = data_size / n^2, total flows = n*(n-1).
        """
        n = len(gpus)
        per_pair = data_size_mb / (n * n)
        flows = []
        fid = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    flows.append(Flow(flow_id=fid, src=gpus[i], dst=gpus[j],
                                      data_size=per_pair))
                    fid += 1
        return flows

    def generate_jobs(self, total_jobs: int = 200) -> List[Job]:
        """
        Generate jobs with Poisson-distributed counts per profile.
        Small jobs (16 GPU) are most frequent, large (128 GPU) least.
        Uses λ as expected proportions, then adds Poisson noise per profile.
        """
        # Use lambdas as base proportions + Poisson noise for variation
        total_lam = sum(POISSON_LAMBDAS)
        counts = []
        for lam in POISSON_LAMBDAS:
            base = lam / total_lam * total_jobs          # expected count
            noise = self.rng.poisson(lam) - lam          # zero-mean noise
            counts.append(max(1, int(round(base + noise))))
        # Adjust to hit exact total
        diff = total_jobs - sum(counts)
        counts[0] += diff  # add/subtract from smallest profile

        jobs = []
        job_id = 0
        for profile_idx, (num_gpus, duration) in enumerate(JOB_PROFILES):
            for _ in range(counts[profile_idx]):
                max_start = max(0, self.num_gpus - num_gpus)
                start_gpu = self.rng.randint(0, max_start + 1)
                gpus = list(range(start_gpu, start_gpu + num_gpus))

                num_collectives = max(1, duration // 10)
                collectives = []
                for c_id in range(num_collectives):
                    data_size = self.rng.uniform(50, 200)  # MB
                    compute_time = duration / num_collectives
                    flows = self._alltoall_flows(gpus, data_size)
                    collectives.append(Collective(
                        collective_id=c_id, flows=flows,
                        compute_time=compute_time))

                submit_time = self.rng.uniform(0, 100)
                jobs.append(Job(
                    job_id=job_id, collectives=collectives, gpu_nodes=gpus,
                    priority=self.rng.randint(0, 3), submit_time=submit_time,
                    duration=duration))
                job_id += 1

        logger.info(f"Job distribution: "
                     + ", ".join(f"{JOB_PROFILES[i][0]}GPU={counts[i]}"
                                for i in range(len(JOB_PROFILES))))
        return jobs


# =============================================================================
# Visualization
# =============================================================================

def plot_results(utopia_stats, fifo_stats, topo_name, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Utopia vs FIFO — 16K GPU Cluster ({topo_name}, AllToAll)",
                 fontsize=14, fontweight='bold')

    # 1) JCT CDF
    ax = axes[0, 0]
    for stats, label, color in [(utopia_stats, 'Utopia', 'blue'),
                                 (fifo_stats, 'FIFO', 'red')]:
        jcts = sorted([j['jct'] for j in stats['job_details']])
        cdf = np.arange(1, len(jcts) + 1) / len(jcts)
        ax.plot(jcts, cdf, label=label, color=color, linewidth=2)
    ax.set_xlabel('Job Completion Time (us)')
    ax.set_ylabel('CDF')
    ax.set_title('JCT Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) Bar chart: avg/median/p95/p99
    ax = axes[0, 1]
    metrics = ['avg_jct', 'median_jct', 'p95_jct', 'p99_jct']
    labels = ['Avg', 'Median', 'P95', 'P99']
    u_vals = [utopia_stats[m] for m in metrics]
    f_vals = [fifo_stats[m] for m in metrics]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, u_vals, w, label='Utopia', color='steelblue')
    ax.bar(x + w / 2, f_vals, w, label='FIFO', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('JCT (us)')
    ax.set_title('JCT Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3) JCT by job size (box plot)
    ax = axes[1, 0]
    for stats, label, color, offset in [(utopia_stats, 'Utopia', 'steelblue', -0.15),
                                         (fifo_stats, 'FIFO', 'salmon', 0.15)]:
        by_size = defaultdict(list)
        for j in stats['job_details']:
            by_size[j['num_gpus']].append(j['jct'])
        sizes = sorted(by_size.keys())
        positions = [i + offset for i in range(len(sizes))]
        data = [by_size[s] for s in sizes]
        bp = ax.boxplot(data, positions=positions, widths=0.25,
                        patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_xticks(range(len(sorted(by_size.keys()))))
    ax.set_xticklabels([f"{s} GPUs" for s in sorted(by_size.keys())])
    ax.set_ylabel('JCT (us)')
    ax.set_title('JCT by Job Size')
    ax.legend(['Utopia', 'FIFO'])
    ax.grid(True, alpha=0.3, axis='y')

    # 4) Speedup by profile
    ax = axes[1, 1]
    u_map = {j['job_id']: j for j in utopia_stats['job_details']}
    f_map = {j['job_id']: j for j in fifo_stats['job_details']}
    profile_speedups = defaultdict(list)
    for jid in u_map:
        if jid in f_map and u_map[jid]['jct'] > 0:
            sp = f_map[jid]['jct'] / u_map[jid]['jct']
            profile_speedups[u_map[jid]['num_gpus']].append(sp)
    gpu_labels = []
    avg_speedups = []
    for gpus, _ in JOB_PROFILES:
        if gpus in profile_speedups:
            gpu_labels.append(f"{gpus} GPUs")
            avg_speedups.append(np.mean(profile_speedups[gpus]))
    if avg_speedups:
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        ax.bar(range(len(gpu_labels)), avg_speedups,
               color=colors[:len(gpu_labels)])
        ax.set_xticks(range(len(gpu_labels)))
        ax.set_xticklabels(gpu_labels)
        ax.set_ylabel('Avg Speedup (FIFO/Utopia)')
        ax.set_title('Speedup by Job Size')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def make_topo(topo_type: str) -> Topology:
    if topo_type == "spine-leaf":
        return SpineLeafTopology(num_gpus=16384, ports_per_switch=64, link_bw=400.0)
    elif topo_type == "dragonfly":
        return DragonflyTopology(num_gpus=16384, ports_per_switch=128, link_bw=400.0)
    elif topo_type == "ocs":
        return OCSTopology(num_gpus=16384, gpus_per_pod=128, num_spines=32, link_bw=200.0)
    else:
        raise ValueError(f"Unknown topology: {topo_type}")


def run_simulation(topo_type: str = "spine-leaf", num_jobs: int = 200, seed: int = 42):
    logger.info("=" * 70)
    logger.info(f"UTOPIA SIMULATION — 16K GPUs — {topo_type} — {num_jobs} jobs — AllToAll")
    logger.info("=" * 70)

    # Build topology (just for info)
    t0 = time.time()
    topo_info = make_topo(topo_type)
    logger.info(f"Topology info built in {time.time() - t0:.2f}s")

    # Generate workload
    wg = WorkloadGenerator(topo_info.num_gpus, rng_seed=seed)
    jobs = wg.generate_jobs(num_jobs)
    total_flows = sum(sum(len(c.flows) for c in j.collectives) for j in jobs)
    logger.info(f"Generated {len(jobs)} jobs, {total_flows} total flows")

    profile_cnt = Counter(len(j.gpu_nodes) for j in jobs)
    for gpus in sorted(profile_cnt):
        logger.info(f"  {gpus}-GPU jobs: {profile_cnt[gpus]}")

    # --- Utopia ---
    # Re-generate identical jobs (same seed) instead of deepcopy (much faster)
    logger.info("Generating Utopia workload...")
    wg_u = WorkloadGenerator(topo_info.num_gpus, rng_seed=seed)
    jobs_u = wg_u.generate_jobs(num_jobs)
    topo_u = make_topo(topo_type)
    t0 = time.time()
    utopia = UtopiaScheduler(topo_u)
    u_stats = utopia.schedule_all_jobs(jobs_u)
    u_time = time.time() - t0
    logger.info(f"Utopia done in {u_time:.2f}s")

    # --- FIFO ---
    logger.info("Generating FIFO workload...")
    wg_f = WorkloadGenerator(topo_info.num_gpus, rng_seed=seed)
    jobs_f = wg_f.generate_jobs(num_jobs)
    topo_f = make_topo(topo_type)
    t0 = time.time()
    fifo = FIFOScheduler(topo_f)
    f_stats = fifo.schedule_all_jobs(jobs_f)
    f_time = time.time() - t0
    logger.info(f"FIFO done in {f_time:.2f}s")

    # --- Results ---
    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS — {topo_type}")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<25} {'Utopia':>15} {'FIFO':>15} {'Improvement':>12}")
    logger.info("-" * 70)
    for key, label in [('avg_jct', 'Avg JCT (us)'),
                        ('median_jct', 'Median JCT (us)'),
                        ('p95_jct', 'P95 JCT (us)'),
                        ('p99_jct', 'P99 JCT (us)'),
                        ('max_jct', 'Max JCT (us)')]:
        u, f = u_stats[key], f_stats[key]
        imp = ((f - u) / f * 100) if f > 0 else 0
        logger.info(f"{label:<25} {u:>15.1f} {f:>15.1f} {imp:>11.1f}%")

    logger.info(f"\nTotal flows: {u_stats['total_flows']}")
    logger.info(f"Time: Utopia={u_time:.2f}s  FIFO={f_time:.2f}s")

    out = f"schedule_results_{topo_type}.png"
    plot_results(u_stats, f_stats, topo_type, out)

    return u_stats, f_stats


if __name__ == "__main__":
    import sys
    topo = sys.argv[1] if len(sys.argv) > 1 else "spine-leaf"
    num_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    run_simulation(topo_type=topo, num_jobs=num_jobs, seed=42)
