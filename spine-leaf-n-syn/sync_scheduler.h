/*
 * sync_scheduler.h  —  512-GPU Leaf-Spine 同步预调度 (v13)
 *
 * v12 crash 根因:
 *   ECMP 映射 spine = (ps + pd) % S 构成 Latin Square.
 *   同一 leaf 多个 GPU 发往同一 spine 时, 共享 srcLeaf egressQ (per uplink port).
 *   egressQ 排水 = dtPort >> dtFabric → 调度器低估 spine 到达时间 → 堆积 crash.
 *
 * v13 修复: 增加 leafEgFree[srcLeaf][spine] 追踪 srcLeaf egressQ.
 *   确保同一 (srcLeaf, spine) 的两个包 sendTime 间隔 ≥ dtPort
 *   → egressQ 无排队 → pipeline timing 确定 → spine 预测准确
 *
 * 完整 pipeline (跨 leaf):
 *   T → txQ(dtPort) → ingressQ(dtPort) → BufferGate → fabricQ(dtFabric)
 *     → BufferRelease → egressQ(dtPort) → spine_ingressQ(dtPort)
 *     → spine_BufferGate → spine_fabricQ(dtFabric) → ...
 *
 *   设 T = sendTime:
 *     srcLeaf fabricQ 到达:  T + 2·dtPort
 *     srcLeaf egressQ 到达:  T + 2·dtPort + dtFabric      ← NEW: 追踪这里
 *     spine BufferGate 到达: T + 4·dtPort + dtFabric
 *     spine fabricQ 到达:    T + 4·dtPort + dtFabric       (same, BG is pass-through)
 *
 * 资源追踪 (5 类):
 *   txFree[gpu]                  — NIC 带宽
 *   fabLeafFree[srcLeaf]         — srcLeaf fabricQ
 *   leafEgFree[srcLeaf][spine]   — srcLeaf egressQ per uplink (NEW)
 *   fabSpineFree[spine]          — spine fabricQ
 *   (dstLeaf 不追踪, BufferGate 4·pktSize 容忍)
 *
 * 利用率影响:
 *   leafEgFree 约束同一 (leaf,spine) 间隔 ≥ dtPort.
 *   但每 leaf 有 NUM_SPINES 个 uplink port, 轮转使用, 很少连续命中同一 spine.
 *   预期利用率损失 < 2%.
 */

#pragma once

#include "constants.h"

#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <iomanip>

struct ScheduleEntry {
    simtime_picosec sendTime;
    uint16_t        dstGpu;
    uint16_t        fragId;
};

class SyncScheduler {
public:
    static simtime_picosec psPerByteLink() {
        return (simtime_picosec)((1.0e12 * 8) / (PROT_RATE_Gbps * 1.0e9));
    }
    static simtime_picosec psPerByteFabric() {
        return (simtime_picosec)((1.0e12 * 8) / (PROT_RATE_Gbps * PORTS_PER_SWITCH * 1.0e9));
    }

    // ================================================================
    //  buildInterleavedDstOrder
    // ================================================================
    static std::vector<uint32_t> buildInterleavedDstOrder(uint32_t srcGpu) {
        uint32_t srcLeaf = gpuToLeaf(srcGpu);
        std::vector<uint32_t> order;
        order.reserve(NUM_GPU_NODES - 1);

        std::vector<uint32_t> remoteLeaves;
        for (uint32_t l = 0; l < NUM_LEAVES; l++)
            if (l != srcLeaf) remoteLeaves.push_back(l);

        std::vector<uint32_t> localDsts;
        for (uint32_t p = 0; p < GPUS_PER_LEAF; p++) {
            uint32_t g = srcLeaf * GPUS_PER_LEAF + p;
            if (g != srcGpu) localDsts.push_back(g);
        }

        uint32_t numRemote = (uint32_t)remoteLeaves.size();
        uint32_t localIdx = 0, round = 0;

        while (order.size() < NUM_GPU_NODES - 1) {
            for (uint32_t li = 0; li < numRemote && order.size() < NUM_GPU_NODES - 1; li++) {
                uint32_t leaf = remoteLeaves[li];
                uint32_t portOffset = (round * numRemote + li) % GPUS_PER_LEAF;
                order.push_back(leaf * GPUS_PER_LEAF + portOffset);
            }
            if (localIdx < localDsts.size())
                order.push_back(localDsts[localIdx++]);
            round++;
        }
        order.resize(NUM_GPU_NODES - 1);
        return order;
    }

    // ================================================================
    //  buildRotatedDstOrder — 旋转 srcGpu 个位置, 分散 rxQ 负载
    // ================================================================
    static std::vector<uint32_t> buildRotatedDstOrder(uint32_t srcGpu) {
        auto order = buildInterleavedDstOrder(srcGpu);
        uint32_t n = (uint32_t)order.size();
        if (n == 0) return order;
        uint32_t offset = srcGpu % n;
        std::rotate(order.begin(), order.begin() + offset, order.end());
        return order;
    }

    // ================================================================
    //  buildLeafInterleavedSrcOrder
    // ================================================================
    static std::vector<uint32_t> buildLeafInterleavedSrcOrder() {
        std::vector<uint32_t> order;
        order.reserve(NUM_GPU_NODES);
        for (uint32_t p = 0; p < GPUS_PER_LEAF; p++)
            for (uint32_t l = 0; l < NUM_LEAVES; l++)
                order.push_back(l * GPUS_PER_LEAF + p);
        return order;
    }

    // ================================================================
    //  computePhaseSchedule (v13)
    //
    //  约束 (跨 leaf):
    //    T ≥ txFree[src]                                        (NIC)
    //    T ≥ fabLeafFree[srcLeaf] - 2·dtPort                   (srcLeaf fabricQ)
    //    T ≥ leafEgFree[srcLeaf][spine] - 2·dtPort - dtFabric  (srcLeaf egressQ) ← NEW
    //    T ≥ fabSpineFree[spine] - 4·dtPort - dtFabric          (spine fabricQ)
    //
    //  更新 (跨 leaf):
    //    txFree[src]                    = T + dtPort
    //    fabLeafFree[srcLeaf]           = T + 2·dtPort + dtFabric
    //    leafEgFree[srcLeaf][spine]     = T + 3·dtPort + dtFabric   ← NEW
    //    fabSpineFree[spine]            = T + 4·dtPort + 2·dtFabric
    // ================================================================
    static std::map<uint32_t, std::vector<ScheduleEntry>>
    computePhaseSchedule(int pktSize, simtime_picosec* outDuration = nullptr) {

        simtime_picosec dtPort   = (simtime_picosec)pktSize * psPerByteLink();
        simtime_picosec dtFabric = (simtime_picosec)pktSize * psPerByteFabric();

        std::cout << "  === Scheduler v13: pipeline-aware + egressQ tracked ===\n"
                  << "  Port drain:    " << std::fixed << std::setprecision(1)
                  << (double)dtPort / 1000.0 << " ns\n"
                  << "  Fabric drain:  " << std::setprecision(2)
                  << (double)dtFabric / 1000.0 << " ns  ("
                  << (dtPort / std::max(dtFabric, (simtime_picosec)1)) << "x faster)\n"
                  << "  Pipeline GPU→srcFab:  " << std::setprecision(1)
                  << (double)(2*dtPort) / 1000.0 << " ns\n"
                  << "  Pipeline GPU→srcEgQ:  "
                  << (double)(2*dtPort + dtFabric) / 1000.0 << " ns\n"
                  << "  Pipeline GPU→spine:   "
                  << (double)(4*dtPort + dtFabric) / 1000.0 << " ns\n"
                  << "  ECMP Latin Square: spine = (ps + pd) % "
                  << NUM_SPINES << "\n";

        // ---- 目标序列 & 源序列 ----
        std::cout << "  Building dst/src orders...\n";
        std::vector<std::vector<uint32_t>> dstOrders(NUM_GPU_NODES);
        for (uint32_t s = 0; s < NUM_GPU_NODES; s++)
            dstOrders[s] = buildRotatedDstOrder(s);
        std::vector<uint32_t> srcOrder = buildLeafInterleavedSrcOrder();

        // ---- 资源时间线 (5 类) ----
        std::vector<simtime_picosec> txFree(NUM_GPU_NODES, 0);       // NIC 带宽
        std::vector<simtime_picosec> fabLeafFree(NUM_LEAVES, 0);     // srcLeaf fabricQ
        std::vector<simtime_picosec> fabSpineFree(NUM_SPINES, 0);    // spine fabricQ

        // NEW: srcLeaf egressQ per uplink port (= per spine)
        std::vector<std::vector<simtime_picosec>> leafEgFree(
            NUM_LEAVES, std::vector<simtime_picosec>(NUM_SPINES, 0));

        // ---- 输出 ----
        std::map<uint32_t, std::vector<ScheduleEntry>> schedule;
        for (uint32_t g = 0; g < NUM_GPU_NODES; g++)
            schedule[g].reserve(TOTAL_FRAGMENTS * (NUM_GPU_NODES - 1));

        simtime_picosec maxFinish = 0;
        uint64_t totalEntries = (uint64_t)TOTAL_FRAGMENTS * NUM_GPU_NODES * (NUM_GPU_NODES - 1);
        uint64_t processed = 0;
        uint64_t reportInterval = std::max(totalEntries / 20, (uint64_t)1);

        std::cout << "  Scheduling " << totalEntries << " entries...\n";

        uint32_t numDsts = NUM_GPU_NODES - 1;

        for (uint32_t frag = 0; frag < TOTAL_FRAGMENTS; frag++) {
            for (uint32_t di = 0; di < numDsts; di++) {
                for (uint32_t si = 0; si < NUM_GPU_NODES; si++) {
                    uint32_t src = srcOrder[si];
                    uint32_t dst = dstOrders[src][di];
                    uint32_t srcLeaf = gpuToLeaf(src);
                    uint32_t dstLeaf = gpuToLeaf(dst);

                    int64_t c_tx   = (int64_t)txFree[src];
                    int64_t c_leaf = (int64_t)fabLeafFree[srcLeaf] - 2 * (int64_t)dtPort;

                    simtime_picosec sendTime;

                    if (srcLeaf == dstLeaf) {
                        // 同 leaf: txQ + srcLeaf fabricQ (egressQ 不经 uplink)
                        int64_t best = std::max({c_tx, c_leaf, (int64_t)0});
                        sendTime = (simtime_picosec)best;

                        txFree[src]           = sendTime + dtPort;
                        fabLeafFree[srcLeaf]  = sendTime + 2 * dtPort + dtFabric;

                        simtime_picosec finish = sendTime + 4 * dtPort + dtFabric;
                        if (finish > maxFinish) maxFinish = finish;

                    } else {
                        // 跨 leaf: 额外约束 spine + srcLeaf egressQ
                        uint32_t spine = spineSelectECMP(src, dst);

                        int64_t c_spine = (int64_t)fabSpineFree[spine]
                                        - 4 * (int64_t)dtPort - (int64_t)dtFabric;

                        // NEW: srcLeaf egressQ constraint
                        // 包到达 egressQ 的时间 = T + 2·dtPort + dtFabric
                        // egressQ 必须空闲: T + 2·dtPort + dtFabric ≥ leafEgFree
                        int64_t c_eg = (int64_t)leafEgFree[srcLeaf][spine]
                                     - 2 * (int64_t)dtPort - (int64_t)dtFabric;

                        int64_t best = std::max({c_tx, c_leaf, c_spine, c_eg, (int64_t)0});
                        sendTime = (simtime_picosec)best;

                        txFree[src]           = sendTime + dtPort;
                        fabLeafFree[srcLeaf]  = sendTime + 2 * dtPort + dtFabric;
                        fabSpineFree[spine]   = sendTime + 4 * dtPort + 2 * dtFabric;

                        // NEW: egressQ free after draining this packet
                        // egressQ starts at T + 2·dtPort + dtFabric, takes dtPort
                        leafEgFree[srcLeaf][spine] = sendTime + 3 * dtPort + dtFabric;

                        simtime_picosec finish = sendTime + 8 * dtPort + 3 * dtFabric;
                        if (finish > maxFinish) maxFinish = finish;
                    }

                    schedule[src].push_back({sendTime, (uint16_t)dst, (uint16_t)frag});

                    if (++processed % reportInterval == 0) {
                        std::cout << "  Schedule: "
                                  << (100 * processed / totalEntries) << "%\r"
                                  << std::flush;
                    }
                }
            }
        }
        std::cout << "  Schedule: 100%                    \n";

        // ---- 排序 ----
        for (auto& [gpu, entries] : schedule)
            std::sort(entries.begin(), entries.end(),
                      [](const ScheduleEntry& a, const ScheduleEntry& b) {
                          return a.sendTime < b.sendTime;
                      });

        if (outDuration) *outDuration = maxFinish;

        // ---- 利用率分析 ----
        simtime_picosec txTheoretical =
            (simtime_picosec)(NUM_GPU_NODES - 1) * TOTAL_FRAGMENTS * dtPort;

        uint64_t fabLeafPkts = (uint64_t)GPUS_PER_LEAF * (GPUS_PER_LEAF - 1) * TOTAL_FRAGMENTS
                             + 2 * (uint64_t)GPUS_PER_LEAF * (NUM_GPU_NODES - GPUS_PER_LEAF) * TOTAL_FRAGMENTS;
        simtime_picosec fabLeafTheoretical = (simtime_picosec)fabLeafPkts * dtFabric;

        uint64_t totalCrossLeafPkts = (uint64_t)NUM_GPU_NODES * (NUM_GPU_NODES - GPUS_PER_LEAF) * TOTAL_FRAGMENTS;
        simtime_picosec fabSpineTheoretical = (simtime_picosec)(totalCrossLeafPkts / NUM_SPINES) * dtFabric;

        // leafEgQ theoretical: per (leaf, spine), packets = GPUS_PER_LEAF * (NUM_LEAVES-1) * TOTAL_FRAGMENTS
        // each takes dtPort → leafEgQ_theoretical = that * dtPort
        uint64_t leafEgPktsPerPair = (uint64_t)GPUS_PER_LEAF * (NUM_LEAVES - 1) * TOTAL_FRAGMENTS;
        simtime_picosec leafEgTheoretical = (simtime_picosec)leafEgPktsPerPair * dtPort;

        simtime_picosec bound = std::max({txTheoretical, fabLeafTheoretical,
                                          fabSpineTheoretical, leafEgTheoretical});

        // 实际瓶颈
        simtime_picosec maxTx = 0, maxLeaf = 0, maxSpine = 0, maxEg = 0;
        for (uint32_t i = 0; i < NUM_GPU_NODES; i++)
            if (txFree[i] > maxTx) maxTx = txFree[i];
        for (uint32_t i = 0; i < NUM_LEAVES; i++) {
            if (fabLeafFree[i] > maxLeaf) maxLeaf = fabLeafFree[i];
            for (uint32_t s = 0; s < NUM_SPINES; s++)
                if (leafEgFree[i][s] > maxEg) maxEg = leafEgFree[i][s];
        }
        for (uint32_t i = 0; i < NUM_SPINES; i++)
            if (fabSpineFree[i] > maxSpine) maxSpine = fabSpineFree[i];

        simtime_picosec maxRes = std::max({maxTx, maxLeaf, maxSpine, maxEg});
        std::string bottleneck;
        if (maxRes == maxTx)         bottleneck = "txQ (NIC bandwidth)";
        else if (maxRes == maxLeaf)  bottleneck = "fabricQ(Leaf)";
        else if (maxRes == maxSpine) bottleneck = "fabricQ(Spine)";
        else                          bottleneck = "egressQ(Leaf→Spine)";

        std::cout << "\n  === Utilization Analysis (v13) ===\n"
                  << "  txQ theoretical:     " << std::fixed << std::setprecision(3)
                  << (double)txTheoretical / 1e9 << " ms\n"
                  << "  fabQ(Leaf) theory:   " << (double)fabLeafTheoretical / 1e9 << " ms\n"
                  << "  fabQ(Spine) theory:  " << (double)fabSpineTheoretical / 1e9 << " ms\n"
                  << "  egQ(Leaf) theory:    " << (double)leafEgTheoretical / 1e9 << " ms (per leaf-spine pair)\n"
                  << "  Lower bound:         " << (double)bound / 1e9 << " ms\n"
                  << "  Actual (max res):    " << (double)maxRes / 1e9 << " ms\n"
                  << "  Actual (last pkt):   " << (double)maxFinish / 1e9 << " ms\n"
                  << "  Utilization:         " << std::setprecision(1)
                  << ((double)bound / maxRes * 100.0) << "%\n"
                  << "  Bottleneck:          " << bottleneck << "\n"
                  << "  Max txQ:             " << (double)maxTx / 1e9 << " ms\n"
                  << "  Max fabQ(Leaf):      " << (double)maxLeaf / 1e9 << " ms\n"
                  << "  Max fabQ(Spine):     " << (double)maxSpine / 1e9 << " ms\n"
                  << "  Max egQ(Leaf):       " << (double)maxEg / 1e9 << " ms\n";

        return schedule;
    }

    static void printScheduleStats(
        const std::map<uint32_t, std::vector<ScheduleEntry>>& schedule,
        simtime_picosec duration)
    {
        std::cout << "=== Sync Schedule Stats ===\n"
                  << "  Phase duration: " << std::fixed << std::setprecision(3)
                  << (double)duration / 1e9 << " ms\n"
                  << "  Total GPUs: " << schedule.size() << "\n";
        int count = 0;
        for (auto& [gpu, entries] : schedule) {
            if (count < 4 || gpu == NUM_GPU_NODES - 1) {
                std::cout << "  GPU" << gpu << ": " << entries.size() << " sends, "
                          << "first=" << std::setprecision(3) << (double)entries.front().sendTime/1e9 << "ms "
                          << "last=" << std::setprecision(3) << (double)entries.back().sendTime/1e9 << "ms\n";
            } else if (count == 4) {
                std::cout << "  ...\n";
            }
            count++;
        }
        std::cout << "\n";
    }
};
