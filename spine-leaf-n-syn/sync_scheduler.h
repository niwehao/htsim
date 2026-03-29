/*
 * sync_scheduler.h  —  512-GPU Leaf-Spine 同步预调度 (v11)
 *
 * v11 关键改进:
 *
 * 1. 精确 pipeline 模型:
 *    仿真路径: sendTime → txQueue(dtPort) → ingressQ(dtPort) → fabricQ(dtFabric) → ...
 *    每个 fabricQ hop 之间: egressQ(dtPort) + ingressQ(dtPort) = 2*dtPort
 *    sendTime 由 srcLeaf fabricQ 时隙反推: sendTime = srcFabStart - 2*dtPort
 *
 * 2. 资源追踪:
 *    - txFree[gpu]: 每 GPU 的 NIC 带宽约束 (发送间隔 >= dtPort)
 *    - fabLeafFree[leaf]: srcLeaf fabricQ 无冲突
 *    - fabSpineFree[spine]: spine fabricQ 无冲突 (从 spine 反向传播到 srcLeaf)
 *    - 不追踪 dstLeaf fabricQ (避免 resFree 空洞问题, BufferGate 容忍少量排队)
 *
 * 3. Dst rotation + leaf-interleaved src order (继承 v10)
 *
 * 迭代: frag → dst_idx → src (交织 leaf)
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
    //  buildInterleavedDstOrder — 基础交织序列
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
    //  buildLeafInterleavedSrcOrder — 交织 leaf, 分散 fabricQ 负载
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
    //  computePhaseSchedule — v11
    //
    //  Pipeline 模型 (仿真实际路径):
    //    GPU send → txQueue(dtPort) → ingressQ(dtPort) → BufferGate → fabricQ(dtFabric)
    //            → BufferRelease → egressQ(dtPort) → [next switch ingressQ(dtPort)] → ...
    //
    //  设 T = sendTime:
    //    srcLeaf fab 开始: T + 2*dtPort
    //    spine fab 开始:   T + 4*dtPort + dtFabric
    //    dstLeaf fab 开始: T + 6*dtPort + 2*dtFabric  (不追踪)
    //
    //  约束:
    //    T >= txFree[gpu]                                      (NIC 带宽)
    //    T >= fabLeafFree[srcLeaf] - 2*dtPort                  (srcLeaf fabricQ)
    //    T >= fabSpineFree[spine]  - 4*dtPort - dtFabric       (spine fabricQ)
    // ================================================================
    static std::map<uint32_t, std::vector<ScheduleEntry>>
    computePhaseSchedule(int pktSize, simtime_picosec* outDuration = nullptr) {

        simtime_picosec dtPort   = (simtime_picosec)pktSize * psPerByteLink();
        simtime_picosec dtFabric = (simtime_picosec)pktSize * psPerByteFabric();

        std::cout << "  Port drain:    " << std::fixed << std::setprecision(1)
                  << (double)dtPort / 1000.0 << " ns\n"
                  << "  Fabric drain:  " << std::setprecision(2)
                  << (double)dtFabric / 1000.0 << " ns  ("
                  << (dtPort / std::max(dtFabric, (simtime_picosec)1)) << "x faster)\n"
                  << "  Hop pipeline:  " << std::setprecision(1)
                  << (double)(2 * dtPort) / 1000.0 << " ns  (egressQ + ingressQ)\n";

        // ---- 目标序列 (rotated) & 源序列 ----
        std::cout << "  Building dst/src orders...\n";
        std::vector<std::vector<uint32_t>> dstOrders(NUM_GPU_NODES);
        for (uint32_t s = 0; s < NUM_GPU_NODES; s++)
            dstOrders[s] = buildRotatedDstOrder(s);
        std::vector<uint32_t> srcOrder = buildLeafInterleavedSrcOrder();

        // ---- 资源时间线 ----
        std::vector<simtime_picosec> txFree(NUM_GPU_NODES, 0);
        std::vector<simtime_picosec> fabLeafFree(NUM_LEAVES, 0);
        std::vector<simtime_picosec> fabSpineFree(NUM_SPINES, 0);

        // ---- 输出 ----
        std::map<uint32_t, std::vector<ScheduleEntry>> schedule;
        for (uint32_t g = 0; g < NUM_GPU_NODES; g++)
            schedule[g].reserve(TOTAL_FRAGMENTS * (NUM_GPU_NODES - 1));

        simtime_picosec maxFinish = 0;
        uint64_t totalEntries = (uint64_t)TOTAL_FRAGMENTS * NUM_GPU_NODES * (NUM_GPU_NODES - 1);
        uint64_t processed = 0;
        uint64_t reportInterval = std::max(totalEntries / 20, (uint64_t)1);

        std::cout << "  Scheduling " << totalEntries
                  << " entries (v11: pipeline + txQ + srcLeaf + spine)...\n";

        uint32_t numDsts = NUM_GPU_NODES - 1;

        for (uint32_t frag = 0; frag < TOTAL_FRAGMENTS; frag++) {
            for (uint32_t di = 0; di < numDsts; di++) {
                for (uint32_t si = 0; si < NUM_GPU_NODES; si++) {
                    uint32_t src = srcOrder[si];
                    uint32_t dst = dstOrders[src][di];
                    uint32_t srcLeaf = gpuToLeaf(src);
                    uint32_t dstLeaf = gpuToLeaf(dst);

                    // 计算 sendTime = max(所有约束)
                    int64_t cTx       = (int64_t)txFree[src];
                    int64_t cSrcLeaf  = (int64_t)fabLeafFree[srcLeaf] - 2 * (int64_t)dtPort;
                    int64_t sendTime64;

                    if (srcLeaf == dstLeaf) {
                        // 同 Leaf: 只约束 txQ + srcLeaf fabricQ
                        sendTime64 = std::max({cTx, cSrcLeaf, (int64_t)0});
                    } else {
                        // 跨 Leaf: 额外约束 spine fabricQ (反向传播)
                        uint32_t spine = spineSelectECMP(src, dst);
                        int64_t cSpine = (int64_t)fabSpineFree[spine]
                                       - 4 * (int64_t)dtPort - (int64_t)dtFabric;
                        sendTime64 = std::max({cTx, cSrcLeaf, cSpine, (int64_t)0});

                        // 更新 spine fabricQ
                        simtime_picosec spineFabStart =
                            (simtime_picosec)sendTime64 + 4 * dtPort + dtFabric;
                        fabSpineFree[spine] = spineFabStart + dtFabric;
                    }

                    simtime_picosec sendTime = (simtime_picosec)sendTime64;

                    // 更新 txQ 和 srcLeaf fabricQ
                    txFree[src] = sendTime + dtPort;
                    simtime_picosec srcFabStart = sendTime + 2 * dtPort;
                    fabLeafFree[srcLeaf] = srcFabStart + dtFabric;

                    // 最后一个 fabricQ 完成时间
                    simtime_picosec lastFabFinish;
                    if (srcLeaf == dstLeaf) {
                        lastFabFinish = srcFabStart + dtFabric;
                    } else {
                        // dstLeaf fab finish = sendTime + 6*dtPort + 3*dtFabric
                        lastFabFinish = sendTime + 6 * dtPort + 3 * dtFabric;
                    }
                    // 加上最后的 egressQ + rxQ 管道延迟
                    simtime_picosec finish = lastFabFinish + 2 * dtPort;
                    if (finish > maxFinish) maxFinish = finish;

                    schedule[src].push_back({sendTime, (uint16_t)dst, (uint16_t)frag});

                    processed++;
                    if (processed % reportInterval == 0) {
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
        // 每 GPU txQ 理论工作量
        simtime_picosec txTheoretical =
            (simtime_picosec)(NUM_GPU_NODES - 1) * TOTAL_FRAGMENTS * dtPort;

        // 每 Leaf fabricQ 理论工作量 (local + outgoing + incoming)
        uint64_t fabLeafPkts = (uint64_t)GPUS_PER_LEAF * (GPUS_PER_LEAF - 1) * TOTAL_FRAGMENTS
                             + 2 * (uint64_t)GPUS_PER_LEAF * (NUM_GPU_NODES - GPUS_PER_LEAF) * TOTAL_FRAGMENTS;
        simtime_picosec fabLeafTheoretical = (simtime_picosec)fabLeafPkts * dtFabric;

        // 每 Spine fabricQ 理论工作量
        uint64_t totalCrossLeafPkts = (uint64_t)NUM_GPU_NODES * (NUM_GPU_NODES - GPUS_PER_LEAF) * TOTAL_FRAGMENTS;
        simtime_picosec fabSpineTheoretical = (simtime_picosec)(totalCrossLeafPkts / NUM_SPINES) * dtFabric;

        simtime_picosec bound = std::max({txTheoretical, fabLeafTheoretical, fabSpineTheoretical});

        // 找瓶颈
        simtime_picosec maxTx = 0, maxLeaf = 0, maxSpine = 0;
        for (uint32_t i = 0; i < NUM_GPU_NODES; i++)
            if (txFree[i] > maxTx) maxTx = txFree[i];
        for (uint32_t i = 0; i < NUM_LEAVES; i++)
            if (fabLeafFree[i] > maxLeaf) maxLeaf = fabLeafFree[i];
        for (uint32_t i = 0; i < NUM_SPINES; i++)
            if (fabSpineFree[i] > maxSpine) maxSpine = fabSpineFree[i];

        std::string bottleneck;
        simtime_picosec maxRes = std::max({maxTx, maxLeaf, maxSpine});
        if (maxRes == maxTx)         bottleneck = "txQ (NIC bandwidth)";
        else if (maxRes == maxLeaf)  bottleneck = "fabricQ(Leaf)";
        else                          bottleneck = "fabricQ(Spine)";

        std::cout << "\n  === Utilization Analysis (v11) ===\n"
                  << "  txQ theoretical:     " << std::fixed << std::setprecision(3)
                  << (double)txTheoretical / 1e9 << " ms\n"
                  << "  fabQ(Leaf) theory:   " << (double)fabLeafTheoretical / 1e9 << " ms\n"
                  << "  fabQ(Spine) theory:  " << (double)fabSpineTheoretical / 1e9 << " ms\n"
                  << "  Lower bound:         " << (double)bound / 1e9 << " ms\n"
                  << "  Actual (max res):    " << (double)maxRes / 1e9 << " ms\n"
                  << "  Actual (last pkt):   " << (double)maxFinish / 1e9 << " ms\n"
                  << "  Utilization:         " << std::setprecision(1)
                  << ((double)bound / maxRes * 100.0) << "%\n"
                  << "  Bottleneck:          " << bottleneck << "\n"
                  << "  Max txQ:             " << (double)maxTx / 1e9 << " ms\n"
                  << "  Max fabQ(Leaf):      " << (double)maxLeaf / 1e9 << " ms\n"
                  << "  Max fabQ(Spine):     " << (double)maxSpine / 1e9 << " ms\n";

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
