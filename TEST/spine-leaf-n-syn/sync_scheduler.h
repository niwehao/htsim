/*
 * sync_scheduler.h  —  512-GPU Leaf-Spine 同步预调度 (v12)
 *
 * v12 关键改进 (相对 v11):
 *
 * 1. 新增 egress queue 资源追踪:
 *    - leafUpEgFree[leaf][spine]:     leaf 上行 egressQ → spine
 *    - spineDownEgFree[spine][leaf]:  spine 下行 egressQ → dstLeaf
 *
 *    v11 问题: leaf fabricQ 以 dtFabric (8.25ns) 间隔输出包, 但 egressQ
 *    以 dtPort (165ns) 逐包排空. 同 leaf 多包经同 spine 时在 egressQ 排队,
 *    导致到达 spine 时间偏移 → spine BufferGate 多包堆叠 → lost packet.
 *
 * 2. 仅追踪源侧核心资源 (类比 Dragonfly-opt 只追踪 srcSw):
 *    不追踪 egressQ / spine egressQ / dstLeaf fabricQ, 避免 gap 问题.
 *    下游少量排队由 BufferGate 容忍 (fabric 20x faster, 阈值16包).
 *
 * 资源追踪 (v12b — 最小集):
 *    - txFree[gpu]:                   NIC 带宽
 *    - fabLeafFree[leaf]:             srcLeaf fabricQ
 *    - fabSpineFree[spine]:           spine fabricQ
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
#include <fstream>
#include <cstring>

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
    //  Schedule Cache — 二进制文件缓存
    //
    //  文件格式:
    //    [magic 4B "SLSC"] [version 4B]
    //    [NUM_GPU_NODES 4B] [NUM_LEAVES 4B] [NUM_SPINES 4B]
    //    [GPUS_PER_LEAF 4B] [TOTAL_FRAGMENTS 4B] [pktSize 4B]
    //    [PORTS_PER_SWITCH 4B] [PROT_RATE_Gbps 8B]
    //    [duration 8B]
    //    for each GPU (0..N-1):
    //      [numEntries 4B]
    //      for each entry: [sendTime 8B] [dstGpu 2B] [fragId 2B]
    // ================================================================
    static constexpr uint32_t CACHE_MAGIC   = 0x534C5343; // "SLSC"
    static constexpr uint32_t CACHE_VERSION = 2;

    static std::string cacheFileName(int pktSize) {
        // schedule_G128_L16_S8_P16_F512_R200_pkt4125.bin
        return "schedule_G" + std::to_string(NUM_GPU_NODES)
             + "_L" + std::to_string(NUM_LEAVES)
             + "_S" + std::to_string(NUM_SPINES)
             + "_P" + std::to_string(PORTS_PER_SWITCH)
             + "_F" + std::to_string(TOTAL_FRAGMENTS)
             + "_R" + std::to_string(PROT_RATE_Gbps)
             + "_pkt" + std::to_string(pktSize) + ".bin";
    }

    static bool saveCache(const std::map<uint32_t, std::vector<ScheduleEntry>>& schedule,
                          simtime_picosec duration, int pktSize) {
        std::string fname = cacheFileName(pktSize);
        std::ofstream f(fname, std::ios::binary);
        if (!f) return false;

        auto w32 = [&](uint32_t v) { f.write((const char*)&v, 4); };
        auto w64 = [&](uint64_t v) { f.write((const char*)&v, 8); };
        auto w16 = [&](uint16_t v) { f.write((const char*)&v, 2); };

        w32(CACHE_MAGIC); w32(CACHE_VERSION);
        w32(NUM_GPU_NODES); w32(NUM_LEAVES); w32(NUM_SPINES);
        w32(GPUS_PER_LEAF); w32(TOTAL_FRAGMENTS); w32((uint32_t)pktSize);
        w32(PORTS_PER_SWITCH); w32(0); // padding
        w64(PROT_RATE_Gbps);
        w64(duration);

        for (uint32_t g = 0; g < NUM_GPU_NODES; g++) {
            auto it = schedule.find(g);
            uint32_t n = (it != schedule.end()) ? (uint32_t)it->second.size() : 0;
            w32(n);
            if (it != schedule.end()) {
                for (auto& e : it->second) {
                    w64(e.sendTime); w16(e.dstGpu); w16(e.fragId);
                }
            }
        }
        std::cout << "  Cache saved: " << fname << "\n";
        return true;
    }

    static bool loadCache(std::map<uint32_t, std::vector<ScheduleEntry>>& schedule,
                          simtime_picosec& duration, int pktSize) {
        std::string fname = cacheFileName(pktSize);
        std::ifstream f(fname, std::ios::binary);
        if (!f) return false;

        auto r32 = [&]() -> uint32_t { uint32_t v; f.read((char*)&v, 4); return v; };
        auto r64 = [&]() -> uint64_t { uint64_t v; f.read((char*)&v, 8); return v; };
        auto r16 = [&]() -> uint16_t { uint16_t v; f.read((char*)&v, 2); return v; };

        if (r32() != CACHE_MAGIC)   { std::cout << "  Cache: bad magic\n"; return false; }
        if (r32() != CACHE_VERSION) { std::cout << "  Cache: version mismatch\n"; return false; }
        if (r32() != NUM_GPU_NODES) { std::cout << "  Cache: GPU count mismatch\n"; return false; }
        if (r32() != NUM_LEAVES)    { std::cout << "  Cache: leaf count mismatch\n"; return false; }
        if (r32() != NUM_SPINES)    { std::cout << "  Cache: spine count mismatch\n"; return false; }
        if (r32() != GPUS_PER_LEAF) { std::cout << "  Cache: GPUs/leaf mismatch\n"; return false; }
        if (r32() != TOTAL_FRAGMENTS) { std::cout << "  Cache: frag count mismatch\n"; return false; }
        if (r32() != (uint32_t)pktSize) { std::cout << "  Cache: pktSize mismatch\n"; return false; }
        if (r32() != PORTS_PER_SWITCH) { std::cout << "  Cache: ports mismatch\n"; return false; }
        r32(); // padding
        if (r64() != PROT_RATE_Gbps) { std::cout << "  Cache: rate mismatch\n"; return false; }

        duration = (simtime_picosec)r64();

        schedule.clear();
        for (uint32_t g = 0; g < NUM_GPU_NODES; g++) {
            uint32_t n = r32();
            auto& vec = schedule[g];
            vec.resize(n);
            for (uint32_t i = 0; i < n; i++) {
                vec[i].sendTime = (simtime_picosec)r64();
                vec[i].dstGpu   = r16();
                vec[i].fragId   = r16();
            }
        }

        if (!f.good()) { std::cout << "  Cache: read error\n"; schedule.clear(); return false; }
        std::cout << "  Cache loaded: " << fname
                  << " (duration=" << std::fixed << std::setprecision(3)
                  << (double)duration / 1e9 << " ms)\n";
        return true;
    }

    // ================================================================
    //  computePhaseSchedule — v12
    //
    //  Pipeline 模型 (跨 Leaf):
    //    sendTime → txQ(dtPort) → srcLeaf ingressQ(dtPort)
    //    → srcLeaf fabricQ(dtFabric) → srcLeaf egressQ[spine](dtPort)
    //    → spine ingressQ(dtPort) → spine fabricQ(dtFabric)
    //    → spine egressQ[dstLeaf](dtPort)
    //    → dstLeaf ingressQ(dtPort) → dstLeaf fabricQ(dtFabric) [不追踪]
    //    → dstLeaf egressQ(dtPort) → dst rxQ
    //
    //  设 T = sendTime:
    //    srcLeaf fab 开始:      T + 2*dtPort
    //    srcLeaf egress 开始:   T + 2*dtPort + dtFabric
    //    spine fab 开始:        T + 4*dtPort + dtFabric
    //    spine egress 开始:     T + 4*dtPort + 2*dtFabric
    //
    //  约束:
    //    T >= txFree[src]                                              (NIC)
    //    T >= fabLeafFree[srcLeaf]          - 2*dtPort                 (srcLeaf fabQ)
    //    T >= fabSpineFree[spine]           - 4*dtPort - dtFabric       (spine fabQ)
    // ================================================================
    static std::map<uint32_t, std::vector<ScheduleEntry>>
    computePhaseSchedule(int pktSize, simtime_picosec* outDuration = nullptr) {

        // ---- 尝试从缓存加载 ----
        {
            std::map<uint32_t, std::vector<ScheduleEntry>> cached;
            simtime_picosec cachedDuration = 0;
            if (loadCache(cached, cachedDuration, pktSize)) {
                if (outDuration) *outDuration = cachedDuration;
                return cached;
            }
            std::cout << "  No valid cache, computing schedule...\n";
        }

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
                  << " entries (v12: pipeline + egressQ tracking)...\n";

        uint32_t numDsts = NUM_GPU_NODES - 1;

        for (uint32_t frag = 0; frag < TOTAL_FRAGMENTS; frag++) {
            for (uint32_t di = 0; di < numDsts; di++) {
                for (uint32_t si = 0; si < NUM_GPU_NODES; si++) {
                    uint32_t src = srcOrder[si];
                    uint32_t dst = dstOrders[src][di];
                    uint32_t srcLeaf = gpuToLeaf(src);
                    uint32_t dstLeaf = gpuToLeaf(dst);

                    int64_t cTx       = (int64_t)txFree[src];
                    int64_t cSrcLeaf  = (int64_t)fabLeafFree[srcLeaf] - 2 * (int64_t)dtPort;
                    int64_t sendTime64;

                    if (srcLeaf == dstLeaf) {
                        sendTime64 = std::max({cTx, cSrcLeaf, (int64_t)0});
                    } else {
                        uint32_t spine = spineSelectECMP(src, dst);

                        int64_t cSpine   = (int64_t)fabSpineFree[spine]
                                         - 4 * (int64_t)dtPort - (int64_t)dtFabric;

                        sendTime64 = std::max({cTx, cSrcLeaf,
                                               cSpine, (int64_t)0});

                        fabSpineFree[spine] = (simtime_picosec)sendTime64
                                            + 4 * dtPort + 2 * dtFabric;
                    }

                    simtime_picosec sendTime = (simtime_picosec)sendTime64;

                    txFree[src] = sendTime + dtPort;
                    fabLeafFree[srcLeaf] = sendTime + 2 * dtPort + dtFabric;

                    simtime_picosec lastFabFinish;
                    if (srcLeaf == dstLeaf) {
                        lastFabFinish = sendTime + 2 * dtPort + dtFabric;
                    } else {
                        lastFabFinish = sendTime + 6 * dtPort + 3 * dtFabric;
                    }
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
        simtime_picosec txTheoretical =
            (simtime_picosec)(NUM_GPU_NODES - 1) * TOTAL_FRAGMENTS * dtPort;

        uint64_t fabLeafPkts = (uint64_t)GPUS_PER_LEAF * (GPUS_PER_LEAF - 1) * TOTAL_FRAGMENTS
                             + (uint64_t)GPUS_PER_LEAF * (NUM_GPU_NODES - GPUS_PER_LEAF) * TOTAL_FRAGMENTS;
        simtime_picosec fabLeafTheoretical = (simtime_picosec)fabLeafPkts * dtFabric;

        uint64_t totalCrossLeafPkts = (uint64_t)NUM_GPU_NODES * (NUM_GPU_NODES - GPUS_PER_LEAF) * TOTAL_FRAGMENTS;
        simtime_picosec fabSpineTheoretical = (simtime_picosec)(totalCrossLeafPkts / NUM_SPINES) * dtFabric;

        simtime_picosec bound = std::max({txTheoretical, fabLeafTheoretical,
                                           fabSpineTheoretical});

        simtime_picosec maxTx = 0, maxLeaf = 0, maxSpine = 0;
        for (uint32_t i = 0; i < NUM_GPU_NODES; i++)
            if (txFree[i] > maxTx) maxTx = txFree[i];
        for (uint32_t i = 0; i < NUM_LEAVES; i++)
            if (fabLeafFree[i] > maxLeaf) maxLeaf = fabLeafFree[i];
        for (uint32_t i = 0; i < NUM_SPINES; i++)
            if (fabSpineFree[i] > maxSpine) maxSpine = fabSpineFree[i];

        simtime_picosec maxRes = std::max({maxTx, maxLeaf, maxSpine});
        std::string bottleneck;
        if      (maxRes == maxTx)         bottleneck = "txQ (NIC bandwidth)";
        else if (maxRes == maxLeaf)       bottleneck = "fabricQ(Leaf)";
        else                               bottleneck = "fabricQ(Spine)";

        std::cout << "\n  === Utilization Analysis (v12) ===\n"
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

        // ---- 保存缓存 ----
        saveCache(schedule, maxFinish, pktSize);

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
