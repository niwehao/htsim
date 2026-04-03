/*
 * sync_scheduler.h  —  N-GPU Dragonfly 同步预调度 (ECMP + srcSw-aware)
 *
 * 路由顺序: ingressQ → BufferGate → fabricQ → egressQ → BufferRelease
 * buffer 占用持续到 egressQ drain 完才释放.
 *
 * 资源追踪 (只跟踪 srcSw, 避免级联延迟):
 *   - txFree[gpu]:                     NIC 带宽
 *   - fabSwFree[srcSw]:               srcSw fabricQ
 *   - egressFree[srcSw][outPort]:     srcSw egressQ (buffer 在此释放)
 *
 * 不追踪 transit/dst switch — fabricQ 16x 快于端口速率不会成为瓶颈,
 * transit egressQ 的短暂排队由 BufferGate 阈值容忍.
 * 追踪多跳会产生 cross-switch cascade 导致 60x 效率损失.
 *
 * Pipeline (srcSw hop 0):
 *   T → txQ(dtPort) → ingressQ(dtPort)
 *     → BufferGate(0) → fabricQ(dtFabric) → egressQ(dtPort)
 *     → BufferRelease(0) → next hop
 *
 * 约束 (对 srcSw):
 *   T >= txFree[src]
 *   T >= fabSwFree[srcSw]        - 2*dtPort
 *   T >= egressFree[srcSw][out]  - 2*dtPort - dtFabric
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
    //  buildInterleavedDstOrder — Dragonfly 拓扑交织
    // ================================================================
    static std::vector<uint32_t> buildInterleavedDstOrder(uint32_t srcGpu) {
        uint32_t srcSw    = gpuToSwitch(srcGpu);
        uint32_t srcGroup = switchToGroup(srcSw);

        std::vector<uint32_t> order;
        order.reserve(NUM_GPU_NODES - 1);

        std::vector<uint32_t> remoteGroups;
        for (uint32_t g = 0; g < NUM_GROUPS; g++)
            if (g != srcGroup) remoteGroups.push_back(g);

        std::vector<uint32_t> localDsts;
        for (uint32_t sw = srcGroup * SWITCHES_PER_GROUP;
             sw < (srcGroup + 1) * SWITCHES_PER_GROUP; sw++) {
            if (sw == srcSw) continue;
            for (uint32_t p = 0; p < GPUS_PER_SWITCH; p++)
                localDsts.push_back(sw * GPUS_PER_SWITCH + p);
        }

        std::vector<uint32_t> sameSw;
        for (uint32_t p = 0; p < GPUS_PER_SWITCH; p++) {
            uint32_t g = srcSw * GPUS_PER_SWITCH + p;
            if (g != srcGpu) sameSw.push_back(g);
        }

        uint32_t numRemoteGroups = (uint32_t)remoteGroups.size();
        uint32_t localIdx = 0, sameIdx = 0, round = 0;

        while (order.size() < NUM_GPU_NODES - 1) {
            for (uint32_t gi = 0; gi < numRemoteGroups && order.size() < NUM_GPU_NODES - 1; gi++) {
                uint32_t group = remoteGroups[gi];
                uint32_t gpuOffset = (round * numRemoteGroups + gi) % GPUS_PER_GROUP;
                order.push_back(group * GPUS_PER_GROUP + gpuOffset);
            }
            if (localIdx < localDsts.size() && order.size() < NUM_GPU_NODES - 1)
                order.push_back(localDsts[localIdx++]);
            if (sameIdx < sameSw.size() && order.size() < NUM_GPU_NODES - 1)
                order.push_back(sameSw[sameIdx++]);
            round++;
        }
        order.resize(NUM_GPU_NODES - 1);
        return order;
    }

    static std::vector<uint32_t> buildRotatedDstOrder(uint32_t srcGpu) {
        auto order = buildInterleavedDstOrder(srcGpu);
        uint32_t n = (uint32_t)order.size();
        if (n == 0) return order;
        uint32_t offset = srcGpu % n;
        std::rotate(order.begin(), order.begin() + offset, order.end());
        return order;
    }

    // ================================================================
    //  buildGroupInterleavedSrcOrder — 组间交织
    // ================================================================
    static std::vector<uint32_t> buildGroupInterleavedSrcOrder() {
        std::vector<uint32_t> order;
        order.reserve(NUM_GPU_NODES);
        for (uint32_t p = 0; p < GPUS_PER_SWITCH; p++)
            for (uint32_t j = 0; j < SWITCHES_PER_GROUP; j++)
                for (uint32_t g = 0; g < NUM_GROUPS; g++)
                    order.push_back((g * SWITCHES_PER_GROUP + j) * GPUS_PER_SWITCH + p);
        return order;
    }

    // ================================================================
    //  Schedule Cache
    // ================================================================
    static constexpr uint32_t CACHE_MAGIC   = 0x44465343; // "DFSC"
    static constexpr uint32_t CACHE_VERSION = 12; // v12: srcSw-only (no cascade)

    static std::string cacheFileName(int pktSize) {
        return "schedule_G" + std::to_string(NUM_GPU_NODES)
             + "_SW" + std::to_string(NUM_SWITCHES)
             + "_GR" + std::to_string(NUM_GROUPS)
             + "_a" + std::to_string(SWITCHES_PER_GROUP)
             + "_p" + std::to_string(GPUS_PER_SWITCH)
             + "_h" + std::to_string(GLOBAL_PORTS_PER_SW)
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
        w32(NUM_GPU_NODES); w32(NUM_SWITCHES); w32(NUM_GROUPS);
        w32(SWITCHES_PER_GROUP); w32(GPUS_PER_SWITCH); w32(GLOBAL_PORTS_PER_SW);
        w32(TOTAL_FRAGMENTS); w32((uint32_t)pktSize); w32(PORTS_PER_SWITCH);
        w32(0); // padding
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

        if (r32() != CACHE_MAGIC)        { std::cout << "  Cache: bad magic\n"; return false; }
        if (r32() != CACHE_VERSION)      { std::cout << "  Cache: version mismatch\n"; return false; }
        if (r32() != NUM_GPU_NODES)      { std::cout << "  Cache: GPU count mismatch\n"; return false; }
        if (r32() != NUM_SWITCHES)       { std::cout << "  Cache: switch count mismatch\n"; return false; }
        if (r32() != NUM_GROUPS)         { std::cout << "  Cache: group count mismatch\n"; return false; }
        if (r32() != SWITCHES_PER_GROUP) { std::cout << "  Cache: a mismatch\n"; return false; }
        if (r32() != GPUS_PER_SWITCH)    { std::cout << "  Cache: p mismatch\n"; return false; }
        if (r32() != GLOBAL_PORTS_PER_SW){ std::cout << "  Cache: h mismatch\n"; return false; }
        if (r32() != TOTAL_FRAGMENTS)    { std::cout << "  Cache: frag count mismatch\n"; return false; }
        if (r32() != (uint32_t)pktSize)  { std::cout << "  Cache: pktSize mismatch\n"; return false; }
        if (r32() != PORTS_PER_SWITCH)   { std::cout << "  Cache: ports mismatch\n"; return false; }
        r32(); // padding
        if (r64() != PROT_RATE_Gbps)     { std::cout << "  Cache: rate mismatch\n"; return false; }

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
    //  computePhaseSchedule — Dragonfly ECMP + egress-aware
    //
    //  Pipeline (per switch hop):
    //    T → txQ(dtPort) → ingressQ(dtPort)
    //      → fabricQ(dtFabric) → egressQ(dtPort)
    //      → next hop ingressQ(dtPort) → ...
    //
    //  约束 (只跟踪 srcSw):
    //    T >= txFree[src]
    //    T >= fabSwFree[srcSw]        - 2*dtPort
    //    T >= egressFree[srcSw][out]  - 2*dtPort - dtFabric
    //
    //  更新:
    //    txFree[src]              = T + dtPort
    //    fabSwFree[srcSw]         = T + 2*dtPort + dtFabric
    //    egressFree[srcSw][out]   = T + 3*dtPort + dtFabric
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

        std::cout << "  Building dst/src orders for Dragonfly (ECMP)...\n";
        std::vector<std::vector<uint32_t>> dstOrders(NUM_GPU_NODES);
        for (uint32_t s = 0; s < NUM_GPU_NODES; s++)
            dstOrders[s] = buildRotatedDstOrder(s);
        std::vector<uint32_t> srcOrder = buildGroupInterleavedSrcOrder();

        // ---- 资源时间线 (只跟踪 srcSw) ----
        std::vector<simtime_picosec> txFree(NUM_GPU_NODES, 0);
        std::vector<simtime_picosec> fabSwFree(NUM_SWITCHES, 0);
        // srcSw egressQ per port: buffer 在此 drain 完才释放
        std::vector<std::vector<simtime_picosec>> egressFree(
            NUM_SWITCHES, std::vector<simtime_picosec>(PORTS_PER_SWITCH, 0));

        // ---- 输出 ----
        std::map<uint32_t, std::vector<ScheduleEntry>> schedule;
        for (uint32_t g = 0; g < NUM_GPU_NODES; g++)
            schedule[g].reserve(TOTAL_FRAGMENTS * (NUM_GPU_NODES - 1));

        simtime_picosec maxFinish = 0;
        uint64_t totalEntries = (uint64_t)TOTAL_FRAGMENTS * NUM_GPU_NODES * (NUM_GPU_NODES - 1);
        uint64_t processed = 0;
        uint64_t reportInterval = std::max(totalEntries / 20, (uint64_t)1);

        std::cout << "  Scheduling " << totalEntries
                  << " entries (Dragonfly ECMP, srcSw-aware)...\n";

        uint32_t numDsts = NUM_GPU_NODES - 1;

        for (uint32_t frag = 0; frag < TOTAL_FRAGMENTS; frag++) {
            for (uint32_t di = 0; di < numDsts; di++) {
                for (uint32_t si = 0; si < NUM_GPU_NODES; si++) {
                    uint32_t src = srcOrder[si];
                    uint32_t dst = dstOrders[src][di];

                    SwitchPath path = computeSwitchPath(src, dst);
                    uint32_t srcSw   = path.sw[0];
                    uint32_t outPort = path.outPort[0];

                    // ---- 约束计算 (srcSw only) ----
                    int64_t cTx      = (int64_t)txFree[src];
                    int64_t cFab     = (int64_t)fabSwFree[srcSw] - 2 * (int64_t)dtPort;
                    int64_t cEgress  = (int64_t)egressFree[srcSw][outPort]
                                     - 2 * (int64_t)dtPort - (int64_t)dtFabric;

                    int64_t sendTime64 = std::max({cTx, cFab, cEgress, (int64_t)0});
                    simtime_picosec sendTime = (simtime_picosec)sendTime64;

                    // ---- 更新资源 (srcSw only) ----
                    txFree[src] = sendTime + dtPort;
                    fabSwFree[srcSw] = sendTime + 2 * dtPort + dtFabric;
                    egressFree[srcSw][outPort] = sendTime + 3 * dtPort + dtFabric;

                    // ---- 完成时间 (到 dstGpu rxQueue) ----
                    uint32_t nHops = path.numHops;
                    simtime_picosec finish = sendTime
                        + (2 * nHops + 1) * dtPort + nHops * dtFabric;
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

        simtime_picosec maxTx = 0, maxFab = 0, maxEgress = 0;
        for (uint32_t i = 0; i < NUM_GPU_NODES; i++)
            if (txFree[i] > maxTx) maxTx = txFree[i];
        for (uint32_t i = 0; i < NUM_SWITCHES; i++)
            if (fabSwFree[i] > maxFab) maxFab = fabSwFree[i];
        for (uint32_t sw = 0; sw < NUM_SWITCHES; sw++)
            for (uint32_t p = 0; p < PORTS_PER_SWITCH; p++)
                if (egressFree[sw][p] > maxEgress) maxEgress = egressFree[sw][p];

        // 每交换机理论 fabric 负载
        std::vector<uint64_t> swPktCount(NUM_SWITCHES, 0);
        for (uint32_t src = 0; src < NUM_GPU_NODES; src++) {
            for (uint32_t dst = 0; dst < NUM_GPU_NODES; dst++) {
                if (src == dst) continue;
                SwitchPath path = computeSwitchPath(src, dst);
                for (uint32_t h = 0; h < path.numHops; h++)
                    swPktCount[path.sw[h]] += TOTAL_FRAGMENTS;
            }
        }
        uint64_t maxSwPkts = *std::max_element(swPktCount.begin(), swPktCount.end());
        simtime_picosec fabTheoretical = (simtime_picosec)maxSwPkts * dtFabric;

        simtime_picosec bound = std::max(txTheoretical, fabTheoretical);
        simtime_picosec maxRes = std::max({maxTx, maxFab, maxEgress});

        std::string bottleneck;
        if (maxRes == maxTx)         bottleneck = "txQ (NIC bandwidth)";
        else if (maxRes == maxFab)   bottleneck = "fabricQ (switch)";
        else                          bottleneck = "egressQ (port)";

        // 分析 port 物理瓶颈
        std::vector<std::vector<uint64_t>> portPktCount(
            NUM_SWITCHES, std::vector<uint64_t>(PORTS_PER_SWITCH, 0));
        for (uint32_t src = 0; src < NUM_GPU_NODES; src++) {
            for (uint32_t dst = 0; dst < NUM_GPU_NODES; dst++) {
                if (src == dst) continue;
                SwitchPath path = computeSwitchPath(src, dst);
                for (uint32_t h = 0; h < path.numHops; h++)
                    portPktCount[path.sw[h]][path.outPort[h]] += TOTAL_FRAGMENTS;
            }
        }
        uint64_t maxPortPkts = 0;
        uint32_t maxPortSw = 0, maxPortId = 0;
        for (uint32_t sw = 0; sw < NUM_SWITCHES; sw++)
            for (uint32_t p = 0; p < PORTS_PER_SWITCH; p++)
                if (portPktCount[sw][p] > maxPortPkts) {
                    maxPortPkts = portPktCount[sw][p];
                    maxPortSw = sw; maxPortId = p;
                }
        simtime_picosec portBottleneck = (simtime_picosec)maxPortPkts * dtPort;

        std::cout << "\n  === Utilization Analysis (Dragonfly ECMP srcSw-aware) ===\n"
                  << "  txQ theoretical:     " << std::fixed << std::setprecision(3)
                  << (double)txTheoretical / 1e9 << " ms\n"
                  << "  fabQ max theory:     " << (double)fabTheoretical / 1e9 << " ms\n"
                  << "  Lower bound:         " << (double)bound / 1e9 << " ms\n"
                  << "  Actual (max res):    " << (double)maxRes / 1e9 << " ms\n"
                  << "  Actual (last pkt):   " << (double)maxFinish / 1e9 << " ms\n"
                  << "  Utilization:         " << std::setprecision(1)
                  << ((double)bound / maxRes * 100.0) << "%\n"
                  << "  Bottleneck:          " << bottleneck << "\n"
                  << "  Max txQ:             " << (double)maxTx / 1e9 << " ms\n"
                  << "  Max fabQ(switch):    " << (double)maxFab / 1e9 << " ms\n"
                  << "  Max egressQ(port):   " << (double)maxEgress / 1e9 << " ms\n"
                  << "\n  Port bottleneck: Sw" << maxPortSw << " port " << maxPortId
                  << " = " << maxPortPkts << " pkts × " << std::setprecision(0)
                  << (double)dtPort << " ps = "
                  << std::setprecision(3) << (double)portBottleneck / 1e9 << " ms\n";

        std::cout << "  Switch fabric load (packets): ";
        for (uint32_t i = 0; i < NUM_SWITCHES; i++) {
            if (i > 0 && i % SWITCHES_PER_GROUP == 0) std::cout << "| ";
            std::cout << "S" << i << "=" << swPktCount[i] << " ";
        }
        std::cout << "\n";

        // ---- 保存缓存 ----
        saveCache(schedule, maxFinish, pktSize);

        return schedule;
    }

    static void printScheduleStats(
        const std::map<uint32_t, std::vector<ScheduleEntry>>& schedule,
        simtime_picosec duration)
    {
        std::cout << "=== Sync Schedule Stats (Dragonfly ECMP) ===\n"
                  << "  Phase duration: " << std::fixed << std::setprecision(3)
                  << (double)duration / 1e9 << " ms\n"
                  << "  Total GPUs: " << (uint32_t)schedule.size() << "\n";
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
