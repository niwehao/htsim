/*
 * main.cpp  —  Dragonfly 同步调度版 MOE htsim 仿真
 *
 * 流程:
 *   1. 离线计算最优调度表 (三遍扫描 SyncScheduler)
 *   2. 验证调度: 确保所有资源无时间冲突
 *   3. 创建 Dragonfly 拓扑 (简化 Tofino: 无 BufferGate)
 *   4. 各 GPU 按调度表定时发送 -> 零拥塞、零重传、无 BufferGate 惩罚
 *   5. 运行仿真, 验证结果
 *
 * Dragonfly 拓扑 (4 个交换机, 8 个 GPU):
 *
 *  Group 0:
 *    Sw0: ports {1,2,3,4}   GPU1@p1, GPU2@p2, local@p3, global@p4
 *    Sw1: ports {5,6,7,8}   GPU3@p5, GPU4@p6, local@p7, global@p8
 *    Sw0 <-local-> Sw1 (port 3 <-> port 7)
 *
 *  Group 1:
 *    Sw2: ports {9,10,11,12}  GPU5@p9, GPU6@p10, local@p11, global@p12
 *    Sw3: ports {13,14,15,16} GPU7@p13, GPU8@p14, local@p15, global@p16
 *    Sw2 <-local-> Sw3 (port 11 <-> port 15)
 *
 *  Global links:
 *    Sw0 <-> Sw2 (port 4 <-> port 12)
 *    Sw1 <-> Sw3 (port 8 <-> port 16)
 */

#include "gpu_node_sync.h"
#include <iostream>
#include <array>

// ================================================================
//  buildRoute: 同步版路由构建 (3 hops per Tofino, not 5)
// ================================================================
Route* buildRoute(int srcGpu, int dstGpu,
                  std::array<TofinoSwitchSync*, 4>& tofs,
                  std::array<GpuNodeSync*, 8>& gpus)
{
    Route* route = new Route();

    int curTofinoIdx = gpuToTofinoIdx(srcGpu);
    int inPort       = gpuToPort(srcGpu);

    for (int hop = 0; hop < 10; hop++) {
        auto rtable = resolvedRoutingTable(curTofinoIdx);
        int outPort = rtable.at(dstGpu);

        // 3 个节点: ingressQ -> fabricQ -> egressQ (无 BufferGate/BufferRelease)
        tofs[curTofinoIdx]->appendToRoute(*route, inPort, outPort);

        if (isGpuPort(outPort) && portToGpu(outPort) == dstGpu) {
            route->push_back(gpus[dstGpu - 1]->rxQueue());
            route->push_back(gpus[dstGpu - 1]->appSink());
            return route;
        }

        int nextInPort    = portMapping(outPort);
        int nextTofinoIdx = portToTofinoIdx(nextInPort);
        curTofinoIdx = nextTofinoIdx;
        inPort       = nextInPort;
    }

    assert(false && "Route tracing exceeded max hops");
    return route;
}

// ================================================================
//  main
// ================================================================
int main() {
    EventList::setEndtime(timeFromSec(2000));

    std::cout << "=== MOE Sync-Scheduled Simulation (Dragonfly Topology) ===\n"
              << "  GPUs=" << NUM_GPU_NODES
              << " Switches=" << NUM_SWITCHES << " (Dragonfly)"
              << " Groups=" << NUM_GROUPS
              << " Layers=" << TOTAL_LAYERS
              << " Frags=" << TOTAL_FRAGMENTS << "x" << (FRAGMENT_PAYLOAD_SIZE/1024) << "KB"
              << " Link=" << PROT_RATE_Gbps << "Gbps\n\n";

    // ---- Step 1: 计算调度表 (三遍扫描) ----
    std::cout << "Computing optimal schedule (3-pass algorithm)...\n";
    int pktSize = 14 + 15 + (int)FRAGMENT_PAYLOAD_SIZE;
    simtime_picosec phaseDuration = 0;
    auto scheduleMap = SyncScheduler::computePhaseSchedule(pktSize, &phaseDuration);
    SyncScheduler::printScheduleStats(scheduleMap, phaseDuration);
    SyncScheduler::printFullSchedule(scheduleMap, 50);

    simtime_picosec totalEstimate = (simtime_picosec)TOTAL_LAYERS * 2 *
        (phaseDuration + INTERPHASE_GAP_PS);
    std::cout << "Estimated total time: " << std::fixed << std::setprecision(3)
              << (double)totalEstimate / 1e9 << " ms ("
              << TOTAL_LAYERS << " layers x 2 phases)\n\n";

    // ---- Step 2: 验证调度 ----
    std::cout << "Validating schedule...\n";
    SyncScheduler::validateSchedule(pktSize);

    // ---- Step 3: 创建简化 Tofino (无 BufferGate) ----
    std::array<TofinoSwitchSync*, 4> tofs;
    for (int i = 0; i < 4; i++)
        tofs[i] = new TofinoSwitchSync(i);

    // ---- Step 4: 创建 GPU 节点 (传入调度表) ----
    std::array<GpuNodeSync*, 8> gpus;
    auto makePeers = [](uint8_t self) {
        std::vector<uint8_t> v;
        for (uint8_t i = 1; i <= 8; i++) if (i != self) v.push_back(i);
        return v;
    };
    for (int i = 0; i < 8; i++) {
        int gpuId = i + 1;
        gpus[i] = new GpuNodeSync(
            (uint8_t)gpuId,
            makePeers(gpuId),
            scheduleMap[gpuId]);   // 该 GPU 的发送调度
    }

    // ---- Step 5: 创建 PacketFlow ----
    std::array<PacketFlow*, 8> flows;
    for (int i = 0; i < 8; i++) {
        flows[i] = new PacketFlow(nullptr);
        flows[i]->set_flowid((flowid_t)(i + 1));
        gpus[i]->setFlow(flows[i]);
    }

    // ---- Step 6: 预计算路由 ----
    std::array<std::array<Route*, 8>, 8> routes{};
    for (int src = 1; src <= 8; src++) {
        for (int dst = 1; dst <= 8; dst++) {
            if (src == dst) continue;
            routes[src-1][dst-1] = buildRoute(src, dst, tofs, gpus);
            gpus[src-1]->setRoute((uint8_t)dst, routes[src-1][dst-1]);
        }
    }

    // ---- 打印路由验证 ----
    std::cout << "Route hops (Dragonfly Sync, 3 hops/Tofino):\n";
    for (int src = 1; src <= 8; src++) {
        for (int dst = 1; dst <= 8; dst++) {
            if (src == dst) continue;
            Route* r = routes[src-1][dst-1];
            int swIdx_src = gpuToTofinoIdx(src);
            int swIdx_dst = gpuToTofinoIdx(dst);
            const char* pathType =
                (swIdx_src == swIdx_dst) ? "same-sw" :
                (swIdx_src / 2 == swIdx_dst / 2) ? "same-group" : "cross-group";
            std::cout << "  GPU" << src << " -> GPU" << dst
                      << ": " << r->size() << " route nodes"
                      << " (" << pathType << ")\n";
        }
    }
    std::cout << "\n";

    // ---- 打印拓扑 ----
    std::cout << "Dragonfly Sync Topology:\n"
              << "  Group 0: Sw0(GPU1,GPU2) <--local--> Sw1(GPU3,GPU4)\n"
              << "  Group 1: Sw2(GPU5,GPU6) <--local--> Sw3(GPU7,GPU8)\n"
              << "  Global:  Sw0 <---> Sw2,  Sw1 <---> Sw3\n"
              << "  Switch model: simplified (no BufferGate, 3 hops/Tofino)\n\n";

    // ---- 运行仿真 ----
    std::cout << "Running sync simulation...\n\n";
    while (EventList::doNextEvent()) {}

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
