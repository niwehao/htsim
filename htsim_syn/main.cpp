/*
 * main.cpp  —  同步调度版 MOE htsim 仿真
 *
 * 流程:
 *   1. 创建网络拓扑 (简化 Tofino: 无 BufferGate)
 *   2. 离线计算最优调度表 (SyncScheduler)
 *   3. 各 GPU 按调度表定时发送 → 零拥塞、零重传
 *   4. 运行仿真, 验证结果
 */

#include "gpu_node_sync.h"
#include <iostream>
#include <array>

// ================================================================
//  buildRoute: 同步版路由构建 (3 hops per Tofino, not 5)
// ================================================================
Route* buildRoute(int srcGpu, int dstGpu,
                  std::array<TofinoSwitchSync*, 6>& tofs,
                  std::array<GpuNodeSync*, 8>& gpus)
{
    Route* route = new Route();

    int curTofinoIdx = gpuToTofinoIdx(srcGpu);
    int inPort       = gpuToPort(srcGpu);

    for (int hop = 0; hop < 10; hop++) {
        auto rtable = resolvedRoutingTable(curTofinoIdx);
        int outPort = rtable.at(dstGpu);

        // 3 个节点: ingressQ → fabricQ → egressQ
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
    EventList::setEndtime(timeFromSec(200));

    std::cout << "=== MOE Sync-Scheduled Simulation (csg-htsim) ===\n"
              << "  GPUs=" << NUM_GPU_NODES
              << " Layers=" << TOTAL_LAYERS
              << " Frags=" << TOTAL_FRAGMENTS << "x" << (FRAGMENT_PAYLOAD_SIZE/1024) << "KB"
              << " Link=" << PROT_RATE_Gbps << "Gbps\n\n";

    // ---- Step 1: 计算调度表 ----
    std::cout << "Computing optimal schedule...\n";
    int pktSize = 14 + 15 + (int)FRAGMENT_PAYLOAD_SIZE;
    simtime_picosec phaseDuration = 0;
    auto scheduleMap = SyncScheduler::computePhaseSchedule(pktSize, &phaseDuration);
    SyncScheduler::printScheduleStats(scheduleMap, phaseDuration);

    simtime_picosec totalEstimate = (simtime_picosec)TOTAL_LAYERS * 2 *
        (phaseDuration + INTERPHASE_GAP_PS);
    std::cout << "Estimated total time: " << std::fixed << std::setprecision(3)
              << (double)totalEstimate / 1e9 << " ms ("
              << TOTAL_LAYERS << " layers x 2 phases)\n\n";

    // ---- Step 2: 创建简化 Tofino ----
    std::array<TofinoSwitchSync*, 6> tofs;
    for (int i = 0; i < 6; i++)
        tofs[i] = new TofinoSwitchSync(i);

    // ---- Step 3: 创建 GPU 节点 (传入调度表) ----
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

    // ---- Step 4: 创建 PacketFlow ----
    std::array<PacketFlow*, 8> flows;
    for (int i = 0; i < 8; i++) {
        flows[i] = new PacketFlow();
        flows[i]->set_flowid((flowid_t)(i + 1));
        gpus[i]->setFlow(flows[i]);
    }

    // ---- Step 5: 预计算路由 ----
    std::array<std::array<Route*, 8>, 8> routes{};
    for (int src = 1; src <= 8; src++) {
        for (int dst = 1; dst <= 8; dst++) {
            if (src == dst) continue;
            routes[src-1][dst-1] = buildRoute(src, dst, tofs, gpus);
            gpus[src-1]->setRoute((uint8_t)dst, routes[src-1][dst-1]);
        }
    }

    // ---- 打印路由验证 ----
    std::cout << "Route examples (sync, 3 hops/Tofino):\n";
    for (int dst : {2, 5}) {
        Route* r = routes[0][dst-1];
        std::cout << "  GPU1->GPU" << dst << ": " << r->hop_count() << " hops\n";
    }
    std::cout << "\n";

    // ---- 运行仿真 ----
    std::cout << "Running sync simulation...\n\n";
    while (EventList::doNextEvent()) {}

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
