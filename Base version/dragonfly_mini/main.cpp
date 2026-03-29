/*
 * main.cpp  —  Dragonfly 拓扑版
 *
 * 拓扑 (4 个交换机, 8 个 GPU):
 *
 *  Group 0:
 *    Sw0: ports {1,2,3,4}   GPU1@p1, GPU2@p2, local@p3, global@p4
 *    Sw1: ports {5,6,7,8}   GPU3@p5, GPU4@p6, local@p7, global@p8
 *    Sw0 ←local→ Sw1 (port 3 ↔ port 7)
 *
 *  Group 1:
 *    Sw2: ports {9,10,11,12}  GPU5@p9, GPU6@p10, local@p11, global@p12
 *    Sw3: ports {13,14,15,16} GPU7@p13, GPU8@p14, local@p15, global@p16
 *    Sw2 ←local→ Sw3 (port 11 ↔ port 15)
 *
 *  Global links:
 *    Sw0 ↔ Sw2 (port 4 ↔ port 12)
 *    Sw1 ↔ Sw3 (port 8 ↔ port 16)
 *
 * 路由:
 *   同交换机: 1 hop (5 route nodes)
 *   同 Group: 2 hops (10 route nodes)
 *   跨 Group: 2-3 hops (10-15 route nodes)
 */

#include "gpu_node.h"
#include <iostream>
#include <array>

// ================================================================
//  buildRoute: 为 (srcGpu → dstGpu) 构建完整 Route
// ================================================================

Route* buildRoute(int srcGpu, int dstGpu,
                  std::array<TofinoSwitch*, 4>& tofs,
                  std::array<GpuNode*, 8>& gpus)
{
    Route* route = new Route();

    int curTofinoIdx = gpuToTofinoIdx(srcGpu);
    int inPort       = gpuToPort(srcGpu);

    for (int hop = 0; hop < 10; hop++) {
        auto rtable = resolvedRoutingTable(curTofinoIdx);
        int outPort = rtable.at(dstGpu);

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

    std::cout << "=== MOE htsim Simulation (Dragonfly Topology) ===\n"
              << "  GPUs=" << NUM_GPU_NODES
              << " Switches=" << NUM_SWITCHES
              << " Groups=" << NUM_GROUPS
              << " Layers=" << TOTAL_LAYERS
              << " Frags=" << TOTAL_FRAGMENTS << "x" << (FRAGMENT_PAYLOAD_SIZE/1024) << "KB"
              << " Link=" << PROT_RATE_Gbps << "Gbps"
              << " Timeout=" << timeAsMs(TIMEOUT_PS) << "ms\n\n";

    // ---- 创建 4 个交换机 ----
    std::array<TofinoSwitch*, 4> tofs;
    for (int i = 0; i < 4; i++)
        tofs[i] = new TofinoSwitch(i);

    // ---- 创建 8 个 GpuNode ----
    std::array<GpuNode*, 8> gpus;
    auto makePeers = [](uint8_t self) {
        std::vector<uint8_t> v;
        for (uint8_t i = 1; i <= 8; i++) if (i != self) v.push_back(i);
        return v;
    };
    for (int i = 0; i < 8; i++)
        gpus[i] = new GpuNode((uint8_t)(i + 1), makePeers(i + 1));

    // ---- 创建 PacketFlow ----
    std::array<PacketFlow*, 8> flows;
    for (int i = 0; i < 8; i++) {
        flows[i] = new PacketFlow(nullptr);
        flows[i]->set_flowid((flowid_t)(i + 1));
        gpus[i]->setFlow(flows[i]);
    }

    // ---- 预计算路由 ----
    std::array<std::array<Route*, 8>, 8> routes{};
    for (int src = 1; src <= 8; src++) {
        for (int dst = 1; dst <= 8; dst++) {
            if (src == dst) continue;
            routes[src-1][dst-1] = buildRoute(src, dst, tofs, gpus);
            gpus[src-1]->setRoute((uint8_t)dst, routes[src-1][dst-1]);
        }
    }

    // ---- 打印路由 ----
    std::cout << "Route hops (Dragonfly):\n";
    for (int src = 1; src <= 8; src++) {
        for (int dst = 1; dst <= 8; dst++) {
            if (src == dst) continue;
            Route* r = routes[src-1][dst-1];
            std::cout << "  GPU" << src << " -> GPU" << dst << ": " << r->size() << " hops\n";
        }
    }
    std::cout << "\n";

    // ---- 打印拓扑 ----
    std::cout << "Dragonfly Topology:\n"
              << "  Group 0: Sw0(GPU1,GPU2) <--local--> Sw1(GPU3,GPU4)\n"
              << "  Group 1: Sw2(GPU5,GPU6) <--local--> Sw3(GPU7,GPU8)\n"
              << "  Global:  Sw0 <---> Sw2,  Sw1 <---> Sw3\n\n";

    // ---- 运行仿真 ----
    std::cout << "Running simulation...\n\n";
    while (EventList::doNextEvent()) {}

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
