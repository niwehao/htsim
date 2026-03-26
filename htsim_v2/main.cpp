/*
 * main.cpp  —  csg-htsim 版
 *
 * 翻译自: test_mqnic_sync_dcn.py (拓扑连线 + 启动)
 *          Spine.py (路由 / port_mapping 逻辑)
 *
 * 拓扑 (6 个 Tofino, 8 个 GPU):
 *
 *  Spine1.Leaf1 (S1L1, idx=0): ports [1,2,3,4]    GPU1@p2, GPU2@p4
 *  Spine1.Leaf2 (S1L2, idx=1): ports [5,6,7,8]    GPU3@p6, GPU4@p8
 *  Spine2.Leaf1 (S2L1, idx=2): ports [9,10,11,12] GPU5@p10, GPU6@p12
 *  Spine2.Leaf2 (S2L2, idx=3): ports [13,14,15,16] GPU7@p14, GPU8@p16
 *  Spine3.Leaf1 (S3L1, idx=4): ports [17,18,19,20] (互联)
 *  Spine3.Leaf2 (S3L2, idx=5): ports [21,22,23,24] (互联)
 *
 * 路由预计算:
 *   对每对 (srcGpu, dstGpu), 追踪经过的 Tofino 链:
 *     srcTofino[inPort→outPort] → port_mapping → nextTofino → ... → dstTofino → GPU.rxQ → AppSink
 *   得到完整 Route (csg-htsim PacketSink* 序列)
 */

#include "gpu_node.h"
#include <iostream>
#include <array>

// ================================================================
//  buildRoute: 为 (srcGpu → dstGpu) 构建完整 Route
//
//  翻译自: Spine.send_packet() 的路由追踪逻辑
//
//  路由过程:
//    1. srcGpu 的 txQueue (不在 Route 中) → sendOn() → Route[0]
//    2. Route 经过 1~3 个 Tofino, 每个 Tofino 占 5 个节点
//    3. Route 末尾: dstGpu.rxQueue + dstGpu.appSink
// ================================================================

Route* buildRoute(int srcGpu, int dstGpu,
                  std::array<TofinoSwitch*, 6>& tofs,
                  std::array<GpuNode*, 8>& gpus)
{
    Route* route = new Route();

    // 起点: srcGpu 所在 Tofino 和端口
    int curTofinoIdx = gpuToTofinoIdx(srcGpu);
    int inPort       = gpuToPort(srcGpu);

    // 反复追踪 Tofino 链, 直到到达 dstGpu
    for (int hop = 0; hop < 10; hop++) {  // 最多 10 跳 (安全上限)
        // 查当前 Tofino 的路由表: dstGpu → outPort
        auto rtable = resolvedRoutingTable(curTofinoIdx);
        int outPort = rtable.at(dstGpu);

        // 添加当前 Tofino 的 5 个 Route 节点
        tofs[curTofinoIdx]->appendToRoute(*route, inPort, outPort);

        // 检查 outPort 是否直达 dstGpu
        if (isGpuPort(outPort) && portToGpu(outPort) == dstGpu) {
            // Route 末尾: dstGpu 的 rxQueue + appSink
            route->push_back(gpus[dstGpu - 1]->rxQueue());
            route->push_back(gpus[dstGpu - 1]->appSink());
            return route;
        }

        // outPort 是互联端口 → port_mapping → 下一个 Tofino
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
    // ---- 仿真时长上限 ----
    EventList::setEndtime(timeFromSec(2000));

    std::cout << "=== MOE htsim Simulation (csg-htsim) ===\n"
              << "  GPUs=" << NUM_GPU_NODES
              << " Layers=" << TOTAL_LAYERS
              << " Frags=" << TOTAL_FRAGMENTS << "x" << (FRAGMENT_PAYLOAD_SIZE/1024) << "KB"
              << " Link=" << PROT_RATE_Gbps << "Gbps"
              << " Timeout=" << timeAsMs(TIMEOUT_PS) << "ms\n\n";

    // ---- 创建 6 个 TofinoSwitch ----
    std::array<TofinoSwitch*, 6> tofs;
    for (int i = 0; i < 6; i++)
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

    // ---- 创建 PacketFlow (每 GPU 一个) ----
    std::array<PacketFlow*, 8> flows;
    for (int i = 0; i < 8; i++) {
        flows[i] = new PacketFlow(nullptr);
        flows[i]->set_flowid((flowid_t)(i + 1));
        gpus[i]->setFlow(flows[i]);
    }

    // ---- 预计算所有 56 条路由 (8 GPU × 7 目标) ----
    // routes[src-1][dst-1] = Route*  (src != dst)
    std::array<std::array<Route*, 8>, 8> routes{};
    for (int src = 1; src <= 8; src++) {
        for (int dst = 1; dst <= 8; dst++) {
            if (src == dst) continue;
            routes[src-1][dst-1] = buildRoute(src, dst, tofs, gpus);
            gpus[src-1]->setRoute((uint8_t)dst, routes[src-1][dst-1]);
        }
    }

    // ---- 打印路由验证 (前 2 条) ----
    std::cout << "Route examples:\n";
    for (int dst : {2, 5}) {
        Route* r = routes[0][dst-1];
        std::cout << "  GPU1→GPU" << dst << ": " << r->hop_count() << " hops\n";
    }
    std::cout << "\n";

    // ---- 打印拓扑 ----
    std::cout << "Topology:\n"
              << "  S1L1(0): GPU1,GPU2   S1L2(1): GPU3,GPU4\n"
              << "  S2L1(2): GPU5,GPU6   S2L2(3): GPU7,GPU8\n"
              << "  S3L1(4): interconnect S3L2(5): interconnect\n\n";

    // ---- 运行仿真 ----
    std::cout << "Running simulation...\n\n";
    while (EventList::doNextEvent()) {}
 

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
