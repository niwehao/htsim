/*
 * main.cpp  —  512-GPU 同步调度版 MOE 仿真
 *
 * 流程:
 *   1. 离线计算最优调度表 (SyncScheduler)
 *   2. 创建 2 层 Leaf-Spine 拓扑 (自动路由)
 *   3. 各 GPU 按调度表定时发送 → 零拥塞、零重传
 *   4. 运行仿真, 验证结果
 *
 * 拓扑:
 *   32 Leaf (端口 0..15 下行, 16..31 上行) × 16 Spine (端口 0..31)
 *   路由自动计算, 修改 NUM_GPU_NODES 即可缩放.
 */

#include "gpu_node_sync.h"
#include <iostream>
#include <vector>

// ================================================================
//  buildRoute: 自动路由 (5 hops per switch: ingress→gate→fabric→release→egress)
// ================================================================

Route* buildRoute(uint32_t srcGpu, uint32_t dstGpu,
                  std::vector<TofinoSwitchSync*>& leaves,
                  std::vector<TofinoSwitchSync*>& spines,
                  std::vector<GpuNodeSync*>& gpus)
{
    Route* route = new Route();

    uint32_t srcLeaf = gpuToLeaf(srcGpu);
    uint32_t dstLeaf = gpuToLeaf(dstGpu);
    uint32_t srcPort = gpuToLocalPort(srcGpu);
    uint32_t dstPort = gpuToLocalPort(dstGpu);

    if (srcLeaf == dstLeaf) {
        leaves[srcLeaf]->appendToRoute(*route, srcPort, dstPort);
    } else {
        uint32_t spine = spineSelectECMP(srcGpu, dstGpu);
        leaves[srcLeaf]->appendToRoute(*route, srcPort, leafUpPort(spine));
        spines[spine]->appendToRoute(*route, srcLeaf, dstLeaf);
        leaves[dstLeaf]->appendToRoute(*route, leafUpPort(spine), dstPort);
    }

    route->push_back(gpus[dstGpu]->rxQueue());
    route->push_back(gpus[dstGpu]->appSink());
    return route;
}

// ================================================================
//  main
// ================================================================

int main() {
    EventList::setEndtime(timeFromSec(200000));

    std::cout << "=== MOE Sync-Scheduled Simulation — 2-Level Leaf-Spine ===\n"
              << "  GPUs=" << NUM_GPU_NODES
              << " Leaves=" << NUM_LEAVES
              << " Spines=" << NUM_SPINES
              << " PortsPerSwitch=" << PORTS_PER_SWITCH << "\n"
              << "  Layers=" << TOTAL_LAYERS
              << " Frags=" << TOTAL_FRAGMENTS << "x" << (FRAGMENT_PAYLOAD_SIZE/1024) << "KB"
              << " Link=" << PROT_RATE_Gbps << "Gbps\n\n";

    // ---- Step 1: 计算调度表 ----
    std::cout << "Computing optimal schedule (this may take a while for "
              << NUM_GPU_NODES << " GPUs)...\n";
    int pktSize = 14 + 15 + (int)FRAGMENT_PAYLOAD_SIZE;
    simtime_picosec phaseDuration = 0;
    auto scheduleMap = SyncScheduler::computePhaseSchedule(pktSize, &phaseDuration);
    SyncScheduler::printScheduleStats(scheduleMap, phaseDuration);

    simtime_picosec totalEstimate = (simtime_picosec)TOTAL_LAYERS * 2 *
        (phaseDuration + INTERPHASE_GAP_PS);
    std::cout << "Estimated total time: " << std::fixed << std::setprecision(3)
              << (double)totalEstimate / 1e9 << " ms ("
              << TOTAL_LAYERS << " layers x 2 phases)\n\n";

    // ---- Step 2: 创建 Leaf 交换机 ----
    std::vector<TofinoSwitchSync*> leaves(NUM_LEAVES);
    for (uint32_t l = 0; l < NUM_LEAVES; l++) {
        std::vector<uint32_t> ports;
        for (uint32_t p = 0; p < PORTS_PER_SWITCH; p++)
            ports.push_back(p);
        leaves[l] = new TofinoSwitchSync((int)l, ports);
    }
    std::cout << "  Created " << NUM_LEAVES << " Leaf switches\n";

    // ---- Step 3: 创建 Spine 交换机 ----
    std::vector<TofinoSwitchSync*> spines(NUM_SPINES);
    for (uint32_t s = 0; s < NUM_SPINES; s++) {
        std::vector<uint32_t> ports;
        for (uint32_t p = 0; p < NUM_LEAVES; p++)
            ports.push_back(p);
        spines[s] = new TofinoSwitchSync((int)(NUM_LEAVES + s), ports);
    }
    std::cout << "  Created " << NUM_SPINES << " Spine switches\n";

    // ---- Step 4: 创建 GPU 节点 (传入调度表) ----
    std::vector<GpuNodeSync*> gpus(NUM_GPU_NODES);
    for (uint32_t g = 0; g < NUM_GPU_NODES; g++) {
        std::vector<uint16_t> peers;
        for (uint32_t p = 0; p < NUM_GPU_NODES; p++)
            if (p != g) peers.push_back((uint16_t)p);
        gpus[g] = new GpuNodeSync((uint16_t)g, peers, scheduleMap[g]);
    }
    std::cout << "  Created " << NUM_GPU_NODES << " GPU nodes\n";

    // ---- Step 5: PacketFlow ----
    std::vector<PacketFlow*> flows(NUM_GPU_NODES);
    for (uint32_t g = 0; g < NUM_GPU_NODES; g++) {
        flows[g] = new PacketFlow(nullptr);
        flows[g]->set_flowid((flowid_t)g);
        gpus[g]->setFlow(flows[g]);
    }

    // ---- Step 6: 预计算路由 ----
    std::cout << "\n  Computing routes...\n";
    uint64_t sameLeafCount = 0, crossLeafCount = 0;
    for (uint32_t src = 0; src < NUM_GPU_NODES; src++) {
        for (uint32_t dst = 0; dst < NUM_GPU_NODES; dst++) {
            if (src == dst) continue;
            Route* r = buildRoute(src, dst, leaves, spines, gpus);
            gpus[src]->setRoute((uint16_t)dst, r);
            if (gpuToLeaf(src) == gpuToLeaf(dst)) sameLeafCount++;
            else crossLeafCount++;
        }
    }
    std::cout << "  Routes: " << sameLeafCount << " intra-leaf, "
              << crossLeafCount << " inter-leaf\n";

    // ---- Step 7: ECMP 分布验证 ----
    std::vector<uint64_t> spineLoad(NUM_SPINES, 0);
    for (uint32_t src = 0; src < NUM_GPU_NODES; src++)
        for (uint32_t dst = 0; dst < NUM_GPU_NODES; dst++)
            if (src != dst && gpuToLeaf(src) != gpuToLeaf(dst))
                spineLoad[spineSelectECMP(src, dst)]++;
    std::cout << "  ECMP distribution: ";
    for (uint32_t s = 0; s < NUM_SPINES; s++)
        std::cout << "S" << s << "=" << spineLoad[s] << " ";
    std::cout << "\n";

    // ---- 运行仿真 ----
    std::cout << "\nRunning sync simulation...\n\n";
    while (EventList::doNextEvent()) {}

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
