/*
 * main.cpp  —  512-GPU 2-Level Leaf-Spine 拓扑
 *
 * 拓扑自动构建 + 路由自动计算:
 *
 *   32 Leaf switches:  端口 0..15 (下行→GPU), 端口 16..31 (上行→Spine)
 *   16 Spine switches: 端口 0..31 (→Leaf)
 *
 *   GPU g (0-indexed):
 *     leaf  = g / 16
 *     port  = g % 16
 *
 *   路由 src → dst:
 *     同 Leaf: Leaf[leaf](port_src → port_dst)
 *     跨 Leaf: Leaf[src_leaf](port_src → 16+spine)
 *              → Spine[spine](src_leaf → dst_leaf)
 *              → Leaf[dst_leaf](16+spine → port_dst)
 *     ECMP:  spine = (src + dst) % NUM_SPINES
 *
 *   修改 constants.h 中 NUM_GPU_NODES 即可自动缩放.
 */

#include "gpu_node.h"
#include <iostream>
#include <vector>
#include <ctime>    // 包含 time
#include <cstdlib>  // 包含 srand, rand

// ================================================================
//  buildRoute: 自动路由计算
// ================================================================

Route* buildRoute(uint32_t srcGpu, uint32_t dstGpu,
                  std::vector<TofinoSwitch*>& leaves,
                  std::vector<TofinoSwitch*>& spines,
                  std::vector<GpuNode*>& gpus)
{
    Route* route = new Route();

    uint32_t srcLeaf  = gpuToLeaf(srcGpu);
    uint32_t dstLeaf  = gpuToLeaf(dstGpu);
    uint32_t srcPort  = gpuToLocalPort(srcGpu);  // 0..15
    uint32_t dstPort  = gpuToLocalPort(dstGpu);  // 0..15

    if (srcLeaf == dstLeaf) {
        // 同 Leaf: 1 switch hop
        leaves[srcLeaf]->appendToRoute(*route, srcPort, dstPort);
    } else {
        // 跨 Leaf: 3 switch hops
        uint32_t spine = spineSelectECMP(srcGpu, dstGpu);

        // Hop 1: Leaf[srcLeaf] — GPU 下行端口 → Spine 上行端口
        leaves[srcLeaf]->appendToRoute(*route, srcPort, leafUpPort(spine));

        // Hop 2: Spine[spine] — srcLeaf 端口 → dstLeaf 端口
        spines[spine]->appendToRoute(*route, srcLeaf, dstLeaf);

        // Hop 3: Leaf[dstLeaf] — Spine 上行端口 → GPU 下行端口
        leaves[dstLeaf]->appendToRoute(*route, leafUpPort(spine), dstPort);
    }
   
    // Route 末尾: dstGpu 的 rxQueue + appSink
    route->push_back(gpus[dstGpu]->rxQueue());
    route->push_back(gpus[dstGpu]->appSink());
    return route;
}

// ================================================================
//  main
// ================================================================

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    EventList::setEndtime(timeFromSec(20000));
    

    std::cout << "=== MOE htsim Simulation — 2-Level Leaf-Spine ===\n"
              << "  GPUs=" << NUM_GPU_NODES
              << " Leaves=" << NUM_LEAVES
              << " Spines=" << NUM_SPINES
              << " PortsPerSwitch=" << PORTS_PER_SWITCH << "\n"
              << "  Layers=" << TOTAL_LAYERS
              << " Frags=" << TOTAL_FRAGMENTS << "x" << (FRAGMENT_PAYLOAD_SIZE/1024) << "KB"
              << " Link=" << PROT_RATE_Gbps << "Gbps"
              << " Timeout=" << timeAsMs(TIMEOUT_PS) << "ms\n\n";

    // ---- 创建 Leaf 交换机 (32 个, 各 32 端口) ----
    std::vector<TofinoSwitch*> leaves(NUM_LEAVES);
    for (uint32_t l = 0; l < NUM_LEAVES; l++) {
        std::vector<uint32_t> ports;
        for (uint32_t p = 0; p < PORTS_PER_SWITCH; p++)
            ports.push_back(p);
        leaves[l] = new TofinoSwitch((int)l, ports);
    }
    std::cout << "  Created " << NUM_LEAVES << " Leaf switches\n";

    // ---- 创建 Spine 交换机 (16 个, 各 NUM_LEAVES 端口) ----
    std::vector<TofinoSwitch*> spines(NUM_SPINES);
    for (uint32_t s = 0; s < NUM_SPINES; s++) {
        std::vector<uint32_t> ports;
        for (uint32_t p = 0; p < NUM_LEAVES; p++)
            ports.push_back(p);
        spines[s] = new TofinoSwitch((int)(NUM_LEAVES + s), ports);
    }
    std::cout << "  Created " << NUM_SPINES << " Spine switches\n";

    // ---- 创建 GPU 节点 (512 个) ----
    std::vector<GpuNode*> gpus(NUM_GPU_NODES);
    for (uint32_t g = 0; g < NUM_GPU_NODES; g++) {
        std::vector<uint16_t> peers;
        for (uint32_t p = 0; p < NUM_GPU_NODES; p++)
            if (p != g) peers.push_back((uint16_t)p);
        gpus[g] = new GpuNode((uint16_t)g, peers);
    }
    std::cout << "  Created " << NUM_GPU_NODES << " GPU nodes\n";

    // ---- PacketFlow (每 GPU 一个) ----
    std::vector<PacketFlow*> flows(NUM_GPU_NODES);
    for (uint32_t g = 0; g < NUM_GPU_NODES; g++) {
        flows[g] = new PacketFlow(nullptr);
        flows[g]->set_flowid((flowid_t)g);
        gpus[g]->setFlow(flows[g]);
    }

    // ---- 预计算所有路由 ----
    std::cout << "\n  Computing routes for " << NUM_GPU_NODES
              << " x " << (NUM_GPU_NODES - 1) << " = "
              << (uint64_t)NUM_GPU_NODES * (NUM_GPU_NODES - 1) << " pairs...\n";

    uint64_t sameLeafCount = 0, crossLeafCount = 0;
    for (uint32_t src = 0; src < NUM_GPU_NODES; src++) {
        for (uint32_t dst = 0; dst < NUM_GPU_NODES; dst++) {
            if (src == dst) continue;
            Route* r = buildRoute(src, dst, leaves, spines, gpus);
            gpus[src]->setRoute((uint16_t)dst, r);

            if (gpuToLeaf(src) == gpuToLeaf(dst))
                sameLeafCount++;
            else
                crossLeafCount++;
        }
    }
    std::cout << "  Routes: " << sameLeafCount << " intra-leaf (7 hops), "
              << crossLeafCount << " inter-leaf (17 hops)\n";

    // ---- 打印拓扑摘要 ----
    std::cout << "\nTopology:\n";
    for (uint32_t l = 0; l < NUM_LEAVES; l++) {
        if (l % 8 == 0) std::cout << "  ";
        std::cout << "L" << l << "[GPU" << l*GPUS_PER_LEAF
                  << "-" << (l+1)*GPUS_PER_LEAF - 1 << "]";
        if (l % 8 == 7) std::cout << "\n";
        else std::cout << "  ";
    }
    std::cout << "  Spine0..Spine" << NUM_SPINES - 1
              << " (each connects all " << NUM_LEAVES << " Leaves)\n";

    // ---- ECMP 分布验证 ----
    std::vector<uint64_t> spineLoad(NUM_SPINES, 0);
    for (uint32_t src = 0; src < NUM_GPU_NODES; src++)
        for (uint32_t dst = 0; dst < NUM_GPU_NODES; dst++)
            if (src != dst && gpuToLeaf(src) != gpuToLeaf(dst))
                spineLoad[spineSelectECMP(src, dst)]++;

    std::cout << "\nECMP distribution (cross-leaf flows per spine):\n  ";
    for (uint32_t s = 0; s < NUM_SPINES; s++)
        std::cout << "S" << s << "=" << spineLoad[s] << " ";
    std::cout << "\n";

    // ---- 运行仿真 ----
    std::cout << "\nRunning simulation...\n\n";
    while (EventList::doNextEvent()) {}

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
