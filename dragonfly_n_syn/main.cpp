/*
 * main.cpp  —  N-GPU Dragonfly 同步调度版 MOE 仿真 (全自动拓扑 + ECMP)
 *
 * 流程:
 *   1. initDragonflyTopology() 构建全局链路表 + ECMP
 *   2. SyncScheduler 离线计算最优调度表 (带缓存)
 *   3. 创建 Dragonfly 拓扑 + GPU 节点
 *   4. 各 GPU 按调度表定时发送 → 零拥塞、零重传
 *   5. 运行仿真, 验证结果
 *
 * 只需修改 NUM_GPU_NODES / PORTS_PER_SWITCH, 一切自动推导.
 */

#include "gpu_node_sync.h"
#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>

// ================================================================
//  buildRoute: 基于 computeSwitchPath 自动构建 Route
// ================================================================

Route* buildRoute(uint32_t srcGpu, uint32_t dstGpu,
                  std::vector<TofinoSwitchSync*>& switches,
                  std::vector<GpuNodeSync*>& gpus)
{
    Route* route = new Route();
    SwitchPath path = computeSwitchPath(srcGpu, dstGpu);

    for (uint32_t h = 0; h < path.numHops; h++)
        switches[path.sw[h]]->appendToRoute(*route, path.inPort[h], path.outPort[h]);

    route->push_back(gpus[dstGpu]->rxQueue());
    route->push_back(gpus[dstGpu]->appSink());
    return route;
}

// ================================================================
//  printTopology: 打印 Dragonfly 拓扑
// ================================================================

void printTopology() {
    std::cout << "\n  === Dragonfly Topology (auto-derived) ===\n"
              << "  Groups=" << NUM_GROUPS
              << " SwitchesPerGroup(a)=" << SWITCHES_PER_GROUP
              << " GPUsPerSwitch(p)=" << GPUS_PER_SWITCH << "\n"
              << "  Ports: [0.." << GPUS_PER_SWITCH-1 << "] GPU"
              << " | [" << LOCAL_PORT_BASE << ".." << LOCAL_PORT_BASE+LOCAL_PORTS-1 << "] local(" << LOCAL_PORTS << ")"
              << " | [" << GLOBAL_PORT_BASE << ".." << PORTS_PER_SWITCH-1 << "] global(" << GLOBAL_PORTS_PER_SW << ")\n"
              << "  Total: " << NUM_SWITCHES << " switches, "
              << NUM_GPU_NODES << " GPUs\n\n";

    std::cout << "  Global link map:\n";
    for (uint32_t g = 0; g < NUM_GROUPS; g++) {
        for (uint32_t j = 0; j < SWITCHES_PER_GROUP; j++) {
            uint32_t sw = g * SWITCHES_PER_GROUP + j;
            std::cout << "    Sw" << sw << " (G" << g << ".s" << j << "): ";
            for (uint32_t gp = 0; gp < GLOBAL_PORTS_PER_SW; gp++) {
                uint32_t tSw = g_glTarget[sw][gp];
                if (tSw == UINT32_MAX) { std::cout << "p" << (GLOBAL_PORT_BASE+gp) << "=- "; continue; }
                std::cout << "p" << (GLOBAL_PORT_BASE+gp) << "→Sw" << tSw
                          << "(G" << switchToGroup(tSw) << ") ";
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n  ECMP routing table (sw → target group → via):\n";
    for (uint32_t sw = 0; sw < std::min(NUM_SWITCHES, 4u); sw++) {
        uint32_t myG = switchToGroup(sw);
        std::cout << "    Sw" << sw << ": ";
        for (uint32_t g = 0; g < NUM_GROUPS; g++) {
            if (g == myG) { std::cout << "G" << g << "=local "; continue; }
            auto& links = g_routeLinks[sw][g];
            std::cout << "G" << g << "→[";
            for (size_t i = 0; i < links.size(); i++) {
                if (i > 0) std::cout << ",";
                std::cout << "Sw" << links[i].remSw
                          << "(p" << (GLOBAL_PORT_BASE + links[i].gp) << ")";
            }
            std::cout << "] ";
        }
        std::cout << "\n";
    }
    if (NUM_SWITCHES > 4) std::cout << "    ... (" << NUM_SWITCHES - 4 << " more)\n";
    std::cout << "\n";
}

// ================================================================
//  main
// ================================================================

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    EventList::setEndtime(timeFromSec(200000));

    std::cout << "=== MOE Sync-Scheduled Simulation — Dragonfly (balanced-topo + ECMP) ===\n"
              << "  GPUs=" << NUM_GPU_NODES
              << " Switches=" << NUM_SWITCHES
              << " Groups=" << NUM_GROUPS
              << " a=" << SWITCHES_PER_GROUP
              << " p=" << GPUS_PER_SWITCH
              << " h=" << GLOBAL_PORTS_PER_SW
              << " PortsPerSwitch=" << PORTS_PER_SWITCH
              << " |a-2h|=" << ((int)SWITCHES_PER_GROUP > 2*(int)GLOBAL_PORTS_PER_SW
                  ? (int)SWITCHES_PER_GROUP - 2*(int)GLOBAL_PORTS_PER_SW
                  : 2*(int)GLOBAL_PORTS_PER_SW - (int)SWITCHES_PER_GROUP) << "\n"
              << "  Layers=" << TOTAL_LAYERS
              << " Frags=" << TOTAL_FRAGMENTS << "x" << (FRAGMENT_PAYLOAD_SIZE/1024) << "KB"
              << " Link=" << PROT_RATE_Gbps << "Gbps\n";

    // ---- Step 0: 初始化 Dragonfly 全局链路表 ----
    initDragonflyTopology();
    printTopology();

    // ---- Step 1: 计算调度表 ----
    std::cout << "Computing optimal schedule (this may take a while for "
              << NUM_GPU_NODES << " GPUs)...\n";
    int pktSize = DATA_PKT_SIZE;
    simtime_picosec phaseDuration = 0;
    auto scheduleMap = SyncScheduler::computePhaseSchedule(pktSize, &phaseDuration);
    SyncScheduler::printScheduleStats(scheduleMap, phaseDuration);

    simtime_picosec totalEstimate = (simtime_picosec)TOTAL_LAYERS * 2 *
        (phaseDuration + INTERPHASE_GAP_PS);
    std::cout << "Estimated total time: " << std::fixed << std::setprecision(3)
              << (double)totalEstimate / 1e9 << " ms ("
              << TOTAL_LAYERS << " layers x 2 phases)\n\n";

    // ---- Step 2: 创建交换机 ----
    std::vector<TofinoSwitchSync*> switches(NUM_SWITCHES);
    for (uint32_t sw = 0; sw < NUM_SWITCHES; sw++) {
        std::vector<uint32_t> ports;
        for (uint32_t p = 0; p < PORTS_PER_SWITCH; p++)
            ports.push_back(p);
        switches[sw] = new TofinoSwitchSync((int)sw, ports);
    }
    std::cout << "  Created " << NUM_SWITCHES << " switches ("
              << NUM_GROUPS << " groups x " << SWITCHES_PER_GROUP << ")\n";

    // ---- Step 3: 创建 GPU 节点 (传入调度表) ----
    std::vector<GpuNodeSync*> gpus(NUM_GPU_NODES);
    for (uint32_t g = 0; g < NUM_GPU_NODES; g++) {
        std::vector<uint16_t> peers;
        for (uint32_t p = 0; p < NUM_GPU_NODES; p++)
            if (p != g) peers.push_back((uint16_t)p);
        gpus[g] = new GpuNodeSync((uint16_t)g, peers, scheduleMap[g]);
    }
    std::cout << "  Created " << NUM_GPU_NODES << " GPU nodes\n";

    // ---- Step 4: PacketFlow ----
    std::vector<PacketFlow*> flows(NUM_GPU_NODES);
    for (uint32_t g = 0; g < NUM_GPU_NODES; g++) {
        flows[g] = new PacketFlow(nullptr);
        flows[g]->set_flowid((flowid_t)g);
        gpus[g]->setFlow(flows[g]);
    }

    // ---- Step 5: 预计算路由 ----
    std::cout << "\n  Computing routes...\n";
    uint64_t hop1 = 0, hop2 = 0, hop3 = 0;
    for (uint32_t src = 0; src < NUM_GPU_NODES; src++) {
        for (uint32_t dst = 0; dst < NUM_GPU_NODES; dst++) {
            if (src == dst) continue;
            Route* r = buildRoute(src, dst, switches, gpus);
            gpus[src]->setRoute((uint16_t)dst, r);

            SwitchPath path = computeSwitchPath(src, dst);
            if (path.numHops == 1) hop1++;
            else if (path.numHops == 2) hop2++;
            else hop3++;
        }
    }
    std::cout << "  Routes: " << hop1 << " x 1-hop (same-switch), "
              << hop2 << " x 2-hop, "
              << hop3 << " x 3-hop\n";

    // ---- Step 6: Global 链路负载验证 ----
    std::cout << "\n  Global link load (routes per switch global port):\n";
    for (uint32_t sw = 0; sw < NUM_SWITCHES; sw++) {
        uint32_t total = 0;
        std::string detail;
        for (uint32_t gp = 0; gp < GLOBAL_PORTS_PER_SW; gp++) {
            uint32_t count = 0;
            for (uint32_t src = 0; src < NUM_GPU_NODES; src++) {
                for (uint32_t dst = 0; dst < NUM_GPU_NODES; dst++) {
                    if (src == dst) continue;
                    SwitchPath path = computeSwitchPath(src, dst);
                    for (uint32_t h = 0; h < path.numHops; h++) {
                        if (path.sw[h] == sw && path.outPort[h] == GLOBAL_PORT_BASE + gp)
                            count++;
                    }
                }
            }
            if (count > 0)
                detail += "p" + std::to_string(GLOBAL_PORT_BASE + gp) + "=" + std::to_string(count) + " ";
            total += count;
        }
        if (total > 0)
            std::cout << "    Sw" << sw << " (G" << switchToGroup(sw) << ".s" << switchInGroup(sw) << "): " << detail << "\n";
    }

    // ---- 运行仿真 ----
    std::cout << "\nRunning Dragonfly sync simulation...\n\n";
    while (EventList::doNextEvent()) {}

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
