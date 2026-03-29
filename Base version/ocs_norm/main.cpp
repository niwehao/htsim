/*
 * main.cpp  —  OCS direct-only routing (no multi-hop)
 *
 * Topology: 8 GPUs connected via 1 OCS switch
 *
 * Routing: direct-only (no BFS, no multi-hop)
 *   - If current slice connects to destination → send
 *   - Otherwise → buffer until direct connection slice
 *
 * Route per packet: [EcsBuffer@src, rxQ_dst, AppSink_dst]
 */

#include "gpu_node.h"
#include <iostream>
#include <array>

// ================================================================
//  buildRoute: [EcsBuffer@src, rxQ_dst, AppSink_dst]
// ================================================================

Route* buildRoute(int srcGpu, int dstGpu,
                  OcsSwitch* ocs,
                  std::array<GpuNode*, 8>& gpus)
{
    Route* route = new Route();
    route->push_back(ocs->getBuffer(srcGpu));
    route->push_back(gpus[dstGpu - 1]->rxQueue());
    route->push_back(gpus[dstGpu - 1]->appSink());
    return route;
}

// ================================================================
//  main
// ================================================================

int main() {
    EventList::setEndtime(timeFromSec(2000));

    std::cout << "=== MOE htsim Simulation (OCS Direct-Only) ===\n"
              << "  GPUs=" << NUM_GPU_NODES
              << " Switch=1 (OCS, direct-only, no multi-hop)"
              << " Layers=" << TOTAL_LAYERS
              << " Frags=" << TOTAL_FRAGMENTS << "x" << (FRAGMENT_PAYLOAD_SIZE/1024) << "KB"
              << " Link=" << PROT_RATE_Gbps << "Gbps\n"
              << "  SliceActive=" << std::fixed << std::setprecision(1)
              << (double)SLICE_ACTIVE_PS / 1e6 << "us"
              << " Reconfig=" << (double)RECONFIG_DELAY_PS / 1e6 << "us"
              << " SliceTotal=" << (double)SLICE_TOTAL_PS / 1e6 << "us"
              << " Cycle=" << (double)CYCLE_PS / 1e6 << "us"
              << " EcsBuf=" << (ECS_BUFFER_SIZE / 1024 / 1024) << "MB"
              << " OcsTxBuf=" << (OCS_TX_BUFFER / 1024 / 1024) << "MB"
              << " Timeout=" << timeAsMs(TIMEOUT_PS) << "ms\n\n";

    // ---- Generate perfect matchings (Circle Method) ----
    HohooSchedule sched = computeHohooSchedule(NUM_GPU_NODES);
    printHohooSchedule(sched);

    // ---- Create topology (no BFS routing) ----
    DynOcsTopology* topo = new DynOcsTopology(sched, SLICE_ACTIVE_PS, RECONFIG_DELAY_PS);

    std::cout << "Direct-Only Topology:\n"
              << "  Slices: " << topo->num_slices() << "\n"
              << "  Slice active: " << (double)topo->slice_active() / 1e6 << " us\n"
              << "  Reconfig gap: " << (double)topo->reconfig_time() / 1e6 << " us\n"
              << "  Cycle time: " << (double)topo->cycle_time() / 1e6 << " us\n"
              << "  OCS link delay: " << (double)OCS_LINK_DELAY_PS / 1e3 << " ns\n\n";

    // ---- Print direct connection table ----
    topo->printConnectionTable();

    // ---- Create OCS switch (8 EcsBuffers) ----
    OcsSwitch* ocs = new OcsSwitch(topo);

    // ---- Create 8 GpuNodes ----
    std::array<GpuNode*, 8> gpus;
    auto makePeers = [](uint8_t self) {
        std::vector<uint8_t> v;
        for (uint8_t i = 1; i <= 8; i++) if (i != self) v.push_back(i);
        return v;
    };
    for (int i = 0; i < 8; i++) {
        gpus[i] = new GpuNode((uint8_t)(i + 1), makePeers(i + 1));
    }

    // ---- Create PacketFlows ----
    std::array<PacketFlow*, 8> flows;
    for (int i = 0; i < 8; i++) {
        flows[i] = new PacketFlow(nullptr);
        flows[i]->set_flowid((flowid_t)(i + 1));
        gpus[i]->setFlow(flows[i]);
    }

    // ---- Build routes ----
    std::array<std::array<Route*, 8>, 8> routes{};
    for (int src = 1; src <= 8; src++) {
        for (int dst = 1; dst <= 8; dst++) {
            if (src == dst) continue;
            routes[src-1][dst-1] = buildRoute(src, dst, ocs, gpus);
            gpus[src-1]->setRoute((uint8_t)dst, routes[src-1][dst-1]);
        }
    }

    // ---- Print route info ----
    std::cout << "Route structure (OCS Direct-Only):\n";
    for (int src = 1; src <= 8; src++) {
        for (int dst = 1; dst <= 8; dst++) {
            if (src == dst) continue;
            int directSlice = topo->get_slice_for_pair(src, dst);
            std::cout << "  GPU" << src << " -> GPU" << dst
                      << ": direct in slice " << directSlice << "\n";
        }
    }
    std::cout << "\n";

    // ---- Print topology ----
    std::cout << "OCS Direct-Only Topology:\n"
              << "  8 GPUs, each with one EcsBuffer\n"
              << "  OCS time-division: 7 slices per cycle\n"
              << "  Each slice: one perfect matching (4 bidirectional pairs)\n"
              << "  Routing: direct-only (wait for direct connection)\n"
              << "  No multi-hop, no relay\n"
              << "  Physical model: 1 OCS port per node, 200Gbps\n\n";

    // ---- Run simulation ----
    std::cout << "Running simulation...\n\n";
    while (EventList::doNextEvent()) {}

    // ---- Print stats ----
    ocs->printStats();

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
