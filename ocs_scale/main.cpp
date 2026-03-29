/*
 * main.cpp  —  Scalable OCS + BFS multi-hop routing (max 5 hops)
 *
 * N GPUs connected via 1 OCS switch, N-1 time slots per cycle.
 * Each phase: GPU sends to NUM_ACTIVE_EXPERTS targets (MoE top-K).
 * Route: [EcsBuffer@src, rxQ_dst, AppSink_dst]
 */

#include "gpu_node.h"
#include <iostream>
#include <vector>

Route* buildRoute(int srcGpu, int dstGpu,
                  OcsSwitch* ocs,
                  std::vector<GpuNode*>& gpus)
{
    Route* route = new Route();
    route->push_back(ocs->getBuffer(srcGpu));
    route->push_back(gpus[dstGpu - 1]->rxQueue());
    route->push_back(gpus[dstGpu - 1]->appSink());
    return route;
}

int main() {
    int N = (int)NUM_GPU_NODES;

    EventList::setEndtime(timeFromSec(10000));

    std::cout << "=== MOE Scalable Simulation (OCS + BFS " << DynOcsTopology::MAX_HOPS << "-hop) ===\n"
              << "  GPUs=" << N
              << " Experts=" << N
              << " ActiveExperts=" << NUM_ACTIVE_EXPERTS
              << " Layers=" << TOTAL_LAYERS
              << " Frags=" << TOTAL_FRAGMENTS << "x" << (FRAGMENT_PAYLOAD_SIZE/1024) << "KB"
              << " Link=" << PROT_RATE_Gbps << "Gbps\n"
              << "  SliceActive=" << std::fixed << std::setprecision(1)
              << (double)SLICE_ACTIVE_PS / 1e6 << "us"
              << " Reconfig=" << (double)RECONFIG_DELAY_PS / 1e6 << "us"
              << " SliceTotal=" << (double)SLICE_TOTAL_PS / 1e6 << "us"
              << " Cycle=" << (double)CYCLE_PS / 1e6 << "us"
              << " (" << (double)CYCLE_PS / 1e9 << "ms)"
              << " Timeout=" << timeAsMs(TIMEOUT_PS) << "ms\n\n";

    // ---- Generate schedule (Circle Method) ----
    HohooSchedule sched = computeHohooSchedule(N);
    printHohooSchedule(sched);

    // ---- Create topology + BFS routing table ----
    DynOcsTopology* topo = new DynOcsTopology(sched, SLICE_ACTIVE_PS, RECONFIG_DELAY_PS);

    std::cout << "Topology:\n"
              << "  Ports: " << topo->num_ports() << "\n"
              << "  Slices: " << topo->num_slices() << "\n"
              << "  Slice active: " << (double)topo->slice_active() / 1e6 << " us\n"
              << "  Cycle time: " << (double)topo->cycle_time() / 1e6 << " us"
              << " (" << (double)topo->cycle_time() / 1e9 << " ms)\n\n";

    topo->printRoutingTable();

    // ---- Create OCS switch ----
    OcsSwitch* ocs = new OcsSwitch(topo);

    // ---- Create N GpuNodes ----
    std::vector<GpuNode*> gpus(N);
    for (int i = 0; i < N; i++)
        gpus[i] = new GpuNode((uint16_t)(i + 1));

    // ---- Create PacketFlows ----
    std::vector<PacketFlow*> flows(N);
    for (int i = 0; i < N; i++) {
        flows[i] = new PacketFlow(nullptr);
        flows[i]->set_flowid((flowid_t)(i + 1));
        gpus[i]->setFlow(flows[i]);
    }

    // ---- Build routes for all pairs ----
    std::cout << "Building " << (N * (N-1)) << " routes...\n";
    for (int src = 1; src <= N; src++) {
        for (int dst = 1; dst <= N; dst++) {
            if (src == dst) continue;
            Route* route = buildRoute(src, dst, ocs, gpus);
            gpus[src-1]->setRoute((uint16_t)dst, route);
        }
    }

    // ---- Pre-compute phase assignments (MoE target selection) ----
    std::cout << "Pre-computing MoE target assignments ("
              << TOTAL_LAYERS << " layers x 2 phases x " << N << " GPUs x "
              << NUM_ACTIVE_EXPERTS << " targets)...\n";

    for (int layer = 0; layer < (int)TOTAL_LAYERS; layer++) {
        for (int phase = 0; phase < 2; phase++) {
            auto assignments = computePhaseAssignments(layer, phase);
            for (int gpu = 1; gpu <= N; gpu++) {
                gpus[gpu-1]->setPhaseConfig(layer, phase,
                    assignments[gpu].send_targets,
                    assignments[gpu].recv_from);
            }
        }
    }

    // Print sample assignments
    std::cout << "Sample MoE assignments (L0 DISPATCH):\n";
    auto sample = computePhaseAssignments(0, 0);
    int printLimit = std::min(N, 8);
    for (int gpu = 1; gpu <= printLimit; gpu++) {
        std::cout << "  GPU" << gpu << " sends to:";
        for (auto t : sample[gpu].send_targets) std::cout << " " << t;
        std::cout << "  | recv from:";
        for (auto r : sample[gpu].recv_from) std::cout << " " << r;
        std::cout << "\n";
    }
    if (N > 8) std::cout << "  ... (" << N << " GPUs total)\n";
    std::cout << "\n";

    // ---- Run simulation ----
    std::cout << "Running simulation...\n\n";
    while (EventList::doNextEvent()) {}

    // ---- Print stats ----
    ocs->printStats();

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
