/*
 * main.cpp  —  OCS Synchronized Scheduling Simulation
 *
 * N GPUs connected via 1 OCS switch, N-1 time slots per cycle.
 * All-to-all communication with pre-computed optimal send times.
 * GPU scheduler assigns each flow to its direct connection slot.
 * EcsBuffer handles actual packet routing via BFS.
 */

#include "gpu_node.h"
#include <iostream>
#include <vector>
#include <chrono>

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

    std::cout << "=== OCS Synchronized Scheduling Simulation ===\n"
              << "  GPUs=" << N
              << " All-to-all (" << NUM_ACTIVE_EXPERTS << " targets/GPU)"
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

    // ---- Generate OCS schedule (Circle Method) ----
    HohooSchedule sched = computeHohooSchedule(N);
    printHohooSchedule(sched);

    // ---- Create topology + BFS routing table ----
    auto t0 = std::chrono::steady_clock::now();
    DynOcsTopology* topo = new DynOcsTopology(sched, SLICE_ACTIVE_PS, RECONFIG_DELAY_PS);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[TIMER] BFS routing table: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    std::cout << "Topology:\n"
              << "  Ports: " << topo->num_ports() << "\n"
              << "  Slices: " << topo->num_slices() << "\n"
              << "  Slice active: " << (double)topo->slice_active() / 1e6 << " us\n"
              << "  Cycle time: " << (double)topo->cycle_time() / 1e6 << " us"
              << " (" << (double)topo->cycle_time() / 1e9 << " ms)\n\n";

    topo->printRoutingTable();

    // ---- Compute GPU flow schedule (all-to-all direct) ----
    auto t2 = std::chrono::steady_clock::now();
    auto flowSchedules = computeGlobalFlowSchedule(topo, (int)TOTAL_LAYERS);
    auto t3 = std::chrono::steady_clock::now();
    std::cout << "[TIMER] Flow scheduling: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms\n";

    // ---- Create OCS switch ----
    OcsSwitch* ocs = new OcsSwitch(topo);

    // ---- Create N GpuNodes ----
    std::vector<GpuNode*> gpus(N);
    for (int i = 0; i < N; i++)
        gpus[i] = new GpuNode((uint16_t)(i + 1), topo);

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

    // ---- Pre-compute phase assignments (all-to-all) ----
    std::cout << "Setting up all-to-all phase assignments ("
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

    // ---- Assign flow schedules to GPUs ----
    std::cout << "Assigning flow schedules to " << N << " GPUs...\n";
    for (int i = 0; i < N; i++)
        gpus[i]->setFlowSchedule(flowSchedules[i]);

    std::cout << "\nExpected per-phase time: "
              << std::fixed << std::setprecision(2)
              << (double)CYCLE_PS / 1e9 << " ms (1 cycle = "
              << NUM_SLOTS << " slots)\n"
              << "Expected total time: ~"
              << (double)(TOTAL_LAYERS * 2 * CYCLE_PS
                          + (TOTAL_LAYERS * 2 - 1) * INTERPHASE_GAP_PS) / 1e9
              << " ms\n\n";

    // ---- Run simulation ----
    std::cout << "Running simulation...\n\n";
    while (EventList::doNextEvent()) {}

    // ---- Print stats ----
    ocs->printStats();

    std::cout << "\nSimulation ended at t=" << timeAsMs(EventList::now()) << " ms\n";
    return 0;
}
