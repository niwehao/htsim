/*
 * gpu_scheduler.h — Static flow scheduling for all-to-all OCS
 *
 * Exploits the Circle Method property: every pair (A,B) has exactly
 * one direct connection slot per cycle. For all-to-all, the optimal
 * schedule is to send each flow at its direct connection slot.
 *
 * Result: makespan = N-1 slots = 1 cycle (provably optimal).
 * Each GPU sends exactly 1 flow per slot, no relay needed.
 */

#pragma once
#include "hoho_routing.h"

// ================================================================
//  Data structures
// ================================================================

struct ScheduledFlow {
    uint16_t target;
    int      delaySlots;   // delay from phase start (in time slots)
};

// Per-GPU schedule for one phase: indexed by starting slice
// phases[{layer,phase}][starting_slice] -> ordered list of sends
struct PerGpuFlowSchedule {
    std::map<std::pair<int,int>,
             std::vector<std::vector<ScheduledFlow>>> phases;
};

// ================================================================
//  All-to-all direct connection scheduler
//
//  For each (src, dst) pair:
//    directSlice = topo->get_slice_for_pair(src, dst)
//    delay = (directSlice - baseSlice + S) % S
//
//  This guarantees:
//    - Each GPU's ocs_tx is used at most once per slot (no conflicts)
//    - Each flow uses direct connection (1 hop, 1 slot delivery)
//    - Makespan = N-1 slots for any starting slice
// ================================================================

inline std::vector<PerGpuFlowSchedule> computeGlobalFlowSchedule(
    const DynOcsTopology* topo,
    int totalLayers)
{
    int N = topo->num_ports();
    int S = topo->num_slices();

    std::vector<PerGpuFlowSchedule> schedules(N);   // 0-indexed

    std::cout << "Computing GPU flow schedules (K=" << NUM_ACTIVE_EXPERTS
              << ", " << totalLayers << " layers x 2 phases x "
              << S << " starting slices)...\n";

    for (int layer = 0; layer < totalLayers; layer++) {
        for (int phase = 0; phase < 2; phase++) {
            auto phaseKey = std::make_pair(layer, phase);

            // Get actual targets for each GPU in this phase
            auto assignments = computePhaseAssignments(layer, phase);

            // Init per-GPU storage
            for (int g = 0; g < N; g++)
                schedules[g].phases[phaseKey].resize(S);

            int worstMakespan = 0;
            int totalFlows = 0;

            for (int base = 0; base < S; base++) {
                int makespan = 0;

                for (int src = 1; src <= N; src++) {
                    int gpuIdx = src - 1;
                    auto& entries = schedules[gpuIdx].phases[phaseKey][base];
                    auto& targets = assignments[src].send_targets;

                    for (uint16_t dst : targets) {
                        int directSlice = topo->get_slice_for_pair(src, (int)dst);
                        int delay = (directSlice - base + S) % S;
                        entries.push_back({dst, delay});
                        makespan = std::max(makespan, delay + 1);
                    }

                    if (base == 0) totalFlows += (int)targets.size();

                    // Sort by delay for ordered execution
                    std::sort(entries.begin(), entries.end(),
                        [](const ScheduledFlow& a, const ScheduledFlow& b) {
                            return a.delaySlots < b.delaySlots;
                        });
                }

                worstMakespan = std::max(worstMakespan, makespan);
            }

            std::cout << "  L" << layer << " "
                      << (phase == 0 ? "DISPATCH" : "COMBINE")
                      << ": " << totalFlows << " flows, makespan="
                      << worstMakespan << " slots ("
                      << std::fixed << std::setprecision(1)
                      << (double)worstMakespan * SLICE_TOTAL_PS / 1e6
                      << " us = "
                      << std::setprecision(2)
                      << (double)worstMakespan * SLICE_TOTAL_PS / 1e9
                      << " ms)\n";

            // Print sample schedule for first phase
            if (layer == 0 && phase == 0) {
                int sampleSlice = 0;
                std::cout << "  Sample schedule (L0 DISPATCH, startSlice=0):\n";
                int printGpus = std::min(N, 4);
                for (int g = 0; g < printGpus; g++) {
                    auto& ent = schedules[g].phases[phaseKey][sampleSlice];
                    std::cout << "    GPU" << (g + 1) << ": "
                              << ent.size() << " flows, first 8:";
                    int limit = std::min((int)ent.size(), 8);
                    for (int i = 0; i < limit; i++)
                        std::cout << " ->" << ent[i].target
                                  << "@d" << ent[i].delaySlots;
                    if ((int)ent.size() > 8) std::cout << " ...";
                    std::cout << "\n";
                }
                if (N > 4)
                    std::cout << "    ... (" << N << " GPUs total)\n";
            }
        }
    }

    std::cout << "GPU flow schedules computed.\n\n";
    return schedules;
}
