/*
 * gpu_scheduler.h — Static flow scheduling for OCS
 *
 * Pre-computes optimal send times for every flow in each MoE phase.
 * Goal: minimize makespan (time until all flows complete).
 * Constraint: ocs_tx[node][slot] serves at most 1 flow per slot.
 * Algorithm: Greedy LPT (Longest Processing Time first).
 *
 * The schedule is computed for every possible starting slice so that
 * at runtime the GPU can look up the correct plan regardless of when
 * the phase actually begins.
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
// phases[{layer,phase}][starting_slice] → ordered list of sends
struct PerGpuFlowSchedule {
    std::map<std::pair<int,int>,
             std::vector<std::vector<ScheduledFlow>>> phases;
};

// ================================================================
//  BFS path tracing
// ================================================================

struct PathHop {
    int node;       // node whose ocs_tx is used
    int relSlot;    // relative slot offset from flow injection
};

struct PathInfo {
    std::vector<PathHop> hops;
    int deliverySlots;          // total slots src → dst
};

// Trace the BFS-optimal path from src to dst starting at startSlice.
// Returns ocs_tx usage along the path and total delivery time.
inline PathInfo traceBfsPath(const DynOcsTopology* topo,
                             int src, int dst, int startSlice)
{
    PathInfo info;
    int node  = src;
    int slice = startSlice;
    int S     = topo->num_slices();
    int rel   = 0;
    int limit = S * 3;          // safety bound

    while (node != dst && rel < limit) {
        if (topo->getAction(node, dst, slice) == DynOcsTopology::FORWARD) {
            info.hops.push_back({node, rel});
            node = topo->getPeer(node, slice);
        }
        slice = (slice + 1) % S;
        rel++;
    }

    if (node != dst) {
        std::cerr << "FATAL: BFS path " << src << "->" << dst
                  << " from slice " << startSlice << " did not converge\n";
        abort();
    }
    info.deliverySlots = rel;
    return info;
}

// ================================================================
//  Global schedule computation
// ================================================================

inline std::vector<PerGpuFlowSchedule> computeGlobalFlowSchedule(
    const DynOcsTopology* topo,
    int totalLayers)
{
    int N = topo->num_ports();
    int S = topo->num_slices();

    std::vector<PerGpuFlowSchedule> schedules(N);   // 0-indexed

    std::cout << "Computing GPU flow schedules ("
              << totalLayers << " layers x 2 phases x "
              << S << " starting slices)...\n";

    for (int layer = 0; layer < totalLayers; layer++) {
        for (int phase = 0; phase < 2; phase++) {
            auto phaseKey = std::make_pair(layer, phase);
            auto assignments = computePhaseAssignments(layer, phase);

            // --- Collect all flows for this phase ---
            struct Flow { uint16_t src, dst; };
            std::vector<Flow> allFlows;
            for (uint16_t gpu = 1; gpu <= (uint16_t)N; gpu++)
                for (auto dst : assignments[gpu].send_targets)
                    allFlows.push_back({gpu, dst});

            size_t F = allFlows.size();

            // --- Pre-compute paths for every (flow, startSlice) ---
            // pathCache[flowIdx][slice] → PathInfo
            std::vector<std::vector<PathInfo>> pathCache(
                F, std::vector<PathInfo>(S));

            for (size_t fi = 0; fi < F; fi++)
                for (int s = 0; s < S; s++)
                    pathCache[fi][s] = traceBfsPath(
                        topo, allFlows[fi].src, allFlows[fi].dst, s);

            // --- Init per-GPU storage ---
            for (int g = 0; g < N; g++)
                schedules[g].phases[phaseKey].resize(S);

            // --- Solve for each starting slice ---
            int worstMakespan = 0;

            for (int base = 0; base < S; base++) {

                // Build flow list sorted by delivery time (LPT)
                struct IdxFlow { size_t idx; int delivery; };
                std::vector<IdxFlow> sorted;
                sorted.reserve(F);
                for (size_t fi = 0; fi < F; fi++)
                    sorted.push_back({fi,
                        pathCache[fi][base].deliverySlots});

                std::sort(sorted.begin(), sorted.end(),
                    [](const IdxFlow& a, const IdxFlow& b) {
                        return a.delivery > b.delivery;
                    });

                // Resource tracking: ocsTxUsed[node] → set of abs slots
                std::map<int, std::set<int>> ocsTxUsed;
                int makespan = 0;
                int maxSearch = S * 3;

                for (auto& sf : sorted) {
                    auto& flow = allFlows[sf.idx];
                    bool placed = false;

                    for (int delay = 0; delay <= maxSearch; delay++) {
                        int actualSlice = (base + delay) % S;
                        const PathInfo& path =
                            pathCache[sf.idx][actualSlice];

                        // Check ocs_tx conflicts
                        bool conflict = false;
                        for (auto& h : path.hops) {
                            if (ocsTxUsed[h.node].count(
                                    delay + h.relSlot)) {
                                conflict = true;
                                break;
                            }
                        }

                        if (!conflict) {
                            // Mark resources
                            for (auto& h : path.hops)
                                ocsTxUsed[h.node].insert(
                                    delay + h.relSlot);

                            int gpuIdx = (int)flow.src - 1;
                            schedules[gpuIdx]
                                .phases[phaseKey][base]
                                .push_back({flow.dst, delay});

                            makespan = std::max(makespan,
                                delay + path.deliverySlots);
                            placed = true;
                            break;
                        }
                    }

                    if (!placed) {
                        std::cerr << "FATAL: cannot schedule flow "
                                  << flow.src << "->" << flow.dst
                                  << " base=" << base << "\n";
                        abort();
                    }
                }

                worstMakespan = std::max(worstMakespan, makespan);
            }

            std::cout << "  L" << layer << " "
                      << (phase == 0 ? "DISPATCH" : "COMBINE")
                      << ": " << F << " flows, worst-case makespan="
                      << worstMakespan << " slots ("
                      << std::fixed << std::setprecision(1)
                      << (double)worstMakespan * SLICE_TOTAL_PS / 1e6
                      << " us)\n";

            // Print sample schedule for first phase
            if (layer == 0 && phase == 0) {
                int sampleSlice = 0;
                std::cout << "  Sample schedule (L0 DISPATCH, "
                             "startSlice=0):\n";
                int printGpus = std::min(N, 8);
                for (int g = 0; g < printGpus; g++) {
                    auto& entries =
                        schedules[g].phases[phaseKey][sampleSlice];
                    std::cout << "    GPU" << (g + 1) << ":";
                    for (auto& e : entries)
                        std::cout << " ->" << e.target
                                  << "@d" << e.delaySlots;
                    std::cout << "\n";
                }
                if (N > 8)
                    std::cout << "    ... (" << N
                              << " GPUs total)\n";
            }
        }
    }

    std::cout << "GPU flow schedules computed.\n\n";
    return schedules;
}
