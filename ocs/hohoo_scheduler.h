/*
 * hohoo_scheduler.h  —  Generalized Circle Method scheduler
 *
 * Round-Robin Tournament (Circle Method) for any even N:
 *   - N participants, N-1 rounds, N/2 pairs per round
 *   - Fix participant N at top, rotate 1..N-1 in a circle
 *   - Generates all perfect matchings for complete graph K_N
 */

#pragma once

#include <vector>
#include <map>
#include <iostream>
#include <cassert>

struct HohooSchedule {
    // schedule[gpu_id] = per-slot target list
    std::map<int, std::vector<int>> schedule;
    // matchings[round] = pairs for that slot
    std::vector<std::vector<std::pair<int,int>>> matchings;
};

// Circle Method: works for any even N
inline HohooSchedule computeHohooSchedule(int numGpus) {
    assert(numGpus >= 2 && numGpus % 2 == 0 && "Circle Method requires even N >= 2");

    HohooSchedule sched;
    int N = numGpus;

    // Circle: [1, 2, ..., N-1]
    std::vector<int> circle;
    for (int i = 1; i < N; i++)
        circle.push_back(i);

    // N-1 rounds
    for (int round = 0; round < N - 1; round++) {
        std::vector<std::pair<int,int>> pairs;

        // GPU N (fixed) matches circle[0]
        pairs.push_back({N, circle[0]});

        // circle[i] matches circle[N-2-i] (symmetric pairing)
        for (int i = 1; i < N / 2; i++) {
            int a = circle[i];
            int b = circle[N - 1 - i];
            pairs.push_back({a, b});
        }

        sched.matchings.push_back(pairs);

        // Rotate: last element moves to front
        int last = circle.back();
        circle.pop_back();
        circle.insert(circle.begin(), last);
    }

    // Build per-GPU schedule
    for (int gpu = 1; gpu <= N; gpu++)
        sched.schedule[gpu] = std::vector<int>();

    for (int round = 0; round < (int)sched.matchings.size(); round++) {
        for (auto& [a, b] : sched.matchings[round]) {
            sched.schedule[a].push_back(b);
            sched.schedule[b].push_back(a);
        }
    }

    return sched;
}

inline void printHohooSchedule(const HohooSchedule& sched) {
    int numSlots = (int)sched.matchings.size();
    int numGpus = numSlots + 1;

    std::cout << "OCS Schedule (Circle Method, N=" << numGpus
              << ", " << numSlots << " slots):\n";

    // Print first few and last few slots
    int printLimit = std::min(numSlots, 8);
    for (int round = 0; round < printLimit; round++) {
        std::cout << "  Slot " << round << ": ";
        int pairLimit = std::min((int)sched.matchings[round].size(), 8);
        for (int i = 0; i < pairLimit; i++) {
            auto& [a, b] = sched.matchings[round][i];
            std::cout << "(" << a << "," << b << ") ";
        }
        if ((int)sched.matchings[round].size() > 8)
            std::cout << "... (" << sched.matchings[round].size() << " pairs)";
        std::cout << "\n";
    }
    if (numSlots > 8)
        std::cout << "  ... (" << numSlots << " slots total)\n";

    // Print first few GPU schedules
    std::cout << "\nSample per-GPU schedule:\n";
    int gpuPrintLimit = std::min(numGpus, 4);
    for (int gpu = 1; gpu <= gpuPrintLimit; gpu++) {
        auto it = sched.schedule.find(gpu);
        if (it == sched.schedule.end()) continue;
        std::cout << "  GPU" << gpu << ": ";
        int slotPrintLimit = std::min((int)it->second.size(), 10);
        for (int i = 0; i < slotPrintLimit; i++)
            std::cout << it->second[i] << " ";
        if ((int)it->second.size() > 10)
            std::cout << "... (" << it->second.size() << " slots)";
        std::cout << "\n";
    }
    std::cout << "\n";
}
