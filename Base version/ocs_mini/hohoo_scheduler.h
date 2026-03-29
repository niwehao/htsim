/*
 * hohoo_scheduler.h  —  HOHOO (Hierarchical Optical Hybrid Overlay and Orchestration)
 *
 * HOHOO 算法用于 OCS 电路调度:
 *   - 将 All-to-All 通信分解为多个时隙
 *   - 每个时隙配置一个完美匹配 (perfect matching)
 *   - 8 个 GPU 需要 7 个时隙完成全部通信
 *   - 每个时隙中, 4 对 GPU 同时双向通信
 *
 * 调度算法: Round-Robin Tournament (Circle Method)
 *   - 固定 GPU8 在顶部
 *   - 将 GPU1-GPU7 排成环, 每轮旋转一个位置
 *   - 每轮产生 4 对匹配
 *
 * 时隙调度:
 *   Round 0: (8,1) (2,7) (3,6) (4,5)
 *   Round 1: (8,7) (1,6) (2,5) (3,4)
 *   Round 2: (8,6) (7,5) (1,4) (2,3)
 *   Round 3: (8,5) (6,4) (7,3) (1,2)
 *   Round 4: (8,4) (5,3) (6,2) (7,1)
 *   Round 5: (8,3) (4,2) (5,1) (6,7)
 *   Round 6: (8,2) (3,1) (4,7) (5,6)
 *
 * 每对匹配是双向的: (A,B) 表示 A→B 和 B→A 同时通信
 */

#pragma once

#include <vector>
#include <map>
#include <iostream>
#include <cassert>

struct HohooSchedule {
    // schedule[gpu_id] = 按时隙顺序排列的目标 GPU 列表
    // 例: schedule[1] = {8, 3, 5, 7, 2, 4, 6} 表示
    //   时隙 0: GPU1 与 GPU8 通信
    //   时隙 1: GPU1 与 GPU3 通信
    //   ...
    std::map<int, std::vector<int>> schedule;

    // matchings[round] = 该时隙的所有匹配对 {(a,b), ...}
    std::vector<std::vector<std::pair<int,int>>> matchings;
};

// ================================================================
//  HOHOO 调度算法
//
//  使用 Circle Method 生成 Round-Robin Tournament:
//    N=8 个参与者, N-1=7 轮, 每轮 N/2=4 对
// ================================================================

inline HohooSchedule computeHohooSchedule(int numGpus) {
    assert(numGpus == 8 && "HOHOO scheduler currently supports 8 GPUs");

    HohooSchedule sched;
    int N = numGpus;

    // 初始化环: [1, 2, 3, 4, 5, 6, 7]
    std::vector<int> circle;
    for (int i = 1; i < N; i++)
        circle.push_back(i);

    // N-1 = 7 轮
    for (int round = 0; round < N - 1; round++) {
        std::vector<std::pair<int,int>> pairs;

        // GPU N (固定在顶部) 与 circle[0] 匹配
        pairs.push_back({N, circle[0]});

        // circle[i] 与 circle[N-1-i] 匹配 (i = 1..N/2-1)
        for (int i = 1; i < N / 2; i++) {
            int a = circle[i];
            int b = circle[N - 1 - i];
            pairs.push_back({a, b});
        }

        sched.matchings.push_back(pairs);

        // 旋转环: 将最后一个元素移到前面
        int last = circle.back();
        circle.pop_back();
        circle.insert(circle.begin(), last);
    }

    // 从 matchings 生成每个 GPU 的时隙调度
    for (int gpu = 1; gpu <= N; gpu++) {
        sched.schedule[gpu] = std::vector<int>();
    }

    for (int round = 0; round < (int)sched.matchings.size(); round++) {
        for (auto& [a, b] : sched.matchings[round]) {
            sched.schedule[a].push_back(b);
            sched.schedule[b].push_back(a);
        }
    }

    return sched;
}

// 打印调度信息
inline void printHohooSchedule(const HohooSchedule& sched) {
    std::cout << "HOHOO Schedule (Round-Robin Tournament):\n";
    for (int round = 0; round < (int)sched.matchings.size(); round++) {
        std::cout << "  Slot " << round << ": ";
        for (auto& [a, b] : sched.matchings[round]) {
            std::cout << "(" << a << "," << b << ") ";
        }
        std::cout << "\n";
    }

    std::cout << "\nPer-GPU schedule:\n";
    for (auto& [gpu, targets] : sched.schedule) {
        std::cout << "  GPU" << gpu << ": ";
        for (int t : targets) std::cout << t << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}
