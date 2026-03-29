/*
 * sync_scheduler.h  —  Dragonfly 同步预调度算法 (三遍扫描)
 *
 * 核心思想:
 *   在仿真开始前, 离线计算每个数据包的最优发送时刻.
 *   将网络中的每个队列视为一个"资源", 每个资源同一时刻只能服务一个包.
 *
 * 三遍扫描算法 (比 htsim_syn 的一遍贪心更优):
 *   第一遍 (正向): 不更新 resFree, 只计算各跳的最早开始时间
 *   第二遍 (反向): 找瓶颈跳, 反推 sendTime — 使包"刚好"在瓶颈资源空闲时到达
 *   第三遍 (正向): 用最终 sendTime 更新 resFree
 *
 * 效果:
 *   - 包不会在中间队列等待 -> 无 BufferGate 2ms 惩罚
 *   - Just-In-Time 到达: 减少中间队列占用
 *   - 无重传
 *   - 理论最优完成时间
 *
 * Dragonfly 资源模型:
 *   - GPU txQueue:  8 个, 各 200 Gbps
 *   - Tofino ingressQ[port]: 每端口 200 Gbps (4 个交换机 x 4 端口 = 16 个)
 *   - Tofino fabricQ:  每 Tofino 800 Gbps (4 个)
 *   - Tofino egressQ[port]: 每端口 200 Gbps (16 个)
 *   - GPU rxQueue:  8 个, 各 200 Gbps
 */

#pragma once

#include "constants.h"

#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <iomanip>

// ================================================================
//  调度条目: 一个 GPU 在某阶段的一次定时发送
// ================================================================
struct ScheduleEntry {
    simtime_picosec sendTime;   // 相对于阶段开始的发送时刻 (ps)
    uint8_t         dstGpu;     // 目标 GPU (1-8)
    uint16_t        fragId;     // fragment ID (0..TOTAL_FRAGMENTS-1)
};

// ================================================================
//  SyncScheduler — 离线三遍扫描调度器
// ================================================================
class SyncScheduler {
public:
    // ---- 资源 ID 编码 (唯一标识网络中的每个队列) ----
    //
    // Dragonfly: 4 交换机, ports 1-16
    //   txQueue:   1-8   (GPU ID)
    //   rxQueue:   11-18 (10 + GPU ID)
    //   ingressQ:  101-116 (100 + port)
    //   fabricQ:   200-203 (200 + switch_idx)
    //   egressQ:   301-316 (300 + port)
    static int rTx(int gpu)       { return gpu; }            // 1-8
    static int rRx(int gpu)       { return 10 + gpu; }       // 11-18
    static int rIngress(int port) { return 100 + port; }     // 101-116
    static int rFabric(int tof)   { return 200 + tof; }      // 200-203
    static int rEgress(int port)  { return 300 + port; }     // 301-316

    // ---- 每字节的串行化时间 (ps) ----
    static simtime_picosec psPerByteLinkSpeed() {
        return (simtime_picosec)((1.0e12 * 8) / (PROT_RATE_Gbps * 1.0e9));
    }
    static simtime_picosec psPerByteFabricSpeed() {
        return (simtime_picosec)((1.0e12 * 8) / (PROT_RATE_Gbps * PORT_NUM * 1.0e9));
    }

    // ---- 路由跳: 一个资源 + 在该资源上的占用时间 ----
    struct RouteHop {
        int             resourceId;
        simtime_picosec drainTime;
    };

    // ---- 计算 srcGpu -> dstGpu 的抽象路由 (资源序列) ----
    //
    // Dragonfly minimal routing:
    //   同交换机: txQ -> [ingress, fabric, egress] -> rxQ                  (1 switch, 5 hops)
    //   同 Group: txQ -> [ing,fab,egr] -> [ing,fab,egr] -> rxQ            (2 switches, 8 hops)
    //   跨 Group: txQ -> [ing,fab,egr] -> ... -> [ing,fab,egr] -> rxQ     (2-3 switches, 8-11 hops)
    static std::vector<RouteHop> abstractRoute(int srcGpu, int dstGpu, int pktSize) {
        std::vector<RouteHop> hops;
        simtime_picosec dtLink   = (simtime_picosec)pktSize * psPerByteLinkSpeed();
        simtime_picosec dtFabric = (simtime_picosec)pktSize * psPerByteFabricSpeed();

        // GPU 发送端 txQueue
        hops.push_back({rTx(srcGpu), dtLink});

        // 追踪 Tofino 链 (与 dragonfly/main.cpp::buildRoute 相同逻辑)
        int curTof = gpuToTofinoIdx(srcGpu);
        int inPort = gpuToPort(srcGpu);

        for (int h = 0; h < 10; h++) {
            auto rt = resolvedRoutingTable(curTof);
            int outPort = rt.at(dstGpu);

            hops.push_back({rIngress(inPort),  dtLink});
            hops.push_back({rFabric(curTof),   dtFabric});
            hops.push_back({rEgress(outPort),  dtLink});

            if (isGpuPort(outPort) && portToGpu(outPort) == dstGpu) {
                // 到达目标 GPU
                hops.push_back({rRx(dstGpu), dtLink});
                return hops;
            }

            // 互联端口 -> 下一个 Tofino
            int nextIn = portMapping(outPort);
            curTof = portToTofinoIdx(nextIn);
            inPort = nextIn;
        }

        assert(false && "abstractRoute: exceeded max hops");
        return hops;
    }

    // ================================================================
    //  computePhaseSchedule — 三遍扫描计算一个阶段的完整调度
    //
    //  调度顺序: fragment -> src -> dst (轮询)
    //  对每个包:
    //    第一遍: 正向扫描, 找每跳最早开始时间 (不更新 resFree)
    //    第二遍: 反向推导最优 sendTime (Just-In-Time 到达瓶颈)
    //    第三遍: 用最终 sendTime 正向更新 resFree
    //
    //  返回: gpuId -> 按 sendTime 排序的 ScheduleEntry 列表
    // ================================================================
    static std::map<int, std::vector<ScheduleEntry>>
    computePhaseSchedule(int pktSize, simtime_picosec* outDuration = nullptr) {
        // 资源时间线: resource_id -> 该资源的下一个空闲时刻
        std::map<int, simtime_picosec> resFree;
        std::map<int, std::vector<ScheduleEntry>> schedule;
        simtime_picosec maxFinish = 0;

        for (int frag = 0; frag < (int)TOTAL_FRAGMENTS; frag++) {
            for (int src = 1; src <= (int)NUM_GPU_NODES; src++) {
                for (int dst = 1; dst <= (int)NUM_GPU_NODES; dst++) {
                    if (src == dst) continue;

                    auto route = abstractRoute(src, dst, pktSize);

                    // ---- 第一遍: 正向扫描, 找每跳的最早开始时间 ----
                    // 暂不更新 resFree, 只计算各跳 start
                    std::vector<simtime_picosec> hopStart(route.size());
                    simtime_picosec prevFinish = 0;

                    for (int i = 0; i < (int)route.size(); i++) {
                        auto& hop = route[i];
                        simtime_picosec start = std::max(resFree[hop.resourceId],
                                                          prevFinish);
                        simtime_picosec finish = start + hop.drainTime;
                        prevFinish = finish;
                        hopStart[i] = start;
                    }

                    // ---- 第二遍: 反向推导 sendTime ----
                    // 找到瓶颈跳 (等待时间最长的跳), 从它反推第一跳的最晚开始时间
                    // 使得包"刚好"在每个资源空闲时到达, 不提前占用中间队列
                    //
                    // 瓶颈跳 k 的 start 由 resFree[k] 决定 (而非 prevFinish)
                    // 从 k 反推: hop[0].start = hop[k].start - sum(drainTime[0..k-1])
                    simtime_picosec sendTime = hopStart[0];
                    for (int i = 1; i < (int)route.size(); i++) {
                        // 如果 hop[i] 被资源限制 (resFree > 来自前一跳的 prevFinish)
                        // 则从此跳反推更晚的 sendTime
                        simtime_picosec sumDrain = 0;
                        for (int j = 0; j < i; j++)
                            sumDrain += route[j].drainTime;
                        simtime_picosec candidateSend = hopStart[i] - sumDrain;
                        if (candidateSend > sendTime)
                            sendTime = candidateSend;
                    }

                    // ---- 第三遍: 用最终 sendTime 正向确定各跳时间, 更新 resFree ----
                    prevFinish = sendTime;  // hop[0] 不早于 sendTime
                    for (int i = 0; i < (int)route.size(); i++) {
                        auto& hop = route[i];
                        simtime_picosec start = std::max(resFree[hop.resourceId],
                                                          prevFinish);
                        simtime_picosec finish = start + hop.drainTime;
                        resFree[hop.resourceId] = finish;
                        prevFinish = finish;
                    }

                    schedule[src].push_back({sendTime, (uint8_t)dst, (uint16_t)frag});
                    maxFinish = std::max(maxFinish, prevFinish);
                }
            }
        }

        // 按发送时间排序
        for (auto& [gpu, entries] : schedule)
            std::sort(entries.begin(), entries.end(),
                      [](const ScheduleEntry& a, const ScheduleEntry& b) {
                          return a.sendTime < b.sendTime;
                      });

        if (outDuration) *outDuration = maxFinish;
        return schedule;
    }

    // ================================================================
    //  printScheduleStats — 打印调度统计
    // ================================================================
    static void printScheduleStats(
        const std::map<int, std::vector<ScheduleEntry>>& schedule,
        simtime_picosec duration)
    {
        std::cout << "=== Sync Schedule Stats (Dragonfly, 3-pass) ===\n"
                  << "  Phase duration: " << std::fixed << std::setprecision(3)
                  << (double)duration / 1e9 << " ms\n"
                  << "  Scheduling: fragment -> src -> dst (round-robin)\n"
                  << "  Algorithm: 3-pass (forward scan, backward inference, forward commit)\n";

        for (auto& [gpu, entries] : schedule) {
            simtime_picosec first = entries.front().sendTime;
            simtime_picosec last  = entries.back().sendTime;
            std::cout << "  GPU" << gpu << ": " << entries.size() << " sends, "
                      << "first=" << std::setprecision(3) << (double)first/1e9 << "ms "
                      << "last=" << std::setprecision(3) << (double)last/1e9 << "ms\n";
        }
        std::cout << "\n";
    }

    // ================================================================
    //  printFullSchedule — 打印完整调度表 (可限制条目数)
    // ================================================================
    static void printFullSchedule(
        const std::map<int, std::vector<ScheduleEntry>>& schedule,
        int maxEntries = 0)
    {
        std::cout << "=== Full Schedule (Dragonfly Sync) ===\n"
                  << "  GPU  sendTime(ms)  dst  fragId\n"
                  << "  ---  -----------  ---  ------\n";

        int count = 0;
        struct GlobalEntry {
            int gpu;
            simtime_picosec sendTime;
            uint8_t dst;
            uint16_t fragId;
        };
        std::vector<GlobalEntry> all;
        for (auto& [gpu, entries] : schedule)
            for (auto& e : entries)
                all.push_back({gpu, e.sendTime, e.dstGpu, e.fragId});

        std::sort(all.begin(), all.end(),
                  [](const GlobalEntry& a, const GlobalEntry& b) {
                      return a.sendTime < b.sendTime;
                  });

        for (int i = (int)all.size() - 1; i >= 0; i--) {
            auto& e = all[i];
            std::cout << "  GPU" << e.gpu
                      << "  t=" << std::fixed << std::setprecision(6)
                      << (double)e.sendTime / 1e9 << "ms"
                      << "  -> GPU" << (int)e.dst
                      << "  frag=" << e.fragId << "\n";
            if (maxEntries > 0 && ++count >= maxEntries) {
                std::cout << "  ... (" << all.size() - count << " more entries)\n";
                break;
            }
        }
        std::cout << "=== Total: " << all.size() << " entries ===\n\n";
    }

    // ================================================================
    //  验证调度: 检查所有资源在任意时刻只被一个包占用
    // ================================================================
    static bool validateSchedule(int pktSize) {
        auto route_cache = [&](int src, int dst) {
            return abstractRoute(src, dst, pktSize);
        };

        // 收集所有资源的 [start, finish) 区间
        struct Interval {
            simtime_picosec start, finish;
            int src, dst, frag;
        };
        std::map<int, std::vector<Interval>> resourceIntervals;

        std::map<int, simtime_picosec> resFree;
        for (int frag = 0; frag < (int)TOTAL_FRAGMENTS; frag++) {
            for (int src = 1; src <= (int)NUM_GPU_NODES; src++) {
                for (int dst = 1; dst <= (int)NUM_GPU_NODES; dst++) {
                    if (src == dst) continue;
                    auto route = route_cache(src, dst);

                    // 3-pass 重放
                    std::vector<simtime_picosec> hopStart(route.size());
                    simtime_picosec prevFinish = 0;
                    for (int i = 0; i < (int)route.size(); i++) {
                        auto& hop = route[i];
                        simtime_picosec start = std::max(resFree[hop.resourceId], prevFinish);
                        hopStart[i] = start;
                        prevFinish = start + hop.drainTime;
                    }
                    simtime_picosec sendTime = hopStart[0];
                    for (int i = 1; i < (int)route.size(); i++) {
                        simtime_picosec sumDrain = 0;
                        for (int j = 0; j < i; j++) sumDrain += route[j].drainTime;
                        simtime_picosec cand = hopStart[i] - sumDrain;
                        if (cand > sendTime) sendTime = cand;
                    }
                    prevFinish = sendTime;
                    for (int i = 0; i < (int)route.size(); i++) {
                        auto& hop = route[i];
                        simtime_picosec start = std::max(resFree[hop.resourceId], prevFinish);
                        simtime_picosec finish = start + hop.drainTime;
                        resFree[hop.resourceId] = finish;
                        prevFinish = finish;
                        resourceIntervals[hop.resourceId].push_back(
                            {start, finish, src, dst, frag});
                    }
                }
            }
        }

        // 检查每个资源的区间不重叠
        bool valid = true;
        for (auto& [rid, intervals] : resourceIntervals) {
            std::sort(intervals.begin(), intervals.end(),
                      [](const Interval& a, const Interval& b) {
                          return a.start < b.start;
                      });
            for (int i = 1; i < (int)intervals.size(); i++) {
                if (intervals[i].start < intervals[i-1].finish) {
                    std::cout << "CONFLICT on resource " << rid
                              << ": [" << intervals[i-1].start << "," << intervals[i-1].finish << ")"
                              << " overlaps [" << intervals[i].start << "," << intervals[i].finish << ")"
                              << "\n";
                    valid = false;
                }
            }
        }
        if (valid)
            std::cout << "Schedule validation: PASSED (no resource conflicts)\n\n";
        return valid;
    }
};
