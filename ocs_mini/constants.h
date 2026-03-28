/*
 * constants.h  —  OCS + EcsBuffer HOHO 路由版
 *
 * 拓扑: 8 个 GPU 通过 1 个 OCS 交换机互联
 *   OCS 交换机有 8 个端口, 每个端口连一个 GPU
 *   每个节点一个 EcsBuffer (Edge Circuit Switch Buffer)
 *   BFS 路由表: rt[node][dst][slice] = FORWARD / WAIT
 *   7 个时隙, 每个时隙配置一个完美匹配 (Circle Method)
 *
 * 包含:
 *   S1  网络/MOE 常量
 *   S2  HOHO 时隙参数
 *   S3  OCS 端口映射
 *   S4  MoePacket (含 target_slice 标签)
 *   S5  TimerEvent
 *   S6  NodeStats
 */

#pragma once

#include "eventlist.h"
#include "network.h"
#include "queue.h"
#include "config.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// ================================================================
//  S1  网络 / MOE 常量
// ================================================================

static constexpr uint64_t PROT_RATE_Gbps       = 200;
static constexpr uint32_t OCS_PORT_NUM          = 8;       // OCS 端口数
static constexpr uint32_t HIDDEN_DIM            = 4096;
static constexpr uint32_t BYTES_PER_ELEM        = 2;
static constexpr uint32_t TOKENS_PER_TARGET     = 1024;
static constexpr uint32_t PAYLOAD_BYTES_PER_TARGET =
    TOKENS_PER_TARGET * HIDDEN_DIM * BYTES_PER_ELEM;                    // 8 MB
static constexpr uint32_t FRAGMENT_PAYLOAD_SIZE = 4 * 1024;            // 4 KB
static constexpr uint32_t TOTAL_FRAGMENTS       =
    PAYLOAD_BYTES_PER_TARGET / FRAGMENT_PAYLOAD_SIZE;                   // 2048
static constexpr uint32_t NUM_GPU_NODES         = 8;
static constexpr uint32_t TOTAL_LAYERS          = 32;
static constexpr uint32_t NUM_SLOTS             = NUM_GPU_NODES - 1;   // 7 time slots per phase

static constexpr int DATA_PKT_SIZE = 14 + 15 + (int)FRAGMENT_PAYLOAD_SIZE;
static constexpr int ACK_PKT_SIZE  = 14 + 15;

// 定时 (皮秒)
static constexpr simtime_picosec TIMEOUT_PS        = 20ULL * 1000000000ULL;  // 20 ms
static constexpr simtime_picosec INTERPHASE_GAP_PS =   1ULL * 1000000000ULL; // 1 ms

// OCS 重配置延迟: 10 us (典型 MEMS OCS 重配置时间)
static constexpr simtime_picosec RECONFIG_DELAY_PS = 10ULL * 1000000ULL;     // 10 us

// 速率
static const linkspeed_bps LINK_SPEED_BPS = speedFromGbps(PROT_RATE_Gbps);

// 队列大容量上限 (GPU NIC 队列用, 需要容纳一个目标的全部数据)
static constexpr mem_b HUGE_BUFFER = (mem_b)1 * 1024 * 1024 * 1024;

// ================================================================
//  S2  HOHO 时隙参数
//
//  对应 operasim DynExpTopology 的 _slicetime 概念
//  每个时隙 = 活动传输时间 + 重配置间隙
//  7 个时隙构成一个完整周期
// ================================================================

// 每目标数据传输时间 at 200 Gbps
//   2048 fragments * 4125 bytes * 8 bits / 200 Gbps = 337,920,000 ps ~ 338 us
static constexpr simtime_picosec SLOT_TX_TIME_PS =
    (simtime_picosec)TOTAL_FRAGMENTS * DATA_PKT_SIZE * 8ULL * 1000ULL
    / PROT_RATE_Gbps;

// 时隙活动时间 (传输时间 + 5% 余量, 容纳 ACK 交错和排队抖动)
static constexpr simtime_picosec SLICE_ACTIVE_PS =
    SLOT_TX_TIME_PS + SLOT_TX_TIME_PS / 20;

// 时隙总时间 = 活动 + 重配置
static constexpr simtime_picosec SLICE_TOTAL_PS =
    SLICE_ACTIVE_PS + RECONFIG_DELAY_PS;

// 完整周期 = 7 个时隙
static constexpr simtime_picosec CYCLE_PS =
    (simtime_picosec)NUM_SLOTS * SLICE_TOTAL_PS;

// EcsBuffer 内部缓冲区大小 (每节点 128 MB, 容纳中转数据)
static constexpr mem_b ECS_BUFFER_SIZE = 128LL * 1024 * 1024;

// OCS TX 串行化队列缓冲 (每节点 128 MB, 需容纳中转 + 本地数据)
static constexpr mem_b OCS_TX_BUFFER = 128LL * 1024 * 1024;

// OCS 光纤传播延迟: 100 ns (机架内短距光纤)
static constexpr simtime_picosec OCS_LINK_DELAY_PS = 100ULL * 1000ULL;     // 100 ns

// ================================================================
//  S3  OCS 端口映射
//
//  OCS 只有 1 个交换机, 8 个端口
//  GPU ID = 端口号 (1-8)
//  路由: GPU.txQ -> EcsBuffer@src -> [OCS] -> EcsBuffer@dst -> GPU.rxQ
// ================================================================

// GPU ID -> OCS 端口号 (直接映射)
inline int gpuToPort(int gpuId) { return gpuId; }

// 端口号 -> GPU ID
inline int portToGpu(int port) {
    if (port >= 1 && port <= 8) return port;
    return 0;
}

// GPU ID -> 所在交换机索引 (只有 1 个 OCS, 索引始终为 0)
inline int gpuToSwitchIdx(int /*gpuId*/) { return 0; }

// ================================================================
//  S4  MoePacket (含 target_slice 标签)
//
//  target_slice: EcsBuffer 为每个包标记的目标转发时隙
//    -1 = 未标记 (刚从 GPU 发出, 或刚被转发)
//    0-6 = 目标时隙 (在该时隙到来时从 buffer 中取出转发)
// ================================================================

class MoePacket : public Packet {
public:
    uint8_t  pktType   = 0;
    uint64_t roundId   = 0;
    uint8_t  srcId     = 0;
    uint8_t  targetId  = 0;
    uint16_t fragId    = 0;
    uint16_t totalFrags = 0;
    int8_t   target_slice = -1;  // EcsBuffer routing tag

    static constexpr uint8_t PKT_DATA = 0;
    static constexpr uint8_t PKT_ACK  = 1;

    static uint64_t encodeRound(int layer, int phase) {
        return ((uint64_t)layer << 32) | ((uint64_t)(phase & 0xFFFFFFFF));
    }
    static void decodeRound(uint64_t rid, int& layer, int& phase) {
        layer = (int)((rid >> 32) & 0xFFFFFFFF);
        phase = (int)(rid & 0xFFFFFFFF);
    }

    static MoePacket* newpkt(PacketFlow& flow, const Route& route,
                              uint8_t pktType, uint64_t roundId,
                              uint8_t srcId, uint8_t targetId,
                              uint16_t fragId, int pktSize)
    {
        MoePacket* p = _db.allocPacket();
        p->set_route(flow, route, pktSize, _nextId++);
        p->pktType      = pktType;
        p->roundId       = roundId;
        p->srcId         = srcId;
        p->targetId      = targetId;
        p->fragId        = fragId;
        p->totalFrags    = (uint16_t)TOTAL_FRAGMENTS;
        p->target_slice  = -1;  // untagged
        return p;
    }

    void free() override { _db.freePacket(this); }
    PktPriority priority() const override { return Packet::PRIO_LO; }

    static PacketDB<MoePacket> _db;
    static packetid_t _nextId;
};

// ================================================================
//  S5  TimerEvent
// ================================================================

class TimerEvent : public EventSource {
public:
    TimerEvent(const std::string& name)
        : EventSource(name) {}

    void arm(simtime_picosec delay, std::function<void()> cb) {
        _gen ++;
        _cb = std::move(cb);
        EventList::sourceIsPending(*this, EventList::now() + delay);
    }

    void doNextEvent() override {
       _gen --;
        if (_gen == 0) { if (_cb) _cb(); }
    }

private:
    std::function<void()> _cb;
    uint64_t _gen = 0;
};

// ================================================================
//  S6  NodeStats
// ================================================================

struct TaskRecord {
    uint32_t totalAttempts = 0, retransmitFrags = 0, fragRetransmits = 0;
};

class NodeStats {
public:
    uint32_t nodeId = 0;
    uint64_t totalTx = 0, totalFragRetransmits = 0, totalTasks = 0;
    std::map<std::tuple<int,int,int>, TaskRecord> taskRecords;

    void recordTaskDone(int layer, int phase, int target,
                        const std::map<int,int>& fragAttempts)
    {
        TaskRecord rec;
        for (auto& [fid, cnt] : fragAttempts) {
            rec.totalAttempts += cnt;
            if (cnt > 1) { rec.retransmitFrags++; rec.fragRetransmits += cnt - 1; }
        }
        taskRecords[{layer, phase, target}] = rec;
        totalTx += rec.totalAttempts;
        totalFragRetransmits += rec.fragRetransmits;
        totalTasks++;
    }
    uint32_t perfectCount() const {
        uint32_t n = 0;
        for (auto& [k,v] : taskRecords) if (v.fragRetransmits == 0) n++;
        return n;
    }
    void printSummary() const {
        std::cout << "===== Node " << nodeId << " =====\n"
                  << "  TX=" << totalTx << " tasks=" << totalTasks
                  << " retx=" << totalFragRetransmits
                  << " perfect=" << perfectCount() << "/" << totalTasks << "\n";
    }
};

inline std::vector<NodeStats*> g_allStats;

inline void printGlobalSummary() {
    uint64_t tx=0, retx=0, tasks=0, perf=0, exp=0;
    for (auto* s : g_allStats) {
        tx += s->totalTx; retx += s->totalFragRetransmits;
        tasks += s->totalTasks; perf += s->perfectCount();
        exp += s->totalTasks * TOTAL_FRAGMENTS;
    }
    std::string sep(60, '=');
    std::cout << "\n" << sep << "\n"
              << "        GLOBAL MOE COMMUNICATION STATS (OCS + EcsBuffer)\n" << sep << "\n"
              << "  Nodes=" << g_allStats.size() << " TX=" << tx
              << " minTX=" << exp << " tasks=" << tasks
              << " retx=" << retx << " perfect=" << perf << "/" << tasks;
    if (tasks) std::cout << " (" << std::fixed << std::setprecision(1)
                         << 100.0*perf/tasks << "%)";
    if (exp)   std::cout << " overhead=" << std::setprecision(2)
                         << 100.0*retx/exp << "%";
    std::cout << "\n" << sep << "\n";
    for (auto* s : g_allStats)
        std::cout << "  Node" << s->nodeId << ": tx=" << s->totalTx
                  << " retx=" << s->totalFragRetransmits
                  << " perfect=" << s->perfectCount() << "\n";
    std::cout << sep << "\n";
}
