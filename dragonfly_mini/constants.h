/*
 * constants.h  —  Dragonfly 拓扑版
 *
 * 基于 htsim_v2 修改，将 3 级 Clos 替换为 Dragonfly 拓扑。
 *
 * Dragonfly 参数:
 *   2 个 Group, 每 Group 2 个交换机, 每交换机连 2 个 GPU
 *   每交换机 4 端口: 2 GPU + 1 local + 1 global
 *
 * 拓扑:
 *   Group 0: Sw0 (GPU1,GPU2), Sw1 (GPU3,GPU4)
 *             Sw0 ←local→ Sw1
 *   Group 1: Sw2 (GPU5,GPU6), Sw3 (GPU7,GPU8)
 *             Sw2 ←local→ Sw3
 *   Global: Sw0 ←→ Sw2, Sw1 ←→ Sw3
 *
 * 包含:
 *   §1  网络/MOE 常量
 *   §2  路由表 (Dragonfly minimal routing)
 *   §3  端口 / GPU 映射
 *   §4  MoePacket
 *   §5  TimerEvent
 *   §6  NodeStats
 */

#pragma once

#include "eventlist.h"
#include "network.h"
#include "queue.h"
#include "config.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// ================================================================
//  §1  网络 / MOE 常量
// ================================================================

static constexpr uint64_t PROT_RATE_Gbps       = 200;
static constexpr uint32_t PORT_NUM              = 4;
static constexpr mem_b    SW_QUEUE_SIZE_BYTES   = 64LL * 1024 * 1024;   // 32 MB
static constexpr uint32_t HIDDEN_DIM            = 4096;
static constexpr uint32_t BYTES_PER_ELEM        = 2;
static constexpr uint32_t TOKENS_PER_TARGET     = 1024;
static constexpr uint32_t PAYLOAD_BYTES_PER_TARGET =
    TOKENS_PER_TARGET * HIDDEN_DIM * BYTES_PER_ELEM;                    // 8 MB
static constexpr uint32_t FRAGMENT_PAYLOAD_SIZE = 4 * 1024;            // 4 KB
static constexpr uint32_t TOTAL_FRAGMENTS       =
    PAYLOAD_BYTES_PER_TARGET / FRAGMENT_PAYLOAD_SIZE;                   // 2048
static constexpr uint32_t NUM_GPU_NODES         = 8;
static constexpr uint32_t NUM_SWITCHES          = 4;
static constexpr uint32_t TOTAL_LAYERS          = 32;

// Dragonfly 参数
static constexpr uint32_t NUM_GROUPS            = 2;
static constexpr uint32_t SWITCHES_PER_GROUP    = 2;
static constexpr uint32_t GPUS_PER_SWITCH       = 2;

static constexpr int DATA_PKT_SIZE = 14 + 15 + (int)FRAGMENT_PAYLOAD_SIZE;
static constexpr int ACK_PKT_SIZE  = 14 + 15;

static constexpr simtime_picosec TIMEOUT_PS        = 15ULL * 1000000000ULL;
static constexpr simtime_picosec INTERPHASE_GAP_PS =   1ULL * 1000000000ULL;
static constexpr simtime_picosec BUFFER_DELAY_PS   =   2ULL * 1000000000ULL;

static const linkspeed_bps LINK_SPEED_BPS   = speedFromGbps(PROT_RATE_Gbps);
static const linkspeed_bps FABRIC_SPEED_BPS = speedFromGbps(PROT_RATE_Gbps * PORT_NUM);

static constexpr mem_b HUGE_BUFFER = (mem_b)1 * 1024 * 1024 * 1024;

// ================================================================
//  §2  路由表 — Dragonfly Minimal Routing
//
//  4 个交换机:
//    Sw0 (Group 0): ports {1,2,3,4}
//      port 1→GPU1, port 2→GPU2, port 3→local(Sw1), port 4→global(Sw2)
//    Sw1 (Group 0): ports {5,6,7,8}
//      port 5→GPU3, port 6→GPU4, port 7→local(Sw0), port 8→global(Sw3)
//    Sw2 (Group 1): ports {9,10,11,12}
//      port 9→GPU5, port 10→GPU6, port 11→local(Sw3), port 12→global(Sw0)
//    Sw3 (Group 1): ports {13,14,15,16}
//      port 13→GPU7, port 14→GPU8, port 15→local(Sw2), port 16→global(Sw1)
//
//  Minimal routing:
//    同交换机: 直达 GPU 端口
//    同 Group 不同交换机: local link
//    不同 Group: global link → (可能需 local link 到达目标交换机)
// ================================================================

inline std::map<int,int> resolvedRoutingTable(int idx) {
    switch (idx) {
    case 0: // Sw0 (Group 0): GPU1@p1, GPU2@p2, local@p3, global@p4
        return {
            {1, 1}, {2, 2},         // 本交换机 GPU
            {3, 3}, {4, 3},         // 同 Group → local link → Sw1
            {5, 4}, {6, 4},         // 不同 Group → global → Sw2
            {7, 4}, {8, 4}          // 不同 Group → global → Sw2 → local → Sw3
        };
    case 1: // Sw1 (Group 0): GPU3@p5, GPU4@p6, local@p7, global@p8
        return {
            {1, 7}, {2, 7},         // 同 Group → local link → Sw0
            {3, 5}, {4, 6},         // 本交换机 GPU
            {5, 8}, {6, 8},         // 不同 Group → global → Sw3 → local → Sw2
            {7, 8}, {8, 8}          // 不同 Group → global → Sw3
        };
    case 2: // Sw2 (Group 1): GPU5@p9, GPU6@p10, local@p11, global@p12
        return {
            {1, 12}, {2, 12},       // 不同 Group → global → Sw0
            {3, 12}, {4, 12},       // 不同 Group → global → Sw0 → local → Sw1
            {5,  9}, {6, 10},       // 本交换机 GPU
            {7, 11}, {8, 11}        // 同 Group → local link → Sw3
        };
    case 3: // Sw3 (Group 1): GPU7@p13, GPU8@p14, local@p15, global@p16
        return {
            {1, 16}, {2, 16},       // 不同 Group → global → Sw1 → local → Sw0
            {3, 16}, {4, 16},       // 不同 Group → global → Sw1
            {5, 15}, {6, 15},       // 同 Group → local link → Sw2
            {7, 13}, {8, 14}        // 本交换机 GPU
        };
    default: assert(false); return {};
    }
}

inline std::vector<int> tofinoPorts(int idx) {
    switch (idx) {
    case 0: return {1,2,3,4};
    case 1: return {5,6,7,8};
    case 2: return {9,10,11,12};
    case 3: return {13,14,15,16};
    default: assert(false); return {};
    }
}

// ================================================================
//  §3  端口 / GPU 映射
// ================================================================

// port_mapping: 交换机间互联
//   local:  port 3 ↔ port 7  (Sw0 ↔ Sw1, Group 0)
//           port 11 ↔ port 15 (Sw2 ↔ Sw3, Group 1)
//   global: port 4 ↔ port 12 (Sw0 ↔ Sw2)
//           port 8 ↔ port 16 (Sw1 ↔ Sw3)
inline int portMapping(int port) {
    static const std::map<int,int> m = {
        {3, 7},  {7, 3},    // local: Sw0 ↔ Sw1
        {11, 15},{15, 11},   // local: Sw2 ↔ Sw3
        {4, 12}, {12, 4},   // global: Sw0 ↔ Sw2
        {8, 16}, {16, 8}    // global: Sw1 ↔ Sw3
    };
    auto it = m.find(port);
    assert(it != m.end());
    return it->second;
}

// GPU ID → 端口号
inline int gpuToPort(int gpuId) {
    static const int t[] = {0, 1,2,5,6,9,10,13,14}; // 1-indexed
    return t[gpuId];
}

// 端口号 → GPU ID
inline int portToGpu(int port) {
    static const std::map<int,int> m = {
        {1,1},{2,2},{5,3},{6,4},{9,5},{10,6},{13,7},{14,8}
    };
    auto it = m.find(port);
    return (it != m.end()) ? it->second : 0;
}

// GPU ID → 所在交换机索引 (0-3)
inline int gpuToTofinoIdx(int gpuId) {
    static const int t[] = {0, 0,0,1,1,2,2,3,3}; // 1-indexed
    return t[gpuId];
}

// 端口 → 所在交换机索引
inline int portToTofinoIdx(int port) {
    if (port >= 1  && port <= 4)  return 0;
    if (port >= 5  && port <= 8)  return 1;
    if (port >= 9  && port <= 12) return 2;
    if (port >= 13 && port <= 16) return 3;
    assert(false); return -1;
}

// 端口是否连接 GPU
inline bool isGpuPort(int port) { return portToGpu(port) != 0; }

// ================================================================
//  §4  MoePacket
// ================================================================

class MoePacket : public Packet {
public:
    uint8_t  pktType   = 0;
    uint64_t roundId   = 0;
    uint8_t  srcId     = 0;
    uint8_t  targetId  = 0;
    uint16_t fragId    = 0;
    uint16_t totalFrags = 0;

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
        p->pktType    = pktType;
        p->roundId    = roundId;
        p->srcId      = srcId;
        p->targetId   = targetId;
        p->fragId     = fragId;
        p->totalFrags = (uint16_t)TOTAL_FRAGMENTS;
        return p;
    }

    void free() override { _db.freePacket(this); }
    PktPriority priority() const override { return Packet::PRIO_LO; }

    static PacketDB<MoePacket> _db;
    static packetid_t _nextId;
};

// ================================================================
//  §5  TimerEvent
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
//  §6  NodeStats
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
              << "        GLOBAL MOE COMMUNICATION STATS (Dragonfly)\n" << sep << "\n"
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
