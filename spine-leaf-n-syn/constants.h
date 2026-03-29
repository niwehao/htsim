/*
 * constants.h  —  512-GPU Leaf-Spine 同步调度版
 *
 * 拓扑: 2 层 Leaf-Spine (自动路由, 无硬编码)
 *
 *   32 Leaf × 16 Spine, 每交换机 32 端口
 *   Leaf: 下行端口 0..15 (连 GPU), 上行端口 16..31 (连 Spine)
 *   Spine: 端口 0..31 (连 Leaf)
 *
 * 只需修改 NUM_GPU_NODES 即可自动调整拓扑.
 *
 * 包含:
 *   §1  网络/MOE 常量 + 拓扑参数
 *   §2  自动路由
 *   §3  MoePacket
 *   §4  TimerEvent
 *   §5  NodeStats
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
//  §1  网络 / MOE 常量 + 拓扑参数
// ================================================================

// ---- 可调参数 ----
static constexpr uint32_t NUM_GPU_NODES       = 128;//start512 32
static constexpr uint32_t PRINT_MOD       = NUM_GPU_NODES/32;//start512 32
static constexpr uint32_t PORTS_PER_SWITCH    = 16;
static constexpr int      LOG_LEVEL           = 1;  // 0=silent, 1=milestone, 2=verbose

// ---- 自动推导 ----
static constexpr uint32_t GPUS_PER_LEAF       = PORTS_PER_SWITCH / 2;  // 16
static constexpr uint32_t NUM_UPLINKS_PER_LEAF= PORTS_PER_SWITCH / 2;  // 16
static constexpr uint32_t NUM_LEAVES          = NUM_GPU_NODES / GPUS_PER_LEAF; // 32
static constexpr uint32_t NUM_SPINES          = NUM_UPLINKS_PER_LEAF;  // 16
static constexpr uint32_t NUM_SWITCHES        = NUM_LEAVES + NUM_SPINES; // 48

static_assert(NUM_GPU_NODES % GPUS_PER_LEAF == 0,
              "NUM_GPU_NODES must be divisible by GPUS_PER_LEAF");
static_assert(NUM_LEAVES <= PORTS_PER_SWITCH,
              "NUM_LEAVES exceeds Spine port count");

// ---- 链路参数 ----
static constexpr uint64_t PROT_RATE_Gbps      = 200;
static constexpr mem_b    SW_QUEUE_SIZE_BYTES  = 64LL * 1024 * 1024;  // 64 MB

// ---- MOE 参数 ----
static constexpr uint32_t HIDDEN_DIM           = 4096;
static constexpr uint32_t BYTES_PER_ELEM       = 2;
static constexpr uint32_t TOKENS_PER_TARGET    = 256;
static constexpr uint32_t PAYLOAD_BYTES_PER_TARGET =
    TOKENS_PER_TARGET * HIDDEN_DIM * BYTES_PER_ELEM;                  // 8 MB
static constexpr uint32_t FRAGMENT_PAYLOAD_SIZE = 4 * 1024;           // 4 KB
static constexpr uint32_t TOTAL_FRAGMENTS      =
    PAYLOAD_BYTES_PER_TARGET / FRAGMENT_PAYLOAD_SIZE;                  // 2048
static constexpr uint32_t TOTAL_LAYERS         = 2;

static constexpr int DATA_PKT_SIZE = 14 + 15 + (int)FRAGMENT_PAYLOAD_SIZE;
static constexpr int ACK_PKT_SIZE  = 14 + 15;

// 定时 (皮秒)
static constexpr simtime_picosec TIMEOUT_PS        = 10ULL * 1000000000ULL;  // 10 ms
static constexpr simtime_picosec INTERPHASE_GAP_PS =  1ULL * 1000000000ULL;  // 1 ms
static constexpr simtime_picosec BUFFER_DELAY_PS   =  2ULL * 1000000000ULL;  // 2 ms

// 速率 (bps)
static const linkspeed_bps LINK_SPEED_BPS   = speedFromGbps(PROT_RATE_Gbps);
static const linkspeed_bps FABRIC_SPEED_BPS =
    speedFromGbps(PROT_RATE_Gbps * PORTS_PER_SWITCH);

static constexpr mem_b HUGE_BUFFER = (mem_b)1 * 1024 * 1024 * 1024;

// ================================================================
//  §2  自动路由
// ================================================================

inline uint32_t gpuToLeaf(uint32_t gpuId) {
    return gpuId / GPUS_PER_LEAF;
}

inline uint32_t gpuToLocalPort(uint32_t gpuId) {
    return gpuId % GPUS_PER_LEAF;
}

inline uint32_t leafUpPort(uint32_t spineIdx) {
    return GPUS_PER_LEAF + spineIdx;
}

inline uint32_t spineSelectECMP(uint32_t src, uint32_t dst) {
    return (src + dst) % NUM_SPINES;
}

// ================================================================
//  §3  MoePacket
// ================================================================

class MoePacket : public Packet {
public:
    uint8_t  pktType    = 0;
    uint64_t roundId    = 0;
    uint16_t srcId      = 0;   // 0..511
    uint16_t targetId   = 0;
    uint16_t fragId     = 0;
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
                              uint16_t srcId, uint16_t targetId,
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
//  §4  TimerEvent
// ================================================================

class TimerEvent : public EventSource {
public:
    TimerEvent(const std::string& name) : EventSource(name) {}

    void arm(simtime_picosec delay, std::function<void()> cb) {
        _gen++;
        _cb = std::move(cb);
        EventList::sourceIsPending(*this, EventList::now() + delay);
    }

    void doNextEvent() override {
        _gen--;
        if (_gen == 0) { if (_cb) _cb(); }
    }

private:
    std::function<void()> _cb;
    uint64_t _gen = 0;
};

// ================================================================
//  §5  NodeStats
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
        std::cout << "  Node " << nodeId
                  << ": TX=" << totalTx
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
              << "        GLOBAL MOE COMMUNICATION STATS (SYNC)\n" << sep << "\n"
              << "  Nodes=" << g_allStats.size() << " TX=" << tx
              << " minTX=" << exp << " tasks=" << tasks
              << " retx=" << retx << " perfect=" << perf << "/" << tasks;
    if (tasks) std::cout << " (" << std::fixed << std::setprecision(1)
                         << 100.0*perf/tasks << "%)";
    if (exp)   std::cout << " overhead=" << std::setprecision(2)
                         << 100.0*retx/exp << "%";
    std::cout << "\n" << sep << "\n";

    uint32_t printCount = 0;
    for (auto* s : g_allStats) {
        if (s->totalFragRetransmits > 0) {
            s->printSummary();
            printCount++;
        }
    }
    if (printCount == 0)
        std::cout << "  All nodes: 0 retransmissions (perfect delivery)\n";
    else
        std::cout << "  (" << (g_allStats.size() - printCount)
                  << " nodes with 0 retransmissions omitted)\n";
    std::cout << sep << "\n";
}
