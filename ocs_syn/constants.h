/*
 * constants.h  —  OCS Synchronized scheduling simulation constants
 *
 * Topology: N GPUs connected via 1 OCS switch
 *   N-1 time slots per cycle (Circle Method)
 *   All-to-all communication (every GPU sends to every other GPU)
 *   GPU controls send timing, EcsBuffer handles routing
 */

#pragma once

#include "eventlist.h"
#include "network.h"
#include "queue.h"
#include "config.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// ================================================================
//  S1  Network / MOE constants
// ================================================================

static constexpr uint32_t NUM_GPU_NODES         = 128;
static constexpr uint32_t TOTAL_LAYERS          = 2;
static constexpr uint32_t NUM_ACTIVE_EXPERTS    = NUM_GPU_NODES - 1;  // all-to-all

static constexpr uint64_t PROT_RATE_Gbps        = 200;
static constexpr uint32_t OCS_PORT_NUM          = NUM_GPU_NODES;
static constexpr uint32_t HIDDEN_DIM            = 4096;
static constexpr uint32_t BYTES_PER_ELEM        = 2;
static constexpr uint32_t TOKENS_PER_TARGET     = 256;
static constexpr uint32_t PAYLOAD_BYTES_PER_TARGET =
    TOKENS_PER_TARGET * HIDDEN_DIM * BYTES_PER_ELEM;
static constexpr uint32_t FRAGMENT_PAYLOAD_SIZE = 4 * 1024;
static constexpr uint32_t TOTAL_FRAGMENTS       =
    PAYLOAD_BYTES_PER_TARGET / FRAGMENT_PAYLOAD_SIZE;
static constexpr uint32_t NUM_SLOTS             = NUM_GPU_NODES - 1;

static constexpr int DATA_PKT_SIZE = 14 + 15 + (int)FRAGMENT_PAYLOAD_SIZE;
static constexpr int ACK_PKT_SIZE  = 14 + 15;

// Timing (picoseconds)
static constexpr simtime_picosec TIMEOUT_PS        = 20ULL * 1000000000ULL;
static constexpr simtime_picosec INTERPHASE_GAP_PS =   1ULL * 1000000000ULL;

// OCS reconfiguration delay: 10 us
static constexpr simtime_picosec RECONFIG_DELAY_PS = 10ULL * 1000000ULL;

// Link speed
static const linkspeed_bps LINK_SPEED_BPS = speedFromGbps(PROT_RATE_Gbps);

// GPU <-> EcsBuffer: instant transfer
static const linkspeed_bps GPU_LOCAL_SPEED = speedFromGbps(10000);

// Queue buffer
static constexpr mem_b HUGE_BUFFER = (mem_b)2LL * 1024 * 1024 * 1024;

// ================================================================
//  S2  HOHO time slot parameters
// ================================================================

static constexpr simtime_picosec SLOT_TX_TIME_PS =
    (simtime_picosec)TOTAL_FRAGMENTS * DATA_PKT_SIZE * 8ULL * 1000ULL * 1.2
    / PROT_RATE_Gbps;

static constexpr simtime_picosec SLICE_ACTIVE_PS =
    SLOT_TX_TIME_PS + SLOT_TX_TIME_PS / 20;

static constexpr simtime_picosec SLICE_TOTAL_PS =
    SLICE_ACTIVE_PS + RECONFIG_DELAY_PS;

static constexpr simtime_picosec CYCLE_PS =
    (simtime_picosec)NUM_SLOTS * SLICE_TOTAL_PS;

static constexpr mem_b OCS_TX_BUFFER = 256LL * 1024 * 1024;

static constexpr simtime_picosec OCS_LINK_DELAY_PS = 100ULL * 1000ULL;

static constexpr bool VERBOSE_LOG = (NUM_GPU_NODES <= 16);

// ================================================================
//  S3  All-to-all target selection
// ================================================================

inline std::vector<uint16_t> selectMoeTargets(uint16_t nodeId, int /*layer*/, int /*phase*/) {
    std::vector<uint16_t> targets;
    for (uint16_t i = 1; i <= NUM_GPU_NODES; i++)
        if (i != nodeId) targets.push_back(i);
    return targets;
}

struct PhaseAssignment {
    std::vector<uint16_t> send_targets;
    std::vector<uint16_t> recv_from;
};

inline std::map<uint16_t, PhaseAssignment> computePhaseAssignments(int layer, int phase) {
    std::map<uint16_t, PhaseAssignment> assignments;
    for (uint16_t gpu = 1; gpu <= NUM_GPU_NODES; gpu++)
        assignments[gpu] = PhaseAssignment{};

    for (uint16_t gpu = 1; gpu <= NUM_GPU_NODES; gpu++) {
        auto targets = selectMoeTargets(gpu, layer, phase);
        assignments[gpu].send_targets = targets;
        for (auto t : targets)
            assignments[t].recv_from.push_back(gpu);
    }
    return assignments;
}

// ================================================================
//  S4  MoePacket
// ================================================================

class MoePacket : public Packet {
public:
    uint8_t  pktType     = 0;
    uint64_t roundId     = 0;
    uint16_t srcId       = 0;
    uint16_t targetId    = 0;
    uint16_t fragId      = 0;
    uint16_t totalFrags  = 0;
    int16_t  target_slice = -1;

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
        p->pktType      = pktType;
        p->roundId       = roundId;
        p->srcId         = srcId;
        p->targetId      = targetId;
        p->fragId        = fragId;
        p->totalFrags    = (uint16_t)TOTAL_FRAGMENTS;
        p->target_slice  = -1;
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
        std::cout << "  TX=" << totalTx << " tasks=" << totalTasks
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
              << "        GLOBAL MOE COMMUNICATION STATS\n" << sep << "\n"
              << "  Nodes=" << g_allStats.size() << " TX=" << tx
              << " minTX=" << exp << " tasks=" << tasks
              << " retx=" << retx << " perfect=" << perf << "/" << tasks;
    if (tasks) std::cout << " (" << std::fixed << std::setprecision(1)
                         << 100.0*perf/tasks << "%)";
    if (exp)   std::cout << " overhead=" << std::setprecision(2)
                         << 100.0*retx/exp << "%";
    std::cout << "\n" << sep << "\n";
}
