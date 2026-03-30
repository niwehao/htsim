/*
 * constants.h  —  N-GPU Dragonfly 版 (平衡拓扑, 全自动推导)
 *
 * 拓扑: Dragonfly (自动路由, 无硬编码)
 *
 *   只需修改 NUM_GPU_NODES 和 PORTS_PER_SWITCH 即可自动调整拓扑.
 *   (p, a, h) 由编译期自动推导, 满足 a ≈ 2h 平衡条件.
 *
 *   平衡原则 (Kim et al. Dragonfly 论文):
 *     当 a = 2h 时, 全局流量和局部流量在交换机端口上分配最为均衡.
 *     算法在所有合法整数解中选择 |a - 2h| 最小的方案.
 *
 *   每交换机端口分配:
 *     [0, p)                  — GPU 下行端口 (p 不再硬编码为 k/2)
 *     [p, p+a-1)              — 组内 local 端口 (全互联)
 *     [p+a-1, PORTS_PER_SWITCH) — 组间 global 端口
 *
 *   约束:
 *     k = p + (a-1) + h       (端口预算)
 *     N = p * a * g            (GPU 总数)
 *     h ≥ g - 1               (组间全连接)
 *     a ≈ 2h                  (带宽平衡, 目标函数)
 *
 * 包含:
 *   §1  网络/MOE 常量 + Dragonfly 拓扑自动推导
 *   §2  全局链路表 + 路径计算
 *   §3  MoePacket
 *   §4  TimerEvent
 *   §5  NodeStats
 */

#pragma once

#include "eventlist.h"
#include "network.h"
#include "queue.h"
#include "config.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

// ================================================================
//  §1  网络 / MOE 常量 + Dragonfly 拓扑自动推导
// ================================================================

// ---- 可调参数 (只需改这两个) ----
static constexpr uint32_t NUM_GPU_NODES       = 128;
static constexpr uint32_t PORTS_PER_SWITCH    = 16;
static constexpr int      LOG_LEVEL           = 1;  // 0=silent, 1=milestone, 2=verbose
static constexpr uint32_t PRINT_MOD           = (NUM_GPU_NODES >= 32) ? NUM_GPU_NODES / 32 : 1;

// ---- 编译期自动推导 (p, a, h) ----
//
//  Dragonfly 平衡规则:  a ≈ 2h
//    k = p + (a-1) + h   (端口预算)
//    N = p * a * g        (GPU 总数)
//    h ≥ g - 1            (组间全连接)
//
//  算法: 遍历所有合法 (p, a, h) 组合, 选 |a - 2h| 最小者.
//  同分时优先选 p 更大的方案 (更高计算密度).
//
struct DragonflyCfg {
    uint32_t p;   // GPU 下行端口数
    uint32_t a;   // 每组交换机数
    uint32_t h;   // 每交换机 global 端口数
    uint32_t g;   // 组数
};

static constexpr DragonflyCfg autoDeriveDragonfly() {
    uint32_t k = PORTS_PER_SWITCH;
    uint32_t N = NUM_GPU_NODES;

    DragonflyCfg best = {0, 0, 0, 0};
    int bestScore = 99999;

    for (uint32_t a = 2; a < k; a++) {
        for (uint32_t h = 1; (a - 1) + h < k; h++) {
            uint32_t p = k - (a - 1) - h;
            if (p < 1)        continue;
            if (N % p != 0)   continue;
            uint32_t numSw = N / p;
            if (numSw % a != 0) continue;
            uint32_t g = numSw / a;
            if (g < 2)        continue;
            if (h < g - 1)    continue;

            int score = (int)a > 2 * (int)h
                      ? (int)a - 2 * (int)h
                      : 2 * (int)h - (int)a;

            if (score < bestScore ||
                (score == bestScore && p > best.p)) {
                best = {p, a, h, g};
                bestScore = score;
            }
        }
    }
    return best;
}

static constexpr DragonflyCfg DF_CFG = autoDeriveDragonfly();
static_assert(DF_CFG.a >= 2,
              "Cannot build balanced Dragonfly: try adjusting PORTS_PER_SWITCH or NUM_GPU_NODES");

// ---- 由推导结果定义常量 ----
static constexpr uint32_t GPUS_PER_SWITCH     = DF_CFG.p;
static constexpr uint32_t SWITCHES_PER_GROUP  = DF_CFG.a;
static constexpr uint32_t GLOBAL_PORTS_PER_SW = DF_CFG.h;
static constexpr uint32_t NUM_GROUPS          = DF_CFG.g;
static constexpr uint32_t NUM_SWITCHES        = NUM_GPU_NODES / GPUS_PER_SWITCH;
static constexpr uint32_t LOCAL_PORTS         = SWITCHES_PER_GROUP - 1;
static constexpr uint32_t GPUS_PER_GROUP      = SWITCHES_PER_GROUP * GPUS_PER_SWITCH;

// 端口基址
static constexpr uint32_t LOCAL_PORT_BASE     = GPUS_PER_SWITCH;
static constexpr uint32_t GLOBAL_PORT_BASE    = GPUS_PER_SWITCH + LOCAL_PORTS;

static_assert(NUM_GPU_NODES % GPUS_PER_SWITCH == 0,
              "NUM_GPU_NODES must be divisible by GPUS_PER_SWITCH");
static_assert(NUM_SWITCHES % SWITCHES_PER_GROUP == 0,
              "NUM_SWITCHES must be divisible by SWITCHES_PER_GROUP");
static_assert(GLOBAL_PORTS_PER_SW >= NUM_GROUPS - 1,
              "Not enough global ports per switch for full inter-group connectivity");
static_assert(GPUS_PER_SWITCH + LOCAL_PORTS + GLOBAL_PORTS_PER_SW == PORTS_PER_SWITCH,
              "Port budget p + (a-1) + h must equal PORTS_PER_SWITCH");

// ---- 链路参数 ----
static constexpr uint64_t PROT_RATE_Gbps      = 200;
static constexpr mem_b    SW_QUEUE_SIZE_BYTES  = 64LL * 1024 * 1024;  // 64 MB

// ---- MOE 参数 ----
static constexpr uint32_t HIDDEN_DIM           = 4096;
static constexpr uint32_t BYTES_PER_ELEM       = 2;
static constexpr uint32_t TOKENS_PER_TARGET    = 256;
static constexpr uint32_t PAYLOAD_BYTES_PER_TARGET =
    TOKENS_PER_TARGET * HIDDEN_DIM * BYTES_PER_ELEM;
static constexpr uint32_t FRAGMENT_PAYLOAD_SIZE = 4 * 1024;
static constexpr uint32_t TOTAL_FRAGMENTS      =
    PAYLOAD_BYTES_PER_TARGET / FRAGMENT_PAYLOAD_SIZE;
static constexpr uint32_t TOTAL_LAYERS         = 2;

static constexpr int DATA_PKT_SIZE = 14 + 15 + (int)FRAGMENT_PAYLOAD_SIZE;
static constexpr int ACK_PKT_SIZE  = 14 + 15;

// 定时 (皮秒)
static constexpr simtime_picosec TIMEOUT_PS        = 10ULL * 1000000000ULL;
static constexpr simtime_picosec INTERPHASE_GAP_PS =  1ULL * 1000000000ULL;
static constexpr simtime_picosec BUFFER_DELAY_PS   =  0.2*1000000000ULL;

// 速率 (bps)
static const linkspeed_bps LINK_SPEED_BPS   = speedFromGbps(PROT_RATE_Gbps);
static const linkspeed_bps FABRIC_SPEED_BPS =
    speedFromGbps(PROT_RATE_Gbps * PORTS_PER_SWITCH);

static constexpr mem_b HUGE_BUFFER = (mem_b)1 * 1024 * 1024 * 1024;

// ================================================================
//  §2  自动路由 — 全局链路表 + 路径计算
// ================================================================

// ---- GPU ↔ Switch 基本映射 ----
inline uint32_t gpuToSwitch(uint32_t gpuId)     { return gpuId / GPUS_PER_SWITCH; }
inline uint32_t gpuToLocalPort(uint32_t gpuId)  { return gpuId % GPUS_PER_SWITCH; }
inline uint32_t gpuToGroup(uint32_t gpuId)      { return gpuToSwitch(gpuId) / SWITCHES_PER_GROUP; }
inline uint32_t switchToGroup(uint32_t swIdx)   { return swIdx / SWITCHES_PER_GROUP; }
inline uint32_t switchInGroup(uint32_t swIdx)   { return swIdx % SWITCHES_PER_GROUP; }

// ---- 组内全互联: local 端口映射 (通用 a) ----
// Switch j 的 local port l (0..a-2) 连到同组的 peer switch
inline uint32_t localPeer(uint32_t mySwIG, uint32_t localIdx) {
    return localIdx < mySwIG ? localIdx : localIdx + 1;
}
// 从 peer 到 myself 用哪个 local port index?
inline uint32_t localPortIdx(uint32_t peerSwIG, uint32_t mySwIG) {
    return mySwIG < peerSwIG ? mySwIG : mySwIG - 1;
}
// 实际端口号
inline uint32_t localOutPort(uint32_t mySwIG, uint32_t dstSwIG) {
    uint32_t li = dstSwIG < mySwIG ? dstSwIG : dstSwIG - 1;
    return LOCAL_PORT_BASE + li;
}
inline uint32_t localInPort(uint32_t mySwIG, uint32_t srcSwIG) {
    uint32_t li = srcSwIG < mySwIG ? srcSwIG : srcSwIG - 1;
    return LOCAL_PORT_BASE + li;
}

// ================================================================
//  全局链路表 (运行时初始化)
// ================================================================

// g_glTarget[sw][gp]     = 对端交换机 index
// g_glRemotePort[sw][gp] = 对端绝对端口号

inline std::vector<std::vector<uint32_t>> g_glTarget;
inline std::vector<std::vector<uint32_t>> g_glRemotePort;

// ECMP 路由表: 每个 (sw, dstGroup) 存储所有可用的 global 链路
struct GlobalLink {
    uint32_t gp;       // global port index (0..h-1)
    uint32_t remSw;    // 对端交换机 index
    uint32_t remPort;  // 对端绝对端口号
};
// g_routeLinks[sw][dstGroup] = vector of GlobalLink
inline std::vector<std::vector<std::vector<GlobalLink>>> g_routeLinks;

// ECMP 选路: hash(srcGpu, dstGpu) 选择 global link
inline const GlobalLink& globalSelectECMP(uint32_t srcSw, uint32_t dstGroup,
                                           uint32_t srcGpu, uint32_t dstGpu) {
    auto& links = g_routeLinks[srcSw][dstGroup];
    uint32_t idx = (srcGpu + dstGpu) % (uint32_t)links.size();
    return links[idx];
}

inline void initDragonflyTopology() {
    g_glTarget.assign(NUM_SWITCHES, std::vector<uint32_t>(GLOBAL_PORTS_PER_SW, UINT32_MAX));
    g_glRemotePort.assign(NUM_SWITCHES, std::vector<uint32_t>(GLOBAL_PORTS_PER_SW, UINT32_MAX));
    g_routeLinks.assign(NUM_SWITCHES, std::vector<std::vector<GlobalLink>>(NUM_GROUPS));

    // ---- Step 1: 每交换机的 global 端口分配到目标 group (round-robin) ----
    using LinkEnd = std::pair<uint32_t, uint32_t>;
    std::vector<std::vector<std::vector<LinkEnd>>> pending(
        NUM_GROUPS, std::vector<std::vector<LinkEnd>>(NUM_GROUPS));

    for (uint32_t g = 0; g < NUM_GROUPS; g++) {
        uint32_t tgIdx = 0;
        for (uint32_t j = 0; j < SWITCHES_PER_GROUP; j++) {
            uint32_t sw = g * SWITCHES_PER_GROUP + j;
            for (uint32_t gp = 0; gp < GLOBAL_PORTS_PER_SW; gp++) {
                uint32_t tg = (g + (tgIdx % (NUM_GROUPS - 1)) + 1) % NUM_GROUPS;
                pending[g][tg].push_back({sw, gp});
                tgIdx++;
            }
        }
    }

    // ---- Step 2: 匹配组对之间的链路 ----
    for (uint32_t g1 = 0; g1 < NUM_GROUPS; g1++) {
        for (uint32_t g2 = g1 + 1; g2 < NUM_GROUPS; g2++) {
            auto& from1 = pending[g1][g2];
            auto& from2 = pending[g2][g1];
            uint32_t numLinks = (uint32_t)std::min(from1.size(), from2.size());
            for (uint32_t i = 0; i < numLinks; i++) {
                auto [sw1, gp1] = from1[i];
                auto [sw2, gp2] = from2[i];
                g_glTarget[sw1][gp1] = sw2;
                g_glRemotePort[sw1][gp1] = GLOBAL_PORT_BASE + gp2;
                g_glTarget[sw2][gp2] = sw1;
                g_glRemotePort[sw2][gp2] = GLOBAL_PORT_BASE + gp1;
            }
        }
    }

    // ---- Step 3: 构建 ECMP 路由表 (收集每交换机到每组的所有可用链路) ----
    for (uint32_t sw = 0; sw < NUM_SWITCHES; sw++) {
        for (uint32_t gp = 0; gp < GLOBAL_PORTS_PER_SW; gp++) {
            uint32_t remSw = g_glTarget[sw][gp];
            if (remSw == UINT32_MAX) continue;
            uint32_t tg = switchToGroup(remSw);
            g_routeLinks[sw][tg].push_back({gp, remSw, g_glRemotePort[sw][gp]});
        }
    }

    // ---- 验证: 每交换机到每个其他组都有路由 ----
    for (uint32_t sw = 0; sw < NUM_SWITCHES; sw++) {
        uint32_t myGroup = switchToGroup(sw);
        for (uint32_t g = 0; g < NUM_GROUPS; g++) {
            if (g == myGroup) continue;
            if (g_routeLinks[sw][g].empty()) {
                std::cerr << "FATAL: Sw" << sw << " (G" << myGroup
                          << ") has no global route to G" << g << "!\n";
                exit(1);
            }
        }
    }
}

// ---- 路径计算 (需先调用 initDragonflyTopology) ----
struct SwitchPath {
    uint32_t sw[4];
    uint32_t inPort[4];
    uint32_t outPort[4];
    uint32_t numHops;
};

inline SwitchPath computeSwitchPath(uint32_t srcGpu, uint32_t dstGpu) {
    SwitchPath path = {};
    uint32_t srcSw    = gpuToSwitch(srcGpu);
    uint32_t dstSw    = gpuToSwitch(dstGpu);
    uint32_t srcGroup = switchToGroup(srcSw);
    uint32_t dstGroup = switchToGroup(dstSw);
    uint32_t srcSwIG  = switchInGroup(srcSw);
    uint32_t dstSwIG  = switchInGroup(dstSw);
    uint32_t srcPort  = gpuToLocalPort(srcGpu);
    uint32_t dstPort  = gpuToLocalPort(dstGpu);

    if (srcSw == dstSw) {
        // ---- 同交换机: 1 hop ----
        path.numHops = 1;
        path.sw[0] = srcSw;
        path.inPort[0] = srcPort;
        path.outPort[0] = dstPort;

    } else if (srcGroup == dstGroup) {
        // ---- 同组不同交换机: 2 hops (组内全互联) ----
        path.numHops = 2;
        path.sw[0] = srcSw;
        path.inPort[0] = srcPort;
        path.outPort[0] = localOutPort(srcSwIG, dstSwIG);
        path.sw[1] = dstSw;
        path.inPort[1] = localInPort(dstSwIG, srcSwIG);
        path.outPort[1] = dstPort;

    } else {
        // ---- 跨组: ECMP 选择 global 链路 ----
        const GlobalLink& gl = globalSelectECMP(srcSw, dstGroup, srcGpu, dstGpu);
        uint32_t outP      = GLOBAL_PORT_BASE + gl.gp;
        uint32_t remoteSw  = gl.remSw;
        uint32_t remInPort = gl.remPort;
        uint32_t remSwIG   = switchInGroup(remoteSw);

        if (remoteSw == dstSw) {
            // 2 hops: srcSw —global→ dstSw
            path.numHops = 2;
            path.sw[0] = srcSw;  path.inPort[0] = srcPort; path.outPort[0] = outP;
            path.sw[1] = dstSw;  path.inPort[1] = remInPort; path.outPort[1] = dstPort;
        } else {
            // 3 hops: srcSw —global→ remoteSw —local→ dstSw
            path.numHops = 3;
            path.sw[0] = srcSw;
            path.inPort[0] = srcPort;
            path.outPort[0] = outP;
            path.sw[1] = remoteSw;
            path.inPort[1] = remInPort;
            path.outPort[1] = localOutPort(remSwIG, dstSwIG);
            path.sw[2] = dstSw;
            path.inPort[2] = localInPort(dstSwIG, remSwIG);
            path.outPort[2] = dstPort;
        }
    }
    return path;
}

// ================================================================
//  §3  MoePacket
// ================================================================

class MoePacket : public Packet {
public:
    uint8_t  pktType    = 0;
    uint64_t roundId    = 0;
    uint16_t srcId      = 0;
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
              << "    GLOBAL MOE COMMUNICATION STATS (DRAGONFLY)\n" << sep << "\n"
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
