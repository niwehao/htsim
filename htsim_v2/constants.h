/*
 * constants.h  —  csg-htsim 版
 *
 * 翻译自: Constants.py, Statics.py
 *
 * 依赖: Broadcom/csg-htsim  (https://github.com/Broadcom/csg-htsim)
 *
 * 包含:
 *   §1  网络/MOE 常量
 *   §2  路由表 (已按 leaf_id 解析 ECMP)
 *   §3  端口 / GPU 映射
 *   §4  MoePacket (继承 csg-htsim Packet)
 *   §5  TimerEvent (通用定时回调)
 *   §6  NodeStats (统计)
 */

#pragma once

// ---- csg-htsim 头文件 ----
#include "eventlist.h"   // EventList, EventSource, simtime_picosec
#include "network.h"     // Packet, PacketSink, PacketFlow, Route, PacketDB
#include "queue.h"       // Queue
#include "config.h"      // linkspeed_bps, mem_b, speedFromGbps, timeFromMs …

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
//  §1  网络 / MOE 常量  (Constants.py)
// ================================================================

static constexpr uint64_t PROT_RATE_Gbps       = 200;
static constexpr uint32_t PORT_NUM              = 4;
static constexpr mem_b    SW_QUEUE_SIZE_BYTES   = 64LL * 1024 * 1024;   // 32 MB
static constexpr uint32_t HIDDEN_DIM            = 4096;
static constexpr uint32_t BYTES_PER_ELEM        = 2;
static constexpr uint32_t TOKENS_PER_TARGET     = 1024;//128个碎片比较合理
static constexpr uint32_t PAYLOAD_BYTES_PER_TARGET =
    TOKENS_PER_TARGET * HIDDEN_DIM * BYTES_PER_ELEM;                    // 8 MB
static constexpr uint32_t FRAGMENT_PAYLOAD_SIZE = 4 * 1024;            // 4 KB
static constexpr uint32_t TOTAL_FRAGMENTS       =
    PAYLOAD_BYTES_PER_TARGET / FRAGMENT_PAYLOAD_SIZE;                   // 128
static constexpr uint32_t NUM_GPU_NODES         = 8;
static constexpr uint32_t TOTAL_LAYERS          = 32;

// 数据包字节数  (Ether 14B + 自定义 header 15B + payload)
//   注意: csg-htsim Packet::_size 是 uint16_t (max 65535)
//   DATA_PKT_SIZE = 65565 > 65535, 编译时截断为 65535
//   如需精确值, 请将 csg-htsim network.h 中 _size 改为 uint32_t
static constexpr int DATA_PKT_SIZE = 14 + 15 + (int)FRAGMENT_PAYLOAD_SIZE;
static constexpr int ACK_PKT_SIZE  = 14 + 15;

// 定时 (皮秒)  1 ms = 1,000,000,000 ps
static constexpr simtime_picosec TIMEOUT_PS        = 20ULL * 1000000000ULL;   // 100 ms
static constexpr simtime_picosec INTERPHASE_GAP_PS =   1ULL * 1000000000ULL;   // 1 ms
static constexpr simtime_picosec BUFFER_DELAY_PS   =   2000000000ULL;   // 0.5ms

// 速率 (bps)
static const linkspeed_bps LINK_SPEED_BPS   = speedFromGbps(PROT_RATE_Gbps);
static const linkspeed_bps FABRIC_SPEED_BPS = speedFromGbps(PROT_RATE_Gbps * PORT_NUM);

// 队列大容量上限 (仅作 csg-htsim Queue 构造参数, 溢出由 BufferGate 管理)
static constexpr mem_b HUGE_BUFFER = (mem_b)1 * 1024 * 1024 * 1024;   // 1 GB

// ================================================================
//  §2  路由表 — 已按 leaf_id 解析 ECMP
//
//  原始 Python:
//    spine1_leaf1_routing_table = { 5: [1,3], ... }
//    select_out_port: out_ports[self.leaf_id - 1]
//
//  这里直接给出解析后的 map<gpuId, outPort>
// ================================================================
//  6 个 Tofino 的索引:
//    0 = S1L1, 1 = S1L2, 2 = S2L1, 3 = S2L2, 4 = S3L1, 5 = S3L2

inline std::map<int,int> resolvedRoutingTable(int idx) {
    switch (idx) {
    case 0: // S1L1 (Spine1.Leaf1, leaf_id=1, ports 1-4)
        return {{1,2},{2,4},{3,1},{4,3},{5,1},{6,3},{7,1},{8,3}};
    case 1: // S1L2 (Spine1.Leaf2, leaf_id=2, ports 5-8)
        return {{1,5},{2,7},{3,6},{4,8},{5,7},{6,5},{7,7},{8,5}};
    case 2: // S2L1 (Spine2.Leaf1, leaf_id=1, ports 9-12)
        return {{1,9},{2,11},{3,9},{4,11},{5,10},{6,12},{7,9},{8,11}};
    case 3: // S2L2 (Spine2.Leaf2, leaf_id=2, ports 13-16)
        return {{1,15},{2,13},{3,15},{4,13},{5,13},{6,15},{7,14},{8,16}};
    case 4: // S3L1 (Spine3.Leaf1, ports 17-20)
        return {{1,17},{2,17},{3,18},{4,18},{5,19},{6,19},{7,20},{8,20}};
    case 5: // S3L2 (Spine3.Leaf2, ports 21-24)
        return {{1,21},{2,21},{3,22},{4,22},{5,23},{6,23},{7,24},{8,24}};
    default: assert(false); return {};
    }
}

inline std::vector<int> tofinoPorts(int idx) {
    switch (idx) {
    case 0: return {1,2,3,4};
    case 1: return {5,6,7,8};
    case 2: return {9,10,11,12};
    case 3: return {13,14,15,16};
    case 4: return {17,18,19,20};
    case 5: return {21,22,23,24};
    default: assert(false); return {};
    }
}

// ================================================================
//  §3  端口 / GPU 映射  (Constants.py :: port_mapping, 测试文件中的连线)
// ================================================================

// port_mapping: 交换机间互联  (uplink port ↔ Spine3 port)
inline int portMapping(int port) {
    static const std::map<int,int> m = {
        {1,17},{3,21},{5,18},{7,22},{9,19},{11,23},{13,20},{15,24},
        {17,1},{21,3},{18,5},{22,7},{19,9},{23,11},{20,13},{24,15}
    };
    auto it = m.find(port);
    assert(it != m.end());
    return it->second;
}

// GPU ID → 端口号
inline int gpuToPort(int gpuId) {
    static const int t[] = {0, 2,4,6,8,10,12,14,16}; // 1-indexed
    return t[gpuId];
}

// 端口号 → GPU ID  (仅 GPU 下行端口有效)
inline int portToGpu(int port) {
    static const std::map<int,int> m = {
        {2,1},{4,2},{6,3},{8,4},{10,5},{12,6},{14,7},{16,8}
    };
    auto it = m.find(port);
    return (it != m.end()) ? it->second : 0;
}

// GPU ID → 所在 Tofino 索引 (0-5)
inline int gpuToTofinoIdx(int gpuId) {
    static const int t[] = {0, 0,0,1,1,2,2,3,3}; // 1-indexed
    return t[gpuId];
}

// 端口 → 所在 Tofino 索引
inline int portToTofinoIdx(int port) {
    if (port >= 1  && port <= 4)  return 0;
    if (port >= 5  && port <= 8)  return 1;
    if (port >= 9  && port <= 12) return 2;
    if (port >= 13 && port <= 16) return 3;
    if (port >= 17 && port <= 20) return 4;
    if (port >= 21 && port <= 24) return 5;
    assert(false); return -1;
}

// 端口是否连接 GPU (偶数端口 2-16)
inline bool isGpuPort(int port) { return portToGpu(port) != 0; }

// ================================================================
//  §4  MoePacket  (继承 csg-htsim Packet)
//
//  翻译自 GPU.py :: create_packet()
// ================================================================

class MoePacket : public Packet {
public:
    uint8_t  pktType   = 0;   // PKT_TYPE_DATA / PKT_TYPE_ACK
    uint64_t roundId   = 0;
    uint8_t  srcId     = 0;   // 1-8
    uint8_t  targetId  = 0;   // 1-8
    uint16_t fragId    = 0;   // 0..127
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

    // 创建数据包
    // flow: 发送方的 PacketFlow
    // route: 预计算的完整路径 (Route 对象必须在包生命周期内有效)
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
//  §5  TimerEvent — 通用定时回调
//
//  对应 cocotb:
//    await Timer(delay, 'ns')            → timer.arm(delay_ps, callback)
//    await with_timeout(..., timeout)    → timer.arm(timeout_ps, retransmit_cb)
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
//  §6  NodeStats  (Statics.py :: MoeStats + print_global_summary)
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
              << "        GLOBAL MOE COMMUNICATION STATS\n" << sep << "\n"
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
