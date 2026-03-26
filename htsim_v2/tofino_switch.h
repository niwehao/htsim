/*
 * tofino_switch.h  —  csg-htsim 版
 *
 * 翻译自: Tofino.py + Spine.py (路由部分)
 *
 * 使用 csg-htsim 的 source-routing:
 *   每个 MoePacket 在创建时就携带完整 Route (PacketSink* 序列)
 *   Queue::completeService() 调用 pkt->sendOn() 自动推进到 Route 中下一跳
 *
 * 每个 Tofino 在 Route 中占 5 个节点:
 *
 *   ingressQueue[port_in]          ← 入端口 25 Gbps 串行化
 *        ↓ sendOn()
 *   BufferGate                     ← 32 MB 缓冲检查 + 2ms 拥塞延迟
 *        ↓ sendOn() / delayed sendOn()
 *   fabricQueue                    ← 100 Gbps 内部转发串行化
 *        ↓ sendOn()
 *   BufferRelease                  ← 释放缓冲计数
 *        ↓ sendOn()
 *   egressQueue[port_out]          ← 出端口 25 Gbps 串行化
 *        ↓ sendOn()  → 下一个 Tofino 的 ingressQueue 或 GPU 的 rxQueue
 *
 * 延迟对应:
 *   入端口处理包  = ingressQueue (25 Gbps)  ← Tofino._port_worker_in
 *   内部转发处理包 = fabricQueue  (100 Gbps) ← Tofino._transfer_worker
 *   出端口处理包  = egressQueue  (25 Gbps)  ← Tofino._port_worker_out
 */

#pragma once
#include "constants.h"
#include <deque>

// ================================================================
//  BufferGate
//
//  翻译自: Tofino.add_queue_transfer()
//
//  Route 中 ingressQueue → BufferGate → fabricQueue 之间的中间节点
//  功能:
//    1. 检查 32 MB 缓冲上限 → 超出则 DROP (free 包)
//    2. 缓冲为空 (仅本包) → 直接 sendOn() 进入 fabricQueue
//    3. 缓冲非空 → 延迟 2ms 后 sendOn()
//
//  因为 sendOn() 由包自身的 Route 驱动, BufferGate 不需要知道
//  fabricQueue 的指针 — 只需在正确的时间对包调用 sendOn()
// ================================================================

class BufferGate : public EventSource, public PacketSink {
public:
    BufferGate(const std::string& name)
        : EventSource(name), _nodename(name) {}

    // ---- PacketSink: 从 ingressQueue 的 sendOn() 到达 ----
    void receivePacket(Packet& pkt) override {
        int sz = pkt.size();

        // 1. 溢出检查
        if (_bufUsage + sz > SW_QUEUE_SIZE_BYTES) {
            std::cout << "[" << _nodename << "] DROP: buffer overflow ("
                      << _bufUsage << "+" << sz << ">" << SW_QUEUE_SIZE_BYTES << ")\n";
            pkt.free();
            return;
        }

        _bufUsage += sz;

        // 2. 缓冲为空 → 直接转发 (sendOn 进入 Route 下一跳 = fabricQueue)
        if (_bufUsage == sz) {
            pkt.sendOn();
            return;
        }

        // 3. 缓冲非空 → 延迟 2ms
        _delayed.push_back({EventList::now() + BUFFER_DELAY_PS, &pkt});
        // 注册最早到期时间
        EventList::sourceIsPending(*this, _delayed.front().due);
    }

    // ---- EventSource: 延迟到期 → sendOn() ----
    void doNextEvent() override {
        simtime_picosec now = EventList::now();
        while (!_delayed.empty() && _delayed.front().due <= now) {
            Packet* pkt = _delayed.front().pkt;
            _delayed.pop_front();
            pkt->sendOn();
        }
        if (!_delayed.empty())
            EventList::sourceIsPending(*this, _delayed.front().due);
    }

    // 由 BufferRelease 调用: fabricQueue 处理完成后释放缓冲
    void releaseBuffer(int sz) { _bufUsage -= sz; }

    const std::string& nodename() override { return _nodename; }

private:
    struct Delayed { simtime_picosec due; Packet* pkt; };
    std::deque<Delayed> _delayed;
    int64_t     _bufUsage = 0;
    std::string _nodename;
};

// ================================================================
//  BufferRelease
//
//  翻译自: Tofino._transfer_worker 中 fabric 串行化完成后:
//    self.current_buffer_usage -= pkt_size_bytes
//
//  Route 中位于 fabricQueue 与 egressQueue 之间:
//    fabricQueue → sendOn() → BufferRelease → sendOn() → egressQueue
// ================================================================

class BufferRelease : public PacketSink {
public:
    BufferRelease(BufferGate* gate, const std::string& name)
        : _gate(gate), _nodename(name) {}

    void receivePacket(Packet& pkt) override {
        _gate->releaseBuffer(pkt.size());
        pkt.sendOn();  // 继续 Route: → egressQueue
    }

    const std::string& nodename() override { return _nodename; }

private:
    BufferGate* _gate;
    std::string _nodename;
};

// ================================================================
//  TofinoSwitch
//
//  翻译自: Tofino 类
//
//  管理一个 leaf 的全部队列组件.
//  Route 构建时使用 getRouteNodes() 获取经过本 Tofino 的 5 个 PacketSink*
// ================================================================

class TofinoSwitch {
public:
    // idx: Tofino 全局索引 (0-5)
    TofinoSwitch(int idx) : _idx(idx) {
        std::string pfx = "tof" + std::to_string(idx);
        auto ports = tofinoPorts(idx);

        // fabricQueue: 100 Gbps 共享内部带宽
        _fabricQueue = new Queue(FABRIC_SPEED_BPS, HUGE_BUFFER,
                                 EventList::getTheEventList(), nullptr);

        // BufferGate + BufferRelease
        _bufferGate    = new BufferGate(pfx + "_bufGate");
        _bufferRelease = new BufferRelease(_bufferGate, pfx + "_bufRel");

        // 每端口 ingress / egress queue (25 Gbps)
        for (int p : ports) {
            _ingressQ[p] = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                                      EventList::getTheEventList(), nullptr);
            _egressQ[p]  = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                                      EventList::getTheEventList(), nullptr);
        }
    }

    // ---- Route 构建接口 ----

    // 返回经过本 Tofino 从 inPort 到 outPort 的 5 个 Route 节点
    // 调用方按顺序 push_back 到 Route 中:
    //   route.push_back(ingressQ[inPort])
    //   route.push_back(bufferGate)
    //   route.push_back(fabricQ)
    //   route.push_back(bufferRelease)
    //   route.push_back(egressQ[outPort])
    void appendToRoute(Route& route, int inPort, int outPort) const {
        route.push_back(_ingressQ.at(inPort));
        route.push_back(_bufferGate);
        route.push_back(_fabricQueue);
        route.push_back(_bufferRelease);
        route.push_back(_egressQ.at(outPort));
    }

    // 获取指定端口的 ingress queue (用于 GPU txQueue 的 Route 首跳)
    Queue* ingressQueue(int port) const { return _ingressQ.at(port); }

    // 获取指定端口的 egress queue (用于连接下游 GPU rxQueue)
    Queue* egressQueue(int port) const { return _egressQ.at(port); }

    int idx() const { return _idx; }

private:
    int _idx;
    std::map<int, Queue*> _ingressQ;    // port → 25G ingress
    std::map<int, Queue*> _egressQ;     // port → 25G egress
    Queue*                _fabricQueue;  // 100G fabric
    BufferGate*           _bufferGate;
    BufferRelease*        _bufferRelease;
};
