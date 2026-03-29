/*
 * hoho_routing.h  —  OCS direct-only routing (no multi-hop)
 *
 * Architecture: 8 GPUs connected via 1 OCS switch.
 * Each GPU node has one EcsBuffer (Edge Circuit Switch Buffer).
 *
 * 路由策略 (与 ocs/ 的 BFS 多跳路由不同):
 *   - 只使用直连: 如果当前时隙直接连接目的地 → 发送
 *   - 否则等待: 缓存到 buffer, 标记 target_slice = 直连时隙
 *   - 没有中转, 没有多跳, 每个包只经过一次 OCS
 *
 * EcsBuffer 四个逻辑端口:
 *   gpu_rx: 从 GPU txQueue 接收 (via sendOn, route[0])
 *   gpu_tx: 交付本地 GPU (via pkt.sendOn → rxQueue → AppSink)
 *   ocs_tx: OCS 上行串行化队列 (200Gbps)
 *   ocs_rx: OCS 下行串行化队列 (200Gbps)
 *
 * 安全机制:
 *   processPacket: 检查剩余活动时间是否够发
 *   onSliceStart: 预算控制, 发不完的重新标记下一轮
 *   onOcsTxDone: 时隙变了则重新路由
 *
 * Components:
 *   DynOcsTopology  — 调度 (Circle Method) + 直连表
 *   SerDesPort      — 回调式串行化队列
 *   OcsLinkDelay    — 光纤传播延迟
 *   EcsBuffer       — 每节点直连路由 + 缓存
 *   SliceAlarm      — 时隙边界事件
 *
 * Route: [EcsBuffer@src, rxQueue@dst, AppSink@dst]
 */

#pragma once

#include "constants.h"
#include "hohoo_scheduler.h"

#include <cassert>
#include <deque>
#include <list>
#include <map>
#include <queue>
#include <string>
#include <vector>

// Forward declarations
class EcsBuffer;

// ================================================================
//  DynOcsTopology — schedule + direct connection table (no BFS)
//
//  Only stores:
//    _peer[node][slice] = connected peer
//    _connected[src][dst] = direct connection slice
//  No routing table computation.
// ================================================================

class DynOcsTopology {
public:
    DynOcsTopology(const HohooSchedule& sched,
                   simtime_picosec slice_active,
                   simtime_picosec reconfig)
        : _matchings(sched.matchings)
        , _schedule(sched.schedule)
        , _num_ports((int)NUM_GPU_NODES)
        , _num_slices((int)NUM_SLOTS)
        , _slice_active(slice_active)
        , _reconfig(reconfig)
        , _slice_total(slice_active + reconfig)
        , _cycle((simtime_picosec)NUM_SLOTS * (slice_active + reconfig))
    {
        // Build peer table: _peer[node][slice] = connected peer
        memset(_peer, 0, sizeof(_peer));
        _connected.assign(_num_ports + 1, std::vector<int>(_num_ports + 1, -1));

        for (int s = 0; s < _num_slices; s++) {
            for (auto& [a, b] : _matchings[s]) {
                _peer[a][s] = b;
                _peer[b][s] = a;
                _connected[a][b] = s;
                _connected[b][a] = s;
            }
        }
    }

    // ---- Time / slice queries ----

    int time_to_slice(simtime_picosec t) const {
        return (int)((t % _cycle) / _slice_total);
    }

    int time_to_absolute_slice(simtime_picosec t) const {
        return (int)(t / _slice_total);
    }

    simtime_picosec get_slice_start_time(int abs_slice) const {
        return (simtime_picosec)abs_slice * _slice_total;
    }

    bool is_in_reconfig(simtime_picosec t) const {
        simtime_picosec within = (t % _cycle) % _slice_total;
        return within >= _slice_active;
    }

    int effective_slice(simtime_picosec t) const {
        if (is_in_reconfig(t))
            return (time_to_slice(t) + 1) % _num_slices;
        return time_to_slice(t);
    }

    // ---- Connection queries ----

    int getPeer(int node, int slice) const {
        return _peer[node][slice];
    }

    // Which slice directly connects src to dst? (-1 if none)
    int get_slice_for_pair(int src, int dst) const {
        return _connected[src][dst];
    }

    int get_connected_target(int node, int slice) const {
        return _peer[node][slice];
    }

    const std::vector<std::pair<int,int>>& get_matching(int slice) const {
        return _matchings[slice];
    }

    // ---- Getters ----
    int num_slices() const { return _num_slices; }
    simtime_picosec slice_active() const { return _slice_active; }
    simtime_picosec reconfig_time() const { return _reconfig; }
    simtime_picosec slice_total() const { return _slice_total; }
    simtime_picosec cycle_time() const { return _cycle; }
    const std::map<int, std::vector<int>>& schedule() const { return _schedule; }

    // ---- Print connection table ----
    void printConnectionTable() const {
        std::cout << "OCS Direct Connection Table:\n";
        for (int src = 1; src <= _num_ports; src++) {
            for (int dst = 1; dst <= _num_ports; dst++) {
                if (src == dst) continue;
                std::cout << "  GPU" << src << " -> GPU" << dst
                          << ": direct in slice " << _connected[src][dst] << "\n";
            }
        }
        std::cout << "\n";
    }

private:
    std::vector<std::vector<std::pair<int,int>>> _matchings;
    std::map<int, std::vector<int>> _schedule;
    std::vector<std::vector<int>> _connected;  // [src][dst] -> direct slice
    int _peer[9][7];  // [node][slice] -> peer (1-indexed)

    int _num_ports;
    int _num_slices;
    simtime_picosec _slice_active;
    simtime_picosec _reconfig;
    simtime_picosec _slice_total;
    simtime_picosec _cycle;
};

// ================================================================
//  SerDesPort — serialization queue with callback or sendOn
// ================================================================

class SerDesPort : public EventSource, public PacketSink {
public:
    enum Mode { CALLBACK, SEND_ON };

    SerDesPort(const std::string& name, linkspeed_bps rate, mem_b maxbuf,
              Mode mode = CALLBACK)
        : EventSource(name)
        , _nodename(name)
        , _rate(rate)
        , _maxsize(maxbuf)
        , _mode(mode)
    {
        _ps_per_byte = (simtime_picosec)((double)8e12 / (double)rate);
    }

    void setCallback(std::function<void(Packet*)> cb) {
        _onComplete = std::move(cb);
    }

    void receivePacket(Packet& pkt) override {
        if (_queuesize + pkt.size() > _maxsize) {
            _drops++;
            pkt.free();
            return;
        }
        _queue.push_back(&pkt);
        _queuesize += pkt.size();
        if (_queuesize > _maxQueuesize)
            _maxQueuesize = _queuesize;

        if (!_transmitting)
            beginService();
    }

    void doNextEvent() override {
        assert(!_queue.empty());
        Packet* pkt = _queue.front();
        _queue.pop_front();
        _queuesize -= pkt->size();
        _transmitting = false;
        _totalPkts++;

        if (_mode == SEND_ON)
            pkt->sendOn();
        else if (_onComplete)
            _onComplete(pkt);

        if (!_queue.empty())
            beginService();
    }

    const std::string& nodename() override { return _nodename; }

    mem_b queuesize() const { return _queuesize; }
    mem_b maxQueuesize() const { return _maxQueuesize; }
    uint64_t drops() const { return _drops; }
    uint64_t totalPkts() const { return _totalPkts; }

private:
    void beginService() {
        if (_queue.empty()) return;
        Packet* pkt = _queue.front();
        simtime_picosec drain = (simtime_picosec)pkt->size() * _ps_per_byte;
        if (drain == 0) drain = 1;
        _transmitting = true;
        EventList::sourceIsPending(*this, EventList::now() + drain);
    }

    std::string _nodename;
    linkspeed_bps _rate;
    mem_b _maxsize;
    mem_b _queuesize = 0;
    mem_b _maxQueuesize = 0;
    simtime_picosec _ps_per_byte;
    std::list<Packet*> _queue;
    bool _transmitting = false;
    Mode _mode;
    std::function<void(Packet*)> _onComplete;
    uint64_t _totalPkts = 0;
    uint64_t _drops = 0;
};

// ================================================================
//  OcsLinkDelay — optical fiber propagation delay
// ================================================================

class OcsLinkDelay : public EventSource {
public:
    OcsLinkDelay(const std::string& name, simtime_picosec delay)
        : EventSource(name)
        , _delay(delay)
    {}

    void push(Packet* pkt, PacketSink* receiver) {
        simtime_picosec due = EventList::now() + _delay;
        bool wasEmpty = _delayed.empty();
        _delayed.push_back({due, pkt, receiver});
        if (wasEmpty)
            EventList::sourceIsPending(*this, due);
    }

    void doNextEvent() override {
        simtime_picosec now = EventList::now();
        while (!_delayed.empty() && _delayed.front().due <= now) {
            auto entry = _delayed.front();
            _delayed.pop_front();
            entry.receiver->receivePacket(*entry.pkt);
        }
        if (!_delayed.empty())
            EventList::sourceIsPending(*this, _delayed.front().due);
    }

private:
    struct Delayed {
        simtime_picosec due;
        Packet* pkt;
        PacketSink* receiver;
    };
    std::deque<Delayed> _delayed;
    simtime_picosec _delay;
};

// ================================================================
//  EcsBuffer — Edge Circuit Switch Buffer (direct-only routing)
//
//  路由策略 (无多跳):
//    dst == me       → LOCAL: pkt.sendOn()
//    peer == dst     → DIRECT: 当前时隙直连目的地, 立即发送
//    peer != dst     → WAIT: 标记 target_slice = 直连时隙, 缓存
//
//  onSliceStart:
//    当前时隙连接 peer, 从 buffer 取出 dst==peer 的包发送
//    预算控制: 发不完的重新标记等下一轮
// ================================================================

class EcsBuffer : public PacketSink {
public:
    EcsBuffer(int nodeId, DynOcsTopology* topo)
        : _nodeId(nodeId)
        , _topo(topo)
        , _nodename("ecs_" + std::to_string(nodeId))
    {
        _ps_per_byte = (simtime_picosec)((double)8e12 / (double)LINK_SPEED_BPS);

        // OCS TX serialization port (200 Gbps)
        _ocs_tx = new SerDesPort(
            "ocs_tx_" + std::to_string(nodeId),
            LINK_SPEED_BPS,
            OCS_TX_BUFFER
        );
        _ocs_tx->setCallback([this](Packet* pkt) { onOcsTxDone(pkt); });

        // OCS RX serialization port (200 Gbps)
        _ocs_rx = new SerDesPort(
            "ocs_rx_" + std::to_string(nodeId),
            LINK_SPEED_BPS,
            OCS_TX_BUFFER
        );
        _ocs_rx->setCallback([this](Packet* pkt) { processPacket(pkt); });

        // OCS link delay (100 ns propagation)
        _ocsLink = new OcsLinkDelay(
            "ocs_link_" + std::to_string(nodeId),
            OCS_LINK_DELAY_PS
        );
    }

    // ---- PacketSink interface ----
    // Only receives from GPU txQueue (via route sendOn)
    void receivePacket(Packet& pkt) override {
        processPacket(&pkt);
    }

    // ---- OCS RX port accessor ----
    SerDesPort* ocsRxPort() { return _ocs_rx; }

    // ---- Slice boundary callback ----
    void onSliceStart(int new_slice) {
        int peer = _topo->getPeer(_nodeId, new_slice);
        if (peer <= 0) return;  // no connection this slice

        // Calculate safe send budget
        simtime_picosec now = EventList::now();
        simtime_picosec remaining = sliceRemainingActive(now);
        simtime_picosec ocs_tx_drain = (simtime_picosec)_ocs_tx->queuesize()
                                       * _ps_per_byte;
        simtime_picosec available = (remaining > ocs_tx_drain)
                                    ? (remaining - ocs_tx_drain) : 0;
        mem_b budget = (mem_b)(available / _ps_per_byte);

        // Scan buffer: forward packets whose target_slice matches (dst == peer)
        std::vector<Packet*> to_forward;
        auto it = _buffer.begin();
        while (it != _buffer.end()) {
            MoePacket* mpkt = static_cast<MoePacket*>(*it);
            if ((int)mpkt->target_slice == new_slice) {
                if ((mem_b)mpkt->size() > budget) {
                    // Budget exhausted — this packet waits for next cycle
                    // target_slice stays the same (same direct slice next cycle)
                    ++it;
                    continue;
                }
                budget -= (mem_b)mpkt->size();
                _bufferSize -= mpkt->size();
                to_forward.push_back(*it);
                it = _buffer.erase(it);
            } else {
                ++it;
            }
        }

        if (!to_forward.empty()) {
            for (Packet* pkt : to_forward) {
                forwardToOcs(pkt, peer);
            }
        }
    }

    // ---- Wiring ----
    void setPeerBuffer(int peer, EcsBuffer* buf) {
        _peerBuffers[peer] = buf;
    }

    const std::string& nodename() override { return _nodename; }

    // ---- Stats ----
    uint64_t localCount()   const { return _localCount; }
    uint64_t directCount()  const { return _directCount; }
    uint64_t waitCount()    const { return _waitCount; }
    uint64_t drops()        const { return _drops; }
    uint64_t misroutes()    const { return _misroutes; }
    mem_b    maxBufferUsed() const { return _maxBufferUsed; }
    uint64_t ocsTxDrops()   const { return _ocs_tx->drops(); }
    mem_b    ocsTxMaxQueue() const { return _ocs_tx->maxQueuesize(); }
    uint64_t ocsTxPkts()    const { return _ocs_tx->totalPkts(); }
    uint64_t ocsRxDrops()   const { return _ocs_rx->drops(); }
    mem_b    ocsRxMaxQueue() const { return _ocs_rx->maxQueuesize(); }
    uint64_t ocsRxPkts()    const { return _ocs_rx->totalPkts(); }
    size_t   bufferPkts()   const { return _buffer.size(); }

private:
    // ---- Calculate remaining active time in current slice ----
    simtime_picosec sliceRemainingActive(simtime_picosec now) const {
        int abs_slice = _topo->time_to_absolute_slice(now);
        simtime_picosec slice_start = _topo->get_slice_start_time(abs_slice);
        simtime_picosec active_end = slice_start + _topo->slice_active();
        if (now >= active_end) return 0;
        return active_end - now;
    }

    // ---- Check if enough time to send a packet in current slice ----
    bool canSendInSlice(simtime_picosec now, mem_b pkt_size) const {
        simtime_picosec remaining = sliceRemainingActive(now);
        if (remaining == 0) return false;
        simtime_picosec ocs_tx_drain = (simtime_picosec)_ocs_tx->queuesize()
                                       * _ps_per_byte;
        simtime_picosec pkt_drain = (simtime_picosec)pkt_size * _ps_per_byte;
        return (ocs_tx_drain + pkt_drain) <= remaining;
    }

    // ---- Routing logic (direct-only, no multi-hop) ----
    void processPacket(Packet* pkt) {
        MoePacket* mpkt = static_cast<MoePacket*>(pkt);
        int dst = (int)mpkt->targetId;

        // Local delivery
        if (dst == _nodeId) {
            _localCount++;
            pkt->sendOn();  // → rxQueue@dst → AppSink@dst
            return;
        }

        simtime_picosec now = EventList::now();
        bool in_reconfig = _topo->is_in_reconfig(now);
        int slice = _topo->effective_slice(now);
        int peer = _topo->getPeer(_nodeId, slice);

        // Direct connection: current slice's peer IS the destination
        if (peer == dst && !in_reconfig
            && canSendInSlice(now, (mem_b)pkt->size())) {
            _directCount++;
            forwardToOcs(pkt, peer);
        } else {
            // WAIT: buffer until the direct connection slice
            _waitCount++;
            int direct_slice = _topo->get_slice_for_pair(_nodeId, dst);

            // If direct slice is current but can't send (reconfig or no budget),
            // still tag with direct_slice — will be picked up next cycle
            mpkt->target_slice = (int8_t)direct_slice;

            // Buffer overflow check
            if (_bufferSize + pkt->size() > ECS_BUFFER_SIZE) {
                _drops++;
                pkt->free();
                return;
            }
            _buffer.push_back(pkt);
            _bufferSize += pkt->size();
            if (_bufferSize > _maxBufferUsed)
                _maxBufferUsed = _bufferSize;
        }
    }

    // ---- Forward packet through OCS ----
    void forwardToOcs(Packet* pkt, int peer) {
        static_cast<MoePacket*>(pkt)->target_slice = -1;
        _ocs_tx_nexthop.push(peer);
        _ocs_tx->receivePacket(*pkt);
    }

    // ---- Callback when ocs_tx finishes serialization ----
    void onOcsTxDone(Packet* pkt) {
        assert(!_ocs_tx_nexthop.empty());
        int nh = _ocs_tx_nexthop.front();
        _ocs_tx_nexthop.pop();

        // Safety check: is the current slice's peer still nh?
        simtime_picosec now = EventList::now();
        bool in_reconfig = _topo->is_in_reconfig(now);
        if (in_reconfig) {
            _misroutes++;
            processPacket(pkt);
            return;
        }
        int current_slice = _topo->time_to_slice(now);
        int current_peer = _topo->getPeer(_nodeId, current_slice);
        if (current_peer != nh) {
            _misroutes++;
            processPacket(pkt);
            return;
        }

        // Safe to send: deliver to peer's ocs_rx
        auto it = _peerBuffers.find(nh);
        assert(it != _peerBuffers.end() && "Missing peer buffer");
        _ocsLink->push(pkt, it->second->ocsRxPort());
    }

    int _nodeId;
    DynOcsTopology* _topo;
    std::string _nodename;
    simtime_picosec _ps_per_byte;

    // OCS TX port + RX port + link
    SerDesPort* _ocs_tx;
    SerDesPort* _ocs_rx;
    OcsLinkDelay* _ocsLink;
    std::map<int, EcsBuffer*> _peerBuffers;
    std::queue<int> _ocs_tx_nexthop;

    // Internal buffer (WAIT packets with target_slice = direct connection slice)
    std::list<Packet*> _buffer;
    mem_b _bufferSize = 0;
    mem_b _maxBufferUsed = 0;

    // Stats
    uint64_t _localCount   = 0;
    uint64_t _directCount  = 0;  // sent via direct connection
    uint64_t _waitCount    = 0;
    uint64_t _drops        = 0;
    uint64_t _misroutes    = 0;
};

// ================================================================
//  SliceAlarm — fires at every slice boundary
// ================================================================

class SliceAlarm : public EventSource {
public:
    SliceAlarm(DynOcsTopology* topo)
        : EventSource("SliceAlarm")
        , _topo(topo)
    {
        EventList::sourceIsPending(*this, 0);
    }

    void addBuffer(EcsBuffer* buf) { _buffers.push_back(buf); }

    void doNextEvent() override {
        simtime_picosec now = EventList::now();
        int slice = _topo->time_to_slice(now);

        for (auto* buf : _buffers)
            buf->onSliceStart(slice);

        _totalActivations++;

        // Schedule next slice boundary
        int abs = _topo->time_to_absolute_slice(now);
        simtime_picosec next = _topo->get_slice_start_time(abs + 1);
        if (next <= now) next = now + 1;
        EventList::sourceIsPending(*this, next);
    }

    uint64_t totalActivations() const { return _totalActivations; }

private:
    DynOcsTopology* _topo;
    std::vector<EcsBuffer*> _buffers;
    uint64_t _totalActivations = 0;
};
