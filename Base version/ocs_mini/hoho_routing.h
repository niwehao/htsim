/*
 * hoho_routing.h  —  EcsBuffer-based HOHO routing for OCS
 *
 * Architecture: 8 GPUs connected via 1 OCS switch.
 * Each GPU node has one EcsBuffer (Edge Circuit Switch Buffer).
 *
 * EcsBuffer 四个逻辑端口:
 *   gpu_rx: 从 GPU txQueue 接收 (via sendOn, route[0])
 *   gpu_tx: 交付本地 GPU (via pkt.sendOn → rxQueue → AppSink)
 *   ocs_tx: OCS 上行串行化队列 (200Gbps)
 *   ocs_rx: 从 OCS 接收 (via OcsLinkDelay)
 *
 * 路由表: rt[node][dst][slice] = FORWARD / WAIT
 *   BFS 在时间展开图 (8 nodes × 7 slices = 56 states) 上计算
 *   FORWARD: 发给当前时隙的对端
 *   WAIT: 缓存, 标记 target_slice
 *
 * 每个时隙边界: SliceAlarm → EcsBuffer.onSliceStart
 *   遍历 buffer, 匹配 target_slice 的包转发
 *
 * Components:
 *   DynOcsTopology  — 调度 + BFS 路由表
 *   SerDesPort      — 回调式串行化队列
 *   OcsLinkDelay    — 光纤传播延迟 (per-packet 目的地)
 *   EcsBuffer       — 每节点路由 + 缓存
 *   SliceAlarm      — 时隙边界事件
 *
 * Data flow (2-hop: GPU_A → GPU_B → GPU_D):
 *   GPU_A.txQ → sendOn → EcsBuffer@A → ocs_tx → OcsLinkDelay
 *     → EcsBuffer@B → ocs_tx → OcsLinkDelay → EcsBuffer@D
 *     → sendOn → rxQ_D → AppSink_D
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
//  DynOcsTopology — BFS routing on time-expanded graph
//
//  Time-expanded graph: 8 nodes × 7 slices = 56 states
//  Edges from (node, s):
//    WAIT:    (node, s) → (node, (s+1)%7)  cost 1
//    FORWARD: (node, s) → (peer(node,s), (s+1)%7)  cost 1
//
//  For each dst, reverse BFS from all (dst, s) states.
//  Result: rt[node][dst][slice] = FORWARD or WAIT
// ================================================================

class DynOcsTopology {
public:
    enum Action { FORWARD = 0, WAIT = 1 };

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

        computeRoutingTable();
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

    // Effective slice for routing decisions (skips reconfig gap)
    int effective_slice(simtime_picosec t) const {
        if (is_in_reconfig(t))
            return (time_to_slice(t) + 1) % _num_slices;
        return time_to_slice(t);
    }

    // ---- Routing queries ----

    Action getAction(int node, int dst, int slice) const {
        return _rt[node][dst][slice];
    }

    int getDistance(int node, int dst, int slice) const {
        return _dist[node][dst][slice];
    }

    int getPeer(int node, int slice) const {
        return _peer[node][slice];
    }

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

    // ---- Print routing table ----
    void printRoutingTable() const {
        std::cout << "HOHO Routing Table (BFS on time-expanded graph):\n";
        int fwd_count = 0, wait_count = 0;
        int max_dist = 0;

        for (int node = 1; node <= _num_ports; node++) {
            for (int dst = 1; dst <= _num_ports; dst++) {
                if (node == dst) continue;
                for (int s = 0; s < _num_slices; s++) {
                    if (_rt[node][dst][s] == FORWARD) fwd_count++;
                    else wait_count++;
                    if (_dist[node][dst][s] > max_dist)
                        max_dist = _dist[node][dst][s];
                }
            }
        }

        std::cout << "  FORWARD entries: " << fwd_count
                  << "  WAIT entries: " << wait_count
                  << "  Max distance: " << max_dist << " slots\n";

        // Show routes for all (src, dst) pairs
        std::cout << "  Sample routing decisions:\n";
        for (int src = 1; src <= std::min(4, _num_ports); src++) {
            for (int dst = 1; dst <= _num_ports; dst++) {
                if (src == dst) continue;
                std::cout << "    " << src << "->" << dst << ":";
                for (int s = 0; s < _num_slices; s++) {
                    if (_rt[src][dst][s] == FORWARD)
                        std::cout << " s" << s << "=FWD(" << _peer[src][s]
                                  << ",d" << _dist[src][dst][s] << ")";
                    else
                        std::cout << " s" << s << "=WAIT(d"
                                  << _dist[src][dst][s] << ")";
                }
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }

private:
    // ---- BFS routing table computation ----
    void computeRoutingTable() {
        // Initialize all to WAIT with high distance
        for (int n = 0; n <= _num_ports; n++)
            for (int d = 0; d <= _num_ports; d++)
                for (int s = 0; s < _num_slices; s++) {
                    _rt[n][d][s] = WAIT;
                    _dist[n][d][s] = 999;
                }

        // BFS for each destination
        for (int dst = 1; dst <= _num_ports; dst++)
            bfsForDst(dst);
    }

    // Reverse BFS from destination: find optimal action at every (node, slice)
    void bfsForDst(int dst) {
        // Local distance array for this BFS
        int d[9][7];
        memset(d, -1, sizeof(d));

        std::queue<std::pair<int,int>> q;

        // Initialize: all (dst, s) states have distance 0
        for (int s = 0; s < _num_slices; s++) {
            d[dst][s] = 0;
            _dist[dst][dst][s] = 0;
            q.push({dst, s});
        }

        while (!q.empty()) {
            auto [node, s] = q.front();
            q.pop();
            int cost = d[node][s];
            int prev_s = (s - 1 + _num_slices) % _num_slices;

            // Reverse of WAIT edge: (node, prev_s) --WAIT--> (node, s)
            // At (node, prev_s), waiting one slot leads to (node, s)
            if (d[node][prev_s] == -1) {
                d[node][prev_s] = cost + 1;
                _rt[node][dst][prev_s] = WAIT;
                _dist[node][dst][prev_s] = cost + 1;
                q.push({node, prev_s});
            }

            // Reverse of FORWARD edge: (x, prev_s) --FWD--> (node, s)
            // where x is connected to node at slice prev_s
            // i.e., peer(node, prev_s) == x, and peer(x, prev_s) == node
            int x = _peer[node][prev_s];
            if (x > 0 && d[x][prev_s] == -1) {
                d[x][prev_s] = cost + 1;
                _rt[x][dst][prev_s] = FORWARD;
                _dist[x][dst][prev_s] = cost + 1;
                q.push({x, prev_s});
            }
        }
    }

    // Schedule data
    std::vector<std::vector<std::pair<int,int>>> _matchings;
    std::map<int, std::vector<int>> _schedule;
    std::vector<std::vector<int>> _connected;  // [src][dst] -> direct slice
    int _peer[9][7];  // [node][slice] -> peer (1-indexed)

    // BFS routing table (1-indexed nodes)
    Action _rt[9][9][7];   // [node][dst][slice] = FORWARD or WAIT
    int    _dist[9][9][7]; // [node][dst][slice] = distance in slots

    int _num_ports;
    int _num_slices;
    simtime_picosec _slice_active;
    simtime_picosec _reconfig;
    simtime_picosec _slice_total;
    simtime_picosec _cycle;
};

// ================================================================
//  SerDesPort — serialization queue with callback or sendOn
//
//  Replaces Queue for cases where we need custom delivery:
//    CALLBACK mode: after serialization, invoke _onComplete(pkt)
//    SEND_ON mode:  after serialization, call pkt->sendOn()
//
//  Each packet is serialized at the configured link rate.
//  Packets are served FIFO.
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
//
//  Per-packet destination: each pushed packet carries its receiver.
//  After the configured delay, delivers to the specified PacketSink.
//  Uses batched event scheduling (single event for queue head).
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
//  EcsBuffer — Edge Circuit Switch Buffer (one per node)
//
//  核心组件: 替代原来的 56 个 OcsCircuit + 8 个 HohoForwarder
//  每个节点一个 EcsBuffer, 物理上只有一个 OCS 端口
//
//  接收来源:
//    1. GPU txQueue → sendOn → EcsBuffer.receivePacket (route[0])
//    2. OcsLinkDelay → EcsBuffer._ocs_rx → processPacket (中间跳)
//
//  四个串行化端口:
//    gpu_rx: GPU txQueue 已串行化, EcsBuffer 直接 processPacket
//    gpu_tx: rxQueue 已串行化, EcsBuffer 直接 sendOn
//    ocs_tx: SerDesPort 200Gbps, FORWARD 发送方向
//    ocs_rx: SerDesPort 200Gbps, 从光纤接收方向 (新增)
//
//  路由决策 (processPacket):
//    dst == me  → LOCAL: pkt.sendOn() → rxQueue → AppSink
//    FORWARD    → 安全检查后发到 ocs_tx → OcsLinkDelay → 对端 ocs_rx
//    WAIT       → 标记 target_slice, 存入 buffer
//
//  时隙边界 (onSliceStart):
//    计算安全发送预算, 遍历 buffer, 匹配且预算内的包转发
//
//  安全机制:
//    processPacket FORWARD: 检查剩余活动时间是否够发
//    onSliceStart: 预算控制, 发不完的留在 buffer
//    onOcsTxDone: 时隙变了则重新路由 (不发给错误对端)
// ================================================================

class EcsBuffer : public PacketSink {
public:
    EcsBuffer(int nodeId, DynOcsTopology* topo)
        : _nodeId(nodeId)
        , _topo(topo)
        , _nodename("ecs_" + std::to_string(nodeId))
    {
        _ps_per_byte = (simtime_picosec)((double)8e12 / (double)LINK_SPEED_BPS);

        // OCS TX serialization port (200 Gbps) — 发送方向
        _ocs_tx = new SerDesPort(
            "ocs_tx_" + std::to_string(nodeId),
            LINK_SPEED_BPS,
            OCS_TX_BUFFER
        );
        _ocs_tx->setCallback([this](Packet* pkt) { onOcsTxDone(pkt); });

        // OCS RX serialization port (200 Gbps) — 接收方向 (从光纤)
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
    // OCS direction goes through _ocs_rx (SerDesPort) instead
    void receivePacket(Packet& pkt) override {
        processPacket(&pkt);
    }

    // ---- OCS RX port accessor (for OcsLinkDelay to deliver to) ----
    SerDesPort* ocsRxPort() { return _ocs_rx; }

    // ---- Slice boundary callback ----
    void onSliceStart(int new_slice) {
        // Calculate safe send budget for this slice
        simtime_picosec now = EventList::now();
        simtime_picosec remaining = sliceRemainingActive(now);

        // Budget = remaining time - time to drain ocs_tx queue
        simtime_picosec ocs_tx_drain = (simtime_picosec)_ocs_tx->queuesize()
                                       * _ps_per_byte;
        simtime_picosec available = (remaining > ocs_tx_drain)
                                    ? (remaining - ocs_tx_drain) : 0;//只要进入光纤就不受时隙限制了
        mem_b budget = (mem_b)(available / _ps_per_byte);

        // Scan buffer: forward packets whose target_slice matches, within budget
        std::vector<Packet*> to_forward;
        auto it = _buffer.begin();
        while (it != _buffer.end()) {
            MoePacket* mpkt = static_cast<MoePacket*>(*it);
            if ((int)mpkt->target_slice == new_slice) {
                if ((mem_b)mpkt->size() > budget) {
                    // No more budget — leave remaining matched packets in buffer
                    // They keep target_slice, will be picked up next cycle
                    ++it;
                    continue;// 也许可以再次查表，看看下一个时隙能不能发,否则需要等待一轮次 TODO
                }
                budget -= (mem_b)mpkt->size();
                _bufferSize -= mpkt->size();
                to_forward.push_back(*it);
                it = _buffer.erase(it);
            } else {
                ++it;//也许可以再次查表
            }
        }

        if (!to_forward.empty()) {
            int peer = _topo->getPeer(_nodeId, new_slice);
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
    uint64_t forwardCount() const { return _forwardCount; }
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

    // ---- Routing logic ----
    void processPacket(Packet* pkt) {
        MoePacket* mpkt = static_cast<MoePacket*>(pkt);
        int dst = (int)mpkt->targetId;

        // Local delivery: packet destination is this node
        if (dst == _nodeId) {
            _localCount++;
            pkt->sendOn();  // → rxQueue@dst → AppSink@dst
            return;
        }

        simtime_picosec now = EventList::now();
        bool in_reconfig = _topo->is_in_reconfig(now);
        int slice = _topo->effective_slice(now);
        auto action = _topo->getAction(_nodeId, dst, slice);

        if (action == DynOcsTopology::FORWARD && !in_reconfig
            && canSendInSlice(now, (mem_b)pkt->size())) {
            // Forward immediately via OCS to current peer (safe to send)
            _forwardCount++;
            int peer = _topo->getPeer(_nodeId, slice);
            forwardToOcs(pkt, peer);
        } else {
            // WAIT: tag with target_slice and buffer
            _waitCount++;
            int ts;
            if (action == DynOcsTopology::FORWARD && in_reconfig) {
                // Should forward this slice, but in reconfig — wait for slice start
                ts = slice;
            } else if (action == DynOcsTopology::FORWARD && !in_reconfig) {
                // Should forward but not enough time — wait for this slice next cycle
                ts = slice;
            } else {
                // WAIT action: find next slice where FORWARD
                ts = computeTargetSlice(dst, slice);
            }
            mpkt->target_slice = (int8_t)ts;

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
        // Clear old target_slice tag (will be recomputed at next node)
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
            // In reconfig gap — can't send, re-route
            _misroutes++;
            processPacket(pkt);
            return;
        }
        int current_slice = _topo->time_to_slice(now);
        int current_peer = _topo->getPeer(_nodeId, current_slice);
        if (current_peer != nh) {
            // Slice changed, peer mismatch — re-route
            _misroutes++;
            processPacket(pkt);
            return;
        }

        // Safe to send: deliver to peer's ocs_rx (with serialization)
        auto it = _peerBuffers.find(nh);
        assert(it != _peerBuffers.end() && "Missing peer buffer");
        _ocsLink->push(pkt, it->second->ocsRxPort());
    }

    // ---- Find next FORWARD slice from current_slice ----
    int computeTargetSlice(int dst, int current_slice) {
        int ns = _topo->num_slices();
        for (int offset = 1; offset <= ns; offset++) {
            int s = (current_slice + offset) % ns;
            if (_topo->getAction(_nodeId, dst, s) == DynOcsTopology::FORWARD)
                return s;
        }
        // Fallback: shouldn't happen (every pair has a direct connection)
        return _topo->get_slice_for_pair(_nodeId, dst);
    }

    int _nodeId;
    DynOcsTopology* _topo;
    std::string _nodename;
    simtime_picosec _ps_per_byte;

    // OCS TX port (sending to OCS) + RX port (receiving from OCS) + link
    SerDesPort* _ocs_tx;
    SerDesPort* _ocs_rx;
    OcsLinkDelay* _ocsLink;
    std::map<int, EcsBuffer*> _peerBuffers;
    std::queue<int> _ocs_tx_nexthop;

    // Internal buffer (WAIT packets with target_slice tags)
    std::list<Packet*> _buffer;
    mem_b _bufferSize = 0;
    mem_b _maxBufferUsed = 0;

    // Stats
    uint64_t _localCount   = 0;
    uint64_t _forwardCount = 0;
    uint64_t _waitCount    = 0;
    uint64_t _drops        = 0;
    uint64_t _misroutes    = 0;
};

// ================================================================
//  SliceAlarm — fires at every slice boundary
//
//  At each slice start, notifies all EcsBuffers to scan their
//  buffers and forward packets with matching target_slice.
// ================================================================

class SliceAlarm : public EventSource {
public:
    SliceAlarm(DynOcsTopology* topo)
        : EventSource("SliceAlarm")
        , _topo(topo)
    {
        // Schedule first event at time 0
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
