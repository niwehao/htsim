/*
 * hoho_routing.h  —  Scalable BFS multi-hop routing (max 5 hops)
 *
 * Dynamic arrays for N-port OCS (no static array limits).
 * BFS on time-expanded graph: N nodes × (N-1) slices.
 * Hop limit: if no path within 5 hops, WAIT for direct connection.
 * GPU↔EcsBuffer transfer is instant (handled by caller).
 * No internal buffer size limit.
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

class EcsBuffer;

// ================================================================
//  DynOcsTopology — BFS routing on time-expanded graph (5-hop limit)
// ================================================================

class DynOcsTopology {
public:
    enum Action : uint8_t { FORWARD = 0, WAIT = 1 };
    static constexpr int MAX_HOPS = 16;

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
        int N = _num_ports;
        int S = _num_slices;

        // Allocate peer table: [node][slice] (1-indexed nodes)
        _peer.assign(N + 1, std::vector<int>(S, 0));
        _connected.assign(N + 1, std::vector<int>(N + 1, -1));

        for (int s = 0; s < S; s++) {
            for (auto& [a, b] : _matchings[s]) {
                _peer[a][s] = b;
                _peer[b][s] = a;
                _connected[a][b] = s;
                _connected[b][a] = s;
            }
        }

        // Allocate routing table: [node][dst][slice]
        _rt.assign(N + 1, std::vector<std::vector<uint8_t>>(
            N + 1, std::vector<uint8_t>(S, WAIT)));
        _dist.assign(N + 1, std::vector<std::vector<int16_t>>(
            N + 1, std::vector<int16_t>(S, -1)));
        _fwd_hops.assign(N + 1, std::vector<std::vector<uint8_t>>(
            N + 1, std::vector<uint8_t>(S, 255)));

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

    int effective_slice(simtime_picosec t) const {
        if (is_in_reconfig(t))
            return (time_to_slice(t) + 1) % _num_slices;
        return time_to_slice(t);
    }

    // ---- Routing queries ----

    Action getAction(int node, int dst, int slice) const {
        return (Action)_rt[node][dst][slice];
    }

    int getDistance(int node, int dst, int slice) const {
        return (int)_dist[node][dst][slice];
    }

    int getFwdHops(int node, int dst, int slice) const {
        return (int)_fwd_hops[node][dst][slice];
    }

    int getPeer(int node, int slice) const {
        return _peer[node][slice];
    }

    int get_slice_for_pair(int src, int dst) const {
        return _connected[src][dst];
    }

    // ---- Getters ----
    int num_ports() const { return _num_ports; }
    int num_slices() const { return _num_slices; }
    simtime_picosec slice_active() const { return _slice_active; }
    simtime_picosec reconfig_time() const { return _reconfig; }
    simtime_picosec slice_total() const { return _slice_total; }
    simtime_picosec cycle_time() const { return _cycle; }

    // ---- Print routing table summary ----
    void printRoutingTable() const {
        int fwd_count = 0, wait_count = 0, max_dist = 0, max_fhops = 0;
        int unreachable = 0;
        for (int node = 1; node <= _num_ports; node++) {
            for (int dst = 1; dst <= _num_ports; dst++) {
                if (node == dst) continue;
                for (int s = 0; s < _num_slices; s++) {
                    if (_rt[node][dst][s] == FORWARD) fwd_count++;
                    else wait_count++;
                    int d = (int)_dist[node][dst][s];
                    if (d >= 0 && d > max_dist) max_dist = d;
                    if (d < 0) unreachable++;
                    int h = (int)_fwd_hops[node][dst][s];
                    if (h < 255 && h > max_fhops) max_fhops = h;
                }
            }
        }

        std::cout << "BFS Routing Table (max " << MAX_HOPS << " forward hops):\n"
                  << "  States: " << _num_ports << " nodes x " << _num_slices << " slices = "
                  << (_num_ports * _num_slices) << "\n"
                  << "  FORWARD entries: " << fwd_count
                  << "  WAIT entries: " << wait_count
                  << "  Unreachable: " << unreachable
                  << "\n  Max total distance: " << max_dist << " slots"
                  << "  Max forward hops: " << max_fhops << "\n";

        // Sample routes for first 4 nodes
        if (_num_ports <= 16) {
            std::cout << "  Sample routing decisions:\n";
            for (int src = 1; src <= std::min(4, _num_ports); src++) {
                for (int dst = 1; dst <= std::min(8, _num_ports); dst++) {
                    if (src == dst) continue;
                    std::cout << "    " << src << "->" << dst << ":";
                    for (int s = 0; s < std::min(8, _num_slices); s++) {
                        if (_rt[src][dst][s] == FORWARD)
                            std::cout << " s" << s << "=F(d"
                                      << (int)_dist[src][dst][s]
                                      << ",h" << (int)_fwd_hops[src][dst][s] << ")";
                        else
                            std::cout << " s" << s << "=W(d"
                                      << (int)_dist[src][dst][s] << ")";
                    }
                    if (_num_slices > 8) std::cout << " ...";
                    std::cout << "\n";
                }
            }
        }
        std::cout << "\n";
    }

private:
    void computeRoutingTable() {
        std::cout << "  Computing BFS routing table for " << _num_ports
                  << " destinations..." << std::flush;
        for (int dst = 1; dst <= _num_ports; dst++)
            bfsForDst(dst);
        std::cout << " done.\n";
    }

    // 3D Reverse BFS: state = (node, slice, fwd_hops_to_dst)
    //   WAIT  does NOT consume a forward hop (only costs 1 time slot)
    //   FORWARD consumes 1 forward hop + 1 time slot
    //   Constraint: fwd_hops ≤ MAX_HOPS
    //   Objective: minimize total time slots
    void bfsForDst(int dst) {
        int N = _num_ports, S = _num_slices;
        int H = MAX_HOPS + 1;  // h in [0 .. MAX_HOPS]

        // d[node][s][h] = min total time slots from (node,s) to dst using h fwd hops
        std::vector<std::vector<std::vector<int>>> d(
            N + 1, std::vector<std::vector<int>>(S, std::vector<int>(H, -1)));

        // Track whether (node, slice) has been decided in routing table
        std::vector<std::vector<bool>> decided(
            N + 1, std::vector<bool>(S, false));

        struct State { int node, s, h; };
        std::queue<State> q;

        // Initialize: (dst, s, 0) = 0 time slots for all slices
        for (int s = 0; s < S; s++) {
            d[dst][s][0] = 0;
            decided[dst][s] = true;
            _dist[dst][dst][s] = 0;
            _fwd_hops[dst][dst][s] = 0;
            q.push({dst, s, 0});
        }

        while (!q.empty()) {
            auto [node, s, h] = q.front();
            q.pop();
            int cost = d[node][s][h];
            int prev_s = (s - 1 + S) % S;

            // --- Reverse WAIT: (node, prev_s, h) --WAIT--> (node, s, h) ---
            // WAIT keeps same fwd_hops, costs 1 time slot
            if (d[node][prev_s][h] == -1) {
                d[node][prev_s][h] = cost + 1;
                q.push({node, prev_s, h});
                if (!decided[node][prev_s]) {
                    decided[node][prev_s] = true;
                    _rt[node][dst][prev_s] = WAIT;
                    _dist[node][dst][prev_s] = (int16_t)(cost + 1);
                    _fwd_hops[node][dst][prev_s] = (uint8_t)h;
                }
            }

            // --- Reverse FORWARD: (x, prev_s, h+1) --FWD--> (node, s, h) ---
            // FORWARD uses 1 fwd hop, costs 1 time slot
            if (h + 1 <= MAX_HOPS) {
                int x = _peer[node][prev_s];
                if (x > 0 && d[x][prev_s][h + 1] == -1) {
                    d[x][prev_s][h + 1] = cost + 1;
                    q.push({x, prev_s, h + 1});
                    if (!decided[x][prev_s]) {
                        decided[x][prev_s] = true;
                        _rt[x][dst][prev_s] = FORWARD;
                        _dist[x][dst][prev_s] = (int16_t)(cost + 1);
                        _fwd_hops[x][dst][prev_s] = (uint8_t)(h + 1);
                    }
                }
            }
        }
    }

    std::vector<std::vector<std::pair<int,int>>> _matchings;
    std::map<int, std::vector<int>> _schedule;
    std::vector<std::vector<int>> _peer;       // [node][slice]
    std::vector<std::vector<int>> _connected;  // [src][dst] -> direct slice

    // BFS routing table (dynamic, 1-indexed)
    std::vector<std::vector<std::vector<uint8_t>>>  _rt;        // [node][dst][slice] → FORWARD/WAIT
    std::vector<std::vector<std::vector<int16_t>>>  _dist;      // [node][dst][slice] → total time slots (-1 = unreachable)
    std::vector<std::vector<std::vector<uint8_t>>>  _fwd_hops;  // [node][dst][slice] → forward hops used

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
//  EcsBuffer — Edge Circuit Switch Buffer (BFS multi-hop, no buffer limit)
//
//  Routing:
//    dst == me  → LOCAL: pkt.sendOn()
//    FORWARD    → safe check → ocs_tx → OcsLinkDelay → peer ocs_rx
//    WAIT       → tag target_slice, store in buffer
//    Unreachable (dist=255) → wait for direct connection slice
//
//  No internal buffer size limit.
// ================================================================

class EcsBuffer : public PacketSink {
public:
    EcsBuffer(int nodeId, DynOcsTopology* topo)
        : _nodeId(nodeId)
        , _topo(topo)
        , _nodename("ecs_" + std::to_string(nodeId))
    {
        _ps_per_byte = (simtime_picosec)((double)8e12 / (double)LINK_SPEED_BPS);

        _ocs_tx = new SerDesPort(
            "ocs_tx_" + std::to_string(nodeId),
            LINK_SPEED_BPS, OCS_TX_BUFFER);
        _ocs_tx->setCallback([this](Packet* pkt) { onOcsTxDone(pkt); });

        _ocs_rx = new SerDesPort(
            "ocs_rx_" + std::to_string(nodeId),
            LINK_SPEED_BPS, OCS_TX_BUFFER);
        _ocs_rx->setCallback([this](Packet* pkt) { processPacket(pkt); });

        _ocsLink = new OcsLinkDelay(
            "ocs_link_" + std::to_string(nodeId),
            OCS_LINK_DELAY_PS);
    }

    // From GPU (instant transfer — no serialization needed)
    void receivePacket(Packet& pkt) override {
        processPacket(&pkt);
    }

    SerDesPort* ocsRxPort() { return _ocs_rx; }

    // ---- Slice boundary ----
    void onSliceStart(int new_slice) {
        simtime_picosec now = EventList::now();
        simtime_picosec remaining = sliceRemainingActive(now);
        simtime_picosec ocs_tx_drain = (simtime_picosec)_ocs_tx->queuesize()
                                       * _ps_per_byte;
        simtime_picosec available = (remaining > ocs_tx_drain)
                                    ? (remaining - ocs_tx_drain) : 0;
        mem_b budget = (mem_b)(available / _ps_per_byte);

        std::vector<Packet*> to_forward;
        auto it = _buffer.begin();
        while (it != _buffer.end()) {
            MoePacket* mpkt = static_cast<MoePacket*>(*it);
            if ((int)mpkt->target_slice == new_slice) {
                if ((mem_b)mpkt->size() > budget) {
                    // Budget exhausted — re-tag for next FORWARD opportunity
                    int dst = (int)mpkt->targetId;
                    int new_ts = computeTargetSlice(dst, new_slice);
                    mpkt->target_slice = (int16_t)new_ts;
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
            int peer = _topo->getPeer(_nodeId, new_slice);
            for (Packet* pkt : to_forward)
                forwardToOcs(pkt, peer);
        }
    }

    void setPeerBuffer(int peer, EcsBuffer* buf) {
        _peerBuffers[peer] = buf;
    }

    const std::string& nodename() override { return _nodename; }

    // Stats
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
    simtime_picosec sliceRemainingActive(simtime_picosec now) const {
        int abs_slice = _topo->time_to_absolute_slice(now);
        simtime_picosec slice_start = _topo->get_slice_start_time(abs_slice);
        simtime_picosec active_end = slice_start + _topo->slice_active();
        if (now >= active_end) return 0;
        return active_end - now;
    }

    bool canSendInSlice(simtime_picosec now, mem_b pkt_size) const {
        simtime_picosec remaining = sliceRemainingActive(now);
        if (remaining == 0) return false;
        simtime_picosec ocs_tx_drain = (simtime_picosec)_ocs_tx->queuesize()
                                       * _ps_per_byte;
        simtime_picosec pkt_drain = (simtime_picosec)pkt_size * _ps_per_byte;
        return (ocs_tx_drain + pkt_drain) <= remaining;
    }

    void processPacket(Packet* pkt) {
        MoePacket* mpkt = static_cast<MoePacket*>(pkt);
        int dst = (int)mpkt->targetId;

        if (dst == _nodeId) {
            _localCount++;
            pkt->sendOn();
            return;
        }

        simtime_picosec now = EventList::now();
        bool in_reconfig = _topo->is_in_reconfig(now);
        int slice = _topo->effective_slice(now);
        auto action = _topo->getAction(_nodeId, dst, slice);
        int dist = _topo->getDistance(_nodeId, dst, slice);

        // FORWARD: BFS says forward AND link is available AND budget allows
        if (action == DynOcsTopology::FORWARD && dist >= 0
            && !in_reconfig && canSendInSlice(now, (mem_b)pkt->size())) {
            _forwardCount++;
            int peer = _topo->getPeer(_nodeId, slice);
            forwardToOcs(pkt, peer);
        } else {
            // WAIT: tag and buffer
            _waitCount++;
            int ts;
            if (action == DynOcsTopology::FORWARD && dist >= 0 && !in_reconfig) {
                // FORWARD but no time budget — wait for this slice next cycle
                ts = slice;
            } else if (action == DynOcsTopology::FORWARD && dist >= 0 && in_reconfig) {
                // FORWARD but in reconfig — wait for next occurrence of this slice
                ts = slice;
            } else {
                // WAIT action: find next FORWARD slice
                ts = computeTargetSlice(dst, slice);
            }
            mpkt->target_slice = (int16_t)ts;

            // No buffer size limit
            _buffer.push_back(pkt);
            _bufferSize += pkt->size();
            if (_bufferSize > _maxBufferUsed)
                _maxBufferUsed = _bufferSize;
        }
    }

    void forwardToOcs(Packet* pkt, int peer) {
        static_cast<MoePacket*>(pkt)->target_slice = -1;
        _ocs_tx_nexthop.push(peer);
        _ocs_tx->receivePacket(*pkt);
    }

    void onOcsTxDone(Packet* pkt) {
        assert(!_ocs_tx_nexthop.empty());
        int nh = _ocs_tx_nexthop.front();
        _ocs_tx_nexthop.pop();

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

        auto it = _peerBuffers.find(nh);
        assert(it != _peerBuffers.end());
        _ocsLink->push(pkt, it->second->ocsRxPort());
    }

    int computeTargetSlice(int dst, int current_slice) {
        int ns = _topo->num_slices();
        // Find next FORWARD slice (BFS guarantees ≤ MAX_HOPS fwd hops)
        for (int offset = 1; offset <= ns; offset++) {
            int s = (current_slice + offset) % ns;
            if (_topo->getAction(_nodeId, dst, s) == DynOcsTopology::FORWARD
                && _topo->getDistance(_nodeId, dst, s) >= 0)
                return s;
        }
        // Fallback: direct connection slice (should not happen with 3D BFS)
        return _topo->get_slice_for_pair(_nodeId, dst);
    }

    int _nodeId;
    DynOcsTopology* _topo;
    std::string _nodename;
    simtime_picosec _ps_per_byte;

    SerDesPort* _ocs_tx;
    SerDesPort* _ocs_rx;
    OcsLinkDelay* _ocsLink;
    std::map<int, EcsBuffer*> _peerBuffers;
    std::queue<int> _ocs_tx_nexthop;

    std::list<Packet*> _buffer;
    mem_b _bufferSize = 0;
    mem_b _maxBufferUsed = 0;

    uint64_t _localCount   = 0;
    uint64_t _forwardCount = 0;
    uint64_t _waitCount    = 0;
    uint64_t _drops        = 0;
    uint64_t _misroutes    = 0;
};

// ================================================================
//  SliceAlarm
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
