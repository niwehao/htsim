/*
 * tofino_switch.h  —  Dragonfly 版 (与 htsim_v2 相同的交换机模型)
 *
 * 每个交换机在 Route 中占 5 个节点:
 *   ingressQueue[port_in] → BufferGate → fabricQueue → BufferRelease → egressQueue[port_out]
 */

#pragma once
#include "constants.h"
#include <deque>

// ================================================================
//  BufferGate
// ================================================================

class BufferGate : public EventSource, public PacketSink {
public:
    BufferGate(const std::string& name)
        : EventSource(name), _nodename(name) {}

    void receivePacket(Packet& pkt) override {
        int sz = pkt.size();

        if (_bufUsage + sz > SW_QUEUE_SIZE_BYTES) {
            // std::cout << "[" << _nodename << "] DROP: buffer overflow ("
            //           << _bufUsage << "+" << sz << ">" << SW_QUEUE_SIZE_BYTES << ")\n";
            pkt.free();
            return;
        }

        _bufUsage += sz;
        if (_bufUsage <= sz+pkt.size() ||  pkt.size()== 29) {  // 仅本包, 或者是ACK包（不占用缓冲）直接转发
            pkt.sendOn();
            return;
        }

        // 3. 缓冲非空 → 延迟 2ms
        bool wasEmpty = _delayed.empty();
        _delayed.push_back({EventList::now() + BUFFER_DELAY_PS, &pkt});
        // 仅在队列从空变非空时注册事件, 避免每包一次 sourceIsPending
        if (wasEmpty)
            EventList::sourceIsPending(*this, _delayed.front().due);
    }

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
// ================================================================

class BufferRelease : public PacketSink {
public:
    BufferRelease(BufferGate* gate, const std::string& name)
        : _gate(gate), _nodename(name) {}

    void receivePacket(Packet& pkt) override {
        _gate->releaseBuffer(pkt.size());
        pkt.sendOn();
    }

    const std::string& nodename() override { return _nodename; }

private:
    BufferGate* _gate;
    std::string _nodename;
};

// ================================================================
//  TofinoSwitch
// ================================================================

class TofinoSwitch {
public:
    TofinoSwitch(int idx) : _idx(idx) {
        std::string pfx = "tof" + std::to_string(idx);
        auto ports = tofinoPorts(idx);

        _fabricQueue = new Queue(FABRIC_SPEED_BPS, HUGE_BUFFER,
                                 EventList::getTheEventList(), nullptr);

        _bufferGate    = new BufferGate(pfx + "_bufGate");
        _bufferRelease = new BufferRelease(_bufferGate, pfx + "_bufRel");

        for (int p : ports) {
            _ingressQ[p] = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                                      EventList::getTheEventList(), nullptr);
            _egressQ[p]  = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                                      EventList::getTheEventList(), nullptr);
        }
    }

    void appendToRoute(Route& route, int inPort, int outPort) const {
        route.push_back(_ingressQ.at(inPort));
        route.push_back(_bufferGate);
        route.push_back(_fabricQueue);
        route.push_back(_bufferRelease);
        route.push_back(_egressQ.at(outPort));
    }

    Queue* ingressQueue(int port) const { return _ingressQ.at(port); }
    Queue* egressQueue(int port) const { return _egressQ.at(port); }
    int idx() const { return _idx; }

private:
    int _idx;
    std::map<int, Queue*> _ingressQ;
    std::map<int, Queue*> _egressQ;
    Queue*                _fabricQueue;
    BufferGate*           _bufferGate;
    BufferRelease*        _bufferRelease;
};
