/*
 * tofino_switch.h  —  N-GPU Dragonfly 版
 *
 * 与 spine-leaf-n 相同的 5 节点模型:
 *   ingressQueue → BufferGate → fabricQueue → BufferRelease → egressQueue
 *
 * 非同步版: BufferGate 在拥塞时引入随机延迟.
 */

#pragma once
#include "constants.h"
#include <deque>
#include <cmath>

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
            // std::cout << "[" << nodename() << "] Buffer overflow! Dropping packet of size "
            //           << sz << " bytes (usage=" << _bufUsage << ")\n";
            pkt.free();
            return;
        }
       

        _bufUsage += sz;
        pkt.sendOn();
        return;
        if (_bufUsage <= (int64_t)DATA_PKT_SIZE * 16 || pkt.size() == ACK_PKT_SIZE) {
            pkt.sendOn();
            return;
        }

        bool wasEmpty = _delayed.empty();
        double randVal = (double)rand() / RAND_MAX;
        randVal = std::round(randVal * 10) / 10.0;
        simtime_picosec delay = BUFFER_DELAY_PS + (simtime_picosec)(randVal * 0.8*1000000000ULL);
        _delayed.push_back({EventList::now() + delay, &pkt});
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
//  TofinoSwitch — 通用交换机 (动态端口)
// ================================================================

class TofinoSwitch {
public:
    TofinoSwitch(int switchId, const std::vector<uint32_t>& ports)
        : _switchId(switchId)
    {
        std::string pfx = "sw" + std::to_string(switchId);

        _fabricQueue = new Queue(FABRIC_SPEED_BPS, HUGE_BUFFER,
                                 EventList::getTheEventList(), nullptr);
        _bufferGate    = new BufferGate(pfx + "_bufGate");
        _bufferRelease = new BufferRelease(_bufferGate, pfx + "_bufRel");

        for (uint32_t p : ports) {
            _ingressQ[p] = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                                      EventList::getTheEventList(), nullptr);
            _egressQ[p]  = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                                      EventList::getTheEventList(), nullptr);
        }
    }

    void appendToRoute(Route& route, uint32_t inPort, uint32_t outPort) const {
        route.push_back(_ingressQ.at(inPort));
        route.push_back(_bufferGate);
        route.push_back(_fabricQueue);
        route.push_back(_egressQ.at(outPort));
        route.push_back(_bufferRelease);
    }

    int switchId() const { return _switchId; }

private:
    int _switchId;
    std::map<uint32_t, Queue*> _ingressQ;
    std::map<uint32_t, Queue*> _egressQ;
    Queue*                     _fabricQueue;
    BufferGate*                _bufferGate;
    BufferRelease*             _bufferRelease;
};
