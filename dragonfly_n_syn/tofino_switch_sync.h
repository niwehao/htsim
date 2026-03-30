/*
 * tofino_switch_sync.h  —  Dragonfly 同步调度版交换机
 *
 * 路由顺序 (每 switch 5 节点):
 *   ingressQ → BufferGate → fabricQ → egressQ → BufferRelease
 *
 * BufferRelease 在 egressQ 之后, 即 buffer 占用持续到包离开
 * egressQ 才释放. 调度器需考虑 egressQ 约束.
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
            std::cout << "[" << _nodename << "] HARD LIMIT: lost packet "
                      << _bufUsage << " bytes (+" << sz << ")\n";
            pkt.free();
            exit(1);
            return;
        }

        _bufUsage += sz;
        if (_bufUsage > _maxBufUsage) _maxBufUsage = _bufUsage;

        // srcSw-only 调度: transit/dst switch 有短暂排队, 阈值放宽
        // 实测 max ~540KB, 远低于 64MB hard limit
        static constexpr int64_t BUF_WARN_THRESHOLD = 2LL * 1024 * 1024; // 2MB
        if (_bufUsage > BUF_WARN_THRESHOLD && sz == DATA_PKT_SIZE) {
            std::cout << "[" << _nodename << "] WARNING: bufUsage="
                      << _bufUsage << " bytes (+" << sz << ")\n";
        }

        pkt.sendOn();
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
    int64_t maxBufUsage() const { return _maxBufUsage; }

private:
    struct Delayed { simtime_picosec due; Packet* pkt; };
    std::deque<Delayed> _delayed;
    int64_t     _bufUsage = 0;
    int64_t     _maxBufUsage = 0;
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
//  TofinoSwitchSync — 通用交换机 (动态端口)
//
//  路由顺序: ingressQ → BufferGate → fabricQ → egressQ → BufferRelease
// ================================================================

class TofinoSwitchSync {
public:
    TofinoSwitchSync(int switchId, const std::vector<uint32_t>& ports)
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

    // 5 节点: ingressQ → BufferGate → fabricQ → egressQ → BufferRelease
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
