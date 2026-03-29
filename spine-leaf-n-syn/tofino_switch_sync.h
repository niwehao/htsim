/*
 * tofino_switch_sync.h  —  512-GPU 同步调度版交换机
 *
 * 保留 BufferGate/BufferRelease 结构 (与 htsim_syn 一致),
 * 但同步调度保证包到达时队列为空, BufferGate 实际只做 pass-through.
 *
 * 每个 Tofino 在 Route 中占 5 个节点:
 *   ingressQ → BufferGate → fabricQ → BufferRelease → egressQ
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
            pkt.free();
            return;
        }

        _bufUsage += sz;
        //如果大于max,则打印现在的bufUsage,并且更新max
        if(_bufUsage > _bufMax){
            _bufMax = _bufUsage;
            std::cout << "[" << _nodename << "] BUFUSAGE: " << _bufUsage/(ACK_PKT_SIZE+FRAGMENT_PAYLOAD_SIZE) << "个)\n";
        }
        pkt.sendOn();
        return;
        // 同步调度下, 包到达时缓冲应为空 → 直接转发
        if (_bufUsage <= (ACK_PKT_SIZE+FRAGMENT_PAYLOAD_SIZE)*10 || pkt.size() == ACK_PKT_SIZE) {
            pkt.sendOn();
            return;
        }
        //打印bufferoverflow日志
        std::cout << "[" << _nodename << "] BUFF is over: " << _bufUsage << " bytes (+" << sz << ")\n";
        //结束程序运行
        exit(1);


        // fallback: 缓冲非空 → 延迟 (正常不应触发)
        bool wasEmpty = _delayed.empty();
        
        if(wasEmpty){
            _delayed.push_back({EventList::now() + BUFFER_DELAY_PS, &pkt});
        }
        else{

        }
        
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
    int64_t     _bufMax = 0;
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

    // 5 节点 per switch: ingressQ → BufferGate → fabricQ → BufferRelease → egressQ
    void appendToRoute(Route& route, uint32_t inPort, uint32_t outPort) const {
        route.push_back(_ingressQ.at(inPort));
        route.push_back(_bufferGate);
        route.push_back(_fabricQueue);
        route.push_back(_bufferRelease);
        route.push_back(_egressQ.at(outPort));
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
