/*
 * tofino_switch_sync.h  —  同步版简化 Tofino
 *
 * 与 htsim_v2/tofino_switch.h 的区别:
 *   - 移除 BufferGate (无 2ms 延迟, 无 32MB 缓冲检查)
 *   - 移除 BufferRelease
 *   - 每个 Tofino 在 Route 中占 3 个节点 (而非 5 个):
 *       ingressQueue[port_in] → fabricQueue → egressQueue[port_out]
 *
 * 同步调度保证:
 *   包到达每个队列时, 队列空闲 → 无排队等待
 *   因此无需 BufferGate 的缓冲管理
 */

#pragma once
#include "constants.h"

class TofinoSwitchSync {
public:
    TofinoSwitchSync(int idx) : _idx(idx) {
        auto ports = tofinoPorts(idx);

        // fabricQueue: 100 Gbps 共享内部带宽
        _fabricQueue = new Queue(FABRIC_SPEED_BPS, HUGE_BUFFER,
                                 EventList::getTheEventList(), nullptr);

        // 每端口 ingress / egress queue (25 Gbps)
        for (int p : ports) {
            _ingressQ[p] = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                                      EventList::getTheEventList(), nullptr);
            _egressQ[p]  = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                                      EventList::getTheEventList(), nullptr);
        }
    }

    // Route 构建: 3 个节点 (比 htsim_v2 少了 BufferGate 和 BufferRelease)
    void appendToRoute(Route& route, int inPort, int outPort) const {
        route.push_back(_ingressQ.at(inPort));
        route.push_back(_fabricQueue);
        route.push_back(_egressQ.at(outPort));
    }

    Queue* ingressQueue(int port) const { return _ingressQ.at(port); }
    Queue* egressQueue(int port)  const { return _egressQ.at(port); }
    int idx() const { return _idx; }

private:
    int                   _idx;
    std::map<int, Queue*> _ingressQ;
    std::map<int, Queue*> _egressQ;
    Queue*                _fabricQueue;
};
