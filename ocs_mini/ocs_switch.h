/*
 * ocs_switch.h  —  EcsBuffer-based OCS switch model
 *
 * Creates 8 EcsBuffers (one per GPU node) + SliceAlarm.
 * Each EcsBuffer has:
 *   - Internal routing logic (FORWARD / WAIT / LOCAL)
 *   - ocs_tx serialization port (200 Gbps)
 *   - OcsLinkDelay (100 ns propagation)
 *   - Internal buffer with target_slice tags
 *
 * Wiring:
 *   EcsBuffer@A.ocs_tx → OcsLinkDelay@A → EcsBuffer@peer
 *   (peer determined by _ocs_tx_nexthop tag at forwarding time)
 *
 * Physical model:
 *   Each node has ONE OCS port, connecting to ONE peer per time slot.
 *   8 EcsBuffers replace the old 56 OcsCircuits + 8 HohoForwarders.
 */

#pragma once
#include "hoho_routing.h"

#include <iostream>
#include <map>

class OcsSwitch {
public:
    OcsSwitch(DynOcsTopology* topo)
        : _topo(topo)
        , _alarm(new SliceAlarm(topo))
    {
        // Phase 1: Create 8 EcsBuffers
        for (int n = 1; n <= (int)NUM_GPU_NODES; n++) {
            _buffers[n] = new EcsBuffer(n, topo);
            _alarm->addBuffer(_buffers[n]);
        }

        // Phase 2: Wire peer references (each EcsBuffer knows all others)
        for (int a = 1; a <= (int)NUM_GPU_NODES; a++) {
            for (int b = 1; b <= (int)NUM_GPU_NODES; b++) {
                if (a == b) continue;
                _buffers[a]->setPeerBuffer(b, _buffers[b]);
            }
        }
    }

    EcsBuffer* getBuffer(int nodeId) const {
        auto it = _buffers.find(nodeId);
        assert(it != _buffers.end());
        return it->second;
    }

    DynOcsTopology* topology() const { return _topo; }
    SliceAlarm* alarm() const { return _alarm; }

    void printStats() const {
        uint64_t total_local = 0, total_forward = 0, total_wait = 0;
        uint64_t total_drops = 0, total_ocs_tx_drops = 0, total_ocs_rx_drops = 0;
        uint64_t total_ocs_tx_pkts = 0, total_ocs_rx_pkts = 0;
        uint64_t total_misroutes = 0;
        mem_b max_buf = 0, max_ocs_tx = 0, max_ocs_rx = 0;

        for (auto& [id, buf] : _buffers) {
            total_local      += buf->localCount();
            total_forward    += buf->forwardCount();
            total_wait       += buf->waitCount();
            total_drops      += buf->drops();
            total_misroutes  += buf->misroutes();
            total_ocs_tx_drops += buf->ocsTxDrops();
            total_ocs_tx_pkts  += buf->ocsTxPkts();
            total_ocs_rx_drops += buf->ocsRxDrops();
            total_ocs_rx_pkts  += buf->ocsRxPkts();
            if (buf->maxBufferUsed() > max_buf)
                max_buf = buf->maxBufferUsed();
            if (buf->ocsTxMaxQueue() > max_ocs_tx)
                max_ocs_tx = buf->ocsTxMaxQueue();
            if (buf->ocsRxMaxQueue() > max_ocs_rx)
                max_ocs_rx = buf->ocsRxMaxQueue();
        }

        std::string sep(60, '=');
        std::cout << "\n" << sep << "\n"
                  << "        OCS EcsBuffer ROUTING STATISTICS\n" << sep << "\n"
                  << "  EcsBuffers: " << _buffers.size()
                  << " (one per GPU node)\n"
                  << "  Packets delivered locally: " << total_local << "\n"
                  << "  Packets forwarded (immediate): " << total_forward << "\n"
                  << "  Packets buffered (WAIT): " << total_wait << "\n"
                  << "  Packets through ocs_tx: " << total_ocs_tx_pkts << "\n"
                  << "  Packets through ocs_rx: " << total_ocs_rx_pkts << "\n"
                  << "  Buffer drops: " << total_drops << "\n"
                  << "  OCS TX drops: " << total_ocs_tx_drops << "\n"
                  << "  OCS RX drops: " << total_ocs_rx_drops << "\n"
                  << "  Misroutes (slice changed): " << total_misroutes << "\n"
                  << "  Max buffer used: "
                  << (max_buf / 1024) << " KB\n"
                  << "  Max ocs_tx queue: "
                  << (max_ocs_tx / 1024) << " KB\n"
                  << "  Max ocs_rx queue: "
                  << (max_ocs_rx / 1024) << " KB\n"
                  << "  Slice activations: " << _alarm->totalActivations() << "\n";

        if (total_local + total_forward > 0) {
            uint64_t total_routed = total_local + total_forward + total_wait;
            double relay_pct = 100.0 * (total_forward + total_wait)
                               / total_routed;
            std::cout << "  Transit rate: " << std::fixed
                      << std::setprecision(2) << relay_pct << "%\n";
        }

        // Per-node stats
        std::cout << "\n  Per-node stats:\n";
        for (auto& [id, buf] : _buffers) {
            std::cout << "    Node " << id
                      << ": local=" << buf->localCount()
                      << " fwd=" << buf->forwardCount()
                      << " wait=" << buf->waitCount()
                      << " misrt=" << buf->misroutes()
                      << " buf=" << (buf->maxBufferUsed()/1024) << "KB"
                      << " ocs_tx=" << buf->ocsTxPkts()
                      << " ocs_rx=" << buf->ocsRxPkts()
                      << " drops=" << buf->drops()
                      << "\n";
        }

        std::cout << sep << "\n";
    }

private:
    DynOcsTopology* _topo;
    SliceAlarm*     _alarm;
    std::map<int, EcsBuffer*> _buffers;
};
