/*
 * ocs_switch.h  —  OCS switch model (direct-only routing)
 *
 * Creates 8 EcsBuffers (one per GPU node) + SliceAlarm.
 * No multi-hop routing — packets only sent on direct connection.
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
        for (int n = 1; n <= (int)NUM_GPU_NODES; n++) {
            _buffers[n] = new EcsBuffer(n, topo);
            _alarm->addBuffer(_buffers[n]);
        }

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
        uint64_t total_local = 0, total_direct = 0, total_wait = 0;
        uint64_t total_drops = 0, total_ocs_tx_drops = 0, total_ocs_rx_drops = 0;
        uint64_t total_ocs_tx_pkts = 0, total_ocs_rx_pkts = 0;
        uint64_t total_misroutes = 0;
        mem_b max_buf = 0, max_ocs_tx = 0, max_ocs_rx = 0;

        for (auto& [id, buf] : _buffers) {
            total_local      += buf->localCount();
            total_direct     += buf->directCount();
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
                  << "        OCS DIRECT-ONLY ROUTING STATISTICS\n" << sep << "\n"
                  << "  EcsBuffers: " << _buffers.size()
                  << " (one per GPU node, direct-only)\n"
                  << "  Packets delivered locally: " << total_local << "\n"
                  << "  Packets sent direct: " << total_direct << "\n"
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

        // Per-node stats
        std::cout << "\n  Per-node stats:\n";
        for (auto& [id, buf] : _buffers) {
            std::cout << "    Node " << id
                      << ": local=" << buf->localCount()
                      << " direct=" << buf->directCount()
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
