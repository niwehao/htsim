/*
 * gpu_node.h  —  EcsBuffer HOHO version
 *
 * GPU sends ALL targets in parallel at phase start.
 * Network handles routing via EcsBuffer (hop-by-hop with BFS routing).
 * Route: [EcsBuffer@src, rxQ_dst, AppSink_dst]
 *
 * Parameters:
 *   - 8 GPU nodes, 7 peers each
 *   - MOE 32 layers, DISPATCH + COMBINE
 *   - 2048 fragments × 4 KB = 8 MB per target
 *   - Timeout retransmission (20 ms)
 */

#pragma once
#include "constants.h"
#include "ocs_switch.h"
#include "hohoo_scheduler.h"

// ================================================================
//  AppSink
// ================================================================
class GpuNode;

class AppSink : public PacketSink {
public:
    AppSink(const std::string& name) : _nodename(name) {}
    void setGpu(GpuNode* gpu) { _gpu = gpu; }
    void receivePacket(Packet& pkt) override;
    const std::string& nodename() override { return _nodename; }
private:
    GpuNode*    _gpu = nullptr;
    std::string _nodename;
};

// ================================================================
//  GpuNode (EcsBuffer HOHO version)
// ================================================================

class GpuNode {
public:
    GpuNode(uint8_t nodeId, const std::vector<uint8_t>& peerIds);

    Queue*    txQueue() { return _txQueue; }
    Queue*    rxQueue() { return _rxQueue; }
    AppSink*  appSink() { return &_appSink; }

    void setRoute(uint8_t targetId, const Route* route) {
        _routes[targetId] = route;
    }
    void setFlow(PacketFlow* flow) { _flow = flow; }

    void onPacketReceived(Packet& pkt);
    bool done() const { return _done; }

private:
    void startInference();
    void startLayerPhase();

    void sendAllFragments(uint8_t targetId, int layer, int phase);
    void sendFragment(uint8_t targetId, int layer, int phase,
                      uint16_t fragId, uint64_t roundId);
    void retransmitCheck(uint8_t targetId, int layer, int phase);

    void processDataPacket(MoePacket& pkt, int layer, int phase);
    void processAckPacket (MoePacket& pkt, int layer, int phase);
    void sendAck(uint8_t targetId, int layer, int phase, uint16_t fragId);

    void checkPhaseComplete(int layer, int phase);
    void advancePhase();

    static std::string phaseName(int p) { return p==0?"DISPATCH":"COMBINE"; }

    struct TxPeerState {
        std::set<int>     pending;
        std::map<int,int> attempts;
        bool              done = false;
        simtime_picosec   batchSendTime = 0;
    };
    struct TxState {
        std::map<uint8_t, TxPeerState> peers;
        uint32_t doneCount = 0;
    };
    struct RxState {
        std::set<uint8_t> completedPeers;
        bool              barrierMet = false;
    };

    uint8_t              _nodeId;
    std::vector<uint8_t> _peerIds;
    PacketFlow*          _flow = nullptr;

    Queue*   _txQueue;
    Queue*   _rxQueue;
    AppSink  _appSink;

    std::map<uint8_t, const Route*> _routes;

    int  _currentLayer = 0;
    int  _currentPhase = 0;
    bool _done = false;

    std::map<std::pair<int,int>, TxState>  _txState;
    std::map<std::pair<int,int>, RxState>  _rxState;
    std::set<std::pair<int,int>>           _phaseFinished;

    std::set<std::tuple<int,int,int>>                _processedSet;
    std::map<std::tuple<int,int,int>, std::set<int>> _rxFragTracker;

    std::map<uint8_t, TimerEvent*> _retxTimers;
    TimerEvent _gapTimer;
    TimerEvent _startEvent;

    NodeStats _stats;
    static uint32_t s_doneCount;
};
