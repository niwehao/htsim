/*
 * gpu_node.h  —  GPU node with synchronized scheduling
 *
 * Key difference from ocs_scale:
 *   GPU uses pre-computed schedule to send each flow at the optimal time
 *   (when the direct OCS connection to the target is active).
 *   EcsBuffer still handles actual packet routing via BFS.
 */

#pragma once
#include "constants.h"
#include "ocs_switch.h"
#include "gpu_scheduler.h"

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
//  GpuNode — synchronized scheduling version
// ================================================================

class GpuNode {
public:
    GpuNode(uint16_t nodeId, DynOcsTopology* topo);

    Queue*    txQueue() { return _txQueue; }
    Queue*    rxQueue() { return _rxQueue; }
    AppSink*  appSink() { return &_appSink; }

    void setRoute(uint16_t targetId, const Route* route) {
        _routes[targetId] = route;
    }
    void setFlow(PacketFlow* flow) { _flow = flow; }

    void setPhaseConfig(int layer, int phase,
                        const std::vector<uint16_t>& send_targets,
                        const std::vector<uint16_t>& recv_from) {
        auto key = std::make_pair(layer, phase);
        _phaseConfigs[key] = {send_targets, recv_from};
    }

    void setFlowSchedule(const PerGpuFlowSchedule& sched) {
        _flowSchedule = sched;
    }

    void onPacketReceived(Packet& pkt);
    bool done() const { return _done; }

private:
    void startInference();
    void startLayerPhase();

    void sendAllFragments(uint16_t targetId, int layer, int phase);
    void sendFragment(uint16_t targetId, int layer, int phase,
                      uint16_t fragId, uint64_t roundId);
    void retransmitCheck(uint16_t targetId, int layer, int phase);

    void processDataPacket(MoePacket& pkt, int layer, int phase);
    void processAckPacket (MoePacket& pkt, int layer, int phase);
    void sendAck(uint16_t targetId, int layer, int phase, uint16_t fragId);

    void checkPhaseComplete(int layer, int phase);
    void advancePhase();

    static std::string phaseName(int p) { return p==0?"DISPATCH":"COMBINE"; }

    struct PhaseConfig {
        std::vector<uint16_t> send_targets;
        std::vector<uint16_t> recv_from;
    };

    struct TxPeerState {
        std::set<int>     pending;
        std::map<int,int> attempts;
        bool              done = false;
        simtime_picosec   batchSendTime = 0;
    };
    struct TxState {
        std::map<uint16_t, TxPeerState> peers;
        uint32_t doneCount = 0;
        uint32_t expectedCount = 0;
    };
    struct RxState {
        std::set<uint16_t> completedPeers;
        uint32_t expectedCount = 0;
        bool barrierMet = false;
    };

    uint16_t             _nodeId;
    DynOcsTopology*      _topo;
    PacketFlow*          _flow = nullptr;

    Queue*   _txQueue;
    Queue*   _rxQueue;
    AppSink  _appSink;

    std::map<uint16_t, const Route*> _routes;
    std::map<std::pair<int,int>, PhaseConfig> _phaseConfigs;
    PerGpuFlowSchedule _flowSchedule;

    int  _currentLayer = 0;
    int  _currentPhase = 0;
    bool _done = false;

    std::map<std::pair<int,int>, TxState>  _txState;
    std::map<std::pair<int,int>, RxState>  _rxState;
    std::set<std::pair<int,int>>           _phaseFinished;

    std::set<std::tuple<int,int,int>>                _processedSet;
    std::map<std::tuple<int,int,int>, std::set<int>> _rxFragTracker;

    std::map<uint16_t, TimerEvent*> _retxTimers;
    std::map<uint16_t, TimerEvent*> _sendTimers;  // scheduled send timers
    TimerEvent _gapTimer;
    TimerEvent _startEvent;

    NodeStats _stats;
    static uint32_t s_doneCount;
};
