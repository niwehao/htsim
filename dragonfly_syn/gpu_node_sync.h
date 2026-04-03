/*
 * gpu_node_sync.h  —  N-GPU Dragonfly 同步调度版 GPU 节点
 *
 * 按预计算 schedule 定时注入, 无重传.
 * 保留 ACK 机制用于验证和阶段完成判断.
 */

#pragma once
#include "constants.h"
#include "tofino_switch_sync.h"
#include "sync_scheduler.h"

// ================================================================
//  AppSinkSync
// ================================================================
class GpuNodeSync;

class AppSinkSync : public PacketSink {
public:
    AppSinkSync(const std::string& name) : _nodename(name) {}
    void setGpu(GpuNodeSync* gpu) { _gpu = gpu; }
    void receivePacket(Packet& pkt) override;
    const std::string& nodename() override { return _nodename; }
private:
    GpuNodeSync* _gpu = nullptr;
    std::string  _nodename;
};

// ================================================================
//  GpuNodeSync
// ================================================================
class GpuNodeSync {
public:
    GpuNodeSync(uint16_t nodeId, const std::vector<uint16_t>& peerIds,
                const std::vector<ScheduleEntry>& phaseSchedule);

    Queue*       txQueue()  { return _txQueue; }
    Queue*       rxQueue()  { return _rxQueue; }
    AppSinkSync* appSink()  { return &_appSink; }

    void setRoute(uint16_t targetId, const Route* route) { _routes[targetId] = route; }
    void setFlow(PacketFlow* flow) { _flow = flow; }
    void onPacketReceived(Packet& pkt);
    bool done() const { return _done; }

private:
    void startInference();
    void startLayerPhase();
    void processSchedule();

    void processDataPacket(MoePacket& pkt, int layer, int phase);
    void processAckPacket(MoePacket& pkt, int layer, int phase);
    void sendAck(uint16_t targetId, int layer, int phase, uint16_t fragId);

    void checkPhaseComplete(int layer, int phase);
    void advancePhase();
    void cleanupPhase(int layer, int phase);

    static std::string phaseName(int p) { return p == 0 ? "DISPATCH" : "COMBINE"; }

    struct TxPeerState {
        std::set<int> ackedFrags;
        bool          done = false;
    };
    struct TxState {
        std::map<uint16_t, TxPeerState> peers;
        uint32_t doneCount = 0;
    };

    struct RxState {
        std::set<uint16_t> completedPeers;
        bool               barrierMet = false;
    };

    uint16_t              _nodeId;
    std::vector<uint16_t> _peerIds;
    PacketFlow*           _flow = nullptr;

    Queue*       _txQueue;
    Queue*       _rxQueue;
    AppSinkSync  _appSink;

    std::map<uint16_t, const Route*> _routes;

    int  _currentLayer = 0;
    int  _currentPhase = 0;
    bool _done = false;

    std::map<std::pair<int,int>, TxState>  _txState;
    std::map<std::pair<int,int>, RxState>  _rxState;
    std::set<std::pair<int,int>>           _phaseFinished;

    std::set<std::tuple<int,int,int>>                _processedSet;
    std::map<std::tuple<int,int,int>, std::set<int>> _rxFragTracker;

    std::vector<ScheduleEntry> _phaseSchedule;
    size_t                     _schedIdx = 0;
    simtime_picosec            _phaseStartTime = 0;

    TimerEvent _sendTimer;
    TimerEvent _gapTimer;
    TimerEvent _startEvent;

    NodeStats _stats;
    static uint32_t s_doneCount;
    static uint32_t s_phaseReportCount;
};
