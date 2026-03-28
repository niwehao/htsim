/*
 * gpu_node_sync.h  —  Dragonfly 同步调度版 GPU 节点
 *
 * 与 dragonfly/gpu_node.h 的区别:
 *   1. 发送按预计算的 schedule 定时注入, 而非一次性 burst
 *   2. 无重传定时器 (调度保证无丢包/无 BufferGate 惩罚)
 *   3. 仍保留 ACK 机制用于验证数据到达 + 阶段完成判断
 */

#pragma once
#include "constants.h"
#include "tofino_switch_sync.h"
#include "sync_scheduler.h"

// ================================================================
//  AppSinkSync — Route 终点
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
    GpuNodeSync(uint8_t nodeId, const std::vector<uint8_t>& peerIds,
                const std::vector<ScheduleEntry>& phaseSchedule);

    // ---- 拓扑接口 ----
    Queue*       txQueue()  { return _txQueue; }
    Queue*       rxQueue()  { return _rxQueue; }
    AppSinkSync* appSink()  { return &_appSink; }

    void setRoute(uint8_t targetId, const Route* route) { _routes[targetId] = route; }
    void setFlow(PacketFlow* flow) { _flow = flow; }

    // ---- 应用层入口 ----
    void onPacketReceived(Packet& pkt);
    bool done() const { return _done; }

private:
    void startInference();
    void startLayerPhase();

    // 调度发送
    void processSchedule();

    // RX
    void processDataPacket(MoePacket& pkt, int layer, int phase);
    void processAckPacket(MoePacket& pkt, int layer, int phase);
    void sendAck(uint8_t targetId, int layer, int phase, uint16_t fragId);

    // 阶段控制
    void checkPhaseComplete(int layer, int phase);
    void advancePhase();

    static std::string phaseName(int p) { return p == 0 ? "DISPATCH" : "COMBINE"; }

    // TX 状态: 简化 — 无 pending/重传, 仅追踪已 ACK 的 fragment
    struct TxPeerState {
        std::set<int> ackedFrags;
        bool          done = false;
    };
    struct TxState {
        std::map<uint8_t, TxPeerState> peers;
        uint32_t doneCount = 0;
    };

    // RX 状态
    struct RxState {
        std::set<uint8_t> completedPeers;
        bool              barrierMet = false;
    };

    // ---- 成员 ----
    uint8_t              _nodeId;
    std::vector<uint8_t> _peerIds;
    PacketFlow*          _flow = nullptr;

    Queue*       _txQueue;
    Queue*       _rxQueue;
    AppSinkSync  _appSink;

    std::map<uint8_t, const Route*> _routes;

    int  _currentLayer = 0;
    int  _currentPhase = 0;
    bool _done = false;

    std::map<std::pair<int,int>, TxState>  _txState;
    std::map<std::pair<int,int>, RxState>  _rxState;
    std::set<std::pair<int,int>>           _phaseFinished;

    // RX 去重
    std::set<std::tuple<int,int,int>>                _processedSet;
    std::map<std::tuple<int,int,int>, std::set<int>> _rxFragTracker;

    // 调度相关
    std::vector<ScheduleEntry> _phaseSchedule;   // 一个阶段的发送计划 (每阶段复用)
    size_t                     _schedIdx = 0;    // 当前阶段已处理到第几个条目
    simtime_picosec            _phaseStartTime = 0;

    TimerEvent _sendTimer;
    TimerEvent _gapTimer;
    TimerEvent _startEvent;

    NodeStats _stats;
    static uint32_t s_doneCount;
};
