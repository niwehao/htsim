/*
 * gpu_node.h  —  csg-htsim 版
 *
 * 翻译自: GPU.py :: GpuNode
 *
 * GPU 端口延迟模型:
 *
 *   TX (发送): app → txQueue (25G 串行化) → pkt->sendOn() → Route 首跳
 *              对应 _port_worker_out
 *
 *   RX (接收): Route 末跳 → rxQueue (25G 串行化) → pkt->sendOn() → AppSink
 *              对应 _port_worker_in → _process_rx_packet
 *
 * 在 csg-htsim 的 source-routing 模型中:
 *   - txQueue 不在 Route 中 (它是入口点)
 *   - rxQueue 和 AppSink 在 Route 末尾 (每条 Route 最后两个元素)
 *   - txQueue.completeService() 调 sendOn() → Route[0] (第一个 Tofino ingress)
 *   - rxQueue.completeService() 调 sendOn() → AppSink.receivePacket()
 */

#pragma once
#include "constants.h"
#include "tofino_switch.h"

// ================================================================
//  AppSink — Route 终点, 将包交给 GpuNode 应用层
// ================================================================
class GpuNode;  // forward

class AppSink : public PacketSink {
public:
    AppSink(const std::string& name) : _nodename(name) {}
    void setGpu(GpuNode* gpu) { _gpu = gpu; }
    void receivePacket(Packet& pkt) override;   // 定义在 gpu_node.cpp
    const std::string& nodename() override { return _nodename; }
private:
    GpuNode*    _gpu = nullptr;
    std::string _nodename;
};

// ================================================================
//  GpuNode
// ================================================================

class GpuNode {
public:
    // peerIds: [1..8] 中除自身外的 7 个
    GpuNode(uint8_t nodeId, const std::vector<uint8_t>& peerIds);

    // ---- 拓扑接口 ----
    Queue*    txQueue() { return _txQueue; }     // main.cpp 连线用
    Queue*    rxQueue() { return _rxQueue; }     // Route 末尾倒数第二个
    AppSink*  appSink() { return &_appSink; }    // Route 最后一个

    // 注册 Route: 发往 targetId 的包使用此路由
    void setRoute(uint8_t targetId, const Route* route) {
        _routes[targetId] = route;
    }

    // 设置 PacketFlow (main.cpp 创建后传入)
    void setFlow(PacketFlow* flow) { _flow = flow; }

    // ---- 应用层入口 (由 AppSink 调用) ----
    void onPacketReceived(Packet& pkt);

    bool done() const { return _done; }

private:
    // 推理主循环
    void startInference();
    void startLayerPhase();

    // TX
    void sendAllFragments(uint8_t targetId, int layer, int phase);
    void sendFragment(uint8_t targetId, int layer, int phase,
                      uint16_t fragId, uint64_t roundId);
    void retransmitCheck(uint8_t targetId, int layer, int phase);

    // RX
    void processDataPacket(MoePacket& pkt, int layer, int phase);
    void processAckPacket (MoePacket& pkt, int layer, int phase);
    void sendAck(uint8_t targetId, int layer, int phase, uint16_t fragId);

    // 阶段控制
    void checkPhaseComplete(int layer, int phase);
    void advancePhase();

    static std::string phaseName(int p) { return p==0?"DISPATCH":"COMBINE"; }

    // TX 状态
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
    // RX 状态
    struct RxState {
        std::set<uint8_t> completedPeers;
        bool              barrierMet = false;
    };

    // 成员
    uint8_t              _nodeId;
    std::vector<uint8_t> _peerIds;
    PacketFlow*          _flow = nullptr;

    // 端口队列 (25 Gbps)
    Queue*   _txQueue;   // _port_worker_out
    Queue*   _rxQueue;   // _port_worker_in
    AppSink  _appSink;   // Route 终点

    // 预计算路由
    std::map<uint8_t, const Route*> _routes;  // targetId → Route*

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
