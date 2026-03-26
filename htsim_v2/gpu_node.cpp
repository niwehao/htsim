/*
 * gpu_node.cpp  —  csg-htsim 版
 *
 * 翻译自: GPU.py :: GpuNode (完整实现)
 */

#include "gpu_node.h"
#include <algorithm>
#include <iostream>

// ---- 静态成员 ----
PacketDB<MoePacket> MoePacket::_db;
packetid_t          MoePacket::_nextId = 1;
uint32_t            GpuNode::s_doneCount = 0;

// ================================================================
//  AppSink::receivePacket  (Route 终点 → GpuNode 应用层)
// ================================================================
void AppSink::receivePacket(Packet& pkt) {
    _gpu->onPacketReceived(pkt);
}

// ================================================================
//  构造
// ================================================================
GpuNode::GpuNode(uint8_t nodeId, const std::vector<uint8_t>& peerIds)
    : _nodeId(nodeId)
    , _peerIds(peerIds)
    , _appSink("gpu" + std::to_string(nodeId) + "_app")
    , _gapTimer("gap_" + std::to_string(nodeId))
    , _startEvent("start_" + std::to_string(nodeId))
{
    _appSink.setGpu(this);
    _stats.nodeId = nodeId;
    g_allStats.push_back(&_stats);

    std::string pfx = "gpu" + std::to_string(nodeId);

    // TX queue: 25 Gbps 串行化 (_port_worker_out)
    _txQueue = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);

    // RX queue: 25 Gbps 串行化 (_port_worker_in)
    _rxQueue = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);

    // per-peer 重传定时器
    for (uint8_t pid : peerIds)
        _retxTimers[pid] = new TimerEvent(
            "retx_" + std::to_string(nodeId) + "_" + std::to_string((int)pid));

    // t=0 启动推理
    _startEvent.arm(0, [this]() { startInference(); });
}

// ================================================================
//  startInference  (run_multi_layer_inference)
// ================================================================
void GpuNode::startInference() {
    std::cout << "[Node " << (int)_nodeId << "] INFERENCE START — "
              << TOTAL_LAYERS << " layers, "
              << TOTAL_FRAGMENTS << " frags/target, "
              << (FRAGMENT_PAYLOAD_SIZE/1024) << " KB/frag\n";
    _currentLayer = 0;
    _currentPhase = 0;
    startLayerPhase();
}

// ================================================================
//  startLayerPhase  (_run_all_to_all_phase)
// ================================================================
void GpuNode::startLayerPhase() {
    int layer = _currentLayer, phase = _currentPhase;
    auto key = std::make_pair(layer, phase);

    std::cout << "[Node " << (int)_nodeId << "] L" << layer
              << " " << phaseName(phase) << " START\n";

    TxState& tx = _txState[key];
    tx = TxState{};
    for (uint8_t pid : _peerIds) {
        TxPeerState& ps = tx.peers[pid];
        for (int f = 0; f < (int)TOTAL_FRAGMENTS; f++) {
            ps.pending.insert(f);
            ps.attempts[f] = 0;
        }
    }
    // 不要无条件重置 _rxState!
    // 快节点可能在本节点开始当前阶段之前就已经发完数据,
    // processDataPacket 已将它们记入 _rxState[key].completedPeers.
    // 如果这里清零, 这些 COMPLETE 将永远丢失 (因为 _processedSet 会阻止重新计数),
    // 导致 barrierMet 永远为 false → 本节点永远无法完成该阶段.
    if (_rxState.find(key) == _rxState.end())
        _rxState[key] = RxState{};

    for (uint8_t pid : _peerIds)
        sendAllFragments(pid, layer, phase);

    // 如果早到的数据已满足 RX barrier, 立即检查是否可以完成阶段
    if (_rxState[key].barrierMet)
        checkPhaseComplete(layer, phase);
}

// ================================================================
//  sendAllFragments  (_sender_worker 首轮)
// ================================================================
void GpuNode::sendAllFragments(uint8_t targetId, int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key)) return;
    TxState& tx = _txState[key];
    if (tx.peers[targetId].done) return;

    uint64_t roundId = MoePacket::encodeRound(layer, phase);
    TxPeerState& ps = tx.peers[targetId];
    ps.batchSendTime = EventList::now();   // 记录本批次发送时刻
    std::vector<int> frags(ps.pending.begin(), ps.pending.end());
    std::sort(frags.begin(), frags.end());
    for (int fid : frags)
        sendFragment(targetId, layer, phase, (uint16_t)fid, roundId);
    //打印信息
    // std::cout << "ARM:   [Node " << (int)_nodeId << "] SEND -> Node " << (int)targetId
    //           << " L" << layer << " " << phaseName(phase)
    //           << " frags=" << frags.size() <<"start:"+std::to_string(ps.batchSendTime/1e9) + "\n";


    _retxTimers[targetId]->arm(TIMEOUT_PS, [this, targetId, layer, phase]() {
        retransmitCheck(targetId, layer, phase);
    });
}

// ================================================================
//  sendFragment  (create_packet + _enqueue_tx for DATA)
// ================================================================
void GpuNode::sendFragment(uint8_t targetId, int layer, int phase,
                            uint16_t fragId, uint64_t roundId)
{
    uint32_t offset = (uint32_t)fragId * FRAGMENT_PAYLOAD_SIZE;
    uint32_t remain = PAYLOAD_BYTES_PER_TARGET - offset;
    int fragSize = (int)std::min(FRAGMENT_PAYLOAD_SIZE, remain);
    int pktSize  = 14 + 15 + fragSize;

    assert(_routes.count(targetId) && "Route not set for target");
    MoePacket* pkt = MoePacket::newpkt(
        *_flow, *_routes[targetId],
        MoePacket::PKT_DATA, roundId, _nodeId, targetId, fragId, pktSize);

    // 入 txQueue → 串行化 → sendOn() → Route[0] (第一个 Tofino ingress)
    _txQueue->receivePacket(*pkt);

    _txState[{layer, phase}].peers[targetId].attempts[fragId]++;
}

// ================================================================
//  retransmitCheck
// ================================================================
void GpuNode::retransmitCheck(uint8_t targetId, int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key)) return;
    if (layer < _currentLayer || (layer == _currentLayer && phase < _currentPhase))
        return;

    TxState& tx = _txState[key];
    TxPeerState& ps = tx.peers[targetId];
    if (ps.done || ps.pending.empty()){ return;}
    //获取当前时间
    double nowMs = (double)EventList::now() / 1e9; // ps → ms
    //减去ps.batchSendTime得到RTT
    double rttMs = nowMs - ps.batchSendTime/1e9; // ps → ms

    std::cout << "[Node " << (int)_nodeId << "] RETRY -> Node " << (int)targetId
              << " L" << layer << " " << phaseName(phase)
              << " pending=" << ps.pending.size() << " RTT=" << rttMs <<"ms\n";

    uint64_t roundId = MoePacket::encodeRound(layer, phase);
    std::vector<int> frags(ps.pending.begin(), ps.pending.end());
    std::sort(frags.begin(), frags.end());
    for (int fid : frags)
        sendFragment(targetId, layer, phase, (uint16_t)fid, roundId);

    _retxTimers[targetId]->arm(TIMEOUT_PS, [this, targetId, layer, phase]() {
        retransmitCheck(targetId, layer, phase);
    });
}

// ================================================================
//  onPacketReceived  (AppSink → 应用层)
// ================================================================
void GpuNode::onPacketReceived(Packet& rawPkt) {
    MoePacket& pkt = static_cast<MoePacket&>(rawPkt);
    int layer, phase;
    MoePacket::decodeRound(pkt.roundId, layer, phase);

    if (pkt.pktType == MoePacket::PKT_DATA)
        processDataPacket(pkt, layer, phase);
    else
        processAckPacket(pkt, layer, phase);
}

// ================================================================
//  processDataPacket
// ================================================================
void GpuNode::processDataPacket(MoePacket& pkt, int layer, int phase) {
    uint8_t  srcId  = pkt.srcId;
    uint16_t fragId = pkt.fragId;
    uint16_t totalFrags = pkt.totalFrags;

    sendAck(srcId, layer, phase, fragId);

    auto dedup = std::make_tuple((int)srcId, layer, phase);
    if (_processedSet.count(dedup)) { pkt.free(); return; }

    auto& fragSet = _rxFragTracker[dedup];
    if (fragSet.count((int)fragId)) { pkt.free(); return; }
    fragSet.insert((int)fragId);
    pkt.free();

    if ((int)fragSet.size() < totalFrags) return;

    _processedSet.insert(dedup);
    std::cout << "[Node " << (int)_nodeId << "] COMPLETE from Node "
              << (int)srcId << " L" << layer << " " << phaseName(phase) << "\n";

    auto key = std::make_pair(layer, phase);
    RxState& rx = _rxState[key];
    rx.completedPeers.insert(srcId);
    if (rx.completedPeers.size() >= _peerIds.size()) {
        rx.barrierMet = true;
        checkPhaseComplete(layer, phase);
    }
}

// ================================================================
//  sendAck
// ================================================================
void GpuNode::sendAck(uint8_t targetId, int layer, int phase, uint16_t fragId) {
    assert(_routes.count(targetId));
    MoePacket* ack = MoePacket::newpkt(
        *_flow, *_routes[targetId],
        MoePacket::PKT_ACK,
        MoePacket::encodeRound(layer, phase),
        _nodeId, targetId, fragId, ACK_PKT_SIZE);
    _txQueue->receivePacket(*ack);
}

// ================================================================
//  processAckPacket
// ================================================================
void GpuNode::processAckPacket(MoePacket& pkt, int layer, int phase) {
    uint8_t srcId = pkt.srcId;
    int fragId = (int)pkt.fragId;
    pkt.free();

    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key) || !_txState.count(key)) return;
    TxState& tx = _txState[key];
    if (!tx.peers.count(srcId) || tx.peers[srcId].done) return;

    TxPeerState& ps = tx.peers[srcId];
    ps.pending.erase(fragId);

    // ---- DEBUG: 打印每个 ACK 的批次 RTT ----
    {
        simtime_picosec now = EventList::now();
        double rttMs = (double)(now - ps.batchSendTime) / 1e9; // ps → ms
        // if(rttMs > 100.0) // 只打印 RTT > 1 ms 的 ACK
        // std::cout << "[Node " << (int)_nodeId << "] ACK from Node " << (int)srcId
        //           << " frag=" << fragId
        //           << " L" << layer << " " << phaseName(phase)
        //           << " RTT=" << std::fixed << std::setprecision(3) << rttMs << "ms"
        //           << "\n";
    }

    if (!ps.pending.empty()) return;

    ps.done = true;
    tx.doneCount++;
    //_retxTimers[srcId]->cancel();
    _stats.recordTaskDone(layer, phase, (int)srcId, ps.attempts);

    std::cout << "[Node " << (int)_nodeId << "] TX DONE -> Node "
              << (int)srcId << " L" << layer << " " << phaseName(phase)
              << " (" << tx.doneCount << "/" << _peerIds.size() << ")\n";

    checkPhaseComplete(layer, phase);
}

// ================================================================
//  checkPhaseComplete
// ================================================================
void GpuNode::checkPhaseComplete(int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key)) return;
    auto& tx = _txState[key];
    auto& rx = _rxState[key];
    if (tx.doneCount < _peerIds.size() || !rx.barrierMet) return;

    _phaseFinished.insert(key);
    std::cout << "[Node " << (int)_nodeId << "] Phase COMPLETE L"
              << layer << " " << phaseName(phase) << "\n";
    advancePhase();
}

// ================================================================
//  advancePhase
// ================================================================
void GpuNode::advancePhase() {
    if (_currentPhase == 0) {
        _currentPhase = 1;
        _gapTimer.arm(INTERPHASE_GAP_PS, [this]() { startLayerPhase(); });
    } else {
        _currentPhase = 0;
        _currentLayer++;
        if (_currentLayer < (int)TOTAL_LAYERS) {
            // 层间间隙: 等待网络排空, 与 DISPATCH→COMBINE 的 gap 对称
            _gapTimer.arm(INTERPHASE_GAP_PS, [this]() { startLayerPhase(); });
        } else {
            std::cout << "[Node " << (int)_nodeId << "] INFERENCE FINISHED  t="
                      << timeAsMs(EventList::now()) << " ms\n";
            _stats.printSummary();
            _done = true;
            if (++s_doneCount >= NUM_GPU_NODES) {
                std::cout << "All " << NUM_GPU_NODES << " nodes finished.\n";
                printGlobalSummary();
                EventList::setEndtime(EventList::now());
            }
        }
    }
}
