/*
 * gpu_node.cpp  —  512-GPU Leaf-Spine 版
 *
 * 与 htsim_v2 逻辑一致, 主要变更:
 *   - 0-indexed uint16_t GPU ID
 *   - 阶段完成后清理状态 (cleanupPhase), 控制内存
 *   - 精简日志: 只打印里程碑 (推理开始/结束, 层完成, 重传)
 */

#include "gpu_node.h"
#include <algorithm>
#include <iostream>
#include <random>

// ---- 静态成员 ----
PacketDB<MoePacket> MoePacket::_db;
packetid_t          MoePacket::_nextId = 1;
uint32_t            GpuNode::s_doneCount = 0;
uint32_t            GpuNode::s_phaseReportCount = 0;

// ================================================================
//  AppSink::receivePacket
// ================================================================
void AppSink::receivePacket(Packet& pkt) {
    _gpu->onPacketReceived(pkt);
}

// ================================================================
//  构造
// ================================================================
GpuNode::GpuNode(uint16_t nodeId, const std::vector<uint16_t>& peerIds)
    : _nodeId(nodeId)
    , _peerIds(peerIds)
    , _appSink("gpu" + std::to_string(nodeId) + "_app")
    , _gapTimer("gap_" + std::to_string(nodeId))
    , _startEvent("start_" + std::to_string(nodeId))
{
    _appSink.setGpu(this);
    _stats.nodeId = nodeId;
    g_allStats.push_back(&_stats);

    _txQueue = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);
    _rxQueue = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);

    for (uint16_t pid : peerIds)
        _retxTimers[pid] = new TimerEvent(
            "retx_" + std::to_string(nodeId) + "_" + std::to_string((int)pid));

    _startEvent.arm(0, [this]() { startInference(); });
}

// ================================================================
//  startInference
// ================================================================
void GpuNode::startInference() {
    if (_nodeId%PRINT_MOD ) {
        std::cout << "[Node " << _nodeId << "] INFERENCE START — "
                  << TOTAL_LAYERS << " layers, "
                  << TOTAL_FRAGMENTS << " frags/target, "
                  << _peerIds.size() << " peers\n";
    }
    _currentLayer = 0;
    _currentPhase = 0;
    startLayerPhase();
}

// ================================================================
//  startLayerPhase
// ================================================================
void GpuNode::startLayerPhase() {
    int layer = _currentLayer, phase = _currentPhase;
    auto key = std::make_pair(layer, phase);

    TxState& tx = _txState[key];
    tx = TxState{};
    for (uint16_t pid : _peerIds) {
        TxPeerState& ps = tx.peers[pid];
        for (int f = 0; f < (int)TOTAL_FRAGMENTS; f++) {
            ps.pending.insert(f);
            ps.attempts[f] = 0;
        }
    }
    if (_rxState.find(key) == _rxState.end())
        _rxState[key] = RxState{};

    // 乱序发射
    std::vector<uint16_t> shuffled(_peerIds.begin(), _peerIds.end());
    std::shuffle(shuffled.begin(), shuffled.end(),
                std::mt19937{std::random_device{}()});
    for (uint16_t pid : shuffled)
        sendAllFragments(pid, layer, phase);


    if (_rxState[key].barrierMet)
        checkPhaseComplete(layer, phase);
}

// ================================================================
//  sendAllFragments
// ================================================================
void GpuNode::sendAllFragments(uint16_t targetId, int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key)) return;
    TxState& tx = _txState[key];
    if (tx.peers[targetId].done) return;

    uint64_t roundId = MoePacket::encodeRound(layer, phase);
    TxPeerState& ps = tx.peers[targetId];
    ps.batchSendTime = EventList::now();
    std::vector<int> frags(ps.pending.begin(), ps.pending.end());
    std::sort(frags.begin(), frags.end());
    for (int fid : frags)
        sendFragment(targetId, layer, phase, (uint16_t)fid, roundId);

    _retxTimers[targetId]->arm(TIMEOUT_PS, [this, targetId, layer, phase]() {
        retransmitCheck(targetId, layer, phase);
    });
}

// ================================================================
//  sendFragment
// ================================================================
void GpuNode::sendFragment(uint16_t targetId, int layer, int phase,
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

    _txQueue->receivePacket(*pkt);
    _txState[{layer, phase}].peers[targetId].attempts[fragId]++;
}

// ================================================================
//  retransmitCheck
// ================================================================
void GpuNode::retransmitCheck(uint16_t targetId, int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key)) return;
    if (layer < _currentLayer || (layer == _currentLayer && phase < _currentPhase))
        return;

    TxState& tx = _txState[key];
    TxPeerState& ps = tx.peers[targetId];
    if (ps.done || ps.pending.empty()) return;

    double nowMs = (double)EventList::now() / 1e9;
    double rttMs = nowMs - ps.batchSendTime / 1e9;

    // std::cout << "[Node " << (int)_nodeId << "] RETRY -> Node " << (int)targetId
    //           << " L" << layer << " " << phaseName(phase)
    //           << " pending=" << ps.pending.size() << " RTT=" << rttMs << "ms\n";

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
//  onPacketReceived
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
    uint16_t srcId     = pkt.srcId;
    uint16_t fragId    = pkt.fragId;
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
    if (_nodeId%PRINT_MOD == 0 && srcId % PRINT_MOD == 0) {
        std::cout << "[Node " << (int)_nodeId << "] COMPLETE from Node "
                  << (int)srcId << " L" << layer << " " << phaseName(phase) << "\n";
    }
 

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
void GpuNode::sendAck(uint16_t targetId, int layer, int phase, uint16_t fragId) {
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
    uint16_t srcId = pkt.srcId;
    int fragId = (int)pkt.fragId;
    pkt.free();

    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key) || !_txState.count(key)) return;
    TxState& tx = _txState[key];
    if (!tx.peers.count(srcId) || tx.peers[srcId].done) return;

    TxPeerState& ps = tx.peers[srcId];
    ps.pending.erase(fragId);

    if (!ps.pending.empty()) return;

    ps.done = true;
    tx.doneCount++;
    _stats.recordTaskDone(layer, phase, (int)srcId, ps.attempts);
    if (_nodeId%PRINT_MOD == 0 && srcId % PRINT_MOD == 0) {
        std::cout << "[Node " << (int)_nodeId << "] Receive ACK DONE -> Node "
                  << (int)srcId << " L" << layer << " " << phaseName(phase)
                  << " (" << tx.doneCount << "/" << _peerIds.size() << ")\n";
    }


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

    // 汇总报告: 每当所有 GPU 完成一个 phase, 打印一行
    s_phaseReportCount++;
    // if (s_phaseReportCount == NUM_GPU_NODES) {
    //     std::cout << "  All " << NUM_GPU_NODES << " GPUs completed L"
    //               << layer << " " << phaseName(phase)
    //               << "  t=" << std::fixed << std::setprecision(2)
    //               << timeAsMs(EventList::now()) << " ms\n";
        
    // }

    cleanupPhase(layer, phase);
    advancePhase();
}

// ================================================================
//  cleanupPhase — 释放已完成阶段的状态, 控制内存
// ================================================================
void GpuNode::cleanupPhase(int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    _txState.erase(key);
    _rxState.erase(key);

    // 清理 _rxFragTracker 和 _processedSet 中该阶段的条目
    for (auto it = _rxFragTracker.begin(); it != _rxFragTracker.end(); ) {
        auto& [k, v] = *it;
        if (std::get<1>(k) == layer && std::get<2>(k) == phase)
            it = _rxFragTracker.erase(it);
        else
            ++it;
    }
    for (auto it = _processedSet.begin(); it != _processedSet.end(); ) {
        if (std::get<1>(*it) == layer && std::get<2>(*it) == phase)
            it = _processedSet.erase(it);
        else
            ++it;
    }
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
            _gapTimer.arm(INTERPHASE_GAP_PS, [this]() { startLayerPhase(); });
        } else {
            if (_nodeId%PRINT_MOD == 0) {
                std::cout << "[Node " << _nodeId << "] INFERENCE FINISHED  t="
                          << timeAsMs(EventList::now()) << " ms\n";
            }
            _stats.printSummary();
            _done = true;
            if (++s_doneCount >= NUM_GPU_NODES) {
                std::cout << "\nAll " << NUM_GPU_NODES << " nodes finished.\n";
                printGlobalSummary();
                EventList::setEndtime(EventList::now());
            }
        }
    }
}
