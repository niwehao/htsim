/*
 * gpu_node_sync.cpp  —  N-GPU Dragonfly 同步调度版 GPU 节点实现
 *
 * 按预计算 schedule 定时发送, 无 burst, 无重传.
 * ACK 仅用于验证和阶段完成判断.
 */

#include "gpu_node_sync.h"
#include <algorithm>
#include <iostream>

PacketDB<MoePacket> MoePacket::_db;
packetid_t          MoePacket::_nextId = 1;
uint32_t            GpuNodeSync::s_doneCount = 0;
uint32_t            GpuNodeSync::s_phaseReportCount = 0;

// ================================================================
//  AppSinkSync::receivePacket
// ================================================================
void AppSinkSync::receivePacket(Packet& pkt) {
    _gpu->onPacketReceived(pkt);
}

// ================================================================
//  构造
// ================================================================
GpuNodeSync::GpuNodeSync(uint16_t nodeId, const std::vector<uint16_t>& peerIds,
                          const std::vector<ScheduleEntry>& phaseSchedule)
    : _nodeId(nodeId)
    , _peerIds(peerIds)
    , _appSink("gpu" + std::to_string(nodeId) + "_app")
    , _phaseSchedule(phaseSchedule)
    , _sendTimer("send_" + std::to_string(nodeId))
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

    _startEvent.arm(0, [this]() { startInference(); });
}

// ================================================================
//  startInference
// ================================================================
void GpuNodeSync::startInference() {
    if ((_nodeId % PRINT_MOD) == 0 && LOG_LEVEL >= 1)
        std::cout << "[Node " << _nodeId << "] INFERENCE START (DRAGONFLY SYNC) — "
                  << TOTAL_LAYERS << " layers, "
                  << TOTAL_FRAGMENTS << " frags/target, "
                  << _peerIds.size() << " peers\n";
    _currentLayer = 0;
    _currentPhase = 0;
    startLayerPhase();
}

// ================================================================
//  startLayerPhase
// ================================================================
void GpuNodeSync::startLayerPhase() {
    int layer = _currentLayer, phase = _currentPhase;
    auto key = std::make_pair(layer, phase);

    TxState& tx = _txState[key];
    tx = TxState{};
    for (uint16_t pid : _peerIds)
        tx.peers[pid] = TxPeerState{};

    if (_rxState.find(key) == _rxState.end())
        _rxState[key] = RxState{};

    _schedIdx = 0;
    _phaseStartTime = EventList::now();
    processSchedule();

    if (_rxState[key].barrierMet)
        checkPhaseComplete(layer, phase);
}

// ================================================================
//  processSchedule — 按预计算时间逐包发送
// ================================================================
void GpuNodeSync::processSchedule() {
    uint64_t roundId = MoePacket::encodeRound(_currentLayer, _currentPhase);

    while (_schedIdx < _phaseSchedule.size()) {
        auto& entry = _phaseSchedule[_schedIdx];
        simtime_picosec absTime = _phaseStartTime + entry.sendTime;

        if (absTime > EventList::now()) {
            _sendTimer.arm(absTime - EventList::now(), [this]() { processSchedule(); });
            return;
        }

        uint32_t offset = (uint32_t)entry.fragId * FRAGMENT_PAYLOAD_SIZE;
        uint32_t remain = PAYLOAD_BYTES_PER_TARGET - offset;
        int fragSize = (int)std::min(FRAGMENT_PAYLOAD_SIZE, remain);
        int pktSize  = 14 + 15 + fragSize;

        assert(_routes.count(entry.dstGpu) && "Route not set for target");
        MoePacket* pkt = MoePacket::newpkt(
            *_flow, *_routes[entry.dstGpu],
            MoePacket::PKT_DATA, roundId, _nodeId, entry.dstGpu, entry.fragId, pktSize);

        _txQueue->receivePacket(*pkt);
        _schedIdx++;
    }
}

// ================================================================
//  onPacketReceived
// ================================================================
void GpuNodeSync::onPacketReceived(Packet& rawPkt) {
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
void GpuNodeSync::processDataPacket(MoePacket& pkt, int layer, int phase) {
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
    if (_nodeId % PRINT_MOD == 0 && srcId % PRINT_MOD == 0) {
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
void GpuNodeSync::sendAck(uint16_t targetId, int layer, int phase, uint16_t fragId) {
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
void GpuNodeSync::processAckPacket(MoePacket& pkt, int layer, int phase) {
    uint16_t srcId = pkt.srcId;
    int fragId = (int)pkt.fragId;
    pkt.free();

    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key) || !_txState.count(key)) return;
    TxState& tx = _txState[key];
    if (!tx.peers.count(srcId) || tx.peers[srcId].done) return;

    TxPeerState& ps = tx.peers[srcId];
    ps.ackedFrags.insert(fragId);

    if ((int)ps.ackedFrags.size() < (int)TOTAL_FRAGMENTS) return;

    ps.done = true;
    tx.doneCount++;

    std::map<int,int> fragAttempts;
    for (int f = 0; f < (int)TOTAL_FRAGMENTS; f++)
        fragAttempts[f] = 1;
    _stats.recordTaskDone(layer, phase, (int)srcId, fragAttempts);

    if (_nodeId % PRINT_MOD == 0 && srcId % PRINT_MOD == 0) {
        std::cout << "[Node " << (int)_nodeId << "] Receive ACK DONE -> Node "
                  << (int)srcId << " L" << layer << " " << phaseName(phase)
                  << " (" << tx.doneCount << "/" << _peerIds.size() << ")\n";
    }
    checkPhaseComplete(layer, phase);
}

// ================================================================
//  checkPhaseComplete
// ================================================================
void GpuNodeSync::checkPhaseComplete(int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key)) return;
    auto& tx = _txState[key];
    auto& rx = _rxState[key];
    if (tx.doneCount < _peerIds.size() || !rx.barrierMet) return;

    _phaseFinished.insert(key);

    s_phaseReportCount++;
    if (s_phaseReportCount % NUM_GPU_NODES == 0) {
        uint32_t phaseNum = s_phaseReportCount / NUM_GPU_NODES;
        std::cout << "  Phase " << phaseNum << "/4: All " << NUM_GPU_NODES
                  << " GPUs completed L" << layer << " " << phaseName(phase)
                  << "  t=" << std::fixed << std::setprecision(2)
                  << timeAsMs(EventList::now()) << " ms\n";
    }

    cleanupPhase(layer, phase);
    advancePhase();
}

// ================================================================
//  cleanupPhase
// ================================================================
void GpuNodeSync::cleanupPhase(int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    _txState.erase(key);
    _rxState.erase(key);

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
void GpuNodeSync::advancePhase() {
    if (_currentPhase == 0) {
        _currentPhase = 1;
        _gapTimer.arm(INTERPHASE_GAP_PS, [this]() { startLayerPhase(); });
    } else {
        _currentPhase = 0;
        _currentLayer++;
        if (_currentLayer < (int)TOTAL_LAYERS) {
            _gapTimer.arm(INTERPHASE_GAP_PS, [this]() { startLayerPhase(); });
        } else {
            if ((_nodeId % PRINT_MOD) == 0 && LOG_LEVEL >= 1) {
                std::cout << "[Node " << _nodeId << "] INFERENCE FINISHED (DRAGONFLY SYNC)  t="
                          << timeAsMs(EventList::now()) << " ms\n";
            }
            _done = true;
            if (++s_doneCount >= NUM_GPU_NODES) {
                std::cout << "\nAll " << NUM_GPU_NODES << " nodes finished.\n";
                printGlobalSummary();
                EventList::setEndtime(EventList::now());
            }
        }
    }
}
