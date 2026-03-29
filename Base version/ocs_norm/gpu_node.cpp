/*
 * gpu_node.cpp  —  EcsBuffer HOHO version
 *
 * GPU sends ALL targets in parallel at phase start.
 * The network (EcsBuffer + BFS routing) handles routing and timing.
 * GPU just injects packets and waits for ACKs.
 *
 * Data flow:
 *   sendAllFragments(target) → txQueue → sendOn → EcsBuffer@src
 *     → [HOHO routing: FORWARD/WAIT via OCS] → EcsBuffer@dst
 *     → sendOn → rxQueue → AppSink → processDataPacket → sendAck
 */

#include "gpu_node.h"
#include "hoho_routing.h"
#include <algorithm>
#include <iostream>

PacketDB<MoePacket> MoePacket::_db;
packetid_t          MoePacket::_nextId = 1;
uint32_t            GpuNode::s_doneCount = 0;

void AppSink::receivePacket(Packet& pkt) {
    _gpu->onPacketReceived(pkt);
}

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

    _txQueue = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);
    _rxQueue = new Queue(LINK_SPEED_BPS, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);

    for (uint8_t pid : peerIds)
        _retxTimers[pid] = new TimerEvent(
            "retx_" + std::to_string(nodeId) + "_" + std::to_string((int)pid));

    _startEvent.arm(0, [this]() { startInference(); });
}

void GpuNode::startInference() {
    std::cout << "[Node " << (int)_nodeId << "] INFERENCE START (EcsBuffer HOHO) -- "
              << TOTAL_LAYERS << " layers, "
              << TOTAL_FRAGMENTS << " frags/target, "
              << (FRAGMENT_PAYLOAD_SIZE/1024) << " KB/frag\n";
    _currentLayer = 0;
    _currentPhase = 0;
    startLayerPhase();
}

// ================================================================
//  startLayerPhase — send ALL targets in parallel
//
//  Unlike sequential model, GPU injects data for all 7 targets at once.
//  HohoForwarder handles routing (direct or multi-hop).
//  OcsCircuits handle timing (slice-gated transmission).
// ================================================================
void GpuNode::startLayerPhase() {
    int layer = _currentLayer, phase = _currentPhase;
    auto key = std::make_pair(layer, phase);

    std::cout << "[Node " << (int)_nodeId << "] L" << layer
              << " " << phaseName(phase) << " START (EcsBuffer HOHO)\n";

    // Initialize TX state for all targets
    TxState& tx = _txState[key];
    tx = TxState{};
    for (uint8_t pid : _peerIds) {
        TxPeerState& ps = tx.peers[pid];
        for (int f = 0; f < (int)TOTAL_FRAGMENTS; f++) {
            ps.pending.insert(f);
            ps.attempts[f] = 0;
        }
    }
    if (_rxState.find(key) == _rxState.end())
        _rxState[key] = RxState{};

    // Send ALL targets in parallel — network handles routing & timing
    for (uint8_t target : _peerIds) {
        sendAllFragments(target, layer, phase);
    }

    // Check if RX barrier already met (early data from fast peers)
    if (_rxState[key].barrierMet)
        checkPhaseComplete(layer, phase);
}

// ================================================================
//  sendAllFragments
// ================================================================
void GpuNode::sendAllFragments(uint8_t targetId, int layer, int phase) {
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
    if (ps.done || ps.pending.empty()) return;

    double nowMs = (double)EventList::now() / 1e9;
    double rttMs = nowMs - ps.batchSendTime/1e9;

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
//  processAckPacket — simplified (no slot tracking)
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

    if (!ps.pending.empty()) return;

    ps.done = true;
    tx.doneCount++;
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
              << layer << " " << phaseName(phase) << " (EcsBuffer HOHO)\n";
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
            _gapTimer.arm(INTERPHASE_GAP_PS, [this]() { startLayerPhase(); });
        } else {
            std::cout << "[Node " << (int)_nodeId << "] INFERENCE FINISHED (EcsBuffer HOHO)  t="
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
