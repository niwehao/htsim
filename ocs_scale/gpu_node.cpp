/*
 * gpu_node.cpp  —  Scalable GPU node with dynamic MoE target selection
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

GpuNode::GpuNode(uint16_t nodeId)
    : _nodeId(nodeId)
    , _appSink("gpu" + std::to_string(nodeId) + "_app")
    , _gapTimer("gap_" + std::to_string(nodeId))
    , _startEvent("start_" + std::to_string(nodeId))
{
    _appSink.setGpu(this);
    _stats.nodeId = nodeId;
    g_allStats.push_back(&_stats);

    // GPU↔EcsBuffer: instant transfer (very high speed)
    _txQueue = new Queue(GPU_LOCAL_SPEED, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);
    _rxQueue = new Queue(GPU_LOCAL_SPEED, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);

    _startEvent.arm(0, [this]() { startInference(); });
}

void GpuNode::startInference() {
    if (VERBOSE_LOG) {
        std::cout << "[Node " << (int)_nodeId << "] INFERENCE START -- "
                  << TOTAL_LAYERS << " layers\n";
    }
    _currentLayer = 0;
    _currentPhase = 0;
    startLayerPhase();
}

void GpuNode::startLayerPhase() {
    int layer = _currentLayer, phase = _currentPhase;
    auto key = std::make_pair(layer, phase);

    // Get per-phase targets
    auto cfgIt = _phaseConfigs.find(key);
    assert(cfgIt != _phaseConfigs.end() && "Phase config not set");
    const PhaseConfig& cfg = cfgIt->second;

    if (VERBOSE_LOG) {
        std::cout << "[Node " << (int)_nodeId << "] L" << layer
                  << " " << phaseName(phase) << " START -> "
                  << cfg.send_targets.size() << " targets, recv from "
                  << cfg.recv_from.size() << "\n";
    }

    // Initialize TX state
    TxState& tx = _txState[key];
    tx = TxState{};
    tx.expectedCount = (uint32_t)cfg.send_targets.size();
    for (uint16_t pid : cfg.send_targets) {
        TxPeerState& ps = tx.peers[pid];
        for (int f = 0; f < (int)TOTAL_FRAGMENTS; f++) {
            ps.pending.insert(f);
            ps.attempts[f] = 0;
        }
    }

    // Initialize RX state
    RxState& rx = _rxState[key];
    if (rx.expectedCount == 0) {
        rx.expectedCount = (uint32_t)cfg.recv_from.size();
        if (rx.expectedCount == 0) {
            rx.barrierMet = true;  // no one sends to us → barrier trivially met
        } else if (rx.completedPeers.size() >= rx.expectedCount) {
            rx.barrierMet = true;  // early arrivals already met the barrier
        }
    }

    // Send ALL targets in parallel
    for (uint16_t target : cfg.send_targets) {
        // Create retx timer if needed
        if (_retxTimers.find(target) == _retxTimers.end()) {
            _retxTimers[target] = new TimerEvent(
                "retx_" + std::to_string(_nodeId) + "_" + std::to_string(target));
        }
        sendAllFragments(target, layer, phase);
    }

    // Check if RX barrier already met
    if (rx.expectedCount == 0 || rx.barrierMet)
        checkPhaseComplete(layer, phase);
}

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

void GpuNode::retransmitCheck(uint16_t targetId, int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key)) return;
    if (layer < _currentLayer || (layer == _currentLayer && phase < _currentPhase))
        return;

    TxState& tx = _txState[key];
    TxPeerState& ps = tx.peers[targetId];
    if (ps.done || ps.pending.empty()) return;

    double nowMs = (double)EventList::now() / 1e9;

    std::cout << "[Node " << (int)_nodeId << "] RETRY -> Node " << (int)targetId
              << " L" << layer << " " << phaseName(phase)
              << " pending=" << ps.pending.size() << " t=" << nowMs << "ms\n";

    uint64_t roundId = MoePacket::encodeRound(layer, phase);
    std::vector<int> frags(ps.pending.begin(), ps.pending.end());
    std::sort(frags.begin(), frags.end());
    for (int fid : frags)
        sendFragment(targetId, layer, phase, (uint16_t)fid, roundId);

    _retxTimers[targetId]->arm(TIMEOUT_PS, [this, targetId, layer, phase]() {
        retransmitCheck(targetId, layer, phase);
    });
}

void GpuNode::onPacketReceived(Packet& rawPkt) {
    MoePacket& pkt = static_cast<MoePacket&>(rawPkt);
    int layer, phase;
    MoePacket::decodeRound(pkt.roundId, layer, phase);

    if (pkt.pktType == MoePacket::PKT_DATA)
        processDataPacket(pkt, layer, phase);
    else
        processAckPacket(pkt, layer, phase);
}

void GpuNode::processDataPacket(MoePacket& pkt, int layer, int phase) {
    uint16_t srcId  = pkt.srcId;
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
    if (VERBOSE_LOG) {
        std::cout << "[Node " << (int)_nodeId << "] COMPLETE from Node "
                  << (int)srcId << " L" << layer << " " << phaseName(phase) << "\n";
    }

    auto key = std::make_pair(layer, phase);
    RxState& rx = _rxState[key];
    rx.completedPeers.insert(srcId);
    if (rx.expectedCount > 0 && rx.completedPeers.size() >= rx.expectedCount) {
        rx.barrierMet = true;
        checkPhaseComplete(layer, phase);
    }
}

void GpuNode::sendAck(uint16_t targetId, int layer, int phase, uint16_t fragId) {
    assert(_routes.count(targetId));
    MoePacket* ack = MoePacket::newpkt(
        *_flow, *_routes[targetId],
        MoePacket::PKT_ACK,
        MoePacket::encodeRound(layer, phase),
        _nodeId, targetId, fragId, ACK_PKT_SIZE);
    _txQueue->receivePacket(*ack);
}

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

    if (VERBOSE_LOG) {
        std::cout << "[Node " << (int)_nodeId << "] TX DONE -> Node "
                  << (int)srcId << " L" << layer << " " << phaseName(phase)
                  << " (" << tx.doneCount << "/" << tx.expectedCount << ")\n";
    }

    checkPhaseComplete(layer, phase);
}

void GpuNode::checkPhaseComplete(int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key)) return;
    auto& tx = _txState[key];
    auto& rx = _rxState[key];
    if (tx.doneCount < tx.expectedCount || !rx.barrierMet) return;

    _phaseFinished.insert(key);
    if (VERBOSE_LOG) {
        std::cout << "[Node " << (int)_nodeId << "] Phase COMPLETE L"
                  << layer << " " << phaseName(phase) << "\n";
    }
    advancePhase();
}

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
