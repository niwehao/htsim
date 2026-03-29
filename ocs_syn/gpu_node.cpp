/*
 * gpu_node.cpp  —  GPU node with synchronized scheduling
 *
 * Key difference: startLayerPhase() uses the pre-computed schedule
 * to send each flow at the optimal time slot, instead of dumping
 * all flows immediately.
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

GpuNode::GpuNode(uint16_t nodeId, DynOcsTopology* topo)
    : _nodeId(nodeId)
    , _topo(topo)
    , _appSink("gpu" + std::to_string(nodeId) + "_app")
    , _gapTimer("gap_" + std::to_string(nodeId))
    , _startEvent("start_" + std::to_string(nodeId))
{
    _appSink.setGpu(this);
    _stats.nodeId = nodeId;
    g_allStats.push_back(&_stats);

    _txQueue = new Queue(GPU_LOCAL_SPEED, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);
    _rxQueue = new Queue(GPU_LOCAL_SPEED, HUGE_BUFFER,
                          EventList::getTheEventList(), nullptr);

    _startEvent.arm(0, [this]() { startInference(); });
}

void GpuNode::startInference() {
    if (VERBOSE_LOG) {
        std::cout << "[Node " << (int)_nodeId << "] INFERENCE START -- "
                  << TOTAL_LAYERS << " layers, all-to-all ("
                  << NUM_ACTIVE_EXPERTS << " targets)\n";
    }
    _currentLayer = 0;
    _currentPhase = 0;
    startLayerPhase();
}

void GpuNode::startLayerPhase() {
    int layer = _currentLayer, phase = _currentPhase;
    auto key = std::make_pair(layer, phase);

    auto cfgIt = _phaseConfigs.find(key);
    assert(cfgIt != _phaseConfigs.end() && "Phase config not set");
    const PhaseConfig& cfg = cfgIt->second;

    if (VERBOSE_LOG) {
        std::cout << "[Node " << (int)_nodeId << "] L" << layer
                  << " " << phaseName(phase) << " START -> "
                  << cfg.send_targets.size() << " targets\n";
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
            rx.barrierMet = true;
        } else if (rx.completedPeers.size() >= rx.expectedCount) {
            rx.barrierMet = true;
        }
    }

    // --- Synchronized scheduling: send each flow at its optimal time ---
    simtime_picosec now = EventList::now();
    int S = _topo->num_slices();
    int absSlice = _topo->time_to_absolute_slice(now);
    simtime_picosec sliceStart = _topo->get_slice_start_time(absSlice);

    // Align to next slice boundary if we're mid-slice
    if (now > sliceStart)
        absSlice++;

    int base = absSlice % S;

    auto schedIt = _flowSchedule.phases.find(key);
    assert(schedIt != _flowSchedule.phases.end() && "Flow schedule not set");
    auto& entries = schedIt->second[base];

    for (auto& entry : entries) {
        uint16_t target = entry.target;
        simtime_picosec targetTime = _topo->get_slice_start_time(
            absSlice + entry.delaySlots);
        simtime_picosec delay = (targetTime > now) ? (targetTime - now) : 0;

        // Create send timer if needed
        if (_sendTimers.find(target) == _sendTimers.end())
            _sendTimers[target] = new TimerEvent(
                "send_" + std::to_string(_nodeId) + "_" + std::to_string(target));

        // Create retx timer if needed (safety net)
        if (_retxTimers.find(target) == _retxTimers.end())
            _retxTimers[target] = new TimerEvent(
                "retx_" + std::to_string(_nodeId) + "_" + std::to_string(target));

        if (delay == 0) {
            sendAllFragments(target, layer, phase);
        } else {
            _sendTimers[target]->arm(delay,
                [this, target, layer, phase]() {
                    sendAllFragments(target, layer, phase);
                });
        }
    }

    // Check if RX barrier already met
    if (rx.barrierMet)
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

    // With synchronized scheduling, retransmission indicates a problem
    std::cerr << "ERROR: [Node " << (int)_nodeId << "] RETRANSMIT needed -> Node "
              << (int)targetId << " L" << layer << " " << phaseName(phase)
              << " pending=" << ps.pending.size()
              << " t=" << std::fixed << std::setprecision(2)
              << (double)EventList::now() / 1e9 << "ms\n";

    // Retransmit anyway to keep simulation running
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

    checkPhaseComplete(layer, phase);
}

void GpuNode::checkPhaseComplete(int layer, int phase) {
    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key)) return;
    auto& tx = _txState[key];
    auto& rx = _rxState[key];
    if (tx.doneCount < tx.expectedCount || !rx.barrierMet) return;

    _phaseFinished.insert(key);

    // Clean up completed phase state to save memory
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
            _done = true;
            if (++s_doneCount >= NUM_GPU_NODES) {
                std::cout << "\nAll " << NUM_GPU_NODES
                      << " nodes finished at t="
                      << timeAsMs(EventList::now()) << " ms\n";
                printGlobalSummary();
                EventList::setEndtime(EventList::now() + 1);
            }
        }
    }
}
