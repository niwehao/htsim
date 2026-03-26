/*
 * gpu_node_sync.cpp  —  同步调度版 GPU 节点实现
 */

#include "gpu_node_sync.h"
#include <algorithm>
#include <iostream>

// ---- 静态成员 ----
PacketDB<MoePacket> MoePacket::_db;
packetid_t          MoePacket::_nextId = 1;
uint32_t            GpuNodeSync::s_doneCount = 0;

// ================================================================
//  AppSinkSync::receivePacket
// ================================================================
void AppSinkSync::receivePacket(Packet& pkt) {
    _gpu->onPacketReceived(pkt);
}

// ================================================================
//  构造
// ================================================================
GpuNodeSync::GpuNodeSync(uint8_t nodeId, const std::vector<uint8_t>& peerIds,
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

    // t=0 启动推理
    _startEvent.arm(0, [this]() { startInference(); });
}

// ================================================================
//  startInference
// ================================================================
void GpuNodeSync::startInference() {
    if (LOG_LEVEL >= 1)
        std::cout << "[Node " << (int)_nodeId << "] INFERENCE START (SYNC) — "
                  << TOTAL_LAYERS << " layers, "
                  << TOTAL_FRAGMENTS << " frags/target, "
                  << (FRAGMENT_PAYLOAD_SIZE/1024) << " KB/frag\n";
    _currentLayer = 0;
    _currentPhase = 0;
    startLayerPhase();
}

// ================================================================
//  startLayerPhase — 启动一个阶段的调度发送
// ================================================================
void GpuNodeSync::startLayerPhase() {
    int layer = _currentLayer, phase = _currentPhase;
    auto key = std::make_pair(layer, phase);

    if (LOG_LEVEL >= 1)
        std::cout << "[Node " << (int)_nodeId << "] L" << layer
                  << " " << phaseName(phase) << " START\n";

    // 初始化 TX 状态
    TxState& tx = _txState[key];
    tx = TxState{};
    for (uint8_t pid : _peerIds)
        tx.peers[pid] = TxPeerState{};

    // 初始化 RX 状态 (不要覆盖已有的早到数据)
    if (_rxState.find(key) == _rxState.end())
        _rxState[key] = RxState{};

    // 按调度表开始发送
    _schedIdx = 0;
    _phaseStartTime = EventList::now();
    processSchedule();

    // 检查是否早到数据已满足 RX barrier
    if (_rxState[key].barrierMet)
        checkPhaseComplete(layer, phase);
}

// ================================================================
//  processSchedule — 按预计算时间逐包发送
//
//  处理所有当前时刻应发送的包, 然后 arm 定时器等待下一个发送时刻
// ================================================================
void GpuNodeSync::processSchedule() {
    uint64_t roundId = MoePacket::encodeRound(_currentLayer, _currentPhase);

    while (_schedIdx < _phaseSchedule.size()) {
        auto& entry = _phaseSchedule[_schedIdx];
        simtime_picosec absTime = _phaseStartTime + entry.sendTime;

        if (absTime > EventList::now()) {
            // 下一个包在未来 → 设置定时器
            _sendTimer.arm(absTime - EventList::now(), [this]() { processSchedule(); });
            return;
        }

        // 当前时刻发送此包
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
    // 所有包已发送完毕
}

// ================================================================
//  onPacketReceived  (AppSinkSync → 应用层)
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
    uint8_t  srcId  = pkt.srcId;
    uint16_t fragId = pkt.fragId;
    uint16_t totalFrags = pkt.totalFrags;

    // 回复 ACK
    sendAck(srcId, layer, phase, fragId);

    // 去重
    auto dedup = std::make_tuple((int)srcId, layer, phase);
    if (_processedSet.count(dedup)) { pkt.free(); return; }

    auto& fragSet = _rxFragTracker[dedup];
    if (fragSet.count((int)fragId)) { pkt.free(); return; }
    fragSet.insert((int)fragId);
    pkt.free();

    if ((int)fragSet.size() < totalFrags) return;

    // 该 peer 的所有 fragment 收齐
    _processedSet.insert(dedup);
    if (LOG_LEVEL >= 2)
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
void GpuNodeSync::sendAck(uint8_t targetId, int layer, int phase, uint16_t fragId) {
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
    uint8_t srcId = pkt.srcId;   // ACK 发送方 = 数据接收方
    int fragId = (int)pkt.fragId;
    pkt.free();

    auto key = std::make_pair(layer, phase);
    if (_phaseFinished.count(key) || !_txState.count(key)) return;
    TxState& tx = _txState[key];
    if (!tx.peers.count(srcId) || tx.peers[srcId].done) return;

    TxPeerState& ps = tx.peers[srcId];
    ps.ackedFrags.insert(fragId);

    if ((int)ps.ackedFrags.size() < (int)TOTAL_FRAGMENTS) return;

    // 该 peer 所有 fragment 均已 ACK
    ps.done = true;
    tx.doneCount++;

    // 记录统计 (同步模式下所有 fragment 应只发送一次)
    std::map<int,int> fragAttempts;
    for (int f = 0; f < (int)TOTAL_FRAGMENTS; f++)
        fragAttempts[f] = 1;
    _stats.recordTaskDone(layer, phase, (int)srcId, fragAttempts);

    if (LOG_LEVEL >= 2)
        std::cout << "[Node " << (int)_nodeId << "] TX DONE -> Node "
                  << (int)srcId << " L" << layer << " " << phaseName(phase)
                  << " (" << tx.doneCount << "/" << _peerIds.size() << ")\n";

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
    if (LOG_LEVEL >= 1)
        std::cout << "[Node " << (int)_nodeId << "] Phase COMPLETE L"
                  << layer << " " << phaseName(phase) << "\n";
    advancePhase();
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
            if (LOG_LEVEL >= 1)
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
