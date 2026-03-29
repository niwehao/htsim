# MOE All-to-All 网络仿真 — Dragonfly 拓扑 + 同步预调度

基于 [Broadcom/csg-htsim](https://github.com/Broadcom/csg-htsim) 事件驱动模拟器，将 Dragonfly 拓扑与同步预调度 (Sync Scheduling) 结合，**消除 BufferGate 2ms 惩罚**，实现零拥塞、零重传的理论最优通信。

## 问题: BufferGate 延迟惩罚

在异步版 Dragonfly (`dragonfly/`) 中，8 个 GPU 同时向所有目标 burst 发送数据包。当多条路径共享同一交换机时，共享缓冲区 (32 MB) 快速填满，触发 **BufferGate**:

```
包到达交换机 → BufferGate 检查共享缓冲占用
  ├─ 缓冲 < 32 MB → 正常转发 (0 额外延迟)
  └─ 缓冲 >= 32 MB → 强制延迟 2 ms 再释放
                       ↑ 这就是 BufferGate 惩罚
```

### 惩罚放大效应

```
8 GPU × 7 目标 × 2048 片/目标 = 114,688 个数据包
交叉路径共享 Dragonfly 的 local/global 链路
  → 大量包在交换机排队 → 触发 BufferGate
  → 2ms 延迟 × 多次触发 → 总延迟显著增加
  → 超时 → 重传 → 进一步拥塞 (恶性循环)
```

## 解决方案: 三遍扫描同步调度

### 核心思想

**在仿真开始前，离线计算每个包的精确发送时刻**，保证：
1. 每个包到达每个交换机队列时，该队列为空
2. 缓冲占用始终为零 → BufferGate **永远不被触发**
3. 无排队延迟、无重传

### 三遍扫描算法 (优于 htsim_syn 的一遍贪心)

```
输入: 所有 (src, dst, frag) 组合, 按 fragment → src → dst 轮询顺序

对每个包:

  第一遍 — 正向扫描 (只读, 不更新 resFree):
    对路由链上的每一跳:
      start[i] = max(resFree[resource_i], prevFinish)
    → 得到各跳的最早开始时间 hopStart[]

  第二遍 — 反向推导 sendTime (Just-In-Time):
    sendTime = hopStart[0]
    对 i = 1..N:
      sumDrain = sum(drainTime[0..i-1])
      candidateSend = hopStart[i] - sumDrain
      sendTime = max(sendTime, candidateSend)
    → 找到瓶颈跳, 反推使包"刚好"到达瓶颈时资源空闲的 sendTime

  第三遍 — 正向提交 (更新 resFree):
    prevFinish = sendTime
    对每一跳:
      start = max(resFree[resource], prevFinish)
      resFree[resource] = start + drainTime
```

### 为什么三遍比一遍好？

| | 一遍贪心 (htsim_syn) | 三遍扫描 (dragonfly_syn) |
|---|---|---|
| 发送时刻 | 资源最早可用时 | 瓶颈决定的最晚可用时 (JIT) |
| 中间队列占用 | 包可能在中间队列等待 | 包到达时队列刚好空闲 |
| BufferGate 安全 | 理论安全, 但中间有排队 | **严格保证零排队** |
| 适用场景 | 简单拓扑 | 多跳 Dragonfly (路径共享严重) |

## Dragonfly 拓扑

```
  Group 0                              Group 1
┌────────────┐                    ┌────────────┐
│ Sw0        │   global (p4↔p12)  │ Sw2        │
│ GPU1 (p1)  │◄──────────────────►│ GPU5 (p9)  │
│ GPU2 (p2)  │                    │ GPU6 (p10) │
│ local (p3) │                    │ local (p11)│
└─────┬──────┘                    └─────┬──────┘
      │ p3↔p7                           │ p11↔p15
┌─────┴──────┐                    ┌─────┴──────┐
│ Sw1        │   global (p8↔p16)  │ Sw3        │
│ GPU3 (p5)  │◄──────────────────►│ GPU7 (p13) │
│ GPU4 (p6)  │                    │ GPU8 (p14) │
│ local (p7) │                    │ local (p15)│
└────────────┘                    └────────────┘
```

### 路由类型与跳数

| 类型 | 示例 | 交换机跳数 | Route 节点数 | 资源数 |
|------|------|-----------|-------------|--------|
| 同交换机 | GPU1→GPU2 | 1 | 5 (3+rxQ+app) | 5 |
| 同 Group | GPU1→GPU3 | 2 | 8 (3+3+rxQ+app) | 8 |
| 跨 Group (直达) | GPU1→GPU5 | 2 | 8 | 8 |
| 跨 Group (转发) | GPU1→GPU7 | 3 | 11 (3+3+3+rxQ+app) | 11 |

### 资源模型

```
每条路径经过的资源 (同步调度视角):

GPU_src.txQ  →  Sw_a.ingressQ[port_in]  →  Sw_a.fabricQ  →  Sw_a.egressQ[port_out]
             →  Sw_b.ingressQ[port_in]  →  Sw_b.fabricQ  →  Sw_b.egressQ[port_out]
             →  ...
             →  GPU_dst.rxQ

每个资源同一时刻只能服务一个包 (串行化约束)
```

| 资源 | 数量 | 速率 | 资源 ID |
|------|------|------|---------|
| GPU txQueue | 8 | 200 Gbps | 1-8 |
| GPU rxQueue | 8 | 200 Gbps | 11-18 |
| Tofino ingressQ | 16 (4sw × 4port) | 200 Gbps | 101-116 |
| Tofino fabricQ | 4 | 800 Gbps | 200-203 |
| Tofino egressQ | 16 | 200 Gbps | 301-316 |

## 交换机模型对比

### 异步版 (dragonfly/tofino_switch.h): 5 级流水线

```
ingressQ → BufferGate → fabricQ → BufferRelease → egressQ
           ↑ 检查 32MB        ↑ 释放缓冲计数
           共享缓冲
```

### 同步版 (tofino_switch_sync.h): 3 级流水线

```
ingressQ → fabricQ → egressQ
```

同步调度保证包到达时队列为空 → **不需要 BufferGate/BufferRelease**

## 性能分析

### 理论下界

```
每包串行化时间 (200 Gbps):
  DATA: (14+15+4096) B × 8 / 200e9 = 165 ns = 165,000 ps
  ACK:  (14+15) B × 8 / 200e9 = 1.16 ns

每 GPU 发送总量: 7 目标 × 2048 片 = 14,336 包
txQueue 占用: 14,336 × 165,000 ps ≈ 2.37 ms

跨 Group 路径共享 global 链路:
  Sw0→Sw2 (port 4→12): 最多承载 4 GPU 的跨 Group 流量
  瓶颈 = global egress 端口: 需要序列化所有经过的包

理论下界 (受瓶颈资源限制): 调度器输出的 phaseDuration
```

### 与异步版对比

| 指标 | Dragonfly 异步 | Dragonfly 同步 |
|------|---------------|---------------|
| 交换机模型 | 5 级 (含 BufferGate) | 3 级 (无 BufferGate) |
| BufferGate 触发 | 频繁 (2ms/次) | **永远不触发** |
| 重传 | 有 (超时后) | **无** |
| 发送策略 | Burst (全部同时) | 定时 (按 schedule) |
| 拥塞 | 严重 | **零** |
| Route 节点/Tofino | 5 | 3 |
| 完成时间 | 受 BufferGate + 重传影响 | **理论最优** |

### 与 Clos 同步版对比

| 指标 | Clos Sync (htsim_syn) | Dragonfly Sync |
|------|----------------------|----------------|
| 交换机数量 | 6 | 4 |
| 调度算法 | 一遍贪心 | **三遍扫描 (JIT)** |
| 最大路径长度 | 3 Tofino | 3 Tofino |
| 路径多样性 | 高 (ECMP) | 低 (minimal routing) |
| 瓶颈 | 分散 | 集中 (global link) |

## 文件说明

| 文件 | 说明 |
|------|------|
| `constants.h` | Dragonfly 网络常量, 路由表, 端口映射, MoePacket, TimerEvent, NodeStats |
| `sync_scheduler.h` | 三遍扫描同步调度算法, 资源模型, 调度验证 |
| `tofino_switch_sync.h` | 简化 Tofino (无 BufferGate, 3 级流水线) |
| `gpu_node_sync.h/cpp` | 同步版 GPU 节点 (定时发送, 无重传, ACK 验证) |
| `main.cpp` | 拓扑创建, 调度计算, 路由构建, 仿真主循环 |

## 编译与运行

```bash
# 1. 确保 csg-htsim 已编译
cd ../htsim_v2/csg-htsim/sim
make

# 2. 编译 Dragonfly 同步版
cd ../../../dragonfly_syn
make

# 3. 运行
./moe_dragonfly_syn.bin
```

## 关键参数

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| GPU 数量 | 8 | `NUM_GPU_NODES` |
| 交换机数量 | 4 | `NUM_SWITCHES` |
| Group 数 | 2 | `NUM_GROUPS` |
| 推理层数 | 32 | `TOTAL_LAYERS` |
| 每目标数据量 | 8 MB | `PAYLOAD_BYTES_PER_TARGET` |
| 分片大小 | 4 KB | `FRAGMENT_PAYLOAD_SIZE` |
| 分片数 | 2048 | `TOTAL_FRAGMENTS` |
| 链路速率 | 200 Gbps | `PROT_RATE_Gbps` |
| Fabric 速率 | 800 Gbps | `FABRIC_SPEED_BPS` |
| BufferGate 延迟 | 2 ms | `BUFFER_DELAY_PS` (同步版不触发) |
| 共享缓冲上限 | 32 MB | `SW_QUEUE_SIZE_BYTES` (同步版无需) |
| 层间间隔 | 1 ms | `INTERPHASE_GAP_PS` |
| 日志级别 | 2 | `LOG_LEVEL` |

## 调度验证

`SyncScheduler::validateSchedule()` 在仿真前验证：
- 对每个网络资源，检查所有包的占用区间 `[start, finish)` 不重叠
- 如果通过验证，保证仿真中零拥塞

## 扩展方向

1. **更大规模 Dragonfly**: 自动生成路由表替代硬编码，支持 N 组 × M 交换机
2. **自适应路由**: 非最短路径 (Valiant routing) 避免 global link 热点
3. **混合调度**: 热点流用同步调度，冷流用异步 burst
4. **ACK 调度**: 将 ACK 也纳入同步调度，进一步减少干扰
