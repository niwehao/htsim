# MOE All-to-All 网络仿真 — Dragonfly 拓扑版

基于 [Broadcom/csg-htsim](https://github.com/Broadcom/csg-htsim) 事件驱动模拟器，参照 htsim_v2 实现，将 3 级 Clos 拓扑替换为 Dragonfly 拓扑。

## 仿真内容

8 个 GPU 节点通过 Dragonfly 网络（4 个交换机，2 个 Group）执行 32 层 MOE 推理，每层包含 DISPATCH 和 COMBINE 两个 All-to-All 阶段。每个 GPU 向其余 7 个 GPU 发送 8 MB 数据（2048 个 4 KB 分片），并通过 ACK 确认 + 超时重传保证可靠交付。

## Dragonfly 拓扑

### 拓扑结构

```
  Group 0                           Group 1
┌──────────────────┐             ┌──────────────────┐
│                  │             │                  │
│  GPU1 ─┐        │   global    │        ┌─ GPU5   │
│  GPU2 ─┤─ Sw0 4──┼─────────────┼──12 Sw2 ─┤─ GPU6   │
│        │  │ 3    │             │  11  │   │         │
│        │ local  │             │  local │         │
│        │  │ 7   │             │   15 │   │         │
│  GPU3 ─┤─ Sw1 8 ──┼─────────────┼──16 Sw3 ─┤─ GPU7   │
│  GPU4 ─┘        │   global    │        └─ GPU8   │
│                  │             │                  │
└──────────────────┘             └──────────────────┘
```

### 交换机端口分配

| 交换机 | Group | 端口 1-2 (GPU) | 端口 3 (Local) | 端口 4 (Global) |
|--------|-------|----------------|----------------|-----------------|
| Sw0    | 0     | GPU1, GPU2  1,2    | → Sw1 (p7)     | → Sw2 (p12)     |
| Sw1    | 0     | GPU3, GPU4   5,6   | → Sw0 (p3)     | → Sw3 (p16)     |
| Sw2    | 1     | GPU5, GPU6  9,10   | → Sw3 (p15)    | → Sw0 (p4)      |
| Sw3    | 1     | GPU7, GPU8  13,14   | → Sw2 (p11)    | → Sw1 (p8)      |

### 链路类型

| 链路类型 | 端口对 | 说明 |
|----------|--------|------|
| Local    | p3↔p7, p11↔p15 | Group 内交换机互联 |
| Global   | p4↔p12, p8↔p16 | 跨 Group 互联 |
| Access   | p1,p2,p5,p6,p9,p10,p13,p14 | GPU 接入 |

## Dragonfly 路由

采用 **最小路由 (Minimal Routing)**:

| 路径类型 | 交换机跳数 | Route 节点数 | 示例 |
|----------|-----------|-------------|------|
| 同交换机 | 1 | 7 | GPU1→GPU2 |
| 同 Group | 2 | 12 | GPU1→GPU3 |
| 跨 Group (直达) | 2 | 12 | GPU1→GPU5 |
| 跨 Group (中转) | 3 | 17 | GPU1→GPU7 |

### 路由策略

- **同交换机**: 直接从入端口到出端口
- **同 Group**: 经 local link 到达同 Group 的另一交换机
- **跨 Group (直达)**: 经 global link 到达目标 Group 中有直连的交换机
- **跨 Group (中转)**: 经 global link 到达目标 Group，再经 local link 到目标交换机

### 路由表

```
Sw0: {GPU1→p1, GPU2→p2, GPU3→p3, GPU4→p3, GPU5→p4, GPU6→p4, GPU7→p4, GPU8→p4}
Sw1: {GPU1→p7, GPU2→p7, GPU3→p5, GPU4→p6, GPU5→p8, GPU6→p8, GPU7→p8, GPU8→p8}
Sw2: {GPU1→p12, GPU2→p12, GPU3→p12, GPU4→p12, GPU5→p9, GPU6→p10, GPU7→p11, GPU8→p11}
Sw3: {GPU1→p16, GPU2→p16, GPU3→p16, GPU4→p16, GPU5→p15, GPU6→p15, GPU7→p13, GPU8→p14}
```

## 与 Clos (htsim_v2) 对比

| 指标 | 3 级 Clos | Dragonfly |
|------|-----------|-----------|
| 交换机数量 | 6 | 4 |
| 最大跳数 | 3 (17 节点) | 3 (17 节点) |
| 最小跳数 | 1 (7 节点) | 1 (7 节点) |
| 同 Group 跳数 | 1 (同 Leaf) | 1-2 |
| 跨 Group 跳数 | 3 | 2-3 |
| Global link 负载 | 分散 (2 条 Spine) | 集中 (2 条 Global) |
| 拓扑特点 | 均匀多路径 | 分层，local 带宽高 |

### 负载分析

- **Global link** (Sw0↔Sw2, Sw1↔Sw3): 各承载 16 条流（8 条去程 + 8 条回程的跨 Group 流量）
- **Local link** (Sw0↔Sw1, Sw2↔Sw3): 各承载 ~16 条流（同 Group 流量 + 部分中转流量）
- **瓶颈**: Global link 集中承载所有跨 Group 流量，是性能瓶颈

## 延迟模型

每个交换机精确建模五级流水线（与 htsim_v2 相同）:

| 阶段 | 组件 | 速率 |
|------|------|------|
| 入端口处理 | `ingressQueue` | 200 Gbps |
| 缓冲检查 | `BufferGate` | 32 MB 上限 |
| 内部转发 | `fabricQueue` | 800 Gbps |
| 缓冲释放 | `BufferRelease` | — |
| 出端口处理 | `egressQueue` | 200 Gbps |

## 文件说明

| 文件 | 说明 |
|------|------|
| `constants.h` | Dragonfly 拓扑常量、路由表、端口映射 |
| `tofino_switch.h` | 交换机模型 (同 htsim_v2) |
| `gpu_node.h/cpp` | GPU 节点逻辑 (同 htsim_v2) |
| `main.cpp` | Dragonfly 拓扑创建、路由预计算、仿真主循环 |

## 编译与运行

```bash
# 1. 确保 csg-htsim 已编译 (在 htsim_v2 目录中)
cd ../htsim_v2
make lib

# 2. 编译 Dragonfly 版
cd ../dragonfly
make

# 3. 运行
./moe_dragonfly.bin
```

## 关键参数

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| GPU 数量 | 8 | `NUM_GPU_NODES` |
| 交换机数量 | 4 | `NUM_SWITCHES` |
| Group 数量 | 2 | `NUM_GROUPS` |
| 推理层数 | 32 | `TOTAL_LAYERS` |
| 每目标数据量 | 8 MB | `PAYLOAD_BYTES_PER_TARGET` |
| 分片大小 | 4 KB | `FRAGMENT_PAYLOAD_SIZE` |
| 分片数 | 2048 | `TOTAL_FRAGMENTS` |
| 链路速率 | 200 Gbps | `PROT_RATE_Gbps` |
| 交换机缓冲 | 32 MB | `SW_QUEUE_SIZE_BYTES` |
| 重传超时 | 20 ms | `TIMEOUT_PS` |
| 层间间隔 | 1 ms | `INTERPHASE_GAP_PS` |
