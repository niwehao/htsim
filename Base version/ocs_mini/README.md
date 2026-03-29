# OCS + EcsBuffer HOHO Multi-hop MOE Simulation

基于 [operasim](../operasim/src/opcc/) 的 HOHO (Hop-by-hop Opportunistic) 算法，在 OCS 单交换机拓扑上实现多跳路由。

## 核心架构

每个 GPU 节点配备一个 **EcsBuffer** (Edge Circuit Switch Buffer)。EcsBuffer 是整个系统的核心组件，替代了原来 56 个 OcsCircuit + 8 个 HohoForwarder 的设计。

**物理约束**: 每个节点只有一个 OCS 端口，每个时隙只连接一个对端。

### 路由决策

BFS 在时间展开图 (8 nodes × 7 slices = 56 states) 上预计算路由表：

```
rt[node][dst][slice] = FORWARD 或 WAIT
  FORWARD: 发给当前时隙连接的对端（不需要指定目的地，因为只有一个对端）
  WAIT:    留在 buffer，标记 target_slice
```

### 与旧设计的对比

| 旧设计 | 新设计 |
|--------|--------|
| 56 个 OcsCircuit (每对 src-dst 一条) | 8 个 EcsBuffer (每节点一个) |
| 8 个 HohoForwarder | EcsBuffer 内置路由逻辑 |
| 2跳路由 (直连/中转) | BFS 最短路径 (最多 ~5 跳) |
| 一次性路由决策 | 每个时隙边界重新评估 |
| 物理上不正确 (多条并行电路) | 物理正确 (单 OCS 端口) |

## 数据流

### EcsBuffer 内部结构

```
GPU.txQ ──→ EcsBuffer ──→ ocs_tx (200Gbps) ──→ OcsLinkDelay ──→ 对端 EcsBuffer
              │                                                      │
              ├── buffer (target_slice 标记)                          ├── 路由决策
              │                                                      │
              └── sendOn ──→ GPU.rxQ ──→ AppSink (本地交付)           └── ...
```

### 四个逻辑端口

| 端口 | 方向 | 说明 |
|------|------|------|
| gpu_rx | GPU → EcsBuffer | GPU 的 txQueue 串行化 (200Gbps) |
| gpu_tx | EcsBuffer → GPU | GPU 的 rxQueue 串行化 (200Gbps) |
| ocs_tx | EcsBuffer → OCS | OCS 上行串行化 (200Gbps) |
| ocs_rx | OCS → EcsBuffer | 从 OcsLinkDelay 直接接收 |

### 延迟模型 (单跳直连)

```
GPU_A.txQ 串行化 → EcsBuffer@A 路由 → ocs_tx 串行化 → OCS 传播(100ns)
  → EcsBuffer@B 路由 → GPU_B.rxQ 串行化 → AppSink
```

### 直连路径 (A→D, 当前时隙 A↔D 连通)

```
GPU_A.txQ → sendOn → EcsBuffer@A [FORWARD]
  → ocs_tx → OcsLinkDelay → EcsBuffer@D [LOCAL]
  → sendOn → rxQ_D → AppSink_D
```

### 2跳中转路径 (A→D, 经过中间节点 B)

```
GPU_A.txQ → sendOn → EcsBuffer@A [FORWARD to B]
  → ocs_tx → OcsLinkDelay → EcsBuffer@B [FORWARD to D]
  → ocs_tx → OcsLinkDelay → EcsBuffer@D [LOCAL]
  → sendOn → rxQ_D → AppSink_D
```

### 等待+转发路径 (A→D, 当前无好的中转)

```
GPU_A.txQ → sendOn → EcsBuffer@A [WAIT, target_slice=3]
  → buffer → (slice 3 到来) → EcsBuffer@A [FORWARD to D]
  → ocs_tx → OcsLinkDelay → EcsBuffer@D [LOCAL]
  → sendOn → rxQ_D → AppSink_D
```

## BFS 路由表计算

### 时间展开图

将 8 个节点 × 7 个时隙展开为 56 个状态 `(node, slice)`。

边:
- **WAIT**: `(node, s)` → `(node, (s+1)%7)`, 代价 1
- **FORWARD**: `(node, s)` → `(peer(node,s), (s+1)%7)`, 代价 1

### 反向 BFS

对每个目的地 `dst`:
1. 初始化: 所有 `(dst, s)` 状态距离为 0
2. 反向 BFS: 从已访问状态回溯 WAIT 和 FORWARD 边
3. 结果: `rt[node][dst][slice]` = 到达 `dst` 的最优动作

### 示例 (8 GPU, 7 时隙)

```
src=1, dst=5, 当前 slice=0:
  直连: slice[1][5]=5, 需要等 5 个时隙
  BFS 路径: FORWARD→GPU8(slice 0) → WAIT(slice 1) → FORWARD→GPU5(slice 2)
  代价: 3 个时隙 (vs 直接等 5+1=6 个时隙)
  rt[1][5][0] = FORWARD
```

## target_slice 标记系统

每个包携带 `target_slice` 标签:
- **-1**: 未标记 (从 GPU 新发出或刚被转发)
- **0-6**: 目标转发时隙

### 工作流程

1. 包到达 EcsBuffer → 查表 `rt[me][dst][current_slice]`
2. 如果 **FORWARD**: 立即转发到 ocs_tx
3. 如果 **WAIT**: 计算下一个 FORWARD 的时隙，标记为 `target_slice`，放入 buffer
4. 每个时隙边界 (SliceAlarm): 遍历 buffer，`target_slice` 匹配的包取出转发
5. 包到达下一个 EcsBuffer: 清除旧标签，重新路由
6. 我理解了,谢谢,现在实现一个普通版本ocs,在ocs_norm文件夹下实现,网络架构没有任何区别,依然是gpu-ecsbuffer-optics,唯一区别就是不需要任何最短路径规划,如果目前没有直接连接目标地址,就进入buffer,每次时间slot变化就去buffer里取出发送,等到可以连接到了目的地址

## 组件说明

| 组件 | 文件 | 功能 |
|------|------|------|
| **DynOcsTopology** | `hoho_routing.h` | 管理 7 时隙 + BFS 路由表 (56 state graph) |
| **EcsBuffer** | `hoho_routing.h` | 每节点一个, 路由决策 + buffer + ocs_tx |
| **SerDesPort** | `hoho_routing.h` | 回调式串行化队列 (替代 Queue) |
| **OcsLinkDelay** | `hoho_routing.h` | 光纤传播延迟 (per-packet 目的地) |
| **SliceAlarm** | `hoho_routing.h` | 时隙边界事件, 通知所有 EcsBuffer |
| **OcsSwitch** | `ocs_switch.h` | 创建 8 个 EcsBuffer + 布线 |
| **GpuNode** | `gpu_node.h/cpp` | MOE GPU 节点, 并行发送所有目标 |
| **MoePacket** | `constants.h` | MOE 数据/ACK 包 (含 target_slice) |

## GPU 发送模型

与之前相同，使用**并行发送**:

1. 阶段开始时，GPU 向所有 7 个目标同时注入数据
2. 所有 fragment 进入 txQueue (FIFO, 200Gbps 串行化)
3. txQueue 输出到 EcsBuffer，由 BFS 路由表决定 FORWARD 或 WAIT
4. EcsBuffer 在正确时隙转发，中间节点继续路由
5. 目标收到数据后发送 ACK (ACK 也经过 EcsBuffer 路由)
6. GPU 等待所有 7 个目标的 ACK 完成后进入下一阶段

## 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| GPU 数量 | 8 | |
| MOE 层数 | 32 | DISPATCH + COMBINE |
| Fragment 数 | 2048 | 每目标 |
| Fragment 大小 | 4 KB | |
| 每目标数据量 | 8 MB | |
| 链路速率 | 200 Gbps | |
| 时隙活动期 | ~355 us | 传输时间 + 5% 余量 |
| 重配置延迟 | 10 us | MEMS OCS |
| 时隙总长 | ~365 us | 活动 + 重配置 |
| 周期 | ~2.55 ms | 7 个时隙 |
| EcsBuffer 缓冲 | 128 MB | 含中转数据 |
| OCS TX 缓冲 | 128 MB | 串行化队列 |
| 光纤延迟 | 100 ns | |
| 超时重传 | 20 ms | |

## 构建和运行

```bash
make clean && make
make run
```

```
GPU.txQueue (Queue, 200Gbps)          ← htsim 原生 Queue
    → sendOn
    → EcsBuffer.receivePacket         ← 无串行化，瞬时
        → processPacket
            → forwardToOcs
                → _ocs_tx (SerDesPort, 200Gbps)   ← 唯一的 SerDesPort
                    → onOcsTxDone
                        → OcsLinkDelay (100ns)
                            → 对端 EcsBuffer.receivePacket  ← 无串行化，瞬时
                                → processPacket
                                    → LOCAL: sendOn → rxQueue (Queue, 200Gbps)
                                    → FORWARD: → _ocs_tx (另一个 SerDesPort)
                                    → WAIT: → buffer

```

