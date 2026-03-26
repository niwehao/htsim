# MOE All-to-All 同步预调度仿真

## 问题背景

在异步版本 (`htsim_v2`) 中，每个 GPU 在阶段开始时将所有数据包一次性 burst 注入网络：

- 8 个 GPU × 7 个目标 × 128 个 fragment = **7168 个包同时注入**
- Tofino 交换机的 BufferGate 检测到缓冲非空时，对每个包施加 **2ms 延迟惩罚**
- 尾部包的端到端延迟可能超过重传超时阈值 → 触发不必要的重传
- 重传注入更多包 → 拥塞加剧 → **拥塞雪崩**

## 核心思想

**在仿真开始前，离线计算每个数据包的最优发送时刻**，使得：

1. 每个包到达每个交换机队列时，队列恰好空闲
2. 无 BufferGate 2ms 延迟惩罚
3. 无丢包、无重传
4. 达到理论最优完成时间

## 算法原理

### 网络资源模型

将网络中的每个队列抽象为一个**资源**，每个资源同一时刻只能服务一个包：

```
资源类型              数量    带宽
─────────────────────────────────
GPU txQueue           8个    25 Gbps   (每 GPU 一个发送队列)
Tofino ingressQ       24个   25 Gbps   (每端口一个入口队列)
Tofino fabricQ        6个    100 Gbps  (每交换机一个内部队列)
Tofino egressQ        24个   25 Gbps   (每端口一个出口队列)
GPU rxQueue           8个    25 Gbps   (每 GPU 一个接收队列)
```

### 数据包路由

每个数据包从源 GPU 到目标 GPU 经过一系列资源：

**同 Leaf (如 GPU1→GPU2，同在 Tofino0 下):**
```
txQ[1] → ingressQ[port2] → fabricQ[tof0] → egressQ[port4] → rxQ[2]
共 5 个资源
```

**跨 Leaf (如 GPU1→GPU3，经过 Spine3):**
```
txQ[1] → ingressQ[p2] → fabricQ[tof0] → egressQ[p1]
       → ingressQ[p17] → fabricQ[tof4] → egressQ[p18]
       → ingressQ[p5] → fabricQ[tof1] → egressQ[p6] → rxQ[3]
共 11 个资源
```

### 贪心调度算法

```
输入: 所有 (源GPU, 目标GPU, fragmentID) 三元组
输出: 每个包的精确发送时刻

对每个资源 R, 维护 nextFree[R] = 该资源的下一个空闲时刻 (初始为 0)

按 fragment → src → dst 的顺序遍历每个包:
    route = 该包经过的资源序列 [(R1, drain1), (R2, drain2), ...]
    prevFinish = 0

    对 route 中的每一跳 (Ri, draini):
        start  = max(nextFree[Ri], prevFinish)   // 资源空闲 且 上一跳完成
        finish = start + draini                    // 占用时间 = 包大小 × 每字节皮秒数
        nextFree[Ri] = finish                      // 更新资源时间线
        prevFinish = finish

    sendTime = 第一跳的 start 时间  // GPU 应在此时刻注入包到 txQueue
```

其中 `drain_time = packet_size × ps_per_byte`：
- 25 Gbps: `ps_per_byte = 320`，65535 字节包 → 20.97 μs
- 100 Gbps: `ps_per_byte = 80`，65535 字节包 → 5.24 μs

### 调度顺序

采用 **fragment → src → dst** 的三层循环：

```
for frag in 0..127:
    for src in GPU1..GPU8:
        for dst in GPU1..GPU8 (≠src):
            schedule(src, dst, frag)
```

这种顺序实现了自然的轮询：每轮为所有 (src,dst) 对各调度一个 fragment，避免单个 GPU 连续占用共享资源。

## 瓶颈分析

| 资源 | 负载 | 占用时间 |
|------|------|---------|
| GPU txQueue (25G) | 896 包 × 20.97μs | **18.79 ms** |
| GPU rxQueue (25G) | 896 包 × 20.97μs | **18.79 ms** |
| Leaf fabricQ (100G) | 1792 包 × 5.24μs | **9.39 ms** |
| Spine3 fabricQ (100G) | 3072 包 × 5.24μs | **16.1 ms** |

**理论下界 = txQueue 瓶颈 = 18.79 ms/阶段**

32 层 × 2 阶段 × (18.79ms + 1ms 间隙) ≈ **1267 ms** 总推理时间 (理论最优)

## 与异步版本的对比

| | 异步版 (htsim_v2) | 同步版 (htsim_syn) |
|---|---|---|
| 发送策略 | Burst (一次性全发) | 按调度表定时发送 |
| BufferGate | 非空缓冲 +2ms 延迟 | 移除 (不需要) |
| Tofino Route 跳数 | 5 跳/交换机 | 3 跳/交换机 |
| 重传机制 | 100ms 超时重传 | 无 (调度保证送达) |
| ACK | 用于确认 + 触发重传 | 仅用于验证阶段完成 |
| 拥塞风险 | 高 (burst 导致) | 无 |

## 文件结构

```
htsim_syn/
├── sync_scheduler.h        # 离线贪心调度算法
├── tofino_switch_sync.h    # 简化 Tofino (无 BufferGate, 3 跳)
├── gpu_node_sync.h         # GPU 节点头文件
├── gpu_node_sync.cpp       # GPU 节点实现 (按调度发送)
├── main.cpp                # 入口: 计算调度 → 建拓扑 → 运行仿真
└── Makefile
```

## 编译与运行

```bash
# 前置: 编译 csg-htsim
git clone https://github.com/Broadcom/csg-htsim.git
cd csg-htsim/sim && make

# 编译同步版
cd htsim_syn
make 

# 运行
make run
```

## 局限性

1. **需要全局信息**: 调度器需要预知所有 (src, dst) 对和包数量，适用于 All-to-All 等确定性通信模式
2. **未调度 ACK**: ACK 包 (29 字节) 未纳入调度，但因体积极小 (占 DATA 的 0.04%) 影响可忽略
3. **贪心非全局最优**: 当前 fragment→src→dst 遍历顺序可能非全局最优，但实际接近下界
4. **静态调度**: 不适应动态负载变化，每个阶段复用相同调度表
