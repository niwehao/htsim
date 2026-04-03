# htsim_syn_512 — 512-GPU 同步调度 MOE 仿真 (零拥塞)

## 概述

基于 `htsim_syn` (8 GPU) 扩展到 **512 GPU** 的同步预调度 MOE 仿真器。
通过离线贪心调度算法预计算每个包的最优发送时刻，消除网络拥塞和重传。

## 核心原理

### 问题: 为什么需要同步调度?

在异步版本 (`htsim_v2_512`) 中，512 个 GPU 在每个阶段同时 burst 发送所有数据包：
- 每 GPU 向 511 个目标各发 2048 个 fragment = **1,046,528 包/GPU**
- 全网瞬间注入 **5.36 亿包**
- 交换机缓冲溢出 → 丢包 → 2ms 拥塞惩罚 → 重传级联

### 解决: 离线贪心调度

将每个网络队列视为"资源"，每个资源同一时刻只能服务一个包：

```
调度顺序: fragment → src → dst (轮询)

对每个包:
  1. 正向扫描: 沿路由链计算每跳最早可用时间
  2. 反向推导: 找瓶颈跳, 反推最优 sendTime
  3. 正向确认: 用 sendTime 更新所有资源时间线

结果: 每个包到达每个交换机时队列恰好为空
```

## 拓扑 (与 htsim_v2_512 相同)

```
        16 Spine Switches (各 32 端口)
   S0   S1   S2   ...   S15
   |\\   |\\   |\\        |\\
   (每 Spine 连接所有 32 Leaf)
   |//   |//   |//        |//
   L0   L1   L2   ...   L31      ← 32 Leaf Switches
   |     |     |          |
  16G  16G  16G  ...    16G      ← 512 GPUs
```

## 资源模型

| 资源类型 | 数量 | 速率 | 说明 |
|----------|------|------|------|
| GPU txQueue | 512 | 200 Gbps | 每 GPU 发送队列 |
| Leaf ingressQ | 32×32=1024 | 200 Gbps | 每端口入队列 |
| Leaf fabricQ | 32 | 6.4 Tbps | 交叉开关 |
| Leaf egressQ | 32×32=1024 | 200 Gbps | 每端口出队列 |
| Spine ingressQ | 16×32=512 | 200 Gbps | 每端口入队列 |
| Spine fabricQ | 16 | 6.4 Tbps | 交叉开关 |
| Spine egressQ | 16×32=512 | 200 Gbps | 每端口出队列 |
| GPU rxQueue | 512 | 200 Gbps | 每 GPU 接收队列 |
| **合计** | **4,176** | | |

## 路由 (抽象资源序列)

```
同 Leaf (GPU 在同一 Leaf 下):
  txQ → Leaf[ingress→fabric→egress] → rxQ        (5 资源)

跨 Leaf (GPU 在不同 Leaf):
  txQ → Leaf[ingress→fabric→egress]               (4 资源)
      → Spine[ingress→fabric→egress]              (+3 资源)
      → Leaf[ingress→fabric→egress] → rxQ         (+4 资源)
                                                   (11 资源)
```

## 与 htsim_v2_512 (异步版) 的对比

| 方面 | htsim_v2_512 (异步) | htsim_syn_512 (同步) |
|------|---------------------|----------------------|
| 发送策略 | Burst (一次性注入) | Schedule (按时间表) |
| 调度 | 无 (被动响应) | 离线贪心预计算 |
| 拥塞 | BufferGate 2ms 惩罚 | 零拥塞 (调度保证) |
| 重传 | ACK 超时 → 重传 | 无重传 (保证送达) |
| 完成时间 | 不确定 (依赖拥塞) | 确定性 (理论最优) |
| 调度计算 | 无 | 需要预计算 (大规模耗时) |

## 与 htsim_syn (8 GPU) 的对比

| 方面 | htsim_syn (8 GPU) | htsim_syn_512 |
|------|-------------------|---------------|
| GPU | 8 | 512 |
| 拓扑 | 3 层 6 Tofino | 2 层 48 Switch |
| 路由 | 硬编码 | 自动计算 |
| 调度条目 | 7,168 | ~536M |
| GPU ID | uint8_t, 1-indexed | uint16_t, 0-indexed |
| 内存管理 | 无清理 | cleanupPhase() |

## 文件结构

```
htsim_syn_512/
├── constants.h           # 拓扑参数 + 路由函数 + MoePacket
├── tofino_switch_sync.h  # 交换机 (含 BufferGate, 调度下为 pass-through)
├── sync_scheduler.h      # 离线贪心调度器 (Leaf-Spine 资源模型)
├── gpu_node_sync.h       # GPU 节点接口 (无重传)
├── gpu_node_sync.cpp     # GPU 应用层 (定时发送 + ACK 验证)
├── main.cpp              # 调度 + 拓扑 + 仿真入口
├── Makefile
└── README.md
```

## 编译与运行

```bash
# 确保 csg-htsim 已编译
make -C ../htsim_v2/csg-htsim/sim

# 编译
make

# 运行
make run
```

## 性能注意事项

512 GPU 的调度计算规模很大:
- 调度条目: `2048 frags × 512 src × 511 dst ≈ 5.36 亿`
- 每条目 3 遍扫描 × 5~11 跳 → 数十亿次资源查找
- 预计调度计算时间: **数分钟到数十分钟**
- 调度只需计算一次, 每个阶段复用同一份调度表

如需加速测试, 可临时减小参数:
```cpp
// constants.h
static constexpr uint32_t NUM_GPU_NODES = 64;    // 减少 GPU
static constexpr uint32_t TOTAL_LAYERS  = 2;     // 减少层数
```

## MOE 参数

| 参数 | 值 |
|------|-----|
| 推理层数 | 32 |
| 每层阶段 | 2 (DISPATCH + COMBINE) |
| Fragment 大小 | 4 KB |
| 每目标 Fragment 数 | 2048 |
| 每目标数据量 | 8 MB |
| 阶段间间隙 | 1 ms |
