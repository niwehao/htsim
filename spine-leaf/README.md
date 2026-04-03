# htsim_v2_512 — 512-GPU 异步 MOE 仿真 (Burst 模式)

## 概述

基于 `htsim_v2` (8 GPU) 扩展到 **512 GPU** 的异步 all-to-all MOE 通信仿真器。
采用 **2 层 Leaf-Spine 拓扑**，路由全自动计算，只需修改 `NUM_GPU_NODES` 即可缩放。

## 拓扑

```
        16 Spine Switches (各 32 端口)
   S0   S1   S2   ...   S15
   |\\   |\\   |\\        |\\
   (每个 Spine 各 1 端口连接每个 Leaf, 共 32 端口)
   |//   |//   |//        |//
   L0   L1   L2   ...   L31      ← 32 Leaf Switches (各 32 端口)
   |     |     |          |
  16G  16G  16G  ...    16G      ← 512 GPUs (每 Leaf 16 个)
```

| 参数 | 值 |
|------|-----|
| GPU 数量 | 512 (可配置) |
| Leaf 交换机 | 32 个, 16 下行 + 16 上行 |
| Spine 交换机 | 16 个, 32 端口 |
| 总交换机 | 48 |
| 每端口速率 | 200 Gbps |
| 过载比 | 1:1 (无阻塞) |
| 最大跳数 | 同 Leaf: 2 跳, 跨 Leaf: 4 跳 |

## 自动路由算法

```
GPU g 的位置:
  leaf  = g / 16      (Leaf 索引 0..31)
  port  = g % 16      (Leaf 下行端口 0..15)

路由 src → dst:
  同 Leaf:  GPU → Leaf → GPU
  跨 Leaf:  GPU → Leaf[src] → Spine[s] → Leaf[dst] → GPU
  ECMP:     s = (src + dst) % 16
```

无需硬编码任何路由表。修改 `constants.h` 中 `NUM_GPU_NODES` 自动调整：
- 256 GPUs → 16 Leaf × 16 Spine = 32 switches
- 512 GPUs → 32 Leaf × 16 Spine = 48 switches
- 约束: `NUM_GPU_NODES / 16 ≤ 32`

## 与 htsim_v2 (8 GPU) 的区别

| 方面 | htsim_v2 | htsim_v2_512 |
|------|----------|--------------|
| GPU 数量 | 8 | 512 |
| 拓扑 | 3 层 6 Tofino | 2 层 48 Switch |
| 路由 | 硬编码 56 条 | 自动计算 261,632 条 |
| GPU ID | uint8_t, 1-indexed | uint16_t, 0-indexed |
| 内存管理 | 无清理 | cleanupPhase() 阶段后释放 |
| 日志 | 全量 | 里程碑 + 重传 |

## 文件结构

```
htsim_v2_512/
├── constants.h           # 拓扑参数自动推导 + 路由函数 + MoePacket
├── tofino_switch.h       # 通用交换机 (动态端口, BufferGate 拥塞模型)
├── gpu_node.h            # GPU 节点接口
├── gpu_node.cpp          # GPU 应用层 (burst 发送 + ACK 重传)
├── main.cpp              # 拓扑构建 + 路由计算 + 仿真入口
├── Makefile
├── topology_diagram.html # 交互式拓扑可视化
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

## MOE 参数

| 参数 | 值 |
|------|-----|
| 推理层数 | 32 |
| 每层阶段 | 2 (DISPATCH + COMBINE) |
| Fragment 大小 | 4 KB |
| 每目标 Fragment 数 | 2048 |
| 每目标数据量 | 8 MB |
| 重传超时 | 20 ms |
| 阶段间间隙 | 1 ms |
