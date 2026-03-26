# MOE All-to-All 网络仿真 — csg-htsim 版

基于 [Broadcom/csg-htsim](https://github.com/Broadcom/csg-htsim) 事件驱动模拟器，从 cocotb 仿真代码翻译而来。

## 仿真内容

8 个 GPU 节点通过 3 级 Clos 网络（6 个 Tofino 交换机）执行 32 层 MOE 推理，每层包含 DISPATCH 和 COMBINE 两个 All-to-All 阶段。每个 GPU 向其余 7 个 GPU 发送 8 MB 数据（128 个 64 KB 分片），并通过 ACK 确认 + 100ms 超时重传保证可靠交付。

## 网络拓扑

```
GPU1 ─┐                              ┌─ GPU5
GPU2 ─┤─ S1L1 ─┐          ┌─ S2L1 ─┤─ GPU6
      │         ├─ S3L1 ─┤         │
GPU3 ─┤─ S1L2 ─┤          ├─ S2L2 ─┤─ GPU7
GPU4 ─┘         └─ S3L2 ─┘         └─ GPU8
```

- **S1L1~S2L2**: Spine1/2 的 Leaf 交换机，各连 2 个 GPU
- **S3L1, S3L2**: Spine3 互联交换机，连接 Spine1 和 Spine2
- 所有链路 25 Gbps，内部 Fabric 100 Gbps (25G × 4 端口)

## 延迟模型

每个 Tofino 交换机精确建模三级流水线：

| 阶段 | 组件 | 速率 | 对应原始代码 |
|------|------|------|-------------|
| 入端口处理 | `ingressQueue` | 25 Gbps | `Tofino._port_worker_in` |
| 内部转发 | `fabricQueue` | 100 Gbps | `Tofino._transfer_worker` |
| 出端口处理 | `egressQueue` | 25 Gbps | `Tofino._port_worker_out` |

额外建模：
- **缓冲管理** (`BufferGate`): 32 MB 上限，溢出丢包，拥塞时 2ms 延迟入队
- **GPU 端口** (`txQueue` / `rxQueue`): 各 25 Gbps 串行化

## 文件说明

| 文件 | 说明 | 翻译自 |
|------|------|--------|
| `constants.h` | 网络常量、路由表、MoePacket、TimerEvent、NodeStats | `Constants.py` + `Statics.py` |
| `tofino_switch.h` | TofinoSwitch (BufferGate + BufferRelease + 队列) | `Tofino.py` |
| `gpu_node.h/cpp` | GpuNode 应用层逻辑 (发送/接收/重传/统计) | `GPU.py` |
| `main.cpp` | 拓扑创建、路由预计算、仿真主循环 | `test_mqnic_sync_dcn.py` + `Spine.py` |

## 编译与运行

```bash
# 1. 克隆 csg-htsim
git clone https://github.com/Broadcom/csg-htsim.git
cd csg-htsim/sim && make
cd -

# 2. 编译本项目
cd htsim_v2
make HTSIM_DIR=/path/to/csg-htsim/sim

# 3. 运行
./moe_sim
```

## 关键参数

| 参数 | 值 | 定义位置 |
|------|-----|---------|
| GPU 数量 | 8 | `NUM_GPU_NODES` |
| 推理层数 | 32 | `TOTAL_LAYERS` |
| 每目标数据量 | 8 MB | `PAYLOAD_BYTES_PER_TARGET` |
| 分片大小 | 64 KB | `FRAGMENT_PAYLOAD_SIZE` |
| 分片数 | 128 | `TOTAL_FRAGMENTS` |
| 链路速率 | 25 Gbps | `PROT_RATE_Gbps` |
| 交换机缓冲 | 32 MB | `SW_QUEUE_SIZE_BYTES` |
| 重传超时 | 100 ms | `TIMEOUT_PS` |
| 拥塞延迟 | 2 ms | `BUFFER_DELAY_PS` |
| 层间间隔 | 1 ms | `INTERPHASE_GAP_PS` |

所有参数在 `constants.h` 中定义，修改后重新编译即可。

## csg-htsim Source Routing

与原始 cocotb 仿真的动态路由不同，csg-htsim 使用 **source routing**：每个包在创建时绑定预计算的完整 `Route`（`PacketSink*` 序列）。`Queue::completeService()` 调用 `pkt->sendOn()` 自动沿 Route 推进到下一跳。

同 Leaf 路径 (如 GPU1→GPU2)：
```
txQueue → [ingressQ → BufferGate → fabricQ → BufferRelease → egressQ] → rxQueue → AppSink
            └──────────── 1 个 Tofino (7 hops) ────────────┘
```

跨 Spine 路径 (如 GPU1→GPU5)：
```
txQueue → [Tofino×3, 每个 5 hops] → rxQueue → AppSink  (17 hops)
```

## 已知限制

- `csg-htsim` 的 `Packet::_size` 为 `uint16_t`（最大 65535），数据包实际 65565 字节会截断。如需精确值，将 `network.h` 中 `_size` 改为 `uint32_t`。
- 原始代码无传播延迟（节点间无物理距离建模），本翻译保持一致。如需添加，在 `appendToRoute` 中插入 `Pipe` 节点即可。
