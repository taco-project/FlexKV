# FlexKV 正式合入 Dynamo & vLLM 官方社区

---

## 第一部分：合入官方社区公告

### FlexKV 正式成为 NVIDIA Dynamo 和 vLLM 官方 KV Cache Offloading 方案

我们非常高兴地宣布，**FlexKV** 已正式合入 **NVIDIA Dynamo** 和 **vLLM** 两大主流 LLM 推理框架的官方社区！

- **Dynamo PR**: [ai-dynamo/dynamo#5858](https://github.com/ai-dynamo/dynamo/pull/5858) — *feat: FlexKV integration in Dynamo*（2026年3月3日合入）
- **vLLM PR**: [vllm-project/vllm#34328](https://github.com/vllm-project/vllm/pull/34328) — *[KV Connector] Support using FlexKV as KV Cache Offloading option*（2026年3月12日合入）

FlexKV 是由**腾讯云 TACO 团队**联合 NVIDIA 及社区共同开发的**分布式 KV Store 与多级缓存管理系统**，专为超大规模 LLM 推理场景设计。此次合入意味着 FlexKV 成为 Dynamo 和 vLLM 原生支持的 KV Cache Offloading 方案，用户无需额外打 patch，即可直接通过配置启用 FlexKV 能力。

**核心价值：**

| 指标 | 提升效果 |
|------|---------|
| TTFT（首 Token 延迟） | 降低约 **60%** |
| TPOT（每 Token 延迟） | 提升约 **13%** |
| QPM（每分钟请求数） | 提升约 **16%** |

**核心特性：**

- [x] **拥抱开源，无缝集成**
  - [x] 已合入 **Dynamo** 官方社区 — [ai-dynamo/dynamo#5858](https://github.com/ai-dynamo/dynamo/pull/5858)
  - [x] 已合入 **vLLM** 官方社区 — [vllm-project/vllm#34328](https://github.com/vllm-project/vllm/pull/34328)
  - [x] 与 **Mooncake** 社区合作共同开发 Distributed KV Cache Reuse — [taco-project/FlexKV#105](https://github.com/taco-project/FlexKV/pull/105)
  - [ ] 正在合入 **TensorRT-LLM** 官方社区
  - [ ] 正在与 **SGLang** 社区共建 KV Cache 解决方案
- [x] **多级缓存层次**：GPU → CPU 内存 → SSD → 远程存储（云存储），突破 GPU 显存瓶颈，缓存容量可扩展至 GPU 显存的 100 倍以上
- [x] **分布式 KV Cache 复用**：设计了一套分布式 RadixTree 机制，避免了中心化索引方案的单点性能瓶颈、网络延迟瓶颈和单点故障问题，实现跨节点 KV 缓存高效共享 — [分布式复用文档](https://github.com/taco-project/FlexKV/blob/main/docs/dist_reuse/README_zh.md)
- [x] **高性能 I/O**：支持 io_uring、GPU Direct Storage (GDS) 加速数据传输
- [x] **异步流水线**：Get/Put 操作与 GPU 计算重叠，隐藏数据搬运延迟

> 项目地址：[https://github.com/taco-project/FlexKV](https://github.com/taco-project/FlexKV)

---

## 第二部分：最佳实践 — Dynamo + vLLM + FlexKV

> **注意**：本部分为框架性指南，详细测试数据和调优参数将在后续补充完善。

### 2.1 方案概述

Dynamo + vLLM + FlexKV 是当前业界领先的 **KV Cache 感知路由 + 多级缓存卸载** 联合方案：

```
                         ┌─────────────────┐
                         │   用户请求       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Dynamo Frontend │
                         │  + KV Router     │  ← 基于全局 KV Cache 分布智能路由
                         └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
             ┌───────────┐ ┌───────────┐  ┌───────────┐
             │  Worker 0 │ │  Worker 1 │  │  Worker N │
             │  vLLM     │ │  vLLM     │  │  vLLM     │
             │ + FlexKV  │ │ + FlexKV  │  │ + FlexKV  │
             └───────────┘ └───────────┘  └───────────┘
                  │              │               │
            ┌─────┴─────┐ ┌─────┴─────┐   ┌─────┴─────┐
            │GPU→CPU→SSD│ │GPU→CPU→SSD│   │GPU→CPU→SSD│
            │ 多级缓存   │ │ 多级缓存   │   │ 多级缓存   │
            └───────────┘ └───────────┘   └───────────┘
```

**核心工作流程：**
1. 用户请求到达 Dynamo Frontend
2. KV Router 根据全局 KV Cache 分布信息，将请求路由到缓存命中率最高的 Worker
3. Worker 上的 vLLM 通过 FlexKV Connector 查询本地多级缓存，命中的 KV Cache 从 CPU/SSD 加载到 GPU
4. 请求完成后，FlexKV 异步将新生成的 KV Cache 卸载到 CPU/SSD
5. FlexKV 通过事件机制将缓存变更上报给 Dynamo KV Router，更新全局视图

### 2.2 部署模式：聚合部署 + KV 感知路由（推荐）

多 Worker 部署，利用 KV-Aware Routing 最大化缓存复用：

```bash
# 启动前端（开启 KV 路由）
python -m dynamo.frontend --router-mode kv --router-reset-states &

# 启动 Worker 0
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_0" \
CUDA_VISIBLE_DEVICES=0 \
python -m dynamo.vllm \
    --model YOUR_MODEL \
    --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

# 启动 Worker 1
DYNAMO_USE_FLEXKV=1 \
FLEXKV_CPU_CACHE_GB=32 \
FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_1" \
CUDA_VISIBLE_DEVICES=1 \
python -m dynamo.vllm \
    --model YOUR_MODEL \
    --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}'
```

### 2.3 配置参考

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `DYNAMO_USE_FLEXKV` | 启用 FlexKV 集成 | `0`（禁用） |
| `FLEXKV_CPU_CACHE_GB` | CPU 缓存大小（GB） | 必填 |
| `FLEXKV_CONFIG_PATH` | FlexKV 配置文件路径 | 不设置 |
| `FLEXKV_SERVER_RECV_PORT` | FlexKV IPC 端口 | 自动分配 |
| `FLEXKV_SSD_CACHE_GB` | SSD 缓存大小（GB） | 不启用 |
| `FLEXKV_SSD_CACHE_DIR` | SSD 缓存目录 | 不启用 |

**CPU + SSD 多级配置示例：**

```yaml
# flexkv_config.yml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data0/flexkv_ssd/;/data1/flexkv_ssd/
enable_gds: false
```

### 2.4 性能测试

> TODO：补充详细的 benchmark 数据，包括不同模型、不同缓存配置下的 TTFT / TPOT / QPM 对比。

推荐使用 [`aiperf`](https://github.com/ai-dynamo/aiperf) + [Mooncake Trace](https://github.com/kvcache-ai/Mooncake) 进行端到端性能评估：

```bash
aiperf profile \
  --model YOUR_MODEL \
  --tokenizer YOUR_TOKENIZER \
  --endpoint-type 'chat' \
  --endpoint '/v1/chat/completions' \
  --streaming \
  --url http://localhost:8000 \
  --input-file YOUR_TRACE \
  --random-seed 100
```

---

## 第三部分：技术深度解析

本部分详细介绍 Dynamo + vLLM + FlexKV 整个链路的核心技术实现，帮助大家理解背后的工作原理。

### 3.1 整体架构总览

FlexKV 的整体架构分为三层：控制面（GlobalCacheEngine）、数据面（TransferEngine）和存储面（StorageEngine），并通过 vLLM KVConnector 接口与推理引擎对接。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        vLLM 推理引擎                                          │
│  ┌────────────────────────────┐    ┌────────────────────────────────────┐    │
│  │      Scheduler 进程        │    │         Worker 进程                 │    │
│  │  ┌──────────────────────┐  │    │  ┌──────────────────────────────┐  │    │
│  │  │ FlexKVConnectorV1    │  │    │  │  FlexKVConnectorV1           │  │    │
│  │  │ (Scheduler 侧)       │  │    │  │  (Worker 侧)                 │  │    │
│  │  │                      │  │    │  │                              │  │    │
│  │  │ • get_num_new_       │  │    │  │  • register_kv_caches()      │  │    │
│  │  │   matched_tokens()   │  │    │  │  • start_load_kv()           │  │    │
│  │  │ • update_state_      │  │    │  │  • save_kv_layer()           │  │    │
│  │  │   after_alloc()      │  │    │  │  • get_finished()            │  │    │
│  │  │ • build_connector_   │  │    │  │                              │  │    │
│  │  │   meta()             │  │    │  └──────────────────────────────┘  │    │
│  │  │ • request_finished() │  │    │                                    │    │
│  │  │ • take_events()      │  │    │                                    │    │
│  │  └──────────┬───────────┘  │    │                                    │    │
│  └─────────────┼──────────────┘    └────────────────────────────────────┘    │
└────────────────┼────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FlexKV 核心引擎                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    KVManager / KVTaskEngine                         │     │
│  │              (任务生命周期管理：创建 / 启动 / 等待 / 回调)             │     │
│  └─────────────────────────┬───────────────────────────────────────────┘     │
│                            │                                                 │
│         ┌──────────────────┼──────────────────┐                              │
│         ▼                  ▼                  ▼                              │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐                       │
│  │GlobalCache  │   │ Transfer     │   │ Storage      │                       │
│  │Engine       │   │ Engine       │   │ Engine       │                       │
│  │(控制面)      │   │ (数据面)     │   │ (存储面)      │                       │
│  │             │   │              │   │              │                       │
│  │• RadixTree  │   │• 异步传输     │    │• GPU 缓冲区   │                     │
│  │  前缀匹配    │   │  调度器      │    │• CPU pin内存 │                       │
│  │• MemPool    │   │• 多进程       │    │• SSD mmap    │                      │
│  │  空间管理    │   │  Worker      │   │• 远程存储     │                      │
│  │• 淘汰策略    │   │• io_uring    │   │              │                       │
│  │             │   │• GDS / P2P   │   │              │                       │
│  └─────────────┘   └──────────────┘   └──────────────┘                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                    KVEventCollector                                 │     │
│  │          (事件收集器：收集 BlockStored / BlockRemoved 事件)           │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 KV Cache 存取全链路详解

这是整个系统最核心的数据通路。下面分别详细解析 **Store（存储/卸载）** 和 **Retrieve（读取/加载）** 两条路径。

---

#### 3.2.1 Store 路径：GPU → CPU → SSD（KV Cache 卸载）

当一个请求完成推理后，其在 GPU 上的 KV Cache 需要被卸载到更低层级的存储中，以释放宝贵的 GPU 显存，同时保留这些缓存以供后续相同前缀的请求复用。

**流程图：**

```
  vLLM Scheduler                          FlexKV 核心引擎
  ─────────────                          ──────────────────
       │
  请求生成完成
       │
       ▼
  ① request_finished(request, block_ids)
       │
       ▼
  ② put_match()  ──────────────────────► GlobalCacheEngine.put()
       │                                       │
       │                                  查询 RadixTree:
       │                                  哪些 block 已在 CPU/SSD 缓存?
       │                                  哪些是新的、需要写入?
       │                                       │
       │                                       ▼
       │                               ③ 分配物理块
       │                               CPU MemPool → 分配需要写入的 CPU 块
       │                               SSD MemPool → 分配需要写入的 SSD 块
       │                                       │
       │                                       ▼
       │                               ④ 插入 RadixTree（is_ready=False）
       │                               在索引中标记为"写入中，不可读取"
       │                                       │
       │                                       ▼
       │                               ⑤ 构建传输任务图
       │                               按依赖关系编排多级传输操作:
       │                               D2H(GPU→CPU) 完成后 → H2DISK(CPU→SSD)
       │                                       │
       ▼                                       ▼
  ⑥ launch_tasks() ──────────────────► TransferEngine.submit()
       │                               调度器将就绪的传输操作
       │                               分发到对应的 Worker 进程执行
       │                                       │
       │                                 ┌─────┴──────┐
       │                                 ▼            ▼
       │                          GPUCPUTransfer  CPUSSDDisk
       │                          Worker          TransferWorker
       │                          (D2H 拷贝)     (H2DISK 写入)
       │                          cudaMemcpy     io_uring
       │                                 │            │
       │                                 └─────┬──────┘
       │                                       │
       │                                       ▼
       │                               ⑦ 传输完成回调
       │                               • RadixTree 节点标记 is_ready=True
       │                               • 解锁节点（允许后续请求读取）
       │                               • 如启用分布式：发布到 Redis
       │                               • KVEventCollector.publish_stored()
       │                                       │
       │                                       ▼
       │                               ⑧ 事件上报（详见 3.3 节）
       │
  ⑨ get_finished() 返回已完成的请求 ID
       │
       ▼
  释放 GPU blocks，继续调度新请求
```

**关键代码路径（FlexKV 源码）：**

```python
# FlexKVSchedulerConnector.request_finished()
# 入口：vLLM 通知请求完成
def request_finished(self, request, block_ids):
    # 1. 调用 put_match 确定需要写入哪些块
    put_result = self.flexkv_manager.put_match(sequence_meta)

    # 2. put_match 内部调用 GlobalCacheEngine.put()
    #    → cpu_cache_engine.match_local(seq_meta)  # 查本地索引
    #    → ssd_cache_engine.match_local(seq_meta)
    #    → 确定哪些 block 是新的需要写入
    #    → 分配 CPU/SSD 物理块
    #    → 构建传输任务图

    # 3. 后续 launch_tasks() 提交传输任务到 TransferEngine
    return True  # 表示异步保存中，暂不释放 GPU 块
```

**Store 路径中的关键设计：**

- **增量写入**：只写入真正新增的 block，已缓存的跳过，避免重复写入
- **依赖调度**：H2DISK 操作必须等 D2H 完成后才执行，通过传输任务图的依赖边表达，同时最大化各层级之间的并行度
- **先插入后就绪**：先在 RadixTree 中占位（is_ready=False），传输完成后标记为就绪，防止并发读取到不完整数据
- **异步不阻塞推理**：整个 Store 过程与 vLLM 的下一批推理并行执行

---

#### 3.2.2 Retrieve 路径：CPU/SSD → GPU（KV Cache 加载）

当新请求到来时，如果其前缀已经存在于 FlexKV 的缓存中，可以直接加载到 GPU，避免重复计算。

**流程图：**

```
  vLLM Scheduler                          FlexKV 核心引擎
  ─────────────                          ──────────────────
       │
  新请求到达调度器
       │
       ▼
  ① get_num_new_matched_tokens(request, num_computed_tokens)
       │
       ▼
  ② get_match()  ──────────────────────► GlobalCacheEngine.get()
       │                                       │
       │                                  查询所有缓存层级的 RadixTree:
       │                                  • CPU 本地索引
       │                                  • SSD 本地索引
       │                                  • 分布式索引（如启用 P2P）
       │                                  • 远程存储索引（如启用）
       │                                       │
       │                               ┌───────┴────────┐
       │                               │   匹配结果       │
       │                               │ 假设命中 800 tokens │
       │                               │ 部分在 CPU 中    │
       │                               │ 部分在 SSD 中    │
       │                               └───────┬────────┘
       │                                       │
       │◄──── 返回 (800, True) ────────────────┘
       │      (800个可加载的token, 异步加载)
       │
       ▼
  ③ vLLM 为该请求分配 GPU blocks
       │
       ▼
  ④ update_state_after_alloc(request, blocks, num_external_tokens)
       │
       │  记录 GPU 目标块的 slot_mapping：
       │  确定每个外部 token 应写入 GPU 的哪个位置
       │
       ▼
  ⑤ build_connector_meta() ───────────► launch_tasks()
       │                                       │
       │                               构建传输任务图：
       │                               按依赖关系编排多级传输操作
       │                               DISK2H(SSD→CPU) 完成后 → H2D(CPU→GPU)
       │                               * 如启用 GDS：SSD 可直接传输到 GPU
       │                               * 如启用 P2P：先从远程节点传输到本地
       │                                       │
       │                                       ▼
       │                               TransferEngine 调度执行
       │                               • DISK2H: io_uring 异步读取
       │                               • H2D: cudaMemcpyAsync
       │                               • DISK2D: GDS 直通（如启用）
       │                                       │
       │                                       ▼
       │                               ⑥ 传输完成回调
       │                               • 释放临时 CPU 块
       │                               • 解锁 RadixTree 节点
       │
       ▼
  ⑦ GPU 上已有完整的 KV Cache
     vLLM 从 token 800 开始 Prefill（跳过前 800 token 的计算）
     大幅减少 TTFT！
```

**关键代码路径：**

```python
# FlexKVSchedulerConnector.get_num_new_matched_tokens()
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    # 1. 构建 SequenceMeta（包含 token hash）
    seq_meta = SequenceMeta.from_request(request)

    # 2. 查询 GlobalCacheEngine
    task_id, num_matched = self.flexkv_manager.get_match(seq_meta)
    #    → cpu_cache_engine.match(seq_meta)      # 查 CPU RadixTree
    #    → ssd_cache_engine.match(seq_meta)      # 查 SSD RadixTree
    #    → 如果 P2P: match_all() 同时查本地+远程
    #    → 取最长匹配

    return num_matched, True  # 返回匹配数和"异步加载"标志
```

**Retrieve 路径中的关键设计：**

- **多级查找优先级**：CPU（最快）→ SSD → 远程节点 → 远程存储，选择最长前缀匹配
- **临时缓冲区**：SSD 数据需要先读到 CPU 临时块，再统一上传 GPU；传输完成后释放临时块
- **GDS 直通优化**：启用 GPU Direct Storage 时，SSD 数据可直接 DMA 到 GPU，跳过 CPU 中转
- **P2P 跨节点复用**：通过分布式 RadixTree 发现其他节点上的缓存，RDMA 传输到本地

---

#### 3.2.3 StorageEngine：多级存储管理

StorageEngine 负责物理层的内存/磁盘管理：

```
┌────────────────────────────────────────────────┐
│               StorageEngine                      │
│                                                  │
│  ┌────────────┐  注册 vLLM 的 KV Cache 张量      │
│  │  GPU 层     │  register_gpu_blocks()           │
│  │  (不分配)   │  直接引用 vLLM 已分配的显存        │
│  └────────────┘                                   │
│                                                   │
│  ┌────────────┐  分配 pinned memory               │
│  │  CPU 层     │  torch.cuda.pin_memory()         │
│  │             │  保证 H2D/D2H 高速传输            │
│  └────────────┘                                   │
│                                                   │
│  ┌────────────┐  mmap 映射文件                     │
│  │  SSD 层     │  支持多盘并行 I/O                 │
│  │             │  io_uring 异步 / GDS 直通         │
│  └────────────┘                                   │
│                                                   │
│  ┌────────────┐  PCFS / 云存储后端                 │
│  │  远程存储层  │  用于持久化和超大容量场景          │
│  └────────────┘                                   │
└────────────────────────────────────────────────┘
```

**Block 布局：**

每个 Block 的数据布局与 GPU 上的 KV Cache 形状保持一致：
```
Block = [num_layers × 2(K+V) × block_size × num_heads × head_dim]
```
支持 `BLOCKFIRST`（按 block 连续存储）和 `LAYERFIRST`（按 layer 连续存储）两种布局模式。

---

#### 3.2.4 TransferEngine：异步传输调度器

TransferEngine 是数据面的核心，采用 **事件驱动的依赖调度** 模型：

```
┌─────────────────────────────────────────────────────────────┐
│                    TransferEngine                             │
│                                                              │
│  ┌─────────────────────────────────────────────────┐        │
│  │               调度器主循环                        │        │
│  │          _scheduler_loop() (selectors)           │        │
│  │                                                   │        │
│  │  1. 从 task_queue 接收新的传输任务图               │        │
│  │  2. 找出无依赖的操作，分发到对应 Worker            │        │
│  │  3. 从 finished_ops_queue 收集已完成的操作         │        │
│  │  4. 更新依赖状态，触发后续操作                     │        │
│  │  5. 全部完成 → 触发回调                           │        │
│  └─────────────────────────────────────────────────┘        │
│                         │                                    │
│           ┌─────────────┼─────────────┐                      │
│           ▼             ▼             ▼                       │
│  ┌──────────────┐ ┌──────────┐ ┌──────────────┐             │
│  │GPUCPUTransfer│ │CPUSSDDisk│ │CPURemote     │             │
│  │Worker        │ │Transfer  │ │TransferWorker│             │
│  │              │ │Worker    │ │              │              │
│  │• D2H (GPU→  │ │• H2DISK  │ │• H2REMOTE    │             │
│  │  CPU)       │ │  (CPU→   │ │  (CPU→远程)  │              │
│  │• H2D (CPU→  │ │   SSD)   │ │• REMOTE2H    │             │
│  │  GPU)       │ │• DISK2H  │ │  (远程→CPU)  │              │
│  │             │ │  (SSD→   │ │              │              │
│  │cudaMemcpy  │ │   CPU)   │ │              │              │
│  │Async       │ │          │ │              │              │
│  │             │ │io_uring  │ │              │              │
│  └──────────────┘ └──────────┘ └──────────────┘             │
│                                                              │
│  ┌──────────────┐ ┌──────────────┐                           │
│  │GDSTransfer   │ │PEER2CPU      │                           │
│  │Worker        │ │TransferWorker│                           │
│  │              │ │              │                           │
│  │• D2DISK      │ │• PEERH2H     │                           │
│  │  (GPU→SSD)  │ │  (远程CPU→   │                           │
│  │• DISK2D     │ │   本地CPU)   │                           │
│  │  (SSD→GPU)  │ │• PEERSSD2H   │                           │
│  │             │ │  (远程SSD→   │                           │
│  │GDS 直通    │ │   本地CPU)   │                           │
│  └──────────────┘ └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘

每个 Worker 是独立的子进程（mp.spawn），确保 CUDA context 隔离
```

**为什么使用依赖调度？**

因为 Store/Retrieve 操作涉及多级存储之间的级联传输，存在天然的依赖关系：
- Store: `D2H` 完成后才能执行 `H2DISK`（CPU 里有数据后才能写 SSD）
- Retrieve: `DISK2H` 完成后才能执行 `H2D`（SSD 数据读到 CPU 后才能上传 GPU）
- 但同一级别的不同传输操作可以并行执行

传输任务图通过依赖边自然地表达了这些关系，同时最大化并行度。

---

### 3.3 take_events：全局 KV Cache 分布上报机制

这是 FlexKV 与 Dynamo KV Router 协作的关键通路。当 FlexKV 的缓存状态发生变化时，需要通知 Dynamo 的 KV Router，让路由器知道每个 Worker 上缓存了哪些 token 前缀，从而做出最优的路由决策。

#### 3.3.1 整体事件流转图

```
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Worker 0 │     │ Worker 1 │     │ Worker N │
  │ vLLM +   │     │ vLLM +   │     │ vLLM +   │
  │ FlexKV   │     │ FlexKV   │     │ FlexKV   │
  └────┬─────┘     └────┬─────┘     └────┬─────┘
       │                │                │
  ①缓存变更事件产生  ①缓存变更事件产生  ①缓存变更事件产生
  (Store/Evict)     (Store/Evict)     (Store/Evict)
       │                │                │
       ▼                ▼                ▼
  ②KVEventCollector ②KVEventCollector ②KVEventCollector
  (FlexKV内部)      (FlexKV内部)      (FlexKV内部)
  收集 BlockStored  收集 BlockStored  收集 BlockStored
  / BlockRemoved   / BlockRemoved   / BlockRemoved
       │                │                │
       ▼                ▼                ▼
  ③take_events()    ③take_events()    ③take_events()
  (vLLM Scheduler   (vLLM Scheduler   (vLLM Scheduler
   每步调用)          每步调用)          每步调用)
       │                │                │
       ▼                ▼                ▼
  ④ZmqEventPublisher ④ZmqEventPublisher ④ZmqEventPublisher
  ZMQ PUB socket    ZMQ PUB socket    ZMQ PUB socket
  tcp://*:20080     tcp://*:20081     tcp://*:2008N
       │                │                │
       │    msgpack 编码序列化            │
       │    [topic, seq_number, payload]  │
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼ (ZMQ SUB 订阅)
  ┌─────────────────────────────────────────────────────┐
  │              Dynamo Event Plane                       │
  │                                                       │
  │  ⑤ KvEventPublisher (Rust)                            │
  │     • start_zmq_listener() — 订阅各 Worker 的 ZMQ     │
  │     • 解码 msgpack → 转换为 KvCacheEvent               │
  │     • 批量聚合（合并连续的 Stored/Removed 事件）        │
  │     • 发布到 NATS Event Plane 或直连 Router            │
  │                                                       │
  │  ⑥ 同时写入 LocalKvIndexer（每 Worker 一个本地索引）   │
  │     • 支持 Router 启动时拉取全量状态快照               │
  │     • 支持 gap detection 和状态恢复                    │
  └─────────────────────┬───────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────────┐
  │              Dynamo KV Router                         │
  │                                                       │
  │  ⑦ Subscriber 接收事件                                │
  │     • 从 NATS / ZMQ 接收 RouterEvent                  │
  │     • 检测 event_id gap → 触发从 Worker 恢复          │
  │                                                       │
  │  ⑧ KvIndexer 维护全局 RadixTree                       │
  │     • apply_event(BlockStored): 在树中插入新节点       │
  │       标记该 Worker 拥有这些 block                     │
  │     • apply_event(BlockRemoved): 从树中移除节点        │
  │       解除该 Worker 与这些 block 的关联                │
  │                                                       │
  │  ⑨ 路由决策                                           │
  │     新请求到达 → compute_block_hash() → find_matches()│
  │     → 计算每个 Worker 的缓存重叠分数                   │
  │     → 综合考虑缓存命中 + 负载均衡                     │
  │     → 选择最优 Worker                                 │
  └─────────────────────────────────────────────────────┘
```

#### 3.3.2 事件产生：FlexKV 内部的 KVEventCollector

FlexKV 在以下两个时机产生事件：

**1. BlockStored 事件** — 当新 Block 写入缓存时，在 `CacheEngine.insert()` 中触发。FlexKV 将 block 的哈希值、block_size、存储介质类型（CPU/SSD）等信息封装为 `BlockStored` 事件。

**2. BlockRemoved 事件** — 当缓存空间不足需要淘汰旧 Block 时，在 `CacheEngine.take()` 的淘汰逻辑中触发。被淘汰 block 的哈希值和存储介质类型被封装为 `BlockRemoved` 事件。

**KVEventCollector 的核心设计：**
- 使用 **后台线程 + 队列** 异步处理事件，确保事件创建不阻塞 GPU 计算的热路径
- `take_events()` 方法通过加锁原子性地取出并清空所有缓冲事件，供 vLLM 调度器周期性调用
- 仅在 `DYNAMO_USE_FLEXKV=1` 时创建，即仅在 Dynamo 集成场景下启用事件收集

#### 3.3.3 事件消费：vLLM Scheduler 调用 take_events()

在 vLLM 的调度循环中，Scheduler 每一步都会调用 connector 的 `take_events()` 方法：

```python
# vLLM Scheduler 主循环中
def _schedule(self):
    # ... 调度逻辑 ...

    # 从 KV Cache Manager 收集事件
    events = self.kv_cache_manager.take_events()

    # 从 KV Connector（FlexKV）收集事件
    connector_events = self.connector.take_events()

    # 合并所有事件
    all_events = list(events) + list(connector_events)

    # 事件被包含在 SchedulerOutput 中传递给 EventPublisher
```

#### 3.3.4 事件发布：ZmqEventPublisher

vLLM 将收集到的事件通过 ZMQ PUB socket 发布出去。`ZmqEventPublisher` 在后台线程中运行，从事件队列取出事件，通过 msgpack 编码后以 `[topic, seq_number, payload]` 的格式通过 ZMQ 多帧消息发送。

**可靠性保障：**
- 每条消息携带递增 **序列号**（seq_number），接收端可检测是否丢消息
- **回放缓冲区**（deque，默认保存最近 10,000 条），支持通过 ROUTER socket 请求重传
- 每个 data_parallel_rank 使用 **独立端口**，避免事件混淆

#### 3.3.5 事件接收：Dynamo KV Router

Dynamo 侧使用 Rust 实现的 `KvEventPublisher` 订阅 vLLM 的 ZMQ 事件：

```
Dynamo 侧事件处理流程:

  ZMQ SUB socket ──► start_zmq_listener() ──► mpsc channel
                     解码 msgpack               │
                     转换事件格式                 ▼
                                          run_event_processor_loop()
                                          批量聚合优化
                                                │
                                    ┌───────────┼───────────┐
                                    ▼                       ▼
                             LocalKvIndexer          Event Plane (NATS)
                             (本地索引快照)           (广播给 Router)
                                                         │
                                                         ▼
                                                    KvIndexer
                                                    (全局 RadixTree)
```

**事件处理的批量聚合优化：**

Dynamo 的 `run_event_processor_loop()` 在转发事件前会进行智能聚合：
- 连续的 `BlockStored` 事件如果具有链式 parent_hash 关系，合并为一个大事件
- 连续的 `BlockRemoved` 事件合并
- 减少 NATS 消息数量，提升吞吐

#### 3.3.6 路由决策：基于全局 RadixTree 的 KV-Aware Routing

Dynamo KV Router 核心的路由逻辑：

```
新请求到达
    │
    ▼
compute_block_hash_for_seq(tokens)
  将 token 序列按 block_size 分块
  每块计算 XXH3 hash → [hash_0, hash_1, ..., hash_N]
    │
    ▼
indexer.find_matches([hash_0, hash_1, ..., hash_N])
  在全局 RadixTree 中查找：

          root
         / | \
     hash_0  hash_0  hash_0
     (W0)   (W1)    (W2)
       |      |
    hash_1  hash_1
    (W0)    (W1)
       |
    hash_2
    (W0)

  结果: W0 匹配 3 blocks, W1 匹配 2 blocks, W2 匹配 1 block
    │
    ▼
Worker 选择（代价函数）:
  对每个 Worker 计算代价:
  cost = overlap_weight × (ISL - matched_blocks × block_size) / block_size
         + current_decode_blocks

  解释：
  • 第一项：需要重新 prefill 的 token 数（匹配越多，代价越低）
  • 第二项：当前 decode 负载（均衡考虑）
  • overlap_weight：可配置的权重（越大越倾向缓存命中）
    │
    ▼
  选择代价最低的 Worker（支持 softmax 采样）
    │
    ▼
  将请求路由到该 Worker
```

**示例场景：**

假设有 3 个 Worker，一个新请求包含系统 prompt + 用户问题：

```
请求 tokens: [系统prompt 10个block] + [用户问题 2个block]

Worker 0 的缓存: [系统prompt 10个block] + [之前用户A的问题 3个block]
Worker 1 的缓存: [系统prompt 10个block]
Worker 2 的缓存: [其他系统prompt 5个block]

匹配结果:
  Worker 0: 10 blocks（系统prompt完全匹配）
  Worker 1: 10 blocks（系统prompt完全匹配）
  Worker 2: 0 blocks

路由决策: Worker 0 和 Worker 1 缓存命中相同，
         但 Worker 0 当前 decode 负载更高 → 选择 Worker 1
```

这样，请求被路由到 Worker 1 后，FlexKV 只需从 CPU/SSD 加载 10 个 block 的 KV Cache 到 GPU，跳过系统 prompt 的 prefill 计算，**TTFT 大幅降低**。

---

### 3.4 关键技术总结

| 技术点 | 实现方式 | 核心价值 |
|-------|---------|---------|
| **多级缓存层次** | GPU → CPU(pinned) → SSD(mmap/io_uring) → Remote(PCFS) | 突破 GPU 显存瓶颈，缓存容量提升 100 倍+ |
| **RadixTree 前缀索引** | C++ 加速 RadixTree + 内容哈希 block 去重 | O(L) 前缀匹配，支持高效缓存查找 |
| **异步传输调度** | 事件驱动依赖调度器 + 多进程 Worker | 最大化并行度，异步不阻塞推理 |
| **事件驱动分布式感知** | KVEventCollector → ZMQ PUB → NATS → KvIndexer | 全局缓存视图实时更新，毫秒级延迟 |
| **KV-Aware Routing** | 全局 RadixTree + 代价函数路由 | 最大化缓存复用，减少冗余计算 |
| **GDS 直通** | GPU Direct Storage: SSD ↔ GPU 直接 DMA | 消除 CPU 中转开销，带宽提升 2-3 倍 |
| **分布式 P2P 复用** | Redis 分布式 RadixTree + RDMA/ZMQ P2P 传输 | 跨节点缓存共享，集群级别缓存复用 |
| **可靠事件传输** | ZMQ PUB + 序列号 + 回放缓冲区 + gap detection | 保证事件不丢失，支持故障恢复 |

---

### 3.5 端到端数据流总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        完整请求处理流程                                   │
│                                                                         │
│  用户请求                                                                │
│    │                                                                    │
│    ▼                                                                    │
│  Dynamo Frontend                                                        │
│    │                                                                    │
│    ├──► KV Router: 计算 token hash → 查 RadixTree → 路由到 Worker 2     │
│    │                                                                    │
│    ▼                                                                    │
│  Worker 2 (vLLM + FlexKV)                                               │
│    │                                                                    │
│    ├──► Scheduler: get_num_new_matched_tokens()                         │
│    │    FlexKV 查缓存: "找到了！CPU 里有 800 tokens 的 KV Cache"         │
│    │                                                                    │
│    ├──► FlexKV Retrieve: CPU → GPU (H2D, 异步传输)                      │
│    │                                                                    │
│    ├──► vLLM Prefill: 只需计算剩余 200 tokens（跳过 800 tokens！）      │
│    │                                                                    │
│    ├──► vLLM Decode: 生成响应                                           │
│    │                                                                    │
│    ├──► FlexKV Store: GPU → CPU → SSD（异步卸载新 KV Cache）            │
│    │                                                                    │
│    └──► take_events() → ZMQ → Dynamo → KV Router 更新 RadixTree        │
│                                                                         │
│  响应返回给用户                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*本文档由腾讯云 TACO 团队编写，如有疑问欢迎联系我们。*

*项目地址：[https://github.com/taco-project/FlexKV](https://github.com/taco-project/FlexKV)*
