# FlexKV 支持 Gemma4 31B：多组异构 KV Cache 传输

## 1. 背景

### 1.1 Gemma4 31B 的 KV Cache 异构结构

Gemma4 31B 是一个混合注意力模型，其不同层的 KV cache 形状不同：

| 层类型 | 层数 | KV Head 数 | Head Dim | 每 token KV 大小 (bf16) |
|--------|------|-----------|----------|------------------------|
| Sliding Window Attention | 50 | 16 | 256 | 16×256×2×2 = 16,384 B |
| Full Attention | 10 | 4 | 512 | 4×512×2×2 = 8,192 B |
| **合计** | **60** | — | — | **每 token 总计 ≈ 900 KB** |

> 注：vLLM 中 Gemma4 部分层共享 KV tensor（同一个 data_ptr），去重后实际为 60 层。

### 1.2 vLLM 侧的 Gemma4 支持

vLLM 对 Gemma4 有以下关键处理：

- **强制使用 TRITON_ATTN 后端**：由于 head_dim 不统一（256 vs 512），flash_attn 无法同时处理两种 head_dim，vLLM 在检测到 Gemma4 的异构 head_dim 后自动切换到 triton attention
- **Hybrid KV Cache Manager 被禁用**：当 `--kv-transfer-config` 启用时，vLLM 关闭混合 KV cache 管理器
- **KV tensor 布局**：triton attention 的 GPU KV cache tensor shape 为 `[num_blocks, 2, block_size, num_kv_heads, head_size]`，而 flash_attn 为 `[2, num_blocks, block_size, num_kv_heads, head_size]`，dim 0 和 dim 1 互换

### 1.3 FlexKV 的核心挑战

FlexKV 原先假设所有层的 KV cache 形状一致（统一的 `num_kv_heads` 和 `head_size`）。这个假设贯穿了从配置、buffer 分配、stride 计算到数据传输的整个流程。支持 Gemma4 需要引入 **"Layer Group"（层组）** 概念，让不同组的层使用不同的 KV 参数。

---

## 2. FlexKV 代码修改

### 2.1 整体思路

核心设计：引入 `LayerGroupSpec` 描述每组层的 KV 参数，使用单一 BLOCKFIRST CPU buffer 存储所有组的数据（每个 block 内按组排列），传输时按组分别调用一次 `transfer_kv_blocks()`。

数据流：

```
vLLM Worker (GPU KV tensors, 分组)
    ↓ register_to_server() 探测 shape → 构建 LayerGroupSpec
FlexKV Client (RegisterTPClientRequest + layer_groups)
    ↓ ZMQ
TransferManager (接收 layer_groups → 重算 block count → 创建 StorageEngine)
    ↓
TransferEngine (per-group kwargs → Worker)
    ↓
GPU<->CPU Worker (per-group strides, offset → N 次 transfer_kv_blocks 调用)
```

### 2.2 配置层

**`flexkv/common/config.py`**

```python
@dataclass
class LayerGroupSpec:
    """一组共享相同 KV cache shape 的层"""
    num_layers: int        # 该组的层数
    num_kv_heads: int      # KV head 数
    head_size: int         # head 维度
    layer_indices: List[int]  # 在全部层中的索引
```

- `ModelConfig` 新增 `layer_groups: Optional[List[LayerGroupSpec]]`
- `ModelConfig.token_size_in_bytes` 改为按组求和：当 `layer_groups` 有值时，`sum(g.num_layers * g.num_kv_heads * g.head_size * kv_dim for g in layer_groups) * dtype_size`
- `CacheConfig` 新增 `_user_cpu_cache_gb` / `_user_ssd_cache_gb`，保存用户原始配置，用于延迟重算 block 数量

### 2.3 存储布局

**`flexkv/common/storage.py`**

`KVCacheLayout` 新增 `layer_groups` 字段。当 `layer_groups` + BLOCKFIRST 模式时：

- `_compute_kv_shape()` → `[num_block, elements_per_block]`，其中 `elements_per_block = sum(每组的 num_layers × kv_dim × tpb × num_kv_heads × head_size)`
- `get_block_stride()` 正常工作（返回总 elements per block）
- `get_layer_stride()` / `get_kv_stride()` / `get_chunk_size()` 抛出 ValueError（多组无单一值）
- 新增 `get_group_strides()` 返回每组的 (offset, layer_stride, kv_stride, chunk_size)

CPU block 内存布局示意（单个 block 内）：

```
[group0_layer0_k | group0_layer0_v | group0_layer1_k | ... | group0_layer49_v |
 group1_layer0_k | group1_layer0_v | group1_layer1_k | ... | group1_layer9_v ]
```

### 2.4 vLLM 适配器

**`flexkv/integration/vllm/vllm_v1_adapter.py`** — `FlexKVWorkerConnector.register_to_server()`

1. **去重**：遍历 `kv_caches` dict，按 `data_ptr()` 去重（Gemma4 部分层共享同一 tensor）
2. **分组**：按 `(num_kv_heads, head_size)` 将去重后的层分组（使用 `OrderedDict` 保持顺序）
3. **检测**：如果只有 1 组 → 走原有统一路径；如果 >1 组 → 构建 `LayerGroupSpec` 列表、每组 GPU layout 和 handles，调用多组注册

### 2.5 请求与客户端

**`flexkv/server/request.py`** — `RegisterTPClientRequest` 新增：
- `layer_groups: Optional[List[LayerGroupSpec]]`
- `gpu_layouts: Optional[List[KVCacheLayout]]`（每组的 GPU layout）
- `handles_per_group: Optional[List[List[TensorSharedHandle]]]`（每组的 GPU tensor handles）

**`flexkv/server/client.py`** — `KVTPClient.register_to_server()` 新增对应参数，序列化后通过 ZMQ 发送

### 2.6 传输管理器

**`flexkv/transfer_manager.py`**

关键改动：**延迟 StorageEngine 创建**

原来 `StorageEngine` 在 `__init__` 中创建（此时 `layer_groups` 未知）→ 使用 max(num_kv_heads)×max(head_size) 计算 → buffer 过大、block 数过少。

修改后：
1. `__init__` 中 `self.storage_engine = None`
2. `initialize_transfer_engine()` 中：先完成 GPU 注册（获得 `layer_groups`）→ 用正确的 `token_size_in_bytes` 重算 `num_cpu_blocks` → 再创建 `StorageEngine`

Gemma4 效果：16GB CPU cache，`num_cpu_blocks` 从 546（过大估算）→ 1191（正确值）

### 2.7 传输引擎

**`flexkv/transfer/transfer_engine.py`**

- 构造函数接收 `gpu_blocks_per_group` 和 `gpu_layouts_per_group`（按 dp_client_id 索引）
- 提供 helper 方法：`_get_multi_group_kwargs_tp1()`, `_get_multi_group_kwargs_tp()` 等，从按 dp_client_id 组织的数据中提取按 (group, device) 组织的参数
- 创建 Worker 时通过 `**kwargs` 传入多组参数

### 2.8 GPU<->CPU 传输 Worker

**`flexkv/transfer/worker.py`** — `GPUCPUTransferWorker`

**stride 自动探测**（解决 triton vs flash_attn 布局差异）：
```python
@staticmethod
def _get_gpu_strides_from_tensor(tensor, tokens_per_block, dtype_size, is_mla):
    # 5D tensor: 前 3 维是 (num_blocks, kv_dim=2, block_size) 的某种排列
    # 通过 dim size 识别哪个是 kv_dim(=2)、block_size(=tpb)、num_blocks
    # 用 tensor.stride() 得到实际内存 stride
```

**多组初始化** `_init_multi_group()`：
- 对每组计算：GPU ptrs、GPU strides（从 tensor 探测）、CPU strides（基于 BLOCKFIRST 布局）、CPU offset
- 所有组共享同一个 `cpu_block_stride`（整个 block 大小）

**多组传输**：在 `_transfer_impl()` 中，对每组调用一次 `transfer_kv_blocks()`：
```python
for gi, gp in enumerate(self.group_transfer_params):
    gpu_ptrs = gp['gpu_ptrs']
    cpu_tensor_for_group = self.cpu_tensor[gp['cpu_offset_elements']:]
    transfer_kv_blocks(
        gpu_block_ids, gpu_ptrs,
        gp['gpu_kv_stride'], gp['gpu_block_stride'], gp['gpu_layer_stride'],
        cpu_block_ids, cpu_tensor_for_group,
        gp['cpu_kv_stride'], gp['cpu_layer_stride'], gp['cpu_block_stride'],
        gp['chunk_size'],
        start_layer_id=0, num_layers=gp['num_layers'],
        ...
    )
```

TP 模式（`tpGPUCPUTransferWorker`）有对应的 `_init_tp_multi_group()`，逻辑类似但需处理多 device。

### 2.9 CPU<->SSD / GDS Worker

**`flexkv/transfer/worker.py`** 中的 `CPUSSDDiskTransferWorker` 和 `GDSTransferWorker` 也有对应的多组支持：

- `_init_multi_group_ssd()`：计算每组的 CPU/SSD stride 和 offset
- `_init_multi_group_gds()` / `_init_tp_multi_group_gds()`：GDS 变体

### 2.10 任务引擎修复

**`flexkv/kvtask.py`** — `KVTaskEngine`

- `get_match()` 和 `put_match()` 开头新增 `self._update_tasks(timeout=0)`
- 目的：在查询 radix tree 之前，先 flush 管道中待处理的 D2H 完成事件，确保 `set_ready` 回调已执行

**`flexkv/server/server.py`** — `KVServer`

- 主循环在分发请求前调用 `self.kv_task_engine._update_tasks(timeout=0)`（用于 server_client_mode 场景）

---

## 3. 验证过程中发现和解决的问题

### 3.1 问题一：GPU Stride 计算错误（dp1 验证阶段发现）

**现象**：dp1 模式下 KV cache 传输后数据损坏，Round 2/3 结果与 Round 1 不一致

**根因**：FlexKV 从 `KVCacheLayout`（LAYERFIRST 类型）计算 GPU stride，假设 flash_attn 的 `[2, N, B, H, D]` 布局。但 Gemma4 使用 triton attention，GPU tensor 实际布局为 `[N, 2, B, H, D]`，dim 0 和 dim 1 互换，导致 `kv_stride` 和 `block_stride` 计算错误。

**修复**：新增 `_get_gpu_strides_from_tensor()` 静态方法，直接从 GPU tensor 的实际 `shape` 和 `stride()` 推断正确的 kv_stride / block_stride / layer_stride，而不依赖 layout 类型的假设。该方法通过 dim size 识别各维度（kv_dim=2, block_size=tpb, 剩余为 num_blocks），然后用 `tensor.stride()` 获取实际内存步长。

**涉及文件**：`flexkv/transfer/worker.py`

### 3.2 问题二：CPU Buffer 过度分配

**现象**：16GB CPU cache 只能分配 546 个 block（过少），大量 KV cache 无法缓存

**根因**：在 GPU 注册之前（`layer_groups` 未知时），FlexKV 用 `max(num_kv_heads)=16` 和 `max(head_size)=512` 为所有 60 层计算 `token_size_in_bytes`，得到的 per-token 大小是实际值的 ~2.18 倍。后续 `StorageEngine` 在 `__init__` 中创建，使用了这个错误的大小。

**修复**：
1. 延迟 `StorageEngine` 创建到 `initialize_transfer_engine()` 中，在 GPU 注册完成后执行
2. GPU 注册时从 `RegisterTPClientRequest` 获取 `layer_groups`，写入 `model_config`
3. 用正确的 `token_size_in_bytes`（考虑每组实际大小）重算 `num_cpu_blocks`
4. 效果：`num_cpu_blocks` 从 546 → 1191

**涉及文件**：`flexkv/transfer_manager.py`, `flexkv/common/config.py`

### 3.3 问题三：D2H 完成事件未及时处理（dp2 验证阶段发现）

**现象**：dp2 模式下 Round 2 的 `get_match` 返回 `cpu_matched=2/0`——radix tree 中存在节点但 `is_ready=false`，导致 0 个 block 可用于 H2D 传输

**根因分析**：

1. dp2 模式下 `server_client_mode=False`，每个 DP engine 拥有独立的 `KVTaskEngine`（不经过 KVServer）
2. D2H 传输完成后，`CompletedOp` 通过 `mp.Pipe` 从 TransferManager 子进程发送到 KVTaskEngine
3. `set_ready` 回调只在 `_update_tasks()` 读取管道并处理 CompletedOp 时才执行
4. vLLM scheduler 的调用顺序是：`get_match`（调度阶段） → `build_connector_meta`（启动传输） → forward pass → `update_connector_output`（调用 `query_finished_task` → `try_wait` → `_update_tasks`）
5. 当 scheduler 在两轮请求间空闲时，不会调用 `update_connector_output`，因此 `_update_tasks` 不被调用
6. D2H 完成事件积压在管道中 → 新请求到来时 `get_match` 先执行 → 看到节点未 ready

**修复**：在 `KVTaskEngine.get_match()` 和 `put_match()` 开头添加 `self._update_tasks(timeout=0)`，在查询 radix tree 前先 flush 管道中的待处理完成事件。

**涉及文件**：`flexkv/kvtask.py`

---

## 4. 复现流程

### 4.1 环境准备

**硬件要求**：2× GPU（Gemma4 31B 需要约 60GB 显存，2×A100-80G 或类似配置）

**软件环境**：
- vLLM（需包含 Gemma4 支持和 KV connector 接口，版本 >= 0.19.0）
- FlexKV（包含本文档描述的所有修改）
- FlexKV C++ 库已编译（`build/lib` 下）

**环境变量**：
```bash
export CUDA_VISIBLE_DEVICES=0,1
export FLEXKV_CPU_CACHE_GB=16       # CPU cache 大小
export FLEXKV_SSD_CACHE_GB=0        # 不使用 SSD cache
export LD_LIBRARY_PATH=/workspace/FlexKV/build/lib:$LD_LIBRARY_PATH
```

### 4.2 dp1 模式（tp=1, dp=1）

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/gemma-4-31B-it \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --enforce-eager \
    --trust-remote-code \
    --port 8000 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector": "FlexKVConnectorV1", "kv_role": "kv_both"}'
```

### 4.3 dp2 模式（tp=1, dp=2）

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/gemma-4-31B-it \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --enforce-eager \
    --trust-remote-code \
    --port 8000 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector": "FlexKVConnectorV1", "kv_role": "kv_both"}'
```

### 4.4 验证测试

测试思路：发送 3 轮相同的请求（temperature=0），验证 KV cache 复用的正确性。

- **Round 1**：首次计算，触发 D2H offload（GPU → CPU）
- **Round 2**：相同 prompt，应触发 H2D reload（CPU → GPU），跳过 prefill
- **Round 3**：再次相同 prompt，同样触发 H2D reload

**正确性判定**：
- Round 2 == Round 3（**必须通过**，两轮都走 cache reload 路径）
- Round 1 != Round 2 是可接受的（vLLM prefill chunking 在有/无 cache 时可能有微小非确定性）

发送测试请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/gemma-4-31B-it",
    "messages": [{"role": "user", "content": "What is the square root of 144?"}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

### 4.5 关键日志确认

启动成功后，检查以下日志：

**1) Layer groups 检测正确**：
```
Set model_config.layer_groups from GPU 0: [(50, 16, 256), (10, 4, 512)]
```

**2) Block 数量正确重算**：
```
Recomputed num_cpu_blocks with layer_groups: 546 -> 1191
```

**3) CPU 分配大小正确**（约 16GB 而非过大值）：
```
CPU allocate total_size: 15.99 GB
```

**4) D2H 传输完成**（Round 1 后）：
```
D2H transfer request: N finished transfer data size: X GB
```

**5) H2D 传输完成**（Round 2/3）：
```
H2D transfer request: N finished transfer data size: X GB
```

### 4.6 完整的 dp2 自动化测试脚本

项目中包含 `test_dp2_kv_reuse.sh`，自动化执行上述流程：

```bash
cd /workspace/FlexKV
bash test_dp2_kv_reuse.sh
```

脚本会：
1. 启动 dp2 vLLM server
2. 等待 server ready
3. 发送 3 组 × 3 轮请求
4. 验证 Round2 == Round3
5. 输出 PASS/FAIL 结果
6. 打印 D2H/H2D 传输日志摘要

---

## 5. 修改文件汇总

| 文件 | 修改内容 |
|------|---------|
| `flexkv/common/config.py` | 新增 `LayerGroupSpec`；`ModelConfig.layer_groups`；`token_size_in_bytes` 按组求和；`CacheConfig._user_*_cache_gb` |
| `flexkv/common/storage.py` | `KVCacheLayout.layer_groups`；多组 shape 计算；`get_group_strides()` |
| `flexkv/integration/vllm/vllm_v1_adapter.py` | KV tensor 去重、分组探测、多组注册 |
| `flexkv/server/request.py` | `RegisterTPClientRequest` 多组字段 |
| `flexkv/server/client.py` | `register_to_server()` 多组参数 |
| `flexkv/transfer_manager.py` | 延迟 StorageEngine 创建；layer_groups 传播；block 数重算 |
| `flexkv/storage/storage_engine.py` | 传递 `layer_groups` 到 CPU/SSD/Remote layout |
| `flexkv/transfer/transfer_engine.py` | 存储并分发 per-group GPU 数据到 worker |
| `flexkv/transfer/worker.py` | GPU stride 自动探测；多组 init/transfer（GPU↔CPU, CPU↔SSD, GDS） |
| `flexkv/kvtask.py` | `get_match()`/`put_match()` 中 flush D2H 完成事件 |
| `flexkv/server/server.py` | 主循环 flush D2H 完成事件（server_client_mode） |

---

## 6. 已知限制与后续工作

- **CPU↔SSD 多组传输**：框架已搭建（`_init_multi_group_ssd`），但 C++ 层的 `transfer_kv_blocks_ssd` 尚未添加 `ssd_copy_offset` 参数，多组 SSD 传输暂不可用
- **GDS 多组传输**：框架已搭建，实际验证待进行
- **Remote/PEER 多组传输**：优先级较低，待后续实现
- **非 BLOCKFIRST 布局**：多组支持仅针对 BLOCKFIRST 实现，LAYERFIRST 的 CPU 端多组传输在 worker 中有基本支持但未充分测试
