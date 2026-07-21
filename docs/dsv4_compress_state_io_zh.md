# DeepSeek-V4 compress state 在 FlexKV 中的存取设计

## 1. 背景与目标

DeepSeek-V4 的 DSA attention 除了读取 C4 KV 和 C4 indexer KV，还会读取两组运行时 score：

- C4 attention state：`compress_state_pools[i].kv_score_buffer.kv_score`
- C4 indexer state：`indexer_compress_state_pools[i].kv_score_buffer.kv_score`

原来的 FlexKV 只存取 KV buffer。请求 restore 时，KV 会从 Host 恢复到 GPU，但上述 state 仍可能保留旧值或空值，导致 DSA 使用“正确 KV + 错误 score”。

本实现补齐这两组 state 的完整数据通路：

```text
GPU
  ├─ C4 KV
  ├─ C4 indexer KV
  ├─ SWA KV
  ├─ C4 attention state   ─┐
  └─ C4 indexer state     ─┴─ SWA multi-group sidecar
                             │
                             ▼
                   CPU / SSD / Remote
```

需要特别说明：当前实现做的是 state 快照的 offload/restore，不是在 restore 后重算 state。它补齐了之前缺失的 I/O；恢复后的 state 是否满足新一轮 decode 的语义，仍需通过 CP8 精度实验最终确认。

## 2. 为什么 state 复用 SWA page，而不是 main-KV page

这两组 compress state 的 GPU 地址由 SWA physical page 和各自的 `ring_size` 决定，不应使用 full-KV slot mapping。

SGLang 已经可以通过 `translate_loc_from_full_to_swa` 把 full-KV token slot 转换成 SWA token slot。FlexKV 因此把 state 作为 SWA page 的 sidecar：

```text
同一个逻辑 SWA page id
  ├─ SWA KV bytes
  ├─ attention state bytes
  └─ indexer state bytes
```

这样 PUT 和 GET 都只需要一份 `swa_slot_mapping`，SWA KV 与两组 state 始终使用相同的 source/destination page id，不会和 main-KV slot 错位。

## 3. SGLang 侧：发现并注册 state

代码入口：

```text
python/sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py
```

### 3.1 发现 state tensor

`FlexKVConnector` 从 DSv4 KV cache 中读取 C4 layer 对应的两个 pool：

```python
kvcache.compress_state_pools[i].kv_score_buffer.kv_score
kvcache.indexer_compress_state_pools[i].kv_score_buffer.kv_score
```

每组 state 在注册前会检查：

- 所有 C4 layer 的 `ring_size` 一致；
- state tensor 是 contiguous 2D tensor；
- 同组各层的 shape 和 dtype 一致；
- `swa_page_size == FlexKV page_size`；
- `swa_page_size` 能被 `ring_size` 整除；
- state pool 至少能覆盖 SWA pool 的全部 physical page。

缺失或不完整的 pool 不会被静默当成有效 state 注册；不满足布局约束时会直接报错，避免只恢复部分 state。

### 3.2 构造三个 SWA layer group

SWA 通道注册为 heterogeneous multi-group：

| Group | GPU 数据 | 每个 FlexKV page 搬运的行数 | `compress_ratio` |
|---|---|---:|---:|
| SWA KV | `swa_kv_pool` | `swa_page_size` | 1 |
| C4 attention state | attention `kv_score` | `ring_size` | `swa_page_size / ring_size` |
| C4 indexer state | indexer `kv_score` | `ring_size` | `swa_page_size / ring_size` |

每个 state group 保留自身真实的 `dtype` 和 `head_size`。例如 page size 为 256、ring size 为 8 时，state 的 `compress_ratio` 为 32，一个 SWA page 只搬运对应的 8 行 score。

Connector 最终向 FlexKV 传递三份一一对应的注册信息：

```text
swa_layer_groups
swa_gpu_layouts
swa_handles_per_group
```

同时设置：

```python
cache_config.swa.multi_group = True
```

这个标志还用于选择正确的 layerwise GET 时序，见第 6 节。

### 3.3 用户配置开关

用户配置项为 `swa_multi_group`：

| 配置 | 行为 |
|---|---|
| 不配置 | 默认启用：SWA KV + attention/indexer state |
| `swa_multi_group: true` | 显式启用：SWA KV + attention/indexer state |
| `swa_multi_group: false` | SWA-only：保留 SWA KV，不注册、不存取 state |

也可以在未使用配置文件时通过环境变量 `FLEXKV_SWA_MULTI_GROUP=0/1` 设置。用户配置和内部运行标记是分开的：只有 state groups 实际构造成功后，Connector 才会设置 `cache_config.swa.multi_group = True`；SWA-only 路径始终保持 `False`。

### 3.4 生成 page mapping

PUT 和 GET 都调用 `_build_swa_slot_mapping`：

1. 输入 main-KV 的 token slot；
2. 调用 `translate_loc_from_full_to_swa` 得到 SWA token slot；
3. 去掉未映射的 slot-0 sentinel 前缀；
4. 把 CPU `int64` mapping 传给 `kv_manager.launch(..., swa_slot_mappings=...)`；
5. FlexKV 按 page size fold 成 SWA physical page id。

state 没有单独再生成 mapping；它和 SWA KV 共享这份 physical page mapping。

## 4. FlexKV 注册协议

FlexKV 在原有 `swa_handles` 和 `swa_layout` 之外增加了可选字段：

```python
swa_layer_groups
swa_gpu_layouts
swa_handles_per_group
```

三个字段必须同时出现，而且 group 数量必须一致。Client 把各组 CUDA tensor 转成 `TensorSharedHandle` 后，通过 `RegisterTPClientRequest` 交给 TransferManager。

TransferManager 会：

- 保存每个 device 的 per-group handle 和 layout；
- 检查不同 GPU 注册的 `swa_layer_groups` 完全一致；
- 按 `WorkerKey` 重新组织 TP/CP device；
- 把 group metadata 和 handles 传给 `StorageEngine`、`TransferEngine`。

没有提供这些字段时，FlexKV 继续使用原来的 uniform SWA 路径，其他模型不受影响。

## 5. CPU、SSD 和 Remote 的数据布局

### 5.1 BLOCKFIRST byte-flat block

SWA multi-group 要求：

```text
FLEXKV_CPU_LAYOUT=BLOCKFIRST
```

原因是三组数据的 dtype 和单页 shape 不同，不能再用一个统一 tensor dtype 表示。Host 侧每个 physical page 被表示成一个 byte-flat block：

```text
block N
  [SWA KV group][attention state group][indexer state group]
```

每个 block 的字节数为：

```text
bytes_per_block = tp_size × Σ(
    num_layers[g]
    × kv_dim
    × (page_size / compress_ratio[g])
    × num_kv_heads[g]
    × head_size[g]
    × dtype_size[g]
)
```

当前 SWA/state 通道按 MLA byte layout 处理，`kv_dim = 1`。`KVCacheLayout.kv_shape` 为：

```text
[num_blocks, bytes_per_block]
```

第二维已经是字节数，不能再乘 Host tensor 的 `dtype.itemsize`。

### 5.2 GPU 与 CPU 之间

GPU 上每个 group 保留原生 tensor、dtype 和 layout。`GPUCPUTransferWorker` 或 `tpGPUCPUTransferWorker` 根据 group metadata 计算：

- GPU block/layer stride；
- 每个 group 在 Host block 中的 byte offset；
- `page_size / compress_ratio` 对应的实际搬运行数；
- TP 场景下每个 device 的 group handles。

D2H 时，各组数据写入同一个 Host block 的不同 byte region；H2D 时执行完全相反的拷贝。

### 5.3 CPU 与 SSD/Remote 之间

CPU、SSD 和 Remote 使用完全相同的 BLOCKFIRST byte-flat layout，因此下层存储不再解析 group：

- CPU ↔ SSD：每个 page 作为一个 opaque byte block 整块读写；
- CPU ↔ Remote：每个 page 作为一个 MLA byte block 整块读写；
- GDS GPU ↔ SSD：使用相同的 per-group GPU metadata 和 byte-flat SSD layout。

整块 I/O 还能避免高压缩 state group 的单个 chunk 小于 4 KiB 时产生碎片化读写。

## 6. PUT：state 如何 offload

一次 PUT 的时序如下：

```text
SGLang start_store_kv
  │
  ├─ put_match(token_ids)              分配 FlexKV destination page
  ├─ full-KV slot_mapping              绑定 main-KV GPU page
  ├─ translate full slot -> SWA slot
  └─ launch(..., swa_slot_mappings)
       │
       ├─ main KV D2H
       └─ is_swa=True D2H
            ├─ SWA KV       ─┐
            ├─ attn state    ├─ 写入同一个 Host page block
            └─ index state  ─┘
                         │
                         └─ 可继续 H2DISK / H2REMOTE
```

只对 `put_match` 返回的 unmatched 完整 page 发起搬运。SWA KV 和 state 使用同一个 SWA page id，因此它们在 CPU/SSD/Remote 上也作为同一条 cache entry 的组成部分一起淘汰、命中和迁移。

## 7. GET：state 如何 restore

一次 GET 的时序如下：

```text
FlexKV get_match
  │
  ├─ main KV source pages
  └─ SWA/state source pages
          │
SGLang start_load_kv
  ├─ full-KV destination slot_mapping
  ├─ SWA destination slot_mapping
  └─ launch(...)
       │
       ├─ DISK/Remote -> Host（如需要）
       ├─ SWA/state multi-group H2D
       │    ├─ SWA KV
       │    ├─ attn state
       │    └─ index state
       └─ main KV H2D
```

H2D worker 按注册时的 group byte offset，从一个 Host page block 中分别恢复 SWA KV、attention state 和 indexer state。

### 7.1 Layerwise GET 的依赖

Layerwise 路径把 main-KV 与 SWA/state 全部 fuse 进同一个 `LayerwiseTransferOp`：

```text
LayerwiseTransferOp
  ├─ main-KV DISK2H（如需要，opaque multi-group block）
  ├─ SWA/state DISK2H（如需要，opaque multi-group block）
  └─ per original layer:
       ├─ main-KV H2D（各 layer group members）
       ├─ SWA KV / attn state / indexer state H2D（各 SWA group members）
       └─ per-layer ready eventfd
```

`swa.multi_group=True` 时，SWA 池使用与 main-KV 同构的 `GroupParams + layer_members`；C4 层会等该层 state 写完再发 eventfd，无 state 的层只等 SWA KV + main。

`swa_multi_layer` 控制上述融合路径，默认开启。显式设为 `false`（或通过环境变量设置 `FLEXKV_SWA_MULTI_LAYER=0`）时，main-KV 仍使用 layerwise restore，但 SWA/state 改由独立 H2D worker 搬运；主 layerwise op 会依赖该 worker 完成，因此不会提前暴露未恢复的 state。该开关不改变 `swa_multi_group` 是否注册 compress-state sidecar 的语义。

非 layerwise GET 会等待整个 FlexKV task 完成后再把请求交还给 SGLang。

## 8. 关键约束与失败方式

| 约束 | 原因/行为 |
|---|---|
| 必须有 SWA pool 和 SWA config | state 复用 SWA physical page id |
| `swa_page_size == FlexKV page_size` | 保证逻辑 page 一一对应 |
| `page_size % ring_size == 0` | `compress_ratio` 必须是整数 |
| CPU layout 必须为 BLOCKFIRST | heterogeneous dtype 需要 byte-flat block |
| 三个 multi-group 注册字段同时出现 | 防止 metadata 与 handle 不完整 |
| 各 GPU 的 group spec 一致 | TP/CP worker 必须使用相同 byte layout |
| restore 测试必须关闭 `FLEXKV_P3_CLEAR` | restore 后清零会覆盖刚恢复的 state |
| `swa_multi_group: false` | 只存取 SWA KV，不注册或搬运 compress state |

当前实现不会自动重算 state，也不会在 state pool 缺失时伪造零值。

## 9. 验证建议

### 9.1 注册和布局

启动日志应出现：

```text
[FlexKV-DSv4-State] Prepared SWA-page state sidecars
[FlexKV-DSv4-State] Registered group 'c4_attention_state'
[FlexKV-DSv4-State] Registered group 'c4_indexer_state'
```

并确认 CPU pool 使用 BLOCKFIRST，Host `bytes_per_block` 等于三组数据逐组计算后的字节数之和。

### 9.2 数据正确性

建议分别对两组 `kv_score` 做 page-level checksum：

1. R1 PUT 前 GPU state；
2. D2H 后 CPU block 中对应 state region；
3. R2 H2D 后 GPU state。

三处应字节一致。只比较全局 sum-hash 可以发现明显 stale/空值，但 page-level hash 更容易定位 page mapping 或 group offset 错误。

### 9.3 时序正确性

FlexKV 单测 `tests/test_swa_state_sidecars.py` 覆盖：

- SWA KV 和两组 state 是否打包进同一个 byte-flat Host block；
- layerwise main-KV 是否等待 heterogeneous SWA/state H2D 完成。

最终仍需运行 CP8 的 R1 fresh → flush → R2 restore 精度复现，确认 R2 答案与 fresh 一致。

## 10. 代码索引

| 仓库 | 文件 | 职责 |
|---|---|---|
| SGLang | `python/sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py` | 发现 state、构造 group、注册 tensor、生成 SWA slot mapping |
| FlexKV | `flexkv/server/client.py` | 接收并序列化 per-group CUDA handles |
| FlexKV | `flexkv/server/request.py` | multi-group 注册协议字段 |
| FlexKV | `flexkv/transfer_manager.py` | 汇总各 device group metadata/handles |
| FlexKV | `flexkv/storage/storage_engine.py` | 分配 CPU/SSD/Remote byte-flat SWA pool |
| FlexKV | `flexkv/common/storage.py` | 计算 heterogeneous block layout 和 group stride |
| FlexKV | `flexkv/transfer/transfer_engine.py` | 创建 SWA/state 专用 multi-group worker |
| FlexKV | `flexkv/transfer/worker.py` | GPU/CPU、SSD、Remote 实际搬运 |
| FlexKV | `flexkv/common/transfer.py` | layerwise transfer graph 依赖 |
| FlexKV | `flexkv/kvtask.py` | 根据 `swa.multi_group` 选择 layerwise 路径 |
| FlexKV | `tests/test_swa_state_sidecars.py` | block layout 和 restore 时序单测 |
