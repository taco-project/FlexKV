# Physical CP Layer Split

> 适用：SGLang 0.5.10 NSA / HY4 一类「按层归属」的 Context Parallel 缓存布局。  
> 开关：框架侧 `layer_shard_enabled`（及对应的 NSA cache layer-split 启动项）；FlexKV 侧不另开总开关。  
> `CP=1` 或未启用 layer split 时，本路径不改变 block 编号与 H2D 语义。

## 要解决什么问题

普通 TP/CP 下，每个 rank 持有同一套层的完整 KV 视图（或通过 TP 切头）。  
Physical CP layer split 则是：

- 全局仍有 `cp_size` 个逻辑 owner；
- **每一层 KV 只由其中一个 CP rank 拥有**；
- 非 owner 在该层只有 receive scratch（空行 / remote buffer），不能当作本地可读缓存。

跨节点时（例如两机 CP16），一个节点只能看到本机那一段 owner。若仍按「全局层数 × 全局 holder」去编 Host/L3 block，会出现：

1. 本机 GPU handle 对不上全局 `layer_groups`；
2. 前缀命中后，有的节点缺层，H2D 没法原子恢复；
3. Indexer / NEXTN 与 Full KV 的归属不一致，容易出现半块可用。

FlexKV 的做法是：**控制面继续用全局层号做完成与 fence；数据面在 GPU 注册完成后，把模型和 handle 表收成节点本地布局。**

## 核心概念

| 概念 | 含义 |
|------|------|
| `h2d_layer_holders[i]` | 全局原层 `i` 的 owner，取值在 `[0, cp_size)` |
| `layer_shard_enabled` | 启用 physical owner-only 传输；非 owner 层走 scratch，不写本地 KV |
| `cp_cache_node_rank` | 模型已被 `cp_node_cache_model()` 收成「本节点视角」后的标记 |
| 全局层号 | layerwise / eventfd / milestone 仍按原层编号；远端空 member 视为已满足 |
| 节点本地 `layer_groups` | 只保留本节点 owner 落在本地 CP 段内的层，用于 CPU/SSD/GPU handle 切片 |

约束（跨节点 physical CP）：

- `PP = DP = attention-TP = instance = 1`
- `cp_size` 能被 `nnodes` 整除
- `layer_groups` 与 `h2d_layer_holders` 完整
- 每个 group 在本地至少有一层；否则拒绝（避免 Host 空洞）

## 数据流

```text
SGLang NSA pool
  layer_shard_enabled / layer_shard_rank / _get_layer_owner_rank
        │
        ▼
connector._configure_h2d_holders()
  写出全局 h2d_layer_holders + layer_shard_enabled
        │
        ▼
GPU 注册完成
        │
        ▼
cp_node_cache_model(model, node_rank)
  按节点切 layer_groups / holders（非本地 owner → -1）
        │
        ├─► slice_cp_gpu_groups / slice_cp_storage_handle
        │     只保留本节点层对应的 GPU tensor handle
        │
        └─► GPUCPUTransferWorker(h2d_layer_holders, layer_shard_enabled)
              owner 层：H2D/D2H 打真实 KV
              非 owner：不写本地池；完成契约仍按全局层号推进
```

Lookup / store / prefetch 的 token→block 逻辑不变。变的是「一个 block 在本节点上对应哪些层的物理字节」。

## 与 Indexer / NEXTN 的关系

- **Target Indexer**：physical CP 下每个原层仍要有 Indexer buffer；非 owner 使用 remote Indexer scratch。`FLEXKV_DEDUP_INDEXER_GROUP` **尚不支持** DSA cache layer split，二者不要同时开。
- **Draft / NEXTN**：通常是复制的一层池，不走 layer split。connector 把 draft KV + Indexer 打成额外 `LayerGroupSpec`，挂在 **每个 CP owner 的代表 target 层** 上，与 target 同一 Host block，避免第二套 KVManager。
- **Namespace**：target/draft 几何指纹（含 CP holder 布局）会进 cache namespace，避免不同切层方案共用 L3 对象。这与 `RadixKey.extra_key` 转发是正交能力。

## 代码落点

| 模块 | 作用 |
|------|------|
| `flexkv/common/cp_shard.py` | 节点本地化模型与 GPU/Storage handle 切片 |
| `flexkv/common/config.py` | `h2d_layer_holders` / `layer_shard_enabled` / `cp_cache_node_rank`；跨节点时调用 `cp_node_cache_model` |
| `flexkv/transfer/workers/gpu_cpu.py` | owner-only H2D/D2H；校验 holders 长度与范围 |
| `flexkv/integration/sglang/connector.py` | `_configure_h2d_holders`、`_resolve_cp_kv_buffers`、注册前 preflight / geometry vote |

相关测试：`tests/test_rebase_cp_shard.py`、`tests/test_rebase_cp_worker.py`、`tests/test_rebase_cp_connector.py`。

## 验收要点

- `CP=1` 或未开 layer split：与 upstream main 行为一致。
- 单机 physical CP：每个 rank 只恢复自己拥有的层；非 owner 层不污染本地 KV。
- 两机 CP（`cp_size % nnodes == 0`）：前缀命中后，各节点都能恢复自己那一段层；不允许「半节点有层、半节点空洞」的 block 发布。
- layerwise 完成仍按**全局层号**通知；空 remote member 不得卡住 forward。
- 与 GDR/CE 正交：layer split 只改「哪层有本地字节」，不替代传输 backend 选择。

## 非目标

- 不在本 feature 里实现跨节点 remote-buffer DMA 补洞。
- 不恢复已删除的 `csrc/layerwise.cpp` worker 类。
- 不替代 `flexkv/prefetch/` 的 chunked prefetch；L3 Host window 是另一条 opt-in 路径。
