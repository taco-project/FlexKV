# FlexKV 配置使用指南

本指南详细说明如何配置和使用 FlexKV 的在线服务配置文件（`flexkv_config.json`），涵盖所有参数的含义、推荐值及典型使用场景。

---

## 推荐配置方案

以下是一个兼顾性能与稳定性的生产级推荐配置：

```json
{
    "enable_flexkv": true,
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": true,
        "enable_remote": false,
        "use_gds": false,
        "enable_trace": false,
        "ssd_cache_iouring_entries": 512,
        "tokens_per_block": 64,
        "num_cpu_blocks": 233000,
        "num_ssd_blocks": 4096000,
        "ssd_cache_dir": "/data/flexkv_ssd/",
        "evict_ratio": 0.05,
        "index_accel": true
    },
    "num_log_interval_requests": 2000
}
```
- 其中的`num_cpu_blocks`和`num_ssd_blocks`分别代表内存和SSD中block的总数量，需要根据实际机器配置和模型来配置，具体计算方式见下文[缓存容量配置](#cache-capacity-config)
- `ssd_cache_dir`为ssd中KVCache存放的文件目录

---

## 配置文件结构概览

FlexKV 的配置文件是一个 JSON 文件，主要包含三个部分：

- `enable_flexkv`: 是否启用 FlexKV 功能（必须设为 `true` 才生效）
- `server_recv_port`: FlexKV 服务监听的 IPC 端口
- `cache_config`: 核心缓存配置对象，包含所有缓存行为参数
- `num_log_interval_requests`: 日志统计间隔（每处理 N 个请求输出一次性能日志）

---

## cache_config完整参数详解（来自 [`flexkv/common/config.py`](../../flexkv/common/config.py)）

### 基础配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `tokens_per_block` | int | 16 | 每个 KV Block 包含的 token 数量。需要与加速框架（如vLLM）中`block_size`保持一致 |
| `enable_cpu` | bool | true | 是否启用 CPU 内存作为缓存层。强烈建议开启。 |
| `enable_ssd` | bool | false | 是否启用 SSD 作为缓存层。如配备 NVMe SSD，建议开启。 |
| `enable_remote` | bool | false | 是否启用远程缓存（如可扩展云存储等）。需要配合远程缓存和自定义的远程缓存引擎使用 |
| `use_gds` | bool | false | 是否使用 GPU Direct Storage（GDS）加速 SSD 读写。目前暂不支持。 |
| `index_accel` | bool | false | 是否启用C++ RadixTree。推荐开启。 |

---

### KV 缓存布局类型（一般无需修改）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `gpu_kv_layout_type` | enum | LAYERWISE | GPU 上 KV Cache 的组织方式（按层或按块）。目前vLLM在GPU组织方式为`LAYERWISE`，因此FlexKV的`gpu_kv_layout_type`须与vLLM保持一致 |
| `cpu_kv_layout_type` | enum | BLOCKWISE | CPU 上按块组织, 推荐使用`BLOCKWISE`，不需要与vLLM保持一致 |
| `ssd_kv_layout_type` | enum | BLOCKWISE | SSD 上按块组织, 推荐使用`BLOCKWISE`，不需要与vLLM保持一致 |
| `remote_kv_layout_type` | enum | BLOCKWISE | 远程缓存按块组织, 需要按照remote组织形式定义 |

> 注：除非有特殊性能需求，否则不建议修改布局类型。

---

### 缓存容量配置 <a id="cache-capacity-config"></a>

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `num_cpu_blocks` | int | 1000000 | CPU 缓存块数。根据内存大小调整。|
| `num_ssd_blocks` | int | 10000000 | SSD 缓存块数。|
| `num_remote_blocks` | int \| None | None | 远程缓存块数。|

> 注：FlexKV里的各级缓存的block大小与GPU中的block大小保持一致，可以参考GPU的KVCache显存大小与block数量估算各级缓存中的block数量。

> 注：block_size = num_layer * _kv_dim * tokens_per_block * num_head * self.head_size * torch_dtype.size()。

---

### CPU-GPU 传输优化

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_ce_transfer_h2d` | bool | false | 是否使用 cuda copy engine 优化 Host→Device 传输，使用CE可以减少GPU SM在传输上的使用，但是传输速度会降低，实际测试差距不大 |
| `use_ce_transfer_d2h` | bool | false | 是否使用 cuda copy engine 优化 Device→Host 传输 |
| `transfer_sms_h2d` | int | 8 | H2D 传输使用的流处理器数量 |
| `transfer_sms_d2h` | int | 8 | D2H 传输使用的流处理器数量 |

---

### SSD 缓存配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_blocks_per_file` | int | 32000 | 单个 SSD 文件最多包含的 block 数。-1 表示无限制 |
| `ssd_cache_dir` | str \| List[str] | None | SSD 缓存目录路径，**必须设置**，如 `"/data/flexkv_ssd/"` |
| `ssd_cache_iouring_entries` | int | 0 | io_uring 队列深度，推荐设为 `512` 以提升并发 IO 性能，实测比不使用iouring提升较大，推荐使用512 |
| `ssd_cache_iouring_flags` | int | 0 | io_uring 标志位，一般保持 0 |

> 注：为了充分利用多块SSD的带宽上限，可以将多块SSD绑定至不同目录，并使用如 `"ssd cache dir": ["/data0/flexkv_ssd/", "/data1/flexkv_ssd/"]`方式初始化，SSD KVCache会均匀分布在所有SSD中，充分利用多个SSD带宽。

> 注：`ssd_cache_iouring_entries`设置为0即不适用iouring，不推荐设置为0

---

### 远程缓存配置（不启用时无需配置）

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `remote_cache_size_mode` | str | "file_size" | 按文件大小或块数分配远程缓存空间 |
| `remote_file_size` | int \| None | None | 单个远程文件大小（字节） |
| `remote_file_num` | int \| None | None | 远程文件数量 |
| `remote_file_prefix` | str \| None | None | 远程文件名前缀 |
| `remote_cache_path` | str \| List[str] | None | 远程缓存路径（如 Redis URL、S3 路径等） |
| `remote_config_custom` | dict \| None | None | 自定义远程缓存配置（如超时、认证等） |

---

### 追踪与日志

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_trace` | bool | true | 是否启用性能追踪。生产环境建议关闭（`false`）以减少开销 |
| `trace_file_path` | str | "./flexkv_trace.log" | 追踪日志路径 |
| `trace_max_file_size_mb` | int | 100 | 单个追踪文件最大大小（MB） |
| `trace_max_files` | int | 5 | 最多保留的追踪文件数 |
| `trace_flush_interval_ms` | int | 1000 | 追踪日志刷新间隔（毫秒） |

---

### 缓存淘汰策略

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `evict_ratio` | float | 0.0 | cpu，ssd一次evict主动淘汰比例（0.0 = 只淘汰最小的必要的block数量，较多的淘汰次数会影响性能）。建议保持 `0.05`，即每一次淘汰5%的最久未使用的block |
