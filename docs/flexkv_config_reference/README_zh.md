# FlexKV 配置使用指南

本指南详细说明如何配置和使用 FlexKV 的在线服务配置文件（`flexkv_config.json`），涵盖所有参数的含义、推荐值及典型使用场景。

---

## 基础配置选项

### 一、通过文件配置

如果设置了环境变量 `FLEXKV_CONFIG_PATH`，将优先使用该变量指定的配置文件。支持yml和json两种文件类型。

以下是一个同时开启 CPU 和 SSD 缓存层的推荐配置示例：

yml配置：
```yml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data/flexkv_ssd/
enable_gds: false
```
或使用json配置：
```json
{
  "cpu_cache_gb": 32,
  "ssd_cache_gb": 1024,
  "ssd_cache_dir": "/data/flexkv_ssd/",
  "enable_gds": false
}
```
- `cpu_cache_gb`：CPU 缓存层容量，单位为 GB，不能超过物理内存。
- `ssd_cache_gb`：SSD 缓存层容量，单位为 GB。建议大于 `cpu_cache_gb`并为`FLEXKV_MAX_FILE_SIZE_GB`的整数倍，若仅用CPU缓存则设为 0（此时不启用 SSD 缓存）。
- `ssd_cache_dir`：SSD 缓存数据的存放目录。若有多块 SSD，可通过分号 `;` 分隔多个挂载路径。例如 `ssd_cache_dir: /data0/flexkv_ssd/;/data1/flexkv_ssd/`，以提升带宽。
- `enable_gds`：是否启用 GPU Direct Storage（GDS）。如硬件和驱动支持，开启后可提升 SSD 到 GPU 的数据吞吐能力。默认关闭。
- `swa_multi_group`：DeepSeek-V4 SWA sidecar 开关。未配置或设为 `true` 时，SWA KV 会与 attention/indexer compress state 一起存取；显式设为 `false` 时保留 SWA KV 存取，但不注册或存取 state。
- `swa_multi_layer`：控制 layerwise restore 是否把 SWA/state H2D 融合进主 layerwise worker。默认为 `true`；设为 `false` 时使用独立的 SWA/state H2D 前置 worker。

如需切换到 SWA-only，可在配置文件中显式添加：

```yml
swa_multi_group: false
```

---

### 二、通过环境变量配置

如果未设置 `FLEXKV_CONFIG_PATH`环境变量，则可通过以下环境变量进行配置。

> 注：如果设置了`FLEXKV_CONFIG_PATH`，将优先使用`FLEXKV_CONFIG_PATH`指定的配置文件，以下环境变量将被忽略。

| 环境变量             | 类型  | 默认值        | 说明                                                                                                            |
|----------------------|-------|-------------|----------------------------------------------------------------------------------------------------------------|
| `FLEXKV_CPU_CACHE_GB`    | int   | 16          | CPU 缓存层容量，单位为 GB，不能超过物理内存
| `FLEXKV_SSD_CACHE_GB`    | int   | 0           | SSD 缓存层容量，单位为 GB。建议设置大于 `FLEXKV_CPU_CACHE_GB`并为`FLEXKV_MAX_FILE_SIZE_GB`的整数倍，若仅用CPU缓存则设为 0（此时不启用 SSD 缓存）               |
| `FLEXKV_SSD_CACHE_DIR`   | str   | "./flexkv_ssd" | SSD 缓存数据的存放目录。若有多块 SSD，可通过分号 `;` 分隔多个挂载路径。例如 `"/data0/flexkv_ssd/;/data1/flexkv_ssd/"`，以提升带宽                  |
| `FLEXKV_ENABLE_GDS`      | bool  | 0           | 是否启用 GPU Direct Storage（GDS）。如硬件和驱动支持，开启后可提升 SSD 到 GPU 的数据吞吐能力。默认关闭，开启请设为 1                    |
| `FLEXKV_USE_HUGEPAGE_CPU_BUFFER` | bool | 0 | 是否为通用 CPU KV cache 启用 HugePage。默认关闭，开启请设为 1 |
| `FLEXKV_USE_HUGEPAGE_TMP_BUFFER` | bool | 0 | 是否为 `enable_p2p_ssd` 场景下的 tmp CPU staging buffer 启用 HugePage。默认关闭，开启请设为 1 |
| `FLEXKV_HUGEPAGE_SIZE_BYTES` | int | 2097152 | HugePage 大小，默认 2 MiB。如果宿主机准备的是 1 GiB HugePage，可设为 `1073741824` |
| `FLEXKV_SWA_MULTI_GROUP` | bool | 未设置（自动开启） | DeepSeek-V4 下未设置或设为 `1` 时，SWA KV 与 attention/indexer compress state 一起存取；设为 `0` 时只保留 SWA KV 存取 |
| `FLEXKV_SWA_MULTI_LAYER` | bool | 1 | `1` 表示把 SWA/state H2D 融合进 layerwise restore；`0` 表示使用独立的 SWA/state H2D 前置 worker |

---

## 高级配置选项
高级配置主要针对需要精细化性能优化或自定义特殊需求的用户，建议对 FlexKV 具备一定理解的用户使用。
所有高级配置均支持通过环境变量或 yml/json 配置文件进行设置，如有多级配置冲突，最终生效顺序为：**配置文件 > 环境变量 > 默认内置参数**。
如果在配置文件中设置，请去除`FLEXKV_`前缀并全部转换为小写，例如在yml文件中设置`server_client_mode: 1`将会覆盖`FLEXKV_SERVER_CLIENT_MODE`环境变量的值。
部分配置只能通过环境变量设置。

### 启用/禁用FLEXKV

> 注：该配置只能通过环境变量设置

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ENABLE_FLEXKV` | bool | 1 | 0-禁用FLEXKV，1-启用FLEXKV |

---

### 多实例模式配置

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_SERVER_CLIENT_MODE` | bool | 0 | `server_client_mode`: 是否强制启用服务器-客户端模式 |
| `FLEXKV_SERVER_RECV_PORT` | str | "ipc:///tmp/flexkv_server" | `server_recv_port`: 服务器接收端口配置，多实例模式下不同实例应当使用相同的端口 |
| `FLEXKV_INSTANCE_NUM` | int | 1 | 推理引擎实例的数量 |
| `FLEXKV_INSTANCE_ID` | int | 0 | 推理引擎实例ID |

---

### KV 缓存布局类型

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_CPU_LAYOUT` | str | BLOCKFIRST | CPU 存储布局，可选`LAYERFIRST`和`BLOCKFIRST`, 推荐使用`BLOCKFIRST` |
| `FLEXKV_SSD_LAYOUT` | str | BLOCKFIRST | SSD 存储布局，可选`LAYERFIRST`和`BLOCKFIRST`, 推荐使用`BLOCKFIRST` |
| `FLEXKV_REMOTE_LAYOUT` | str | BLOCKFIRST | REMOTE 存储布局，可选`LAYERFIRST`和`BLOCKFIRST`, 推荐使用`BLOCKFIRST` |
| `FLEXKV_GDS_LAYOUT` | str | BLOCKFIRST | GDS 存储布局，可选`LAYERFIRST`和`BLOCKFIRST`, 推荐使用`BLOCKFIRST` |

---

### CPU-GPU 传输优化

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_USE_CE_TRANSFER_H2D` | bool | 0 | 是否使用 cudaMemcpyAsync 实现 Host→Device 传输，可以避免占用 SM，但是传输速度会降低 |
| `FLEXKV_USE_CE_TRANSFER_D2H` | bool | 0 |  是否使用 cudaMemcpyAsync 实现 Device→Host 传输，可以避免占用 SM，但是传输速度会降低 |
| `FLEXKV_TRANSFER_NUM_CTA_H2D` | int | 4 | H2D 传输使用的 CUDA thread block (CTA) 数量，仅在`FLEXKV_USE_CE_TRANSFER_H2D`为0时生效 |
| `FLEXKV_TRANSFER_NUM_CTA_D2H` | int | 4 | D2H 传输使用的 CUDA thread block (CTA) 数量，仅在`FLEXKV_USE_CE_TRANSFER_D2H`为0时生效 |
| `FLEXKV_CE_PATH_OPT` | bool | 1 | 是否启用 CE 优化路径（五路自适应策略）。`0` = baseline（`PER_BLOCK` 逐块 memcpy，无优化）；`1` = 优化路径（`CONTIG_DIRECT` / `SEGMENT_DIRECT` / `SEGMENT_SCATTER` / `GATHER_SCATTER` / `GATHER_DIRECT` 自动选择，**默认**）。仅在 `FLEXKV_USE_CE_TRANSFER_H2D` 或 `FLEXKV_USE_CE_TRANSFER_D2H` 为 1 时生效 |
| `FLEXKV_CE_SEGMENT_THRESHOLD` | int | 8 | `SEGMENT_DIRECT` 与 `GATHER_SCATTER` 的段数切换阈值（仅 `FLEXKV_CE_PATH_OPT=1` 时生效）。**对 LAYERFIRST 场景生效**（非连续段数 ≤ 阈值时走 `SEGMENT_DIRECT`，否则走 `GATHER_SCATTER`）。BLOCKFIRST 场景走 `GATHER_DIRECT`，其内部分别在「整块直传」与「逐块拷贝」两种实现间切换（取决于是否一次性传输整层），其中逐块实现可启用 `cudaMemcpy2DAsync` 快路径（受 `FLEXKV_ENABLE_CE_MEMCPY2D` 控制）；`segment_threshold` 还用于 `SEGMENT_SCATTER` / `GATHER_SCATTER` 的 `cudaMemcpy2DAsync` 分支选择 |
| `FLEXKV_MLA_D2H_MODE` | str | "sharded" | **仅适用于 MLA 场景**（kv_heads=1，所有 TP rank 的 KV 相同）。控制 D2H 时 CPU KV Cache 的写入模式。可选值：<br/>• `rank_rotate` - 跨请求轮换 designated rank 写完整 KV，避免单 rank 持续满载导致热节流，配合 CE 效果最佳<br/>• `sharded` - **默认**。每个 GPU 写 1/N 分片拼成一个完整 KV（需要 `chunk_size % num_gpus == 0`）<br/>• `all_write` - 每个 GPU 写完整 KV 到各自位置（CPU 内存占用 N×）<br/>• `rank0_only` - 仅 rank 0 写完整 KV<br/>• `layer_parallel` - 按 layer 并行写入 |
| `FLEXKV_ENABLE_CE_MEMCPY2D` | bool | 1 | 控制 `SEGMENT_SCATTER` / `GATHER_SCATTER` / `GATHER_DIRECT` 中的 `cudaMemcpy2DAsync` strided 快路径（D2H 与 H2D 双向生效）。`1` = **默认**：使用 `cudaMemcpy2DAsync` 做 strided GPU↔CPU 直传，在 NVIDIA 上快，但在其他平台上慢或不受支持；`0` = 可移植的 staging buffer + CPU scatter/gather 路径，用于这些平台。仅在 CE 传输开启且选择上述三条路径之一时生效；`CONTIG_DIRECT` / `SEGMENT_DIRECT` 不受影响 |
| `FLEXKV_CE_GATHER_THREADS` | int | 4 | CPU 并行 gather/scatter 线程池的线程数，用于 `SEGMENT_SCATTER` / `GATHER_SCATTER` 路径（BLOCKFIRST 散列布局）。未设置 = `4`（开启并行，默认 4 线程）；`0` = 关闭并行（单线程 memcpy 基线）；`N>0` = 开启并行，使用 N 个线程（含调用线程）。线程池为 `thread_local`——每个 GPU 调度线程拥有独立的线程池，无跨 GPU 竞争。LAYERFIRST 布局会将连续 block 合并为一次大拷贝，不使用线程池 |
| `FLEXKV_CE_GATHER_NT` | bool | 1 | 控制 CPU scatter/gather 路径中对 staging buffer / KV pool 目标地址的非时间（NT）存储（AVX-512 / AVX2 streaming store）。`1` = **默认**：使用 NT 存储以避免 RFO 流量和缓存污染；`0` = 回退到 `std::memcpy`。非 x86 平台不支持 NT 存储，无论此设置如何均使用 `memcpy` |
| `FLEXKV_LAYERWISE_NOTIFY_MODE` | str | "hostfunc" | layerwise H2D 传输在每个 layer batch 完成后通知推理引擎的机制：<br/>• `hostfunc` - 默认。使用 `cudaLaunchHostFunc` 在 CUDA stream 上排队一个 host callback，GPU 完成该 batch 时精确触发<br/>• `polling` - 后台线程通过 `cudaEventQuery` 轮询每 batch 的 CUDA event，所有 GPU 完成该 batch 后立即写 eventfd。避免 `cudaLaunchHostFunc` 的跨线程调度开销；在 NVIDIA 8x GPU 上实测快于 `hostfunc`（cuda 引擎：small 0.60→0.40ms 1.50x，medium 4.39→1.87ms 2.35x，large 16.79→16.62ms 1.01x）。代价是每个 `LayerwiseTransferGroup` 实例占用一个 busy-polling CPU 线程 |

---

### CUDA MPS（Multi-Process Service）

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_ENABLE_MPS` | bool | 1 | 是否自动管理 CUDA MPS 的启停。设为 0 可禁用 |

---

### SSD I/O优化

> 注：`iouring_entries`设置为0即禁用iouring，不推荐设置为0。

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_MAX_FILE_SIZE_GB` | float | -1 | 单个 SSD 文件的最大大小，-1表示不限 |
| `FLEXKV_IOURING_ENTRIES` | int | 512 | io_uring 队列深度，推荐设为 `512` 以提升并发 IO 性能 |
| `FLEXKV_IOURING_FLAGS` | int | 0 | io_uring 标志位，默认为 0|



---

### 多节点TP

> 注：这些配置只能通过环境变量设置

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_MASTER_HOST` | str | "localhost" | 多节点TP的主节点IP |
| `FLEXKV_MASTER_PORTS` | str | "5556,5557,5558" | 多节点TP的主节点端口。使用三个端口，用逗号分隔 |

---

### 日志配置

> 注：这些配置只能通过环境变量设置

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_LOGGING_PREFIX` | str | "FLEXKV" | 日志前缀 |
| `FLEXKV_LOG_LEVEL` | str | "INFO" | 日志输出等级，可选："DEBUG"  "INFO" "WARNING"  "ERROR"  "CRITICAL" "OFF" |
| `FLEXKV_NUM_LOG_INTERVAL_REQUESTS` | int | 200 | 日志输出间隔请求数 |

---

### 追踪和调试

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_ENABLE_TRACE` | bool | 0 | 是否启用性能追踪。生产环境建议关闭（`0`）以减少开销 |
| `FLEXKV_TRACE_FILE_PATH` | str | "./flexkv_trace.log" | 追踪日志路径 |
| `FLEXKV_TRACE_MAX_FILE_SIZE_MB` | int | 100 | 单个追踪文件最大大小（MB） |
| `FLEXKV_TRACE_MAX_FILES` | int | 5 | 最多保留的追踪文件数 |
| `FLEXKV_TRACE_FLUSH_INTERVAL_MS` | int | 1000 | 追踪日志刷新间隔（毫秒） |


---

### 控制面优化

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_INDEX_ACCEL` | bool | 1 | 0-启用Python版本RadixTree实现，1-启用C++版本RadixTree实现 |
| `FLEXKV_EVICTION_POLICY` | str | "lru" | 缓存淘汰策略，可选 "lru"、"lfu"、"fifo"、"mru" 和 "filo"。 "lru" 表示最近最少使用，"lfu" 表示最不经常使用，"fifo" 表示先进先出，"mru" 表示最近最多使用，"filo" 表示先进后出 |
| `FLEXKV_EVICT_RATIO` | float | 0.05 | cpu，ssd一次evict主动淘汰比例（0.0 = 只淘汰最小的必要的block数）。建议保持 `0.05`，即每一次淘汰5%的最久未使用的block |
| `FLEXKV_EVICT_START_THRESHOLD` | float | 0.7 | 触发主动淘汰的内存利用率阈值。当缓存利用率达到该比例时，FlexKV 开始主动淘汰节点。例如 `0.7` 表示缓存占用达到 70% 时即开始淘汰。设为 `1.0` 则仅在缓存满时才淘汰 |
| `FLEXKV_HIT_REWARD_SECONDS` | int | 0 | 每次缓存命中时向节点的有效访问时间叠加的额外秒数，为 LRU 增加频率感知能力。设为 `0`（默认值）时为标准 LRU 行为。设为正数时，频繁命中的节点会累积额外的保护时间，使其更难被驱逐。详见[驱逐策略指南](../eviction_policy/README_zh.md) |
