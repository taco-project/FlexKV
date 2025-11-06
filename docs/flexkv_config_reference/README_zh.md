# FlexKV 配置使用指南

本指南详细说明如何配置和使用 FlexKV 的在线服务配置文件（`flexkv_config.json`），涵盖所有参数的含义、推荐值及典型使用场景。

---

## 配置方案示例

### 一、通过yml文件配置

如果设置了环境变量 `FLEXKV_CONFIG_FILE`，将优先使用该变量指定的配置文件。

以下是一个同时开启 CPU 和 SSD 缓存层的推荐配置示例：

```yml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data/flexkv_ssd/
enable_gds: false
```
- `cpu_cache_gb`：CPU 缓存层容量，单位为 GB，不能超过物理内存。
- `ssd_cache_gb`：SSD 缓存层容量，单位为 GB。建议设置大于 `cpu_cache_gb`，如仅用CPU缓存则设为 0（此时不启用 SSD 缓存）。
- `ssd_cache_dir`：SSD 缓存数据的存放目录。若有多块 SSD，可通过分号 `;` 分隔多个挂载路径。例如 `ssd_cache_dir: /data0/flexkv_ssd/;/data1/flexkv_ssd/`，以提升带宽。
- `enable_gds`：是否启用 GPU Direct Storage（GDS）。如硬件和驱动支持，开启后可提升 SSD 到 GPU 的数据吞吐能力。默认关闭。

---

### 二、通过环境变量配置

如果未指定 `FLEXKV_CONFIG_FILE`，则可通过以下环境变量初始化：

| 环境变量             | 类型  | 默认值        | 说明                                                                                                            |
|----------------------|-------|-------------|----------------------------------------------------------------------------------------------------------------|
| `FLEXKV_CPU_CACHE_GB`    | int   | 16          | CPU 缓存层容量，单位为 GB，不能超过物理内存。
| `FLEXKV_SSD_CACHE_GB`    | int   | 0           | SSD 缓存层容量，单位为 GB。建议设置大于 `FLEXKV_CPU_CACHE_GB`，如仅用CPU缓存则设为 0（此时不启用 SSD 缓存）。               |
| `FLEXKV_SSD_CACHE_DIR`   | str   | "./flexkv_ssd" | SSD 缓存数据的存放目录。若有多块 SSD，可通过分号 `;` 分隔多个挂载路径。例如 `"/data0/flexkv_ssd/;/data1/flexkv_ssd/"`，以提升带宽。                  |
| `FLEXKV_ENABLE_GDS`      | bool  | 0           | 是否启用 GPU Direct Storage（GDS）。如硬件和驱动支持，开启后可提升 SSD 到 GPU 的数据吞吐能力。默认关闭，开启请设为 1。                    |

---

## 其他细粒度配置（来自 [`flexkv/common/config.py`](../../flexkv/common/config.py)）

### 服务器模式配置

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_SERVER_CLIENT_MODE` | bool | 0 | `server_client_mode`: 是否强制启用服务器-客户端模式 |
| `FLEXKV_SERVER_RECV_PORT` | str | "ipc:///tmp/flexkv_server" | `server_recv_port`: 服务器接收端口配置 |

---

### KV 缓存布局类型

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_CPU_LAYOUT` | str | BLOCKWISE | CPU 存储布局，可选`LAYERWISE`和`BLOCKWISE`, 推荐使用`BLOCKWISE` |
| `FLEXKV_SSD_LAYOUT` | str | BLOCKWISE | SSD 存储布局，可选`LAYERWISE`和`BLOCKWISE`, 推荐使用`BLOCKWISE` |
| `FLEXKV_REMOTE_LAYOUT` | str | BLOCKWISE | REMOTE 存储布局，可选`LAYERWISE`和`BLOCKWISE`, 推荐使用`BLOCKWISE` |
| `FLEXKV_GDS_LAYOUT` | str | BLOCKWISE | GDS 存储布局，可选`LAYERWISE`和`BLOCKWISE`, 推荐使用`BLOCKWISE`` |

> 注：除非有特殊性能需求，否则不建议修改布局类型。

---

### CPU-GPU 传输优化

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_USE_CE_TRANSFER_H2D` | bool | 0 | 是否使用 cudaMemcpyAsync 实现 Host→Device 传输，可以避免占用 SM，但是传输速度会降低 |
| `FLEXKV_USE_CE_TRANSFER_D2H` | bool | 0 |  是否使用 cudaMemcpyAsync 实现 Device→Host 传输，可以避免占用 SM，但是传输速度会降低 |
| `FLEXKV_TRANSFER_SMS_H2D` | int | 8 | `transfer_sms_h2d`: H2D 传输使用的流处理器数量，仅在`FLEXKV_USE_CE_TRANSFER_H2D`为0时生效 |
| `FLEXKV_TRANSFER_SMS_D2H` | int | 8 | `transfer_sms_d2h`: D2H 传输使用的流处理器数量，仅在`FLEXKV_USE_CE_TRANSFER_D2H`为0时生效 |

---

### SSD 相关配置

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_MAX_BLOCKS_PER_FILE` | int | 32000 | `max_blocks_per_file`: 单个 SSD 文件最多包含的 block 数。-1 表示无限制 |
| `FLEXKV_SSD_CACHE_IORING_ENTRIES` | int | 512 | `ssd_cache_iouring_entries`: io_uring 队列深度，推荐设为 `512` 以提升并发 IO 性能，实测比不使用iouring提升较大 |
| `FLEXKV_SSD_CACHE_IORING_FLAGS` | int | 1 | `ssd_cache_iouring_flags`: io_uring 标志位，推荐设置为 1。|

> 注：`ssd_cache_iouring_entries`设置为0即不使用iouring，不推荐设置为0

---

### 追踪与日志

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_ENABLE_TRACE` | bool | 0 (false) | `enable_trace`: 是否启用性能追踪。生产环境建议关闭（`false`）以减少开销 |
| `FLEXKV_TRACE_FILE_PATH` | str | "./flexkv_trace.log" | `trace_file_path`: 追踪日志路径 |
| `FLEXKV_TRACE_MAX_FILE_SIZE_MB` | int | 100 | `trace_max_file_size_mb`: 单个追踪文件最大大小（MB） |
| `FLEXKV_TRACE_MAX_FILES` | int | 5 | `trace_max_files`: 最多保留的追踪文件数 |
| `FLEXKV_TRACE_FLUSH_INTERVAL_MS` | int | 1000 | `trace_flush_interval_ms`: 追踪日志刷新间隔（毫秒） |
| `FLEXKV_NUM_LOG_INTERVAL_REQUESTS` | int | 200 | `num_log_interval_requests`: 日志输出间隔请求数 |

---

### 缓存淘汰策略

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_EVICT_RATIO` | float | 0.05 | `evict_ratio`: cpu，ssd一次evict主动淘汰比例（0.0 = 只淘汰最小的必要的block数量，较多的淘汰次数会影响性能）。建议保持 `0.05`，即每一次淘汰5%的最久未使用的block |
