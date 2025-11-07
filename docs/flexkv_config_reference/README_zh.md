# FlexKV 配置使用指南

本指南详细说明如何配置和使用 FlexKV 的在线服务配置文件（`flexkv_config.json`），涵盖所有参数的含义、推荐值及典型使用场景。

---

## 基础配置选项

### 一、通过yml文件配置

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

### 服务器模式配置

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_SERVER_CLIENT_MODE` | bool | 0 | `server_client_mode`: 是否强制启用服务器-客户端模式 |
| `FLEXKV_SERVER_RECV_PORT` | str | "ipc:///tmp/flexkv_server" | `server_recv_port`: 服务器接收端口配置 |

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
| `FLEXKV_TRANSFER_SMS_H2D` | int | 8 | H2D 传输使用的流处理器数量，仅在`FLEXKV_USE_CE_TRANSFER_H2D`为0时生效 |
| `FLEXKV_TRANSFER_SMS_D2H` | int | 8 | D2H 传输使用的流处理器数量，仅在`FLEXKV_USE_CE_TRANSFER_D2H`为0时生效 |

---

### SSD I/O优化

> 注：`iouring_entries`设置为0即禁用iouring，不推荐设置为0。

| 环境变量 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `FLEXKV_MAX_FILE_SIZE_GB` | float | 32 | 单个 SSD 文件的最大大小，-1表示不限 |
| `FLEXKV_IORING_ENTRIES` | int | 512 | io_uring 队列深度，推荐设为 `512` 以提升并发 IO 性能 |
| `FLEXKV_IORING_FLAGS` | int | 0 | io_uring 标志位，默认为 0|



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
| `FLEXKV_EVICT_RATIO` | float | 0.05 | cpu，ssd一次evict主动淘汰比例（0.0 = 只淘汰最小的必要的block数）。建议保持 `0.05`，即每一次淘汰5%的最久未使用的block |
