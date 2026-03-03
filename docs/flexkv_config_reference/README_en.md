# FlexKV Configuration Guide

This guide provides detailed instructions on how to configure and use FlexKV's online service configuration file (`flexkv_config.json`), covering the meaning of all parameters, recommended values, and typical usage scenarios.

---

## Basic Configuration Options

### 1. Configuration via Config File

If the `FLEXKV_CONFIG_PATH` environment variable is set, the configuration file specified by this variable will be used with priority. Both yml and json file formats are supported.

Below is a recommended configuration example that enables both CPU and SSD cache layers:

YML configuration:
```yml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data/flexkv_ssd/
enable_gds: false
```
Or using JSON configuration:
```json
{
  "cpu_cache_gb": 32,
  "ssd_cache_gb": 1024,
  "ssd_cache_dir": "/data/flexkv_ssd/",
  "enable_gds": false
}
```
- `cpu_cache_gb`: CPU cache layer capacity in GB, must not exceed physical memory.
- `ssd_cache_gb`: SSD cache layer capacity in GB. Recommended to be greater than `cpu_cache_gb` and a multiple of `FLEXKV_MAX_FILE_SIZE_GB`. Set to 0 if only using CPU cache (SSD cache will not be enabled).
- `ssd_cache_dir`: Directory where SSD cache data is stored. If multiple SSDs are available, separate multiple mount paths with semicolons `;`. For example, `ssd_cache_dir: /data0/flexkv_ssd/;/data1/flexkv_ssd/` to improve bandwidth.
- `enable_gds`: Whether to enable GPU Direct Storage (GDS). If hardware and drivers support it, enabling this can improve SSD to GPU data throughput. Disabled by default.

---

### 2. Configuration via Environment Variables

If the `FLEXKV_CONFIG_PATH` environment variable is not set, configuration can be done through the following environment variables.

> Note: If `FLEXKV_CONFIG_PATH` is set, the configuration file specified by `FLEXKV_CONFIG_PATH` will take priority, and the following environment variables will be ignored.

| Environment Variable | Type | Default | Description |
|----------------------|------|---------|-------------|
| `FLEXKV_CPU_CACHE_GB` | int | 16 | CPU cache layer capacity in GB, must not exceed physical memory |
| `FLEXKV_SSD_CACHE_GB` | int | 0 | SSD cache layer capacity in GB. Recommended to be greater than `FLEXKV_CPU_CACHE_GB` and a multiple of `FLEXKV_MAX_FILE_SIZE_GB`. Set to 0 if only using CPU cache (SSD cache will not be enabled) |
| `FLEXKV_SSD_CACHE_DIR` | str | "./flexkv_ssd" | Directory where SSD cache data is stored. If multiple SSDs are available, separate multiple mount paths with semicolons `;`. For example, `"/data0/flexkv_ssd/;/data1/flexkv_ssd/"` to improve bandwidth |
| `FLEXKV_ENABLE_GDS` | bool | 0 | Whether to enable GPU Direct Storage (GDS). If hardware and drivers support it, enabling this can improve SSD to GPU data throughput. Disabled by default, set to 1 to enable |

---

## Advanced Configuration Options
Advanced configuration is mainly for users who need fine-tuned performance optimization or custom special requirements. It is recommended for users with some understanding of FlexKV.
All advanced configurations support configuration via environment variables or yml/json configuration files. In case of conflicts with multiple configuration levels, the final priority order is: **Configuration file > Environment variables > Built-in default parameters**.
If setting in a configuration file, remove the `FLEXKV_` prefix and convert everything to lowercase. For example, setting `server_client_mode: 1` in a yml file will override the value of the `FLEXKV_SERVER_CLIENT_MODE` environment variable.
Some configurations can only be set through environment variables.

### Enable/Disable FLEXKV

> Note: This configuration can only be set through environment variables

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `ENABLE_FLEXKV` | bool | 1 | 0-Disable FLEXKV, 1-Enable FLEXKV |



---

### Multi-Instance Mode Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_SERVER_CLIENT_MODE` | bool | 0 | `server_client_mode`: Whether to force enable server-client mode |
| `FLEXKV_SERVER_RECV_PORT` | str | "ipc:///tmp/flexkv_server" | `server_recv_port`: Server receive port configuration. Different instances in multi-instance mode should use the same port |
| `FLEXKV_INSTANCE_NUM` | int | 1 | Number of inference engine instances |
| `FLEXKV_INSTANCE_ID` | int | 0 | Inference engine instance ID |

---

### KV Cache Layout Types

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_CPU_LAYOUT` | str | BLOCKFIRST | CPU storage layout, options: `LAYERFIRST` and `BLOCKFIRST`, recommended to use `BLOCKFIRST` |
| `FLEXKV_SSD_LAYOUT` | str | BLOCKFIRST | SSD storage layout, options: `LAYERFIRST` and `BLOCKFIRST`, recommended to use `BLOCKFIRST` |
| `FLEXKV_REMOTE_LAYOUT` | str | BLOCKFIRST | REMOTE storage layout, options: `LAYERFIRST` and `BLOCKFIRST`, recommended to use `BLOCKFIRST` |
| `FLEXKV_GDS_LAYOUT` | str | BLOCKFIRST | GDS storage layout, options: `LAYERFIRST` and `BLOCKFIRST`, recommended to use `BLOCKFIRST` |

---

### CPU-GPU Transfer Optimization

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_USE_CE_TRANSFER_H2D` | bool | 0 | Whether to use cudaMemcpyAsync for Host→Device transfers. Can avoid occupying SM, but transfer speed will be reduced |
| `FLEXKV_USE_CE_TRANSFER_D2H` | bool | 0 | Whether to use cudaMemcpyAsync for Device→Host transfers. Can avoid occupying SM, but transfer speed will be reduced |
| `FLEXKV_TRANSFER_SMS_H2D` | int | 8 | Number of streaming multiprocessors used for H2D transfer, only effective when `FLEXKV_USE_CE_TRANSFER_H2D` is 0 |
| `FLEXKV_TRANSFER_SMS_D2H` | int | 8 | Number of streaming multiprocessors used for D2H transfer, only effective when `FLEXKV_USE_CE_TRANSFER_D2H` is 0 |

---

### SSD I/O Optimization

> Note: Setting `iouring_entries` to 0 disables iouring. Not recommended to set to 0.

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_MAX_FILE_SIZE_GB` | float | -1 | Maximum size of a single SSD file, -1 means unlimited |
| `FLEXKV_IOURING_ENTRIES` | int | 512 | io_uring queue depth. Recommended to set to `512` to improve concurrent I/O performance |
| `FLEXKV_IOURING_FLAGS` | int | 0 | io_uring flags, default is 0 |



---

### Multi-Node TP

> Note: These configurations can only be set through environment variables

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_MASTER_HOST` | str | "localhost" | Master node IP for multi-node TP |
| `FLEXKV_MASTER_PORTS` | str | "5556,5557,5558" | Master node ports for multi-node TP. Uses three ports, separated by commas |


---

### Logging Configuration

> Note: These configurations can only be set through environment variables

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_LOGGING_PREFIX` | str | "FLEXKV" | Logging prefix |
| `FLEXKV_LOG_LEVEL` | str | "INFO" | Log output level, options: "DEBUG" "INFO" "WARNING" "ERROR" "CRITICAL" "OFF" |
| `FLEXKV_NUM_LOG_INTERVAL_REQUESTS` | int | 200 | Log output interval request count |



---

### Tracing and Debugging

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_ENABLE_TRACE` | bool | 0 | Whether to enable performance tracing. Recommended to disable (`0`) in production to reduce overhead |
| `FLEXKV_TRACE_FILE_PATH` | str | "./flexkv_trace.log" | Trace log file path |
| `FLEXKV_TRACE_MAX_FILE_SIZE_MB` | int | 100 | Maximum size (MB) per trace log file |
| `FLEXKV_TRACE_MAX_FILES` | int | 5 | Maximum number of trace log files to retain |
| `FLEXKV_TRACE_FLUSH_INTERVAL_MS` | int | 1000 | Trace log flush interval (milliseconds) |


---

### Control Plane Optimization

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_INDEX_ACCEL` | bool | 1 | 0-Enable Python version RadixTree implementation, 1-Enable C++ version RadixTree implementation |
| `FLEXKV_EVICTION_POLICY` | str | "lru" | Cache eviction policy, options: "lru", "lfu", "fifo", "mru", and "filo". "lru" means Least Recently Used, "lfu" means Least Frequently Used, "fifo" means First In First Out, "mru" means Most Recently Used, "filo" means First In Last Out |
| `FLEXKV_EVICT_RATIO` | float | 0.05 | CPU and SSD eviction ratio for proactive eviction per cycle (0.0 = only evict the minimal necessary blocks). Recommended to keep at `0.05`, i.e., evict 5% of least recently used blocks per cycle |
| `FLEXKV_EVICT_START_THRESHOLD` | float | 0.7 | Memory utilization threshold to trigger proactive eviction. When the cache utilization reaches this ratio, FlexKV starts evicting nodes proactively. For example, `0.7` means eviction begins when 70% of the cache is occupied. Set to `1.0` to only evict when the cache is full |
| `FLEXKV_HIT_REWARD_SECONDS` | int | 0 | Number of bonus seconds added to a node's effective access time on each cache hit, enhancing LRU with frequency awareness. When set to `0` (default), standard LRU behavior applies. When set to a positive value, frequently hit nodes accumulate extra protection time, making them harder to evict. See [Eviction Policy Guide](../eviction_policy/README_en.md) for details |
