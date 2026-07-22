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
- `swa_multi_group`: DeepSeek-V4 SWA sidecar switch. When omitted or set to `true`, SWA KV is stored and restored together with the attention/indexer compress states. Set it explicitly to `false` to keep SWA KV I/O while skipping state registration and I/O.
- `swa_multi_layer`: Controls whether layerwise restore fuses SWA/state H2D into the main layerwise worker. It defaults to `true`; set it to `false` to use the standalone SWA/state H2D predecessor path.

To switch to SWA-only mode, add the following explicit setting:

```yml
swa_multi_group: false
```

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
| `FLEXKV_SWA_MULTI_GROUP` | bool | unset (auto-enabled) | For DeepSeek-V4, unset or `1` stores/restores SWA KV with attention/indexer compress states; `0` keeps SWA KV I/O only |
| `FLEXKV_SWA_MULTI_LAYER` | bool | 1 | `1` fuses SWA/state H2D into layerwise restore; `0` uses the standalone SWA/state H2D predecessor worker |

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
| `FLEXKV_TRANSFER_NUM_CTA_H2D` | int | 4 | Number of CUDA thread blocks (CTAs) used for H2D transfer, only effective when `FLEXKV_USE_CE_TRANSFER_H2D` is 0 |
| `FLEXKV_TRANSFER_NUM_CTA_D2H` | int | 4 | Number of CUDA thread blocks (CTAs) used for D2H transfer, only effective when `FLEXKV_USE_CE_TRANSFER_D2H` is 0 |
| `FLEXKV_CE_PATH_OPT` | bool | 1 | Enable the CE optimized five-path adaptive strategy. `0` = baseline (`PER_BLOCK` per-block memcpy, no optimization); `1` = optimized paths (`CONTIG_DIRECT` / `SEGMENT_DIRECT` / `SEGMENT_SCATTER` / `GATHER_SCATTER` / `GATHER_DIRECT` auto-select, **default**). Only effective when `FLEXKV_USE_CE_TRANSFER_H2D` or `FLEXKV_USE_CE_TRANSFER_D2H` is 1 |
| `FLEXKV_CE_SEGMENT_THRESHOLD` | int | 8 | Segment count threshold for `SEGMENT_DIRECT` vs `GATHER_SCATTER` selection (only when `FLEXKV_CE_PATH_OPT=1`). **Applies to LAYERFIRST scenarios** (non-contiguous segments ≤ threshold → `SEGMENT_DIRECT`, otherwise → `GATHER_SCATTER`). BLOCKFIRST scenarios use `GATHER_DIRECT`, whose internal copy switches between a contiguous full-block copy and a per-block copy by batch granularity; the per-block variant can take the `cudaMemcpy2DAsync` fast path gated by `FLEXKV_ENABLE_CE_MEMCPY2D`. The threshold also selects the `cudaMemcpy2DAsync` branch in `SEGMENT_SCATTER` / `GATHER_SCATTER` |
| `FLEXKV_MLA_D2H_MODE` | str | "sharded" | **Only applicable to MLA scenarios** (kv_heads=1, all TP ranks have identical KV). Controls how CPU KV Cache is written during D2H transfer. Available options:<br/>• `rank_rotate` - Rotates the designated rank that writes the complete KV per request, avoiding sustained single-rank overload that causes thermal throttling; best with CE<br/>• `sharded` - **Default**. Each GPU writes 1/N shard to form one complete KV (requires `chunk_size % num_gpus == 0`)<br/>• `all_write` - Each GPU writes complete KV to its own location (N× CPU memory)<br/>• `rank0_only` - Only rank 0 writes complete KV<br/>• `layer_parallel` - Writes in parallel by layer |
| `FLEXKV_ENABLE_CE_MEMCPY2D` | bool | 1 | Controls the `cudaMemcpy2DAsync` strided fast path in `SEGMENT_SCATTER` / `GATHER_SCATTER` / `GATHER_DIRECT` (effective for both D2H and H2D). `1` = **Default**: uses `cudaMemcpy2DAsync` for the strided GPU↔CPU step, which is fast on NVIDIA but slow or unsupported on other platforms; `0` = portable staging-buffer + CPU scatter/gather path for those platforms. Only effective when CE transfer is enabled and one of those three paths is selected; `CONTIG_DIRECT` / `SEGMENT_DIRECT` are unaffected |
| `FLEXKV_CE_GATHER_THREADS` | int | 4 | Thread count for the parallel CPU gather/scatter thread pool used in `SEGMENT_SCATTER` / `GATHER_SCATTER` paths (BLOCKFIRST scattered layouts). Unset = `4` (parallel ON, 4 threads); `0` = disable parallel (single-thread memcpy baseline); `N>0` = parallel ON with N threads (including the calling thread). The thread pool is `thread_local` — each GPU dispatch thread gets its own pool with zero cross-GPU contention. LAYERFIRST layouts merge consecutive blocks into a single large copy and do not use the thread pool |
| `FLEXKV_CE_GATHER_NT` | bool | 1 | Controls non-temporal (NT) stores (AVX-512 / AVX2 streaming store) on the write-once staging / KV-pool destination in the CPU scatter/gather paths. `1` = **Default**: use NT stores to avoid RFO traffic and cache pollution; `0` = fall back to `std::memcpy`. On non-x86 platforms, NT stores are unavailable and `memcpy` is always used regardless of this setting |
| `FLEXKV_LAYERWISE_NOTIFY_MODE` | str | "hostfunc" | Notification mechanism used by layerwise H2D transfer to signal the inference engine after each layer batch completes:<br/>• `hostfunc` - Default. Uses `cudaLaunchHostFunc` to enqueue a host callback on the CUDA stream; fires precisely when the GPU finishes the batch<br/>• `polling` - A background thread polls per-batch CUDA events via `cudaEventQuery` and writes the eventfd as soon as all GPUs complete each batch. Avoids the cross-thread scheduling overhead of `cudaLaunchHostFunc`; benchmarked faster than `hostfunc` on NVIDIA 8x GPU (e.g. cuda engine: small 0.60→0.40ms 1.50x, medium 4.39→1.87ms 2.35x, large 16.79→16.62ms 1.01x). Costs one busy-polling CPU thread per `LayerwiseTransferGroup` instance |

---

### CUDA MPS (Multi-Process Service)

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_ENABLE_MPS` | bool | 1 | Whether to automatically manage CUDA MPS startup and shutdown. Set to 0 to disable |

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
