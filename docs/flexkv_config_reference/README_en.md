# FlexKV Configuration Guide

This guide explains how to configure and use the FlexKV online serving configuration file (`flexkv_config.json`), including the meaning of all parameters, recommended values, and typical usage scenarios.

---

## Recommended Configuration

Below is a production-grade recommended configuration that balances performance and stability:

```json
{
    "enable_flexkv": true,
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
        "enable_cpu": true,
        "enable_ssd": true,
        "enable_remote": false,
        "enable_gds": false,
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
- `num_cpu_blocks` and `num_ssd_blocks` represent the total number of blocks in CPU memory and SSD respectively. These values must be configured according to your machine specs and model size. See [Cache Capacity Configuration](#cache-capacity-config) for calculation details.
- `ssd_cache_dir` specifies the directory where SSD-stored KV cache files are saved.

---

## Configuration File Structure Overview

The FlexKV configuration file is a JSON file, primarily consisting of three parts:

- `enable_flexkv`: Whether to enable FlexKV (must be set to `true` to take effect).
- `server_recv_port`: The IPC port on which the FlexKV service listens.
- `cache_config`: The core cache configuration object, containing all cache behavior parameters.
- `num_log_interval_requests`: Log statistics interval (outputs performance log every N requests).

---

## Complete `cache_config` Parameter Reference (from [`flexkv/common/config.py`](../../flexkv/common/config.py))

### Basic Configuration

| Parameter Name | Type | Default | Description |
|----------------|------|---------|-------------|
| `tokens_per_block` | int | 16 | Number of tokens per KV block. Must match the `block_size` used in the acceleration framework (e.g., vLLM). |
| `enable_cpu` | bool | true | Whether to enable CPU memory as a cache layer. Strongly recommended to enable. |
| `enable_ssd` | bool | false | Whether to enable SSD as a cache layer. Recommended if NVMe SSD is available. |
| `enable_remote` | bool | false | Whether to enable remote cache (e.g., scalable cloud storage). Requires remote cache engine and custom implementation. |
| `enable_gds` | bool | false | Whether to use GPU Direct Storage (GDS) to accelerate SSD I/O. Not currently supported. |
| `index_accel` | bool | false | Whether to enable C++ RadixTree. Recommended to enable. |

---

### KV Cache Layout Types (Generally No Need to Modify)

| Parameter Name | Type | Default | Description |
|----------------|------|---------|-------------|
| `gpu_kv_layout_type` | enum | LAYERWISE | Organization of KV cache on GPU (layer-wise or block-wise). Must match vLLM’s layout (currently `LAYERWISE`). |
| `cpu_kv_layout_type` | enum | BLOCKWISE | Organization on CPU. Recommended to use `BLOCKWISE`. Does not need to match vLLM. |
| `ssd_kv_layout_type` | enum | BLOCKWISE | Organization on SSD. Recommended to use `BLOCKWISE`. Does not need to match vLLM. |
| `remote_kv_layout_type` | enum | BLOCKWISE | Organization for remote cache. Must be defined according to remote backend’s layout. |

> Note: Do not modify layout types unless you have specific performance requirements.

---

### Cache Capacity Configuration <a id="cache-capacity-config"></a>

| Parameter Name | Type | Default | Description |
|----------------|------|---------|-------------|
| `num_cpu_blocks` | int | 1000000 | Number of blocks allocated in CPU memory. Adjust based on available RAM. |
| `num_ssd_blocks` | int | 10000000 | Number of blocks allocated on SSD. |
| `num_remote_blocks` | int \| None | None | Number of blocks allocated in remote cache. |

> Note: Block size in all cache levels (CPU/SSD/Remote) matches the GPU block size. Estimate cache capacities based on GPU KV cache memory usage and block count.

> Note: `block_size = num_layer * _kv_dim * tokens_per_block * num_head * head_size * dtype_size`.

---

### CPU-GPU Transfer Optimization

| Parameter Name | Type | Default | Description |
|----------------|------|---------|-------------|
| `use_ce_transfer_h2d` | bool | false | Whether to use CUDA Copy Engine for Host→Device transfers. Reduces SM usage but may slightly reduce bandwidth. Real-world difference is minimal. |
| `use_ce_transfer_d2h` | bool | false | Whether to use CUDA Copy Engine for Device→Host transfers. |
| `transfer_sms_h2d` | int | 8 | Number of SMs (Streaming Multiprocessors) allocated for H2D transfers. |
| `transfer_sms_d2h` | int | 8 | Number of SMs allocated for D2H transfers. |

---

### SSD Cache Configuration

| Parameter Name | Type | Default | Description |
|----------------|------|---------|-------------|
| `max_blocks_per_file` | int | 32000 | Maximum number of blocks per SSD file. `-1` means unlimited. |
| `ssd_cache_dir` | str \| List[str] | None | **Required.** Path to SSD cache directory, e.g., `"/data/flexkv_ssd/"`. |
| `ssd_cache_iouring_entries` | int | 0 | io_uring queue depth. Recommended: `512` for significantly improved concurrent I/O performance. |
| `ssd_cache_iouring_flags` | int | 0 | io_uring flags. Recommended: `1`.|

> Note: To maximize bandwidth across multiple SSDs, bind each SSD to a separate directory and specify them as a list:  
> `"ssd_cache_dir": ["/data0/flexkv_ssd/", "/data1/flexkv_ssd/"]`.  
> KV blocks will be evenly distributed across all SSDs.

> Note: Setting `ssd_cache_iouring_entries` to `0` disables io_uring. Not recommended.

---

### Remote Cache Configuration (Skip if not enabled)

| Parameter Name | Type | Default | Description |
|----------------|------|---------|-------------|
| `remote_cache_size_mode` | str | "file_size" | Allocate remote cache space by file size or block count. |
| `remote_file_size` | int \| None | None | Size (in bytes) of each remote file. |
| `remote_file_num` | int \| None | None | Number of remote files. |
| `remote_file_prefix` | str \| None | None | Prefix for remote file names. |
| `remote_cache_path` | str \| List[str] | None | Remote cache path (e.g., Redis URL, S3 path). |
| `remote_config_custom` | dict \| None | None | Custom remote cache configurations (e.g., timeout, authentication). |

---

### Tracing and Logging

| Parameter Name | Type | Default | Description |
|----------------|------|---------|-------------|
| `enable_trace` | bool | true | Whether to enable performance tracing. Disable (`false`) in production to reduce overhead. |
| `trace_file_path` | str | "./flexkv_trace.log" | Path to trace log file. |
| `trace_max_file_size_mb` | int | 100 | Maximum size (MB) per trace log file. |
| `trace_max_files` | int | 5 | Maximum number of trace log files to retain. |
| `trace_flush_interval_ms` | int | 1000 | Trace log flush interval (milliseconds). |

---

### Cache Eviction Policy

| Parameter Name | Type | Default | Description |
|----------------|------|---------|-------------|
| `evict_ratio` | float | 0.0 | Ratio of blocks to proactively evict from CPU/SSD per eviction cycle. `0.0` = evict only the minimal necessary blocks (more eviction cycles may impact performance). Recommended: `0.05` (evict 5% of least recently used blocks per cycle). |
