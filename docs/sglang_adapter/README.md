# FlexKV + SGLang Integration

## Overview

FlexKV integrates with SGLang v0.5.6+ as a HiCacheStorage backend, providing a third-level
storage tier beneath SGLang's GPU and Host memory caches.

```
SGLang 3-Level Cache:
  L1: GPU KV Cache          (managed by SGLang)
  L2: Host (CPU pinned) mem (managed by SGLang HiRadixCache)
  L3: FlexKV CPU cache      (managed by this adapter via KVManager)
```

## Architecture

### KVManager CPU-only PUT Path

The adapter uses FlexKV's KVManager in **thread mode** (`FLEXKV_CPU_ONLY=1`), which shares
the process address space and allows direct access to the CPU cache tensor. This avoids GPU
block registration (which is unavailable in the HiCacheStorage context) while retaining full
FlexKV capabilities: radix tree index, CPU cache, and optional SSD persistence via io_uring.

### Data Flow

```
PUT (batch_set_v1 - backup from Host to FlexKV):
  SGLang Host mem
    -> mem_pool_host.get_data_page(index, flat=False)
    -> layout transform: SGLang (2,L,T,H,D) -> FlexKV (L,2,T,H,D)
    -> direct memcpy to FlexKV CPU cache tensor (via put_cpu)
    -> KVManager radix tree insert
    -> TransferEngine H2DISK async (if SSD enabled)

GET (batch_get_v1 - prefetch from FlexKV to Host):
  CPU cache engine radix tree match
    -> read FlexKV CPU cache tensor block
    -> layout transform: FlexKV (L,2,T,H,D) -> SGLang (2,L,T,H,D)
    -> mem_pool_host.set_from_flat_data_page(index, data)

EXISTS (batch_exists - query):
  CPU cache engine radix tree match -> count consecutive ready pages from start
```

### Layout Transform

SGLang `layer_first` layout per page: `(2, num_layers, page_size, num_kv_heads, head_size)`
FlexKV `BLOCKFIRST` layout per block: `(num_layers, 2, tokens_per_block, num_kv_heads, head_size)`

Transform is a single `permute(1,0,2,3,4).contiguous()` in both directions.
Other SGLang layouts (`page_first`, `page_first_direct`, `page_head`) are also supported.

## Files

### FlexKV Side

| File | Purpose |
|------|---------|
| `flexkv/integration/sglang/__init__.py` | Package marker |
| `flexkv/integration/sglang/hicache_storage_adapter.py` | Main adapter: `FlexKVHiCacheStorage` class |
| `flexkv/integration/sglang/test_hicache_storage_adapter.py` | Unit tests (8 test cases) |

### SGLang Side (separate repo/branch)

| File | Changes |
|------|---------|
| `sglang/srt/managers/cache_controller.py` | Add `"flexkv"` to zero-copy whitelist; pass `token_ids` via `extra_info` in 3 places (`_page_backup`, `_page_transfer`, `_storage_hit_query`) |
| `sglang/srt/mem_cache/storage/backend_factory.py` | Register `"flexkv"` backend pointing to `FlexKVHiCacheStorage` |

## Configuration

### SGLang Server Launch

```bash
python3 -m sglang.launch_server \
    --model <model_path> \
    --enable-hierarchical-cache \
    --page-size 16 \
    --hicache-storage-backend flexkv \
    --hicache-storage-backend-extra-config '{
      "enable_cpu": true,
      "enable_ssd": false,
      "num_cpu_blocks": 100000
    }'
```

Model parameters (`num_layers`, `num_kv_heads`, `head_size`) are automatically detected from
SGLang's `mem_pool_host` at runtime. They can be explicitly overridden in `extra_config` if needed.

### Key Parameters

| Parameter | Where | Description |
|-----------|-------|-------------|
| `--page-size 16` | SGLang CLI | Tokens per page. Must be >= 2 (FlexKV C++ radix tree limitation). Adapter auto-adopts this value. |
| `--hicache-ratio` | SGLang CLI | Host cache size as ratio of GPU cache (default 2.0) |
| `--mem-fraction-static` | SGLang CLI | GPU memory fraction for KV cache (default 0.863) |
| `num_cpu_blocks` | extra_config | CPU cache block count. Memory = num_cpu_blocks * block_size_bytes |
| `enable_cpu` | extra_config | Must be `true` |
| `enable_ssd` | extra_config | Enable SSD persistence via io_uring (default `false`) |
| `num_layers` | extra_config | (Optional) Model layer count — auto-detected from SGLang |
| `num_kv_heads` | extra_config | (Optional) KV head count — auto-detected from SGLang |
| `head_size` | extra_config | (Optional) Head dimension — auto-detected from SGLang |

### Build for Development

```bash
# Debug mode (skip Cython, source changes take effect on restart)
FLEXKV_DEBUG=1 pip install -e .

# Release mode
pip install -e . --no-build-isolation
```

## Running Tests

```bash
# Unit tests (no GPU required, needs FlexKV + SGLang installed)
python3 -m flexkv.integration.sglang.test_hicache_storage_adapter

# SGLang E2E tests (needs GPU + model)
cd /path/to/sglang
pytest test/srt/hicache/test_hicache_storage_flexkv_backend.py -v -s
```

## Known Limitations

1. **tokens_per_block=1**: FlexKV's C++ CRadixTreeIndex triggers SIGFPE (integer division).
   Workaround: use `--page-size 16` (or any value >= 2).

2. **Two memcpy per page**: Data flows Host -> FlexKV CPU cache -> Host (read path). This is
   inherent to the HiCacheStorage interface design which separates Host memory from storage.

## Verification Checklist

- [x] FlexKV backend registered in SGLang StorageBackendFactory
- [x] Adapter initializes with deferred KVManager (created in register_mem_pool_host)
- [x] Model params auto-detected from SGLang mem_pool_host
- [x] page_size auto-adopted from SGLang mem_pool_host
- [x] batch_set_v1: writes KV data to FlexKV CPU cache via put_cpu
- [x] batch_exists: queries CPU cache engine radix tree for cached pages
- [x] batch_get_v1: reads data back with layout transform
- [x] Deduplication: second set of same tokens returns True (no re-allocation)
- [x] MLA non-rank-0: backup skipped
- [x] Round-trip data correctness: set -> get produces identical data
- [x] SGLang E2E: backup/prefetch cycle verified (816/832 tokens cached)
- [x] SGLang E2E: GSM8K accuracy consistent across cache flushes
