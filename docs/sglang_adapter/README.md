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

The adapter supports two operating modes:

- **Local mode** (default): Single-node isolated cache, no cross-node communication
- **Distributed mode**: Multi-node KV Cache sharing via Distributed RadixTree + Redis GMS.
  Cross-node blocks are fetched via P2P transfer (Mooncake Transfer Engine).

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
  CPU cache engine radix tree match (local + remote in distributed mode)
    -> if remote: prefetch_async + wait (P2P transfer to local CPU cache)
    -> read FlexKV CPU cache tensor block
    -> layout transform: FlexKV (L,2,T,H,D) -> SGLang (2,L,T,H,D)
    -> mem_pool_host.set_from_flat_data_page(index, data)

EXISTS (batch_exists - query):
  CPU cache engine radix tree match (local + remote in distributed mode)
    -> count consecutive ready pages from start
```

### Layout Transform

SGLang `layer_first` layout per page: `(2, num_layers, page_size, num_kv_heads, head_size)`
FlexKV `BLOCKFIRST` layout per block: `(num_layers, 2, tokens_per_block, num_kv_heads, head_size)`

Transform is a single `permute(1,0,2,3,4).contiguous()` in both directions.
Other SGLang layouts (`page_first`, `page_first_direct`, `page_head`) are also supported.

### Distributed Mode Architecture

In distributed mode, each Prefill node runs its own FlexKV instance with a
Distributed RadixTree that syncs metadata via Redis GMS:

```
Local Mode:
  Prefill A (node 1)              Prefill B (node 2)
       |                               |
  FlexKV cache                    FlexKV cache
  (isolated)                      (isolated)
       X No coordination X

Distributed Mode:
  Prefill A (node 1)                    Prefill B (node 2)
       |                                     |
  FlexKV cache (local)                  FlexKV cache (local)
       |                                     |
  Distributed RadixTree <--Redis GMS--> Distributed RadixTree
  (local tree + global index)        (local tree + global index)
```

**Metadata sharing flow:**

1. **PUT** (Prefill A generates KV Cache):
   - `batch_set_v1()` calls `KVManager.put_cpu()`
   - Block inserted to local RadixTree, metadata published to Redis GMS
   - Other nodes' Distributed RadixTree syncs via Redis

2. **EXISTS** (Prefill B checks prefix):
   - Queries `HierarchyLRCacheEngine.match_all()` which includes distributed index
   - Returns whether prefix exists locally or on any peer

3. **GET** (Prefill B fetches KV Cache):
   - Queries distributed index via `match_all()`
   - Local match: fetches from local CPU cache
   - Remote match: fetches via P2P transfer (`prefetch_async` + `wait`) into local CPU cache, then reads locally

### Cross-Node GET (Distributed Mode)

When `batch_get_v1` discovers blocks on a remote node via `match_all()`:

```
batch_get_v1 called with token_ids
  -> match_all() queries local + remote radix trees
  -> matched_pos == "remote": blocks exist on another Prefill node
  -> prefetch_async(token_ids): KVManager creates PEERH2H transfer ops
     -> TransferEngine resolves remote node address via Redis
     -> Mooncake batch_transfer_sync_read() pulls blocks over P2P (RDMA/TCP)
     -> Blocks land in local CPU cache tensor
  -> wait(task_id): blocks until transfer completes
  -> Re-query local tree with match() to get local block IDs
  -> Read from local CPU cache tensor (same as local path)
  -> Layout transform + write to SGLang Host memory
```

If P2P transfer fails, the adapter returns False (cache miss) and SGLang
proceeds with normal prefill computation.

## Files

### FlexKV Side

| File | Purpose |
|------|---------|
| `flexkv/integration/sglang/__init__.py` | Package marker |
| `flexkv/integration/sglang/hicache_storage_adapter.py` | Main adapter: `FlexKVHiCacheStorage` class |
| `flexkv/integration/sglang/test_hicache_storage_adapter.py` | Unit tests (15 test cases) |

### SGLang Side (separate repo/branch)

| File | Changes |
|------|---------|
| `sglang/srt/managers/cache_controller.py` | Add `"flexkv"` to zero-copy whitelist; pass `token_ids` via `extra_info` in 3 places (`_page_backup`, `_page_transfer`, `_storage_hit_query`) |
| `sglang/srt/mem_cache/storage/backend_factory.py` | Register `"flexkv"` backend pointing to `FlexKVHiCacheStorage` |

## Configuration

### Local Mode (Default)

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

### Distributed Mode

```bash
python3 -m sglang.launch_server \
    --model <model_path> \
    --enable-hierarchical-cache \
    --page-size 16 \
    --hicache-storage-backend flexkv \
    --hicache-storage-backend-extra-config '{
      "mode": "distributed",
      "redis_host": "<redis-server-ip>",
      "redis_port": 6379,
      "redis_password": "optional_password",
      "enable_cpu": true,
      "enable_ssd": false,
      "num_cpu_blocks": 100000
    }' \
    --disaggregation-mode prefill
```

Distributed mode also requires `MOONCAKE_CONFIG_PATH` environment variable pointing to a
Mooncake Transfer Engine config file:

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

### Key Parameters

| Parameter | Where | Description |
|-----------|-------|-------------|
| `--page-size 16` | SGLang CLI | Tokens per page. Must be >= 2 (FlexKV C++ radix tree limitation). |
| `--hicache-ratio` | SGLang CLI | Host cache size as ratio of GPU cache (default 2.0) |
| `--mem-fraction-static` | SGLang CLI | GPU memory fraction for KV cache (default 0.863) |
| `num_cpu_blocks` | extra_config | CPU cache block count. Memory = num_cpu_blocks * block_size_bytes |
| `enable_cpu` | extra_config | Must be `true` |
| `enable_ssd` | extra_config | Enable SSD persistence via io_uring (default `false`) |
| `mode` | extra_config | `"local"` (default) or `"distributed"` |
| `redis_host` | extra_config | Redis server address (distributed mode, default `127.0.0.1`) |
| `redis_port` | extra_config | Redis server port (distributed mode, default `6379`) |
| `redis_password` | extra_config | Redis password (distributed mode, optional) |
| `prefetch_timeout` | extra_config | Timeout in seconds for cross-node P2P prefetch (distributed mode, default `5.0`) |
| `num_layers` | extra_config | (Optional) Model layer count -- auto-detected from SGLang |
| `num_kv_heads` | extra_config | (Optional) KV head count -- auto-detected from SGLang |
| `head_size` | extra_config | (Optional) Head dimension -- auto-detected from SGLang |

### Build for Development

```bash
# Debug mode (skip Cython, source changes take effect on restart)
FLEXKV_DEBUG=1 pip install -e .

# Release mode
pip install -e . --no-build-isolation

# Release mode with P2P support (required for distributed mode)
FLEXKV_ENABLE_P2P=1 pip install -e . --no-build-isolation
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

3. **No dynamic mode switching**: Mode is set at adapter init time. Restart required to change.

4. **`clear()` only resets stats**: The adapter's `clear()` method resets internal statistics but
   does **not** evict cached blocks. FlexKV's KVManager currently has no bulk-clear API — the
   underlying C++ RadixTree index and block allocator lack a reset/clear operation. Cached data
   persists until the process exits or blocks are naturally evicted.

5. **Synchronous remote prefetch**: In distributed mode, `batch_get_v1` performs cross-node block
   fetching synchronously — it calls `prefetch_async()` followed by a blocking `wait()` before
   proceeding. This simplifies the initial implementation (no need to manage async state or
   coordinate partial readiness across a batch of blocks) but means the caller is blocked for the
   entire P2P transfer duration. A future optimization could pipeline prefetch with layout
   transform / host memory writes for already-local blocks, or expose a fully async GET interface
   so SGLang's cache controller can overlap prefetch with other work.

## Verification Checklist

- [x] FlexKV backend registered in SGLang StorageBackendFactory
- [x] Adapter initializes with deferred KVManager (created in register_mem_pool_host)
- [x] Model params auto-detected from SGLang mem_pool_host
- [x] page_size auto-adopted from SGLang mem_pool_host
- [x] batch_set_v1: writes KV data to FlexKV CPU cache via put_cpu
- [x] batch_exists: queries CPU cache engine radix tree for cached pages (local + remote in distributed mode)
- [x] batch_get_v1: reads data back with layout transform (fetches remote blocks via P2P in distributed mode)
- [x] Deduplication: second set of same tokens returns True (no re-allocation)
- [x] MLA non-rank-0: backup skipped
- [x] Round-trip data correctness: set -> get produces identical data
- [x] SGLang E2E: backup/prefetch cycle verified (816/832 tokens cached)
- [x] SGLang E2E: GSM8K accuracy consistent across cache flushes
- [x] Unit tests: 15 test cases pass (including 7 mode configuration tests)
- [x] Distributed mode: single-node with Redis GMS verified
- [x] Distributed mode: cross-node GET via P2P transfer (prefetch_async + wait)
