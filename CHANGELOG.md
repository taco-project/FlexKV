# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Feature

Universal:
- radixshmem mode now uses radixshmem's data plane: the CPU KV pool is the radix-server's SlotStore (one slot per block, attached by name in the TE and every transfer worker) and cross-node reuse is `RadixClient.pull_async` from the prefetch path (server-side RDMA READ), replacing FlexKV's own CPU allocation, `PEER2CPUTransferWorker`, mooncake wrapper and Redis address book on this path. The radix-server runs as a subprocess of the bootstrap DP (`FLEXKV_RADIX_SERVER_LAUNCH_MODE`). radixshmem mode is CPU-tier only (`ssd_cache_gb` must be 0). See `docs/radixshmem_integration.md` and `docs/radixshmem_cross_node.md`
- radixshmem mode is configured by one YAML (`FLEXKV_RADIXSHMEM_CONFIG_PATH`, `flexkv/common/radixshmem_config.py`): `cluster` / `data` / `index` / `server` sections pass through by key to radixshmem's `ClusterConfig` / `DataPlaneConfig` / `IndexConfig` / `RadixServerConfig` (validated against the installed dataclasses, geometry keys rejected), a `client` section holds the prefetch limits; `index.register_chunk_size` defaults to `4096 / tokens_per_block` blocks (one RHT registration chunk per 4096 tokens). The file is global: `cluster.cluster_id` is the only namespace (etcd keys and every shm / socket / TE channel name), node identity derives from `cluster.rpc_interface`. The `FLEXKV_RADIX_*` cluster variables and `FLEXKV_SHM_RADIX_ID` are gone; `FLEXKV_RADIX_NODE_NAME` / `FLEXKV_RADIX_RPC_ADDRESS` remain as per-node overrides for co-located nodes. Examples in `examples/radixshmem_configs/`, reference `docs/radixshmem/config_zh.md`.
- radixshmem planning moved out of `GlobalCacheEngine` into the subclass `RadixShmemCacheEngine` (`flexkv/cache/radix_shmem_planner.py`, selected by `KVTaskEngine` when `FLEXKV_ENABLE_RADIXSHMEM=1`). `GlobalCacheEngine` keeps two hooks only (`_prepare_request`, `_build_cpu_cache_engine`); its plan dataclasses, `TransferPlanHandle` and completion callbacks are back to their non-radixshmem shape. The subclass's handles now roll back a plan cancelled before launch (match pin released, staged PUT slots returned), which the old planners did not. Peer reuse in this mode follows the radixshmem YAML (`distributed`) and no longer reads `enable_p2p_cpu`, which must stay off.
- `CacheEngineRadixShmem` (`flexkv/cache/radix_shmem_engine.py`) is trimmed to what `RadixShmemCacheEngine` uses: the `CacheEngineAccel`-compatibility parameters (`device_type`, `evict_ratio`, `evict_start_threshold`, `hit_reward_seconds`, `eviction_policy`, `protected_threshold`, `tokens_per_block=-1`), `take(strict=)`, `match(gpu_matched_blocks=)`, the `mempool` view, `start()`, `store` / `cluster_rank` and the `FLEXKV_TRACE_RADIX_PEER` variable are gone (prefetch logs at debug level; the planner reports mempool metrics itself).

Targeting SGLang:
- The native FlexKV backend is available in upstream SGLang `v0.5.16` and later; no patch is required ([sglang#29701](https://github.com/sgl-project/sglang/pull/29701))
- Add DeepSeek-V4 support for heterogeneous C4/C128/indexer KV groups, FullKV + SWA dual caches, attention/indexer compress-state sidecars, and layerwise restore ([#225](https://github.com/taco-project/FlexKV/pull/225))
- The matching DeepSeek-V4 SGLang adaptation is not merged yet. Use [sglang#31781](https://github.com/sgl-project/sglang/pull/31781) pinned to [`ee0465a`](https://github.com/sgl-project/sglang/commit/ee0465a09196421a6e4d53a3103eccdef1dd32ac) until it is merged

### Documentation

- Replace the legacy SGLang patch workflow with version-specific English and Chinese integration instructions
- Add English and Chinese CI guides covering the runner, release-wheel build, CPU unit-test scope, reference timing, local reproduction, and COS upload policy

## [1.2.0] - 2025-11-25
### Feature
Universal:
- Add support for distributed sharing of the KV Cache, to suppot KV Cache sharing between CPU and SSD, as well as distributed sharing of PCFS  ([#17](https://github.com/taco-project/FlexKV/pull/17))
- Add GDS (GPU Direct Storage) Support ([#25](https://github.com/taco-project/FlexKV/pull/25))
- TP16 support ([#26](https://github.com/taco-project/FlexKV/pull/26))
- Support more kv cache layout. Now include: vLLM, SGLang, TensorRT-LM ([#27](https://github.com/taco-project/FlexKV/pull/27))
- GDS refactor & gtensor support ([#42](https://github.com/taco-project/FlexKV/pull/42))
- Support construct TensorSharedHandle directly from CUDA IPC Handle ([#44](https://github.com/taco-project/FlexKV/pull/44))


Targeting vllm: 
- Support dp > 1 while integrated with vllm ([#18](https://github.com/taco-project/FlexKV/pull/18))
- Add launch scripts for vllm adaption ([#47](https://github.com/taco-project/FlexKV/pull/47))
- Support TP16 for vLLM+FlexKV ([#59](https://github.com/taco-project/FlexKV/pull/59))

Targeting TensorRT-LLM:
- Support using FlexKV on TensorRT-LLM ([#48](https://github.com/taco-project/FlexKV/pull/48))
- Support TP16 for TensorRT-LLM+FlexKV ([#53](https://github.com/taco-project/FlexKV/pull/53))

### Optimization
- Mla d2h transfer optimization ([#19](https://github.com/taco-project/FlexKV/pull/19))
- optimize SSD I/O ([#33](https://github.com/taco-project/FlexKV/pull/33))
- Enhance cache eviction with frequency-aware grace time mechanism ([#38](https://github.com/taco-project/FlexKV/pull/38))
- Replace std::map with std::unordered_map in RadixTree ([#41](https://github.com/taco-project/FlexKV/pull/41))

### Bugfix
- Fix wrong head number for DeepSeek for vllm integration ([#23](https://github.com/taco-project/FlexKV/pull/23))
- Fix bug, if cpu match len is bigger than ssd when put, it will cause error ([#24](https://github.com/taco-project/FlexKV/pull/24))
- Fix benchmark_worker ([#31](https://github.com/taco-project/FlexKV/pull/31))
- Fix segfault caused by radix tree array out-of-bounds access ([#39](https://github.com/taco-project/FlexKV/pull/39))
- Fix cache_info ([#40](https://github.com/taco-project/FlexKV/pull/40))
- Fix port for GPU registration ([#45](https://github.com/taco-project/FlexKV/pull/45))
- Fix SSD allocator ([#46](https://github.com/taco-project/FlexKV/pull/46))
- Fix vllm init num_kv_heads bug ([#67](https://github.com/taco-project/FlexKV/pull/67))
- Fix model_config for non-MLA models ([#68](https://github.com/taco-project/FlexKV/pull/68))

### Misc
- Add doc for: 
  FlexKV + TensorRT-LLM ([#52](https://github.com/taco-project/FlexKV/pull/52))
- For config: Simplify user configuration ([#37](https://github.com/taco-project/FlexKV/pull/37)), and other slight update ([#43](https://github.com/taco-project/FlexKV/pull/43))

## [1.1.0] - 2025-09-15 
- Add op-level callback for local get/put [#13](https://github.com/taco-project/FlexKV/pull/13)
- Add doc for: 
  FlexKV + Dynamo ([#14](https://github.com/taco-project/FlexKV/pull/14)), 
  flexkv_config.json ([#15](https://github.com/taco-project/FlexKV/pull/15)),

## [1.0.0] - 2025-09-11

### Added
- C++ radix tree for fast match, need set "index_accel": true in cache_config
- sync kernel launch
- a huge change that move cache engine to a library for accelerator(vLLM e.g.) to use instead of server-client mode.
  This accelerate the get and put when no KVCache is matched. This version includes breaking API changes and is not backward compatible. 
- add evict_ratio, need set "evict_ratio": 0.05 in cache_config
- reducing the bubble inner the launch kernel
- add vLLM 0.10.1.1 adapter

### Fixed
- cython release package


## [0.1.0] - 2025-08-29

### Init
- init version
- add license
