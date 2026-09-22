# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Feature

Universal:
- radixshmem mode: the CPU tier is an operator-run `radix-server` (the radixshmem project; requires radixshmem f939910 or later). FlexKV no longer creates a server: the operator starts one per node with a name and a byte budget (`radix-server --name /flexkv --data-bytes 64G [--swa-ratio R] [cluster flags]`), FlexKV's clients hand it the geometry (`shmradix.RadixClient(name, Geometry)`: tokens per block, bytes per CPU block / SWA page, SWA window, slot alignment), the server plans the slot counts from its budget and every FlexKV process adopts them into `CacheConfig.num_cpu_blocks` / `swa.num_slots` (`shm_radix_bootstrap.adopt_geometry`; `cpu_cache_gb` no longer sizes the CPU tier in this mode). A server serving another geometry is refused (`GeometryMismatch`), so every engine on one server runs the same model, page size and SWA configuration. `RadixServerProcess`, `build_radix_server_config`, `FLEXKV_RADIX_SERVER_LAUNCH_MODE`, `FLEXKV_RADIX_NODE_NAME`, `FLEXKV_RADIX_RPC_ADDRESS`, the `FLEXKV_RADIX_*` cluster variables and `FLEXKV_SHM_RADIX_ID` are gone.
- radixshmem mode is configured by one small YAML (`FLEXKV_RADIXSHMEM_CONFIG_PATH`, `flexkv/common/radixshmem_config.py`): `server` (`name`, `endpoint`, `ready_timeout_s`: which radix-server to attach to and how long to wait for it to be reachable and ready) and `client` (`prefetch_timeout_ms`, `prefetch_max_inflight`, `max_outstanding`). No file means `radix-server --name /flexkv` on the local socket. The former `cluster` / `data` / `index` sections are rejected: those settings are radix-server command-line flags now (migration table in `docs/radixshmem/config_zh.md`, launch scripts and a YAML in `examples/radixshmem_configs/`). FlexKV's own TE channel names derive from `server.name`.
- radixshmem mode brings no RHT registration chunk of its own any more (the former `REGISTER_CHUNK_TOKENS = 4096` constant, handed to the server as `index.register_chunk_size` in blocks): the chunk is the radix-server's `--register-chunk-tokens` (radixshmem's default 4096 tokens). `RadixGeometry.register_chunk_tokens` forwards a pinned value (0 = the server's), `adopt_geometry` and `CacheEngineRadixShmem.register_chunk_tokens` / `register_chunk_blocks` take the published value over, converted to blocks by radixshmem's rule (`tokens // tokens_per_block`, at least 1); `check_geometry` compares a pinned value and warns when the server's chunk is not a whole number of FlexKV blocks.
- radixshmem mode uses radixshmem's data plane: the CPU KV pool is the radix-server's SlotStore (one slot per block, attached by name in the TE and every transfer worker) and cross-node reuse is `RadixClient.pull_async` from the prefetch path (server-side RDMA READ), replacing FlexKV's own CPU allocation, `PEER2CPUTransferWorker`, mooncake wrapper and Redis address book on this path. Peer reuse follows the radix-server itself (`world_size > 1`, asked through `shm_radix_bootstrap.radix_server_is_distributed`) and no longer reads `enable_p2p_cpu`, which must stay off. radixshmem mode is CPU-tier only (`ssd_cache_gb` must be 0). Reference: `docs/radixshmem/config_zh.md`.
- radixshmem mode keeps FlexKV's process model: with one DP the KVTaskEngine runs in the engine process with its own TE subprocess, otherwise (dp_size > 1, `FLEXKV_INSTANCE_NUM` engines on one node) the DPs are clients of the node's KVServer whose KVTaskEngine and TE attach the radix-server like any other. Wherever a KVTaskEngine is built, `shm_radix_bootstrap.adopt_radix_server` first hands the server FlexKV's geometry and takes its slot counts and cluster rank over into `CacheConfig`. The embedded KVServer child now inherits `PYTHONPATH` / `LD_LIBRARY_PATH` / `PATH` along with the `FLEXKV_*` variables, so it imports what its parent imports (shmradix included) when the packages are not in site-packages.
- radixshmem planning moved out of `GlobalCacheEngine` into the subclass `RadixShmemCacheEngine` (`flexkv/cache/radix_shmem_planner.py`, selected by `KVTaskEngine` when `FLEXKV_ENABLE_RADIXSHMEM=1`). `GlobalCacheEngine` keeps two hooks only (`_prepare_request`, `_build_cpu_cache_engine`); its plan dataclasses, `TransferPlanHandle` and completion callbacks are back to their non-radixshmem shape. The subclass's handles roll back a plan cancelled before launch (match pin released, staged PUT slots returned), which the old planners did not, and a PUT whose planning fails after its slots were taken returns them and drops the pin before re-raising. `CacheEngineRadixShmem.take` clamps a request to the pool's size instead of letting radixshmem refuse it.
- `CacheEngineRadixShmem` (`flexkv/cache/radix_shmem_engine.py`) is trimmed to what `RadixShmemCacheEngine` uses: the `CacheEngineAccel`-compatibility parameters (`device_type`, `evict_ratio`, `evict_start_threshold`, `hit_reward_seconds`, `eviction_policy`, `protected_threshold`, `tokens_per_block=-1`), `take(strict=)`, `match(gpu_matched_blocks=)`, the `mempool` view, `start()`, `store` / `cluster_rank` and the `FLEXKV_TRACE_RADIX_PEER` variable are gone (prefetch logs at debug level; the planner reports mempool metrics itself).
- `gen_hashes` and `Hasher.update` hash numpy buffers directly (`c_ext.gen_hashes_numpy` / `update_numpy`) instead of going through `torch.from_numpy`, which is not safe to call concurrently. Hashes are unchanged for int64 tokens; `gen_hashes` keeps requiring int64 and the binding now checks dtype, contiguity and sizes instead of reinterpreting the buffer.

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
