# SGLang FlexKV Connector Integration

> **WARNING: THIS INTEGRATION IS NOT YET COMPLETE AND HAS NOT BEEN TESTED.**
>
> The patch was mechanically ported from the development branch and conflicts
> were resolved manually. It has NOT been validated with runtime tests.
> Please do NOT use in production until full testing is completed.

## Overview

This directory contains the patch to integrate FlexKV connector into SGLang. The patch adds KV cache sharing capabilities via FlexKV, supporting cross-node Tensor Parallelism (TP) and Pipeline Parallelism (PP).

## Target SGLang Version

- **Repository**: https://github.com/sgl-project/sglang.git
- **Branch**: `deepseek_v4`
- **Base Commit**: `b69485ff4272b5a306045fa0cf7fc2c5692d391f` (small fix h200 docker)
- **Based on Release**: v0.5.12.post1 + 68 commits (deepseek_v4 branch)

## Patch File

- `sglang_flexkv_connector.patch` — Combined patch containing all FlexKV connector changes

## How to Apply

```bash
# Clone and checkout the target branch
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout deepseek_v4
git reset --hard b69485ff4272b5a306045fa0cf7fc2c5692d391f

# Apply the patch
git apply sglang_flexkv_connector.patch

# Verify
git status
```

## Changes Included

### New Files (7)

| File | Description |
|------|-------------|
| `python/sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py` | FlexKV connector main implementation (KV get/put, cross-node TP/PP) |
| `python/sglang/srt/mem_cache/storage/flexkv/flexkv_comm.py` | FlexKV communication layer (RankInfo, sync groups, async reaper) |
| `python/sglang/srt/mem_cache/extended_radix_cache.py` | Extended radix cache wrapping base cache with connector support |
| `python/sglang/srt/mem_cache/kv_connector.py` | KV connector abstract base class and factory |
| `python/sglang/srt/mem_cache/session_aware_cache.py` | Session-aware cache decorator for streaming session KV management |
| `python/sglang/srt/mem_cache/allocator_ascend.py` | Ascend NPU allocator support |
| `docs/advanced_features/kv_connector.md` | User-facing documentation for KV connector configuration |

### Modified Files (9)

| File | Change Summary |
|------|----------------|
| `python/sglang/srt/server_args.py` | Add `--kv-connector-cls` argument and related config |
| `python/sglang/srt/managers/scheduler.py` | Integrate connector lifecycle: init, prefetch check, event loop, abort handling |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | Add cache breakdown metrics (device/storage/connector) |
| `python/sglang/srt/managers/schedule_batch.py` | Add `cached_tokens_extended_device` field and `update_connector_state` param |
| `python/sglang/srt/managers/schedule_policy.py` | Adapt `init_load_back` call for connector integration |
| `python/sglang/srt/mem_cache/base_prefix_cache.py` | Change `init_load_back` signature, rename `check_hicache_events` -> `check_kv_events` |
| `python/sglang/srt/mem_cache/hiradix_cache.py` | Rename `check_hicache_events` -> `check_kv_events` |
| `python/sglang/srt/layers/attention/nsa_backend.py` | Add FA3 import guard for FlexKV+Blackwell compatibility |
| `python/sglang/srt/mem_cache/cpp_radix_tree/.clang-format` | Clang format config |

## Features

- KV cache prefix matching and sharing across SGLang instances via FlexKV
- Cross-node Tensor Parallelism (TP) support
- Pipeline Parallelism (PP) support with decentralized control plane
- Async prefetch with adaptive work reaper
- Observability: cache hit metrics per request (device/storage breakdown)
- Compatible with HiCache hierarchical caching

## Configuration

```bash
# Launch SGLang with FlexKV connector
python -m sglang.launch_server \
    --model-path <model> \
    --kv-connector-cls FlexKVConnector \
    # ... other args
```

See `docs/advanced_features/kv_connector.md` (in the patched repo) for full configuration details.

## Source Reference

- **Development Repo**: https://github.com/zhuofan1123/sglang.git (branch: `taco_sglang`)
- **Patch derived from**: 23 commits after `ac84702b6186bbcc2c92a8d8356af624a8c39bcc`

## TODO

- [ ] Runtime smoke test: single-node TP launch with FlexKV connector
- [ ] Cross-node TP correctness validation
- [ ] PP mode validation
- [ ] Prefetch / async reaper stress test
- [ ] Compatibility test with HiCache enabled
- [ ] Performance regression check vs. baseline (no connector)
