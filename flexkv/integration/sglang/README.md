# Using FlexKV with SGLang

[中文文档](README_zh.md)

## Version compatibility

Choose the SGLang integration path that matches your model.

> Status last checked: August 6, 2026.

| Use case | SGLang version | Patch required? |
| --- | --- | --- |
| Standard models | SGLang `v0.5.16` or later, or a recent `main` | No |
| DeepSeek V4 | SGLang PR [#31781](https://github.com/sgl-project/sglang/pull/31781), pinned to commit [`ee0465a`](https://github.com/sgl-project/sglang/commit/ee0465a09196421a6e4d53a3103eccdef1dd32ac) | No local patch; use the pinned SGLang commit |
| DeepSeek V4.1 | SGLang PR [#31781](https://github.com/sgl-project/sglang/pull/31781), pinned to the commit listed below | No local patch; use the pinned SGLang commit e74b2ec897256fe50bffaa97a066e14bfc113c1f |

### Standard models: use upstream SGLang directly

The base FlexKV integration was merged into upstream SGLang by
[sglang#29701](https://github.com/sgl-project/sglang/pull/29701) and is included
in SGLang `v0.5.16` and later. Do **not** apply
`sglang_flexkv_connector.patch` to these versions.

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.16  # or use a newer release/main
```

### DeepSeek V4: pin the unmerged SGLang adaptation

FlexKV's DeepSeek V4 support is present on FlexKV `main` (merged by
[FlexKV#225](https://github.com/taco-project/FlexKV/pull/225)), but the matching
SGLang adaptation has **not** been merged upstream yet. Until
[sglang#31781](https://github.com/sgl-project/sglang/pull/31781) is merged, fetch
the PR and pin the commit below instead of using an SGLang release or `main`
alone:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/31781/head
git checkout -b flexkv-dsv4 ee0465a09196421a6e4d53a3103eccdef1dd32ac
```

Use this with the current FlexKV `main` branch:

```bash
git clone https://github.com/taco-project/FlexKV.git
cd FlexKV
git checkout main
```

The PR is still under development, so check its status before updating the
pinned SGLang commit. DeepSeek V4's unified-KV layout and `--enable-hisparse`
are not supported by that commit.

### DeepSeek V4.1: pin the unmerged SGLang adaptation

FlexKV's DeepSeek V4.1 support is on FlexKV `main` (merged by
[FlexKV#303](https://github.com/taco-project/FlexKV/pull/303)), but the matching
SGLang adaptation has **not** been merged upstream yet. Until
[sglang#31781](https://github.com/sgl-project/sglang/pull/31781) is merged, do
not use an SGLang release or `main` alone. Fetch that PR and pin the commit
below:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/31781/head
git checkout -b flexkv-dsv41 e74b2ec897256fe50bffaa97a066e14bfc113c1f
```
The commit above is recorded by this repository; verify it is available and compatible with your FlexKV and model before deployment.


The current FlexKV `main` branch is sufficient for this deployment.

## Launch

After installing SGLang and FlexKV, create a FlexKV configuration file. For
example:

```yaml
# flexkv_config.yaml
cpu_cache_gb: 16
```

Launch SGLang with the built-in FlexKV backend:

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --enable-flexkv \
  --flexkv-config-file /path/to/flexkv_config.yaml \
  # ... other SGLang arguments
```

The equivalent explicit backend option is
`--radix-cache-backend flexkv`. See SGLang's
[`flexkv/README.md`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/flexkv/README.md)
for detailed configuration and validation instructions.

## Chunked remote prefetch

Chunked prefetch is disabled by default. It splits a node-local Mooncake Store
→ CPU prefetch into smaller transfer graphs. It requires the matching SGLang
scheduler/cache hooks; installing FlexKV alone does not add those hooks. Enable
it only with a compatible SGLang revision and validate the deployed model and
runtime together. The chunked path rejects SSD, FlexKV P2P sharing, multi-node
and TensorRT-LLM remote mode.

Add the following to an existing FlexKV YAML configuration, retaining the CPU
capacity and Mooncake Store settings and setting SSD capacity to zero:

```yaml
enable_chunked_prefetch: true
prefetch_options:
  policy: best_effort
  chunk_max_blocks: 128
  max_inflight_chunks: 2
prefetch_max_sessions: 128
prefetch_max_reserved_bytes: 536870912
prefetch_max_pinned_bytes: 2147483648
prefetch_result_ttl_s: 60
```

These limits are within the existing CPU pool, not additional allocations.
`chunk_max_blocks` and `max_inflight_chunks` must be positive; global session,
reserved-byte, pinned-byte and result-TTL limits must be positive. An explicit
FlexKV policy takes precedence over SGLang's prefetch policy. If none is
specified, the connector reads `hicache_storage_prefetch_policy` from SGLang.

| Policy | Behavior in SGLang |
| --- | --- |
| `wait_complete` | Uses the original whole-task `prefetch_async` path and disables the chunk runtime. |
| `timeout` | Stops admitting chunks when the timeout budget expires, then drains submitted work. |
| `best_effort` | Stops admission when the scheduler signals demand, then drains submitted work. |

A timeout limits **admission**, not response latency. There is no hard
cancellation of an in-flight transfer. The result is a contiguous successfully
published prefix; capacity limits or failures can produce a partial or empty
result. For Full+SWA, an eligible checkpoint must include both Full KV and its
SWA/state. The matching SGLang adapter must support the model's checkpoint
semantics. A canceled session can leave already published data cached.

`timeout_budget_s` overrides the formula `min(timeout_max_s, timeout_base_s +
timeout_per_ki_token_s * candidate_tokens / 1024)`. Their defaults are 2.0,
0.1 and 30.0 seconds respectively; `timeout_budget_s` defaults to unset. When
using environment-backed configuration instead of a config file, use
`FLEXKV_ENABLE_CHUNKED_PREFETCH=1` and e.g.
`FLEXKV_PREFETCH_OPTIONS='{"policy":"timeout","timeout_budget_s":0.02}'`;
the four limits also have corresponding `FLEXKV_PREFETCH_MAX_SESSIONS`,
`FLEXKV_PREFETCH_MAX_RESERVED_BYTES`, `FLEXKV_PREFETCH_MAX_PINNED_BYTES` and
`FLEXKV_PREFETCH_RESULT_TTL_S` variables. An already running external server
retains its own configuration; clients do not reconfigure it.

The direct `KVManager` session API exposes `prefetch_capabilities()`,
`start_prefetch`, `progress_prefetch`, `poll_prefetch`,
`notify_prefetch_demand`, `stop_prefetch`, `wait_prefetch` and
`release_prefetch`. Supply the complete token chain, not a sliced suffix;
positions are absolute token offsets. `poll` and `wait` observe state but do
not signal demand; a wait timeout leaves the session active. The terminal
snapshot distinguishes planned hits from `reusable_prefix_end_token` and
`l3_loaded_spans`, and exposes `lease_valid`, submitted and in-flight work.
Take a held GET/match while the result lease is valid before releasing it.
Release of an active session stops further admission; its resources remain
owned until submitted work drains. A released result is not guaranteed to remain resident.

```python
import numpy as np
from flexkv.prefetch import PrefetchOptions

handle = manager.start_prefetch(
    np.asarray(full_token_ids, dtype=np.int64),
    PrefetchOptions(policy="timeout", timeout_budget_s=0.020,
                    chunk_max_blocks=32, max_inflight_chunks=2),
    namespace=["tenant", "model-version"],
)
snapshot = manager.poll_prefetch([handle])[handle]
snapshot = manager.wait_prefetch(handle, timeout_s=5)
# Obtain a held GET/match while snapshot.lease_valid, then release the lease.
manager.release_prefetch(handle)
```

Keep model, weights, dtype, TP and KV layout consistent within a shared Store:
the existing key does not fingerprint the complete layout. Use separate stores
when changing these settings unless every relevant operation's namespace
isolation has been verified with the deployed adapter. For correctness testing,
confirm the remote hit, compare loaded KV bytes and generated outputs with an
appropriate cold baseline, exercise stop/abort and verify resources after drain.
CPU tests alone do not establish GPU or real transport correctness. See
[`tests/prefetch`](../../../tests/prefetch) for the available tests.

## Optional Index-K deduplication

For compatible DSA models, `FLEXKV_DEDUP_INDEXER_GROUP=1` registers only active
Index-K buffers; it is disabled by default. Use a new cache pool or an isolated
layout/version namespace when changing this setting: existing entries are not
migrated or cleared automatically. See [IndexCache deduplication](../../../docs/indexcache_dedup.md)
for supported layouts and the limits of the existing validation.

## Integration boundary

### Read-only local prefix observations

All participating ranks can call `pending_probe_plan(observed_times)` together
to obtain the sync leader's rate-limited refresh indices, then call
`peek_cpu_prefix_tokens(tokens, mask, namespace=...)` for each selected entry.
The latter queries CPU metadata only and releases temporary pins without
changing request lookup/restore/prefetch ownership. `0` is a known miss;
`None` means unavailable and must remain unknown in the load estimate.
Failed cancellation retains at most one temporary task, retries it before
another probe, and makes reset fail consistently across ranks until it can be
released. Use the matching SGLang pending-prefix observer for FCFS load
reporting; optional callers must capability-check both APIs.

`lookup_kv(..., cpu_only=True)` also permits explicit local-only admission
preflight; it retains the normal held-lookup ownership contract.

The core FlexKV connector logic (`FlexKVComm` and `FlexKVConnector`) lives in
this directory:

| File | Description |
|------|-------------|
| `comm.py` | 3-axis (PP × CP × TP) cross-rank communication layer |
| `connector.py` | `FlexKVConnector` wrapping `KVManager` for SGLang |

SGLang's `storage/flexkv/__init__.py` dynamically loads the connector from
`flexkv.integration.sglang.connector` via `importlib`, so future FlexKV logic
changes typically only require updating the FlexKV repo.

Set `FLEXKV_ENABLE_COLLECTIVE_SYNC=0` to disable cross-rank sync
(scatter/barrier/all_reduce). In deployments without Pipeline Parallelism (PP),
disabling it reduces sync overhead and improves performance.

## About the patch in this directory

`sglang_flexkv_connector.patch` is retained only as a legacy reference for the
old pre-upstream integration. It is not required for supported SGLang releases,
and it must not be used for the DeepSeek V4 path above.
