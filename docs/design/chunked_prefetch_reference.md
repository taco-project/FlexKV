# Chunked prefetch — API and lifecycle reference

Read the [Chinese implementation guide and sequence diagrams](chunked_prefetch.md) first.
This reference retains the detailed interface and ownership contracts.

## Scope and semantics

This opt-in implementation adds `wait_complete`, `timeout`, and `best_effort` to
node-local Mooncake → CPU prefetch, including Full+SWA checkpoint publication.
The existing prefetch API remains available. SSD, P2P, multi-node and TRT remote
mode are rejected when chunked prefetch is enabled. SWA requires the C++ radix
engine and registered SWA transfer buffers; multi-group state also requires the
adapter-inferred `SWAPoolConfig.snapshot_bytes` for reservation accounting. No transfer-worker code, GPU kernel,
`TransferOp` schema, or existing graph submission/result wire format is changed.

| Policy | When admission stops | When the caller may consume the result |
| --- | --- | --- |
| `wait_complete` | All planned chunks claimed, or backend/capacity failure | All claimed graphs have reached terminal and publication has finished |
| `timeout` | Monotonic budget expires; checked again immediately before claim | After the claimed window drains |
| `best_effort` | Scheduler demand, when the request is considered for admission | After the claimed window drains |

`timeout` bounds admission, **not total response latency**. The sealed window
includes graphs in the local outbox, IPC pipe, transfer-manager queue, and worker.
There is no hard cancellation of an active transfer. A failed/uncertain sender
stops the runtime and retains buffers; retrying ambiguous submissions is forbidden.

Capacity exhaustion may produce an empty/partial result even for `wait_complete`.
Only a continuous, successfully published prefix can be returned. Out-of-order
completed chunks retain their allocation/credit until their preceding chunks are
resolved. A failed block closes the usable prefix; later chunks are drained and
recycled. Aborted requests may leave successfully fetched bytes in the cache.
Reset/shutdown discard unpublished inflight chunks.

Keep the model, weights, dtype, TP size, and KV layout consistent for a shared
Mooncake pool. The existing BLOCKFIRST payload is partitioned by TP, but its key
does not fingerprint that layout. H20 validation reproduced incorrect output
when TP2 consumed TP1-written KV. Use separate stores (or namespaces with the
direct API) when changing these settings; the SGLang adapter currently needs a
separate store because its foreground path does not propagate namespaces. This
implementation does not change the existing storage key format or transcode KV.

## Components and ownership

- `prefetch/types.py`: immutable, pickle-safe handles, options, capabilities, and snapshots.
- `prefetch/policy.py`: startup policy registry; frozen when the first coordinator starts.
- `prefetch/coordinator.py`: parent sessions, fair bounded admission, seal/drain,
  contiguous publication, leases, and terminal retention.
- `prefetch/planner.py`: async full-chain hashing/metadata query, lazy detached CPU
  allocation, one existing REMOTE2H graph per chunk, and existing deferred-insert commit.
- `prefetch/runtime.py`: a single owner thread for all task state/completions;
  one bounded sender outbox for each existing transfer handle.
- `kvtask.py`: dispatches foreground GET/PUT/match/launch/cancel and completions to
  that owner when enabled. Active tasks have no TTL; only terminal legacy results
  retain the old bounded expiry behavior.
- `kvmanager.py` / `server/*`: equivalent direct and server/client APIs; independent
  per-call local ZMQ control sockets, ownership checks, and nonblocking legacy wait
  registration so a pending wait does not prevent stop/progress RPCs.
- `integration/sglang/connector.py`: leader submits/progresses; ranks receive the
  same handle/terminal snapshot; a held foreground GET takes over the result lease.

The transfer handle still receives `submit_batch(List[TransferOpGraph])`. A
window of two chunks overlaps next-chunk planning/submission with current transfer;
multiple sessions' graphs can share a batch. This is the TBO-inspired pipeline
mechanism. An end-to-end speedup has not been established by the completed correctness tests.

The runtime bridges existing non-selectable completion handles with a 2 ms poll;
commands wake it immediately. It fills another window slot immediately when
possible, and sleeps when no new chunk was admitted. Foreground planning remains
serialized by the owner; a long synchronous foreground operation or an in-flight
SDK metadata call is not preempted. Metadata hashing/query runs in its own executor
without the radix lock and stops between bounded query batches. SDK call latency
also affects shutdown of that executor.

## Interface

```python
import numpy as np
from flexkv.prefetch import PrefetchOptions

handle = manager.start_prefetch(
    np.asarray(full_token_ids, dtype=np.int64),
    PrefetchOptions(policy="timeout", timeout_budget_s=0.020,
                    chunk_max_blocks=32, max_inflight_chunks=2),
    namespace=["tenant", "model-version"],
)

# Non-consuming observations. Neither poll nor wait emits scheduler demand.
snapshot = manager.poll_prefetch([handle])[handle]
manager.notify_prefetch_demand([handle])  # best_effort seal; timeout checks its deadline
snapshot = manager.wait_prefetch(handle, timeout_s=5)
assert snapshot.terminal
# Obtain a held GET/match while this lease is valid before releasing it.
manager.release_prefetch(handle)
```

Other entry points: `prefetch_capabilities()`,
`progress_prefetch(handles, demand_handles=())`,
`stop_prefetch(handle, reason="request_abort")`.
`wait_prefetch()` timing out leaves the session active; explicitly stop/release to
abort. Direct `KVTaskEngine.start_prefetch()` additionally accepts `dp_client_id`.

Positions are absolute token offsets. `candidate_start_token` controls timeout
budget length; callers still provide the **full token chain**, not a sliced suffix.
Snapshots expose `planned_end_token`, `reusable_prefix_end_token`, `l3_loaded_spans`,
`submitted_chunks`, `sealed_submit_seq`, `inflight_chunks`, `inflight_bytes`, and
`lease_valid`. Planned hits are never reported as loaded tokens.

Handles carry a runtime epoch and monotonically increasing session ID. Active
records cannot expire. Terminal leases expire after 60 seconds by default; a
bounded lightweight tombstone then returns an empty `expired` result. Once that
bounded tombstone is evicted, old/unknown handles fail explicitly. Reset changes
the epoch. Release is idempotent. After releasing a lease the snapshot does not
promise continued residency; take a foreground held GET first.

Pin accounting covers entire locked nodes and ancestors, including a concurrently
published suffix beyond this session's boundary. Exact per-block compressed-group
bytes are used. Admission also accounts for staging that will become pinned. If a
concurrent larger node cannot fit the pin budget, new data can remain cached but
the session reports only its previously protected prefix.

## Configuration and SGLang

Default is disabled. Add to the existing FlexKV JSON configuration:

```json
{
  "enable_chunked_prefetch": true,
  "prefetch_options": {
    "policy": "best_effort",
    "chunk_max_blocks": 128,
    "max_inflight_chunks": 2,
    "timeout_base_s": 2.0,
    "timeout_per_ki_token_s": 0.1,
    "timeout_max_s": 30.0
  },
  "prefetch_max_sessions": 128,
  "prefetch_max_reserved_bytes": 536870912,
  "prefetch_max_pinned_bytes": 2147483648,
  "prefetch_result_ttl_s": 60
}
```

This is an additive fragment; retain the existing CPU capacity and Mooncake
connection configuration and set SSD capacity to zero. If loading config entirely
from environment, use `FLEXKV_ENABLE_CHUNKED_PREFETCH=1` and
`FLEXKV_PREFETCH_OPTIONS='{"policy":"timeout","timeout_budget_s":0.02}'`.
The four global limits also support `FLEXKV_PREFETCH_MAX_SESSIONS`,
`FLEXKV_PREFETCH_MAX_RESERVED_BYTES`, `FLEXKV_PREFETCH_MAX_PINNED_BYTES`, and
`FLEXKV_PREFETCH_RESULT_TTL_S`. A JSON config supplied through `FLEXKV_CONFIG_PATH`
takes precedence over these environment-backed UserConfig values.

Timeout budget is `min(max_s, base_s + per_ki_s * candidate_tokens / 1024)`;
`timeout_budget_s` overrides the formula. Configuration policy wins over SGLang's
`--hicache-storage-prefetch-policy` when both are supplied.

The SGLang changes are integrated into the existing
[FlexKV adaptation PR #31781](https://github.com/sgl-project/sglang/pull/31781),
at commit `e5b00611bd`. Use that revision with this FlexKV version. It retains
the adaptation's newer restore/abort/reset ownership and deferred Store handling.

The companion `flexkv/integration/sglang/sglang_chunked_prefetch.patch` is an
incremental patch against that PR's previous commit `16780ea0c8`. From that
checkout, run `git apply --check /path/to/sglang_chunked_prefetch.patch` followed
by `git apply /path/to/sglang_chunked_prefetch.patch`. Do not apply it again on
`e5b00611bd`, or directly on SGLang main. The separate PR #38451 is superseded.

The adapter starts prefetch on queue entry without a foreground remote lookup,
emits demand at scheduler candidacy, propagates abort, and passes actual
L3-loaded spans to existing hit accounting. These hooks are independent of
`hicache_storage_backend`; no HiCache backend needs to be configured.
Chunked-mode foreground LOOKUP searches CPU only after prefetch stop-and-drain.

The first SGLang adapter version skips this optional prefetch for `extra_key` or
`cache_salt` requests because the existing foreground adapter has no consistent
namespace propagation. The direct FlexKV API supports namespaces. Dense MP mode
is the initial end-to-end acceptance target; layerwise mode shares the unchanged
H2D path but requires separate GPU acceptance.

## Chunk sizing and memory limits

`chunk_max_blocks` is the only chunk-size control. The former
`chunk_target_bytes` option was removed before release: remove it from older
experimental JSON files. The planner still computes actual Full KV and SWA/state
bytes and shrinks chunks to fit resource limits and checkpoint boundaries.
Removing the old 32 MiB target can increase actual chunk sizes; retune and rerun
latency/stop-tail measurements rather than carrying old performance results over.

The two byte limits are advanced per-instance safeguards, not duplicate size
controls or copied HiCache parameters. `prefetch_max_reserved_bytes` bounds
uncommitted staging across all sessions, including queued and out-of-order work.
`prefetch_max_pinned_bytes` bounds protected CPU prefixes and reserves room for all
staging to become pinned. Completed results may remain pinned after staging is
released, until foreground GET acquires its own reference or the result lease
expires. These are accounting limits within the existing CPU pool, not two extra
allocations. Pin accounting is conservative across overlapping session leases.

## Monitoring status

Existing Python/C++ exporters use `FLEXKV_ENABLE_METRICS=1`, with
`FLEXKV_PY_METRICS_PORT=8080` and `FLEXKV_CPP_METRICS_PORT=8081`. Both bind to
localhost in their own network namespace; scrape with a colocated agent.
SGLang request/queue/latency metrics require its own `--enable-metrics` flag.

The coordinator logs one `[FlexKV-Prefetch]` terminal record with policy, reason,
submitted/sealed counts, delivered L3 tokens, elapsed/drain milliseconds and error.
Enable INFO for the standard Python logger `flexkv.prefetch.coordinator` in the
application logging configuration. `FLEXKV_LOG_LEVEL` alone configures a different
logger. Per-session snapshots expose inflight counts/bytes and lease validity.

This PR does not add dedicated prefetch Prometheus series. Chunk completions are
routed to the coordinator before legacy transfer accounting, so existing transfer
or joint-prefetch counters must not be interpreted as complete chunked-prefetch
coverage. Metadata hits, CPU publication, H2D completion and model consumption are
distinct. Follow-up instrumentation should add terminal outcome/token counters,
elapsed/drain histograms, reserved/pinned/limit and inflight/state gauges, capacity
source counts and runtime health. Use bounded labels only; keep session/request
IDs and exception text in logs. See the [review document](chunked_prefetch_design_review.md#s9)
for monitoring boundaries and acceptance requirements.

## Failure and lifecycle contract

- Start/admission failure is synchronized across ranks and falls back to normal
  compute. Capability mismatch fails initialization explicitly.
- Progress/IPC uncertainty raises consistently across ranks. It must not be treated
  as successful prefetch, and no inflight slot is recycled on uncertainty.
- Cancel seals and detaches request-facing tracking, retaining the coordinator
  resource ledger until drain. A running legacy task similarly retains callbacks.
- Reset seals admission. If work is inflight, reset raises a draining/retry error;
  the caller retries after drain. The cache is cleared only at the drained boundary.
- Shutdown stops admission and waits for graphs (30-second control drain guard).
  Failure/timeout leaves registered buffers owned, rather than freeing them under
  a worker. Existing lower-level shutdown behavior is unchanged after successful drain.
- `[FlexKV-Prefetch]` terminal logs include epoch/session/policy/reason, submitted
  chunks, loaded tokens, elapsed time, and seal-to-terminal drain time. Snapshot
  fields provide per-session in-flight observability; a Prometheus exporter is
  not added in this first version.

## Validation

Run deterministic state-machine and native integration tests:

```bash
# Lightweight control tests also work without a GPU or native extension.
PYTHONPATH=.:tests:tests/prefetch python -m pytest --confcutdir=tests/prefetch tests/prefetch -q
# Native tests require a matching/importable flexkv.c_ext and runtime libraries.
# A skip means native coverage was NOT exercised.
```

Coverage includes three policies, deadline crossing during reserve, stop before
metadata resolves, missing/short/failed bitmaps, out-of-order terminal events,
resource caps, namespace chains, local-prefix/concurrent insertion, result expiry,
reset/abort, actual CPU slot bytes, actual interprocess graph/result pipes, control
thread drain without request polling, real ZMQ RPC concurrent with pending legacy
waits, DP ownership checks, and follower result consistency. SGLang contract tests
execute actual wrapper/scheduler method bodies with their GPU dependencies omitted;
they do not boot a serving process.

An opt-in real test is available in `tests/prefetch/test_gpu_mooncake.py`. Set
`FLEXKV_TEST_REAL_MOONCAKE=1` and `FLEXKV_MOONCAKE_STORE_CONFIG_PATH` only against
a dedicated test store. It uses real CUDA IPC, the existing worker, and Mooncake;
one fault-injection case force-removes a key in its own unique namespace after
metadata matching. The September 6 dense-path snapshot passed all 25 H20 cases, including window 1/2/4,
byte limits, every returned GPU element, unchanged suffix, inflight stop/drain,
missing remote data, concurrent GET, and allocator/pin cleanup. That snapshot
passed two consecutive complete GPU runs and 367 related regression cases.
SGLang Qwen3-0.6B TP1 passed all three policies (including queued best_effort);
TP2 passed full and timeout-partial restores against data written with the same
TP size. All compared generated token IDs matched cold-compute references.
These runs used Python debug modules with an optimized CUDA extension, not a
Cython release build; they do not establish a performance improvement or soak
qualification.

Before enabling in production, run a dedicated GPU + Mooncake environment with
both matching repository versions:

1. Seed a unique namespace via PUT and wait for full graph completion. Evict only
   that test engine's CPU cache; prove CPU miss + Mooncake hit before prefetch.
2. Exercise all policies at window 1/2/4, chunk sizes 8/32/128 blocks and several
   prompt lengths. For partial policies trigger demand/deadline while a chunk is
   inflight. Require a partial case and `submitted_after_seal == 0`.
3. Follow terminal result with held GET → H2D and compare every loaded KV element
   against seeded data; recompute the remaining tail and compare deterministic
   generated output with a no-prefetch baseline.
4. Repeat abort/reset, shared-prefix concurrency, Mooncake partial-read failures,
   and worker failure. Verify allocator/lock baselines after drain and no early
   terminal at any rank (TP/DP and layerwise each need acceptance).
5. Measure physical REMOTE2H/H2D, IPC submit latency, drain tail, GPU overlap,
   forward stretch, TTFT and throughput against disabled/whole-graph baselines.
   Repeat fixed workloads; report tier-hit proof and topology with results.

CPU tests alone do not establish GPU or transport correctness. The H20 runs above
provide that evidence for the stated configurations; performance and soak
acceptance remain separate.

## SWA checkpoint publication

With `PrefetchOptions.swa_aware=True`, metadata lookup checks Full and SWA keys.
The planner retains all complete checkpoint offsets and clamps the planned end to
the last one. A chunk ending at a checkpoint adds the existing SWA peer op to its
Full graph. Both completion payloads flow into the existing deferred publication
contract. An intermediate Full-only chunk may be cached without advancing the
reusable result. `l3_loaded_spans` is clipped to the last resident checkpoint.

Full and SWA have separate pins and accounting. Abort drains and may retain valid
cache entries; reset/shutdown discard unpublished staging. A failed or absent SWA
result cannot advance the usable checkpoint, even if every Full block succeeded.

The SGLang hybrid wrapper checks admission before allocating restore slots, keeps
an explicit request restore lease until normal cache insertion, and passes the
full hash-chain input and candidate offset to prefetch. Completed GPU-cache
entries and SWA eviction remain owned by the inner UnifiedRadixCache.

## September 7 validation update

The final Python sources passed 445 related regressions (9 Python-radix SWA
cases skipped), 29 SGLang contract/admission tests, all 25 real GPU/Mooncake
prefetch cases, and one existing FULL + SWA byte-exact GPU roundtrip.
GLM-5.2-FP8 and DeepSeek-V4-Flash-FP8 each passed 210 formal requests on H20
TP8: three policies, six C1 requests, and 32 requests at each of C8/C32 per
policy. Every complete 64-token output matched its measured zero-hit reference.
V4 exercised full and partial FULL + SWA/state checkpoints; both models exercised
immediate demand and queued remote loads. These were shared-prefix, single-round
bounded checks, not equal-cache-state performance comparisons or soak acceptance.

An early V4 16K formatting divergence also reproduced with native GPU prefix
reuse and FlexKV disabled; native and FlexKV output IDs were identical. The formal
corpus fixes only the initial output format, then measures a fresh cold reference
and retains the full output-ID comparison. GLM runs preceded the final capacity
classification, error-detail logging and API annotation changes; exact source
differences are retained in the run evidence. GPU teardown emitted a PyTorch CUDA
IPC warning; after process exit each GPU used 4 MiB at 0% utilization, and after
stopping the dedicated keeper/container all eight GPUs used 0 MiB with no GPU processes.
