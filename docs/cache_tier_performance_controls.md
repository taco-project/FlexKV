# Cache tier performance controls

These opt-in controls bound remote read pressure and reduce SGLang control-message
overhead. Defaults retain serial Store reads, unrestricted data sessions within
the existing session/byte limits, and the legacy control framing.

| Environment variable | Default | Contract |
| --- | --- | --- |
| `FLEXKV_MOONCAKE_GET_CONCURRENCY` | `1` | 1–8 SDK Get calls per data client; query-only clients remain serial. |
| `FLEXKV_MOONCAKE_GET_MAX_INFLIGHT_BYTES` | `536870912` | Aggregate padded Get byte budget per client, required positive for parallel reads. |
| `FLEXKV_PREFETCH_DATA_CONCURRENCY` | `0` | Maximum sessions performing data I/O; 0 uses the existing unrestricted policy. Metadata admission and retained result leases still count against `max_sessions`. |
| `FLEXKV_CONTROL_INLINE_BYTES` | `0` | 0, 256, 512, 1024, or 4096; enable the same value on every rank. Enabled only for PP=1. |

With parallel reads enabled, each Get object's budget cost is `ceil(size / 4096) * 4096 + 8192`. Parallel
windows give each call at most the aggregate budget divided by concurrency,
while `max_get_batch_bytes` independently retains its raw-byte SDK batch limit.
Serial reads retain the existing raw-byte batching behavior. A single oversized
object fails before SDK submission. Results preserve input order; short reads,
failed reads, and result-count mismatches do not become successful cache hits.
All accepted calls drain before an error escapes. Worker shutdown drains Get
calls before unregistering destination memory regions.

A session waiting for a peer's byte credits remains pending. It only takes the
permanent-capacity fallback when no outstanding I/O holds credits. Data-session
slots remain occupied until all submitted graphs drain, including cancelled
sessions; finishing I/O does not drop the CPU result/pin lease.

Inline framing combines length and a small serialized object in one fixed-size
message. Larger objects use the fixed header followed by their payload. A leader
serializes once for its fan-out. Legacy and inline framing must not be mixed
within a rank group.

## Validation

```sh
pytest tests/test_mooncake_bounded_io.py tests/test_mooncake_parallel_get.py tests/prefetch/test_coordinator.py
torchrun --standalone --nproc-per-node=4 tests/run_control_framing_distributed.py
```

The SDK tests use controlled thread functions. The distributed test uses real
four-rank CPU Gloo and checks legacy, 256-byte, and 1024-byte framing, including
large payloads and garbage collection while sends are pending. These checks do
not establish live RDMA, GPU/model correctness, or production acceptance.

Model-level validation and measurements must record the exact source heads,
workload, hardware, and matched native GPU-cache baseline. Data-path completion
and throughput do not establish model-output correctness.

## Admitted layerwise restore batches

Serving integrations can prepare each admitted request with
`start_load_kv_layerwise(..., defer_launch=True)` and call
`flush_layerwise_loads(request_keys)` once, before preparing the forward batch.
Immediate launch remains the default. The request keys must preserve the
integration's complete request identity, including retry/attempt information.

All ranks must agree on successful destination allocation with
`sync_load_ready(local_ready)` before entering a restore collective. A failed
vote requires successful peers to release their allocations and cancel their
held lookups. Integrations without this API retain their existing behavior.

Preparation retains the destination mappings and producer counter without
submitting DMA. Flush validates the same admitted members on every rank and
merges their transfers into one operation, with one completion signal per
layer. Cancel unlaunched members with `release_pending(request_key)`. An
unacknowledged launch retains ownership and rejects reset, because accepted
DMA may still be active.

The companion SGLang integration enables this protocol with
`FLEXKV_BATCH_LAYERWISE_RESTORE=1`; it defaults to off and falls back to
immediate launch when the connector lacks the batching API.

```sh
pytest tests/test_sglang_layerwise_batch.py tests/test_sglang_comm_async_reap.py
torchrun --standalone --nproc-per-node=4 tests/run_gloo_async_reap_distributed.py
```

## CE copies with chunked host registration

`FLEXKV_CUDAHOST_CHUNK_SIZE_GB` divides host registration into separate CUDA
allocations. CUDA can reject a DMA spanning those registrations, even when their
virtual addresses are adjacent. The CE path records the exact registration
base, size, and chunk size, then splits linear or pitched copies at those
boundaries. Pitched rows that cross a boundary are split within the row;
consecutive rows within one registration retain a single 2D copy.

Partial BlockFirst H2D transfers with a bounded number of contiguous runs use
direct pitched copies when 2D CE copies are enabled. This avoids a device staging
buffer and a D2D transpose. Full-block, D2H, and highly fragmented cases keep
their existing path selection.

Rebuild the native extension when updating this path. An older extension
rejects the chunked CE configuration before host registration. Unregister uses
the original chunk size even if the environment changes after registration.

A rejected CE copy drains previously submitted DMA before raising, including
work on peer ranks and earlier per-group streams in the transfer process's
device contexts. If CUDA cannot establish stream quiescence, the
transfer process terminates rather than reporting a recoverable failure that
would let the caller recycle blocks still in use.

```sh
pytest tests/test_host_registration_metadata.py tests/test_host_registration_plan.py
pytest tests/test_ce_chunked_registration_cuda.py
```

The planner properties and mocked registration lifecycle run on CPU. The last
test requires CUDA and checks bytes in both directions, including pitched rows
that cross registrations and per-layer completion. These checks do not replace
model-level output validation.
