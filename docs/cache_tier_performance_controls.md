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

With parallel reads enabled, each Get object's budget cost is `ceil(size / 4096) * 4096 + 8192`.
The configured concurrency is an upper bound. Actual concurrency is reduced when
necessary so that the largest object fits its share of the aggregate budget.
For example, eight 64 MiB objects under a 512 MiB budget use at most seven readers
because padding prevents eight from fitting. Objects exceeding the aggregate
budget are rejected before any read is submitted.

Each task stays within the aggregate budget divided by actual concurrency.
Small reads are distributed across workers by object count; SDK batches also
respect `max_get_batch_bytes`, which retains its independent raw-byte limit.
Completed tasks admit new work without waiting for slower peers, and results
are collected in input order. Concurrency and byte bounds cover all outer
callers sharing a client, whose `batch_get` calls remain serialized.
Serial reads retain the existing raw-byte batching behavior. A single oversized
object fails before SDK submission. Results preserve input order; short reads,
failed reads, and result-count mismatches do not become successful cache hits.
All accepted calls drain before an error escapes. Worker shutdown drains Get
calls before unregistering destination memory regions.

A session waiting for a peer's byte credits remains pending. Initial CPU prefix
pinning also waits when only outstanding I/O makes the pin budget insufficient;
a prefix that cannot fit alongside retained result leases stops immediately.
Chunk reservations take the permanent-capacity fallback when no outstanding
I/O holds credits. Data-session
slots remain occupied until all submitted graphs drain, including cancelled
sessions; finishing I/O does not drop the CPU result/pin lease.

Inline framing combines length and a small serialized object in one fixed-size
message. Larger objects use the fixed header followed by their payload. A leader
serializes once for its fan-out. Legacy and inline framing must not be mixed
within a rank group.

## Validation

```sh
pytest tests/test_mooncake_bounded_io.py tests/test_mooncake_parallel_get.py tests/prefetch/test_coordinator.py tests/prefetch/test_prefix_admission.py
torchrun --standalone --nproc-per-node=4 tests/run_control_framing_distributed.py
```

The SDK tests use controlled thread functions. The distributed test uses real
four-rank CPU Gloo and checks legacy, 256-byte, and 1024-byte framing, including
large payloads and garbage collection while sends are pending. These checks do
not establish live RDMA, GPU/model correctness, or production acceptance.

Model-level validation and measurements must record the exact source heads,
workload, hardware, and matched native GPU-cache baseline. Data-path completion
and throughput do not establish model-output correctness.
