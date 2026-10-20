# Transfer trace and synthetic replay

Set `FLEXKV_CE_TRACE=1` **before starting workers** to record native CPU/GPU
transfer launches. This is off by default and does not change transfer selection
or completion. It covers direct calls, TP thread groups and region batches after
rank sharing and region offsets have been resolved.

```bash
FLEXKV_CE_TRACE=1 FLEXKV_CE_TRACE_FILE=/tmp/flexkv-transfer \
FLEXKV_CE_TRACE_MAX_BLOCKS=0 your-normal-launch-command
```

Each process writes `/tmp/flexkv-transfer.<pid>.jsonl`. The parent directory must
exist. Workers drain their private asynchronous logger during shutdown; programs
calling the native API directly should call `flexkv.c_ext.ce_trace_shutdown()`
after their transfers finish. Shutdown is idempotent and terminal for that process.
The logger uses its own thread pool and never shuts down other spdlog loggers.
Initialize tracing in spawned workers; forking an already initialized logger is
not supported.

A record describes a **launch**, not successful completion. Schema version 1
contains process/thread/device identity, direction, backend (copy engine or SM
kernel), tensor layout, complete layer numbering, byte strides and region/rank
offsets, block IDs, CE configuration, analysis and selected CE path. It contains
no KV contents or pointers. Empty launches are omitted. Apply your usual log
retention rules to this workload geometry and block-ID metadata.

The queue holds 4,096 records and drops its oldest records when full rather than
blocking transfers. `dropped_before` is the cumulative count observed when a
record is built; `ce_trace_shutdown()` returns the final count and reports drops
to stderr. Logging initialization/write errors disable tracing without failing
the transfer. Tracing has CPU/file I/O overhead and is intended for diagnosis.

`FLEXKV_CE_TRACE_MAX_BLOCKS` defaults to 256 IDs per record. Use 0 for all IDs, or
an integer from 1 through 1,000,000 to cap them. Truncated records are inspectable
but cannot be replayed. Missing/dropped launches cannot be recovered.

## Inspect without CUDA

```bash
python -m flexkv.transfer.ce_replay /tmp/flexkv-transfer.1234.jsonl
```

Each line reports a zero-based entry number, direction/backend/path, byte count,
truncation and observed drops. Unknown schemas or malformed records fail with
file and line information.

## Replay one launch on CUDA

```bash
python -m flexkv.transfer.ce_replay /tmp/flexkv-transfer.1234.jsonl \
  --replay 0 --device 0 --max-mib 256
# Compare a CE launch with baseline per-block copies:
python -m flexkv.transfer.ce_replay /tmp/flexkv-transfer.1234.jsonl \
  --replay 0 --device 0 --per-block
```

Replay creates deterministic synthetic buffers, preserves strides, offsets,
block IDs and partial layer ranges, and compares **all CPU and GPU bytes** with a
copy oracle, including untouched bytes. Original application memory is never
read. Raw-pointer tensor owners remain live until CUDA has been synchronized,
including when a native call raises.

Allocation is bounded before CUDA initialization. The budget includes buffers,
expected values, readback and ID/pointer tables. Negative or unaligned geometry,
overflow, truncated IDs, overlapping destination writes, more than one million
copy spans and forced benchmark paths are rejected. Increase the budget only
for a trusted capture whose geometry needs it.

The recorded CE policy is rerun on the chosen device; `--per-block` changes only
the CE policy. Output reports correctness and one synchronized wall-clock
duration including launch overhead. This is not a throughput benchmark or a
replay of distributed scheduling, cache lifecycle, compression, eventfd timing,
RDMA, SSD I/O or model outputs.

The direct native binding accepts optional trailing `gpu_startoff_inside_chunks`,
`cpu_startoff_inside_chunks` (bytes, default 0) and `total_num_layers` (default 0
preserves the old `num_layers` behavior). Replay supplies the total so SGLang's
separate K/V pointer arrays retain correct indexing for partial layer captures.
