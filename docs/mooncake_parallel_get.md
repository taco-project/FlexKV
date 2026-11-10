# Bounded parallel Mooncake Get

For opt-in threaded channels on the same Store client, see
[parallel whole-object IO](mooncake_parallel_io.md). That mode keeps one
segment and one Host MR. When it is enabled it replaces this per-client GET
thread pool for that client. When it is off, this implementation is unchanged.

The Mooncake-only Get subset of upstream PR #321 is pinned to
`fc1e207dc93cd7545d048de790a511c1b89362d3`. The chunked prefetch coordinator
and inline control-frame options are not part of this port.

The default `FLEXKV_MOONCAKE_GET_CONCURRENCY=1` preserves serial I/O. Values
2 through 8 create a per-client reader pool. `FLEXKV_MOONCAKE_GET_MAX_INFLIGHT_BYTES`
is a per-client aggregate credit, defaulting to 512 MiB. Each object's charge
is `ceil(raw_bytes / 4096) * 4096 + 8192`; a window contains at most the
configured number of SDK calls and never exceeds the credit. Every object
must fit in `credit / concurrency`, otherwise the request fails before any
SDK call. The existing JSON `max_get_batch_bytes` still limits raw bytes per
SDK call and does not change meaning. Query-only clients remain serial.

Results remain in key order and only exact-byte reads succeed. SDK errors,
malformed result counts and executor submission errors drain every accepted
call before returning. some special machine types and scenarios worker drains the reader pool before MR
unregistration and base buffer cleanup. A failed drain retains registration
and allocations for recovery.

For an opt-in trial, set both variables in the environment inherited by the
FlexKV transfer workers, for example concurrency 4 with a 512 MiB budget.
This enables SDK-level overlap; it does not establish an end-to-end HY4 speedup.
Measure the same request stream, cache state, network setup and output tokens
before changing production defaults.

Run CPU contracts without SDK/torch imports:

```bash
python3 -B tests/test_mooncake_bounded_standalone.py
python3 -B tests/test_mooncake_parallel_standalone.py
```

They execute the actual adapter methods and upstream test bodies with controlled
SDK calls. Live Mooncake/RDMA and some special machine types and scenarios numerical/performance acceptance are
separate checks.
