# HY4 NEXTN/MTP + Mooncake L3

This implementation pairs the HY4 SGLang 0.5.10 adapter with this some special machine types and scenarios branch.
It preserves the CE/GDR implementation and the normal-memory 64 GiB MR split.
It does not replace either repository with upstream main.

## Data and ownership

`prefetch_async` stages REMOTE2H into Host before admission. Foreground GET
continues to use Host only (`ignore_remote=True` for Mooncake), then performs
synchronous or layerwise H2D. L3 writes remain asynchronous H2REMOTE after D2H.
No per-request launch ACK is added.

The scheduler polls the entire prefetch ledger once per scheduling round,
including the separate PD-prefill entry. Only the leader calls `try_wait`;
followers enter the same progress scatter regardless of their local clocks.
An empty ledger does no RPC/scatter. Actual post-commit `return_mask` determines
materialized rows. L3 metrics intersect this span with the Host rows actually
restored; device-tree hits are not attributed to L3.

Cancellation/timeout only detaches a request. The backend writer retains its
Host reservations until a real terminal. Detached tasks count toward the cap.
Reset consumes every drained ID before reporting another still-live writer;
it never clears ownership after a timeout. Immediate external RID reuse gets
a new attempt key and cannot be cancelled by an older Req's completion.

## Bigram and draft pools

Like HiCache, N bigram slots include N+1 raw tokens in their identities. Each
uint32 token pair is packed injectively into an opaque int64 for FlexKV's
existing native hash API; all bits, including signed int64 values, are hashed.
Store, lookup and prefetch use identical encoding and namespace. The model's
input IDs and KV bytes are unchanged. Token IDs outside uint32 fail loudly.

HY4 NEXTN has an independent one-layer KV pool and Indexer, sharing the target
allocator's slot IDs. They are two extra heterogeneous groups in the SAME Host
block as the target pools, not best-effort separate L3 objects. A short read of
any part rejects that block. Pool row dtypes/widths can differ; inline scales
are copied as opaque bytes. Missing or incompatible draft pools fail at startup.
Multi-layer EAGLE and DSV4/SWA draft layouts are not supported by this adapter.

With physical CP layer split, draft is replicated whereas target layers have
one owner. Each CP owner has its own two draft Host rows. Each row is associated
with that owner's first target layer; per-rank GPU handles alias the physical
draft tensor. Existing owner routing copies one row per rank, and existing
layer-member completion fences include these groups before consumption. This
avoids a second KVManager, a second control plane and extra per-load barriers.
Without owner routing there is one draft row per group, using the existing
replicated MLA policy. Extra bytes are included in Host capacity calculation.

A versioned key namespace invalidates previous boundary-dropping entries. A
layout fingerprint separates draft/target geometry, dtype, sparse layer IDs,
page size, CP holder layout and model path; startup checks its agreement across
ranks. The first deployment is a cold cache. Changing weights at the same model
path still requires an application cache namespace/version or storage reset.
Different physical block layouts intentionally do not share L3 entries.

## Controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLEXKV_ENABLE_L3_PREFETCH` | `1` | Used only with Mooncake enabled; `0` keeps Host caching and MTP |
| `FLEXKV_L3_PREFETCH_SKIP_ON_L1_HIT` | `0` | Opt-in read-only L1 GPU hit check before launching `prefetch_async`; `0` avoids L1 eviction TOCTOU while queued |
| `FLEXKV_L3_PREFETCH_MAX_INFLIGHT` | `64` | Per connector, includes detached writers (sized for high concurrency so bursts do not bypass L3 prefetch into L2-only lookup) |
| `FLEXKV_L3_PREFETCH_POLL_INTERVAL_S` | `0.01` | Leader backend-RPC cadence; progress scatter remains ordered |
| `FLEXKV_L3_PREFETCH_WAIT_S` | `1.0` | Leader-authoritative admission wait; timeout detaches, does not stop DMA |
| `FLEXKV_MOONCAKE_GET_CONCURRENCY` | `1` | Optional bounded readers, `2..8`; measure before enabling |
| `FLEXKV_MOONCAKE_GET_MAX_INFLIGHT_BYTES` | `536870912` | Per-client padded-byte budget for parallel Get |

Queued L3 supports PP=1 on this SGLang version. Multi-rank TP/CP requires
collective sync. Unsupported combinations fail at startup. The ordinary
SGLang and HiCache Python paths remain the prior AST when FlexKV is disabled.
The native HY4 launcher still defaults to HiCache; opt in with `enable_flexkv=1`.

## Local validation and hardware acceptance

The workspace CPU runner is `review/upstream-integration-20261103/run_cpu_suite.py`.
Tests execute actual Python methods, wire encoding, layout/routing helpers,
C++ `hash.cpp` plus xxHash, and actual Mooncake adapters with a memory-copy SDK
double. They cover bigram page boundaries, byte-exact inline-scale payloads,
CP2/8/16/32 membership, mixed draft dtype, partial submission, timeout,
cancellation, reset, RID reuse and unchanged non-FlexKV scheduler paths.
This does not exercise CUDA IPC, some special machine types and scenarios kernels, RDMA or model numerics.

Before production acceptance, run both synchronous and layerwise CE/GDR with
NEXTN enabled, CP off/on and layer split on. Compare outputs/logprobs against
cold recomputation, including equal prefixes with different page-boundary
tokens, media namespaces, logprob truncation, chunked prefill, cancellation and
request retraction. Prove L3 reads by evicting L1/L2 while retaining L3 and
observing REMOTE2H -> Host -> H2D; a device hit or successful H2REMOTE is not
proof. Verify actual restored draft bytes, health, long-running ownership and
shutdown. Measure concurrency 1/2/4/8 under identical inputs, CPU capacity and
load; report TTFT/TPM, Host pressure and RDMA route separately.

Streaming-session cache ownership is unsupported and rejected before connector
startup. HY4 target/draft piggyback currently also requires PP=1.

## Extended local CPU validation (2026-11-03)

The follow-up runner is `review/extended-cpu-20261103/run_cpu_suite.py`.
It retains the original 321-case evidence and adds the following tests:

- Execute 37 existing Mooncake deferred-insert/deferred-PUT, prefetch/compute
  and worker-result test bodies. Imports requiring CUDA are isolated; the
  actual Python radix mirror, mempool, global cache engine and native hasher
  execute. The remote double's missing reset is supplied by the real backend's
  no-op reset. No assertion is removed.
- Run the complete connector constructor concurrently, with real ModelConfig,
  RankInfo, layout, capacity recomputation and hierarchical communication.
  GPU handles, KVServer, signal/atexit and eventfd endpoints are doubles.
- Stage requests through the actual SGLang mixin, connector, task engine,
  cache tree and Mooncake adapters. The full D2H -> L3 -> Host eviction ->
  REMOTE2H -> Host publication -> H2D roundtrip checks byte-exact opaque data.
  Real client/server request methods also run over pickle queues in one owner
  thread, including no-ACK rejection and accepted-send exceptions.
- Run actual FlexKVComm methods concurrently across DP/PP/CP/TP replicas,
  including permuted global ranks, slow peers, error propagation, reset and
  sustained asynchronous-work reaping. This found and repaired rank-0-based
  CP/PP routing that excluded a second DP replica. Routing now uses actual CPU
  group membership; the PP+CP+TP stage map uses one startup object gather.
  No new per-request collective is added. Disabled synchronization bypasses
  that startup gather and communication methods.
- Exercise 1,200 seeded publication/abort combinations, including 200 with
  concurrent callbacks under the real cache lock. Check ready-prefix bytes,
  unique physical ownership, no unready residue and all slots free after reset.
- Compile actual native rank participation and submit_layerwise scheduling for
  CP2/8/32 with target, sparse Indexer and draft groups. Delayed memcpy doubles
  stand in for CE/NIC writes; assertions cover aliased draft rows, untouched
  nonowner scratch, all-member readiness, and failure disarm/drain. This is
  native scheduling coverage, not an execution of CUDA/CE/GDR kernels.

The communication fix is folded into the existing multi-node/topology feature;
the other tests extend this L3 feature. Repeated stress runs are reported
separately and are not added to the unique-test total. Real Gloo, ZMQ, GPU IPC,
some special machine types and scenarios numerics and Mooncake RDMA remain outside these CPU tests.

## Serving-order and Mooncake byte stress (2026-11-03)

`review/serving-l3-stress-20261103/` extends the CPU evidence with actual
SGLang finished-request writeback, lock release, L1 and Host eviction,
queued prefetch and synchronous match/H2D. The complete connector constructor
and KVManager forwarding methods execute. Another suite runs the same path
through real KVDPClient/KVServer request handlers over pickle queues, with
one server owner thread; only the transport and worker-progress endpoints
are replaced. No inference, HTTP server, Gloo or ZMQ process is started.

Boundary tests include zero/subpage/exact/adjacent lengths, 128K and 1M
boundaries, uint32 extremes and signed bigram identities, namespace isolation,
fragmented Host rows, reversed GPU pages, Host-hit/remote-tail and GPU-prefix
masking. Long cases retain full real token/native hash chains with compact
payloads; separate smaller cases retain the existing HY4 opaque block size.
They do not allocate a real million-token HY4 KV tensor.

The byte SDK double checks registered address ranges and 33-byte sentinels,
mixed object sizes through 1 MiB+1, byte batch limits, ordered results,
2/8/32 concurrent callers, 1/2/8 readers, partial PUT and failed/short GET,
retry, shutdown and no writes after reader drain. Parallel Get padded credits
are checked separately from the serial raw-byte batching contract.

This path found a SGLang NEXTN writeback API mismatch: the connector exposes
`is_store_sync_leader`, not `is_sync_leader`. It also found accumulated terminal
response masks on empty lookups and fully cached PUTs. The connector now retires
these no-writer tasks using the existing no-ACK cancel message. Successful
transfer paths add no message; miss/no-op paths add one small leader-local
control send, without an extra reply or data transfer. An uncertain retirement
send is not proof of an active writer; the no-op PUT error fan-out uses task ID
-1 so followers do not materialize or lock GPU sources. Public get_match/put_match
response semantics and active writer drain remain unchanged.

The remaining real hardware, SDK, model numerics, long-running service and
performance acceptance requirements above still apply.

## Two-node CP16 and PD serving validation

The current HiCache path uses the global CP group for layer ownership across
nodes, local Host layer indices, and per-CP-rank L3 keys. The FlexKV adapter
now accepts the equivalent HY4 topology: launcher TP16 with CP16, two nodes,
attention TP=1, DP=PP=1, physical layer split, and one-layer NEXTN. The supplied
`native_for_v21/run_p.sh` remains a single-node TP8 launcher; its hardcoded
node count and TP/EP sizes do not constitute a two-node deployment recipe.

Keep the connector and remote-manager configuration global. After GPU
registration, each node derives a separate local storage/transfer model from
the actual contiguous global CP registration IDs. Pack only its owned target,
sparse Indexer and per-owner draft rows, preserving each group's layer order.
The 78-layer logical event space stays global. Off-node layers have no local
members and receive empty-work notifications; the existing framework CP
broadcast waits for the owning rank. Local holder indices are 0..7; -1 is
reserved for off-node layers and is never a DMA request destination.

For page=64 and this HY4 layout, node 0 stores groups [40,11,8,8], or
2,175,744 bytes/block; node 1 stores [38,10,8,8], or 2,083,328 bytes/block.
The shared slot capacity uses the larger local block size, without padding
each smaller-node block. The single-node CP8 block remains 3,855,616 bytes.
L3 keys include `_cpnode_{node}_of_{nodes}`. One block-major batched existence
query requires all node fragments for a hit; publication waits for all node
transfer results and intersects their exact-read masks. Layout fingerprints
also apply when NEXTN is disabled. No per-request collective is added.

PD failure handling follows HiCache's request-level policy. The shared
scheduler catches the sender error, releases the request through the normal
cache lifecycle, reports its abort and continues other requests. FlexKV adds
no sender-Failed exception, service restart, per-request collective or DMA
wait. This preserves the existing PD transport contract; it does not add a
new sender-drain guarantee. Independent FlexKV H2D/D2H/L3 writer ownership,
failure propagation and terminal-drain requirements remain unchanged.

Workspace evidence: `review/pd-session-stress-20261103/`. It includes 489 CPU
tests; 48 isolated scenarios with 200 sessions each, 3..40 turns and client
concurrency 10/16/32/64/96/128; four extra scheduling-seed scenarios; and 21
critical-suite repeat runs. P admission retains the launcher's cap of 16.
Synthetic variable transfer timings cover H2D/D2H/L3/P-to-D, timeout/detach,
finite GPU-page pressure and cache eviction. Exact opaque bytes and an
independent deterministic response oracle are checked. This is one logical
scheduler with per-rank sender endpoints, not a real distributed serving or
HY4 numerical-accuracy benchmark. Real CP16 CUDA IPC, Gloo/NCCL broadcast,
some special machine types and scenarios DMA, Mooncake SDK and worker-group failure teardown remain untested.
