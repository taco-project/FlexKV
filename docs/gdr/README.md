# GDR GPU/CPU transfer feature

GDR is optional and disabled by default. It is owned by `RegionBatchGroup`; no SGLang changes or process-global verbs engine are required. Configuration lives in `flexkv/common/gdr_config.py`; queue/registration code is in `csrc/gdr/transport.*`, and the existing region interface is adapted by `csrc/gdr/runtime.*`. The adapter uses generic `kv_dim` and tensor strides, without model-specific `is_mla` branches.

## Build and configure

On the NVIDIA GPU / RDMA server, with the project's existing build dependencies installed:

```bash
cd FlexKV-PR/FlexKV
# Also needs rdma-core development headers/library, e.g. libibverbs-dev,
# and working GPU peer-memory registration (nvidia-peermem), with sufficient memlock.
git submodule update --init --recursive
# setup.py uses build/include and build/lib; populate them before pip install.
cmake -S . -B build -DFLEXKV_ENABLE_MONITORING=OFF
cmake --build build -j
FLEXKV_ENABLE_METRICS=0 FLEXKV_ENABLE_GDR=1 python -m pip install -v --no-build-isolation -e .
python -c 'from flexkv import c_ext; print("GDR compiled:", c_ext.gdr_compiled())'
```

Without `FLEXKV_ENABLE_GDR=1`, the extension does not link libibverbs; enabled GDR then obeys its initialization-failure policy. GDR requires NVIDIA CUDA >= 11.3 and a GPU/driver capable of establishing RDMA write visibility. The NIC must have an active port 1; Ethernet requires a nonzero RoCE GID with a known v1/v2 type. Like 134ad747, initialization tries non-link-local v2 GIDs first, then link-local v2 and v1; failed QP pairs are destroyed before retry. Workers must use the exact endpoint selected by the allocation owner. `nic_names` can select NICs explicitly when sysfs locality detection is unavailable.

Merge [the annotated YAML](../../tests/gdr/config.yaml) into your existing FlexKV configuration. `gdr.enabled: true` defaults both directions to `auto` / `avgbytes`. H2D and D2H have separate configuration and native selection functions. Their initial formulas are the same; neither direction reads the other's settings.

| Direction `path` | Behavior |
|---|---|
| `original` | Preserve the existing `use_ce_transfer_*` decision and existing CE subpath logic. |
| `ce` | Select the existing CE backend (including its own gather/segment logic). |
| `kernel` | Select the existing SM copy kernel. |
| `gdr` | Force GDR, bypassing performance thresholds. |
| `auto` | Choose GDR using this direction's selector; otherwise use `original`. |

`avgbytes` selects GDR when average contiguous-segment bytes are **strictly below** `gdr_when_ce_avg_bytes_below` and the block fragment is **at most** `gdr_max_block_fragment_bytes`. `segments` instead requires at least `segment_threshold` segments and the same fragment-size bound. These are heuristics inherited from the source implementation, not universal bandwidth crossover points. Neither threshold means “wait until enough bytes arrive.” The former ambiguous names `min_bytes_per_call` / `max_chunk_bytes` are not accepted configuration keys.

`use_dedicated_gdr_thread` adds one persistent GDR execution thread per local GPU, beside the original transfer thread. `batch_requests` is the H2D coalescing limit: 1 disables it; 4/8/16 replace separate gdropt/gdrbatch8 names. Only compatible, already-received layerwise operations sharing a counter are combined. Control messages, different counters, destination aliases and block limits form boundaries. There is no batching sleep. Original operation IDs and success/failure results remain distinct; each layer produces one semaphore token per original operation. A received set can coalesce even if `auto` eventually chooses an original backend. SGLang's compute batching neither guarantees nor prevents these operations from arriving together.

## Correctness and performance boundary

The data path now follows **134ad747's direct owner-target design**. The allocation owner registers the original KV storage and exports address/rkey metadata through FlexKV's existing tensor handles. Each worker establishes its own RC sender QPs against owner target QPs, maps imported aliases to owner addresses, and never registers an imported CUDA IPC alias. H2D RDMA WRITEs land in final KV; D2H RDMA READs source final KV. **There is no GPU staging allocation, gather/scatter kernel, or extra GPU-local payload copy.** CUDA context/stream resources still exist, and NIC accesses still consume PCIe/HBM bandwidth.

`flexkv/transfer/gdr_owner.py` contains the private owner handshake and leases. Owner tensors remain alive while registered. Endpoint checks include GPU PCI identity, NIC, port, GID, MTU and READ depth. Before the first eligible D2H operation in a received worker batch, the worker requests a producer fence in the original allocation process. A failed exchange poisons subsequent transfers rather than accepting a stale ACK. It is intentionally not a worker-context CUDA synchronization. This conservative owner-wide fence has a cost and should be included when interpreting end-to-end results.

One fence per received batch is sound rather than a sampling shortcut, and the reason is worth stating because it is the only thing standing between an RDMA READ and a half-written block. The run loop drains the pipe completely before `_prepare_received_batch` and before the first launch, so every op in the batch had already been sent when the fence runs; the caller cannot ask FlexKV to save KV whose writing kernels it has not launched yet; and the owner's device-wide synchronization waits for everything launched on that device, including work still queued in the driver. `maybe_fence_d2h` enforces that premise instead of trusting it: it stamps the time at which it sends `R` and takes a fresh fence for any op whose `_received_ns` is later, so a save submitted from the forward path rather than from result processing cannot ride a fence that predates it. Coalesced batches report their latest child's arrival time for the same reason. Adding a `recv()` after the fence, or reordering the drain below it, would turn every batch into an extra owner-wide round trip.

The runtime prepares at most 256 spans at a time and posts bounded WR chains across the configured QPs, continuing descriptor preparation while earlier DMA is in flight. SQ/CQ occupancy is bounded; reusing exhausted queue capacity drains outstanding completions. Adjacent spans merge only when both final GPU and host addresses are contiguous. H2D completes the required RDMA write visibility operation before layer-ready notification. D2H completes its reads before reporting host data ready. The dedicated execution thread and configurable H2D coalescing remain available.

The original producer/block ownership contract still applies: buffers and IDs remain live until completion, and KV destination blocks must be safe to overwrite. A local stream event cannot by itself synchronize another process. `GPUCPUTransferWorker` drains and destroys sender resources before closing leases or releasing imported mappings/host registrations. On GPU suspend, the client waits for the worker barrier and then closes owner MRs before releasing exported VMM handles. Resume creates fresh endpoints and mappings. Owners retain registrations for the registration lifetime; there is no automatic eviction of live MR-backed allocations.

### What the caller must guarantee, and why FlexKV cannot

Three ordering edges cross the process boundary, and CUDA events do not. Nothing here is GDR-specific — CE and SM-kernel transfers read and write the same memory from the same worker process — but GDR makes the edges harder to see, because the NIC touches KV outside CUDA stream ordering entirely and never appears in a profile.

**H2D write-after-read.** A destination GPU block must already be free of readers when the operation is submitted. FlexKV has no way to express "wait until the owner's attention kernels are done with this slot": no stream event it can record covers another process's compute stream, and for GDR even a local `cudaStreamWaitEvent` would not hold the NIC back. The SGLang integration satisfies this through allocation lifetime, not stream ordering — `init_load_back` takes fresh slots from the allocator before launching anything, running requests keep their slots allocated and their prefixes radix-locked, and slots return to the free list only from `cache_finished_req`, which SGLang reaches after synchronizing on that batch's completion event. That host-side synchronization is the happens-before edge; the allocator is what carries it. Any change that recycles a slot outside the synchronized result path, or reloads into a slot the current forward is still reading, breaks every backend at once.

**D2H read-after-write.** Symmetric, and additionally backed by the owner producer fence described above. The source blocks must also stay alive for the whole transfer; the integration holds a radix lock until `check_completed_stores` drains the task.

**RDMA write visibility.** See the XDR section below. On NVIDIA this is the documented device-scoped flush; on some special machine types and scenarios it is a NIC-side read-back plus the XMLIR device synchronization.

`on_init_failure: original` permits fallback for missing owner metadata, failed registration or endpoint setup; `error` fails an operation that selects GDR. There is **no staging fallback**. New host-region registration errors fail the operation. Once posting begins, uncertain completion (partial post failure, CQ error/timeout, failed write visibility) terminates the worker rather than reusing potentially live buffers. A normal preflight exception drains already-posted chunks before reporting failure.

This is a migration of the direct address/registration/completion design into the PR's instance-owned region interface, not a wholesale copy of 134ad747's global runtime and experimental options. Hardware performance and cross-process GPU correctness must be verified on the deployment server; CPU contract tests do not establish a physical bandwidth ceiling.

GDR operates on uncompressed GPU/CPU region-batch transfers. It may coexist with Mooncake Store L3 (CPU<->REMOTE), local SSD, GDS, and NIXL GDS_MT (GPU<->SSD): these routes use independent workers and do not replace the GPU/CPU worker. GDR transports own their PDs, CQs, QPs and MRs; Mooncake uses its own registrations and connections, including when both use the same NIC. No exclusive NIC reservation or shared QP is required. Hardware resource limits still apply.

Existing backend requirements remain in effect: GDS requires SSD, NIXL requires GDS and effective TP=1, and GDS cannot be combined with KV sharing. Mooncake REMOTE2H remains prefetch-only; prefetch must complete into CPU before compute H2D, while writeback reaches H2REMOTE after D2H completes. Enabling GDR does not add compressed GDR transport or change these scheduling rules. Defaults and APIs used when GDR is disabled retain original behavior.

`PHASE=unit` runs the CPU-only coexistence regression (`tests/gdr/test_backend_coexistence.py`) together with the native contracts. It checks configuration combinations, remaining GDS/NIXL/KV-sharing guards, GDR attach on the GPU/CPU worker, and production worker routing with mocked worker processes; validate concurrent data correctness and resource availability on the deployment hardware as well.

Resource identifiers are not reserved by GDR: `qp_count` is a count, and lane/rank indices are local bookkeeping. Local QPNs come from `ibv_create_qp` / `ibv_create_qp_ex`; peer QPNs come from the owner handshake. Lkeys/rkeys come from each transport's own `ibv_reg_mr` registrations. Port 1, P_Key index 0, the selected GID and initial PSN 0 do not reserve QP numbers. Completion IDs are scoped to a private CQ, and the control socket lives in a unique `flexkv-gdr-*` temporary directory. GDR neither changes NIC/GID configuration nor binds Mooncake's TCP ports. Its `ibv_fork_init` call is process-wide initialization, not a resource reset.

The native `backend_resource_isolation` contract preallocates another backend's QP/MRs on the same modeled NIC, registers the same CPU buffer and the same GPU pages (Mooncake-like host MRs plus GDS/NIXL-like GPU MRs), and leaves both of its transfers pending on its own CQ while GDR initializes, grows its MR, transfers and shuts down. It checks that GDR only posts on its allocated sender QPNs, uses a distinct GPU rkey, ignores a foreign QPN in `release`, preserves the external registrations/completion, and isolates initialization-failure cleanup. This runs against production transport code with deferred verbs under ASan/UBSan, including the extended-WR and XDR builds; it does not substitute for a real Mooncake SDK or cuFile/NIXL hardware run.

## Tests and output

No-GPU contract tests compile the production runtime/transport against deferred CUDA and verbs models, with address/undefined-behavior sanitizers. They cover layout/kv_dim combinations, rank offsets, remote-address translation, reordered completions, bounded asynchronous queues, init fallback, fail-closed errors, batching limits, alias barriers, result fanout, and a real Unix-socket lease handshake. The native model also asserts zero CUDA payload copies and zero GPU data allocations.

Three of them pin cross-process ordering specifically. The XDR build asserts that every write epoch ends with exactly one read-back per QP that wrote, that a read-only epoch posts none, and that a read-back which cannot be posted aborts (`contracts flush`) just as a failed device synchronization does (`contracts visibility`). `Context::launch` rejects a negative `gpu_block_stride_in_bytes`, because the in-batch conflict envelope brackets a request with the addresses of its smallest and largest block ID and that is only a superset while `ptr_at` grows with `block_idx` — a contract now stated at the `ptr_at` declaration itself. On the Python side, `maybe_fence_d2h` is exercised for the case the batch flag alone would miss: an op that reaches the worker after the fence started takes a fresh one, and a coalesced op reports its latest child's arrival rather than its first child's.

One further contract guards a test rather than the runtime. `tests/gdr/test_visibility.py` (below) is only meaningful while nothing sits between the consumer's `eventfd_read` and its reading kernel, and adding a `synchronize` or a temporary allocation there is a one-line change that no CPU test would notice and that would quietly make the gate pass on a machine with no flush at all. So the contract suite parses that file and asserts the loop body still calls nothing but `torch.ne`, `range`, `max` and `time.perf_counter_ns`. Everything in that test above the device boundary is plain array work, so the suite also runs it: the markers must never repeat back to back nor collide with the untouched-byte value, and the failure classifier is driven through a numpy stand-in over synthetic late/stale/unwritten/torn buffers. That last one matters because the classifier's whole job is to distinguish a byte that arrived late from a byte that is simply wrong, and the only machine that can produce a real failure is the worst place to discover the report is broken.

```bash
PHASE=unit bash tests/batch_run_gdr_modes.sh
```

Hardware smoke first runs `tests/gdr/test_direct_ipc.py`: a separate allocation-owner process exports actual CUDA IPC handles, followed by forced direct H2D/D2H, per-layer validation, owner-context producer synchronization, lease/MR release and a second owner generation. Only then does it run the five-arm comparison:

```bash
TP=8 PHASE=smoke bash tests/batch_run_gdr_modes.sh
TP=8 PHASE=compare DIRECTION=both bash tests/batch_run_gdr_modes.sh
```

The microbenchmark itself uses explicit **same-process owner registration**, so it isolates transfer throughput; the separate IPC gate verifies the cross-process protocol. `CHECK_IPC=0` skips that gate only when intentionally requested.

The five arms are `CE`, `kernel`, `overallperf-origin`, `GDR`, `overallperf-withgdr`. `overallperf-origin` disables GDR and passes the original directional CE flags unchanged. It does not introduce another CE/kernel heuristic: with the original default flags, it can coincide with `kernel`. `overallperf-withgdr` adds the `auto` decision. The test forces initialization failures to error so a failed GDR arm cannot be mistaken for a successful comparison.

Examples for direction, layouts, batching and dedicated-thread comparison:

```bash
TP=8 DIRECTION=H2D BATCH_REQUESTS=8 bash tests/batch_run_gdr_modes.sh
TP=8 DIRECTION=D2H QP_COUNT=4 bash tests/batch_run_gdr_modes.sh
TP=8 PHASE=smoke KV_DIM=2 SHARED_KV=0 LAYOUTS=LAYERFIRST,BLOCKFIRST \
  NOTIFY_MODE=polling bash tests/batch_run_gdr_modes.sh
for batch in 1 4 8 16; do
  TP=8 DIRECTION=H2D ARMS=GDR BATCH_REQUESTS=$batch \
    bash tests/batch_run_gdr_modes.sh
done
TP=8 DIRECTION=H2D USE_DEDICATED_GDR_THREAD=0 ARMS=GDR \
  bash tests/batch_run_gdr_modes.sh
# Detailed logs are optional; use a separate run for diagnosis.
TP=8 PHASE=smoke DIAGNOSTICS=1 bash tests/batch_run_gdr_modes.sh
```

Other controls: `BLOCKS=32,256`, `CONCURRENCY=1,8`, `FRAGMENT_BYTES=8192`, `LAYERS=8`, `SCATTER=0|1`, `QP_COUNT=1..16`, `SELECTOR=avgbytes|segments`, `NIC_NAMES=mlx5_0,...`, `ROUNDS`, `WARMUP`, `OUTPUT`, `PYTHON`. The main and smaller SWA-like regions exercise different layer counts. H2D verifies GPU bytes upon individual layer notification, as well as full buffers including untouched blocks. D2H verifies the full host buffer. Verification and route probing are outside timing.

Each output directory contains:

- `direct_ipc.log` (smoke), `run.log`, `topology.txt`, `metadata.json`: progress, hardware/source/config context.
- `raw.jsonl`: incrementally saved cases, actual route-probe counts, byte counts, samples, configuration and physical submission count; retained if interrupted.
- `summary.csv`, `summary.md`: completed-run comparison.
- `failure.json`: the failing case, if an exception occurs.

`median_ms` / `p95_ms` measure the complete ready-request set, including native WQE preparation, submission and completion, but excluding Python plan construction. `prepare_median_ms` and `with_prepare_median_ms` also expose that cost. This is a transfer microbenchmark using the production coalescer/native interface; it is not SGLang end-to-end latency or a measurement of requests accumulating in its IPC pipe. `aggregate_GB_s` sums participating GPU transfers: shared-KV H2D broadcasts to every rank, whereas sharded D2H writes each byte once. It must not be interpreted as the bandwidth of a single PCIe link. `logical_GB_s` measures logical KV payload progress.

Normal integration defaults both GDR diagnostics off. To suppress routine FlexKV logs as well, set `FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0` in the server environment. The test wrapper does this by default; `DIAGNOSTICS=1` enables INFO summaries and stage timing. Direct-GDR submit timing includes CQ waiting and final write visibility. It excludes the Python owner-fence exchange; use an end-to-end worker benchmark to measure that exchange. Raw rows identify `implementation=direct-owner-v1` and `gpu_staging_bytes=0`; previous staging results are a different implementation and should not be mixed with these rows.


## TP and thread communication

Payload bytes already stay in pinned host memory and final GPU KV storage. Native rank tasks share an immutable request plan and block-ID arrays; each rank resolves its own addresses and builds WQEs. A layerwise engine message contains main/SWA source/destination IDs, not KV payloads. PP sibling replication is separate from the local TP thread fan-out. Replacing these arrays with compressed runs could save IPC bytes for long contiguous ranges, but scattered cache blocks require many runs and extra decoding; no new compression/shared-memory lifetime protocol is introduced without measurements.

With `path: gdr` and a usable endpoint, a batch containing only GDR requests now makes **one dedicated-thread handoff per participating rank**, rather than one per region/layer. That thread walks the shared plan, starts bounded WQE chunks immediately, and records the same per-layer readiness markers after RDMA completion. There is one local stream dependency at batch entry, no per-region CUDA dependency event, and no per-layer round trip to the original thread. The original batch submission also no longer launches a second TP fan-out solely to join GDR jobs. Forced GDR selection skips the unused contiguous-segment scan; transfer bounds/ID validation remains in execution.

`auto`, mixed backend batches, and initialization fallback retain the ordered original-thread dispatch. This deliberately avoids moving CE/kernel launches to a different thread or adding a full descriptor-preparation barrier before the first transfer. `use_dedicated_gdr_thread: false` remains available: for very small transfers, even a single handoff and the extra CPU threads can cost more than they save.

Route-probe output includes `dispatch.batch` and `dispatch.region` counts (not payload bytes). With timing enabled, `[GDR-DISPATCH]` separates queue wake-up time from the local producer-stream fence; `[GDR-DIRECT-PRODUCER]` measures the separate cross-process D2H owner fence. Producer requests are sent to all owners before waiting for acknowledgments. A slow owner/rank can still delay completion, and owner-wide CUDA synchronization can interfere with overlapping compute. Removing that fence requires an allocation-owner producer-event contract; a small message alone cannot replace it safely.

For a short CPU/TP-scaling comparison, run on the GPU server (logs remain quiet):

```bash
for tp in 1 8; do
  for dedicated in 0 1; do
    TP=$tp USE_DEDICATED_GDR_THREAD=$dedicated ARMS=GDR DIRECTION=both \
      BLOCKS=8,256 CONCURRENCY=1,8 LAYERS=64 ROUNDS=5 \
      OUTPUT=result_gdr_dispatch_tp${tp}_thread${dedicated} \
      bash tests/batch_run_gdr_modes.sh
  done
done
```

Compare small/large transfers separately using latency, aggregate/logical bandwidth and preparation time. TP=1 versus TP=8 changes physical traffic and memory/NIC contention as well as dispatch cost; it is not an isolated IPC benchmark. Run a separate `DIAGNOSTICS=1` sample for queue/fence timing; do not use verbose timing runs as the headline performance result.


Initialization/registration follow `flexkv_134ad747/csrc/gdr/backends/verbs_transport.cpp` and `flexkv_134ad747/flexkv/gdr/owner.py`: once-per-process verbs fork setup, candidate GID selection, fresh QPs on connection failure, INIT sender before owner handshake, endpoint validation, owner-only GPU registration with SYNC_MEMOPS, and sender drain/destruction before lease/MR release. The PR retains instance-owned transports instead of copying the old global engine table. Conflicting imported alias ranges fail before connection; consistent overlapping views remain valid. Legacy CUDA IPC additionally preserves the CUDA-allocation-relative offset (including allocator suballocations), fixing the zero-offset assumption also present in the reference. These changes do not alter WQE posting, queue depths, request coalescing, path selection or dedicated-thread scheduling.

### Automatic D2H ownership for GDR

`gdr.d2h.auto_layer_parallel: true` is the default. Before rank fan-out,
shared-KV D2H requests with default `sharded` ownership may be converted to
`layer_parallel`. Conversion requires all ranks' direct registrations/handshakes
to be usable, equal per-rank chunk sizes, backend AUTO, and the complete chunk
(after removing byte sharding) to select GDR under the D2H policy. This last
check keeps auto thresholds meaningful: if a full chunk exceeds the GDR size
ceiling, the request retains its original sharded routing. Requests with an
explicit CE/kernel backend, nonshared KV, H2D, disabled GDR or unavailable ranks
keep their original ownership. Initialization fallback therefore cannot create
a mixture of layer and byte ownership within one request. Existing explicit
non-sharded ownership is preserved.

Set `gdr.d2h.auto_layer_parallel: false` to retain caller ownership. The worker
also honors an explicitly set `FLEXKV_KV_SHARED_ACROSS_RANKS_MODE` by disabling
the automatic override. `tune.py --share-mode ...` and `probe_d2h.py --share-mode
...` likewise disable it, so sharded/layer-parallel ablations remain genuine.
No GDR-disabled CE/kernel routing or H2D layer notification behavior changes.

### D2H shared-KV ownership and descriptor lookup

For worker/SGLang integration, set
`FLEXKV_KV_SHARED_ACROSS_RANKS_MODE=layer_parallel` before starting the process.
This uses the existing D2H layer partition only for identical KV replicas
(`num_kv_heads == 1`). H2D and head-sharded KV are unchanged. Each rank transfers
whole chunks for its assigned layers; few layers or uneven main/SWA layer counts
can limit balance. Explicit ownership still defaults to `sharded`; the GDR-only
automatic override below may repartition eligible requests.

The direct benchmark does not read this environment setting. Use
`tests/gdr/tune.py --share-mode layer_parallel ...` or
`tests/gdr/probe_d2h.py --share-mode layer_parallel ...`; both override the
request field explicitly for all comparison arms. The original workload scripts
remain unchanged. The selected mode is recorded in tuning/probe metadata.

Transport resolution now uses a post-local host MR hint and a checked fast path
for the last owner mapping. Every actual span still undergoes bounds checking,
including scattered/nonmonotonic addresses and registration crossings. The whole
chunk is resolved before any of its WQEs are posted. Layer/KV host offsets are
computed once outside the block loop. Completion, credit, visibility and H2D
per-layer notification rules are unchanged. This removes repeated work without
adding persistent descriptor caches or preparation threads; GPU performance must
still be measured.

### DeepSeek V4 Flash geometry in the transfer comparison

Pass `--profile dsv4-flash` to `compare_modes.py`, `tune.py`, or `probe_d2h.py`.
It fixes the model at 43 layers, KV dimension 1, shared KV, and 256 tokens/page,
following `test_dsv4_flash_concurrency_e2e.py::Dsv4Geometry` and its default YAML:

| Region | Global layers | Bytes per layer per page |
|---|---|---:|
| c4 | 2,4,...,42 | 37440 |
| c128 | 3,5,...,41 | 1728 |
| c4_indexer | 2,4,...,42 | 8448 |
| swa_kv | 0,...,42 | 149760 |
| c4_attention_state | 2,4,...,42 | 65536 |
| c4_indexer_state | 2,4,...,42 | 16384 |

KV pages include 576-byte alignment, the indexer uses 132 bytes/token,
and the two state buffers use an 8-entry float32 ring. H2D maps local region
layers back to global layers before signaling readiness. D2H submits all six
regions; `--share-mode` in the tuning/probe wrappers applies to each region.
`--layers` and `--fragment` do not change this fixed profile. The default
`synthetic` profile retains the old geometry. Metadata and raw results record
all region layer IDs and byte widths.

One block totals 9,158,208 bytes across these regions. Start with
`--blocks 4,16,64` (1024/4096/16384 tokens), not the old 2048-block synthetic
case. Initialization uses bounded temporary buffers. Verification still needs
additional host/GPU memory. This is a direct transfer microbenchmark: each
region transfers the same page count, with independent host allocations;
it does not simulate SWA eviction, request scheduling, or owner RPC.


### some special machine types and scenarios / XMLIR XDR backend

some special machine types and scenarios backend follows the working `../special-platform validation suite/test_xdr.c` and
`test_xdr_torch.py` mechanism. It retains the CE-only build and legacy worker.
`FLEXKV_GDR_BACKEND=auto` selects XDR when `torch_xmlir` is installed; explicitly
select `xdr` when building on some special machine types and scenarios to avoid runtime/library ambiguity.

The allocation owner queries `cudaPointerGetAttributes` once per new registered
storage and registers its **hostPointer** with verbs. This is the XDR peer mapping,
not a host bounce buffer. Layout/validation still use the original XPU pointer;
WQEs and exported IPC metadata use the mapped address plus the checked storage
offset. Workers never register imported IPC aliases. Tensor lifetime and owner
leases remain unchanged. D2H retains the producer fence before RDMA READ.

#### H2D write visibility on XDR

`Transport::finish()` establishes visibility in two steps before any layer-ready
event can be published, and `submit_layerwise` calls it (through `fence_gdr`)
immediately before each `notifier_.record`, so the eventfd is always posted
after both:

1. **A NIC-side read-back.** One small RDMA READ per QP that issued WRITEs in
   this epoch, into a scratch buffer registered once at endpoint construction.
   PCIe does not let a read completion from a completer pass posted writes
   already issued to that same completer, so the CQE for this READ means the
   device has accepted the payload. The CQE of the WRITE itself does not say
   that: it is a posted write to *host* memory, a different completer, and the
   two are unordered. This is the vendor-neutral equivalent of NVIDIA's
   `cudaDeviceFlushGPUDirectRDMAWrites`, it costs one bounded round trip per QP
   per layer, and it is counted separately as `verbs.flush_reads` so it never
   inflates the payload `wqes`/`posts` used for bandwidth.
2. **XMLIR `cudaDeviceSynchronize`.** Retained because `../special-platform validation suite` validated
   the end-to-end path with it and it may additionally settle XPU-side caches.

Step 2 is deliberately *not* treated as sufficient on its own. It drains the
CUDA queues of the process that calls it, and the NIC's writes were never in a
CUDA queue; the consuming kernels also live in the allocation owner process,
not in the worker. The single-process `special-platform validation suite` could not distinguish the two,
which is why the read-back was added rather than assuming the synchronization
carried the ordering. A failed post or CQE in either step aborts the worker
with the same fail-closed rule as the payload path — publishing layer-ready
over data the device has not accepted is worse than crashing.

Do not drop step 1 until an XDR-specific visibility primitive has been
validated **across processes**. `tests/gdr/test_visibility.py` is that
validation, and the next section describes it, including the
`FLEXKV_GDR_SKIP_FLUSH` control that asks whether step 1 is load-bearing at all.
NVIDIA builds are unchanged: they retain SYNC_MEMOPS and the device-scoped flush
API, and post no read-back.

#### The cross-process gate

`test_direct_ipc.py` does not settle the question, because it checks the bytes
with `tensor.cpu()` from the worker process. Both halves are too weak: a copy
engine need not share a consuming kernel's visibility path, and the process
that issued the RDMA is not the one whose view is in doubt.

`tests/gdr/test_visibility.py` puts the roles where production has them. The
parent is the FlexKV worker and drives layerwise H2D over RDMA into KV it
imported over IPC. Each TP rank is a separate allocation-owner process, exactly
as SGLang runs them: its own CUDA context, its own eventfds, its own
interpreter. It waits on the layer's fd and launches the comparison from its
main loop — there is no consumer thread and no queue handoff.

The per-rank process split is load-bearing rather than cosmetic. An earlier
version put all eight ranks in one process, one thread each; because a layer's
per-rank eventfds are posted in one tight loop, every consumer became runnable
at once and then serialized on the GIL, which pushed the tail of the
notification-to-launch window into the hundreds of microseconds. A wide window
is precisely what lets writes become visible on their own, so the harness was
blunting the thing it exists to measure.

The comparison is `torch.ne`, and that choice is the mechanism rather than a
detail. A comparison is arithmetic, so no copy engine can serve it: it is
necessarily an SM kernel reading every byte of the layer from the owner's own
context. A same-dtype `copy_` would likely be a DMA, which is the one path
whose visibility was never in question, and a gather is not portable here —
XDNN has no uint8/int64 `index_select`. A startup probe checks the op on the
device before the IPC and GDR handshakes, so a vendor gap is named immediately
instead of surfacing later as a confusing failure.

What else makes it a visibility test rather than another data test:

* **Nothing** sits between `eventfd_read` and the launch. The expectation
  tensor, the mask and even the per-layer views are built once at startup,
  because an allocation or an `aten::select` in that window is a step the real
  consumer does not take and could establish visibility by itself.
* Every round writes a **new marker** and the owner pre-reads the same blocks
  first, so the caches hold the previous round's bytes going in. A read that
  missed the flush therefore comes back as the previous marker instead of as
  indistinguishable noise.
* The kernel captures the **verdict**; the explanation is read afterwards. A
  byte the kernel rejected that reads correct a moment later arrived late, and
  is reported as `VISIBILITY FAILURE` — distinct from corruption, which reads
  wrong both times. This is the signature the XDR read-back exists to prevent.
* Odd block IDs are never a transfer target and are compared against zero, so
  an out-of-range write is a named failure instead of silent damage.
* A self-check drives the classifier over synthetic late/stale/unwritten/torn
  buffers before the first round. A gate that cannot fail is not evidence, and
  this one proves it can fail on the same machine where it claims success.

```bash
FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0 \
python tests/gdr/test_visibility.py --tp 8 --qps 4 --rounds 200 \
  --notify polling --expect-flush --nics "$NICS"
```

`--expect-flush` fails the run if `verbs.flush_reads` stayed zero: on a build
that posted no read-back, a pass says nothing about whether XDR needs one.
`--background` adds owner-side compute on a side stream, which is closer to a
serving process and dirties the caches the consumer reads through.

#### Interrupting a run

Ctrl-C reaches all nine processes at once, and the eight owners are the ones
holding memory regions the worker still has queue pairs against. An owner that
acted on the signal would tear those down underneath the worker, which is why
the interrupt used to produce eight tracebacks and a `GDR drain acknowledgement
failed` rather than an exit. So the owners set `SIGINT` to `SIG_IGN` before they
allocate anything: shutdown is ordered, and the worker orders it. The worker
prints `interrupted; releasing owner registrations`, releases the leases in the
normal sequence, closes the pipes — which is enough on its own to free an owner
blocked in `recv` — then terminates and, if that is declined, kills. Every join
is bounded, because an unbounded one is indistinguishable from the hang it is
meant to end. `PR_SET_PDEATHSIG` covers the remaining case: a worker killed in a
way it cannot respond to would otherwise leave owners sitting in their
per-layer `select` timeout with nobody left to end them.

If a run still does not die, the hang is below Python — a native call holding
the main thread with the GIL released, where the interpreter cannot raise
`KeyboardInterrupt` at all and pressing Ctrl-C again changes nothing. Do not
guess at that; take a stack:

```bash
py-spy dump --pid $(pgrep -f 'test_visibility.py' | head -1) --native
```

Ctrl-\ (`SIGQUIT`) ends the group in the meantime.

#### Reading the notification-to-launch distribution

The summary prints one entry per check as `n / min / p50 / p95 / max`, and this
is the gate's self-diagnostic: sensitivity is per sample, not aggregate. A check
that launched a few microseconds after the eventfd is a sharp trial, one that
launched half a millisecond later is a blunt one, and the run's power comes from
the sharp end — so read `min` and `p50`, not `max`.

One thing still inflates the number and is not FlexKV's behaviour: XMLIR charges
tens of milliseconds for a process's *first* kernel launch, enough to dominate a
plain maximum on its own. Each owner therefore warms `torch.ne` before waiting
on any fd. Whatever remains is XMLIR's per-launch dispatch cost.

Cross-check with `--tp 1` when a number looks wrong. The owners are separate
processes, so one rank and eight should agree; if `--tp 8` is markedly worse,
the cause is device- or NIC-level contention rather than anything in the
consumer, and the distribution is the place that will show it.

#### Reading notification and consumer wait times

`layer_ready` reports per-layer median owner observation times after submit,
plus completion-to-observation delays computed within each round. It does not
score overlap: all data being ready before compute needs it is also a good
outcome. A slow consumer must not improve a score merely by extending the time
of the last observation. `BURST`/`progressive` labels in the historical results
below came from the retired ratio metric and are not current acceptance rules.

Use the timeline's `fd - cq_last` for producer notification delay and
`notified - fd` for subsequent observation delay. The timestamp `fd` is taken
after writing all rank FDs, so it is an aggregate, not each rank's post time.
`consumer_wait` measures select/read wall time including scheduling and syscall
overhead in the synthetic consumer. It is not SGLang compute stall time, nor
proof of GPU execution overlap. There is no extra allocation, synchronization,
or readiness probe between consuming an eventfd and launching the check kernel.

For serving performance, measure entry/return of SGLang's `wait_until()` and
whether all required restore tokens are already available when needed. Combine
those measurements with TTFT/throughput; do not require temporal overlap when
loads are already ready. A controlled early-notification regression test should
hold a later layer incomplete and verify that the first layer can notify anyway.

XMLIR on some special machine types and scenarios rejects `--notify hostfunc` before spawning owners; use `polling`.
On platforms supporting hostfunc, the pure-GDR stream-free counter assertion
applies only to polling.

##### Result on some special machine types and scenarios, 2026-09-18: burst, as the code predicts

43 layers, tp8, 200 rounds, `polling`, before the fix below:

```
L0=2320 L1=2415 L2=2435 ... L41=2996 L42=3014  (us after submit)
-> BURST: layer 0 at 77% of the window, leaving 23% of it to overlap,
   against the 98% an ideal 43-layer pipeline would leave
```

The transfer itself was over at 1314 us. Nothing was announced until 2320 us —
a full millisecond after the *last* layer was on the GPU — and the 43
announcements then came out over 694 us at a regular 14-17 us apart, which is
the cost of the poller's own walk over a layer (eight `cudaEventQuery` plus
eight eventfd writes), not 43 layers genuinely becoming ready 16 us apart.
Averaged over the layers, each one sat finished for 1902 us before its consumer
was told, on a transfer lasting 1086 us. **Per-layer eventfds bought a consumer
nothing at all on this path.**

That was what the code said would happen rather than a surprise.
`RegionBatchGroup::submit_layerwise` called `notifier_.arm()` *after*
`pool_->run_on_all` returned, so in `polling` mode the poller started only once
every layer's marker was already complete and then posted all of them back to
back. That placement was not an oversight: notification events are pooled and a
previously recorded event stays in the completed state, so arming before the
`record()` calls would make `cudaEventQuery` succeed on a stale event and
publish a **false** layer-ready. `hostfunc` mode does not have this problem —
the callback fires as the stream reaches it — but XMLIR rejects
`cudaLaunchHostFunc`, which is why some special machine types and scenarios runs `polling`.

##### The fix: the arrival is the marker, so the poller can be armed first

The mechanism stays exactly as it is — each rank leaves a completion marker for
the layer, one watcher collects them, the last one in posts the fd. What changes
is what a marker may be.

On a rank whose layer went entirely to the NIC there is nothing an event can add.
`fence_gdr(rank)` returning means that rank's queue pair is drained and, on XDR,
flushed; there is no later moment to wait for. So `LayerNotifier::record` takes a
`used_stream` flag, and when it is false the marker is the rank's arrival at the
boundary — no `cudaEventRecord` here and no `cudaEventQuery` in the poller. On a
43-layer tp8 transfer that is 344 driver calls saved each way. A single rank that
did use a stream puts the whole layer back on the event path, because the GDR
fence says nothing about work the driver has not run yet.

That is also what unblocks the placement. The reason `arm()` had to wait for the
fan-out was never the fan-out; it was that a pooled event reads as complete
before anyone records it. A per-layer arrival counter answers "are this
transfer's markers placed?" without consulting the driver, and the poller checks
it before touching an event. So `arm()` moved ahead of the launches, and a layer
is announced when it finishes instead of after all of them do. The native
contract test proves the guard is load-bearing by arming against a pooled,
already-complete event and asserting nothing is posted.

The rank that completes a layer does not post it. Posting is `tp_size` `write()`
syscalls — 24 us at tp8 — and on the critical path of whichever rank happened to
be last it delays *that rank's* next layer and compounds, measured at 15.8 us per
layer and ~680 us by layer 42. Instead it calls `notify_one` (~1-2 us) and goes
back to work; the poller does the writes on its own thread. The poller's wait is
a `wait_for` with the old 50 us tick as its timeout, so a missed wake costs one
tick and a layer whose ranks used streams is still discovered the old way.

`layer_stream_free_layers()` reports how many layers were announced without
touching a CUDA event. `test_visibility.py` asserts it equals `rounds * layers`,
because a run that quietly took the event path would produce a perfectly
plausible table measuring driver latency that production does not pay.

##### Result on some special machine types and scenarios: the notification is fixed, and rank skew is now the cap

Same 43 layers, tp8, 200 rounds. Layer 0 moved from 77% of the window to 42%,
the verdict from `BURST` to `progressive`, and the per-layer wait between a
layer finishing and its consumer knowing from 1902 us to about 300 us. The
transfer itself is unchanged: the median rank still reaps layer 42 at ~1330 us,
against ~1314 us before.

What the run also showed is where the earlier unattributed millisecond went.
`slowest - cq_done` averages **1004 us**: the median rank drains layer 0 at
274 us and the slowest drains it at 963 us. That skew was always there — the
old table reported only a median across ranks, so it had nowhere to appear —
and it explains the old mystery, because `arm()` could not run until the
slowest rank had finished *all* 43 layers.

It is now the binding constraint. A layer is announced when its slowest rank is
done, so seven of the eight owners wait ~690 us for one. `slowest rank per
layer` reports whether the title is pinned to one rank or moves around, because
the two call for opposite fixes: chase a machine asymmetry, or stop waiting on
every rank at all.

A cost this run still showed, since resolved: the eight `write()` calls landed
on whichever rank thread arrived last, which then posted the next layer ~5 us
later than before (`posted[L+1] - cq_done[L]` went from 14 us to 19 us). They
are the polling thread's now.

Separately, and not addressed by this, the loop fences per layer: `fence_gdr`
reaps layer L's completions before layer L+1's requests are handed to the NIC
lane, so the NIC idles at every layer boundary. In the run above the steady
state was 22 us of fence and 5 us of hand-off per layer against about 21 us of
wire time for 512 KiB at 200 Gb/s — so the NIC was busy roughly 80% of the time
and decoupling the two would be worth around 20% of the transfer, no more.

##### Result at 43 layers: the fabric is innocent, and so is the notifier

At the model's real depth the picture changes, and the change is not in the
notification path. `post_last - post_first` for layer 0 is **891 us**: the last
rank had not finished handing layer 0 to its NIC lane until 1058 us after
submit, while the first was done at 167 us. `cq_last - cq_first` is 895 us — the
same number, and it stays the same number at every layer (1553 against 1554 at
layer 42). **The fabric adds no skew at all.** It is there before the first byte
moves, and the transfer carries it to the end unchanged.

`fd` then acquits the notifier too. The 639 us a layer waits between finishing
and its consumer knowing splits as **18 us** to write the eventfds and **621 us**
for the owners to be scheduled. That second number is not a backlog: layer 0's
share of it is 449 us, and layer 0 is the first layer, so its owner was
certainly blocked in `select` when the fd was written. `notify_to_launch` p50 is
9.8 us. A process that reacts in ten microseconds took four hundred and fifty to
be given a core.

The fan-out entry spread says the same thing about the worker's own threads:
`enqueue` is eight mutex acquisitions and eight `notify_one` calls, yet the
threads it wakes enter the lambda **553 us** apart, and in one round a rank
thread did not reach layer 0 until 2573 us — after the fastest rank had already
finished all 43 layers. Both numbers are the same phenomenon, and
`Transport::finish` names the cause: the completion drain is a bare
`ibv_poll_cq` spin with a `pause` and no yield, so during a transfer eight rank
threads hold eight cores at 100% while eight owner processes try to wake on what
is left.

So the ordering of the remaining work is:

1. **CPU contention** — confirmed and fixed by `OMP_NUM_THREADS=1`; see below.
   It was most of everything, and the two items under it are what remained.
2. **The eventfd `write()` calls**, at 24 us per layer on whichever rank arrived
   last; now the polling thread's, see below.
3. **The per-layer fence**, worth around 20% of the transfer as estimated above.

`slowest rank per layer` stays close to uniform (r7 19% down to r3 4%), so there
is no machine asymmetry to chase and no slow NIC to find; the ranks simply do
not start together.

##### The answer: `OMP_NUM_THREADS=1`, and the pipeline appears

`vmstat` during a run is the tell: the host goes from `r` of 0-8 at idle to
**876-1286 runnable** at 99% user time. Eight rank threads and eight owner
processes cannot produce that. Torch sizes its intra-op pools from the core
count and OpenMP spin-waits by default, so nine processes on a 384-core host
bring thousands of spinning threads, and every wait this test reports measures
them.

Setting `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` takes `worker_threads` to 2 and
settles nearly the whole investigation. No pinning, no code change:

| | inherited | `OMP_NUM_THREADS=1` |
| --- | --- | --- |
| layer 0 announced | 1404 us, 48% of the window | **258 us, 8%** |
| lost against an ideal pipeline | 45% | **6%** |
| `idle` | 500 us | **29 us** |
| ...of which owners waiting for a core | 482 us | **5 us** |
| fan-out entry spread | 294 us | **18 us** |
| `notify_to_launch` max | 571 us | **84 us** |
| wall clock, 100 rounds | 190 s | **4.3 s** |

Per-layer eventfds now do what they exist for: a consumer starts layer 0 while
92% of the transfer is still in flight. Nothing about the notification path or
the poller was ever the binding constraint — thousands of spinning OpenMP
threads were. **Set this before reading any timing from this
stack**, and treat `worker_threads` in the header as a precondition rather than
a curiosity.

Two things survive, and both are now clean enough to act on.

**The eight `write()` calls were the whole remaining notification cost, and they
have been moved.** `idle` was 29 us, of which 24 us was `fd - cq_last` and only
5 us the owners. They also landed badly: the posting rank was by construction the
rank already behind. The launcher skew that dominated every earlier run is *gone
at the start* — `post_last - post_first` for layer 0 is 42 us, against 811 us
before — and instead accumulated at 32 us per layer to 1396 us by layer 42.

Do not read that whole 1396 us as the posting cost. The laggard is persistent
(see below), so the posting rank paid 24 us on roughly half the layers:
`43 × 0.5 × 24 ≈ 500 us`, about a third of it. The rest is the machine asymmetry
plus consumers that now genuinely contend with the NIC.

The writes are now the poller's, as described above, which recovers that ~500 us
without a new thread — the polling thread already existed and was simply not
being used on this path.

It recovered rather more than 500 us, and the surprise is where. Measured again
at 43 layers, tp8, 100 rounds, `OMP_NUM_THREADS=1`:

| | rank thread posts | poller posts |
| --- | --- | --- |
| layer 0 announced | 268 us | **221 us** |
| layer 42 announced | 3180 us | **2062 us** |
| the transfer itself | 3065 us | **1952 us** |
| `idle` per layer | 29 us | 39 us |
| ...of which `fd - cq_last` | 24 us | 31 us |

**Notification got 7 us per layer slower and the whole transfer got 1113 us
faster.** That is the shape of the change: the eight writes were serialised with
the posting rank's *next* layer, and 43 × 24 us ≈ 1032 us accounts for almost all
of what came back. Paying a wakeup to move them off that thread is a trade at
roughly 150:1.

Watch `fd - cq_last` in the timeline table: it is one thread posting layers in
order, so if the writes for a layer ever cost more than the gap between layers
arriving, the column grows down the table instead of staying flat. It does not —
it *shrinks*, from 51 us at layer 0 to 18 us at layer 42 — but the margin is
thinner than it looks. Late in that transfer layers arrive 21 us apart and the
poller spends 18 us on each. It keeps up at 43 layers. More layers, or smaller
ones, will hit this before they hit anything else.

What remains is the AND itself. Layer L is announced to every consumer once
every rank has drained, but on the H2D path each rank writes its own GPU's shard,
so consumer *r* arguably depends only on rank *r*. Per-rank posting would remove
the wait for the slowest rank entirely. It changes what the eventfd *means*, so
it is a decision rather than a patch, and it is deliberately not taken here.

**A rank asymmetry is now visible, and it is not the NICs.** `slowest rank per
layer` was uniform under contention and is not any more. But the title moves
between runs while staying pinned within one:

| run | slowest rank per layer |
| --- | --- |
| before the notification fix | r7 50%, r1 26%, r2 16% |
| after, explicit `--nics` | **r1 73%**, r2 22% |
| after, automatic selection | **r2 92%**, r1 7% |

A NIC on a worse PCIe path would name the same rank every time. Three runs named
three different ranks, each above half, so this is not topology — and the NIC
mapping is in fact already ideal. Running with `gdr.nic_names` unset logs the
automatic choice, and on some special machine types and scenarios it is the identity:

```
[GDR-NIC] logical_device=0 pci=0000:03:00.0 nic=mlx5_bond_0 selection=pcie_topology
[GDR-NIC] logical_device=1 pci=0000:16:00.0 nic=mlx5_bond_1 selection=pcie_topology
... device=N -> mlx5_bond_N for all eight, no ties, no fallback
```

So prefer leaving `nic_names` unset: `select_local_nic` gets the same answer and
cannot be typed wrong. What remains is that one rank per run is behind for the
whole run, which points at where its threads land rather than at the fabric.

`describe_laggards` used to assert "a machine asymmetry to chase" on exactly
this data, which would have sent the work to the NICs. It now refuses to name a
cause from a single run and asks for the second one, because the two causes are
indistinguishable from inside one and call for opposite work.

One cost is worth stating plainly: the transfer itself went from 2425 us to
3065 us, and the fast rank's cadence from 22 us to 39 us per layer against about
21 us of wire time. Overlap is not free — the consumers now really do run
concurrently with the NIC and compete for HBM, which is the point of the
exercise and also its price. This test's consumer sweeps the whole capacity per
layer with `torch.ne`, which is likely heavier than attention over one layer's
KV, so treat the 26% as an upper bound rather than a forecast.

One note for anyone reaching for `taskset` instead. An intermediate run pinned
the eight owners to eight cores, one each, and the owners' waits went to **16
ms** — which reads exactly like a catastrophic notifier bug and is not one. An
owner carries a CUDA runtime whose driver and OpenMP threads share that core,
so its main loop was waiting for a timeslice; once it ran, all 43 layers were
consumed in 616 us. The same run's `r7=94%` was the other half: pinning the
worker to cores 0-15 put all eight rank threads on one NUMA node, away from
half the NICs. Affinity is a sharp tool here and `OMP_NUM_THREADS` is a blunt
one that worked better.

#### The timeline table

`layer_ready` conflates several costs into one number. `set_layer_timeline(True)`
separates them by adding the worker's own timestamps per layer, taken around the
layer boundary that `submit_layerwise` already has — `posted` once the layer's
requests are all handed to the NIC lane, `reaped` once its completions have been
collected. They are `steady_clock` nanoseconds, which is `CLOCK_MONOTONIC` on
Linux, so they line up with the owner processes' wake timestamps without any
clock exchange. `test_visibility.py` turns it on and prints:

```
layer post_first post_last cq_first cq_last   fd notified idle
L0           166       934      188     959  966     1404  446
L1           203      1027      220    1049 1056     1568  519
...
L42         1105      2521     1127    2539 2546     2952  413
us after submit; a layer waits 527us on average between its slowest rank's
completions being collected and its consumer being told -- 7us of that before
the eventfd is written and 520us waiting for the owners to be scheduled --
over a 2373us transfer whose ranks differ by 768us going in and 771us coming
out, having entered the fan-out 12us apart; the fan-out joined at 2694us,
which is the floor for any layer the polling thread had to announce
```

Every column is a within-round extreme *across ranks*, so any two of them can be
subtracted and the answer is a wait some rank really had. Read four separate
questions out of them:

* **Pipelining.** `post_first[L+1]` against `cq_first[L]`: the NIC starts the
  next layer only after the previous one is reaped, so the boundary fence shows
  up here as idle wire time.
* **Skew, and where it comes from.** `post_last - post_first` is how far apart
  the ranks were *before* the fabric did anything; `cq_last - cq_first` is how
  far apart they were after. Comparing the two says whether the skew was created
  by the transfer or was already in the launcher — a distinction the earlier
  single `posted` column could not make, and which turned out to matter. The
  fan-out entry spread narrows it once more: a thread dispatched late and a
  thread dispatched on time that was slow over its first layer look identical in
  `post_last`, and call for completely different work.
* **Notification.** `notified - cq_last`, split by `fd` into the polling thread
  being woken and getting through the layer's writes (`fd - cq_last`) and
  the owners taking that long to be scheduled (`notified - fd`). Those are
  different bugs and one number cannot tell them apart; when the second
  dominates, no amount of work on the posting path will move it. `joined` no
  longer floors notification — the poller is armed before the launches — but it
  still bounds the launcher, so a layer told after it waited on a rank thread
  rather than on the fabric.

The instrumentation is off by default, allocates one pre-sized row per
(layer, rank) so the device threads never lock on the boundary they are
measuring, and costs two `clock_gettime` calls per layer per rank when on.

##### And one round on its own

A second table follows it, and the two answer different questions. The one
above medians each column independently over every round and every rank, which
is how to read a distribution and not how to read a sequence: a difference
between two of those columns is a difference of medians, and `cq_last` is a
maximum by construction, so the subtraction comes out better than any round
actually was.

```
layer post_first post_last cq_first cq_last   fd notified idle
L0           177       279      217     298  305      502  204
...
us after submit, within round 45 alone -- the median of 200 by notification
lateness, which ran from 31us at best through 401us here to 6945us at worst
```

Every number in a row here came from the same transfer, so `notified - cq_last`
is a wait that really happened and `post_first[L+1] - cq_last[L]` is a gap the
NIC really idled through. The round is the median by notification lateness — not
the luckiest and not an outlier — and the range across all rounds is printed
with it, so a reader can see which of those it nearly was.

Reading one round is also the only way to see the shape of a backlog. A median
table shows `idle` as a number per layer; a single round shows it *decaying*,
which is the signature of a consumer that fell behind early and spent the rest
of the transfer catching up, and which no per-layer average can express.

##### And each rank on its own

Both tables above keep an extreme *across* ranks in every column, so a rank that
is behind appears only as the distance between two of them and never as its own
pace. That distance has two causes with the same appearance and opposite fixes,
so a third table breaks it out per rank:

```
 rank   finished  step_p50  step_p95  step_max fence_p50
r0        1190.3      27.7      33.3      38.5      12.5
r2        1917.4      28.3     207.7     216.9      12.5
...
us; rank 2 finished last at 1917us against 1190us for the first, and keeps a
normal per-layer pace (28.3us against 27.8us) with a tail to 217us -- it stalls
rather than crawls, so look at where its threads run
```

`step` is boundary to boundary for that one rank, so it covers the layer's
launches as well as its fence — a rank that is late to *start* a layer is
exactly what `fence_p50` cannot show. The discriminator is the median against
the tail:

* **`step_p50` is high.** Every layer costs that rank more. The suspect is its
  path, its NIC, or memory bandwidth, and the whole transfer pays.
* **`step_p50` is normal and `step_max` is not.** It keeps up and loses its core
  a few times per round. The suspect is placement, and the fabric is innocent.

The verdict picks between those two rather than reporting the total, because the
total is identical in both cases — that is the entire reason the table exists.

Two readings of this table are wrong often enough to be worth naming, both of
them mistakes the verdict made on some special machine types and scenarios before it was taught otherwise:

* **Compare the slowest rank to the *fastest*, not the median.** A run came back
  with `step_p50` running 35.8, 39.1, 46.6, 46.8, 49.5, 49.7, 56.3, 57.0us — a
  smooth gradient with no outlier. Against the median that is 1.15x and reads as
  healthy; against the fastest it is 1.59x, and eight ranks spread over 1.6x is
  the finding. Dividing by the median hides exactly the case where every rank
  differs, because the median moves with them.
* **A tail on every rank is the host, not the rank.** The same run showed
  `step_max` of 3545–3820us on all eight. One rank stalling for 3.8ms is a
  placement bug worth chasing; eight ranks stalling for 3.8ms together is the
  machine, and the column says nothing about whichever rank finished last.

##### Was it the same machine throughout?

Every median here treats the rounds as samples of one thing. When a run degrades
partway that assumption breaks silently: the medians still print to three digits,
and they describe neither half. A 100-round some special machine types and scenarios run reached layer 0 at 252us by
round 25 and at 465us by round 75, and the summary reported 461us — a number that
was never true of any round, and which read as a 90% regression against the run
before it.

So the gate splits the run in half and compares, before any table:

```
NOT STATIONARY: the run slowed from 2395us to 3029us (1.26x) between its halves,
so every median below blends two machines and none of them is comparable to
another run. Find what changed first
```

When that line appears, stop reading the tables and find the other tenant, the
thermal cap, or the leftover process. Comparing a non-stationary run against
anything — a previous run, a threshold, another backend — measures the machine's
mood.

#### Is the read-back load-bearing?

A passing gate shows the read-back *works*. It does not show it is *needed*,
and those are different claims — the second one is what justifies paying a
bounded round trip per QP per layer. The only way to separate them is to remove
the read-back and look, which is what `FLEXKV_GDR_SKIP_FLUSH=1` is for. It drops
step 1 and keeps step 2, so the question it asks is precise: is the read-back
doing work the device synchronization does not already do?

```bash
FLEXKV_GDR_SKIP_FLUSH=1 FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0 \
python tests/gdr/test_visibility.py --tp 8 --qps 4 --rounds 200 \
  --notify polling --control-no-flush --nics "$NICS"
```

The variable is read once per process, defaults to the safe path, and logs at
WARNING when active. `--control-no-flush` asserts `verbs.flush_reads == 0`,
because a control that did not actually disable anything is worse than no
control: a typo in the variable name would leave the read-back on, the run would
pass, and the pass would be read as evidence against a mechanism that was never
removed. For the same reason the native contracts run `contracts noflush`, which
fails if the switch stops switching.

Read the two outcomes differently. **A failure is conclusive**: the read-back is
load-bearing and this is the receipt. **A pass is not** — memory ordering is a
property of the architecture, not something established by failing to observe a
violation at a 10 µs window, so a clean control bounds the risk without
licensing removal. The summary labels such a run `CONTROL RUN, not a gate` so its
output cannot later be quoted as a green result.

##### Result on some special machine types and scenarios, 2026-09-17: null, and the read-back stays

200 rounds × 8 layers × tp8 with `FLEXKV_GDR_SKIP_FLUSH=1` produced no mismatch
in 12 800 trials (`verbs.flush_reads=0`, warning emitted, switch verified by
`contracts noflush`). That is a null result rather than evidence for removal,
for a reason visible in the run's own numbers: `min=6.4us` is the floor of a
Python consumer's launch path, and a PCIe posted write drains in well under
that. **The hazard window probably lies entirely beneath this test's
resolution**, so a clean control was the likely outcome either way, and more
rounds do not help — they add samples outside the window, not sensitivity.

The cost side is equally unresolved: 85.2 s with the read-back against 85.5 s
without, both dominated by harness overhead, so its price here is also below
resolution. Removing it would trade an unquantified correctness margin for an
unmeasurable gain. NVIDIA ships `cudaDeviceFlushGPUDirectRDMAWrites` for exactly
this hazard, so absent vendor documentation stating that an XPU GPUDirect write
is visible to SMs once its CQE is observed, the prior favors keeping it. **That
documentation, not further measurement, is what would settle this.**

The one experiment that would still discriminate is disabling step 2 as well:
if visibility is in fact established by the device synchronization, dropping
both should fail, and a both-off run that stays green would show the gate cannot
reach the hazard at all.

Set these variables **before importing torch**, in every owner/worker process:

```bash
export BKCL_ENABLE_XDR=1 CUDA_ENABLE_P2P_NO_UVA=1
export CUDA_DEVICE_ORDER=OAM_ID MLX5_SCATTER_TO_CQE=0
# Use the same XMLIR environment / LD_LIBRARY_PATH as the successful special-platform validation suite.
FLEXKV_ENABLE_GDR=1 FLEXKV_GDR_BACKEND=xdr \
FLEXKV_ENABLE_KERNEL_TRANSFER=0 FLEXKV_ENABLE_METRICS=0 \
python -m pip install -v --no-deps --no-build-isolation -e .

# Automatically select NICs for the current logical device order:
NICS=''  # Automatically match logical XPU PCIe paths to active NICs
FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0 \
python tests/gdr/tune.py --ipc --tp 8 --qps 4 --nics "$NICS"

FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0 \
python tests/gdr/tune.py --profile dsv4-flash --share-mode layer_parallel \
  --arms CE,GDR --directions H2D,D2H --tp 8 --qps 4 --nics "$NICS" \
  --blocks 4,16,64 --concurrency 1 --warmup 3 --rounds 20 \
  --output result_special_xdr
```

For production, set `gdr.nic_names` to these NIC names **in logical device order**;
if visibility/order is remapped, change the list accordingly. An explicit NIC
avoids requiring XMLIR's PCI bus query. The existing endpoint/GID negotiation is
retained. No implicit NIC-index guess or visibility override is introduced.
Use forced GDR and disable initialization fallback for correctness qualification.
A `kernel` arm on some special machine types and scenarios is not an SM-kernel baseline; compare against CE.

CPU contracts cover a non-identity XDR mapping, storage offsets, remote-owner
translation, READ/WRITE and synchronization failure. They do not establish actual
some special machine types and scenarios cross-process visibility or performance: the IPC smoke test and
`test_visibility.py` must both pass on the device before running larger
workloads.

### Layer-verification timeout and bounded readback

`--notification-timeout` (default 60 seconds) applies separately to each eventfd
notification wait and to native completion waiting. Time spent checking bytes
is not charged against the next notification's budget. Timeout errors include
case, op, global layer, rank, token index, prior waiting, verification and elapsed
time. Byte mismatches additionally report region/local layer, KV component,
GPU/CPU block IDs and byte offset. Native and consumer errors are both retained.

`--verification-chunk-mib` (default 16) bounds each per-layer readback slab,
with a minimum of one GPU page. Copies use contiguous GPU slices and CPU-only
scatter indexing; every requested byte is checked after layer readiness. Existing
full verification, including untouched blocks, remains in place and still needs
substantial memory for large cases. This bounds per-layer temporary copies, not
total test memory or the duration of all correctness checks.

The DSv4 batch script exposes these as `NOTIFICATION_TIMEOUT` and
`VERIFICATION_CHUNK_MIB`. Neither option changes production transfer code.

#### What the timing matrix does and does not check

Know what a green matrix run means, because it is narrower than it looks. Per
case, exactly one transfer — the route probe — is byte-checked: H2D checks every
requested byte per rank per layer as its layer-ready token arrives, and both
directions then compare the whole buffer including untouched blocks. The
`--warmup` + `--rounds` timed iterations that follow run unchecked, because
verification is deliberately outside the measurement.

The compare after the loop does **not** make up for that. Each round regenerates
the same deterministic block pairs, so every round writes identical bytes to
identical addresses: a corruption in round 7 that round 8 overwrites correctly
is invisible, and so is a round that moved nothing at all, since the previous
round's correct bytes are already in place. That compare only sees damage
surviving the last round.

`VERIFY_EVERY_ROUND=1` (`--verify-every-round 1`) closes both. It restores the
baseline before each round — which is what makes a no-op round fail rather than
inherit a pass — and compares every byte afterwards, still outside the timer.
Use a separate `OUT`: the reset and compare perturb steady state, so a soak
run's numbers are not comparable to a clean one, and both `summary.md` and the
`verified_every_round` column in `summary.csv` say so.

D2H needs no layer-granular check and has nowhere to put one: it submits a
single request set with `sync=true`, which drains every stream before returning,
so its whole-buffer compare already sits at the completion point. The asymmetry
is real rather than an oversight — an RDMA READ's completion means the data is
in host memory and visible to the CPU, which is exactly the guarantee a
GPUDirect write into device memory does not give, and the reason the read-back
question above arises only on H2D.

### some special machine types and scenarios batch comparison scripts

Both root-level scripts are self-contained within this tree (no PR or special-platform validation suite
Python imports). `batch_run_ce_kernel_tp8.sh` retains the synthetic workload;
`batch_run_dsv4_transfer_modes.sh` uses the 43-layer DSv4 geometry and defaults
to `KV_SHARED_MODE=layer_parallel`. Both default to CE, overallperf-origin,
GDR and overallperf-withgdr. `kernel` is rejected because some special machine types and scenarios has no compiled
SM transfer path. The original arms retain the CE on some special machine types and scenarios implementation.

Both launchers set XDR environment defaults before importing torch, and set
`FLEXKV_USE_CE_TRANSFER_H2D=1` / `FLEXKV_USE_CE_TRANSFER_D2H=1` for original
and automatic fallback paths. They preserve explicit environment overrides;
setting either CE flag to 0 cannot work for its original path on a CE-only build.
Set `NIC_NAMES` in logical device order; use the validated XMLIR environment.

```bash
export NIC_NAMES=''  # Automatic PCIe mapping, not same-index NIC pairing
# Short qualification (IPC preflight is enabled by default):
HIT_TOKENS=1024 ROUNDS=1 WARMUP_ROUNDS=1 CPU_RUNS=1 GPU_RUNS=none \
  bash batch_run_dsv4_transfer_modes.sh
# Full matrices; output bases are explicit to keep each run separate:
OUT=result_special_dsv4 bash batch_run_dsv4_transfer_modes.sh
OUT=result_special_synthetic bash batch_run_ce_kernel_tp8.sh
```

Outputs retain the PR format: per-cell text logs and JSON/CSV results, plus
merged `raw.jsonl`, `summary.csv`, and `summary.md`. `DRY_RUN=1` prints commands
without importing torch or creating output directories. Shell/mocked-output
checks are not device correctness or performance measurements.

### XMLIR IPC transport

On XMLIR, `TensorSharedHandle` always uses the vendor PyTorch `reduce_tensor`
export/rebuild path, including callers passing `force_direct_ipc=True`. That flag
must not force NVIDIA VMM probes or `cuMemGetAddressRange_v2` on some special machine types and scenarios. The allocator
owns storage offsets, IPC mapping reuse and reference counts. GDR owner MR/XDR
metadata is separate and unchanged: no tensor copy or staging is introduced.
Rebuilding on the wrong logical device fails rather than copying to a new
allocation. Rebuild failures propagate rather than returning an empty tensor.
NVIDIA retains its existing legacy IPC and VMM selection.

Production notification defaults to polling on XMLIR, matching some special machine types and scenarios test
launchers. Explicit `FLEXKV_LAYERWISE_NOTIFY_MODE` still overrides it. The IPC
smoke prints `handle_types=torch_reduce` on some special machine types and scenarios and validates two owner
generations, per-layer data and H2D/D2H. special-platform validation suite's RDMA test is single-process;
these cross-process checks remain mandatory on the real device, and
`test_visibility.py` is the one that covers the consuming kernel.

Whichever mode is active, the eventfd `write` itself can still fail, and a lost
token leaves the consumer parked until `FLEXKV_LAYERWISE_WAIT_TIMEOUT_S`
(240 s by default) turns it into a `TimeoutError` naming the rank, counter and
layer. That is a loud failure rather than a hang, but the consumer cannot say
*why* the token never arrived, so `LayerEventfdTable::post` now logs the fd and
`errno` on a short write. Both modes share this path; `polling` additionally
turns a real `cudaEventQuery` error into a hard failure instead of spinning on
a layer that will never complete.

### Automatic NIC selection on some special machine types and scenarios

Leave YAML `gdr.nic_names: []` (the default), or set `NIC_NAMES=''` in the batch
launchers. Clear previously exported NIC_NAMES lists: an explicit list remains
an intentional override and disables automatic selection.

The backend queries each logical XMLIR device's PCI BDF, normalizes its case and
resolves `/sys/bus/pci/devices/<BDF>`. Active verbs port-1 devices are compared
using canonical `/sys/class/infiniband/<NIC>/device` paths, component by component.
It selects the deepest common PCI ancestor, then the shortest path. Missing
paths, no shared PCI root, or equal-best candidates fail with an actionable
initialization error; `on_init_failure` continues to control original-path
fallback. No same-index, nearest-BDF-number, or first-enumerated-NIC guess is used.

IPC owners select the NIC before registration and share the selected NIC and
endpoint through the existing lease handshake. Workers use that owner endpoint.
Local benchmark allocations use the same selector. Selection happens only at
initialization, not per transfer. With INFO logging, `[GDR-NIC]` shows logical
device, BDF and selected NIC. WARNING mode remains quiet.

On the supplied eight-device OAM topology the expected logical order is
`mlx5_bond_3,mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_7,mlx5_bond_5,mlx5_bond_4,mlx5_bond_6`.
This mapping is a regression fixture, not a hardcoded production mapping.
