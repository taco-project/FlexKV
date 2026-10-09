# Chunked prefetch validation

## Integration cleanup: September 23, 2026

The existing SGLang connector patch and its integration README files remain
unchanged from main. The incremental prefetch patch is excluded from this PR:
the prefetch changes are already in the source branch of SGLang PR #31781
through merged review PR #6. The design docs refer to those source branches
directly. FlexKV owns the connector and communication implementation; SGLang
imports it and maintains the cache/scheduler lifecycle integration.

This cleanup changes only patch artifacts and documentation. Runtime sources
are unchanged from FlexKV `0dff8ea024` and paired SGLang `2ef249ef91`; it does
not add a new GPU or performance result. Earlier companion-patch checks below
describe their original revisions.

## Insert-after integration: September 22, 2026

The prefetch branch now includes main `738ddc141a`, including the merged
insert-after change (#266). The planner uses `num_matched_blocks` and
`last_node`: staging stays outside the radix tree until transfer completion,
so a resident match is already valid. The planner tests use the same contract
and retain checks for invisible staging, out-of-order completion, partial
failure, concurrent publication, prefix pins, and SWA checkpoints.

The previous schema probe reproduces the removed-field `AttributeError` with
the old planner and passes with the migrated `resolve()` and `_pin_prefix()`.
The internal branch's nonnegative span/checkpoint indexing is also included
so release Cython builds with `wraparound=False` preserve Python behavior.
The GPU fixture explicitly sets its 32 MiB staging limit; it no longer relies
on the removed experimental per-chunk byte target.

The initial companion SGLang head was `c6b51c1b5c`, which already contained
its adaptation target `5166ce06aa`. At that revision, the companion patch was
checked to reproduce the head exactly. These patch checks are historical; the
incremental patch was excluded after the changes entered the adaptation branch.

Validation on an isolated H20 container (Python 3.12, PyTorch 2.13.0+cu130,
CUDA 13.0), after rebuilding the current C++/CUDA extension:

- **520 FlexKV tests passed, 9 skipped**, both with Python prefetch modules
  and with all five prefetch modules compiled by Cython 3.2.5 using the release
  compiler directives. Skips are unsupported Python-radix SWA combinations.
- **167 SGLang tests and 12 subtests passed** against each control-plane build.
- **25 real GPU/Mooncake tests passed** against each build: three policies,
  window/chunk limits, timeout/demand/abort drain, overlapping sessions,
  foreground GET, reset, and a key disappearing after metadata matching.
  Returned KV contents, untouched suffixes, and released pin/staging budgets
  were checked. The isolated node-local Mooncake pool uses RDMA.
- Three existing main-branch SWA tests assumed the pre-insert publication
  order. Their assertions now check that Full KV stays invisible until both
  tier writers finish, becomes readable before downstream tiers finish, and
  the SWA slot remains unmounted until graph drain. The cache implementation
  itself is unchanged from main.

These compiled-module checks are not a complete release-wheel validation.
The GPU tests retain the previously observed CUDA IPC producer-exit warning;
they do not establish warning-free shutdown or performance acceptance.

Qwen3-0.6B was also checked on two H20 GPUs (TP2, BF16, eager, Triton
attention, page 16, CPU cache 0.25 GiB). Each scenario restarts SGLang with cold
GPU/CPU caches while retaining the dedicated L3 pool. For prompts of
512/1024/1536 tokens, all 15 restored requests produced exactly the same 32
greedy output token IDs as the three cold references:

| Policy | Restored L3 tokens at the three prompt lengths | Observed stop |
|---|---|---|
| Whole-task wait_complete | 496 / 1008 / 1520 | Complete |
| Timeout 30 s, chunk 8, window 2 | 496 / 1008 / 1520 | Complete, 4 / 8 / 12 graphs |
| Timeout 20 ms, chunk 1, window 2 | 32 / 48 / 48 | Deadline, then drain |
| Best effort, immediately eligible | 0 / 0 / 0 | Demand |
| Best effort, queued behind four requests | 496 / 1008 / 1520 | Complete while queued |

The queued scenario's 12 blocker requests also completed with 128 output
tokens each. The short-timeout sessions returned after 33.8-42.6 ms, including
13.5-22.6 ms of drain: this verifies interruption, not a hard 20 ms return
deadline. These are small-model correctness checks, not a new GLM throughput
or tail-latency comparison.

## Target-branch integration: September 15, 2026

FlexKV was merged with main `6960dfde09`; the companion SGLang review head
`c6b51c1b5c` includes adaptation base `5166ce06aa`. The connector resolution
preserves both physical indexer deduplication and SWA snapshot byte accounting.
The scheduler retains idle admission retry, shared restore deferral, and the
adaptation's new SWA snapshot grid behavior. The companion patch applies to
that SGLang base and produces exactly the review head's source tree.

- **487 FlexKV tests passed; 9 skipped.** Coverage includes prefetch policy and
  threads, Python/C++ radix planning, native IPC, deferred publication,
  cancellation, failure propagation, connector/store ownership, SWA, and indexer
  geometry. Two new constructor cases exercise deduplication on/off while
  preserving the snapshot budget. Skips are unsupported Python-radix SWA cases;
  corresponding C++ cases executed.
- **167 SGLang tests and 12 subtests passed** on the latest adaptation base,
  covering prefetch, ordinary/hybrid restore, admission, ownership, eviction,
  and PrefillAdder. Changed-file lint, formatting, and patch checks passed.
- The current FlexKV C++/CUDA extension was rebuilt in an isolated Linux
  container (Python 3.12.3, PyTorch 2.13.0+cu130, 4 CPUs, 16 GiB memory, no
  network or GPU devices). This validates native imports and CPU control/data
  paths; it is not a Cython release build, GPU/model test, or performance rerun.

Compatibility with [FlexKV #266](https://github.com/taco-project/FlexKV/pull/266)
was checked at `07496038d0`. Git can combine the changes without textual
conflicts, but the APIs were incompatible at that revision: #266 removes
`num_ready_matched_blocks` and `last_ready_node`, which the prefetch planner
still read. Executing the actual planner methods against the combined
match-result schema raises `AttributeError`; the coordinator turns the resolve
failure into a failed prefetch session. This is a focused API reproduction,
not a full test of the combined branches. The September 22 integration above
migrates the planner and tests together with insert-after. This migration
must not be backported alone to the September 15 main, which still permits
unready radix nodes.

## SGLang whole-task policy routing: September 10, 2026

Revision `837d3a4` routes SGLang's effective `wait_complete` policy to the
original `prefetch_async` path and disables the chunk runtime before creating
KVManager. `timeout` and `best_effort` retain chunked sessions when the feature
flag is enabled. Explicit FlexKV policy configuration takes precedence over
the SGLang policy argument. The explicit low-level session API still supports
`wait_complete`; this change does not alter that API or reconfigure an external
KVServer. The worker, native extension and transfer protocol are unchanged.

**197 focused Linux tests passed; 9 skipped.** This includes six new constructor
routing/precedence cases plus the runtime, coordinator, native integration,
planner and task-lifecycle coverage. Skips remain unsupported Python-radix SWA
combinations. Changed-file Ruff and whitespace checks pass.

The same GLM-5.2-FP8 / 8 H20 / TP8 eager configuration below is used for four
arms on this single source revision, with SGLang review head `2f91f9f5f0`:

| Arm | Effective service path | Chunk blocks | Timeout budget |
|---|---|---:|---:|
| `wait_whole` | Original whole-task API; no chunk runtime | Not used | Not used |
| `timeout_whole` | Session API; one graph per request | 4096 | 60 s |
| `timeout128` | Session API | 128 | 60 s |
| `timeout32` | Session API | 32 | 60 s |

The chunk window is two; reserved and pinned limits are 16/60 GiB in each
timeout arm. The largest 16K restore fits the one-graph arm's byte budget.
Comparing chunked timeout against one-graph timeout measures the overall effect
of segmentation within the same framework, including planning, IPC and backend
batch shape. Comparing one-graph timeout against `wait_whole` measures framework
and ownership-path entry cost. Neither comparison isolates pure IPC overhead.

Two rounds used opposite arm orders. **560 measured requests and 24 separate
cold-reference checks passed**, comparing all 32 output token IDs. Each arm had
six serial L3 requests and 32 requests each at client C8/C32; the server admitted
at most four concurrent requests. Every serial arm restored the same 826 blocks
through both REMOTE2H and H2D. The four arms completed 6/6/8/26 remote graphs per
round, respectively; each 16K request used 1/1/2/8 graphs. All timeout serial
sessions ended with `reason=complete`, with zero deadlines across all model logs.
Hot batches used the same 1,984-token GPU prefix and had no transfer completions.
Post-measurement thread snapshots found no chunk control/sender threads for
`wait_whole` and found both threads for each timeout arm.

| Comparison | Serial L3 throughput | Hot C8 throughput | Hot C32 throughput |
|---|---:|---:|---:|
| One-graph timeout / whole-task wait | +0.66% | -0.20% | +0.48% |
| Timeout 128 / one-graph timeout | +0.88% | +0.90% | -0.06% |
| Timeout 32 / one-graph timeout | +0.98% | +0.75% | +0.32% |
| Timeout 128 / whole-task wait | +1.55% | +0.67% | +0.41% |
| Timeout 32 / whole-task wait | +1.65% | +0.54% | +0.80% |

Values are the median of the two within-round percentage differences. For the
segmentation comparisons, per-round ranges were +0.52% to +1.23% (128) and
+0.78% to +1.17% (32) for L3; hot C8 ranges were -0.90% to +2.70% and +0.12% to
+1.39%, while hot C32 ranges were -0.73% to +0.61% and -0.01% to +0.66%.

For 16K prefixes, median TTFT in rounds one/two was 863/869 ms for `wait_whole`,
843/862 ms for one-graph timeout, 748/760 ms for chunk 128, and 760/786 ms for
chunk 32. Each value has only two samples. The latter two also had lower session
elapsed times; their successful restore size was identical. More graphs did not
produce an observed large regression in this workload, but short runs do not
establish zero overhead or an optimal chunk size. Session elapsed and summed
worker transfer durations are not interchangeable with end-to-end TTFT.

Sampled CPU throttling, memory-cgroup failures and RDMA error/discard deltas
were zero. GPU memory peaked at 97,357 MiB per GPU. Existing FlexKV Python and
Mooncake metrics were collected; the C++ endpoint was unavailable. Counters do
not replace worker-completion and output evidence, especially where the new
chunk path is not fully covered by legacy transfer counters.

This matrix covers serial L3 restores and shared-prefix GPU-hot pressure. The
previous mixed-capacity C8 OOM and shutdown resource-hook gap remain open. It is
not concurrent cold-L3, V4 Flash, a fresh native release build or a long soak.

The prior idle-polling results below used chunked `wait_complete` sessions in
SGLang. They describe the earlier routing and remain historical evidence.

## Idle control polling fix: September 10, 2026

Revision `ae634ee` suspends control-thread polling when no prefetch session or
submitted task is active. New commands wake it immediately; retained results
still wake at their TTL. Active work keeps the existing 2 ms polling interval,
and available chunk windows continue without an extra wait. The transfer
protocol, worker and native extension are unchanged.

**191 focused Linux tests passed; 9 skipped.** This includes real background
threads, command/stop wakeup races, TTL reclamation, native radix/CPU publication,
IPC integration and task lifecycle. The nine unsupported Python-radix SWA
combinations were skipped; corresponding C++ cases executed. The same 87-test
runtime/coordinator subset also passed locally.

GLM-5.2-FP8 ran on 8 H20 GPUs, TP8, BF16 KV, eager execution, page size 64,
prefill chunk 256 and a 16,640-token GPU pool. CPU cache was 64 GiB; the
node-local Mooncake RDMA pool was 256 GiB, with SSD disabled. Each request
generated 32 greedy tokens. Client concurrency was 8 or 32 for the hot tests;
the server admitted at most four concurrent requests.

The original baseline uses FlexKV `016c290` and the original SGLang adaptation
plus its independent one-line idle-admission retry fix. The pre-fix and fixed
variants use FlexKV `7df07a5` and `ae634ee`, respectively, with the same SGLang
review head `2f91f9f5f0`. Two rounds used opposite orders, with 420 measured
requests total. Main comparisons used `chunk_max_blocks=128`, window 2.

| Scenario | Pre-fix throughput change | Fixed throughput change |
|---|---:|---:|
| Full L3 restore, serial | -2.68% | +0.56% |
| GPU-hot, client C8 | -2.46% | -0.41% |
| GPU-hot, client C32 | -4.08% | -0.85% |

Changes are the median of the two differences against their own round's
baseline. Fixed per-round ranges were -0.11% to +1.22%, -0.83% to +0.00%, and
-1.09% to -0.62%, respectively. Every measured request matched its complete
reference output token IDs. Each serial group restored 826 blocks through both
REMOTE2H and H2D; hot groups had no transfer completions. These short, controlled
runs show the large regression has receded, not statistically established zero
overhead or production performance.

Separate hot diagnostics, excluded from these throughput comparisons, reduced
control-thread CPU use from 1.70 s / 47.35 s to 0.03 s / 45.33 s (about 98%
less CPU per unit time). This is thread CPU accounting, not a sampled GIL share.

A separate 70-request `chunk_max_blocks=32` check completed all restores,
including eight chunks for each 16K prefix. Serial throughput was 5.298 output
tokens/s with TTFT p95 0.754 s; hot C8/C32 were 21.781/21.790 tokens/s. This is a
single additional check, not a balanced sweep establishing an optimal chunk size.

With chunk 32 and a 30 ms timeout budget, all six serial requests matched their
output references. Four 8K/16K requests sealed on deadline at 30.30-30.99 ms,
restoring only the first 4,096 tokens in two chunks. They then drained for
64.99-66.87 ms before returning at 95.53-97.86 ms. The two 2K requests had already
submitted their single chunk and completed fully. The deadline stops future
submission; it does not cancel already submitted I/O or guarantee a 30 ms return.

All six best-effort requests also matched their output references. Five stopped
on scheduler demand: two had submitted no chunk and returned no storage prefix;
three drained two already submitted chunks and returned 4,096 tokens. One 2K
request had already submitted its sole chunk and completed fully. The interrupted
inflight requests drained for 85.71-87.54 ms. These checks cover both early demand
and demand with active I/O, separately from the noninterrupted performance table.

The prior mixed-capacity C8 OOM and shutdown resource-hook gap remain open and
were excluded from this focused performance fix. They are not cleared by these
successful noninterrupted workloads. No new V4 Flash or long-duration run is
claimed for this revision.

## Adaptation-branch review PR: September 8, 2026

The SGLang changes are proposed in
[XingLiu1/sglang#6](https://github.com/XingLiu1/sglang/pull/6), targeting
`agent/flexkv-dsv4-main`, the branch used by upstream PR #31781. They are not
yet incorporated into that branch. The review head `2f91f9f5f0` has exactly the
same source tree as the tested `e5b00611bd`; only branch ancestry changed.
The prior direct push was reverted at `4b76341435`, restoring the adaptation
branch's source tree exactly to `16780ea0c8`. The companion patch targets that
restored tree. Standalone upstream PR #38451 is superseded by the review PR.

**167 SGLang tests and 12 subtests passed** in the isolated CPU container.
This covers the new prefetch contracts plus the existing ordinary/hybrid restore,
abort/reset ownership, eviction, factory registry, and PrefillAdder regressions.
The existing eviction fixture was updated from the removed runtime-context and
`evict` mock APIs to `get_serving`/`get_spec` and `evict_for_alloc`.
Pinned Ruff/isort, formatting, registered-test rules and patch applicability pass.
The FlexKV runtime is unchanged from the 456-pass result below. No new GPU/model
or performance run was performed for this integration.

## Earlier standalone PR revision: September 8, 2026

The integration was rebased onto FlexKV `016c290` and SGLang `5aab054ec8`.
`chunk_target_bytes` was removed before release; `chunk_max_blocks` is the single
chunk-size option. Actual byte accounting, both instance-wide budgets, and
checkpoint constraints remain in force.

- **456 FlexKV tests passed; 9 skipped.** Includes coordinator/runtime, real
  Python/C++ radix planning, CPU slot allocation, IPC/RPC integration, transfer
  failures, deferred inserts/PUTs, cancellation, namespace isolation, connector
  and store lifecycle, SWA control and publication. The skips are SWA cases on
  the Python radix implementation; C++ SWA cases executed.
- **57 SGLang tests and 12 subtests passed.** Includes 29 FlexKV adapter/admission
  contract tests and 28 existing PrefillAdder tests against the rebased source.
- New tests verify equal block-based granularity across small/large block byte
  geometries, including a chunk larger than the former 32 MiB target, and verify
  that both reserved and pinned budgets still constrain allocation.
- The new upstream store protocol regression caught an unsafe use of `try_wait`
  for store completion. The connector now preserves zero-timeout polling with
  `completely=True`, so store ownership is released only after graph drain.
- Changed Python files pass the repositories' pinned Ruff configurations;
  SGLang also passes isort, Ruff formatting and registered-test checks. The
  companion SGLang patch matches the rebased branch.

Tests ran in an isolated CPU container with no GPU devices and no network,
4 CPUs, 12 GiB RAM and 2 GiB shared memory. Dependencies came from the retained
model-validation image. A matching previously built CUDA extension was mounted
read-only; no C++/CUDA source changed between that build and the FlexKV base.
Dependency deprecation/CPU-capability warnings were retained. This run is not a
fresh native-extension build, GPU test, model serving test, or performance run.

## Earlier snapshots: September 6–7, 2026

These results provide historical evidence for the preceding implementation,
with the old byte-target option and earlier repository bases. They are not
acceptance of the September 8 rebase or its changed chunk sizes.

- H20 + real Mooncake: 25/25 dense-path tests, including elementwise GPU content,
  unchanged tails, in-flight demand/abort, timeout, missing keys, concurrent GET,
  reset/epoch and allocator/pin/reservation cleanup. One additional real
  Full+SWA byte-and-lock round trip passed.
- Qwen3-0.6B: TP1 all three policies; TP2 full and timeout-partial restores.
  Generated tokens matched cold-compute references using matching TP/layout data.
- GLM-5.2-FP8 and V4-Flash-FP8: 210/210 formal requests per model. TP8/DP1;
  each policy used six C1 requests and 32 requests each at C8/C32, comparing all
  64 generated tokens to their reference. The workload included shared prefixes,
  queueing and CPU/GPU hot hits; it was not 32 independent cold-L3 requests.
- Earlier original tests failed on mixed TP cache-pool reuse, unanchored model
  output differences, and idle admission. The underlying cases and fixes were
  preserved in the local validation evidence; only final qualifying cases are
  counted above.

## Before broader enablement

The feature remains disabled by default. The September 10 run covers the
node-local TP8 eager configuration described above; other model and execution
configurations still need GPU/model validation and block/window tuning. Cython release builds,
DP/layerwise configurations, sustained failure injection, dedicated prefetch
Prometheus instrumentation and long-term resource trends remain outstanding.
Broader same-cache-state performance A/B and prolonged load tests are required
before extending the controlled results to production readiness. SSD, P2P,
multi-node and TRT-remote prefetch are explicitly outside this version's scope.
