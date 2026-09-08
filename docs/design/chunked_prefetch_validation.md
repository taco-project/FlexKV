# Chunked prefetch validation

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

The feature remains disabled by default. Repeat GPU/model correctness on the
rebased versions and retune block/window sizes. Cython release builds,
DP/layerwise configurations, sustained failure injection, dedicated prefetch
Prometheus instrumentation and long-term resource trends remain outstanding.
Same-cache-state repeated performance A/B and prolonged load tests are required
before claiming TTFT/throughput improvements or production readiness. SSD, P2P,
multi-node and TRT-remote prefetch are explicitly outside this version's scope.
