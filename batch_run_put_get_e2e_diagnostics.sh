#!/usr/bin/env bash
# Diagnose cumulative forced-GDR degradation in the serving-style PUT/GET path.
#
# Default core suite:
#   bash batch_run_put_get_e2e_diagnostics.sh
# Longer parameter sweep:
#   SUITE=full ROUNDS=20 bash batch_run_put_get_e2e_diagnostics.sh
#
# Every named cell is a fresh process except "cumulative", which intentionally
# runs four workloads through one KVManager/GDR endpoint lifetime.
set -uo pipefail

export BKCL_ENABLE_XDR="${BKCL_ENABLE_XDR:-1}"
export CUDA_ENABLE_P2P_NO_UVA="${CUDA_ENABLE_P2P_NO_UVA:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-OAM_ID}"
export MLX5_SCATTER_TO_CQE="${MLX5_SCATTER_TO_CQE:-0}"
export FLEXKV_USE_CE_TRANSFER_H2D="${FLEXKV_USE_CE_TRANSFER_H2D:-1}"
export FLEXKV_USE_CE_TRANSFER_D2H="${FLEXKV_USE_CE_TRANSFER_D2H:-1}"
# Benchmark TP owners do not send layerwise eventfds. Inheriting a serving
# shell (FLEXKV_ENABLE_LAYERWISE_TRANSFER=1) makes FlexKV wait 0/8 forever.
export FLEXKV_ENABLE_LAYERWISE_TRANSFER=0
export FLEXKV_ENABLE_SWA_TRANSFER=0
export FLEXKV_LOG_LEVEL="${FLEXKV_LOG_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${FLEXKV_ROOT:-${SCRIPT_DIR}}" && pwd)"
PYTHON="${PYTHON:-python3}"

for dependency in \
  benchmarks/benchmark_put_get_e2e.py \
  benchmarks/analyze_put_get_diagnostics.py \
  tests/gdr/bench_process_guard.py; do
  [[ -f "${ROOT}/${dependency}" ]] || {
    echo "Missing ${ROOT}/${dependency}" >&2
    exit 1
  }
done

# Clean only tagged descendants if a run is interrupted or a benchmark crashes.
if [[ "${FLEXKV_BENCH_SUPERVISOR_PID:-}" != "${PPID}" && "$(uname -s)" == Linux ]]; then
  exec "${PYTHON}" "${ROOT}/tests/gdr/bench_process_guard.py" \
    --script "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")" --root "${ROOT}"
fi

SUITE="${SUITE:-core}"
case "${SUITE}" in core|full) ;; *)
  echo "SUITE must be core or full" >&2
  exit 2
esac

TP_SIZE="${TP:-${TP_SIZE:-8}}"
ROUNDS="${ROUNDS:-15}"
WARMUP="${WARMUP:-2}"
LAYERS="${LAYERS:-8}"
KV_HEADS="${KV_HEADS:-1}"
HEAD_SIZE="${HEAD_SIZE:-128}"
KV_DIM="${KV_DIM:-2}"
DTYPE="${DTYPE:-bfloat16}"
TOKENS_PER_BLOCK="${TOKENS_PER_BLOCK:-256}"
CPU_CACHE_GB="${CPU_CACHE_GB:-16}"
TIMEOUT="${TIMEOUT:-180}"
OUT="${OUT:-${SCRIPT_DIR}/result-put-get-diagnostics-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMON=(
  "${PYTHON}" benchmarks/benchmark_put_get_e2e.py
  --storage host
  --tp "${TP_SIZE}"
  --layers "${LAYERS}"
  --kv-heads "${KV_HEADS}"
  --head-size "${HEAD_SIZE}"
  --kv-dim "${KV_DIM}"
  --dtype "${DTYPE}"
  --tokens-per-block "${TOKENS_PER_BLOCK}"
  --cpu-cache-gb "${CPU_CACHE_GB}"
  --rounds "${ROUNDS}"
  --warmup "${WARMUP}"
  --on-init-failure error
  --timeout "${TIMEOUT}"
)

failures=()

run_cell() {
  if (( $# < 12 || $# > 13 )); then
    echo "run_cell expects LABEL ARM BLOCKS CONC QPS DEPTH DEDICATED H2DB D2HB SHARED GAP_MS OWNER_SYNC [DIAGNOSTICS]" >&2
    return 2
  fi
  local label="$1" arm="$2" blocks="$3" concurrency="$4"
  local qps="$5" depth="$6" dedicated="$7" h2d_batch="$8"
  local d2h_batch="$9" shared="${10}" gap_ms="${11}" owner_sync="${12}"
  local diagnostics="${13:-1}"
  local cell="${OUT}/${label}"
  mkdir -p "${cell}"
  local cmd=(
    "${COMMON[@]}"
    --output "${cell}"
    --trace-file "${cell}/request-trace.jsonl"
    --run-label "${label}"
    --arm "${arm}"
    --blocks "${blocks}"
    --concurrency "${concurrency}"
    --qps "${qps}"
    --read-depth "${depth}"
    --dedicated "${dedicated}"
    --h2d-batch-requests "${h2d_batch}"
    --d2h-batch-requests "${d2h_batch}"
    --gdr-diagnostics "${diagnostics}"
    --round-gap-ms "${gap_ms}"
    --owner-sync-after-put "${owner_sync}"
  )
  {
    printf '# label=%s arm=%s blocks=%s concurrency=%s qps=%s depth=%s dedicated=%s h2d_batch=%s d2h_batch=%s shared=%s gap_ms=%s owner_sync=%s diagnostics=%s\n' \
      "${label}" "${arm}" "${blocks}" "${concurrency}" "${qps}" "${depth}" \
      "${dedicated}" "${h2d_batch}" "${d2h_batch}" "${shared}" "${gap_ms}" "${owner_sync}" "${diagnostics}"
    printf '# command='
    printf '%q ' env "FLEXKV_KV_SHARED_ACROSS_RANKS_MODE=${shared}" "${cmd[@]}"
    printf '\n'
  } | tee "${cell}/run.log"
  env "FLEXKV_KV_SHARED_ACROSS_RANKS_MODE=${shared}" \
    "${cmd[@]}" 2>&1 | tee -a "${cell}/run.log"
  local status="${PIPESTATUS[0]}"
  if (( status != 0 )); then
    failures+=("${label}:${status}")
    printf '%s\n' "${status}" > "${cell}/exit-status"
    echo "FAILED ${label} exit=${status}" >&2
  else
    printf '0\n' > "${cell}/exit-status"
  fi
}

# Stable controls and the exact forced-GDR configuration from the first run.
run_cell ce-b64-c1                     ce  64 1 4 16 1 4 1 sharded        0    0
run_cell gdr-b64-c1                    gdr 64 1 4 16 1 4 1 sharded        0    0
run_cell gdr-b64-c1-diagoff            gdr 64 1 4 16 1 4 1 sharded        0    0 0
run_cell ce-b16-c4                     ce  16 4 4 16 1 4 1 sharded        0    0
run_cell gdr-b16-c4                    gdr 16 4 4 16 1 4 1 sharded        0    0

# Match the ownership mode used by the DSv4 direct benchmark.
run_cell gdr-b64-c1-layerparallel      gdr 64 1 4 16 1 4 1 layer_parallel 0   0
run_cell gdr-b16-c4-layerparallel      gdr 16 4 4 16 1 4 1 layer_parallel 0   0

# Does idle time or an explicit allocation-owner synchronization drain state?
run_cell gdr-b64-c1-gap1000            gdr 64 1 4 16 1 4 1 sharded        1000 0
run_cell gdr-b64-c1-owner-sync         gdr 64 1 4 16 1 4 1 sharded        0    1

# Endpoint/CQ pressure controls.
run_cell gdr-b64-c1-qps1               gdr 64 1 1 16 1 4 1 sharded        0    0
run_cell gdr-b64-c1-qps8               gdr 64 1 8 16 1 4 1 sharded        0    0
run_cell gdr-b64-c1-depth1             gdr 64 1 4 1  1 4 1 sharded        0    0
run_cell gdr-b64-c1-depth0             gdr 64 1 4 0  1 4 1 sharded        0    0
run_cell gdr-b64-c1-no-dedicated       gdr 64 1 4 16 0 4 1 sharded        0    0

# Same 64 MiB logical payload: one 64-block request vs four 16-block requests.
# Force batching off/on to prove whether scheduler-adjacent coalescing executes.
run_cell gdr-b16-c4-no-batch           gdr 16 4 4 16 1 1 1 sharded        0    0
run_cell gdr-b16-c4-batch4             gdr 16 4 4 16 1 4 4 sharded        0    0

# Four workloads share one endpoint lifetime. Compare with the isolated cells.
run_cell gdr-cumulative                gdr 16,64 1,4 4 16 1 4 1 sharded   0    0

if [[ "${SUITE}" == full ]]; then
  run_cell gdr-b64-c1-qps16            gdr 64 1 16 16 1 4 1 sharded       0    0
  run_cell gdr-b64-c1-depth4           gdr 64 1 4 4   1 4 1 sharded       0    0
  run_cell gdr-b64-c1-gap100           gdr 64 1 4 16  1 4 1 sharded       100  0
  run_cell gdr-b16-c4-batch8           gdr 16 4 4 16  1 8 8 sharded       0    0
  run_cell gdr-b64-c4                  gdr 64 4 4 16  1 4 1 sharded       0    0
  run_cell ce-b64-c4                   ce  64 4 4 16  1 4 1 sharded       0    0
fi

# Same-machine direct-owner bridge control. With 8 main + 4 SWA layers and a
# fragment of 87,376 bytes, 64 blocks carry 67,104,768 logical bytes (~64 MiB),
# closely matching the E2E 64-block request without replacing its serving path.
if [[ "${RUN_DIRECT_REFERENCE:-1}" == 1 ]]; then
  direct_base="${OUT}/direct-reference"
  ARMS="CE GDR" \
  BLOCKS=64 \
  CONCURRENCY=1 \
  CPU_RUNS=none \
  GPU_RUNS=none \
  LAYERS=8 \
  FRAGMENT_BYTES=87376 \
  KV_DIM=1 \
  SHARED_KV=1 \
  ROUNDS="${ROUNDS}" \
  WARMUP_ROUNDS="${WARMUP}" \
  RUN_IPC_PROBE=0 \
  DIAGNOSTICS=0 \
  OUT="${direct_base}" \
    bash "${ROOT}/batch_run_ce_kernel_tp8.sh" 2>&1 \
    | tee "${OUT}/direct-reference.log"
  direct_status="${PIPESTATUS[0]}"
  if (( direct_status != 0 )); then
    failures+=("direct-reference:${direct_status}")
  fi
fi

"${PYTHON}" benchmarks/analyze_put_get_diagnostics.py "${OUT}"
echo "done: ${OUT}"
echo "share this directory, especially diagnostic-report.md, diagnostic-summary.json, and each run.log/request-trace.jsonl"

if (( ${#failures[@]} )); then
  printf 'failed cells: %s\n' "${failures[*]}" >&2
  exit 1
fi
