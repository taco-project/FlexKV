#!/usr/bin/env bash
# End-to-end FlexKV PUT/GET benchmark through KVTaskEngine and TransferEngine.
# Unlike the RegionBatch microbenchmarks, this includes request planning,
# scheduler/worker queues, D2H, GPU-release eligibility, optional Mooncake
# write/read prefetch, H2D, task completion, and byte verification.
#
# Smoke:
#   BLOCKS=4 CONCURRENCY=1 ROUNDS=1 WARMUP=1 TP_SIZE=1 \
#     bash batch_run_put_get_e2e.sh
# Print commands:
#   DRY_RUN=1 bash batch_run_put_get_e2e.sh
# Mooncake:
#   STORAGE_MODES='host mooncake' MOONCAKE_CONFIG=/path/store.json \
#     bash batch_run_put_get_e2e.sh
#
# Main knobs:
#   ARMS='ce gdr'                    ce, gdr, auto, original
#   STORAGE_MODES='host'             host and/or mooncake
#   BLOCKS=4,16,64 CONCURRENCY=1,4
#   ROUNDS=10 WARMUP=2 TP_SIZE=8
#   LAYERS=8 KV_HEADS=1 HEAD_SIZE=128 KV_DIM=2 DTYPE=bfloat16
#   TOKENS_PER_BLOCK=256 CPU_CACHE_GB=16
#   QP_COUNT=4 READ_DEPTH=16 DEDICATED=1
#   ON_INIT_FAILURE=error            error or original
#   MOONCAKE_CONFIG=                 required for mooncake
#   OUT=result-put-get-e2e-TIMESTAMP
set -euo pipefail

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
export FLEXKV_LOG_LEVEL="${FLEXKV_LOG_LEVEL:-WARNING}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${FLEXKV_ROOT:-${SCRIPT_DIR}}" && pwd)"
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
ARMS="${ARMS:-ce gdr}"
STORAGE_MODES="${STORAGE_MODES:-host}"
TP_SIZE="${TP:-${TP_SIZE:-8}}"
BLOCKS="${BLOCKS:-4,16,64}"
CONCURRENCY="${CONCURRENCY:-1,4}"
ROUNDS="${ROUNDS:-10}"
WARMUP="${WARMUP:-2}"
LAYERS="${LAYERS:-8}"
KV_HEADS="${KV_HEADS:-1}"
HEAD_SIZE="${HEAD_SIZE:-128}"
KV_DIM="${KV_DIM:-2}"
DTYPE="${DTYPE:-bfloat16}"
TOKENS_PER_BLOCK="${TOKENS_PER_BLOCK:-256}"
CPU_CACHE_GB="${CPU_CACHE_GB:-16}"
QP_COUNT="${QP_COUNT:-4}"
READ_DEPTH="${READ_DEPTH:-16}"
DEDICATED="${DEDICATED:-1}"
ON_INIT_FAILURE="${ON_INIT_FAILURE:-error}"
TIMEOUT="${TIMEOUT:-120}"
MOONCAKE_CONFIG="${MOONCAKE_CONFIG:-${FLEXKV_MOONCAKE_STORE_CONFIG_PATH:-}}"
OUT="${OUT:-${SCRIPT_DIR}/result-put-get-e2e-$(date +%Y%m%d_%H%M%S)}"

[[ -f "${ROOT}/benchmarks/benchmark_put_get_e2e.py" ]] || {
  echo "Missing ${ROOT}/benchmarks/benchmark_put_get_e2e.py" >&2
  exit 1
}
case "${DRY_RUN}" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
case "${DEDICATED}" in 0|1) ;; *) echo "DEDICATED must be 0 or 1" >&2; exit 2 ;; esac
case "${ON_INIT_FAILURE}" in error|original) ;;
  *) echo "ON_INIT_FAILURE must be error or original" >&2; exit 2 ;;
esac

canonical_arm() {
  case "$1" in
    ce|CE|memcpy2d) echo ce ;;
    gdr|GDR|gdrforce|gf) echo gdr ;;
    auto|overallperf-withgdr|ga) echo auto ;;
    original|overallperf-origin|oo) echo original ;;
    *) echo "Unknown arm: $1" >&2; return 2 ;;
  esac
}

if [[ "${DRY_RUN}" != 1 ]]; then
  mkdir -p "${OUT}"
  OUT="$(cd "${OUT}" && pwd)"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for storage in ${STORAGE_MODES}; do
  case "${storage}" in
    host) ;;
    mooncake)
      [[ -n "${MOONCAKE_CONFIG}" ]] || {
        echo "MOONCAKE_CONFIG is required for STORAGE_MODES containing mooncake" >&2
        exit 2
      }
      ;;
    *) echo "Unknown storage mode: ${storage}" >&2; exit 2 ;;
  esac
  for raw_arm in ${ARMS}; do
    arm="$(canonical_arm "${raw_arm}")"
    cell="${OUT}/${storage}_${arm}"
    trace="${cell}/request-trace.jsonl"
    command=(
      "${PYTHON}" benchmarks/benchmark_put_get_e2e.py
      --output "${cell}" --trace-file "${trace}"
      --storage "${storage}" --arm "${arm}"
      --tp "${TP_SIZE}" --layers "${LAYERS}" --kv-heads "${KV_HEADS}"
      --head-size "${HEAD_SIZE}" --kv-dim "${KV_DIM}" --dtype "${DTYPE}"
      --tokens-per-block "${TOKENS_PER_BLOCK}" --cpu-cache-gb "${CPU_CACHE_GB}"
      --blocks "${BLOCKS}" --concurrency "${CONCURRENCY}"
      --rounds "${ROUNDS}" --warmup "${WARMUP}"
      --qps "${QP_COUNT}" --read-depth "${READ_DEPTH}"
      --dedicated "${DEDICATED}" --on-init-failure "${ON_INIT_FAILURE}"
      --timeout "${TIMEOUT}"
    )
    if [[ "${storage}" == mooncake ]]; then
      command+=(--mooncake-config "${MOONCAKE_CONFIG}")
    fi
    echo "${storage}/${arm} -> ${cell}"
    if [[ "${DRY_RUN}" == 1 ]]; then
      printf 'cwd=%q command=' "${ROOT}"
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi
    mkdir -p "${cell}"
    export FLEXKV_PUT_TRACE_RUN_ID="put-get-${storage}-${arm}-$(date +%s)-$$"
    {
      printf '# started_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      printf '# root=%s commit=%s\n' "${ROOT}" "$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
      printf '# storage=%s arm=%s tp=%s blocks=%s concurrency=%s rounds=%s warmup=%s\n' \
        "${storage}" "${arm}" "${TP_SIZE}" "${BLOCKS}" "${CONCURRENCY}" "${ROUNDS}" "${WARMUP}"
      printf '# command='
      printf '%q ' "${command[@]}"
      printf '\n'
    } | tee "${cell}/run.log"
    "${command[@]}" 2>&1 | tee -a "${cell}/run.log"
    "${PYTHON}" scripts/report_put_trace.py "${trace}" > "${cell}/request-report.txt"
  done
done

echo "done: ${OUT}"
