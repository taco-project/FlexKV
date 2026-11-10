#!/usr/bin/env bash
# Mixed-length serving-style PUT/GET sweep.
#
# Each profile contains one or more long requests plus many short requests.
# The total resident GPU pool stays bounded; this is deliberately different
# from an infeasible batch of 200 simultaneous 512k-token requests.
set -euo pipefail

export BKCL_ENABLE_XDR="${BKCL_ENABLE_XDR:-1}"
export CUDA_ENABLE_P2P_NO_UVA="${CUDA_ENABLE_P2P_NO_UVA:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-OAM_ID}"
export MLX5_SCATTER_TO_CQE="${MLX5_SCATTER_TO_CQE:-0}"
export FLEXKV_USE_CE_TRANSFER_H2D="${FLEXKV_USE_CE_TRANSFER_H2D:-1}"
export FLEXKV_USE_CE_TRANSFER_D2H="${FLEXKV_USE_CE_TRANSFER_D2H:-1}"
export FLEXKV_ENABLE_LAYERWISE_TRANSFER=0
export FLEXKV_ENABLE_SWA_TRANSFER=0
export FLEXKV_KV_SHARED_ACROSS_RANKS_MODE="${FLEXKV_KV_SHARED_ACROSS_RANKS_MODE:-sharded}"
export FLEXKV_LOG_LEVEL="${FLEXKV_LOG_LEVEL:-WARNING}"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${FLEXKV_ROOT:-${SCRIPT_DIR}}" && pwd)"
PYTHON="${PYTHON:-python3}"

if [[ "${FLEXKV_BENCH_SUPERVISOR_PID:-}" != "${PPID}" && "$(uname -s)" == Linux ]]; then
  exec "${PYTHON}" "${ROOT}/tests/gdr/bench_process_guard.py" \
    --script "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")" --root "${ROOT}"
fi

TP_SIZE="${TP:-${TP_SIZE:-8}}"
ROUNDS="${ROUNDS:-5}"
WARMUP="${WARMUP:-2}"
CPU_CACHE_GB="${CPU_CACHE_GB:-32}"
QP_COUNT="${QP_COUNT:-4}"
READ_DEPTH="${READ_DEPTH:-16}"
H2D_BATCH_REQUESTS="${H2D_BATCH_REQUESTS:-4}"
D2H_BATCH_REQUESTS="${D2H_BATCH_REQUESTS:-1}"
TIMEOUT="${TIMEOUT:-1800}"
PROFILES="${PROFILES:-c4 c16 c64 c128 c200}"
OUT="${OUT:-${SCRIPT_DIR}/result-put-get-mixed-$(date +%Y%m%d_%H%M%S)}"

for dependency in \
  benchmarks/benchmark_put_get_mixed_e2e.py \
  benchmarks/analyze_put_get_diagnostics.py \
  tests/gdr/bench_process_guard.py; do
  [[ -f "${ROOT}/${dependency}" ]] || {
    echo "Missing ${ROOT}/${dependency}" >&2
    exit 1
  }
done

make_blocks() {
  local result="" pair block count i
  for pair in "$@"; do
    block="${pair%%:*}"
    count="${pair##*:}"
    for ((i = 0; i < count; ++i)); do
      if [[ -n "${result}" ]]; then
        result+=","
      fi
      result+="${block}"
    done
  done
  printf '%s' "${result}"
}

profile_blocks() {
  case "$1" in
    # token lengths: 1k, 8k, 128k, 512k
    c4) make_blocks 4:1 32:1 512:1 2048:1 ;;
    # token lengths: 1k×10, 8k×3, 32k, 128k, 512k
    c16) make_blocks 4:10 32:3 128:1 512:1 2048:1 ;;
    # token lengths: 1k×49, 8k×10, 32k×3, 128k, 512k
    c64) make_blocks 4:49 32:10 128:3 512:1 2048:1 ;;
    # token lengths: 1k×100, 8k×20, 32k×5, 128k×2, 512k
    c128) make_blocks 4:100 32:20 128:5 512:2 2048:1 ;;
    # token lengths: 1k×160, 8k×28, 32k×8, 128k×3, 512k
    c200) make_blocks 4:160 32:28 128:8 512:3 2048:1 ;;
    *) echo "Unknown profile: $1" >&2; return 2 ;;
  esac
}

mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for profile in ${PROFILES}; do
  request_blocks="$(profile_blocks "${profile}")"
  profile_dir="${OUT}/${profile}"
  mkdir -p "${profile_dir}"

  for arm in ce gdr; do
    cell="${profile_dir}/${arm}"
    trace="${cell}/request-trace.jsonl"
    mkdir -p "${cell}"
    command=(
      "${PYTHON}" benchmarks/benchmark_put_get_mixed_e2e.py
      --output "${cell}"
      --trace-file "${trace}"
      --request-blocks "${request_blocks}"
      --storage host
      --arm "${arm}"
      --on-init-failure error
      --tp "${TP_SIZE}"
      --tokens-per-block 256
      --cpu-cache-gb "${CPU_CACHE_GB}"
      --rounds "${ROUNDS}"
      --warmup "${WARMUP}"
      --qps "${QP_COUNT}"
      --read-depth "${READ_DEPTH}"
      --dedicated 1
      --h2d-batch-requests "${H2D_BATCH_REQUESTS}"
      --d2h-batch-requests "${D2H_BATCH_REQUESTS}"
      --gdr-diagnostics 0
      --timeout "${TIMEOUT}"
      --run-label "${profile}-${arm}"
    )
    {
      printf '# profile=%s arm=%s request_blocks=%s\n' \
        "${profile}" "${arm}" "${request_blocks}"
      printf '# command='
      printf '%q ' "${command[@]}"
      printf '\n'
    } | tee "${cell}/run.log"
    "${command[@]}" 2>&1 | tee -a "${cell}/run.log"
  done

  "${PYTHON}" benchmarks/analyze_put_get_diagnostics.py "${profile_dir}"
done

echo "done: ${OUT}"
