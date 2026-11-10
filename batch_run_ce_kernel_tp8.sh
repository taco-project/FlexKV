#!/usr/bin/env bash
# XDR on some special machine types and scenarios port of flexkv_134ad747/batch_run_ce_kernel_tp8.sh.
# Same UX (ARMS / HIT_TOKENS / CPU_RUNS / GPU_RUNS / DRY_RUN); the data plane
# is tests/gdr/compare_modes.py. Default four supported arms:
#   CE, overallperf-origin, GDR, overallperf-withgdr
# kernel/kn is rejected on CE-only on some special machine types and scenarios builds.
# NIC_NAMES: comma-separated NICs in logical XPU order; leave empty for PCIe topology selection; do not assume same-index mapping.
# FLEXKV_USE_CE_TRANSFER_H2D=1 FLEXKV_USE_CE_TRANSFER_D2H=1 (original path on some special machine types and scenarios).
# XDR defaults: BKCL_ENABLE_XDR=1 CUDA_ENABLE_P2P_NO_UVA=1
# CUDA_DEVICE_ORDER=OAM_ID MLX5_SCATTER_TO_CQE=0 (all overridable).
# Use the XMLIR environment/LD_LIBRARY_PATH validated by special-platform validation suite.
# NOTIFICATION_TIMEOUT=60 VERIFICATION_CHUNK_MIB=16 control byte verification.
#
# Smoke: HIT_TOKENS=1024 ROUNDS=1 WARMUP_ROUNDS=1 CONCURRENCY=1 bash batch_run_ce_kernel_tp8.sh
# Print commands: DRY_RUN=1 bash batch_run_ce_kernel_tp8.sh
# PUT_TRACE=1: write per-physical-transfer timestamps/bytes to each cell's
# put-trace.jsonl (direct file output; no SGLang scheduler or Mooncake stages).
set -euo pipefail

# Inherited by the supervisor, allocation owners and spawned workers.
export BKCL_ENABLE_XDR="${BKCL_ENABLE_XDR:-1}"
export CUDA_ENABLE_P2P_NO_UVA="${CUDA_ENABLE_P2P_NO_UVA:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-OAM_ID}"
export MLX5_SCATTER_TO_CQE="${MLX5_SCATTER_TO_CQE:-0}"
# The original backend on some special machine types and scenarios requires CE explicitly; auto GDR falls back here.
export FLEXKV_USE_CE_TRANSFER_H2D="${FLEXKV_USE_CE_TRANSFER_H2D:-1}"
export FLEXKV_USE_CE_TRANSFER_D2H="${FLEXKV_USE_CE_TRANSFER_D2H:-1}"


# Persist explicit options through the supervisor's re-exec (argv is dropped).
export FLEXKV_BENCH_QPS="${QP_COUNT:-${FLEXKV_BENCH_QPS:-4}}"
while (( $# )); do
  case "$1" in
    --gdr-qps)
      [[ $# -ge 2 && "$2" =~ ^([1-9]|1[0-6])$ ]] || { echo "--gdr-qps requires 1..16" >&2; exit 2; }
      export FLEXKV_BENCH_QPS="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: PUT_TRACE=1 ARMS='CE GDR' bash $0 [--gdr-qps 1..16]"
      echo "Data plane is tests/gdr/compare_modes.py. D2H is always timed; GDR/auto arms own the GDR path."
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$FLEXKV_BENCH_QPS" =~ ^([1-9]|1[0-6])$ ]] || { echo "QP_COUNT must be 1..16" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FLEXKV_ROOT:-${SCRIPT_DIR}}"
ROOT="$(cd "${ROOT}" && pwd)"
for dependency in tune.py compare_modes.py test_direct_ipc.py bench_process_guard.py; do
  [[ -f "${ROOT}/tests/gdr/${dependency}" ]] || { echo "Missing benchmark dependency: ${ROOT}/tests/gdr/${dependency}" >&2; exit 1; }
done


# Supervise the whole matrix on Linux. Darwin (local laptops) runs unsupervised.
if [[ "${DRY_RUN:-0}" != 1 && "${FLEXKV_BENCH_SUPERVISOR_PID:-}" != "${PPID}" ]]; then
  if [[ "$(uname -s)" == Linux && -f "${ROOT}/tests/gdr/bench_process_guard.py" ]]; then
    exec "${PYTHON:-python3}" "${ROOT}/tests/gdr/bench_process_guard.py" \
      --script "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")" --root "${ROOT}"
  fi
fi

canonicalize_arm() {
  case "$1" in
    CE|ce|memcpy2d) echo CE ;;
    kernel|kn) echo "kernel is unsupported on CE-only on some special machine types and scenarios builds; use CE or overallperf-origin" >&2; return 2 ;;
    overallperf-origin|origin|oo) echo overallperf-origin ;;
    GDR|gdrforce|gf) echo GDR ;;
    overallperf-withgdr|gdrauto|ga) echo overallperf-withgdr ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

arm_tag() {
  case "$1" in
    CE) echo ce ;;
    kernel) echo kn ;;
    overallperf-origin) echo oo ;;
    GDR) echo gf ;;
    overallperf-withgdr) echo ga ;;
  esac
}

ARMS_IN="${ARMS:-CE GDR}"
ARMS=()
ARM_TAG=""
need_gdr=0
for raw in ${ARMS_IN}; do
  arm="$(canonicalize_arm "${raw}")"
  ARMS+=("${arm}")
  short="$(arm_tag "${arm}")"
  ARM_TAG="${ARM_TAG}${ARM_TAG:+_}${short}"
  case "${arm}" in GDR|overallperf-withgdr) need_gdr=1 ;; esac
done
[[ -n "$ARM_TAG" ]] || { echo "ARMS must not be empty" >&2; exit 1; }

TP_SIZE="${TP:-${TP_SIZE:-8}}"
QP_COUNT="${FLEXKV_BENCH_QPS}"
DRY_RUN="${DRY_RUN:-0}"
PUT_TRACE="${PUT_TRACE:-0}"
case "$PUT_TRACE" in 0|1) ;; *) echo "PUT_TRACE must be 0 or 1" >&2; exit 2 ;; esac

OUT="${OUT:-${SCRIPT_DIR}/../result-tp${TP_SIZE}}"
OUT="${OUT%/}-${ARM_TAG}-qp${QP_COUNT}"
if [[ "${DRY_RUN}" != 1 ]]; then
  mkdir -p "${OUT}"
  OUT="$(cd "${OUT}" && pwd)"
fi
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
export FLEXKV_LOG_LEVEL="${FLEXKV_LOG_LEVEL:-WARNING}"
export FLEXKV_TRANSFER_TRACE="${FLEXKV_TRANSFER_TRACE:-0}"

PAGE_SIZE="${PAGE_SIZE:-256}"
if [[ -n "${BLOCKS:-}" ]]; then
  BLOCK_LIST="${BLOCKS}"
else
  HIT_TOKENS="${HIT_TOKENS:-256,1024,4096,16384,65536,131072,524288}"
  BLOCK_LIST="$("${PYTHON}" - "${HIT_TOKENS}" "${PAGE_SIZE}" <<'PY'
import sys
tokens, page = sys.argv[1], int(sys.argv[2])
blocks = []
for item in tokens.split(","):
    n = int(item)
    if n % page:
        raise SystemExit(f"HIT_TOKENS {n} is not a multiple of PAGE_SIZE={page}")
    blocks.append(str(n // page))
print(",".join(blocks))
PY
)"
fi

CPU_RUNS="${CPU_RUNS:-1 4 none}"
GPU_RUNS="${GPU_RUNS:-none 1 4}"
CONCURRENCY="${CONCURRENCY:-1}"
ROUNDS="${ROUNDS:-20}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-3}"
(( WARMUP_ROUNDS >= 1 )) || WARMUP_ROUNDS=1
DIRECTION="${DIRECTION:-H2D,D2H}"
[[ "$DIRECTION" == both ]] && DIRECTION=H2D,D2H
LAYOUTS="${LAYOUTS:-BLOCKFIRST}"
LAYERS="${LAYERS:-8}"
FRAGMENT_BYTES="${FRAGMENT_BYTES:-8192}"
KV_DIM="${KV_DIM:-1}"
SHARED_KV="${SHARED_KV:-1}"
BATCH_REQUESTS="${BATCH_REQUESTS:-4}"
MAX_BATCH_BLOCKS="${MAX_BATCH_BLOCKS:-8192}"
SELECTOR="${SELECTOR:-avgbytes}"
NOTIFY_MODE="${NOTIFY_MODE:-polling}"
NIC_NAMES="${NIC_NAMES:-}"
USE_DEDICATED_GDR_THREAD="${USE_DEDICATED_GDR_THREAD:-1}"
DIAGNOSTICS="${DIAGNOSTICS:-0}"
NOTIFICATION_TIMEOUT="${NOTIFICATION_TIMEOUT:-60}"
VERIFICATION_CHUNK_MIB="${VERIFICATION_CHUNK_MIB:-16}"

COMMON=(
  "${PYTHON}" tests/gdr/compare_modes.py
  --directions "${DIRECTION}" --tp "${TP_SIZE}" --blocks "${BLOCK_LIST}"
  --concurrency "${CONCURRENCY}" --layers "${LAYERS}" --fragment "${FRAGMENT_BYTES}"
  --kv-dim "${KV_DIM}" --shared "${SHARED_KV}" --layouts "${LAYOUTS}"
  --qps "${QP_COUNT}" --batch "${BATCH_REQUESTS}" --max-batch-blocks "${MAX_BATCH_BLOCKS}"
  --dedicated "${USE_DEDICATED_GDR_THREAD}" --selector "${SELECTOR}"
  --notify "${NOTIFY_MODE}" --nics "${NIC_NAMES}"
  --notification-timeout "${NOTIFICATION_TIMEOUT}" --verification-chunk-mib "${VERIFICATION_CHUNK_MIB}"
  --rounds "${ROUNDS}" --warmup "${WARMUP_ROUNDS}" --diagnostics "${DIAGNOSTICS}"
)

log_header() {
  local arm="$1" label="$2" cpu_shape="$3" gpu_shape="$4"
  shift 4
  printf '# ==================== EXPERIMENT CONFIG ====================\n'
  printf '# started_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '# arm=%s label=%s qps=%s\n' "${arm}" "${label}" "${QP_COUNT}"
  printf '# root=%s\n# python=%s\n' "${ROOT}" "${PYTHON}"
  printf '# git_commit=%s\n' "$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  printf '# harness=tests/gdr/compare_modes.py implementation=direct-owner-v1\n'
  printf '# tp_size=%s concurrency=%s directions=%s layouts=%s\n' "${TP_SIZE}" "${CONCURRENCY}" "${DIRECTION}" "${LAYOUTS}"
  printf '# cpu_run=%s gpu_run=%s (none=contiguous; numbers=blocks_per_run + 1-block hole; mix=1,4,2,8)\n' \
    "${cpu_shape}" "${gpu_shape}"
  printf '# blocks=%s host_hit_tokens=%s page_size=%s fragment_bytes=%s layers=%s\n' \
    "${BLOCK_LIST}" "${HIT_TOKENS:-}" "${PAGE_SIZE}" "${FRAGMENT_BYTES}" "${LAYERS}"
  printf '# warmup_rounds=%s measured_rounds=%s kv_dim=%s shared=%s selector=%s batch=%s\n' \
    "${WARMUP_ROUNDS}" "${ROUNDS}" "${KV_DIM}" "${SHARED_KV}" "${SELECTOR}" "${BATCH_REQUESTS}"
  printf '# notify=%s dedicated_gdr_thread=%s diagnostics=%s\n' \
    "${NOTIFY_MODE}" "${USE_DEDICATED_GDR_THREAD}" "${DIAGNOSTICS}"
  printf '# log_level=%s transfer_trace=%s\n' "${FLEXKV_LOG_LEVEL}" "${FLEXKV_TRANSFER_TRACE}"
  printf '# CUDA_VISIBLE_DEVICES=%s XPU_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" "${XPU_VISIBLE_DEVICES-<unset>}"
  printf '# BKCL_ENABLE_XDR=%s CUDA_ENABLE_P2P_NO_UVA=%s CUDA_DEVICE_ORDER=%s MLX5_SCATTER_TO_CQE=%s NIC_NAMES=%s\n' \
    "$BKCL_ENABLE_XDR" "$CUDA_ENABLE_P2P_NO_UVA" "$CUDA_DEVICE_ORDER" "$MLX5_SCATTER_TO_CQE" "$NIC_NAMES"
  printf '# original_ce_h2d=%s original_ce_d2h=%s\n' "$FLEXKV_USE_CE_TRANSFER_H2D" "$FLEXKV_USE_CE_TRANSFER_D2H"
  printf '# put_trace=%s (per-cell put-trace.jsonl; benchmark stages only)\n' "$PUT_TRACE"
  printf '# command='
  printf '%q ' "$@"
  printf '\n# ===========================================================\n\n'
}

run() {
  if (( $# < 6 )); then
    echo "Invalid run arguments: expected ARM LABEL CPU_RUN GPU_RUN PYTHON SCRIPT [args...]" >&2
    return 2
  fi
  local arm="$1" label="$2" cpu_shape="$3" gpu_shape="$4"
  case "${arm}" in
    CE|kernel|overallperf-origin|GDR|overallperf-withgdr|ipc) ;;
    *) echo "Invalid run arm '${arm}'" >&2; return 2 ;;
  esac
  case "${label}" in
    ''|*[!a-zA-Z0-9_-]*) echo "Invalid log label '${label}'" >&2; return 2 ;;
  esac
  shift 4
  local log="${OUT}/test_var_length_h2d_phase_${arm}_${label}.txt"
  if [[ "$PUT_TRACE" == 1 && "$arm" != ipc ]]; then
    local trace_file="${OUT}/${arm}_${label}/put-trace.jsonl"
    set -- "$@" --put-trace-file "$trace_file"
    echo "${arm}/${label} transfer trace -> ${trace_file}"
  fi
  echo "${arm}/${label} -> ${log}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    printf 'cwd=%q command=' "${ROOT}"
    printf '%q ' "$@"
    printf '\n'
    return
  fi
  log_header "${arm}" "${label}" "${cpu_shape}" "${gpu_shape}" "$@" | tee "${log}"
  "$@" 2>&1 | tee -a "${log}"
}

if [[ "${DRY_RUN}" != 1 ]]; then
  "${PYTHON}" - "${ROOT}" <<'PY'
import pathlib, sys
import flexkv
from flexkv import c_ext
root = pathlib.Path(sys.argv[1]).resolve()
for module in (flexkv, c_ext):
    path = pathlib.Path(module.__file__).resolve()
    assert root in path.parents, f"Wrong tree loaded: {path}; expected {root}"
assert hasattr(c_ext, "gdr_compiled"), "Rebuild this FlexKV for some special machine types and scenarios tree with GDR bindings"
print(f"source={flexkv.__file__} extension={c_ext.__file__} gdr_compiled={c_ext.gdr_compiled()}", flush=True)
PY
fi

if [[ "${need_gdr}" == 1 ]]; then
  if [[ "${DRY_RUN}" != 1 ]]; then
    "${PYTHON}" -c "from flexkv import c_ext; assert c_ext.gdr_compiled(), 'rebuild with FLEXKV_ENABLE_GDR=1 FLEXKV_GDR_BACKEND=xdr FLEXKV_ENABLE_KERNEL_TRANSFER=0'"
  fi
  if [[ "${RUN_IPC_PROBE:-1}" == 1 ]]; then
    run ipc probe n/a n/a "${PYTHON}" tests/gdr/test_direct_ipc.py \
      --tp "${TP_SIZE}" --qps "${QP_COUNT}" --kv-dim "${KV_DIM}" \
      --layout "${LAYOUTS%%,*}" --notify "${NOTIFY_MODE}" --nics "${NIC_NAMES}"
  fi
fi

scatter_flags() {
  local side="$1" shape="$2"
  case "${shape}" in
    none)
      SCATTER_ARGS+=(--"${side}"-run-pages 0)
      ;;
    mix)
      SCATTER_ARGS+=(--"${side}"-run-mix "1,4,2,8")
      ;;
    *[!0-9]*|'')
      echo "Invalid ${side} run: ${shape}" >&2
      exit 1
      ;;
    *)
      SCATTER_ARGS+=(--"${side}"-run-pages "${shape}")
      ;;
  esac
}

for arm in "${ARMS[@]}"; do
  for cpu in ${CPU_RUNS}; do
    for gpu in ${GPU_RUNS}; do
      SCATTER_ARGS=(--scatter 0)
      scatter_flags cpu "${cpu}"
      scatter_flags gpu "${gpu}"
      label="pageseg${cpu}"
      [[ "${gpu}" == none ]] || label="${label}_gpuseg${gpu}"
      cell="${OUT}/${arm}_${label}"
      run "${arm}" "${label}" "${cpu}" "${gpu}" \
        "${COMMON[@]}" --arms "${arm}" --output "${cell}" "${SCATTER_ARGS[@]}"
    done
  done
done

if [[ "${DRY_RUN}" != 1 ]]; then
  "${PYTHON}" - "${OUT}" <<'PY'
import csv, json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for path in sorted(out.glob("*/raw.jsonl")):
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            row["case"] = path.parent.name
            rows.append(row)
if not rows:
    raise SystemExit("no raw.jsonl rows to merge")
(out / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
fields = [k for k in rows[0] if k not in ("samples_ms", "probe_routes", "config")]
with (out / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# Transfer comparison",
    "",
    "Byte verification and route probing are outside timing. Aggregate GB/s sums participating GPUs; it is not per-PCIe-link bandwidth.",
    "",
    "| Arm | Direction | Blocks/request | Requests | CPU run | GPU run | Median ms | p95 ms | Aggregate GB/s |",
    "|---|---|---:|---:|---|---|---:|---:|---:|",
]
for row in rows:
    case = row.get("case", "")
    cpu = gpu = ""
    if "pageseg" in case:
        rest = case.split("pageseg", 1)[1]
        if "_gpuseg" in rest:
            cpu, gpu = rest.split("_gpuseg", 1)
        else:
            cpu = rest
    lines.append(
        f"| {row['arm']} | {row['direction']} | {row['blocks_per_request']} | "
        f"{row['concurrency']} | {cpu or '—'} | {gpu or '—'} | {row['median_ms']:.3f} | "
        f"{row['p95_ms']:.3f} | {row['aggregate_GB_s']:.2f} |"
    )
(out / "summary.md").write_text("\n".join(lines) + "\n")
print(f"merged {len(rows)} rows -> {out / 'summary.md'}", flush=True)
PY
fi

echo "done: ${OUT}"
echo "Forced GDR requires h2d.gdr.calls / d2h.gdr.calls > 0 in probe_routes; auto may still pick GDR when fragment << avgbytes threshold."
