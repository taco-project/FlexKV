#!/usr/bin/env bash
# DSv4 Flash XDR on some special machine types and scenarios transfer comparison; run from any working directory.
# Based on batch_run_ce_kernel_tp8.sh; preserves its log/cell/merged output format.
#
# Examples:
#   bash batch_run_dsv4_transfer_modes.sh
#   DIRECTION=D2H ARMS='GDR CE' CPU_RUNS=1 GPU_RUNS=none bash batch_run_dsv4_transfer_modes.sh
#   HIT_TOKENS=1024 ROUNDS=1 WARMUP_ROUNDS=1 CPU_RUNS=1 GPU_RUNS=none bash batch_run_dsv4_transfer_modes.sh
#   KV_SHARED_MODE=sharded QP_COUNT=8 bash batch_run_dsv4_transfer_modes.sh
#   DRY_RUN=1 bash batch_run_dsv4_transfer_modes.sh --gdr-qps 4
#
# Parameters (environment variables; defaults shown):
#   ARMS='CE GDR' (space-separated)
#     Aliases: ce/memcpy2d, origin/oo, gdrforce/gf, gdrauto/ga.
#     kernel/kn is unsupported on CE-only on some special machine types and scenarios builds and is rejected.
#   DIRECTION=H2D,D2H       H2D, D2H, H2D,D2H, or both.
#   KV_SHARED_MODE=layer_parallel  D2H ownership: layer_parallel or sharded.
#     Applies to every comparison arm; H2D still broadcasts to every rank.
#     Each region partitions its own layers; this is not global byte balancing.
#   TP_SIZE=8              Number of visible GPUs; TP overrides TP_SIZE.
#   QP_COUNT=4             1..16 QPs; --gdr-qps N overrides the environment.
#   READ_DEPTH=16          D2H READ depth; 0 selects the supported automatic limit.
#   HIT_TOKENS=1024,4096,16384,65536,131072,524288  Comma-separated multiples of 256.
#   BLOCKS=                Optional page-count list; overrides HIT_TOKENS.
#   CONCURRENCY=1          Comma-separated request counts, e.g. 1,4,8.
#   CPU_RUNS='1 4 none'     Space-separated fragmentation patterns.
#   GPU_RUNS='none 1 4'     none=contiguous; N=N pages/run plus a one-page hole;
#                          mix=cycle 1,4,2,8. Each CPU/GPU pair is a separate cell.
#   LAYOUTS=BLOCKFIRST     BLOCKFIRST, LAYERFIRST, or comma-separated both.
#   ROUNDS=20              Measured iterations per case.
#   WARMUP_ROUNDS=3        At least one warmup iteration.
#   VERIFY_EVERY_ROUND=0   1 = correctness soak: reset the baseline and byte-compare
#                          after every round instead of only the probe. Without it a
#                          case checks 1 transfer in WARMUP_ROUNDS+ROUNDS+1, and the
#                          post-loop compare sees only damage surviving the last
#                          round, since every round rewrites the same bytes. A soak
#                          run's timings are not comparable; use a separate OUT.
#   BATCH_REQUESTS=4       GDR H2D request merging limit (not D2H ownership).
#   MAX_BATCH_BLOCKS=8192  Maximum blocks per merged H2D request.
#   USE_DEDICATED_GDR_THREAD=1  0/1.
#   SELECTOR=avgbytes      Auto-route algorithm: avgbytes or segments.
#   NOTIFY_MODE=polling   H2D layer notification: hostfunc or polling.
#   NOTIFICATION_TIMEOUT=60  Seconds per notification wait, excluding byte checking.
#   VERIFICATION_CHUNK_MIB=16  Bounded per-layer GPU readback slab (at least one page).
#   NIC_NAMES=             Empty=topology-based selection; auto-select by actual logical-device PCIe path.
#                          Explicit lists must follow logical XPU order, not NIC suffix order.
#                          Match logical device order if visibility is remapped.
#   BKCL_ENABLE_XDR=1 CUDA_ENABLE_P2P_NO_UVA=1  Set before importing torch.
#   CUDA_DEVICE_ORDER=OAM_ID MLX5_SCATTER_TO_CQE=0 (overridable defaults).
#   Use the XMLIR environment/LD_LIBRARY_PATH validated by special-platform validation suite.
#   DIAGNOSTICS=0          0/1 transport summaries/timers; keep 0 for timing.
#                          Stage probes remain off; use probe_d2h.py separately.
#   RUN_IPC_PROBE=1        Run generic cross-process correctness test before GDR arms.
#                          This preflight is not the 43-layer performance workload.
#   FLEXKV_LOG_LEVEL=WARNING  FLEXKV_TRANSFER_TRACE=0  (quiet by default).
#   FLEXKV_USE_CE_TRANSFER_H2D / FLEXKV_USE_CE_TRANSFER_D2H:
#                          Default 1 in both directions; original paths on some special machine types and scenarios require CE.
#   CUDA_VISIBLE_DEVICES=  Inherited GPU visibility; normally leave unset.
#   PYTHON=python3         Interpreter with this tree's compiled FlexKV installed.
#   FLEXKV_ROOT=           Defaults to this script's directory.
#   OUT=                   Output base; default result-dsv4-tpN-TIMESTAMP.
#                          Appends -MODE-ARM_TAG-qpN; use a fresh base for each run.
#   DRY_RUN=0              1 prints commands without GPU imports or result writes.
#
# Fixed geometry (not tuning knobs): 43 layers, page_size=256, kv_dim=1,
# shared KV; layers 0/1 SWA only, then even=C4 and odd=C128, all with SWA.
# Includes C4 indexer + float32 attention/indexer state rings and page padding.
# LAYERS/PAGE_SIZE/KV_DIM/SHARED_KV may only repeat these fixed values;
# FRAGMENT_BYTES is invalid because the six groups have different byte widths.
# Each block totals 9,158,208 bytes; large BLOCKS/CONCURRENCY consume much more
# memory than the synthetic benchmark. All groups move the same page count:
# no SGLang scheduling, SWA eviction or owner RPC in the timed microbenchmark.
#
# Outputs (same structure as batch_run_ce_kernel_tp8.sh):
#   test_var_length_h2d_phase_ARM_LABEL.txt (legacy filename, also contains D2H)
#   ARM_LABEL/{metadata.json,raw.jsonl,summary.csv,tuning.json}
#   raw.jsonl, summary.csv, summary.md (merged); optional IPC probe log.
# Internal FLEXKV_BENCH_* variables are reserved for the Linux process supervisor.
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
      echo "DSv4 Flash via tests/gdr/tune.py; KV_SHARED_MODE defaults to layer_parallel. See script header for all parameters."
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

KV_SHARED_MODE="${KV_SHARED_MODE:-layer_parallel}"
case "$KV_SHARED_MODE" in layer_parallel|sharded) ;; *) echo "KV_SHARED_MODE must be layer_parallel or sharded" >&2; exit 2 ;; esac
READ_DEPTH="${READ_DEPTH:-16}"
[[ "$READ_DEPTH" =~ ^[0-9]+$ ]] || { echo "READ_DEPTH must be a nonnegative integer" >&2; exit 2; }
[[ "${PAGE_SIZE:-256}" == 256 && "${LAYERS:-43}" == 43 && "${KV_DIM:-1}" == 1 && "${SHARED_KV:-1}" == 1 && -z "${FRAGMENT_BYTES:-}" ]] || {
  echo "DSv4 geometry is fixed: PAGE_SIZE=256 LAYERS=43 KV_DIM=1 SHARED_KV=1; unset FRAGMENT_BYTES" >&2; exit 2;
}

OUT="${OUT:-${SCRIPT_DIR}/result-dsv4-tp${TP_SIZE}-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT%/}-${KV_SHARED_MODE}-${ARM_TAG}-qp${QP_COUNT}"
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
  HIT_TOKENS="${HIT_TOKENS:-1024,4096,16384,65536,131072,524288}"
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
VERIFY_EVERY_ROUND="${VERIFY_EVERY_ROUND:-0}"
case "$VERIFY_EVERY_ROUND" in 0|1) ;; *) echo "VERIFY_EVERY_ROUND must be 0 or 1" >&2; exit 2 ;; esac
DIRECTION="${DIRECTION:-H2D,D2H}"
[[ "$DIRECTION" == both ]] && DIRECTION=H2D,D2H
LAYOUTS="${LAYOUTS:-BLOCKFIRST}"
LAYERS=43
FRAGMENT_BYTES=per-region
KV_DIM="${KV_DIM:-1}"
SHARED_KV="${SHARED_KV:-1}"
BATCH_REQUESTS="${BATCH_REQUESTS:-4}"
MAX_BATCH_BLOCKS="${MAX_BATCH_BLOCKS:-8192}"
SELECTOR="${SELECTOR:-avgbytes}"
NOTIFY_MODE="${NOTIFY_MODE:-polling}"
NOTIFICATION_TIMEOUT="${NOTIFICATION_TIMEOUT:-60}"
VERIFICATION_CHUNK_MIB="${VERIFICATION_CHUNK_MIB:-16}"
NIC_NAMES="${NIC_NAMES:-}"
USE_DEDICATED_GDR_THREAD="${USE_DEDICATED_GDR_THREAD:-1}"
DIAGNOSTICS="${DIAGNOSTICS:-0}"

COMMON=(
  "${PYTHON}" tests/gdr/tune.py
  --profile dsv4-flash --share-mode "${KV_SHARED_MODE}" --read-depth "${READ_DEPTH}"
  --directions "${DIRECTION}" --tp "${TP_SIZE}" --blocks "${BLOCK_LIST}"
  --concurrency "${CONCURRENCY}"
  --kv-dim "${KV_DIM}" --shared "${SHARED_KV}" --layouts "${LAYOUTS}"
  --qps "${QP_COUNT}" --batch "${BATCH_REQUESTS}" --max-batch-blocks "${MAX_BATCH_BLOCKS}"
  --dedicated "${USE_DEDICATED_GDR_THREAD}" --selector "${SELECTOR}"
  --notify "${NOTIFY_MODE}" --nics "${NIC_NAMES}"
  --notification-timeout "${NOTIFICATION_TIMEOUT}" --verification-chunk-mib "${VERIFICATION_CHUNK_MIB}"
  --rounds "${ROUNDS}" --warmup "${WARMUP_ROUNDS}" --diagnostics "${DIAGNOSTICS}"
  --verify-every-round "${VERIFY_EVERY_ROUND}"
)

log_header() {
  local arm="$1" label="$2" cpu_shape="$3" gpu_shape="$4"
  shift 4
  printf '# ==================== EXPERIMENT CONFIG ====================\n'
  printf '# started_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf '# arm=%s label=%s qps=%s\n' "${arm}" "${label}" "${QP_COUNT}"
  printf '# root=%s\n# python=%s\n' "${ROOT}" "${PYTHON}"
  printf '# git_commit=%s\n' "$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  printf '# harness=tests/gdr/tune.py implementation=direct-owner-v1 profile=dsv4-flash\n'
  printf '# kv_shared_mode=%s read_depth=%s\n' "${KV_SHARED_MODE}" "${READ_DEPTH}"
  printf '# tp_size=%s concurrency=%s directions=%s layouts=%s\n' "${TP_SIZE}" "${CONCURRENCY}" "${DIRECTION}" "${LAYOUTS}"
  printf '# cpu_run=%s gpu_run=%s (none=contiguous; numbers=blocks_per_run + 1-block hole; mix=1,4,2,8)\n' \
    "${cpu_shape}" "${gpu_shape}"
  printf '# blocks=%s host_hit_tokens=%s page_size=%s fragment_bytes=%s layers=%s\n' \
    "${BLOCK_LIST}" "${HIT_TOKENS:-}" "${PAGE_SIZE}" "${FRAGMENT_BYTES}" "${LAYERS}"
  printf '# warmup_rounds=%s measured_rounds=%s kv_dim=%s shared=%s selector=%s batch=%s\n' \
    "${WARMUP_ROUNDS}" "${ROUNDS}" "${KV_DIM}" "${SHARED_KV}" "${SELECTOR}" "${BATCH_REQUESTS}"
  printf '# verify_every_round=%s%s\n' "${VERIFY_EVERY_ROUND}" \
    "$( (( VERIFY_EVERY_ROUND )) && printf ' (SOAK RUN: timings not comparable to a clean run)' )"
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
    run ipc probe n/a n/a "${PYTHON}" tests/gdr/tune.py --ipc --read-depth "${READ_DEPTH}" \
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
fields = [k for k in rows[0] if k not in ("samples_ms", "probe_routes", "config", "geometry")]
with (out / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "# DSv4 Flash transfer comparison",
    "",
    "43 layers; 256 tokens/page; six regions including SWA/state. See per-cell tuning.json for D2H ownership and READ depth.",
    "",
    "Byte verification and route probing are outside timing. Aggregate GB/s sums participating GPUs; it is not per-PCIe-link bandwidth.",
    "",
]
if any(r.get("verified_every_round") for r in rows):
    lines += [
        "**SOAK RUN, not a timing run.** Every round reset the baseline and byte-compared "
        "afterwards, so the numbers below carry that overhead and are not comparable to a "
        "clean run. What they do establish is that every transfer moved correct bytes, "
        "rather than only the probe and whatever survived the last round.",
        "",
    ]
lines += [
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
echo "Forced GDR must execute GDR without fallback; inspect probe_routes. Auto routes are selected per region."
