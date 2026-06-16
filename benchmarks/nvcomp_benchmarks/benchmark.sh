#!/usr/bin/env bash
# Reproduce FlexKV nvCOMP close-loop end-to-end benchmarks.
#
# Typical usage:
#   DATASET=/path/to/datasets--princeton-nlp--SWE-bench_oracle \
#   MODEL_PATH=/path/to/GLM-5.1-FP8 \
#   JIT_CACHE=/path/to/writable/cache \
#   ISL=128k RUN_ID=run_000 \
#   bash benchmarks/nvcomp_benchmarks/benchmark.sh
#
# Paths users usually need to customize:
#   MODEL_PATH    Local model directory. Must contain tokenizer.json.
#                 Default: /tmp/nvidia-mps/GLM-5.1-FP8
#   DATASET       SWE-bench Oracle dataset id or local HuggingFace cache dir.
#                 Example: /workspace/develop/datasets/SWE-bench/datasets--princeton-nlp--SWE-bench_oracle
#                 Default: princeton-nlp/SWE-bench_oracle
#   JIT_CACHE    Writable cache directory for SGLang/FlexKV JIT/cache files.
#                 Default: /workspace/develop/envs/taco-sglang/cache
#   OUT_ROOT      Optional output root. If unset, results go under:
#                 benchmarks/nvcomp_benchmarks/runs/close_loop/$ISL/$RUN_ID
#   INPUT         Optional pre-generated conversations JSON. If unset, this
#                 script auto-generates one under benchmarks/nvcomp_benchmarks/inputs/.
#
# Other common knobs:
#   ISL=128k|64k, RUN_ID=run_000, VARIANTS="baseline nvcomp",
#   PORT=8000, TP_SIZE=8, SEED=42.
#
# By default this runs one baseline sweep and one nvcomp sweep. To run multiple
# repetitions, wrap this script in an outer RUN_ID loop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# User paths
MODEL_PATH="${MODEL_PATH:-/tmp/nvidia-mps/GLM-5.1-FP8}"
DATASET="${DATASET:-princeton-nlp/SWE-bench_oracle}"
JIT_CACHE="${JIT_CACHE:-/workspace/develop/envs/taco-sglang/cache}"

# Benchmark selection
ISL="${ISL:-128k}"
RUN_ID="${RUN_ID:-run_000}"
VARIANTS="${VARIANTS:-baseline nvcomp}"

# JIT cache config
export PYTHONUNBUFFERED=1
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-$JIT_CACHE/deep_gemm}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$JIT_CACHE/sglang}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$JIT_CACHE/flashinfer_workspace}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$JIT_CACHE/cuda}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"


# SGLang config
MODEL_NAME="${MODEL_NAME:-glm-5.1-fp8}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
QUANTIZATION="${QUANTIZATION:-fp8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-nsa}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-140000}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
PAGE_SIZE="${PAGE_SIZE:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"

# Flexkv config
FLEXKV_CPU_CACHE_GB="${FLEXKV_CPU_CACHE_GB:-400}"
FLEXKV_SSD_CACHE_GB="${FLEXKV_SSD_CACHE_GB:-0}"
FLEXKV_CPU_LAYOUT="${FLEXKV_CPU_LAYOUT:-BLOCKFIRST}"
FLEXKV_KV_CACHE_DTYPE="${FLEXKV_KV_CACHE_DTYPE:-bf16}"
FLEXKV_ENABLE_MPS="${FLEXKV_ENABLE_MPS:-0}"

# Client/data config
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-512}"
THINK_TIME_PER_TURN="${THINK_TIME_PER_TURN:-2}"
WAIT_TIME_PER_CONV="${WAIT_TIME_PER_CONV:-6}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-1800}"
FLUSH_TIMEOUT_SEC="${FLUSH_TIMEOUT_SEC:-120}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"
SKIP_WARMUP="${SKIP_WARMUP:-0}"
WARMUP_CLIENTS="${WARMUP_CLIENTS:-8}"
NUM_TURNS="${NUM_TURNS:-5}"


case "$(echo "$ISL" | tr '[:upper:]' '[:lower:]')" in
  64k|64000)
    ISL_NAME="isl64k"
    INPUT_BASENAME="swe-bench-oracle-ISL64k-5turns-conversations.json"
    ISL_TOKENS=64000
    MAX_CONVS="${MAX_CONVS:-40}"
    CONCURRENCIES="${CONCURRENCIES:-16 8 4 2 1}"
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
    ;;
  128k|128000)
    ISL_NAME="isl128k"
    INPUT_BASENAME="swe-bench-oracle-ISL128k-5turns-conversations.json"
    ISL_TOKENS=128000
    MAX_CONVS="${MAX_CONVS:-20}"
    CONCURRENCIES="${CONCURRENCIES:-8 4 3 2 1}"
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
    ;;
  *)
    echo "FATAL: unsupported ISL=$ISL; use ISL=64k or ISL=128k" >&2
    exit 2
    ;;
esac

RUN_ROOT="${OUT_ROOT:-$SCRIPT_DIR/runs/close_loop/$ISL_NAME/$RUN_ID}"
INPUT="${INPUT:-$SCRIPT_DIR/inputs/$INPUT_BASENAME}"
INPUT_CONVERSATIONS="${INPUT_CONVERSATIONS:-40}"

log_step() {
  echo
  echo "[bench] >>> $*"
}

cleanup_gpu() {
  log_step "cleanup_gpu: stop stale server processes and clear IPC state"
  pkill -9 -f "sglang.launch_server"             2>/dev/null || true
  pkill -9 -f "sglang.srt"                       2>/dev/null || true
  pkill -9 -fi "Scheduler"                       2>/dev/null || true
  pkill -9 -fi "TpModelWorker"                   2>/dev/null || true
  pkill -9 -fi "DetokenizerManager"              2>/dev/null || true
  pkill -9 -fi "TokenizerManager"                2>/dev/null || true
  pkill -9 -f "spawn_main"                       2>/dev/null || true
  pkill -9 -f "multiprocessing.spawn"            2>/dev/null || true
  pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null || true
  pkill -9 -f "flexkv"                           2>/dev/null || true
  sleep 2

  if command -v nvidia-smi >/dev/null 2>&1; then
    local a=0 pids
    while [ "$a" -lt 30 ]; do
      pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
      [ -z "$pids" ] && break
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      sleep 2
      a=$((a + 1))
    done

    local b=0 max_used=0
    while [ "$b" -lt 60 ]; do
      max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk 'BEGIN{m=0}{if($1+0>m)m=$1+0}END{print m}')
      [ "${max_used:-0}" -lt 1024 ] && break
      sleep 2
      b=$((b + 1))
    done
    if [ "${max_used:-0}" -ge 1024 ]; then
      echo "WARN cleanup_gpu: GPU still has ${max_used} MiB used after 120s"
    fi
  fi

  rm -f /tmp/flexkv_* 2>/dev/null || true
  for pat in 'torch_*' 'nccl-*' 'sglang_*' 'flexkv_*' 'cuda.*' 'mp-*' 'psm_*' 'sem.mp-*'; do
    find /dev/shm -maxdepth 1 -name "$pat" -delete 2>/dev/null || true
  done
  ipcs -m 2>/dev/null | awk -v u="${USER:-root}" '/^0x/ && $3==u {print $2}' \
    | xargs -r -n1 ipcrm -m 2>/dev/null || true
  sleep 2
  echo "[bench] cleanup_gpu: done"
}

prepare_input() {
  log_step "prepare_input: ensure SWE-bench conversations exist"
  echo "[bench] input=$INPUT"
  echo "[bench] dataset=$DATASET"
  echo "[bench] tokenizer/model=$MODEL_PATH"
  if [ -f "$INPUT" ]; then
    echo "[bench] input exists: $INPUT"
    return
  fi

  echo "[bench] input missing, generating: $INPUT"
  mkdir -p "$(dirname "$INPUT")"
  python3 "$SCRIPT_DIR/swe-bench-multiturn-setup.py" \
    --tokenizer "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-file "$INPUT" \
    --isl "$ISL_TOKENS" \
    --num-turns "$NUM_TURNS" \
    --num-conversations "$INPUT_CONVERSATIONS" \
    --seed "$SEED"
}

start_server() {
  local variant="$1" nvcomp="$2" out_dir="$3"
  local log="$out_dir/server.log"
  mkdir -p "$out_dir"
  : > "$log"

  log_step "start_server: variant=$variant nvcomp=$nvcomp"
  echo "[bench] server log=$log"
  echo "[bench] model=$MODEL_PATH served_name=$MODEL_NAME port=$PORT tp=$TP_SIZE"
  echo "[bench] max_running_requests=$MAX_RUNNING_REQUESTS context_length=$CONTEXT_LENGTH"
  cleanup_gpu
  echo "[bench] starting server: variant=$variant nvcomp=$nvcomp log=$log"

  FLEXKV_ENABLE_NVCOMP="$nvcomp" \
  FLEXKV_ENABLE_MPS="$FLEXKV_ENABLE_MPS" \
  FLEXKV_CPU_CACHE_GB="$FLEXKV_CPU_CACHE_GB" \
  FLEXKV_SSD_CACHE_GB="$FLEXKV_SSD_CACHE_GB" \
  FLEXKV_KV_CACHE_DTYPE="$FLEXKV_KV_CACHE_DTYPE" \
  FLEXKV_CPU_LAYOUT="$FLEXKV_CPU_LAYOUT" \
  python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size "$TP_SIZE" \
    --quantization "$QUANTIZATION" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --context-length "$CONTEXT_LENGTH" \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --page-size "$PAGE_SIZE" \
    --attention-backend "$ATTENTION_BACKEND" \
    --kv-connector-cls flexkv \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --trust-remote-code \
    --log-level info \
    2>&1 | tee "$log" &
  SERVER_PID=$!

  local waited=0
  until grep -q "fired up and ready" "$log" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "FATAL: server exited before ready; see $log" >&2
      exit 1
    fi
    if [ "$waited" -ge "$READY_TIMEOUT_SEC" ]; then
      echo "FATAL: timed out waiting for server readiness; see $log" >&2
      exit 1
    fi
    sleep 10
    waited=$((waited + 10))
    echo "[bench] waiting for server... ${waited}s"
  done
  echo "[bench] server ready after ${waited}s"
}

stop_server() {
  log_step "stop_server"
  echo "[bench] sending SIGINT to sglang.launch_server"
  pkill -INT -f "sglang.launch_server" 2>/dev/null || true
  sleep 10
  echo "[bench] stop_server: done"
}

mark_phase() {
  local out_dir="$1"
  shift
  local ts
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$ts  $*" | tee -a "$out_dir/phases.log"
}

flush_gpu_cache() {
  local label="$1"
  log_step "flush_gpu_cache: $label"
  echo "[bench] flush_cache before $label"
  local rc
  rc=$(curl -sf -o /dev/null -w "%{http_code}" \
    -X POST "http://localhost:$PORT/flush_cache?timeout=$FLUSH_TIMEOUT_SEC" || echo err)
  echo "[bench]   /flush_cache -> http $rc"
  sleep 3
}

run_client() {
  local out_file="$1"
  local num_clients="$2"
  local max_turns="${3:-}"
  local extra_args=()
  if [ -n "$max_turns" ]; then
    extra_args+=(--max-turns-per-conv "$max_turns")
  fi

  log_step "run_client: clients=$num_clients output=$out_file"
  if [ -n "$max_turns" ]; then
    echo "[bench] max_turns_per_conv=$max_turns"
  fi
  python3 -u "$SCRIPT_DIR/close_loop_multi_turn_client.py" \
    --input-file "$INPUT" \
    --output-file "$out_file" \
    --base-url "http://localhost:$PORT" \
    --model-name "$MODEL_NAME" \
    --num-conversations "$MAX_CONVS" \
    --num-clients "$num_clients" \
    --max-tokens "$MAX_TOKENS" \
    --think-time-per-turn "$THINK_TIME_PER_TURN" \
    --wait-time-per-conv "$WAIT_TIME_PER_CONV" \
    --rounds 1 \
    --seed "$SEED" \
    --request-timeout-sec "$REQUEST_TIMEOUT_SEC" \
    "${extra_args[@]}"
}

run_sweep() {
  local out_dir="$1"
  log_step "run_sweep: out_dir=$out_dir"
  echo "[bench] max_convs=$MAX_CONVS concurrencies=[$CONCURRENCIES]"
  echo "[bench] think_time=$THINK_TIME_PER_TURN wait_time=$WAIT_TIME_PER_CONV max_tokens=$MAX_TOKENS"
  mkdir -p "$out_dir"
  : > "$out_dir/phases.log"

  if ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "FATAL: server /health not OK on port $PORT" >&2
    exit 2
  fi

  if [ "$SKIP_WARMUP" = "1" ]; then
    log_step "warmup: skipped"
    mark_phase "$out_dir" "WARMUP_SKIPPED"
  else
    log_step "warmup: turn-0 only"
    mark_phase "$out_dir" "WARMUP_START  convs=$MAX_CONVS c=$WARMUP_CLIENTS turn-0-only"
    echo "[bench] warmup: convs=$MAX_CONVS clients=$WARMUP_CLIENTS"
    run_client "$out_dir/warmup.json" "$WARMUP_CLIENTS" 1 2>&1 | tee "$out_dir/warmup.log"
    mark_phase "$out_dir" "WARMUP_END"
    flush_gpu_cache "sweep"
  fi

  for c in $CONCURRENCIES; do
    local run_dir="$out_dir/c${c}"
    mkdir -p "$run_dir"
    log_step "sweep: concurrency=$c"
    mark_phase "$out_dir" "SWEEP_c${c}_START  convs=$MAX_CONVS"
    echo "[bench] sweep: c=$c convs=$MAX_CONVS"
    run_client "$run_dir/bench.json" "$c" 2>&1 | tee "$run_dir/client.log"
    mark_phase "$out_dir" "SWEEP_c${c}_END"
    flush_gpu_cache "next sweep"
  done
}

run_variant() {
  local variant="$1" nvcomp="$2"
  local out_dir="$RUN_ROOT/$variant"

  log_step "run_variant: $variant"
  echo "=== $variant (NVCOMP=$nvcomp, ISL=$ISL_NAME, RUN_ID=$RUN_ID) ==="
  start_server "$variant" "$nvcomp" "$out_dir"
  run_sweep "$out_dir"
  stop_server
}

main() {
  log_step "main"
  echo "[bench] repo=$REPO_ROOT"
  echo "[bench] isl=$ISL_NAME max_convs=$MAX_CONVS concurrencies=[$CONCURRENCIES]"
  echo "[bench] output=$RUN_ROOT"
  prepare_input

  for variant in $VARIANTS; do
    case "$variant" in
      baseline) run_variant baseline 0 ;;
      nvcomp) run_variant nvcomp 1 ;;
      *)
        echo "FATAL: unsupported variant=$variant; use baseline and/or nvcomp" >&2
        exit 2
        ;;
    esac
  done

  echo "=== all done. results under $RUN_ROOT ==="
}

trap stop_server EXIT
main "$@"
