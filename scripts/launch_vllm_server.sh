#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

VLLM_PORT=30001
MODEL_PATH=Qwen3/Qwen3-8B
TENSOR_PARALLEL_SIZE=1
MAX_NUM_SEQS=128
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.9
ENABLE_FLEXKV=1
LOG_FILE=""
PROFILE=0
PROFILE_DURATION=600
PROFILE_DELAY=60
PROFILE_FILE=""

show_help() {
    cat << EOF
Usage: $0 [options]

Launch vLLM Server

Options:
  -p, --port PORT                           Server port (default: $VLLM_PORT)
  -m, --model MODEL_PATH                    Model path (default: $MODEL_PATH)
  -t, --tensor-parallel-size SIZE           Tensor parallel size (default: $TENSOR_PARALLEL_SIZE)
  -s, --max-num-seqs NUM                    Maximum number of sequences (default: $MAX_NUM_SEQS)
  -l, --max-model-len LEN                   Maximum model length (default: $MAX_MODEL_LEN)
  -g, --gpu-memory-utilization RATIO        GPU memory utilization (default: $GPU_MEMORY_UTIL)
  --enable-flexkv [0|1]                     Enable FlexKV: 0=disabled, 1=enabled (default: $ENABLE_FLEXKV)
  -o, --log FILE                            Log file path (default: auto-generated)
  --profile [0|1]                           Enable profiling (default: $PROFILE)
  --profile-delay SECONDS                   Profiling delay (default: $PROFILE_DELAY)
  --profile-duration SECONDS                Profiling duration (default: $PROFILE_DURATION)
  --profile-file FILE                       Profile output file (default: auto-generated)
  -h, --help                                Show this help message

Examples:
  # Launch with default configuration
  $0

  # Custom port and model
  $0 --port 8000 --model /path/to/model

  # Full configuration using vLLM standard arguments
  $0 -p 8000 -m /workspace/Llama-3-8B -tp 4 --max-num-seqs 256 --gpu-memory-utilization 0.8

  # Disable FlexKV (run vanilla vLLM)
  $0 --enable-flexkv 0

  # Enable profiling
  $0 --profile 1 --profile-duration 600 --profile-delay 60
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            VLLM_PORT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -t|--tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        -s|--max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        -l|--max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        -g|--gpu-memory-utilization)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        --enable-flexkv)
            ENABLE_FLEXKV="$2"
            shift 2
            ;;
        -o|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --profile-delay)
            PROFILE_DELAY="$2"
            shift 2
            ;;
        --profile-duration)
            PROFILE_DURATION="$2"
            shift 2
            ;;
        --profile-file)
            PROFILE_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            echo
            show_help
            exit 1
            ;;
    esac
done

if lsof -i:$VLLM_PORT > /dev/null 2>&1; then
    error "Port $VLLM_PORT is already in use"
    info "Process currently using the port:"
    lsof -i:$VLLM_PORT >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ -z "$LOG_FILE" ]; then
    LOG_DIR="$SCRIPT_DIR/../logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/vllm_server_${TIMESTAMP}.log"
fi

if [ "$PROFILE" = "1" ] && [ -z "$PROFILE_FILE" ]; then
    LOG_DIR="$SCRIPT_DIR/../logs"
    mkdir -p "$LOG_DIR"
    PROFILE_FILE="$LOG_DIR/vllm_profile_${TIMESTAMP}.nsys-rep"
fi

mkdir -p "$(dirname "$LOG_FILE")"

CMD="env VLLM_USE_V1=1 python3 -m vllm.entrypoints.cli.main serve \"$MODEL_PATH\" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --trust-remote-code \
    --port $VLLM_PORT \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_MODEL_LEN \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --enable-chunked-prefill \
    --enable-prefix-caching"

PROFILE_PREFIX=""
if [ "$PROFILE" = "1" ]; then
    PROFILE_PREFIX="nsys profile -o $PROFILE_FILE --trace-fork-before-exec=true --cuda-graph-trace=node --delay $PROFILE_DELAY --duration $PROFILE_DURATION -t cuda,nvtx,osrt --force-overwrite true --sample=cpu"
fi

if [ "$ENABLE_FLEXKV" = "1" ]; then
    CMD="$CMD --kv-transfer-config '{\"kv_connector\":\"FlexKVConnectorV1\",\"kv_role\":\"kv_both\"}'"
fi

CMD="$PROFILE_PREFIX $CMD"

info "Launching vLLM Server with command: $CMD" >&2
info "Launching vLLM Server..." >&2

# launch vllm server in background and get pid
eval "$CMD > \"$LOG_FILE\" 2>&1 &"
VLLM_PID=$!
echo $VLLM_PID
