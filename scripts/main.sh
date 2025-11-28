#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/common.sh"

CACHE_DIR="$HOME/.cache"

# Default values for vLLM configuration
VLLM_PORT=30001
TENSOR_PARALLEL_SIZE=1
MAX_NUM_SEQS=128
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.8

# Default values for FlexKV configuration
ENABLE_FLEXKV=1
FLEXKV_CPU_CACHE_GB=64
FLEXKV_SSD_CACHE_GB=1024
FLEXKV_SSD_CACHE_DIR="$CACHE_DIR/flexkv_ssd_cache/"
FLEXKV_ENABLE_GDS=0

# Default values for Dataset configuration
DATASET_NAME="sharegpt"
DATASET_PATH="$CACHE_DIR/sharegpt/"

# Default values for Benchmark configuration
REQUEST_RATE=32
WORKERS=32
MAX_TURNS=5
CONCURRENCY=2

# Default values for Log configuration
LOG_DIR="$SCRIPT_DIR/../logs"

# Usage function
usage() {
    cat << EOF
Usage: $0 --vllm-path <path> --model-path <path> [options]

Required arguments:
  --vllm-path <path>              vLLM安装路径
  --model-path <path>             模型路径

Optional arguments:
  --vllm-port <port>              vLLM服务端口 (default: $VLLM_PORT)
  --tensor-parallel-size <size>   张量并行大小 (default: $TENSOR_PARALLEL_SIZE)
  --max-num-seqs <num>            最大序列数 (default: $MAX_NUM_SEQS)
  --max-model-len <len>           最大模型长度 (default: $MAX_MODEL_LEN)
  --gpu-memory-util <ratio>       GPU内存使用率 (default: $GPU_MEMORY_UTIL)

  --enable-flexkv <0|1>           启用FlexKV (default: $ENABLE_FLEXKV)
  --flexkv-cpu-cache-gb <size>    FlexKV CPU缓存大小(GB) (default: $FLEXKV_CPU_CACHE_GB)
  --flexkv-ssd-cache-gb <size>    FlexKV SSD缓存大小(GB) (default: $FLEXKV_SSD_CACHE_GB)
  --flexkv-ssd-cache-dir <path>   FlexKV SSD缓存目录 (default: $FLEXKV_SSD_CACHE_DIR)
  --flexkv-enable-gds <0|1>       启用FlexKV GDS (default: $FLEXKV_ENABLE_GDS)

  --dataset-name <name>           数据集名称 (default: $DATASET_NAME)
  --dataset-path <path>           数据集路径 (default: $DATASET_PATH)

  --request-rate <rate>           请求速率 (default: $REQUEST_RATE)
  --workers <num>                 工作线程数 (default: $WORKERS)
  --max-turns <num>               最大轮数 (default: $MAX_TURNS)
  --concurrency <num>             并发数 (default: $CONCURRENCY)

  --log-dir <path>                日志目录 (default: $LOG_DIR)

  -h, --help                      显示帮助信息

EOF
    exit 1
}

# Initialize with default values
VLLM_PATH=""
MODEL_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm-path)
            VLLM_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-memory-util)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        --enable-flexkv)
            ENABLE_FLEXKV="$2"
            shift 2
            ;;
        --flexkv-cpu-cache-gb)
            FLEXKV_CPU_CACHE_GB="$2"
            shift 2
            ;;
        --flexkv-ssd-cache-gb)
            FLEXKV_SSD_CACHE_GB="$2"
            shift 2
            ;;
        --flexkv-ssd-cache-dir)
            FLEXKV_SSD_CACHE_DIR="$2"
            shift 2
            ;;
        --flexkv-enable-gds)
            FLEXKV_ENABLE_GDS="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --request-rate)
            REQUEST_RATE="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$VLLM_PATH" ]; then
    error "Missing required argument: --vllm-path"
    usage
fi

if [ -z "$MODEL_PATH" ]; then
    error "Missing required argument: --model-path"
    usage
fi

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SERVER_LOG="$LOG_DIR/vllm_server_${TIMESTAMP}.log"

if lsof -i:$VLLM_PORT > /dev/null 2>&1; then
    error "Port $VLLM_PORT is already in use"
    exit 1
fi


# Prepare ShareGPT Dataset
info "==================================="
info "Prepare ShareGPT dataset"
info "==================================="
CONVERTED_DATASET_FILE_PATH="$DATASET_PATH/ShareGPT_V3_converted.json"
DATASET_FILE_NAME="ShareGPT_V3_unfiltered_cleaned_split.json"

if [ -f "$CONVERTED_DATASET_FILE_PATH" ]; then
    info "Detected converted dataset: $CONVERTED_DATASET_FILE_PATH, skipping installation and conversion steps"
else
    info "Install ShareGPT dataset..."
    bash "$SCRIPT_DIR/install_sharegpt_dataset.sh" "$DATASET_PATH" "$DATASET_FILE_NAME"
    if [ $? -ne 0 ]; then
        error "Install ShareGPT dataset failed"
        if [ -n "$SERVER_PID" ]; then
            warn "Stop vLLM Server (PID: $SERVER_PID)..."
            kill $SERVER_PID 2>/dev/null || true
        fi
        exit 1
    fi
    info "✓ Install ShareGPT dataset completed"

    info "Convert ShareGPT dataset..."
    bash "$SCRIPT_DIR/convert_sharegpt_to_openai.sh" "$DATASET_PATH/$DATASET_FILE_NAME" "$CONVERTED_DATASET_FILE_PATH"
    if [ $? -ne 0 ]; then
        error "Convert ShareGPT dataset failed"
        if [ -n "$SERVER_PID" ]; then
            warn "Stop vLLM Server (PID: $SERVER_PID)..."
            kill $SERVER_PID 2>/dev/null || true
        fi
        exit 1
    fi
    info "✓ Convert ShareGPT dataset completed"
fi

info "==================================="
info "Configure FlexKV"
info "==================================="
info "FlexKV Configuration:"
info "Enable FlexKV: $ENABLE_FLEXKV"
info "FlexKV CPU Cache: $FLEXKV_CPU_CACHE_GB GB"
info "FlexKV SSD Cache: $FLEXKV_SSD_CACHE_GB GB"
info "FlexKV SSD Cache Dir: $FLEXKV_SSD_CACHE_DIR"
info "FlexKV Enable GDS: $FLEXKV_ENABLE_GDS"
export FLEXKV_CPU_CACHE_GB=$FLEXKV_CPU_CACHE_GB
export FLEXKV_SSD_CACHE_GB=$FLEXKV_SSD_CACHE_GB
export FLEXKV_SSD_CACHE_DIR=$FLEXKV_SSD_CACHE_DIR
export FLEXKV_ENABLE_GDS=$FLEXKV_ENABLE_GDS
export FLEXKV_LOG_LEVEL="INFO"
unset FLEXKV_CONFIG_PATH  # use environment variables instead of config file

info "==================================="
info "Launch vLLM Server"
info "==================================="
info "Server log file: $SERVER_LOG"

# Launch vLLM server in background
SERVER_PID=$(bash "$SCRIPT_DIR/launch_vllm_server.sh" \
        --port "$VLLM_PORT" \
        --model "$MODEL_PATH" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --enable-flexkv "$ENABLE_FLEXKV" \
        --log "$SERVER_LOG")

info "vLLM Server process started with PID: $SERVER_PID"

# Wait for server to be ready
info "Waiting for vLLM Server to be ready (it may take a few minutes)..."
MAX_WAIT=300  # Maximum wait time: 5 minutes
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo ""
        break
    fi

    if ! kill -0 $SERVER_PID 2>/dev/null; then
        error "vLLM Server process exited unexpectedly"
        error "Please check the log: $SERVER_LOG"
        tail -n 50 "$SERVER_LOG"
        exit 1
    fi

    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo -n "."
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    error "vLLM Server startup timed out"
    error "Please check the log: $SERVER_LOG"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

sleep 5
info "✓ vLLM Server is ready, PID: $SERVER_PID"

# Run Benchmark
info "==================================="
info "Run multi-turn benchmark"
info "==================================="
bash "$SCRIPT_DIR/bench_multiturn.sh" \
    --url "http://localhost:$VLLM_PORT" \
    --vllm-path "$VLLM_PATH" \
    --model "$MODEL_PATH" \
    --dataset "$CONVERTED_DATASET_FILE_PATH" \
    --dataset-name "$DATASET_NAME" \
    --workers "$WORKERS" \
    --request-rate "$REQUEST_RATE" \
    --concurrency "$CONCURRENCY" \
    --max-turns "$MAX_TURNS" \
    --log-path "$LOG_DIR" \
    --timestamp "$TIMESTAMP"

BENCHMARK_EXIT_CODE=$?

info "==================================="
info "Stop vLLM Server"
info "==================================="
if [ -n "$SERVER_PID" ]; then
    info "Stop vLLM Server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    info "✓ vLLM Server is stopped"
fi

info "Server log: $SERVER_LOG"

# Print benchmark summary
BENCHMARK_LOG="$LOG_DIR/benchmark_${TIMESTAMP}.log"
if [ -f "$BENCHMARK_LOG" ]; then
    info "==================================="
    info "Benchmark Summary"
    info "==================================="

    if grep -q "Parameters:" "$BENCHMARK_LOG"; then
        info "Benchmark log: $BENCHMARK_LOG"
        echo ""
        # Print from "Parameters:" to end of file, removing ANSI color codes
        sed -n '/Parameters:/,$ { s/\x1b\[[0-9;]*m//g; p }' "$BENCHMARK_LOG"
    else
        warn "No summary found in benchmark log"
    fi
else
    warn "Benchmark log not found: $BENCHMARK_LOG"
fi

if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    info "✓ Benchmark completed successfully"
else
    error "✗ Benchmark failed with exit code: $BENCHMARK_EXIT_CODE"
fi

exit $BENCHMARK_EXIT_CODE
