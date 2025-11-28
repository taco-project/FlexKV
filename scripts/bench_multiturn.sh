#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

VLLM_URL="http://localhost:30001"
VLLM_PATH=""
MODEL_PATH=""
DATASET_PATH=""
DATASET_NAME="sharegpt"
WORKERS=32
REQUEST_RATE=32
MAX_TURNS=10
CONCURRENCY=2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BENCHMARK_LOG_PATH="$SCRIPT_DIR/../logs"

BENCHMARK_LOG="$BENCHMARK_LOG_PATH/benchmark_${TIMESTAMP}.log"
BENCHMARK_RESULT="$BENCHMARK_LOG_PATH/multiturn_output_${WORKERS}clients_${TIMESTAMP}.json"

show_help() {
    cat << EOF
Usage: $0 [options]

Multi-turn conversation benchmark script

Options:
    -u, --url URL                vLLM service URL (default: $VLLM_URL)
    -v, --vllm-path PATH         vLLM path [required]
    -m, --model PATH             Model path [required]
    -d, --dataset PATH           Dataset path [required]
    -n, --dataset-name NAME      Dataset name (default: $DATASET_NAME)
    -w, --workers WORKERS        Worker count (default: $WORKERS)
    -p, --request-rate REQUEST_RATE        Request rate (default: $REQUEST_RATE)
    -c, --concurrency CONCURRENCY          Concurrency multiplier (default: $CONCURRENCY)
    -t, --max-turns MAX_TURNS    Maximum turns per conversation (default: $MAX_TURNS)
    -l, --log-path PATH          Log file path (default: $BENCHMARK_LOG_PATH)
    --timestamp TIMESTAMP        Timestamp for log files (default: current time in YYYYMMDD_HHMMSS format)
    -h, --help                   Show this help message

Examples:
    $0 -m /path/to/model -d /path/to/dataset -v /path/to/vllm
    $0 -m /path/to/model -d /path/to/dataset -v /path/to/vllm --url http://localhost:30002 --workers 32
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--url)
            VLLM_URL="$2"
            shift 2
            ;;
        -v|--vllm-path)
            VLLM_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -n|--dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -p|--request-rate)
            REQUEST_RATE="$2"
            shift 2
            ;;
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        -t|--max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        -l|--log-path)
            BENCHMARK_LOG_PATH="$2"
            shift 2
            ;;
        --timestamp)
            TIMESTAMP="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use --help to see help information"
            exit 1
            ;;
    esac
done

# Update log file names with timestamp
BENCHMARK_LOG="$BENCHMARK_LOG_PATH/benchmark_${TIMESTAMP}.log"
BENCHMARK_RESULT="$BENCHMARK_LOG_PATH/multiturn_output_${WORKERS}clients_${TIMESTAMP}.json"

# Check required parameters
if [ -z "$VLLM_PATH" ]; then
    error "Error: vLLM path must be specified (use -v or --vllm-path parameter)"
    echo "Use --help to see help information"
    exit 1
fi

if [ -z "$MODEL_PATH" ]; then
    error "Error: Model path must be specified (use -m or --model parameter)"
    echo "Use --help to see help information"
    exit 1
fi

if [ -z "$DATASET_PATH" ]; then
    error "Error: Dataset path must be specified (use -d or --dataset parameter)"
    echo "Use --help to see help information"
    exit 1
fi

BENCHMARK_SCRIPT="$VLLM_PATH/benchmarks/multi_turn/benchmark_serving_multi_turn.py"
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    error "Error: Benchmark script not found at $BENCHMARK_SCRIPT"
    echo "Use --help to see help information"
    exit 1
fi

mkdir -p "$(dirname "$BENCHMARK_LOG")"

total_conversations=$((WORKERS * CONCURRENCY))
total_requests=$((total_conversations * MAX_TURNS))

info "==================================="
info "Benchmark Configuration:"
info "  URL: $VLLM_URL"
info "  vLLM Path: $VLLM_PATH"
info "  Model: $MODEL_PATH"
info "  Dataset: $DATASET_PATH"
info "  Dataset Name: $DATASET_NAME"
info "  Workers: $WORKERS"
info "  Concurrency: $CONCURRENCY"
info "  Total Conversations: $total_conversations"
info "  Total Requests: $total_requests"
info "  Request Rate: $REQUEST_RATE"
info "  Max Turns: $MAX_TURNS"
info "  Log File: $BENCHMARK_LOG"
info "  Output File: $BENCHMARK_RESULT"
info "==================================="

CMD="python3 $BENCHMARK_SCRIPT \
    --input-file "$DATASET_PATH" \
    --model "$MODEL_PATH" \
    --url "$VLLM_URL" \
    --num-clients $WORKERS \
    --max-active-conversations $total_conversations \
    --max-num-requests $total_requests \
    --seed 42 \
    --request-rate $REQUEST_RATE \
    --max-turns $MAX_TURNS \
    --conversation-sampling round_robin \
    --verbose \
    --print-content \
    --output-file "$BENCHMARK_RESULT" \
    --warmup-percentages "0%,10%,25%" \
    > "$BENCHMARK_LOG" 2>&1"

info "Running command: $CMD"
eval $CMD

info "✓ All benchmarks completed"
info "End time: $(date)"
