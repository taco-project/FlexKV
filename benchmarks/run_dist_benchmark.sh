#!/bin/bash
# =============================================================================
# FlexKV Distributed KVCache Benchmark - One-Click Launch Script
#
# This script handles:
#   1. Check and start Redis server if not running
#   2. Set up environment variables
#   3. Run the distributed KVCache benchmark
#
# Usage:
#   bash benchmarks/run_dist_benchmark.sh [options]
#
# Options (passed through to benchmark_dist_kvcache.py):
#   --config <path>           Config YAML file (default: benchmarks/example_dist_config.yml)
#   --mode <mode>             Benchmark mode: single, multiturn, or all (default: all)
#   --batch-size <n>          Batch size (default: 1)
#   --sequence-length <n>     Sequence length (default: 1024)
#   --num-users <n>           Number of simulated users (default: 10)
#   --num-turns <n>           Number of conversation turns (default: 3)
#   --clean-redis             Clean up FlexKV & Mooncake residual data in Redis before running benchmark
#                             (removes node:*, meta:*, CPUB:block:*, SSDB:block:*, PCFSB:block:*,
#                              mooncake/*, mooncake:*, segment:*, endpoint:*, mc:* keys)
#   --clean-redis-only        Clean up FlexKV & Mooncake residual data in Redis and exit (no benchmark)
#
# Examples:
#   # Run with defaults
#   bash benchmarks/run_dist_benchmark.sh
#
#   # Custom parameters
#   bash benchmarks/run_dist_benchmark.sh --batch-size 4 --sequence-length 2048
#
#   # Multi-turn only
#   bash benchmarks/run_dist_benchmark.sh --mode multiturn --num-users 20 --num-turns 5
#
#   # Clean Redis residual data before benchmark
#   bash benchmarks/run_dist_benchmark.sh --clean-redis
#
#   # Only clean Redis residual data (no benchmark)
#   bash benchmarks/run_dist_benchmark.sh --clean-redis-only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Default config file
CONFIG_FILE="${SCRIPT_DIR}/example_dist_config.yml"
REDIS_STARTED_BY_US=false
CLEAN_REDIS=false
CLEAN_REDIS_ONLY=false

# Parse script-specific arguments and --config, pass the rest through to benchmark
BENCH_ARGS=()
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--config" ]]; then
        CONFIG_FILE="$arg"
        BENCH_ARGS+=("$arg")
        prev_arg="$arg"
        continue
    fi
    case "$arg" in
        --clean-redis)
            CLEAN_REDIS=true
            ;;
        --clean-redis-only)
            CLEAN_REDIS=true
            CLEAN_REDIS_ONLY=true
            ;;
        *)
            BENCH_ARGS+=("$arg")
            ;;
    esac
    prev_arg="$arg"
done

# ============================================
# Step 1: Parse Redis config from YAML
# ============================================
info "============================================"
info "Step 1: Parsing configuration"
info "============================================"

# Helper function to parse a YAML value using Python (handles comments, quotes, etc. correctly)
# Usage: parse_yaml_value <key> <file> [default]
parse_yaml_value() {
    local key="$1" file="$2" default="${3:-}"
    local val
    val=$(python3 -c "
import yaml, sys
with open('$file') as f:
    d = yaml.safe_load(f)
v = d.get('$key')
if v is None:
    print('$default')
else:
    print(v)
" 2>/dev/null) || val="$default"
    echo "$val"
}

# Simple YAML parser for redis config
REDIS_HOST=$(parse_yaml_value "redis_host" "$CONFIG_FILE" "127.0.0.1")
REDIS_PORT=$(parse_yaml_value "redis_port" "$CONFIG_FILE" "6379")
REDIS_PASSWORD=$(parse_yaml_value "redis_password" "$CONFIG_FILE" "")

info "Config file: ${CONFIG_FILE}"
info "Redis: ${REDIS_HOST}:${REDIS_PORT}"

# ============================================
# Step 2: Check and start Redis
# ============================================
info "============================================"
info "Step 2: Checking Redis server"
info "============================================"

check_redis() {
    local auth_args=""
    if [[ -n "$REDIS_PASSWORD" ]]; then
        auth_args="-a $REDIS_PASSWORD"
    fi
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" $auth_args ping 2>/dev/null | grep -q "PONG"
}

# Build redis-cli auth arguments (reused across the script)
REDIS_AUTH_ARGS=""
if [[ -n "$REDIS_PASSWORD" ]]; then
    REDIS_AUTH_ARGS="-a $REDIS_PASSWORD"
fi

if check_redis; then
    ok "Redis is already running at ${REDIS_HOST}:${REDIS_PORT}"
else
    warn "Redis is not running at ${REDIS_HOST}:${REDIS_PORT}"

    # Only try to start Redis if it's localhost
    if [[ "$REDIS_HOST" == "127.0.0.1" ]] || [[ "$REDIS_HOST" == "localhost" ]]; then
        if command -v redis-server &>/dev/null; then
            info "Starting Redis server on port ${REDIS_PORT}..."
            redis-server --port "$REDIS_PORT" --daemonize yes --save "" --appendonly no \
                --protected-mode no --loglevel warning
            sleep 1

            if check_redis; then
                ok "Redis server started successfully"
                REDIS_STARTED_BY_US=true
            else
                error "Failed to start Redis server"
                error "Please install Redis: sudo apt install redis-server"
                exit 1
            fi
        else
            error "redis-server not found. Please install Redis:"
            error "  sudo apt install redis-server"
            exit 1
        fi
    else
        error "Redis is not running at ${REDIS_HOST}:${REDIS_PORT}"
        error "Please start Redis on the remote host first."
        exit 1
    fi
fi

# ============================================
# Step 2.5: Clean FlexKV residual data in Redis (if requested)
# ============================================
if [[ "$CLEAN_REDIS" == "true" ]]; then
    info "============================================"
    info "Cleaning FlexKV residual data in Redis"
    info "============================================"

    clean_redis_keys() {
        local pattern="$1"
        local count=0
        local cursor=0
        while true; do
            local result
            result=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" $REDIS_AUTH_ARGS SCAN $cursor MATCH "$pattern" COUNT 500 2>/dev/null)
            cursor=$(echo "$result" | head -1)
            local keys
            keys=$(echo "$result" | tail -n +2)
            if [[ -n "$keys" ]]; then
                local batch_keys
                batch_keys=$(echo "$keys" | tr '\n' ' ')
                if [[ -n "$batch_keys" ]]; then
                    local deleted
                    deleted=$(echo "$batch_keys" | xargs redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" $REDIS_AUTH_ARGS DEL 2>/dev/null)
                    count=$((count + deleted))
                fi
            fi
            if [[ "$cursor" == "0" ]]; then
                break
            fi
        done
        echo "$count"
    }

    total_deleted=0

    # Clean node:* keys
    n=$(clean_redis_keys "node:*")
    [[ $n -gt 0 ]] && info "Deleted $n node:* key(s)"
    total_deleted=$((total_deleted + n))

    # Clean meta:* keys
    n=$(clean_redis_keys "meta:*")
    [[ $n -gt 0 ]] && info "Deleted $n meta:* key(s)"
    total_deleted=$((total_deleted + n))

    # Clean CPUB:block:* keys
    n=$(clean_redis_keys "CPUB:block:*")
    [[ $n -gt 0 ]] && info "Deleted $n CPUB:block:* key(s)"
    total_deleted=$((total_deleted + n))

    # Clean SSDB:block:* keys
    n=$(clean_redis_keys "SSDB:block:*")
    [[ $n -gt 0 ]] && info "Deleted $n SSDB:block:* key(s)"
    total_deleted=$((total_deleted + n))

    # Clean PCFSB:block:* keys
    n=$(clean_redis_keys "PCFSB:block:*")
    [[ $n -gt 0 ]] && info "Deleted $n PCFSB:block:* key(s)"
    total_deleted=$((total_deleted + n))

    # Clean Mooncake Transfer Engine residual keys
    # Mooncake uses Redis as metadata backend to store segment/endpoint info
    for mc_pattern in "mooncake/*" "mooncake:*" "segment:*" "endpoint:*" "mc:*"; do
        n=$(clean_redis_keys "$mc_pattern")
        [[ $n -gt 0 ]] && info "Deleted $n ${mc_pattern} key(s)"
        total_deleted=$((total_deleted + n))
    done

    if [[ $total_deleted -gt 0 ]]; then
        ok "Cleaned $total_deleted FlexKV & Mooncake residual key(s) from Redis"
    else
        ok "No FlexKV residual data found in Redis"
    fi

    if [[ "$CLEAN_REDIS_ONLY" == "true" ]]; then
        ok "Clean-only mode, exiting."
        exit 0
    fi
fi

# ============================================
# Step 3: Set up environment
# ============================================
info "============================================"
info "Step 3: Setting up environment"
info "============================================"

# Detect Python (prefer virtual env)
if [[ -n "$VIRTUAL_ENV" ]]; then
    # Prefer 'which python3' to get the actual resolved path in the activated venv,
    # because $VIRTUAL_ENV may point to a path that doesn't match the real filesystem
    # (e.g. symlinks, home dir aliases like ~ vs /data1/home).
    if command -v python3 &>/dev/null; then
        PYTHON="$(which python3)"
    else
        PYTHON="$VIRTUAL_ENV/bin/python3"
    fi
    if [[ ! -x "$PYTHON" ]]; then
        error "Python3 not found at $PYTHON (VIRTUAL_ENV=$VIRTUAL_ENV)"
        exit 1
    fi
    info "Using virtual env Python: $PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON="$(which python3)"
    info "Using system Python: $PYTHON"
else
    error "Python3 not found!"
    exit 1
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Set LD_LIBRARY_PATH for C++ libraries
if [[ -d "${PROJECT_ROOT}/build" ]]; then
    export LD_LIBRARY_PATH="${PROJECT_ROOT}/build:${LD_LIBRARY_PATH:-}"
fi

info "PYTHONPATH=${PYTHONPATH}"
info "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<not set>}"

# Generate mooncake config JSON and export MOONCAKE_CONFIG_PATH if P2P is enabled
ENABLE_P2P_CPU=$(grep -E "^enable_p2p_cpu:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' || echo "false")
ENABLE_P2P_SSD=$(grep -E "^enable_p2p_ssd:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}' || echo "false")

if [[ "$ENABLE_P2P_CPU" == "true" ]] || [[ "$ENABLE_P2P_SSD" == "true" ]]; then
    if [[ -z "${MOONCAKE_CONFIG_PATH:-}" ]]; then
        info "P2P enabled, generating mooncake config..."

        # Parse mooncake config from YAML using helper function
        MC_ENGINE_IP=$(parse_yaml_value "mooncake_engine_ip" "$CONFIG_FILE")
        MC_ENGINE_PORT=$(parse_yaml_value "mooncake_engine_port" "$CONFIG_FILE")
        MC_METADATA_BACKEND=$(parse_yaml_value "mooncake_metadata_backend" "$CONFIG_FILE")
        MC_METADATA_SERVER=$(parse_yaml_value "mooncake_metadata_server" "$CONFIG_FILE")
        MC_METADATA_SERVER_AUTH=$(parse_yaml_value "mooncake_metadata_server_auth" "$CONFIG_FILE")
        MC_PROTOCOL=$(parse_yaml_value "mooncake_protocol" "$CONFIG_FILE")
        MC_DEVICE_NAME=$(parse_yaml_value "mooncake_device_name" "$CONFIG_FILE")
        LOCAL_IP=$(parse_yaml_value "local_ip" "$CONFIG_FILE" "127.0.0.1")

        # Use defaults if not specified
        MC_ENGINE_IP="${MC_ENGINE_IP:-$LOCAL_IP}"
        MC_ENGINE_PORT="${MC_ENGINE_PORT:-5555}"
        MC_METADATA_BACKEND="${MC_METADATA_BACKEND:-redis}"
        MC_METADATA_SERVER="${MC_METADATA_SERVER:-redis://${REDIS_HOST}:${REDIS_PORT}}"
        MC_PROTOCOL="${MC_PROTOCOL:-tcp}"
        MC_DEVICE_NAME="${MC_DEVICE_NAME:-}"

        # Generate JSON config file
        MOONCAKE_CONFIG_FILE=$(mktemp /tmp/mooncake_config_XXXXXX.json)
        cat > "$MOONCAKE_CONFIG_FILE" <<EOF
{
  "engine_ip": "${MC_ENGINE_IP}",
  "engine_port": ${MC_ENGINE_PORT},
  "metadata_backend": "${MC_METADATA_BACKEND}",
  "metadata_server": "${MC_METADATA_SERVER}",
  "metadata_server_auth": "${MC_METADATA_SERVER_AUTH}",
  "protocol": "${MC_PROTOCOL}",
  "device_name": "${MC_DEVICE_NAME}"
}
EOF
        export MOONCAKE_CONFIG_PATH="$MOONCAKE_CONFIG_FILE"
        info "Generated mooncake config: $MOONCAKE_CONFIG_FILE"
        info "Mooncake config content:"
        cat "$MOONCAKE_CONFIG_FILE" | while IFS= read -r line; do info "  $line"; done
    else
        info "Using existing MOONCAKE_CONFIG_PATH: ${MOONCAKE_CONFIG_PATH}"
    fi
fi

# ============================================
# Step 4: Run benchmark
# ============================================
info "============================================"
info "Step 4: Running distributed KVCache benchmark"
info "============================================"

# Build the command - pass all original arguments
CMD="$PYTHON ${SCRIPT_DIR}/benchmark_dist_kvcache.py"

# If no --config was specified in args, add default
HAS_CONFIG=false
for arg in "${BENCH_ARGS[@]}"; do
    if [[ "$arg" == "--config" ]]; then
        HAS_CONFIG=true
        break
    fi
done

if [[ "$HAS_CONFIG" == "false" ]]; then
    CMD="$CMD --config ${CONFIG_FILE}"
fi

# Append benchmark arguments (script-specific args already filtered out)
if [[ ${#BENCH_ARGS[@]} -gt 0 ]]; then
    CMD="$CMD ${BENCH_ARGS[*]}"
fi

info "Running: $CMD"
echo ""

# Run the benchmark
$CMD
BENCH_EXIT_CODE=$?

# ============================================
# Cleanup
# ============================================
# Clean up temporary mooncake config file
if [[ -n "${MOONCAKE_CONFIG_FILE:-}" ]] && [[ -f "$MOONCAKE_CONFIG_FILE" ]]; then
    rm -f "$MOONCAKE_CONFIG_FILE"
    info "Cleaned up temporary mooncake config: $MOONCAKE_CONFIG_FILE"
fi

if [[ "$REDIS_STARTED_BY_US" == "true" ]]; then
    info "Stopping Redis server (started by this script)..."
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" shutdown 2>/dev/null || true
    ok "Redis stopped."
fi

if [[ $BENCH_EXIT_CODE -eq 0 ]]; then
    echo ""
    ok "Benchmark completed successfully!"
else
    echo ""
    error "Benchmark failed with exit code: $BENCH_EXIT_CODE"
fi

exit $BENCH_EXIT_CODE
