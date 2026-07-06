#!/bin/bash
# 场景1: 快速测试（100请求，64并发，按真实时间戳发送）
#bash run_mooncake_benchmark.sh /workspace/mooncake/conversation_trace.jsonl 30001 2 64 100

# 场景2: 高并发压测（不按时间戳，尽快发送）
#bash run_mooncake_benchmark.sh /workspace/mooncake/conversation_trace.jsonl 30001 4 128 500 round-robin 0

# 场景3: 测试 prefix cache（hash 策略，相同前缀发到同一节点）
#bash run_mooncake_benchmark.sh /workspace/mooncake/conversation_trace.jsonl 30001 2 64 100 hash 1.0

# 场景4: 测试跨节点 KV cache reuse（shuffle 策略，打乱后轮询分发）
#bash run_mooncake_benchmark.sh /workspace/mooncake/conversation_trace.jsonl 30001 2 64 100 shuffle 0

# 场景5: 指定 block size（每个 hash_id 对应的 token 数）
#bash run_mooncake_benchmark.sh /workspace/mooncake/conversation_trace.jsonl 30001 2 64 100 round-robin 1.0 256
# ============================================================================
# Mooncake Trace Benchmark 脚本
# 使用 Tokenizer 精确生成 token 序列，测试 KV Cache Reuse
# ============================================================================

# 使用说明
if [ $# -lt 3 ]; then
    echo "用法: $0 <Trace文件> <基础端口> <节点数量> [并发数] [请求数] [策略] [时间缩放] [block_size]"
    echo "示例: $0 /workspace/mooncake/conversation_trace.jsonl 30001 2 64 100 round-robin 1.0 512"
    echo ""
    echo "参数说明:"
    echo "  Trace文件   : Mooncake trace JSONL 文件路径 (必需)"
    echo "  基础端口    : vLLM 起始端口 (必需)"
    echo "  节点数量    : 虚拟节点数 (必需)"
    echo "  并发数      : 并发请求数 (可选，默认: 64)"
    echo "  请求数      : 总请求数 (可选，默认: 全部)"
    echo "  策略        : round-robin/random/hash/shuffle (可选，默认: round-robin)"
    echo "  时间缩放    : 时间缩放因子 (可选，默认: 1.0)"
    echo "              1.0 = 真实时间, 0.5 = 2倍速, 0 = 尽快发送"
    echo "  block_size  : 每个 hash_id 对应的 token 数 (可选，默认: 512)"
    echo ""
    echo "策略说明:"
    echo "  round-robin : 轮询分配，按顺序依次分配到各节点"
    echo "  random      : 随机分配，每个请求随机选择节点"
    echo "  hash        : 基于 hash_ids 分配，相同前缀发到同一节点（测试单节点 prefix cache）"
    echo "  shuffle     : 打乱请求顺序后轮询分发（测试跨节点 KV cache reuse）"
    echo ""
    echo "工作原理:"
    echo "  - 每个 hash_id 映射到一个固定的 token 序列（相同 hash_id = 相同 tokens）"
    echo "  - 相同 hash_ids 前缀的请求会有相同的 prompt 前缀"
    echo "  - 这样可以测试 vLLM + FlexKV 的 prefix cache 复用效果"
    echo ""
    echo "Mooncake trace 格式:"
    echo '  {"timestamp": 27482, "input_length": 6955, "output_length": 52, "hash_ids": [46, 47, ...]}'
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 参数
TRACE_FILE=$1
BASE_PORT=$2
NUM_NODES=$3
CONCURRENCY=${4:-64}
NUM_REQUESTS=${5:-}
STRATEGY=${6:-round-robin}
TIME_SCALE=${7:-1.0}
BLOCK_SIZE=${8:-500}

# 配置
SERVER_HOST="10.6.131.12"
TOKENIZER_PATH="/workspace/Qwen3-8B"
MODEL_NAME="/workspace/Qwen3-8B"

# 检查文件
if [ ! -f "$TRACE_FILE" ]; then
    echo "错误: 文件不存在: $TRACE_FILE"
    exit 1
fi

# 构建 hosts 参数
hosts=""
for ((i=0; i<NUM_NODES; i++)); do
    port=$((BASE_PORT + i))
    if [ -z "$hosts" ]; then
        hosts="${SERVER_HOST}:${port}"
    else
        hosts="${hosts},${SERVER_HOST}:${port}"
    fi
done

# 构建可选参数
EXTRA_ARGS=""
if [ -n "$NUM_REQUESTS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --num-requests $NUM_REQUESTS"
fi

# 时间缩放为 0 时使用 --no-timestamp
if [ "$TIME_SCALE" = "0" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no-timestamp"
else
    EXTRA_ARGS="$EXTRA_ARGS --time-scale $TIME_SCALE"
fi

echo "============================================================================"
echo " Mooncake Trace Benchmark 测试"
echo "============================================================================"
echo "数据文件: $TRACE_FILE"
echo "节点数量: $NUM_NODES"
echo "节点列表: $hosts"
echo "并发数: $CONCURRENCY"
if [ -n "$NUM_REQUESTS" ]; then
    echo "请求数: $NUM_REQUESTS"
else
    echo "请求数: 全部"
fi
echo "策略: $STRATEGY"
if [ "$TIME_SCALE" = "0" ]; then
    echo "时间模式: 尽快发送"
else
    echo "时间缩放: ${TIME_SCALE}x"
fi
echo "Block 大小: $BLOCK_SIZE tokens/hash_id"
echo "Tokenizer: $TOKENIZER_PATH"
echo "============================================================================"
echo ""

# 运行 benchmark
python3 ${SCRIPT_DIR}/benchmark_mooncake.py \
    --trace "$TRACE_FILE" \
    --hosts "$hosts" \
    --tokenizer "$TOKENIZER_PATH" \
    --block-size $BLOCK_SIZE \
    --concurrency $CONCURRENCY \
    --model "$MODEL_NAME" \
    --strategy $STRATEGY \
    $EXTRA_ARGS

echo ""
echo "============================================================================"
echo " Benchmark 完成"
echo "============================================================================"




