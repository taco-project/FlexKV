#!/bin/bash
# 场景1: 快速性能测试（1000请求，64并发）
#bash run_jsonl_benchmark.sh /workspace/chansi_a_day_0815.jsonl 30001 2 8 1000

# 场景2: 高并发压测（256并发，4节点）
#bash run_jsonl_benchmark.sh /workspace/chansi_a_day_0815.jsonl 30001 4 256 5000

# 场景3: 持续压测整个数据集（830K条）
#bash run_jsonl_benchmark_continuous.sh /workspace/chansi_a_day_0815.jsonl 30001 2 128
# ============================================================================
# JSONL 格式数据 Benchmark 脚本
# 支持多节点负载均衡测试
# ============================================================================

# 使用说明
if [ $# -lt 3 ]; then
    echo "用法: $0 <JSONL文件> <基础端口> <节点数量> [并发数] [请求数] [策略]"
    echo "示例: $0 /workspace/chansi_a_day_0815.jsonl 30001 2 64 1000 shuffle"
    echo ""
    echo "参数说明:"
    echo "  JSONL文件   : 数据文件路径 (必需)"
    echo "  基础端口    : vLLM 起始端口 (必需)"
    echo "  节点数量    : 虚拟节点数 (必需)"
    echo "  并发数      : 并发请求数 (可选，默认: 64)"
    echo "  请求数      : 总请求数 (可选，默认: 1000)"
    echo "  策略        : round-robin/random/shuffle (可选，默认: round-robin)"
    echo ""
    echo "策略说明:"
    echo "  round-robin : 轮询分配，按顺序依次分配到各节点"
    echo "  random      : 随机分配，每个请求随机选择节点"
    echo "  shuffle     : 打乱请求顺序后再轮询分配（推荐用于测试 KV Cache 共享）"
    exit 1
fi

# 参数
JSONL_FILE=$1
BASE_PORT=$2
NUM_NODES=$3
CONCURRENCY=${4:-64}
NUM_REQUESTS=${5:-1000}
STRATEGY=${6:-round-robin}

# 配置
SERVER_HOST="10.6.131.12"
MAX_TOKENS=1024
MODEL_NAME="/workspace/Qwen3-8B"

# 检查文件
if [ ! -f "$JSONL_FILE" ]; then
    echo "错误: 文件不存在: $JSONL_FILE"
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

echo "============================================================================"
echo " JSONL Benchmark 测试"
echo "============================================================================"
echo "数据文件: $JSONL_FILE"
echo "节点数量: $NUM_NODES"
echo "节点列表: $hosts"
echo "并发数: $CONCURRENCY"
echo "请求数: $NUM_REQUESTS"
echo "策略: $STRATEGY"
echo "最大输出: $MAX_TOKENS tokens"
echo "============================================================================"
echo ""

# 运行 benchmark
python3 benchmark_jsonl.py \
    --jsonl "$JSONL_FILE" \
    --hosts "$hosts" \
    --num-requests $NUM_REQUESTS \
    --concurrency $CONCURRENCY \
    --max-tokens $MAX_TOKENS \
    --model "$MODEL_NAME" \
    --strategy $STRATEGY

echo ""
echo "============================================================================"
echo " Benchmark 完成"
echo "============================================================================"

