#!/bin/bash

# ============================================================================
# 多轮对话 Benchmark 测试脚本
# 使用 benchmark_serving_multi_turn.py 测试完整的多轮对话性能
# 与 benchmark_serving.py 不同，这个脚本会测试完整的对话流程
# ============================================================================

# 使用说明
if [ $# -eq 0 ]; then
    echo "Usage: $0 <port> [request_rate]"
    echo "Example: $0 30001 8"
    echo ""
    echo "Parameters:"
    echo "  port         : Server port (required)"
    echo "  request_rate : Request rate in RPS (optional, default: 8)"
    exit 1
fi

# 配置参数 (可以根据需要修改)
DATASET_PATH=$2 #"/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
MODEL_PATH="/workspace/Qwen3-8B"
SERVER_HOST="10.6.131.12"
SERVER_PORT=$1  # 第一个参数作为端口
REQUEST_RATE=8
MAX_TURNS=8  # 10

current_time=$(date +"%Y-%m-%d-%H:%M:%S")
# 多轮对话benchmark配置
for workers in 16; do
    # 计算并发倍数
    concurrency_multiplier=2
    if [ $workers -le 64 ]; then
        concurrency_multiplier=4
    fi
    
    # 计算总的对话数量 (每个对话可能有多轮)
    total_conversations=$((workers*concurrency_multiplier))
    
    # 计算总请求数 (考虑多轮对话，增加请求数以支持多轮)
    # 设置足够大的数值，让多轮对话有时间进行
    total_requests=$((total_conversations*10))
    
    python3 /workspace/vllm/benchmarks/benchmark_serving.py \
        --backend vllm \
        --dataset-name sharegpt \
        --dataset-path $DATASET_PATH \
        --model $MODEL_PATH \
        --request-rate $REQUEST_RATE \
        --host $SERVER_HOST \
        --port $SERVER_PORT \
        --endpoint /v1/completions \
        --num-prompts 1024 \
        --trust-remote-code \
        2>&1 | tee "benchmark_port${SERVER_PORT}_${workers}clients_${current_time}.log"
        
    echo "============================================================================"
    echo " 对话benchmark测试完成!"
    echo " 结果文件:"
    echo "   - 日志文件: benchmark_port${SERVER_PORT}_${workers}clients_${current_time}.log"
    echo "============================================================================"
    echo ""
done