#!/bin/bash

# ============================================================================
# FlexKV 多节点 Benchmark 测试脚本
# 支持多种请求分发策略：round-robin、顺序切片、随机
# ============================================================================

set -e

# 使用说明
if [ $# -lt 3 ]; then
    echo "用法: $0 <数据集路径> <基础端口> <节点数量> [请求速率] [分发策略]"
    echo "示例: $0 /workspace/datasets/ShareGPT.json 30001 3 8 round-robin"
    echo ""
    echo "参数说明:"
    echo "  数据集路径      : ShareGPT 格式的数据集文件 (必需)"
    echo "  基础端口        : vLLM serving的起始端口 (必需)"
    echo "  节点数量        : 虚拟节点数 (必需)"
    echo "  请求速率        : 每个节点的请求速率 RPS (可选，默认: 8)"
    echo "  分发策略        : round-robin/slice/random (可选，默认: round-robin)"
    echo ""
    echo "分发策略说明:"
    echo "  round-robin  : 轮询方式，请求依次分配给不同节点"
    echo "  slice        : 顺序切片，数据集平均分成N份，每个节点处理一份"
    echo "  random       : 随机分配，每个请求随机选择一个节点"
    exit 1
fi

# ============================================================================
# 配置参数
# ============================================================================
DATASET_PATH=$1
BASE_PORT=$2
NUM_NODES=$3
REQUEST_RATE=${4:-8}
STRATEGY=${5:-round-robin}

MODEL_PATH="/workspace/Qwen3-8B"
SERVER_HOST="10.6.131.12"
NUM_PROMPTS=1024

# 创建日志目录
mkdir -p logs
current_time=$(date +"%Y-%m-%d-%H:%M:%S")

echo "============================================================================"
echo " FlexKV 多节点 Benchmark 测试"
echo "============================================================================"
echo "数据集: $DATASET_PATH"
echo "节点数量: $NUM_NODES"
echo "端口范围: $BASE_PORT - $((BASE_PORT + NUM_NODES - 1))"
echo "请求速率: $REQUEST_RATE RPS (每节点)"
echo "总请求数: $NUM_PROMPTS"
echo "分发策略: $STRATEGY"
echo "============================================================================"

# 检查数据集文件
if [ ! -f "$DATASET_PATH" ]; then
    echo "错误: 数据集文件不存在: $DATASET_PATH"
    exit 1
fi

# ============================================================================
# 策略 1: Round-Robin - 使用多进程并发测试所有节点
# ============================================================================
if [ "$STRATEGY" = "round-robin" ]; then
    echo ""
    echo "使用 Round-Robin 策略 - 并发向所有节点发送请求"
    echo ""
    
    pids=()
    for ((i=1; i<=NUM_NODES; i++)); do
        port=$((BASE_PORT + i - 1))
        log_file="logs/benchmark_node${i}_${current_time}.log"
        
        echo "启动节点 $i 的 benchmark (端口: $port)..."
        
        python3 /workspace/vllm/benchmarks/benchmark_serving.py \
            --backend vllm \
            --dataset-name sharegpt \
            --dataset-path $DATASET_PATH \
            --model $MODEL_PATH \
            --request-rate $REQUEST_RATE \
            --host $SERVER_HOST \
            --port $port \
            --endpoint /v1/completions \
            --num-prompts $NUM_PROMPTS \
            --trust-remote-code \
            > $log_file 2>&1 &
        
        pids+=($!)
        echo "  节点 $i benchmark 已启动 (PID: ${pids[-1]}, 日志: $log_file)"
        
        # 稍微延迟避免同时启动
        sleep 1
    done
    
    echo ""
    echo "等待所有 benchmark 完成..."
    
    # 等待所有后台任务完成
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    echo ""
    echo "所有节点 benchmark 完成!"

# ============================================================================
# 策略 2: Slice - 顺序切片，每个节点处理数据集的一部分
# ============================================================================
elif [ "$STRATEGY" = "slice" ]; then
    echo ""
    echo "使用 Slice 策略 - 将数据集切片分配给不同节点"
    echo ""
    
    # 计算每个节点处理的请求数
    prompts_per_node=$((NUM_PROMPTS / NUM_NODES))
    
    pids=()
    for ((i=1; i<=NUM_NODES; i++)); do
        port=$((BASE_PORT + i - 1))
        log_file="logs/benchmark_node${i}_${current_time}.log"
        
        # 计算偏移量
        offset=$(( (i - 1) * prompts_per_node ))
        
        # 最后一个节点处理剩余的所有请求
        if [ $i -eq $NUM_NODES ]; then
            num_prompts=$((NUM_PROMPTS - offset))
        else
            num_prompts=$prompts_per_node
        fi
        
        echo "启动节点 $i 的 benchmark (端口: $port, 请求: $offset-$((offset + num_prompts)))..."
        
        # 使用 Python 脚本切片数据集
        temp_dataset="/tmp/dataset_node${i}_${current_time}.json"
        python3 -c "
import json
import sys

with open('$DATASET_PATH', 'r') as f:
    data = json.load(f)

conversations = data if isinstance(data, list) else data.get('conversations', [])
slice_data = conversations[$offset:$((offset + num_prompts))]

with open('$temp_dataset', 'w') as f:
    json.dump(slice_data, f)
    
print(f'切片数据集已保存: {len(slice_data)} 条记录')
"
        
        python3 /workspace/vllm/benchmarks/benchmark_serving.py \
            --backend vllm \
            --dataset-name sharegpt \
            --dataset-path $temp_dataset \
            --model $MODEL_PATH \
            --request-rate $REQUEST_RATE \
            --host $SERVER_HOST \
            --port $port \
            --endpoint /v1/completions \
            --num-prompts $num_prompts \
            --trust-remote-code \
            > $log_file 2>&1 &
        
        pids+=($!)
        echo "  节点 $i benchmark 已启动 (PID: ${pids[-1]}, 日志: $log_file)"
        
        sleep 1
    done
    
    echo ""
    echo "等待所有 benchmark 完成..."
    
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    # 清理临时文件
    rm -f /tmp/dataset_node*_${current_time}.json
    
    echo ""
    echo "所有节点 benchmark 完成!"

# ============================================================================
# 策略 3: Random - 随机分配
# ============================================================================
elif [ "$STRATEGY" = "random" ]; then
    echo ""
    echo "使用 Random 策略 - 使用自定义脚本随机分发请求"
    echo ""
    
    # 生成端口列表
    ports=""
    for ((i=1; i<=NUM_NODES; i++)); do
        port=$((BASE_PORT + i - 1))
        if [ -z "$ports" ]; then
            ports="$port"
        else
            ports="$ports,$port"
        fi
    done
    
    log_file="logs/benchmark_random_${current_time}.log"
    
    # 创建随机分发的 Python 脚本
    cat > /tmp/benchmark_random_dispatch.py <<'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
随机分发 Benchmark 脚本
将请求随机分配给不同的 vLLM 节点
"""
import asyncio
import aiohttp
import json
import random
import time
import argparse
from typing import List

async def send_request(session, url, prompt, semaphore):
    """发送单个请求"""
    async with semaphore:
        payload = {
            "model": "dummy",  # 会被服务器忽略
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        
        start_time = time.time()
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                result = await response.json()
                latency = time.time() - start_time
                return {
                    "success": True,
                    "latency": latency,
                    "url": url
                }
        except Exception as e:
            latency = time.time() - start_time
            return {
                "success": False,
                "latency": latency,
                "url": url,
                "error": str(e)
            }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--ports", type=str, required=True, help="逗号分隔的端口列表")
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--request-rate", type=float, default=8.0)
    args = parser.parse_args()
    
    # 解析端口列表
    ports = [int(p.strip()) for p in args.ports.split(",")]
    urls = [f"http://{args.host}:{port}/v1/completions" for port in ports]
    
    print(f"目标节点: {urls}")
    
    # 加载数据集
    with open(args.dataset_path, 'r') as f:
        dataset = json.load(f)
    
    conversations = dataset if isinstance(dataset, list) else dataset.get('conversations', [])
    prompts = [conv['conversations'][0]['value'] if 'conversations' in conv else conv.get('prompt', '') 
               for conv in conversations[:args.num_prompts]]
    
    print(f"加载了 {len(prompts)} 个提示")
    
    # 创建信号量控制并发
    max_concurrent = 100
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 发送请求
    async with aiohttp.ClientSession() as session:
        tasks = []
        interval = 1.0 / (args.request_rate * len(ports))  # 总请求速率
        
        start_time = time.time()
        for i, prompt in enumerate(prompts):
            # 随机选择一个节点
            url = random.choice(urls)
            
            task = send_request(session, url, prompt, semaphore)
            tasks.append(task)
            
            # 控制请求速率
            if i < len(prompts) - 1:
                await asyncio.sleep(interval)
        
        print(f"所有请求已发送，等待完成...")
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r["success"])
    latencies = [r["latency"] for r in results if r["success"]]
    
    print("\n" + "="*70)
    print("Benchmark 结果:")
    print("="*70)
    print(f"总请求数: {len(results)}")
    print(f"成功数: {success_count}")
    print(f"失败数: {len(results) - success_count}")
    print(f"总耗时: {total_time:.2f}s")
    
    if latencies:
        latencies.sort()
        print(f"\n延迟统计:")
        print(f"  平均: {sum(latencies)/len(latencies):.3f}s")
        print(f"  中位数: {latencies[len(latencies)//2]:.3f}s")
        print(f"  P95: {latencies[int(len(latencies)*0.95)]:.3f}s")
        print(f"  P99: {latencies[int(len(latencies)*0.99)]:.3f}s")
    
    # 按节点统计
    print(f"\n每个节点的请求分布:")
    for url in urls:
        count = sum(1 for r in results if r["url"] == url)
        print(f"  {url}: {count} 请求")
    
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
PYTHON_SCRIPT
    
    chmod +x /tmp/benchmark_random_dispatch.py
    
    echo "启动随机分发 benchmark..."
    python3 /tmp/benchmark_random_dispatch.py \
        --dataset-path $DATASET_PATH \
        --host $SERVER_HOST \
        --ports $ports \
        --num-prompts $NUM_PROMPTS \
        --request-rate $REQUEST_RATE \
        2>&1 | tee $log_file
    
    echo ""
    echo "Random 策略 benchmark 完成!"
    
else
    echo "错误: 不支持的分发策略: $STRATEGY"
    echo "支持的策略: round-robin, slice, random"
    exit 1
fi

# ============================================================================
# 汇总结果
# ============================================================================
echo ""
echo "============================================================================"
echo " Benchmark 测试完成!"
echo "============================================================================"
echo "分发策略: $STRATEGY"
echo "节点数量: $NUM_NODES"
echo ""
echo "日志文件:"
ls -lh logs/*_${current_time}.log 2>/dev/null || true
echo ""
echo "查看汇总结果:"
if [ "$STRATEGY" = "random" ]; then
    echo "  cat logs/benchmark_random_${current_time}.log"
else
    echo "  tail -n 50 logs/benchmark_node*_${current_time}.log"
fi
echo "============================================================================"


