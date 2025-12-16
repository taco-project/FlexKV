#!/bin/bash

# ============================================================================
# FlexKV 多虚拟节点一键启动脚本
# 在单个物理节点上启动多个虚拟节点的 vLLM serving
# bash start_multi_node_serving.sh 2 30001 true
# ============================================================================

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 使用说明
if [ $# -lt 1 ]; then
    echo "用法: $0 <节点数量> [基础端口] [是否清理Redis]"
    echo "示例: $0 3 30001 true"
    echo ""
    echo "参数说明:"
    echo "  节点数量        : 要启动的虚拟节点数 (必需)"
    echo "  基础端口        : vLLM serving的起始端口 (可选，默认: 30001)"
    echo "  是否清理Redis   : true/false (可选，默认: true)"
    echo ""
    echo "说明:"
    echo "  - 各节点端口会从基础端口开始递增 (30001, 30002, 30003...)"
    echo "  - 配置文件会自动生成到 node<N>.yml"
    echo "  - 日志文件会保存到 logs/node<N>_serving.log"
    exit 1
fi

# ============================================================================
# 配置参数
# ============================================================================
NUM_NODES=$1
BASE_PORT=${2:-30001}
CLEAN_REDIS=${3:-true}

# 固定配置
MODEL_PATH="/workspace/Qwen3-8B"
MAX_NUM_SEQS=128
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=8192

# GPU 配置
TOTAL_GPUS=8  # 总共可用的GPU数量（0-7）

# 检查节点数量是否能均匀分配GPU
if [ $((TOTAL_GPUS % NUM_NODES)) -ne 0 ]; then
    echo "错误: 节点数量 ($NUM_NODES) 必须是 GPU 总数 ($TOTAL_GPUS) 的因数"
    echo "支持的节点数量: 1, 2, 4, 8"
    exit 1
fi

# 根据节点数量计算每个节点的TP大小和显存利用率
TENSOR_PARALLEL_SIZE=$((TOTAL_GPUS / NUM_NODES))
# GPU显存利用率 = 0.5 / TP_SIZE (每个节点总共使用单卡显存的0.5倍)
GPU_MEMORY_UTIL=$(python3 -c "print(round(0.5 / $TENSOR_PARALLEL_SIZE, 2))")

# 网络配置
LOCAL_IP="10.6.131.12"
REDIS_HOST="10.6.131.12"
REDIS_PORT=6379
REDIS_PASSWORD="redis-serving-passwd"

# 端口基础配置
BASE_ZMQ_PORT=5454
BASE_RPC_PORT=12841
BASE_MOONCAKE_ENGINE_PORT=12345

# Redis 配置（所有节点共享）
# 6379: FlexKV 后端
# 6380: Mooncake 后端

# RDMA 设备列表（如果没有足够的设备，可以留空或重复使用）
RDMA_DEVICES=("mlx5_1" "mlx5_2" "mlx5_3" "mlx5_4" "mlx5_5" "mlx5_6" "mlx5_7" "mlx5_0")

# 创建日志目录
mkdir -p "${SCRIPT_DIR}/logs"

echo "============================================================================"
echo " FlexKV 多虚拟节点启动"
echo "============================================================================"
echo "节点数量: $NUM_NODES"
echo "基础端口: $BASE_PORT"
echo "端口范围: $BASE_PORT - $((BASE_PORT + NUM_NODES - 1))"
echo "是否清理Redis: $CLEAN_REDIS"
echo ""
echo "GPU 配置:"
echo "  总GPU数: $TOTAL_GPUS"
echo "  每节点TP: $TENSOR_PARALLEL_SIZE"
echo "  显存利用率: $GPU_MEMORY_UTIL (0.5 / $TENSOR_PARALLEL_SIZE)"
echo "============================================================================"

# ============================================================================
# 清理 Redis (可选)
# ============================================================================
if [ "$CLEAN_REDIS" = "true" ]; then
    echo ""
    echo "正在清理 Redis..."
    redis-cli -h $REDIS_HOST -p 6379 -a $REDIS_PASSWORD FLUSHALL 2>/dev/null || true
    redis-cli -h $REDIS_HOST -p 6380 -a $REDIS_PASSWORD FLUSHALL 2>/dev/null || true
    echo "Redis 清理完成 (端口 6379, 6380)"
fi

# ============================================================================
# 为每个虚拟节点生成配置并启动服务
# ============================================================================
for ((i=1; i<=NUM_NODES; i++)); do
    node_id=$i
    vllm_port=$((BASE_PORT + i - 1))
    zmq_port=$((BASE_ZMQ_PORT + (i - 1) * 4))
    rpc_port=$((BASE_RPC_PORT + i - 1))
    mooncake_engine_port=$((BASE_MOONCAKE_ENGINE_PORT + i - 1))
    
    # 获取 RDMA 设备名称
    device_index=$((i - 1))
    rdma_device="${RDMA_DEVICES[$device_index]}"
    
    # 计算该节点的GPU范围
    gpu_start=$((device_index * TENSOR_PARALLEL_SIZE))
    gpu_end=$((gpu_start + TENSOR_PARALLEL_SIZE - 1))
    
    # 构建CUDA_VISIBLE_DEVICES字符串
    cuda_devices=""
    for ((g=gpu_start; g<=gpu_end; g++)); do
        if [ -z "$cuda_devices" ]; then
            cuda_devices="$g"
        else
            cuda_devices="$cuda_devices,$g"
        fi
    done
    
    config_file="${SCRIPT_DIR}/node${node_id}.yml"
    mooncake_config_file="${SCRIPT_DIR}/mooncake_config_node${node_id}.json"
    log_file="${SCRIPT_DIR}/logs/node${node_id}_serving.log"
    
    echo ""
    echo "------------------------------------------------------------"
    echo "准备启动节点 $node_id:"
    echo "  - vLLM 端口: $vllm_port"
    echo "  - GPU 分配: [$cuda_devices] (TP=$TENSOR_PARALLEL_SIZE)"
    echo "  - GPU 显存利用率: $GPU_MEMORY_UTIL"
    echo "  - ZMQ 端口: $zmq_port"
    echo "  - RPC 端口: $rpc_port"
    echo "  - Mooncake Engine 端口: $mooncake_engine_port"
    echo "  - RDMA 设备: $rdma_device"
    echo "  - Redis 端口: 6379 (FlexKV), 6380 (Mooncake)"
    echo "  - FlexKV 配置: $config_file"
    echo "  - Mooncake 配置: $mooncake_config_file"
    echo "  - 日志文件: $log_file"
    echo "------------------------------------------------------------"
    
    # 生成 FlexKV 配置文件（所有节点都使用 Redis 6379 作为 FlexKV 后端）
    cat > $config_file <<EOF
enable_p2p_cpu: true
enable_p2p_ssd: true
cpu_cache_gb: 50
ssd_cache_gb: 100
local_zmq_ip: "$LOCAL_IP"
local_zmq_port: $zmq_port
redis_host: "$REDIS_HOST"
redis_port: 6379
local_ip: "$LOCAL_IP"
redis_password: "$REDIS_PASSWORD"
ssd_cache_dir: /workspace/FlexKV/flexkv_ssd${node_id}/
EOF
    
    # 生成 Mooncake 配置文件（所有节点都使用 Redis 6380 作为 metadata_server）
    cat > $mooncake_config_file <<EOF
{
    "engine_ip": "$LOCAL_IP",
    "engine_port": $mooncake_engine_port,
    "metadata_backend": "redis",
    "metadata_server": "redis://$REDIS_HOST:6380",
    "metadata_server_auth": "$REDIS_PASSWORD",
    "protocol": "rdma",
    "device_name": "$rdma_device"
}
EOF
    
    # 创建 SSD 缓存目录
    mkdir -p /workspace/FlexKV/flexkv_ssd${node_id}/
    
    # 设置环境变量并启动 vLLM serving
    export FLEXKV_CONFIG_PATH="${config_file}"
    export FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_${node_id}"
    export FLEXKV_LEASE_TTL_MS=30000
    export LD_LIBRARY_PATH=/workspace/sdk/:$LD_LIBRARY_PATH
    export PYTHONPATH=/workspace/sdk/:$PYTHONPATH
    export MOONCAKE_CONFIG_PATH="${mooncake_config_file}"
    export MC_REDIS_PASSWORD="$REDIS_PASSWORD"
    export MC_LEGACY_RPC_PORT_BINDING=$rpc_port
    
    # 后台启动 vLLM serving
    nohup env \
        CUDA_VISIBLE_DEVICES="$cuda_devices" \
        FLEXKV_CONFIG_PATH="${config_file}" \
        FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_${node_id}" \
        FLEXKV_LEASE_TTL_MS=30000 \
        LD_LIBRARY_PATH=/workspace/sdk/:$LD_LIBRARY_PATH \
        PYTHONPATH=/workspace/sdk/:$PYTHONPATH \
        MOONCAKE_CONFIG_PATH="${mooncake_config_file}" \
        MC_REDIS_PASSWORD="$REDIS_PASSWORD" \
        MC_LEGACY_RPC_PORT_BINDING=$rpc_port \
        VLLM_USE_V1=1 python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
            --trust-remote-code \
            --port $vllm_port \
            --max-num-seqs $MAX_NUM_SEQS \
            --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
            --max_model_len $MAX_MODEL_LEN \
            --max-seq-len-to-capture $MAX_MODEL_LEN \
            --gpu-memory-utilization $GPU_MEMORY_UTIL \
            --enable-chunked-prefill \
            --enable-prefix-caching \
            --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}' \
        > $log_file 2>&1 &
    
    echo "节点 $node_id 已在后台启动 (PID: $!)"
    echo "查看日志: tail -f $log_file"
    
    # 稍微延迟，避免资源竞争
    sleep 2
done

echo ""
echo "============================================================================"
echo " 所有节点启动完成!"
echo "============================================================================"
echo ""
echo "查看所有节点状态:"
echo "  ps aux | grep vllm.entrypoints.openai.api_server | grep -v grep"
echo ""
echo "停止所有节点:"
echo "  pkill -f 'vllm.entrypoints.openai.api_server'"
echo ""
echo "查看某个节点日志:"
echo "  tail -f logs/node<N>_serving.log"
echo ""
echo "节点配置列表:"
for ((i=1; i<=NUM_NODES; i++)); do
    port=$((BASE_PORT + i - 1))
    mooncake_port=$((BASE_MOONCAKE_ENGINE_PORT + i - 1))
    
    # 重新计算GPU范围
    device_index=$((i - 1))
    gpu_start=$((device_index * TENSOR_PARALLEL_SIZE))
    gpu_end=$((gpu_start + TENSOR_PARALLEL_SIZE - 1))
    cuda_devices_display=""
    for ((g=gpu_start; g<=gpu_end; g++)); do
        if [ -z "$cuda_devices_display" ]; then
            cuda_devices_display="$g"
        else
            cuda_devices_display="$cuda_devices_display,$g"
        fi
    done
    
    echo "  节点 $i:"
    echo "    - vLLM API: http://${LOCAL_IP}:${port}"
    echo "    - GPU: [$cuda_devices_display] (TP=$TENSOR_PARALLEL_SIZE)"
    echo "    - Mooncake Engine: ${mooncake_port}"
    echo "    - FlexKV 配置: node${i}.yml"
    echo "    - Mooncake 配置: mooncake_config_node${i}.json"
done
echo ""
echo "共享 Redis 端口:"
echo "  - FlexKV 后端: ${REDIS_HOST}:6379"
echo "  - Mooncake 后端: ${REDIS_HOST}:6380"
echo "============================================================================"

