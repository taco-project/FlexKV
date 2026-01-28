# FlexKV 多实例模式使用指南

本文档介绍如何在单机上使用 FlexKV 多实例模式，让多个推理引擎实例共享同一个 FlexKV 缓存服务。

## 概述

FlexKV 多实例模式允许在单台机器上运行多个推理引擎实例（如 vLLM），并共享同一套 CPU/SSD KV 缓存。这种模式适用于：

- 在单机多 GPU 上部署多个独立的推理服务
- 利用 FlexKV 实现跨实例的 KV 缓存共享，提高缓存命中率

## 当前限制

> ⚠️ **重要限制**

1. **仅支持单机部署**：目前多实例模式仅支持在单台机器上运行，不支持跨节点部署
2. **并行配置必须一致**：所有推理引擎实例必须使用相同的并行配置（如 `tensor_parallel_size`、`block_size` 等）

## 环境变量配置

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `FLEXKV_INSTANCE_NUM` | int | 1 | 推理引擎实例总数 |
| `FLEXKV_INSTANCE_ID` | int | 0 | 当前实例的 ID（从 0 开始） |
| `FLEXKV_SERVER_RECV_PORT` | str | "ipc:///tmp/flexkv_server" | FlexKV 服务端口，**所有实例必须使用相同的端口** |

## 工作原理

1. 实例 ID 为 0 的进程会自动创建 FlexKV Server
2. 其他实例作为客户端连接到该 Server
3. 所有实例共享同一套 CPU/SSD 缓存，Server 负责统一管理和调度

## 使用示例

以下示例展示如何在 Dynamo 框架下启动 2 个 vLLM 推理实例，每个实例使用 2 张 GPU（tensor_parallel_size=2）。

```bash
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# 启动 nats 和 etcd（Dynamo 依赖）
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &
sleep 3

# 启动 Dynamo ingress
# --router-mode 可选: kv, round-robin, random
python -m dynamo.frontend --router-mode round-robin --http-port 8001 &

# ===== 多实例配置 =====
NUM_WORKERS=2
export FLEXKV_INSTANCE_NUM=${NUM_WORKERS}

# FlexKV 缓存配置
unset FLEXKV_CONFIG_PATH
export FLEXKV_CPU_CACHE_GB=32
export FLEXKV_SSD_CACHE_GB=128
export FLEXKV_SSD_CACHE_DIR="/data/flexkv_ssd/"
export FLEXKV_NUM_LOG_INTERVAL_REQUESTS=200

# 所有实例使用相同的 FlexKV 服务端口
export FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server"

# 模型路径
MODEL_PATH="/path/to/model"

# 启动多个推理实例
for i in $(seq 0 $((NUM_WORKERS-1))); do
    # 计算 GPU ID（假设每个实例使用 2 张连续的 GPU）
    GPU_START=$((i * 2))
    GPU_END=$((i * 2 + 1))
    
    echo "Starting vLLM worker ${i} on GPU ${GPU_START}, ${GPU_END}"
    
    # 启动推理实例
    # FLEXKV_INSTANCE_ID 用于标识每个实例
    (FLEXKV_INSTANCE_ID=${i} \
     CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} \
     python3 -m dynamo.vllm \
        --model ${MODEL_PATH} \
        --tensor_parallel_size 2 \
        --block-size 64 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 40960 \
     2>&1 | tee -a /tmp/vllm_${i}.log) &
done

echo "All workers started, waiting..."
wait
```
