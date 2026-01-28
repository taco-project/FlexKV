# FlexKV Multi-Instance Mode Guide

This document describes how to use FlexKV multi-instance mode on a single machine, allowing multiple inference engine instances to share the same FlexKV cache service.

## Overview

FlexKV multi-instance mode allows running multiple inference engine instances (e.g., vLLM) on a single machine while sharing the same CPU/SSD KV cache. This mode is suitable for:

- Deploying multiple independent inference services on a single machine with multiple GPUs
- Leveraging FlexKV for cross-instance KV cache sharing to improve cache hit rate

## Current Limitations

> ⚠️ **Important Limitations**

1. **Single-machine only**: Multi-instance mode currently only supports running on a single machine; cross-node deployment is not supported
2. **Identical parallel configuration required**: All inference engine instances must use the same parallel configuration (e.g., `tensor_parallel_size`, `block_size`, etc.)

## Environment Variable Configuration

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `FLEXKV_INSTANCE_NUM` | int | 1 | Total number of inference engine instances |
| `FLEXKV_INSTANCE_ID` | int | 0 | Current instance ID (starting from 0) |
| `FLEXKV_SERVER_RECV_PORT` | str | "ipc:///tmp/flexkv_server" | FlexKV server port, **all instances must use the same port** |

## How It Works

1. The process with instance ID 0 automatically creates the FlexKV Server
2. Other instances connect to the Server as clients
3. All instances share the same CPU/SSD cache, with the Server handling unified management and scheduling

## Usage Example

The following example demonstrates how to start 2 vLLM inference instances under the Dynamo framework, each using 2 GPUs (tensor_parallel_size=2).

```bash
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Start nats and etcd (Dynamo dependencies)
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &
sleep 3

# Start Dynamo ingress
# --router-mode options: kv, round-robin, random
python -m dynamo.frontend --router-mode round-robin --http-port 8001 &

# ===== Multi-Instance Configuration =====
NUM_WORKERS=2
export FLEXKV_INSTANCE_NUM=${NUM_WORKERS}

# FlexKV cache configuration
unset FLEXKV_CONFIG_PATH
export FLEXKV_CPU_CACHE_GB=32
export FLEXKV_SSD_CACHE_GB=128
export FLEXKV_SSD_CACHE_DIR="/data/flexkv_ssd/"
export FLEXKV_NUM_LOG_INTERVAL_REQUESTS=200

# All instances use the same FlexKV server port
export FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server"

# Model path
MODEL_PATH="/path/to/model"

# Start multiple inference instances
for i in $(seq 0 $((NUM_WORKERS-1))); do
    # Calculate GPU IDs (assuming each instance uses 2 consecutive GPUs)
    GPU_START=$((i * 2))
    GPU_END=$((i * 2 + 1))
    
    echo "Starting vLLM worker ${i} on GPU ${GPU_START}, ${GPU_END}"
    
    # Start inference instance
    # FLEXKV_INSTANCE_ID identifies each instance
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
