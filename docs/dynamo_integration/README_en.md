# FlexKV and Dynamo Integration Guide

This document demonstrates how to integrate FlexKV with NVIDIA's [Dynamo](https://github.com/ai-dynamo/dynamo) framework and complete performance testing.

Dynamo is a framework designed by NVIDIA for large-scale distributed deployment, supporting multiple backend engines including TensorRT-LLM, vLLM, and SGLang. The KV Router is an intelligent request routing component that tracks and manages KV caches stored on different workers. It intelligently assigns requests to the most suitable worker based on the overlap between requests and KV cache, as well as the current worker load, thereby reducing expensive KV cache recomputations and improving inference efficiency. This document also explains how to integrate FlexKV into Dynamo when the KV Router is enabled.

## 1. Environment Setup

### Dynamo Image

We use Dynamo 0.4.1 image with vLLM backend, which includes vLLM 0.10.1.1.

```bash
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1
```

### FlexKV Code Preparation

```bash
git clone -b dev https://github.com/taco-project/FlexKV
```

### Install FlexKV

```bash
apt update && apt install liburing-dev

cd FlexKV && ./build.sh
```

### vLLM Apply Patch

```bash
# Navigate to FlexKV directory
git apply examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch
```

### FlexKV Verification

Please refer to the test scripts in [vLLM online serving](https://github.com/taco-project/FlexKV/blob/dev/docs/vllm_adapter/README_zh.md#%E7%A4%BA%E4%BE%8B).

## 2. Dynamo Modifications

### kv_transfer_config

To integrate with FlexKV, you need to modify the kv_transfer_config in the Dynamo image. Change lines 245-248 in /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/args.py to:

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="FlexKVConnectorV1", kv_role="kv_both"
)
logger.info("Using FlexKVConnectorV1 configuration")
```

### CPU Offloading

In Dynamo, the KV router updates its KV index by receiving events sent from workers, allowing it to track the KV cache status on each worker. When CPU offloading is enabled in FlexKV, we remove [BlockRemove](https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/v1/core/block_pool.py#L221) in vLLM, allowing FlexKV to cache all KV blocks through CPU during the serving process. This ensures that the index maintained by the KV router accurately reflects the actual index in FlexKV.

## 3. Starting and Verifying Dynamo Services

### Starting Dynamo + FlexKV

```bash
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Start nats and etcd
nats-server -js &

etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

sleep 3

# run ingress, set routing mode with --router-mode, options include kv, round-robin, random
python -m dynamo.frontend --router-mode kv --http-port 8000 &

# Define number of worker nodes
NUM_WORKERS=4

# When using multiple workers, ensure FlexKV ports are different to avoid hanging at flexkv init
# Adjust num_cpu_blocks and num_ssd_blocks values according to your server configuration
for i in $(seq 0 $((NUM_WORKERS-1))); do
    cat <<EOF > ./flexkv_config_${i}.json
{
        "enable_flexkv": true,
        "server_recv_port": "ipc:///tmp/flexkv_${i}_test",
        "cache_config": {
                        "enable_cpu": true,
                        "enable_ssd": false,
                        "enable_remote": false,
                        "use_gds": false,
                        "enable_trace": false,
                        "ssd_cache_iouring_entries": 512,
                        "tokens_per_block": 64,
                        "num_cpu_blocks": 10240,
                        "num_ssd_blocks": 256000,
                        "ssd_cache_dir": "/data/flexkv_ssd/",
                        "evict_ratio": 0.05,
                        "index_accel": true

        },
        "num_log_interval_requests": 200
}
EOF
done

# Use a loop to start worker nodes
for i in $(seq 0 $((NUM_WORKERS-1))); do
    # Calculate GPU device IDs
    GPU_START=$((i*2))
    GPU_END=$((i*2+1))
    
    if [ $i -lt $((NUM_WORKERS-1)) ]; then
        FLEXKV_CONFIG_PATH="./flexkv_config_${i}.json" CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} python3 -m dynamo.vllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tensor_parallel_size 2  --block-size 64 --gpu-memory-utilization 0.9 --max-model-len 100310 &
    else
        FLEXKV_CONFIG_PATH="./flexkv_config_${i}.json" CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} python3 -m dynamo.vllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tensor_parallel_size 2  --block-size 64 --gpu-memory-utilization 0.9 --max-model-len 100310
    fi
done
```

### Verification

You can verify that the Dynamo service has started correctly with the following command:
```bash
curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "messages": [
    {
        "role": "user",
        "content": "Tell me a joke."
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```

## 4. Benchmark

We use [genai-perf](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf) as our benchmark tool and [mooncake trace](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#-open-source-trace) as our dataset to evaluate the performance of Dynamo + FlexKV.

Mooncake Trace is an open-source request file saved in jsonl format. It records timestamps of request arrivals, ISL, OSL, and KV cache-related hash IDs, containing 23,608 requests over a 1-hour period. For our experiment with 4 LLaMA-70B workers, the concurrency in the mooncake trace was too high, so we sampled every 6th request from the trace to build our benchmark dataset.

genai-perf can send requests according to the timestamps in the trace file and calculate metrics such as TTFT (Time To First Token) and TPOT (Tokens Per Output Token) for the LLM service. The command is as follows. Please use genai-perf==0.0.13, as newer versions have a bug in timestamp parsing.

```bash
genai-perf profile   --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-70B  --endpoint-type chat   --endpoint /v1/chat/completions --streaming  --url http://localhost:8000  --input-file payload:mooncake_trace_1_6.jsonl --random-seed 100  -v  -H 'Authorization: Bearer NOT USED'  -H 'Accept: text/event-stream'   -- --stability-percentage 99
```