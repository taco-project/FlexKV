# FlexKV and Dynamo Integration Guide

This document demonstrates how to integrate FlexKV with NVIDIA's [Dynamo](https://github.com/ai-dynamo/dynamo) framework and complete performance testing.

Dynamo is a framework designed by NVIDIA for large-scale distributed deployment, supporting multiple backend engines including TensorRT-LLM, vLLM, and SGLang. The KV Router is an intelligent request routing component that tracks and manages KV caches stored on different workers. It intelligently assigns requests to the most suitable worker based on the overlap between requests and KV cache, as well as the current worker load, thereby reducing expensive KV cache recomputations and improving inference efficiency. This document also explains how to integrate FlexKV into Dynamo when the KV Router is enabled.

> [!CAUTION]
> - This feature conflicts with [namespace isolation](https://github.com/taco-project/FlexKV/pull/95).
> - This feature intends not to be used with [distributed KV cache reuse](../dist_reuse/README_en.md).

## 1. Environment Setup

### Install vLLM

Refer to vLLM adaptation [README](../vllm_adapter/README_en.md).

### Install Dynamo

```bash
# 1. Clone Dynamo repo
git clone https://github.com/ai-dynamo/dynamo.git

# 2. Apply PR #5858
gh pr checkout 5858 # Make sure GitHub CLI is installed first

# 3. Install NIXL
uv pip install 'nixl[cu12]' # Or 'nixl[cu13]'

# 4. Install Dynamo
cd $DYNAMO_WORKSPACE/lib/bindings/python
maturin develop --uv
cd ../../..
uv pip install -e . # No need to specify backend as vLLM is already installed

# 5. Install nats-server and etcd
```

### FlexKV Code Preparation

```bash
git clone https://github.com/taco-project/FlexKV
```

### Install FlexKV

```bash
apt update && apt install liburing-dev

cd FlexKV && ./build.sh
```

- Refer to GPUDirect Storage (GDS) [README](../gds/README_en.md) to enable GDS.
- Refer to KV cache reuse [README](../dist_reuse/README_en.md) to enable KV cache sharing between peer nodes in a distributed setup.

### FlexKV Verification

Please refer to the test scripts in [vLLM online serving](../../docs/vllm_adapter/README_zh.md#%E7%A4%BA%E4%BE%8B).

## 2. Starting and Verifying Dynamo Services

### Starting Dynamo + FlexKV

The following example starts 4 Dynamo vLLM workers on an 8-GPU compute node with KV router enabled.

```bash
# Start NATS with JetStream
nats-server -js -a 127.0.0.1 -p 4222 --store_dir $NATS_DIR &

# Start etcd
etcd --data-dir /tmp/etcd \
  --listen-client-urls http://127.0.0.1:2379 \
  --advertise-client-urls http://YOUR_IP:2379 & # YOUR_IP is the IP address of this node.

sleep 3

export NATS_SERVER="nats://127.0.0.1:4222"
export ETCD_ENDPOINTS="http://127.0.0.1:2379"

# Start Dynamo frontend
python -m dynamo.frontend --router-mode kv &

# Define number of worker nodes
NUM_WORKERS=4

# Enable collecting KV events in FlexKV
export DYNAMO_USE_FLEXKV=1
# Configure FlexKV using environment variables, disabling config file
unset FLEXKV_CONFIG_PATH
# Adjust CPU and SSD space sizes according to your server configuration
export FLEXKV_CPU_CACHE_GB=32
export FLEXKV_SSD_CACHE_GB=128
# Use a loop to start worker nodes
for i in $(seq 0 $((NUM_WORKERS-1))); do
    # Calculate GPU device IDs
    GPU_START=$((i*2))
    GPU_END=$((i*2+1))

    if [ $i -lt $((NUM_WORKERS-1)) ]; then
        # When using multiple workers, ensure FlexKV ports are different to avoid hanging at flexkv init
        # Set FlexKV port via the `FLEXKV_SERVER_RECV_PORT` environment variable
        FLEXKV_SSD_CACHE_DIR="/data/flexkv_ssd/worker_${i}" \
        FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_${i}" \
        KV_ENDPOINT="tcp://*:2008${i}" \
        KV_EVENTS_CONFIG="$(printf '{"publisher":"zmq","topic":"kv-events","endpoint":"%s","enable_kv_cache_events":true}' "$KV_ENDPOINT")" \
        CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} \
        python3 -m dynamo.vllm \
        --model $YOUR_MODEL \
        --tensor-parallel-size 2 \
        --connector flexkv \
        --kv-events-config "$KV_EVENTS_CONFIG" &
    else
        FLEXKV_SSD_CACHE_DIR="/data/flexkv_ssd/worker_${i}" \
        FLEXKV_SERVER_RECV_PORT="ipc:///tmp/flexkv_server_${i}" \
        KV_ENDPOINT="tcp://*:2008${i}" \
        KV_EVENTS_CONFIG="$(printf '{"publisher":"zmq","topic":"kv-events","endpoint":"%s","enable_kv_cache_events":true}' "$KV_ENDPOINT")" \
        CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} \
        python3 -m dynamo.vllm \
        --model $YOUR_MODEL \
        --tensor-parallel-size 2 \
        --connector flexkv \
        --kv-events-config "$KV_EVENTS_CONFIG"
    fi
done
```

> [!NOTE]
> You can configure FlexKV using YAML or JSON files. The above configuration is provided as a simple example only. For full parameter options, please refer to [`docs/flexkv_config_reference/README_en.md`](../../docs/flexkv_config_reference/README_en.md)

### Verification

You can verify that the Dynamo service has started correctly with the following command:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": YOUR MODEL,
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

We use [`aiperf`](https://github.com/ai-dynamo/aiperf) as our benchmarking tool and [mooncake trace](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#-open-source-trace) as our dataset to evaluate the performance of Dynamo + FlexKV.

Mooncake Trace is an open-source request file saved in jsonl format. It records timestamps of request arrivals, ISL, OSL, and KV cache-related hash IDs, containing 23,608 requests over a 1-hour period. For our experiment with 4 LLaMA-70B workers, the concurrency in the mooncake trace was too high, so we sampled every 6th request from the trace to build our benchmark dataset.

`aiperf` can send requests according to the timestamps in the trace file and calculate metrics such as TTFT (Time To First Token) and TPOT (Tokens Per Output Token) for the LLM service. The command is as follows.

```bash
aiperf profile \
  --model $YOUR_MODEL \
  --tokenizer $YOUR_TOKENIZER \
  --endpoint-type 'chat' \
  --endpoint '/v1/chat/completions' \
  --streaming \
  --url http://localhost:8000 \
  --input-file $YOUR_TRACE \
  --random-seed 100 \
  -H 'Authorization: Bearer NOT USED'
```
