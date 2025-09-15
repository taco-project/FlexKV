# FlexKV 与 Dynamo 集成指南

该文档展示了如何将FlexKV和NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo) 框架集成，并完成性能测试的步骤。

Dynamo是NVIDIA专为大规模分离式部署而设计的框架，支持TensorRT-LLM, vLLM, SGLang等多个后端引擎。其中KV 路由器（KV Router）是一个智能的请求路由组件, 它能够追踪和管理存储在不同worker上的 KV cache，并根据请求与缓存的重叠程度和worker当前负载，智能地将请求分配给最合适的 GPU 节点，从而减少昂贵的 KV 缓存重新计算，提高推理效率。文档也介绍了如何在开启KV Router时，将FlexKV集成进Dynamo。

## 1. 环境准备

### Dynamo 镜像

该文档使用的是后端为vLLM的Dynamo 0.4.1 镜像，内置了vLLM 0.10.1.1。

```bash
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1
```

### FlexKV代码准备

```bash
git clone https://github.com/taco-project/FlexKV
```

### 安装 FlexKV

```bash
apt update && apt install liburing-dev

cd FlexKV && ./build.sh
```

### vLLM Apply Patch

```bash
# 进入 vLLM 目录
cd /opt/vllm 
# apply patch
git apply /your/path/to/FlexKV/examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch
```

### FlexKV 验证

请参考[vLLM online serving](https://github.com/taco-project/FlexKV/blob/dev/docs/vllm_adapter/README_zh.md#%E7%A4%BA%E4%BE%8B)里的测试脚本。


## 2. Dynamo 配置修改

### kv_transfer_config

为了和FlexKV集成，需要修改Dynamo镜像内的kv_transfer_config。将/opt/dynamo/venv/lib/python3.12/site-packages/dynamo/vllm/args.py 的245-248行修改为:

```python
kv_transfer_config = KVTransferConfig(
    kv_connector="FlexKVConnectorV1", kv_role="kv_both"
)
logger.info("Using FlexKVConnectorV1 configuration")
```

### CPU Offloading

在Dynamo中，KV router通过接收worker发送的event来更新KV index，从而感知每个worker上的KV cache情况。当FlexKV开启CPU offloading时，我们删掉vLLM里[BlockRemove](https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/v1/core/block_pool.py#L221)，让FlexKV通过CPU能够缓存住所有serving过程中的KV block，这样KV router维护的index就能反映FlexKV的真实index了。

## 3. 启动和验证Dynamo服务

### 启动Dynamo + FlexKV

```bash
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# 启动nats和etcd
nats-server -js &

etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

sleep 3

# run ingress, 通过--router-mode设置路由方式，可选项为kv, round-robin, random
python -m dynamo.frontend --router-mode kv --http-port 8000 &

# 定义工作节点数量
NUM_WORKERS=4

# 多个worker时注意FlexKV的端口应不同，否则会卡在flexkv init这一步
# 请根据服务器的配置，调整num_cpu_blocks和num_ssd_blocks的数值
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

# 使用for循环启动工作节点
for i in $(seq 0 $((NUM_WORKERS-1))); do
    # 计算GPU设备ID
    GPU_START=$((i*2))
    GPU_END=$((i*2+1))
    
    if [ $i -lt $((NUM_WORKERS-1)) ]; then
        FLEXKV_CONFIG_PATH="./flexkv_config_${i}.json" CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} python3 -m dynamo.vllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tensor_parallel_size 2  --block-size 64 --gpu-memory-utilization 0.9 --max-model-len 100310 &
    else
        FLEXKV_CONFIG_PATH="./flexkv_config_${i}.json" CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} python3 -m dynamo.vllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tensor_parallel_size 2  --block-size 64 --gpu-memory-utilization 0.9 --max-model-len 100310
    fi
done
```

### 验证

可通过如下命令验证Dynamo服务是否正确启动：
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

我们使用[genai-perf](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf)作为benchmark工具、[mooncake trace](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#-open-source-trace)作为数据集来评估Dynamo + FlexKV的性能。

Mooncake Trace 是一个开源请求记录文件，以jsonl格式保存。它记录了请求到达的时间戳、输入文本长度、输出文本长度以及与缓存有关的hash id等信息，包含了1小时内的23608个请求。我们的实验资源是4个LLaMA-70B worker，mooncake trace对于该配置来说并发太高了，于是我们从mooncake trace里每6个抽取1个request，构建了用于benchmark的数据集。

genai-perf可以根据trace文件里的时间戳来发送请求，统计LLM服务的TTFT、TPOT等指标，命令如下。请使用genai-perf==0.0.13，更新的版本存在解析时间戳的bug。

```bash
 genai-perf profile   --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-70B  --endpoint-type chat   --endpoint /v1/chat/completions --streaming  --url http://localhost:8000  --input-file payload:mooncake_trace_1_6.jsonl --random-seed 100  -v  -H 'Authorization: Bearer NOT USED'  -H 'Accept: text/event-stream'   -- --stability-percentage 99
```