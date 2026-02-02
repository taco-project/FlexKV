# FlexKV 与 Dynamo 集成指南

该文档展示了如何将FlexKV和NVIDIA [Dynamo](https://github.com/ai-dynamo/dynamo) 框架集成，并完成性能测试的步骤。

Dynamo是NVIDIA专为大规模分离式部署而设计的框架，支持TensorRT-LLM, vLLM, SGLang等多个后端引擎。其中KV 路由器（KV Router）是一个智能的请求路由组件, 它能够追踪和管理存储在不同worker上的 KV cache，并根据请求与缓存的重叠程度和worker当前负载，智能地将请求分配给最合适的 GPU 节点，从而减少昂贵的 KV 缓存重新计算，提高推理效率。文档也介绍了如何在开启KV Router时，将FlexKV集成进Dynamo。

## 1. 环境准备

### 安装 vLLM

参考 vLLM 适配 [README](../vllm_adapter/README_zh.md)。

### 安装 Dynamo

```bash
# 1. 准备 Dynamo 代码
git clone https://github.com/ai-dynamo/dynamo.git

# 2. 按照 PR #5858 更新代码
gh pr checkout 5858 # 确保已安装 GitHub CLI

# 3. 安装 NIXL
uv pip install 'nixl[cu12]' # 或 'nixl[cu13]'

# 4. 安装 Dynamo
cd $DYNAMO_WORKSPACE/lib/bindings/python
maturin develop --uv
cd ../../..
uv pip install -e . # 因为 vLLM 已安装，所以无需指定后端

# 5. 安装 nats-server 和 etcd
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

- 参考 GPUDirect Storage（GDS）[README](../gds/README_zh.md) 来启用 GDS。
- 参考 KV 缓存复用 [README](../dist_reuse/README_zh.md) 来启用分布式环境下的节点间 KV 缓存共享。

### FlexKV 验证

请参考[vLLM online serving](../../docs/vllm_adapter/README_zh.md#%E7%A4%BA%E4%BE%8B)里的测试脚本。

## 2. 启动和验证Dynamo服务

### 启动Dynamo + FlexKV

下面的示例展示了如何在一台8卡节点上启动4个 Dynamo vLLM worker，并开启KV路由。

```bash
# 启动 NATS 并开启 JetStream
nats-server -js -a 127.0.0.1 -p 4222 --store_dir $NATS_DIR &

# 启动 etcd
etcd --data-dir /tmp/etcd \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://YOUR_IP:2379 & # YOUR_IP 是该节点的IP地址。

sleep 3

export NATS_SERVER="nats://127.0.0.1:4222"
export ETCD_ENDPOINTS="http://127.0.0.1:2379"

# 启动 Dynamo 前端（启用 KV 路由模式）
python -m dynamo.frontend --router-mode kv &

# 定义 worker 节点数量
NUM_WORKERS=4

# 启用 FlexKV KV 事件收集
export DYNAMO_USE_FLEXKV=1
# 使用环境变量配置FlexKV，禁用配置文件
unset FLEXKV_CONFIG_PATH
# 根据服务器的配置，调整CPU和SSD的空间大小
export FLEXKV_CPU_CACHE_GB=32
export FLEXKV_SSD_CACHE_GB=128
# 使用for循环启动工作节点
for i in $(seq 0 $((NUM_WORKERS-1))); do
    # 计算GPU设备ID
    GPU_START=$((i*2))
    GPU_END=$((i*2+1))

    if [ $i -lt $((NUM_WORKERS-1)) ]; then
        # 多个worker时注意Flexkv的端口应不同，否则会卡在flexkv init这一步
        # 通过环境变量 `FLEXKV_SERVER_RECV_PORT` 设置Flexkv的端口
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
        CUDA_VISIBLE_DEVICES=${GPU_START},${GPU_END} 
        python3 -m dynamo.vllm \
        --model $YOUR_MODEL \
        --tensor-parallel-size 2 \
        --connector flexkv \
        --kv-events-config "$KV_EVENTS_CONFIG"
    fi
done
```

> [!NOTE]
> 可使用 YAML 或 JSON 文件配置，上述配置仅为简单示例，更多选项请参考[`docs/flexkv_config_reference/README_zh.md`](../../docs/flexkv_config_reference/README_zh.md)

### 验证

可通过如下命令验证Dynamo服务是否正确启动：

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": 你的模型,
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

我们使用 [`aiperf`](https://github.com/ai-dynamo/aiperf) 作为benchmark工具、[mooncake trace](https://github.com/kvcache-ai/Mooncake?tab=readme-ov-file#-open-source-trace)作为数据集来评估Dynamo + FlexKV的性能。

Mooncake Trace 是一个开源请求记录文件，以jsonl格式保存。它记录了请求到达的时间戳、输入文本长度、输出文本长度以及与缓存有关的hash id等信息，包含了1小时内的23608个请求。我们的实验资源是4个LLaMA-70B worker，mooncake trace对于该配置来说并发太高了，于是我们从mooncake trace里每6个抽取1个request，构建了用于benchmark的数据集。

`aiperf` 可以根据trace文件里的时间戳来发送请求，统计LLM服务的TTFT、TPOT等指标，命令如下。

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
