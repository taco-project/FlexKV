# 在 vLLM 中使用 FlexKV

## 当前版本与 Legacy 版本说明
在 commit [`0290841dce65ae9b036a23d733cf94e47e814934`](https://github.com/taco-project/FlexKV/commit/0290841dce65ae9b036a23d733cf94e47e814934)，我们更新了一个重要功能：
 **FlexKV 从 client-server 模式，变为推理加速引擎（如 vLLM）可直接调用的库函数**，以减少进程间消息传递的开销。
这一变更引发了较大的 API 调整。因此，请注意：

- **版本 >= `1.0.0`**：应使用 **当前版本 API**，vLLM patch位于 `examples/vllm_adaption/`。
- **版本 == `0.1.0`**：仅支持 **Legacy 版本 API**, vLLM patch位于`examples/vllm_adaption_legacy/`。

---

## 当前版本（>= 1.0.0）

### 适用版本
- FlexKV >= `1.0.0`
- vLLM 原则上>= `0.8.5`版本均可参考示例代码进行修改

### 配置

#### 示例一：仅启用CPU卸载
使用32GB的CPU内存作为二级缓存。
```bash
unset FLEXKV_CONFIG_PATH
export FLEXKV_CPU_CACHE_GB=32
```
#### 示例二：启用SSD卸载
使用32GB的CPU内存和1T的SSD存储。两个SSD分别挂载到/data0和/data1两个路径上。
```bash
# generate config
cat <<EOF > ./flexkv_config.yml
cpu_cache_gb: 32
ssd_cache_gb: 1024
ssd_cache_dir: /data0/flexkv_ssd/;/data1/flexkv_ssd/
enable_gds: false
EOF
export FLEXKV_CONFIG_PATH="./flexkv_config.yml"
```

> 注：`flexkv_config.yml`配置仅为简单示例，选项请参考[`docs/flexkv_config_reference/README_zh.md`](../../docs/flexkv_config_reference/README_zh.md)

### 运行
我们提供了基于 **vLLM 0.10.1.1** 的适配示例：

1. apply patch
```bash
# FLEXKV_DIR/examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch
git apply examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch
```

2. offline test
```bash
# VLLM_DIR/examples/offline_inference/prefix_caching_flexkv.py
python examples/offline_inference/prefix_caching_flexkv.py
```

3. online serving
```bash
VLLM_USE_V1=1 python -m vllm.entrypoints.cli.main serve Qwen3/Qwen3-32B \
     --tensor-parallel-size 8 \
     --trust-remote-code \
     --port 30001 \
     --max-num-seqs 128 \
     --max-num-batched-tokens 8192 \
     --max_model_len 8192 \
     --max-seq-len-to-capture 8192 \
     --gpu-memory-utilization 0.8 \
     --enable-chunked-prefill \
     --enable-prefix-caching \
     --kv-transfer-config \
        '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'

```

## Legacy版本（<= 0.1.0）,目前的版本尽量不要使用

### 适用版本
- FlexKV <= `0.1.0`

### 配置

旧版本配置方式如下
```bash
# generate config
cat <<EOF > ./flexkv_config.json
{
    "server_recv_port": "ipc:///tmp/flexkv_test",
    "cache_config": {
          "enable_cpu": true,
          "num_cpu_blocks": 10240
    },
    "num_log_interval_requests": 200
}
EOF
export FLEXKV_CONFIG_PATH="./flexkv_config.json"
```

### 运行
在 vLLM 0.8.4 版本中应用patch `examples/vllm_adaption_legacy/flexkv_vllm_0_8_4.patch`，分别启动 FlexKV、vLLM 和测试脚本：

```bash
# 启动 FlexKV 作为服务端
bash benchmarks/flexkv_benchmark/run_flexkv_server.sh

# 启动 vLLM 作为客户端
bash benchmarks/flexkv_benchmark/serving_vllm.sh

# 启动性能测试
bash benchmarks/flexkv_benchmark/multiturn_benchmark.sh
```
在 vLLM 0.10.0 版本中应用patch `examples/vllm_adaption_legacy/flexkv_vllm_0_10_0.patch`，测试方法同上。
