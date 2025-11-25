# Using FlexKV in vLLM

## Current Version vs. Legacy Version
In commit [`0290841dce65ae9b036a23d733cf94e47e814934`](https://github.com/taco-project/FlexKV/commit/0290841dce65ae9b036a23d733cf94e47e814934), we introduced a major update:  
**FlexKV has transitioned from a client-server architecture to a library function that inference acceleration engines (such as vLLM) can directly invoke**, reducing inter-process communication overhead.

This change involves significant API adjustments. Therefore, please note:

- **Version >= `1.0.0`**: Use the **current version API**; the vLLM patch is located in `examples/vllm_adaption/`.
- **Version == `0.1.0`**: Supports the **legacy version API**; the vLLM patch is located in `examples/vllm_adaption_legacy/`.

---

## Current Version (>= 1.0.0)

### Supported Versions
- FlexKV >= `1.0.0`
- vLLM versions >= `0.8.5` can generally follow the example code for adaptation

### Configuration

#### Example 1: CPU Offloading Only
Use 32GB of CPU memory as secondary cache.
```bash
unset FLEXKV_CONFIG_PATH
export FLEXKV_CPU_CACHE_GB=32
```
#### Example 2: SSD Offloading
Use 32GB of CPU memory and 1TB of SSD storage as secondary and tertiary cache respectively. (Assume the machine has two SSDs mounted at /data0 and /data1 respectively.)
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

> Note: The `flexkv_config.yml` configuration is provided as a simple example only. For full parameter options, please refer to [`docs/flexkv_config_reference/README_en.md`](../../docs/flexkv_config_reference/README_en.md)

### Running
We provide an adaptation example based on **vLLM 0.10.1.1**:

1. apply patch && installation
```bash
cd vllm
git apply FLEXKV_DIR/examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch
pip install -e . # build and install vllm from source
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

## Legacy Version (<= 0.1.0) – Not Recommended for Current Use

### Supported Versions
- FlexKV <= `0.1.0`

### Configuration

Legacy version configuration:
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

### Running
Apply the patch `examples/vllm_adaption_legacy/flexkv_vllm_0_8_4.patch` to vLLM 0.8.4, then start FlexKV, vLLM, and the benchmark script:

```bash
# Start FlexKV as server
bash benchmarks/flexkv_benchmark/run_flexkv_server.sh

# Start vLLM as client
bash benchmarks/flexkv_benchmark/serving_vllm.sh

# Start benchmark
bash benchmarks/flexkv_benchmark/multiturn_benchmark.sh
```
Apply the patch `examples/vllm_adaption_legacy/flexkv_vllm_0_10_0.patch` to vLLM 0.10.0, and use the same testing method as above.
