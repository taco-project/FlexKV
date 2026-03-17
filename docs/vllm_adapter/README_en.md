# Using FlexKV in vLLM

## 🎉 Important Update (March 17, 2026)

**FlexKV has been officially merged into the vLLM mainline on March 17, 2026 (PR [#34328](https://github.com/vllm-project/vllm/pull/34328)).**

Starting from **vLLM v0.17.2rc0** (commit [`8cb24d3`](https://github.com/vllm-project/vllm/commit/8cb24d3aedb9f431fb15a636a3e11a00262f5991)), `FlexKVConnectorV1` is built into vLLM — **no patch is needed anymore**.

> **Recommended**: Install vLLM >= v0.17.2 (stable release) or build from the latest `main` branch directly. The patch files under `examples/vllm_adaption/` are retained for users who need to use vLLM < 0.17.2.

---

## Current Version vs. Legacy Version
In commit [`0290841dce65ae9b036a23d733cf94e47e814934`](https://github.com/taco-project/FlexKV/commit/0290841dce65ae9b036a23d733cf94e47e814934), we introduced a major update:  
**FlexKV has transitioned from a client-server architecture to a library function that inference acceleration engines (such as vLLM) can directly invoke**, reducing inter-process communication overhead.

This change involves significant API adjustments. Therefore, please note:

- **Version >= `1.0.0`**: Use the **current version API**.
- **Version == `0.1.0`**: Supports the **legacy version API** only; the vLLM patch is located in `examples/vllm_adaption_legacy/`.

---

## Current Version (>= 1.0.0)

### Supported Versions
- FlexKV >= `1.0.0`
- **vLLM >= `0.17.2` (recommended)**: FlexKVConnectorV1 is built in — no patch required
- vLLM `0.10.1` ~ `0.17.1`: Manual patch required, see [Using Older vLLM (Patch Required)](#using-older-vllm-patch-required)

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

### Running (Recommended: vLLM >= 0.17.2, No Patch Needed)

1. Install vLLM (>= 0.17.2 stable, or build from official `main` branch)
```bash
pip install vllm>=0.17.2
# Or install from source (latest main):
# git clone https://github.com/vllm-project/vllm.git && cd vllm && pip install -e .
```

2. Install FlexKV
```bash
pip install flexkv  # or build from source: ./build.sh
```

3. offline test
```bash
# VLLM_DIR/examples/offline_inference/prefix_caching_flexkv.py
python examples/offline_inference/prefix_caching_flexkv.py
```

4. online serving
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

### Using Older vLLM (Patch Required)

If you need to use FlexKV with vLLM < 0.17.2, apply the corresponding patch manually:

| vLLM Version | Patch File |
|---|---|
| 0.10.1.1 | `examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch` |
| 0.14.1+ | `examples/vllm_adaption/vllm_up_0_14_1-flexkv-connector.patch` |
| 0.16.0 | `examples/vllm_adaption/vllm_0_16_0-flexkv-connector.patch` |

```bash
cd vllm
git apply FLEXKV_DIR/examples/vllm_adaption/<patch-file>
pip install -e .
```

---

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
