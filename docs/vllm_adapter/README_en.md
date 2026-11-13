# Using FlexKV in vLLM

## 🎉 Important Update (March 17, 2026)

**FlexKV has been officially merged into the vLLM mainline on March 17, 2026 (PR [#34328](https://github.com/vllm-project/vllm/pull/34328)).**

Starting from **vLLM v0.17.2rc0** (commit [`8cb24d3`](https://github.com/vllm-project/vllm/commit/8cb24d3aedb9f431fb15a636a3e11a00262f5991)), `FlexKVConnectorV1` is built into vLLM — **no patch is needed anymore**.

> **Recommended**: Install vLLM v0.17.2 ~ v0.28.x, or v0.33.0 with [its patch](#patching-vllm). Since v0.29.0 the built-in connector can no longer start FlexKV. The other patch files under `examples/vllm_adaption/` are retained for users who need to use vLLM < 0.17.2.

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
- vLLM `0.17.2` ~ `0.28.x`: FlexKVConnectorV1 is built in — no patch required
- vLLM `0.33.0`: the built-in FlexKVConnectorV1 must be patched, see [Patching vLLM](#patching-vllm)
- vLLM `0.10.1` ~ `0.17.1`: Manual patch required, see [Patching vLLM](#patching-vllm)

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
#### Example 3: radixshmem as the CPU Tier
The CPU tier and its prefix index live in a [radixshmem](../radixshmem/config_en.md) `radix-server`, one per node; its `--data-bytes`, not `cpu_cache_gb`, sets the CPU cache size.
```bash
radix-server --name /flexkv --data-bytes 64G
export FLEXKV_ENABLE_RADIXSHMEM=1
```

> Note: The `flexkv_config.yml` configuration is provided as a simple example only. For full parameter options, please refer to [`docs/flexkv_config_reference/README_en.md`](../../docs/flexkv_config_reference/README_en.md)

### Running

1. Install vLLM, then patch it if its version needs it (see [Patching vLLM](#patching-vllm))
```bash
pip install 'vllm==0.33.0'
```

2. Install FlexKV
```bash
pip install flexkv  # or build from source: ./build.sh
```

3. offline test
```bash
# Set VLLM_DIR to a compatible vLLM checkout containing this example.
python "$VLLM_DIR/examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py" --model /path/to/model
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

### Experimental Layer-wise KV Loading

Layer-wise loading overlaps FlexKV H2D transfer for layer `N+1` with vLLM
compute for layer `N`, reducing TTFT when external KV transfer is on the
critical path.

```bash
VLLM_USE_V1=1 python -m vllm.entrypoints.cli.main serve MODEL \
  --kv-transfer-config \
  '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both",
"kv_connector_extra_config":{"use_layerwise":true}}'
```

`use_layerwise=true` is required so vLLM selects PIECEWISE CUDA graph mode;
the per-layer eventfd waits cannot execute correctly inside a captured full
CUDA graph. For safety, setting `FLEXKV_ENABLE_LAYERWISE_TRANSFER=1` alone is
ignored by the vLLM adapter. The vLLM 0.33.0 patch already contains the
matching wrapper change; for other versions, until it is merged upstream, apply
`examples/vllm_adaption/vllm_main-flexkv-layerwise-wrapper.patch` to vLLM.

The feature currently pipelines GET/H2D only. PUT remains request-level async.
Context parallelism (`context_parallel_size > 1`) is not supported yet because
vLLM's runtime DCP/PCP rank has not been mapped into FlexKV's eventfd index.
GDS is also unsupported in this mode because the fused layer-wise graph uses
CPU/SSD/remote staging rather than direct `DISK2D` operations.
If a layer does not become ready within
`FLEXKV_LAYERWISE_WAIT_TIMEOUT_S` (default `60` seconds), the forward fails
instead of hanging indefinitely.

### Patching vLLM

Apply the patch matching your vLLM version:

| vLLM Version | Patch File |
|---|---|
| 0.10.1.1 | `examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch` |
| 0.14.1+ | `examples/vllm_adaption/vllm_up_0_14_1-flexkv-connector.patch` |
| 0.16.0 | `examples/vllm_adaption/vllm_0_16_0-flexkv-connector.patch` |
| 0.33.0 | `examples/vllm_adaption/vllm_0_33_0-flexkv-connector.patch` |

```bash
cd vllm
# Set FLEXKV_DIR and choose the patch matching your vLLM version.
PATCH_FILE=vllm_0_16_0-flexkv-connector.patch
git apply "$FLEXKV_DIR/examples/vllm_adaption/$PATCH_FILE"
pip install -e .
```

The 0.33.0 patch only touches the built-in connector, which then passes vLLM's
`KVCacheConfig` to FlexKV and forwards cache reset, sleep/wake and layer-wise
loading. It also applies to an installed wheel:
```bash
cd "$(python -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
git apply --include='vllm/*' "$FLEXKV_DIR/examples/vllm_adaption/vllm_0_33_0-flexkv-connector.patch"
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

The legacy patches in `examples/vllm_adaption_legacy/` target vLLM 0.8.4 and
0.10.0 respectively. The historical `benchmarks/flexkv_benchmark/` launch
scripts are not present in this checkout. To reproduce a legacy deployment,
use a compatible historical FlexKV revision containing its scripts; the current
branch and its configuration examples target the current API.
