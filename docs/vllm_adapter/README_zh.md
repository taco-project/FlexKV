# 在 vLLM 中使用 FlexKV

## 🎉 重要更新（2026年3月17日）

**FlexKV 已于 2026 年 3 月 17 日正式合并进入 vLLM 官方主干（PR [#34328](https://github.com/vllm-project/vllm/pull/34328)）。**

从 **vLLM v0.17.2rc0**（commit [`8cb24d3`](https://github.com/vllm-project/vllm/commit/8cb24d3aedb9f431fb15a636a3e11a00262f5991)）起，`FlexKVConnectorV1` 已内置于 vLLM，**无需再手动打 patch**。

> **推荐做法**：安装 vLLM v0.17.2 ~ v0.28.x，或 v0.33.0 并打上[对应 patch](#给-vllm-打-patch)。从 v0.29.0 起，内置 connector 已无法启动 FlexKV。  
> 其余 patch 仍保留于 `examples/vllm_adaption/` 目录，供需要使用 vLLM < 0.17.2 的用户参考。

---

## 当前版本与 Legacy 版本说明
在 commit [`0290841dce65ae9b036a23d733cf94e47e814934`](https://github.com/taco-project/FlexKV/commit/0290841dce65ae9b036a23d733cf94e47e814934)，我们更新了一个重要功能：
 **FlexKV 从 client-server 模式，变为推理加速引擎（如 vLLM）可直接调用的库函数**，以减少进程间消息传递的开销。
这一变更引发了较大的 API 调整。因此，请注意：

- **版本 >= `1.0.0`**：应使用 **当前版本 API**。
- **版本 == `0.1.0`**：仅支持 **Legacy 版本 API**, vLLM patch位于`examples/vllm_adaption_legacy/`。

---

## 当前版本（>= 1.0.0）

### 适用版本
- FlexKV >= `1.0.0`
- vLLM `0.17.2` ~ `0.28.x`：内置 FlexKVConnectorV1，无需打 patch，直接使用
- vLLM `0.33.0`：需给内置 FlexKVConnectorV1 打 patch，参见[给 vLLM 打 patch](#给-vllm-打-patch)
- vLLM `0.10.1` ~ `0.17.1`：需手动 apply patch，参见[给 vLLM 打 patch](#给-vllm-打-patch)

### 配置

#### 示例一：仅启用CPU卸载
使用32GB的CPU内存作为二级缓存。
```bash
unset FLEXKV_CONFIG_PATH
export FLEXKV_CPU_CACHE_GB=32
```
#### 示例二：启用SSD卸载
使用32GB的CPU内存和1T的SSD存储分别作为二级和三级缓存。（假设机器有两个SSD，并分别挂载在/data0和/data1两个路径上。）
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
#### 示例三：以 radixshmem 作为 CPU 层
CPU 层及其前缀索引由 [radixshmem](../radixshmem/config_zh.md) 的 `radix-server` 承担，每节点一个；CPU 缓存大小由它的 `--data-bytes` 决定，而不是 `cpu_cache_gb`。
```bash
radix-server --name /flexkv --data-bytes 64G
export FLEXKV_ENABLE_RADIXSHMEM=1
```

> 注：`flexkv_config.yml`配置仅为简单示例，选项请参考[`docs/flexkv_config_reference/README_zh.md`](../../docs/flexkv_config_reference/README_zh.md)

### 运行

1. 安装 vLLM，所用版本需要时再打 patch（参见[给 vLLM 打 patch](#给-vllm-打-patch)）
```bash
pip install 'vllm==0.33.0'
```

2. 安装 FlexKV
```bash
pip install flexkv  # 或从源码编译：./build.sh
```

3. offline test
```bash
# 将 VLLM_DIR 设置为包含此示例的兼容 vLLM 源码目录。
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

### 实验性 Layer-wise KV 加载

Layer-wise 加载可让 FlexKV 搬运第 `N+1` 层 KV 的同时，vLLM 计算第 `N`
层，从而在外部 KV 传输位于关键路径时降低 TTFT。

```bash
VLLM_USE_V1=1 python -m vllm.entrypoints.cli.main serve MODEL \
  --kv-transfer-config \
  '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both",
"kv_connector_extra_config":{"use_layerwise":true}}'
```

必须设置 `use_layerwise=true`，使 vLLM 选择 PIECEWISE CUDA Graph 模式；
逐层 eventfd 等待无法在完整 CUDA Graph 捕获/回放中正确执行。为保证安全，
仅设置 `FLEXKV_ENABLE_LAYERWISE_TRANSFER=1` 会被 vLLM adapter 忽略。vLLM 0.33.0
的 patch 已包含对应的 wrapper 改动；其他版本在该改动合入上游前，请给 vLLM 应用
`examples/vllm_adaption/vllm_main-flexkv-layerwise-wrapper.patch`。

当前仅对 GET/H2D 做逐层流水；PUT 仍是请求级异步。
当前暂不支持 Context Parallel（`context_parallel_size > 1`），因为 vLLM
运行时 DCP/PCP rank 尚未映射到 FlexKV 的 eventfd 索引空间。该模式也暂不
支持 GDS，因为融合后的 layer-wise graph 使用 CPU/SSD/remote staging，而
不是直接 `DISK2D`。如果某一层在
`FLEXKV_LAYERWISE_WAIT_TIMEOUT_S`（默认 `60` 秒）内未就绪，forward 会失败，
而不是永久卡住。

### 给 vLLM 打 patch

请 apply 与 vLLM 版本对应的 patch：

| vLLM 版本 | patch 文件 |
|---|---|
| 0.10.1.1 | `examples/vllm_adaption/vllm_0_10_1_1-flexkv-connector.patch` |
| 0.14.1+ | `examples/vllm_adaption/vllm_up_0_14_1-flexkv-connector.patch` |
| 0.16.0 | `examples/vllm_adaption/vllm_0_16_0-flexkv-connector.patch` |
| 0.33.0 | `examples/vllm_adaption/vllm_0_33_0-flexkv-connector.patch` |

```bash
cd vllm
# 设置 FLEXKV_DIR，并选择与 vLLM 版本匹配的 patch。
PATCH_FILE=vllm_0_16_0-flexkv-connector.patch
git apply "$FLEXKV_DIR/examples/vllm_adaption/$PATCH_FILE"
pip install -e .
```

0.33.0 的 patch 只修改内置 connector：把 vLLM 的 `KVCacheConfig` 传给 FlexKV，
并转发缓存清空、sleep/wake 和逐层加载。它也可以直接打到已安装的 wheel 上：
```bash
cd "$(python -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
git apply --include='vllm/*' "$FLEXKV_DIR/examples/vllm_adaption/vllm_0_33_0-flexkv-connector.patch"
```

---

## Legacy版本（<= 0.1.0），目前的版本尽量不要使用

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

`examples/vllm_adaption_legacy/` 中的历史 patch 分别针对 vLLM 0.8.4 和 0.10.0。
历史 `benchmarks/flexkv_benchmark/` 启动脚本不在当前仓库中。若需复现旧版部署，
请使用包含对应脚本的兼容历史 FlexKV 版本；当前分支及其配置示例针对当前 API。
