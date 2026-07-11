# FlexKV stress benchmark

This benchmark allocates synthetic inference-engine KV tensors and exercises
FlexKV's match, load and store paths over configurable multi-turn conversations.

```bash
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/glm5.yaml --dry-run
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/glm5.yaml --rounds 1000
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/dsv4_pro.yaml --duration 86400
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/cpu_stub.yaml --cpu-stub
```

The shipped presets are `dsv4_pro`, `dsv4_flash`, `glm5`, and `glm5_2`.
They create the physical GPU layouts used by SGLang: DSv4 c4/c128/indexer/SWA
pools and GLM MLA/indexer pools. A local Hugging Face config can refresh the
geometry without loading model weights:

```yaml
model:
  preset: glm5_2
  hf_config_path: /models/GLM-5.2/config.json
  tp_size: 2
  dp_size: 2
  cp_size: 1
```

`layer_groups` may be overridden directly. `layers` accepts an explicit list,
`all`, `{compress_ratio: 4}`, or `{indexer_type: shared}`:

```yaml
model:
  preset: glm5
  layer_groups:
    - {name: main, layers: all, num_kv_heads: 1, head_size: 576, dtype: bfloat16}
    - {name: indexer, layers: all, num_kv_heads: 1, head_size: 132, dtype: uint8}
```

Each turn first loads the reusable history, writes deterministic synthetic KV,
stores only `put_match` misses, and performs an immediate read-back. Input and
output lengths accept a fixed block count, a two-element range, or an explicit
`{mode: list, blocks: [...]}` sequence. Cache state is retained across rounds.

CUDA and ROCm both use PyTorch's `cuda:N` API. On ROCm, tensor sharing uses
PyTorch IPC and never attempts to load or call `libcudart`. The FlexKV transfer
extension itself must still have been built for HIP.

The primary outputs are `rounds.csv`, `turns.csv`, `operations.csv`,
`windows.csv`, `validation_samples.csv`, `resources.csv`, `errors.csv`,
`summary.csv`, and `effective_config.yaml`.

For a single-DP SSD run, `cache.force_ssd_reload_interval_rounds` periodically
clears only the CPU tier through FlexKV's test hook and verifies an SSD reload.
Set it to `0` to rely on natural eviction only.

`get_async`/`put_async` probes run at
`features.async_api_probe_interval_rounds` when layerwise and SWA are disabled.
Those APIs use FlexKV's ordinary H2D worker, which is intentionally absent when
the fused layerwise H2D worker is active.

For logic checks on a host without an accelerator, `--cpu-stub` replaces the
FlexKV manager and inference-engine workers with an in-memory implementation.
It still allocates real CPU PyTorch KV tensors and validates prefix matching,
batch launch, PUT/GET, async APIs, TP/DP routing, and byte-level readback. It
does not validate CUDA/ROCm IPC, native transfer kernels, SSD, or eventfd
signaling, and the reported latency is not a FlexKV performance measurement.
