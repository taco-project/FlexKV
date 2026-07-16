# FlexKV stress benchmark

This benchmark allocates synthetic inference-engine KV tensors and exercises
FlexKV's match, load and store paths over configurable multi-turn conversations.

```bash
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/glm5.yaml --dry-run
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/glm5.yaml --rounds 1000
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/dsv4_pro.yaml --duration 86400
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/cpu_stub.yaml --cpu-stub
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/dsv4_cpu_smoke.yaml
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/glm5_cpu_smoke.yaml
benchmarks/flexkv_stress/run.sh benchmarks/flexkv_stress/configs/glm5_cpu_smoke.yaml --mode bandwidth
```

`run.mode` selects `latency_hit` or `bandwidth`. The latency/hit mode always
runs an unloaded profile (`batch=1`, `concurrency=1`) and a loaded profile
using the configured batch and concurrency. Bandwidth mode independently
reports GPU→CPU save, CPU→GPU load, GPU→SSD save end-to-end, and SSD→GPU reload
end-to-end for each configured concurrency level. SSD paths are omitted when
the SSD tier is disabled.

```yaml
run:
  mode: bandwidth
bandwidth:
  paths: [gpu_to_cpu_save, cpu_to_gpu_load, gpu_to_ssd_save_e2e, ssd_to_gpu_reload_e2e]
  concurrency_levels: [1, 2, 4, 8, 16]
  target_payload_gb: 0
  min_duration_seconds: 30
  min_operations: 100
  window_seconds: 5
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

`model.tp_size` follows SGLang's composite TP world size **per DP shard**.
Q-split CP ranks overlap that world instead of multiplying it. The benchmark
derives `kv_tp_size = tp_size / cp_size`, starts `tp_size` physical GPU workers
per DP shard, and passes `(kv_tp_size, cp_size)` to FlexKV. Therefore TP=4,
CP=2 uses four processes/GPUs arranged as two KV-TP slices duplicated over two
CP ranks, not eight GPUs. `tp_size` must be divisible by `cp_size`.

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

The accelerator path uses PyTorch's `cuda:N` API and requires the FlexKV
transfer extension to be built for CUDA.

Every run produces exactly `summary.csv`, `metrics.csv`, `summary.json`, and
`effective_config.yaml`. `summary.csv` and `metrics.csv` use a mode-specific
schema, while `summary.json` is the versioned (`1.0`) machine interface. An
`errors.csv` file is added only when an error occurs; intermediate operation or
turn CSV files are not emitted.

Capacity is reported in decimal GB, throughput in decimal GB/s, latency in ms,
and all accuracy/rate values in `[0,1]`. Bandwidth is logical KV payload divided
by timed transfer duration. Match, workload creation, and byte readback are not
inside the transfer timing boundary. It includes compressed main-KV groups and
SWA pages, but excludes protocol and driver overhead. See
[`DESIGN.md`](DESIGN.md) for the byte formula, concurrency caveats, execution
flow, validation rules, and DSv4 verification levels.

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
does not validate CUDA IPC, native transfer kernels, SSD, or eventfd signaling,
and the reported latency is not a FlexKV performance measurement. CPU-stub
runs still produce every report, but set `performance_valid=false`.
The real accelerator path launches one OS process per GPU and pins it to one
`cuda:N` device. CPU stub uses lightweight in-process virtual workers, so it
checks topology/routing and duplicated CP KV contents but not process or CUDA
IPC isolation.
