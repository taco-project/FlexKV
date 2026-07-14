# `dpskv4_refactor` 合入 `main` 验证手册

本文档供具备 Linux、CUDA/ROCm、GPU、Redis、Mooncake、SGLang、vLLM 和
TensorRT-LLM 环境权限的测试 agent 使用。所有测试结果必须回填到
[`dpskv4_main_merge_validation_results.md`](dpskv4_main_merge_validation_results.md)，
并附上完整日志或日志链接。不要只在聊天中汇报结果。

## 1. 测试对象

- 集成分支：`integration/dpskv4-main-sync`
- 功能基线：`origin/dpskv4_refactor@f5ee0ebabd98c1eecc58151650b36e46b477a4d2`
- 主干基线：`origin/main@3c968fc09c3d3b4f3f46bd0fcc9c4d6d9bb94ca8`
- 目标：保留主干能力，同时加入 DSv4、LayerGroup、DP/TP/PP/CP、layerwise 和 SWA。

### 当前状态

该分支是 **Draft 集成候选**，不是已验证完成的发布版本：

- 已完成：Git 三方合并、冲突标记清理、`git diff --check`、Python `compileall`。
- 未完成：Linux C++/CUDA 构建、pytest 门禁、GPU 和框架验证。
- 优先风险：Layerwise C++/pybind/Python 参数一致性，以及 SGLang patch 中旧
  Indexer/`attn_*` 接口到 LayerGroup/`cp_*` 的迁移。

## 2. 分阶段执行与停止门槛

不要让一个 agent 一次执行全文。每个阶段单独分配；前一阶段通过后才进入下一阶段。

| 阶段 | 目标 | 预计范围 | 停止门槛 |
|---|---|---|---|
| Phase 0 | 合并与接口审计 | 无 GPU，静态检查和旧接口搜索 | 任一冲突、语法或明确旧接口问题即停止 |
| Phase 1 | Linux 基础构建与 CPU 测试 | 默认 C++/CUDA 构建、import、unit/smoke | 构建、收集或核心单测失败即停止 |
| Phase 2 | 单机 GPU 核心链路 | 单卡 Full-KV；多卡 Layerwise/MLA/LayerGroup/SWA | 数据不一致、死锁、泄漏即停止 |
| Phase 3 | 框架和多节点 | vLLM、SGLang DSv4、TRT-LLM、Redis/Mooncake | 核心集成失败即停止 |
| Phase 4 | 性能与稳定性 | benchmark、1 小时 soak、AMD 矩阵 | 只在 Phase 0-3 全部通过后执行 |

### Phase 0：只做快速审计

```bash
git diff --check origin/main...HEAD
python -m compileall -q flexkv tests benchmarks
rg -n '^(<<<<<<< |>>>>>>> |=======$)' --glob '!docs/pr_analysis/**' .
rg -n 'IndexerCacheConfig|cache_config\.indexer|model_config\.attn_|rank_info\.attn_' \
  flexkv --glob '!integration/sglang/sglang_flexkv_connector.patch'
rg -n 'cache_config\.indexer|model_config\.attn_|rank_info\.attn_' \
  flexkv/integration/sglang/sglang_flexkv_connector.patch
```

同时人工核对以下调用的声明、pybind 参数和 Python 调用顺序：

- `LayerwiseTransferGroup::layerwise_transfer`
- `LayerwiseTransferGroup::layerwise_transfer_multi_group`
- `mla_d2h_mode`
- `notify_mode`
- SWA tensor/stride 参数

发现明确旧接口或签名不一致时，记录一个 blocker 并停止；不要继续构建或跑 GPU。

### Phase 1：基础构建与轻量测试

仅在 Phase 0 通过后执行：

```bash
FLEXKV_ENABLE_METRICS=0 bash build.sh --debug
python -c 'import flexkv.c_ext; print("c_ext import OK")'
pytest --collect-only -q
pytest -q \
  tests/test_recompute_block_counts.py \
  tests/test_merge_to_batch_graph_swa_callbacks.py \
  tests/test_set_gpu_blocks_swa.py \
  tests/test_swa_host_pool.py \
  tests/test_swa_peer_op.py \
  tests/test_swa_rpc_launch.py
```

Phase 1 不运行 GDS、NIXL、nvCOMP、多节点、框架 E2E 或性能测试。

开始测试前记录实际提交：

```bash
git fetch origin
git switch integration/dpskv4-main-sync
git pull --ff-only
git rev-parse HEAD origin/main origin/dpskv4_refactor
git submodule update --init --recursive
```

测试期间不要直接修改集成分支。发现问题时记录命令、完整堆栈、GPU/驱动/框架版本及
最小复现；修复应另开分支和 PR。

## 3. 环境信息

每台机器先保存以下信息：

```bash
uname -a
python --version
python -m pip freeze
nvidia-smi || true
nvcc --version || true
rocminfo | head -100 || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("hip", torch.version.hip)
print("gpu_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

最低测试矩阵：

| 环境 | 必测范围 |
|---|---|
| Linux + NVIDIA 单卡 | 构建、Python 测试、Full-KV、NVFP4、基础 layerwise/SWA |
| Linux + NVIDIA 2 卡及以上 | TP/PP/CP、MLA D2H 三模式、layerwise 两种通知模式、多组/SWA |
| Linux + NVIDIA 多节点 | Redis/P2P、Mooncake、DP/PP/CP 拓扑、SGLang DSv4 |
| Linux + AMD/ROCm | 使用项目内部 AMD 构建链执行与 NVIDIA 相同的核心回归；当前公开 `setup.py` 仍是 CUDA 构建入口，若无内部构建链必须标记 `BLOCKED`，不可记为通过 |

## 4. 完整静态检查与构建（Phase 1 后续）

在干净环境中安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install pytest pre-commit mypy
```

运行静态门禁：

```bash
git diff --check origin/main...HEAD
! rg -n '^(<<<<<<<|=======|>>>>>>>)' --glob '!docs/pr_analysis/**' .
python -m compileall -q flexkv tests benchmarks
ruff check flexkv tests benchmarks
mypy flexkv
pre-commit run --all-files
```

`pre-commit` 可能自动格式化文件；若发生修改，只提交格式化 diff 到独立修复分支。

基础 NVIDIA 构建：

```bash
FLEXKV_ENABLE_METRICS=0 bash build.sh --debug
python -c 'import flexkv.c_ext; print("c_ext import OK")'
```

下列构建配置必须分别从干净构建目录验证，不得只验证默认配置：

```bash
bash build.sh --clean
FLEXKV_ENABLE_METRICS=0 FLEXKV_ENABLE_P2P=1 bash build.sh --debug

bash build.sh --clean
FLEXKV_ENABLE_METRICS=0 FLEXKV_ENABLE_GDS=1 bash build.sh --debug

bash build.sh --clean
FLEXKV_ENABLE_METRICS=0 FLEXKV_ENABLE_NVCOMP=1 bash build.sh --debug
```

GDS、NIXL、nvCOMP 构建前需按仓库文档安装对应系统库。NIXL 还需验证：

- `enable_nixl=True` 且 `enable_gds=False` 必须明确报错。
- effective TP 大于 1 或启用 heterogeneous `layer_groups` 时必须明确报“不支持”，不能静默走错误路径。
- TP=1、无 `layer_groups` 的 `GDS_MT` PUT/GET 必须字节一致。

## 5. Python 与控制面回归（Phase 1/2）

先执行测试分层：

```bash
pytest -m unit -ra
pytest -m smoke -ra
pytest -ra
```

重点测试文件：

```bash
pytest -q \
  tests/test_batch_kvtask.py \
  tests/test_kvtask_lifecycle.py \
  tests/test_recompute_block_counts.py \
  tests/test_merge_to_batch_graph_swa_callbacks.py \
  tests/test_set_gpu_blocks_swa.py \
  tests/test_swa_host_pool.py \
  tests/test_swa_peer_op.py \
  tests/test_swa_rpc_launch.py \
  tests/test_transfer_engine_atomic_eviction.py
```

必须人工确认以下行为：

- 未配置 `layer_groups` 和 SWA 时，Full-KV 默认路径与主干一致。
- SWA 默认关闭；开启后 slot 分配、挂载、释放、空 slot 和 placeholder `[0]` 均正确。
- Batch GET/PUT 的子任务只释放一次，取消、失败和早返回不会泄漏任务或槽位。
- 主干 SLRU、Redis TTL/heartbeat、CPU-only match、LAYERBLOCK、HugePage、NIXL、NVFP4、nvCOMP 路径仍可用。
- `IndexerCacheConfig`、`enable_dp_attention`、`attn_cp_size`、`attn_cp_rank` 不再作为公开配置；调用方应迁移到 `LayerGroupSpec`、`cp_size`、`cp_rank`。旧字段必须产生可诊断错误，不能被静默忽略。

## 6. GPU 数据面回归（Phase 2）

### 6.1 Full-KV 与主干能力

```bash
pytest -q tests/test_kv_transfer_correctness.py -ra
pytest -q tests/test_nvfp4_roundtrip.py -ra
pytest -q tests/nvcomp/test_nvcomp_cpu.py tests/nvcomp/test_nvcomp_ssd.py -ra
pytest -q tests/hugepage -ra
```

覆盖 bf16、fp8、NVFP4，BLOCKFIRST、LAYERFIRST、LAYERBLOCK，CPU、SSD、GDS、
REMOTE，以及 H2D/D2H、DISK2H/H2DISK、D2DISK/DISK2D。每条 roundtrip 必须
字节一致。

MLA D2H 模式分别运行：

```bash
for mode in sharded all_write rank0_only; do
  FLEXKV_MLA_D2H_MODE="$mode" pytest -q tests/test_kv_transfer_correctness.py -ra || exit 1
done
```

### 6.2 LayerGroup、layerwise 与 SWA

```bash
pytest -q \
  tests/test_layerwise_multi_group_swa.py \
  tests/test_layerwise_dsv4_multi_group_swa_roundtrip.py \
  tests/test_layerwise_eventfd_timing.py \
  tests/test_swa_control_plane.py \
  tests/test_swa_control_plane_e2e.py \
  tests/test_swa_dispatch.py \
  tests/test_swa_level2_single_pass.py \
  tests/test_swa_node_mount.py \
  tests/test_swa_ssd_staging_e2e.py \
  tests/test_swa_storage_layout.py -ra
```

layerwise 通知模式分别运行：

```bash
for mode in hostfunc polling; do
  FLEXKV_ENABLE_LAYERWISE_TRANSFER=1 \
  FLEXKV_LAYERWISE_NOTIFY_MODE="$mode" \
  pytest -q \
    tests/test_kv_transfer_correctness.py \
    tests/test_layerwise_eventfd_timing.py \
    tests/test_layerwise_multi_group_swa.py \
    tests/test_layerwise_dsv4_multi_group_swa_roundtrip.py -ra || exit 1
done
```

必须确认每个 original layer 的 eventfd 只触发一次，并且是在该 layer 的所有 group
member、所有 GPU 及 SWA sidecar 完成后触发。`polling` 不得丢通知、重复通知、死锁或
在析构时遗留线程/CUDA event。

## 7. 框架与分布式集成（Phase 3）

### vLLM

- 运行 `.github/workflows/vllm-compat-test.yml` 的 Python 3.10/3.11 矩阵。
- 验证 vLLM 0.23+ 非 MLA `LAYERBLOCK` 注册。
- 验证 NVFP4 packed head size 为 `head_size // 2 + head_size // 16`，PUT/GET 字节一致。
- 验证 PP、DP、CP rank 映射和 batch task 生命周期。

### SGLang / DSv4

- 使用 `flexkv/integration/sglang/sglang_flexkv_connector.patch` 对应的 SGLang 基线。
- 覆盖 DSv4 main KV + indexer LayerGroup + SWA，多池注册、整页匹配、PUT、GET、取消和驱逐。
- 至少运行 TP2、PP2、DP2、CP2 中各一种配置，并运行一个多节点组合。
- `FLEXKV_ENABLE_SWA_TRANSFER=0/1` 都要验证；关闭时必须退化为 Full-KV 且无 SWA op。

### TensorRT-LLM、Redis 与 Mooncake

- TensorRT-LLM：基础 PUT/GET、PP、CP、subprocess TransferManager 冒烟。
- Redis/P2P：多 instance 注册、TTL/heartbeat、进程异常退出后的节点清理、CPU/SSD P2P。
- Mooncake：跨节点 PUT/GET、buffer 注册/注销和失败重试。
- 至少进行一次进程中断测试，确认 KVTask、锁、SWA slot 和临时 buffer 均释放。

## 8. 性能与稳定性（Phase 4）

以相同硬件、模型、请求集分别测试 `origin/main` 与集成提交：

```bash
python benchmarks/benchmark_cache_engine.py --help
python benchmarks/microbenchmark_ce_overhead.py --help
python benchmarks/microbenchmark_notify_mode.py --help
```

- 非 DSv4 工作负载吞吐下降超过 5%，或 P99 延迟上升超过 5%，标记为 `FAIL`。
- 分别记录 `hostfunc` 与 `polling` 的每层通知延迟、CPU 占用和吞吐。
- DSv4/SWA 端到端持续运行至少 1 小时；不得出现数据不一致、slot 泄漏、任务增长、死锁或显存/主存持续增长。

## 9. 结果回填

统一回填文件：`docs/dpskv4_main_merge_validation_results.md`。

- 每个硬件/框架环境新增一个独立小节，不要覆盖其他 agent 的结果。
- 结果必须绑定被测 commit SHA；分支更新后，旧结果不得沿用。
- `FAIL` 必须包含失败命令、完整错误、最小复现和初步归属。
- `BLOCKED` 必须说明缺少的权限、硬件、依赖或外部服务，不能只写“环境问题”。

单环境记录使用以下模板：

```text
Commit:
Tester/agent:
Date:
Host/GPU:
OS/driver/toolkit:
Python/PyTorch:
Framework versions:

Static checks: PASS / FAIL / BLOCKED
Default build: PASS / FAIL / BLOCKED
P2P build: PASS / FAIL / BLOCKED
GDS/NIXL build: PASS / FAIL / BLOCKED
nvCOMP build: PASS / FAIL / BLOCKED
Unit tests: passed / failed / skipped
Smoke tests: passed / failed / skipped
NVIDIA GPU tests: PASS / FAIL / BLOCKED
AMD GPU tests: PASS / FAIL / BLOCKED
SGLang DSv4: PASS / FAIL / BLOCKED
vLLM: PASS / FAIL / BLOCKED
TensorRT-LLM: PASS / FAIL / BLOCKED
Redis/Mooncake multi-node: PASS / FAIL / BLOCKED
1-hour soak: PASS / FAIL / BLOCKED
Performance delta:

Failed command and full log:
Minimal reproduction:
Suspected subsystem/owner:
```

任一必测项为 `FAIL` 或无理由 `BLOCKED` 时，不应将 Draft PR 转为 Ready。
