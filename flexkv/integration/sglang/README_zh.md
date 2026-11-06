# 在 SGLang 中使用 FlexKV

[English](README.md)

## 版本兼容性

请根据模型选择对应的 SGLang 集成路径。

> 状态最后确认日期：2026 年 8 月 6 日。

| 使用场景 | SGLang 版本 | 是否需要 patch |
| --- | --- | --- |
| 普通模型 | SGLang `v0.5.16` 及以上版本，或较新的 `main` | 不需要 |
| DeepSeek V4 | SGLang PR [#31781](https://github.com/sgl-project/sglang/pull/31781)，固定到 commit [`ee0465a`](https://github.com/sgl-project/sglang/commit/ee0465a09196421a6e4d53a3103eccdef1dd32ac) | 不需要本地 patch；使用指定的 SGLang commit |
|DeepSeek V4.1 |SGLang PR [#31781](https://github.com/sgl-project/sglang/pull/31781)，固定到下列 commit | 不需要本地 patch；使用指定的 SGLang commit e74b2ec897256fe50bffaa97a066e14bfc113c1f |

### 普通模型：直接使用 SGLang 官方版本

基础 FlexKV 集成已通过
[sglang#29701](https://github.com/sgl-project/sglang/pull/29701) 合入 SGLang
官方主干，并从 SGLang `v0.5.16` 开始随版本发布。使用这些版本时，**不要**再应用
`sglang_flexkv_connector.patch`。

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.16  # 也可以使用更新的正式版本或 main
```

### DeepSeek V4：固定使用尚未合入的 SGLang 适配

FlexKV 侧的 DeepSeek V4 支持已经通过
[FlexKV#225](https://github.com/taco-project/FlexKV/pull/225) 合入 FlexKV
`main`，但配套的 SGLang 适配**尚未合入**官方主干。在
[sglang#31781](https://github.com/sgl-project/sglang/pull/31781) 合入之前，不要只使用
SGLang 正式版本或 `main`，而应拉取该 PR 并固定到下面的 commit：

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/31781/head
git checkout -b flexkv-dsv4 ee0465a09196421a6e4d53a3103eccdef1dd32ac
```

同时使用当前 FlexKV `main` 分支：

```bash
git clone https://github.com/taco-project/FlexKV.git
cd FlexKV
git checkout main
```

该 PR 仍在开发中，更新固定的 SGLang commit 前请先检查 PR 状态。这个 commit
暂不支持 DeepSeek V4 unified-KV layout 和 `--enable-hisparse`。

### DeepSeek V4.1：固定使用尚未合入的 SGLang 适配
FlexKV 侧的 DeepSeek V4.1 支持已经通过
[FlexKV#303](https://github.com/taco-project/FlexKV/pull/303) 合入 FlexKV
`main`，但配套的 SGLang 适配**尚未合入**官方主干。在
[sglang#31781](https://github.com/sgl-project/sglang/pull/31781) 合入之前，不要只使用
SGLang 正式版本或 `main`，而应拉取该 PR 并固定到下面的 commit：


```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/31781/head
git checkout -b flexkv-dsv41 e74b2ec897256fe50bffaa97a066e14bfc113c1f
```
上述 commit 来自本仓库的历史记录；部署前请确认它仍可获取，并验证与所用 FlexKV 和模型兼容。

同样适用当前FlexKV的`main`分支即可完成部署。

## 启动方式

安装 SGLang 和 FlexKV 后，先创建 FlexKV 配置文件。例如：

```yaml
# flexkv_config.yaml
cpu_cache_gb: 16
```

使用 SGLang 内置的 FlexKV backend 启动服务：

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --enable-flexkv \
  --flexkv-config-file /path/to/flexkv_config.yaml \
  # ... 其他 SGLang 参数
```

也可以使用等价的显式参数 `--radix-cache-backend flexkv`。更详细的配置和验证方法请参考
SGLang 官方的
[`flexkv/README.md`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/flexkv/README.md)。

## 分段远端预取

分段预取默认关闭。它把单节点 Mooncake Store → CPU 的预取拆成较小的传输图，
需要配套 SGLang scheduler/cache 钩子；仅安装 FlexKV 不会添加这些钩子。应使用
兼容的 SGLang 版本，并对实际模型和运行环境一起验证。分段路径不支持 SSD、
FlexKV P2P 共享、多节点和 TensorRT-LLM remote 模式。

在已有 FlexKV YAML 配置中添加以下片段，保留 CPU 容量和 Mooncake Store 设置，
并将 SSD 容量设为 0：

```yaml
enable_chunked_prefetch: true
prefetch_options:
  policy: best_effort
  chunk_max_blocks: 128
  max_inflight_chunks: 2
prefetch_max_sessions: 128
prefetch_max_reserved_bytes: 536870912
prefetch_max_pinned_bytes: 2147483648
prefetch_result_ttl_s: 60
```

这些限额在现有 CPU 池内记账，不是额外内存分配。`chunk_max_blocks`、
`max_inflight_chunks` 以及全局 session、预留字节、pin 字节和结果 TTL 限额均须为
正数。显式 FlexKV policy 优先；未指定时 connector 读取 SGLang 的
`hicache_storage_prefetch_policy`。

| 策略 | SGLang 中的行为 |
| --- | --- |
| `wait_complete` | 使用原有整任务 `prefetch_async`，并关闭分段运行时。 |
| `timeout` | 时间预算到期后停止准入新 chunk，再排空已提交工作。 |
| `best_effort` | scheduler 发出 demand 时停止准入，再排空已提交工作。 |

超时限制的是**准入时间**，不是响应时间；在飞传输不会被强制取消。结果只能是连续
且成功发布的前缀，容量不足或失败时可能是部分结果或空结果。Full+SWA 需要 Full KV
和对应 SWA/state 同时构成可用 checkpoint；配套 SGLang adapter 需支持模型的
checkpoint 语义。取消 session 后，已发布数据可能仍留在缓存中。

`timeout_budget_s` 会覆盖公式 `min(timeout_max_s, timeout_base_s +
timeout_per_ki_token_s * candidate_tokens / 1024)`；后三者默认分别为 30.0、2.0
和 0.1 秒，`timeout_budget_s` 默认未设置。只使用环境变量配置时，可设置
`FLEXKV_ENABLE_CHUNKED_PREFETCH=1` 和例如
`FLEXKV_PREFETCH_OPTIONS='{"policy":"timeout","timeout_budget_s":0.02}'`；
四项限额也分别支持 `FLEXKV_PREFETCH_MAX_SESSIONS`、
`FLEXKV_PREFETCH_MAX_RESERVED_BYTES`、`FLEXKV_PREFETCH_MAX_PINNED_BYTES` 和
`FLEXKV_PREFETCH_RESULT_TTL_S`。已运行的 external server 保留自身配置，客户端
不会重新配置它。

直接调用的 `KVManager` session API 包括 `prefetch_capabilities()`、
`start_prefetch`、`progress_prefetch`、`poll_prefetch`、
`notify_prefetch_demand`、`stop_prefetch`、`wait_prefetch` 和
`release_prefetch`。应传入完整 token 链，位置为绝对 token 偏移。`poll` 和 `wait`
只观察状态，不发送 demand；wait 超时后 session 仍活动。终态 snapshot 区分计划
命中和 `reusable_prefix_end_token` / `l3_loaded_spans`，并提供 `lease_valid`、
已提交和在飞工作。应在结果 lease 有效时取得 held GET/match，再释放 lease；释放
活动 session 会停止后续准入，其资源在已提交工作排空前仍由服务持有，释放后不保证缓存继续驻留。

```python
import numpy as np
from flexkv.prefetch import PrefetchOptions

handle = manager.start_prefetch(
    np.asarray(full_token_ids, dtype=np.int64),
    PrefetchOptions(policy="timeout", timeout_budget_s=0.020,
                    chunk_max_blocks=32, max_inflight_chunks=2),
    namespace=["tenant", "model-version"],
)
snapshot = manager.poll_prefetch([handle])[handle]
snapshot = manager.wait_prefetch(handle, timeout_s=5)
# Obtain a held GET/match while snapshot.lease_valid, then release the lease.
manager.release_prefetch(handle)
```

共享 Store 时必须保持模型、权重、dtype、TP 和 KV 布局一致：现有 key 不包含完整
布局指纹。变更这些配置时，除非已验证所部署 adapter 的所有相关操作都使用正确的
namespace 隔离，否则应使用独立 Store。正确性验证应确认远端命中，将加载的 KV 字节
和生成结果与适当的冷算基线比较，检查 stop/abort 及排空后的资源。CPU 测试不能
证明 GPU 或真实传输正确性；现有测试见 [`tests/prefetch`](../../../tests/prefetch)。

## 可选的 Index-K 去重

对兼容的 DSA 模型，`FLEXKV_DEDUP_INDEXER_GROUP=1` 只注册实际使用的 Index-K
buffer，默认关闭。切换该选项时应使用新缓存池，或使用隔离的布局/版本 namespace：
现有缓存不会自动迁移或清空。支持的布局和已有验证的边界见
[IndexCache 去重说明](../../../docs/indexcache_dedup_zh.md)。

## 集成边界

FlexKV connector 的核心逻辑（`FlexKVComm` 和 `FlexKVConnector`）位于本目录：

| 文件 | 职责 |
|------|------|
| `comm.py` | 3-axis（PP × CP × TP）跨 rank 通信层 |
| `connector.py` | `FlexKVConnector`，封装 `KVManager` 供 SGLang 调用 |

SGLang 侧的 `storage/flexkv/__init__.py` 通过 `importlib` 从 `flexkv.integration.sglang.connector`
动态加载 connector，后续 FlexKV 逻辑改动通常只需更新 FlexKV 仓库。

设置环境变量 `FLEXKV_ENABLE_COLLECTIVE_SYNC=0` 可禁用跨 rank 同步（scatter/barrier/all_reduce），
在不使用 Pipeline Parallelism (PP) 的部署中关闭可减少同步开销、提升性能。

## 关于本目录中的 patch

`sglang_flexkv_connector.patch` 仅作为 FlexKV 尚未合入 SGLang 主干前的历史参考保留。
受支持的 SGLang 正式版本不需要它，上述 DeepSeek V4 路径也不能使用它。
