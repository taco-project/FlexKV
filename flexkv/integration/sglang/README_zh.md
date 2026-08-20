# 在 SGLang 中使用 FlexKV

[English](README.md)

## 版本兼容性

SGLang 目前有两条不同的 FlexKV 集成路径，请根据模型选择。

> 状态最后确认日期：2026 年 8 月 6 日。

| 使用场景 | SGLang 版本 | 是否需要 patch |
| --- | --- | --- |
| 普通模型 | SGLang `v0.5.16` 及以上版本，或较新的 `main` | 不需要 |
| DeepSeek V4 | SGLang PR [#31781](https://github.com/sgl-project/sglang/pull/31781)，固定到 commit [`ee0465a`](https://github.com/sgl-project/sglang/commit/ee0465a09196421a6e4d53a3103eccdef1dd32ac) | 不需要本地 patch；使用指定的 SGLang commit |

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
