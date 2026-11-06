# 更新日志

[English](CHANGELOG.md)

本文记录项目的重要变更，格式参考 Keep a Changelog，并遵循语义化版本约定。

## 未发布

### 功能

通用：

- 增加 radixshmem 模式（`FLEXKV_ENABLE_RADIXSHMEM=1`，需要 radixshmem `3608a23`
  或更新版本）：运维在每个节点启动 `radix-server`，由其管理 CPU radix 索引、
  SlotStore 和跨节点 RDMA 拉取。FlexKV 通过 `radixshmem.RadixClient(name, Geometry)`
  提交每 block token 数、CPU/SWA page 字节数、SWA 窗口和 slot 对齐，采纳服务端按
  字节预算规划的 slot 数。此模式下 `cpu_cache_gb` 不决定 CPU 层容量。
  `FLEXKV_RADIXSHMEM_SERVER_NAME` 默认 `/flexkv`；ready 等待与 prefetch 上限为
  固定默认值。单 DP 使用引擎内模式，多 DP 或多实例使用节点 KVServer。仅支持 CPU
  层，需关闭 SSD、remote 和 FlexKV P2P。详见 [radixshmem 配置](docs/radixshmem/config_zh.md)。
- `KVServer.create_server(inherit_env=False)` 将 `PYTHONPATH`、`LD_LIBRARY_PATH`、
  `PATH` 及 `FLEXKV_*` 变量传给子进程。
- `gen_hashes` / `Hasher.update` 直接对 numpy buffer 求 hash，支持并发调用；
  `gen_hashes_numpy` 校验 dtype、连续性和大小。

SGLang：

- SGLang `v0.5.16` 及以上原生包含 FlexKV backend，无需 patch（sglang#29701）。
- 增加 DeepSeek-V4 异构 C4/C128/indexer KV group、FullKV + SWA、attention/indexer
  compress-state sidecar 和 layerwise restore（#225）。
- 对应 SGLang 适配的历史说明使用 PR #31781 的 `ee0465a`；部署时需核对实际版本。

### 文档

- 以按版本区分的中英文集成指南替代旧 SGLang patch 流程。
- 增加中英文 CI 指南，说明 runner、release wheel、CPU 测试、参考耗时、本地复现及产物上传。

## 1.2.0 — 2025-11-25

### 功能

- 支持 CPU、SSD 和 PCFS 分布式 KV cache 共享（#17）；支持 GDS（#25）和 TP16（#26）。
- 增加 vLLM、SGLang、TensorRT-LLM KV 布局（#27）；重构 GDS 并支持 gtensor（#42）；
  支持从 CUDA IPC handle 构造 TensorSharedHandle（#44）。
- vLLM 支持 DP > 1（#18）、启动脚本（#47）和 TP16（#59）。
- 支持 TensorRT-LLM 集成（#48）和 TP16（#53）。

### 优化

- MLA D2H 传输优化（#19）、SSD I/O 优化（#33）。
- 通过频率感知宽限时间改进驱逐（#38）；RadixTree 使用 `std::unordered_map` 替代 `std::map`（#41）。

### 修复

- 修复 vLLM 的 DeepSeek head 数量（#23）、CPU 命中长度大于 SSD 时的 put 错误（#24）、
  benchmark worker（#31）、radix tree 越界导致的段错误（#39）、cache_info（#40）、
  GPU 注册端口（#45）、SSD allocator（#46）、vLLM 初始化 KV head 数量（#67）及
  非 MLA 模型配置（#68）。

### 其他

- 增加 TensorRT-LLM 文档（#52）；简化并更新用户配置（#37、#43）。

## 1.1.0 — 2025-09-15

- 为本地 get/put 添加操作级 callback（#13）。
- 增加 Dynamo（#14）和 `flexkv_config.json`（#15）文档。

## 1.0.0 — 2025-09-11

### 新增

- 增加 C++ radix tree，可通过 `index_accel` 启用；同步 kernel launch。
- 将 cache engine 改为可由推理引擎直接调用的库，带来不兼容旧版的 API 变化。
- 增加 `evict_ratio` 配置；减少 kernel launch 间的空隙；增加 vLLM 0.10.1.1 适配。

### 修复

- 修复 Cython release package。

## 0.1.0 — 2025-08-29

- 初始版本，添加许可证。
