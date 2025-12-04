# FlexKV: 面向高性能分布式推理的KVCache Manager

**English Version**: [README.md](README.md)

FlexKV是腾讯云TACO团队和社区合作开发推出的面向超大规模 LLM 推理场景的分布式 KV Store 与多级缓存管理系统，利用多级缓存支撑推理引擎以获取更大吞吐及更低延迟。

FlexKV 采用 **Apache-2.0 开源协议**，详细信息请参见 [LICENSE](LICENSE) 文件。

## 最新版本主要变更
### 功能
通用功能:
- 添加本地 get/put 的操作级回调 [#13](https://github.com/taco-project/FlexKV/pull/13)
- 添加分布式 KV Cache 共享支持，支持 CPU 和 SSD 之间的 KV Cache 共享，以及 PCFS 的分布式共享 ([#17](https://github.com/taco-project/FlexKV/pull/17))
- 添加 GDS (GPU Direct Storage) 支持 ([#25](https://github.com/taco-project/FlexKV/pull/25))
- TP16 支持 ([#26](https://github.com/taco-project/FlexKV/pull/26))
- 支持更多 kv cache 布局。现在包括：vLLM、SGLang、TensorRT-LM ([#27](https://github.com/taco-project/FlexKV/pull/27))
- GDS 重构和 gtensor 支持 ([#42](https://github.com/taco-project/FlexKV/pull/42))
- 支持直接从 CUDA IPC Handle 构造 TensorSharedHandle ([#44](https://github.com/taco-project/FlexKV/pull/44))


针对 vLLM:
- 在 vLLM 集成中支持 dp > 1 ([#18](https://github.com/taco-project/FlexKV/pull/18))
- 添加 vLLM 适配的启动脚本 ([#47](https://github.com/taco-project/FlexKV/pull/47))
- 支持 vLLM+FlexKV 的 TP16 ([#59](https://github.com/taco-project/FlexKV/pull/59))

针对 TensorRT-LLM
- 在 TensorRT-LLM 上支持使用 FlexKV ([#48](https://github.com/taco-project/FlexKV/pull/48))
- 支持 TensorRT-LLM+FlexKV 的 TP16 ([#53](https://github.com/taco-project/FlexKV/pull/53))
### 优化
- MLA d2h 传输优化 ([#19](https://github.com/taco-project/FlexKV/pull/19))
- 优化 SSD I/O ([#33](https://github.com/taco-project/FlexKV/pull/33))
- 增强缓存淘汰机制，引入频率感知的宽限时间 ([#38](https://github.com/taco-project/FlexKV/pull/38))
- 在 RadixTree 中使用 std::unordered_map 替代 std::map ([#41](https://github.com/taco-project/FlexKV/pull/41))

更多详细信息，请参阅 [CHANGELOG](CHANGELOG.md)

## 如何使用

### 安装依赖

```bash
apt install liburing-dev
apt install libxxhash-dev
```

### 编译 FlexKV

```bash
./build.sh
#./build.sh --release for cython package
```

### 在 vLLM 中使用 FlexKV

见[docs/vllm_adapter/README_zh.md](docs/vllm_adapter/README_zh.md)

### 在 TensorRT-LLM 中使用 Flexkv

见[docs/trtllm_adaption/README_zh.md](docs/trtllm_adaption/README_zh.md)

### FlexKV和Dynamo框架的集成

见[docs/dynamo_integration/README_zh.md](docs/dynamo_integration/README_zh.md)

## 设计框架

<div align="center">
  <img src="docs/images/flexkv_architecture.png" alt="FlexKV Architecture" width="70%" />
</div>

FlexKV 整体由三个核心模块构成：  
- **StorageEngine**  
- **GlobalCacheEngine**  
- **TransferEngine**

### StorageEngine

StorageEngine 根据配置初始化三级缓存，将请求中的多个 token 组合成一个 block，并以 block 为单位存储 KVCache，保持与 GPU 内部完全一致的 KV 形状（shape）。通过 block ID 计算实际存储偏移量。

此外，用户可启用 *block-wise 模式*，将多个网络层（layer）及 KV 的缓存合并为更大的 block，通过增大 I/O 单位提升传输效率。

### GlobalCacheEngine

GlobalCacheEngine 作为 FlexKV 的控制面，负责决策数据传输的方向，以及源端与目标端的 block ID。

缓存维护：
- 一个 **RadixTree**，用于前缀匹配（match/insert）
- 一个 **内存池（mempool）**，用于追踪空间使用并触发淘汰机制

当新请求到达时，GlobalCacheEngine 会比较三级存储介质中已匹配的 token 数量，并决策是否从 SSD 或 扩展存储 拉取对应 block，经由 内存 中转至 GPU。

### TransferEngine

TransferEngine 作为 FlexKV 的数据面，负责执行 GlobalCacheEngine 的传输决策。

核心特性包括：
- 多进程架构：每个进程通过多线程实现并行传输
- 支持io_uring等高性能 I/O 技术，进一步加速数据传输

### FlexKV 三级缓存

FlexKV 利用成本更低的存储介质，缓解因 GPU 显存不足而导致 KVCache 被丢弃并重新计算的问题。

三级缓存结构如下：
- **CPU 内存** —— 第一级外部缓存
- **本地 SSD** —— 第二级持久化缓存
- **扩展存储（如云存储等）** —— 第三级分布式缓存，支持更大容量与跨节点共享

FlexKV 在处理 *get* 请求时：
- 检索三级缓存，提升命中效率
- 当空间不足时，执行逻辑层面的 LRU 淘汰，无需触发实际数据传输。

#### 异步调用设计：
- *get*请求可以异步调用，*get*匹配和传输时间可以通过预取与之前的计算重合。
- *put*请求可以异步调用，从GPU copy到内存的时间可以与之后的计算重合。内存与SSD以及扩展存储间的传输则完全由TransferEngine之后执行，主进程不感知。

## 分支策略

本项目的分支管理策略如下：

- **`main` 分支**：主开发分支，包含最新的功能和变更。所有拉取请求都直接合并到 `main` 分支，以确保快速迭代和持续集成。

- **`release-*` 分支**：当 `main` 分支达到稳定状态时，我们会创建专门的发布分支（例如 `release-1.0`、`release-1.1`），为用户提供稳定、可用于生产环境的版本。

注意：在已发布版本中发现的关键修复会直接应用到对应的 `release-*` 分支，然后回退到 `main` 分支，以保持所有活跃分支的一致性。

## Roadmap

- **缓存引擎共进程化**：dev 分支将进一步优化 Cache Engine 的实现、集成和调用，并同步更新相关 API 支持
- **加速框架支持**：对 vLLM、SGLang 等主流推理框架的适配将陆续发布
- **分布式查询支持**：实现可扩展的分布式 KVCache 查询能力
- **延迟优化**：通过预取、压缩等手段进一步降低 *get* 请求延迟
