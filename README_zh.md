# FlexKV: 面向高性能分布式推理的KVCache Manager

**English Version**: [README.md](README.md)

FlexKV是腾讯云TACO团队和社区合作开发推出的面向超大规模 LLM 推理场景的分布式 KV Store 与多级缓存管理系统，利用多级缓存支撑推理引擎以获取更大吞吐及更低延迟。

FlexKV 采用 **Apache-2.0 开源协议**，详细信息请参见 [LICENSE](LICENSE) 文件。

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

### 以 vLLM 为例使用 FlexKV

见[docs/vllm_adapter/README_zh.md](docs/vllm_adapter/README_zh.md)

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

## Branch
- main 为稳定分支，维护已经测试过的commit。需要稳定的代码请从此分支拉取。
- dev 为开发分支，维护较新特性。需要新特性和开发新特性请从此分支拉取和合入。
- bugfix 为bug分支，维护需要立即解决的bug或需要立即更新的文档。需要解决bug和立即更新的文档请从此分支拉取和合入。
- stable 为上一个版本的main分支位置，仅用于回滚以及极其保守的情况使用（如产品化）。不鼓励使用此版本。

## Roadmap

- **缓存引擎共进程化**：dev 分支将进一步优化 Cache Engine 的实现、集成和调用，并同步更新相关 API 支持
- **加速框架支持**：对 vLLM、SGLang 等主流推理框架的适配将陆续发布
- **分布式查询支持**：实现可扩展的分布式 KVCache 查询能力
- **延迟优化**：通过预取、压缩等手段进一步降低 *get* 请求延迟