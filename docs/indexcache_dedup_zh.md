# GLM5.2 的 IndexCache 去重

[English](indexcache_dedup.md)

DSA 模型中标记为 `skip_topk` 的层可能复用前面层的 Index-K cache。如果把这些别名
分别注册为物理层，相同的 Index-K 数据会被重复存储和传输。

在每个 SGLang worker 设置 `FLEXKV_DEDUP_INDEXER_GROUP=1` 后，只注册实际使用的
Index-K buffer；默认关闭。紧凑 group 保留原始层 ID，因此 layerwise 传输仍等待
正确的模型层。按 page 打包的 Index-K 每页仍为一行，不会再次乘以 page size。CPU
和 SSD block 容量按紧凑布局重新计算。

例如，合成的八层池中只有 `[0, 3, 7]` 使用独立 Index-K，则注册三个 indexer
buffer。这只是映射示例，不是模型架构或实测容量；节省量取决于实际有效层数、KV
几何及字节预算。

初始实现支持每个 cache rank 都能看到完整 Index-K 池的 TP 和普通 CP。DSA cache
layer split、无效的 skip 元数据、空的或互相别名的有效 buffer、不一致的 buffer
几何或注册 group 成员会被拒绝。

切换开关时使用空缓存池或隔离的布局/版本 namespace。旧布局与紧凑布局不能复用
同一组 FlexKV 或 Mooncake cache 对象；该开关不会自动迁移或清理已有数据。

SGLang 的可选 `FLEXKV_DEFER_DUPLICATE_RESTORES=1` 是独立优化：在生产者发布 GPU
radix 条目前，延迟同一 Host 前缀的并发恢复。它需要相应的 SGLang 适配，与
IndexCache 布局去重无关。

现有回归测试使用合成 CPU tensor，覆盖布局、容量和注册约束。GPU 传输、GLM5.2
输出精度、性能及长时间运行仍需针对实际部署版本单独验证。
