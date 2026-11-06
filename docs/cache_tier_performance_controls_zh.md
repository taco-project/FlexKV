# 缓存层性能控制

[English](cache_tier_performance_controls.md)

以下可选控制项限制远端读压力并减少 SGLang 控制消息开销。默认使用串行 Store 读取、
在已有 session/字节限制内不限制数据 session 数，并保留旧控制消息格式。

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `FLEXKV_MOONCAKE_GET_CONCURRENCY` | `1` | 每个数据 client 同时发起 1–8 个 SDK Get；仅查询的 client 仍为串行。 |
| `FLEXKV_MOONCAKE_GET_MAX_INFLIGHT_BYTES` | `536870912` | 每个 client 所有 Get 的对齐后字节预算；并行读取时必须为正。 |
| `FLEXKV_PREFETCH_DATA_CONCURRENCY` | `0` | 执行数据 I/O 的 session 上限；0 使用原有不限制策略。元数据准入和保留的结果 lease 仍计入 `max_sessions`。 |
| `FLEXKV_CONTROL_INLINE_BYTES` | `0` | 取值 0、256、512、1024 或 4096；同一 rank 组使用相同值，仅 PP=1 时启用。 |

并行读取时，每个 Get 对象占用预算为 `ceil(size / 4096) * 4096 + 8192`。
配置的并发数是上限；为了让最大对象适配其预算份额，实际并发数可能降低。例如，
512 MiB 预算下，八个 64 MiB 对象由于对齐开销最多使用七个 reader。超过总预算的
对象会在提交任何读取前被拒绝。

每项任务不超过总预算除以实际并发数的份额。小读取按对象数分配给 worker；SDK
批次还遵守独立的原始字节限制 `max_get_batch_bytes`。已完成任务可以立即让新任务
准入，不必等待较慢任务；结果按输入顺序收集。并发与字节限制覆盖共享同一 client
的所有外层调用，其 `batch_get` 调用仍串行。串行读取保留原始字节批处理行为；
超大对象在提交 SDK 前失败。短读、失败和结果数量不匹配不会被报告为成功命中。
错误抛出前，所有已接受调用都必须排空；worker 关闭时也先排空 Get，再注销目标
内存区域。

等待其他 session 字节额度的 session 保持 pending。若只有未完成 I/O 导致 pin 预算
暂时不足，初始 CPU 前缀 pin 也会等待；若前缀与保留结果 lease 同时存在时无法
容纳，则立即停止。没有未完成 I/O 占用额度时，chunk 预留使用永久容量不足的回退
路径。数据 session 的位置会保留到已提交图全部排空，即使 session 已取消；I/O 完成
不意味着 CPU 结果/pin lease 已释放。

Inline 格式把长度和小的序列化对象放入一个定长消息；较大对象则使用定长头和后续
payload。leader 只序列化一次用于 fan-out。一个 rank 组中不能混用旧格式和 inline
格式。

## 验证

```sh
pytest tests/test_mooncake_bounded_io.py tests/test_mooncake_parallel_get.py tests/prefetch/test_coordinator.py tests/prefetch/test_prefix_admission.py
torchrun --standalone --nproc-per-node=4 tests/run_control_framing_distributed.py
```

SDK 测试使用受控线程函数；分布式测试使用真实四 rank CPU Gloo，检查旧格式、
256 字节与 1024 字节格式，包括大 payload 和发送未完成时的垃圾回收。这些检查不
证明真实 RDMA、GPU/模型正确性或生产验收。模型级验证和测量需要记录精确源码版本、
工作负载、硬件及匹配的原生 GPU cache 基线；数据通路完成和吞吐量不能证明模型输出
正确。
