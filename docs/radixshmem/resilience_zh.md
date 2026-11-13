# RadixShmem 共享 CPU 模式：请求截止与物理排空

本改动对应系统方案 A 的 FlexKV 接入侧。RadixShmem 继续拥有 CPU
SlotStore；FlexKV 继续负责本地 H2D/D2H。搭配支持 operation protocol v2
的 RadixShmem 服务端和 Python 客户端使用。

`PrefetchWaiter` 为每个请求保留独立截止时间。等待超过预算时，PREFETCH
返回没有新增远端命中的结果，后续本地 GET 或计算继续执行。
底层 PullJob 的源 pin、staging 槽位、后台句柄及节点额度仍由 RadixClient
持有，直到后端 release 成功并完成结果确认。请求取消不会假装物理资源已释放。

planner 按 `resources_released` 保留正在排空的任务；reset 在物理任务
未完成时拒绝执行；close 失败时保留原 client/tree，允许稍后重试。
`peer` 命中计数移到最终结果校验后，计划命中不再当成已完成命中。
同一节点多个 FlexKV 进程的完全相同前缀，由 RadixShmem 服务端在
pin/staging 前做 singleflight；FlexKV 不共享一个会互相取消的 waiter。

方案 B 的人工保护集合、etcd 事务账本、双副本与修复 agent 位于
RadixShmem。FlexKV 不再维护另一份副本数，不使用 RHT 候选数量作为
淘汰许可。部署、指标和故障边界见 RadixShmem 的
`docs/resilient_shared_cache.md`。

新等待器的纯 CPU 单元测试可单独运行，无需导入父目录 CUDA fixtures：

```sh
python -m pytest -q --confcutdir=tests/radixshmem \
  tests/radixshmem/test_prefetch_waiter.py
```

现有 `test_radix_shmem_engine.py` 继续覆盖原生 CPU 池、几何和 planner
集成；其依赖缺失的部分会明确 skip。真实 GPU/RDMA 输出一致性、故障
窗口 TTFT/QPS 和生产恢复目标必须在目标设备另行验收。请求计时从
pull_async 返回后开始，原生远端树发现本身的耗时仍受其已有 transport
超时控制；此改动不能证明整个同步发现阶段受 PREFETCH_TIMEOUT_MS 约束。
