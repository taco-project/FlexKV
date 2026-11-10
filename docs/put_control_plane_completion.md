# PUT 控制面完成处理（2026-09-23）

## 结论与范围

v3 task 5 在 store_created 后 6.696 ms 解锁 GPU，但图级 task_end 在
12249.686 ms 才出现。修复目标是让图级收尾不依赖下一次请求，不改变
D2H、SSD/远端搬运实现，也不把 GPU 解锁延后到整个图结束。

对照 FlexKV-PR/FlexKV 的 connector：新版 wait(timeout=0, completely=True)
把图级完成作为 store 返回条件，本次不直接采用，避免多级存储拖住 GPU。
对照较新 SGLang 的 FlexKV radix：调度检查同时推进 store 和 load tail；
在一些特殊机型和场景下已补齐 scheduler -> radix -> connector 检查入口，并按 pending 状态限流。

## 实现

- kvtask.py：记录提前返回但尚未图级完成的任务；progress_returned_tasks()
  在原拥有者线程非阻塞推进。正常完成/失败/取消经 release 移除跟踪；
  reset/任务过期的防御清理最多每秒扫描一次，避免每 tick 扫描全部待收尾图。
  无待收尾任务时直接返回，不访问 IPC、不读取诊断时钟、不打印。
- kvmanager.py：暴露本地推进接口；server-client 模式不发送空 RPC。
- integration/sglang/connector.py：有 inflight store 时仍由原 try_wait 推进；
  仅此状态进入 TP completion collective。GPU rid 只返回一次。GPU 返回后的
  图级推进拆成 leader-local 接口，推进异常不再触发空 collective，每段连续
  异常仅报告一次。
- sglang-special-platform：有 inflight store 时在每个 scheduler/batch 边界检查；GPU
  解锁后仅在 sticky graph-pending 为真时每 50 ms 推进；完全空闲时不调用
  FlexKV。cache_finished_req()/evict() 的空 drain 也不会进入 collective。
- server/server.py：server 自己推进提前返回任务。存在待收尾图时以 1 ms
  request socket poll 等待新请求并继续回收；没有待收尾图恢复阻塞接收。
  不新建缓存修改线程，不在空 poll 时记录日志。
- transfer/transfer_engine.py：增加成功/失败图终止通知入队前的诊断事件。

非阻塞仅表示不等待新的完成通知，不代表现有批量回调处理耗时有硬上限。
单轮处理预算、背压和调度 collective 优化尚未修改，需要并发数据支持。

## 诊断输出

全部新增正常诊断事件遵循现有 put_trace.enabled/output_file，无新开关：

- graph_terminal_enqueue：每个引擎每图终止一次，含 graph/scheduler/status。
  记录点在 queue.put 前；不是 IPC 接收时间。
- completion_drain：仅本轮实际取得完成事件时一条，含 ops、duration_ns、
  pending_returned、pending_returned_peak、tracked_tasks。耗时包含通知读取、
  回调及该轮已有 trace 开销。
- graph_finalize：正常图终止处理一次，含回调/终态处理 duration_ns、
  response_returned、status；失败图回滚仍由原 task_end 表示。
- graph_progress_error：本地迟到回收连续异常首次一条；同时标准 logger
  报告一次异常。恢复后再次失败可以重新报告，不改变原错误日志配置。
- store_poll_summary：活跃期间每进程/rank 最多每秒一条，并在最后一批
  inflight store 完成时刷新不足一秒的尾窗口；内存累计 active/empty poll
  次数、总/最大 poll、try_wait、collective sync 时间以及最大 poll 间隔。
  同一条记录附带这一窗口内 forward 次数、总耗时和最慢 forward 的区间、
  mode、batch size，用于判断 completion gap 是否落在模型执行期间。
  inflight 超过 32 时停止逐 store 扫描，只保留 O(1) 聚合，并通过
  detail_suppressed_poll_count 标明；避免诊断本身随并发数线性放大。

不逐 poll 或逐 forward 输出；热路径只在 trace 开启时读取单调时钟并更新
进程内整数，JSONL/flock 写入频率上限为每秒一次/rank加每批完成一次。trace 关闭时 polling
装饰器仍直接调用原函数，forward bridge 不读取时钟。图终止发送事件按
run/graph/scheduler 关联 op_completed，再关联 request/task；多引擎应检查
所有终止事件，不能仅取第一个引擎。

## 独立的 store 积压观测

该观测不依赖 put_trace，也不改变限流、合批或 TP collective：

- `FLEXKV_RUNTIME_DIAG=0`：关闭全部运行时诊断。
- `FLEXKV_STORE_Q_DIAG=0`：只关闭 store 积压诊断。
- `FLEXKV_STORE_Q_DIAG_INTERVAL_S`：活跃队列日志间隔，默认 30 秒，
  最小 5 秒。

INFO 日志 `[FlexKV-STORE-Q]` 只由 sync leader 输出，包含 inflight、
oldest_age_s 和 age_p50_s。单任务完成日志包含 wait_ms、inflight_left、
put_match_ms、filter_ms、launch_ms、unmatched 和 slots。wait_ms 从异步
launch 成功返回后开始，到 FlexKV 报告 GPU 可释放为止；它是 store
积压代理，包含 D2H、完成通知和调度消费时间，不能直接当作纯 DMA 时间。
调用方也可以通过只读 `inflight_store_count()` 获取当前积压数量。

未开启 INFO 日志时不扫描队列；开启后最多按配置间隔排序一次任务年龄。
阶段计时只在该诊断启用时读取单调时钟。

## 验证与下一轮

CPU 测试使用生产函数 AST 与模拟传输/缓存，不代表一些特殊机型和场景 性能验证：

    python3 tests/test_store_queue_observation_standalone.py
    python3 tests/test_put_terminal_drain_standalone.py
    python3 tests/test_completion_trace_standalone.py
    python3 tests/test_put_trace_standalone.py

另运行 sglang-special-platform/test/manual/test_flexkv_completion_polling.py。
共 63 项通过，覆盖迟到/乱序完成、失败、空闲停止回收、共享发布、
API 包装层、server 无新请求继续回收、空闲不进入 collective、图尾低频推进
以及 store 积压观测的节流、计时与状态清理。

部署需同步并重新构建 FlexKV（服务器当前加载 .so），不能只覆盖 .py。
SGLang 还需同步 scheduler.py、flexkv_radix_cache.py 和新增的
flexkv_put_trace.py；scheduler 在实际 `_run_batch_forward()` 外记录聚合
span，并执行 pending-aware 调度。

下一轮先做单请求后完全停流，确认最后一笔 task_end 自行出现；再进行
持续并发和突发后停流，比较 TTFT/ITL、GPU 解锁延迟、终止发送到消费延迟、
completion_drain 耗时与 pending_returned。若回收批次确实长时间占用调度线程，
再引入保序、不会丢事件的分批处理预算；不提前增加复杂度。
