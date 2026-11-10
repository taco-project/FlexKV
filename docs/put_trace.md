# GET/PUT 分阶段追踪

基于 special-platform-FlexKV 的 `compute-comm-overlap`，支持 CE、直接 GDR、Mooncake Store。
现有 `[XFER]` 只记录 worker 操作总耗时；新增追踪补充请求关联和阶段时间戳。
PUT、GET 和 Mooncake prefetch 的请求级记录带 `operation=put|get|prefetch`；
旧记录缺少该字段时按 PUT 解析。

## 开启和保存

在现有 FlexKV YAML 的顶层加入（与 `gdr` 同级）：

```yaml
put_trace:
  enabled: true
  output_file: /tmp/flexkv/put-trace.jsonl
```

让服务加载这份 YAML，照常启动即可；例如将 SGLang 的
`--flexkv-config-file` 指向该文件。JSON 配置也支持同样的字段。
**启用后直接写入指定文件，不需要设置日志级别、环境开关或重定向 stdout。**
在一些特殊机型和场景下启动配置已在 [examples/sglang_adaption/flexkv_special_gdr.yml](../examples/sglang_adaption/flexkv_special_gdr.yml)
启用上述设置；通用示例见 [tests/gdr/config.yaml](../tests/gdr/config.yaml)。

- 自动创建父目录，同一次启动里所有本机 rank / worker 追加写入同一个文件，不覆盖已有内容。
- `enabled: false`（默认）不创建文件、不写追踪内容。
- 相对路径相对 FlexKV 仓库根目录，不跟启动 cwd。建议生产配置继续用绝对路径。
- `output_file` 支持 `$VAR` / `${VAR}` 和 `~`；引用的环境变量必须已设置且非空，否则初始化报错。在一些特殊机型和场景下示例使用 `${SGLANG_HICACHE_DIAG_DIR}/put-trace.jsonl`，启动前需导出本次诊断目录。
- `rotate_on_start: true`（在一些特殊机型和场景下示例已开）：每次重新启动 SGLang 写 `put-trace-<mode>-<MMDDHHMMSS>.jsonl`，不接到上一次文件后面。父进程传递解析后的绝对路径，各 rank / worker 追加同一文件，不重复展开变量或轮转。
- 配置初始化时检查文件能否打开；路径无效/不可写会立即报错。
- 多进程完整记录通过短时文件锁保护，防止 JSON 行交错；每条写入不做 fsync。
- 运行中磁盘写入失败会在 stderr 提示一次并停止该进程的文件写入，不让诊断错误使传输失败。
- 每台 Prefill 写自己的本地文件；多机分析时收集各节点文件。

输出是纯 **JSONL**：每行一个 JSON 对象，没有 `[FLEXKV]` 等日志前缀。
JSON 有 schema `v=1`，包含 `host/run/pid/ns/wall_ns`。
`ns` 使用 `perf_counter_ns`，用于**同一主机**进程间时间相减；`wall_ns` 是同一条记录
附近校准的 Unix 纳秒时间，供人定位日志。不要跨主机相减 monotonic 时间，或者把
未校时主机的 wall time 当成精确延迟。run ID 随配置传给子进程，用于区分不同启动。
文件采用追加模式，多次运行的数据可共存；也可以为每次实验设置不同文件名。

直接读取追踪文件生成可读报告：

```bash
python3 scripts/report_put_trace.py /tmp/flexkv/put-trace.jsonl > put-report.txt
python3 scripts/report_put_trace.py /tmp/flexkv/put-trace.jsonl --request '某个请求 rid'
```

脚本只依赖 Python 标准库，不需要 GPU、torch 或安装 FlexKV。支持多个输入文件，
并兼容此前 `[PUT-TRACE] {JSON}` 的 stdout 日志。丢失、截断或未结束的记录不会被
补成成功记录，报告显示 `unknown` / `missing_or_still_running`。

## 阶段定义

| 记录/字段 | 定义 |
|---|---|
| `store_created` | connector 进入 store_kv 的时间；取得 task ID 后补记原始时间，带 rid |
| `request_created` | KVTaskManager 开始处理 PUT/GET/PREFETCH，位于缓存匹配/规划之前 |
| `planned` | 缓存匹配、分配、构图完成；携带 op 前驱关系和原始创建时间 |
| `graph_launch` | slot mapping 已就绪，准备向 transfer manager 提交 |
| `scheduler_enqueue` / `scheduler_admit` | 引擎 graph 队列入队 / scheduler 取到 graph |
| `op_ready` / `op_dispatch` | 前驱满足、取到 ready op / 分派 worker，携带原始 op ID |
| `worker_submit` | worker pipe send 前；携带 queue/ticket/parent_op/request/pool |
| `worker_done.received_ns` | worker 从 pipe 收到描述符 |
| `batch_prepare_begin/end` | worker 已接收批次的合并/描述符准备，共享区间 |
| `worker_start` | 开始 launch_transfer；它与提交之差包含 pipe、排队和批次准备 |
| `d2h_prepare_begin` / `d2h_backend_begin` | Python D2H 准备，包括必要的 owner fence、block ID 获取 |
| `producer_fence_begin/end` | GDR owner producer fence，嵌套在 D2H 准备中；复用 fence 时没有该段 |
| `d2h_backend_begin` / `d2h_host_ready` | 后端调用到同步返回后 host 数据就绪的观察时间 |
| `op_completed` | 引擎收齐该 op 所有 TP/PP worker 副本后确认完成 |
| `gpu_release_eligible` | task manager 收到所有 task_end 所需操作的成功完成，具备提前释放 GPU 源块的条件 |
| `gpu_unlock_notified` | connector 轮询完成且跨 rank 通知后，准备把 rid 交回 SGLang 解锁；不是实际 dec_lock_ref 或显存释放时间 |
| `mooncake_prepare_begin/end` | 构造 key / host 指针 / 长度 |
| `mooncake_exists_begin/end` | Store 查存在 RPC；后面还会做过滤 |
| `mooncake_write_begin/end` | 实际 batch_put_from 调用的前后；全部命中时没有这段 |
| `mooncake_read_begin/end` | GET prefetch 的实际 batch_get 调用前后 |
| `mooncake_results_end` | 结果映射完毕；全部命中时由此标识跳过写入 |
| `worker_end` | launch_transfer 返回或抛出错误后；状态为 ok/failed |
| `task_end` | 图级回调和状态处理后的正常/失败/取消终态；原有 task 状态为准 |

**后端执行时间不是纯 PCIe/RDMA DMA 时间。** CE 的 gather/scatter、GPU 转换以及
GDR 的 region/WQE 准备可能与搬运重叠；这些都在后端执行段里。不增加 CUDA event
或同步来强行拆串行阶段。若需要更深的解释，配合 GDR diagnostics 或设备 profiler。
同理，Mooncake SDK 内部分配、网络排队和提交属于 write 段，这里不声称能观察其内部队列。

host_ready 是现有同步调用成功返回后、性能日志输出前的 CPU 观察点，包含少量
Python 返回/计算开销。未完成或失败的阶段没有伪造 end；worker_end 仍会记录失败。

## 数据量与带宽口径

`worker_done` 增加 `logical_blocks`（当前逻辑 op 的 block 数）及 `volume`。
报告脚本也显示这些字段。例如，一次未压缩 D2H 的记录片段：

```json
{"logical_blocks":64,"volume":{"payload_bytes":268435456,"blocks":64,"bytes_scope":"worker_payload"}}
```

- `volume.payload_bytes`：复用 worker 已有的传输大小计算，单位 byte；D2H 为该
  worker 操作的有效 KV payload，TP-group 路径为整组口径，不再乘 TP 数。
  包含该 worker 负责的 KV region；不是 PCIe/NIC 线路上的实际流量，不含协议头、
  重试、临时缓冲区额外拷贝或后端重复读写。可用于计算**有效 payload 带宽**。
- `volume.blocks`：上述 payload 对应的 block 数。GDR 合批时为合并执行的 block 数，
  `logical_blocks` 仍是各原始逻辑 op 的数量。一个 KV block 包含其对应的层/KV 数据，
  不因包含多个 region 或多个 TP rank 再乘 block 数。
- `volume.uncompressed_bytes`：压缩后端提供时额外记录压缩前大小。
- Mooncake `payload_bytes` 为计划处理的全部 block payload；额外记录
  `mooncake_write_bytes` / `mooncake_write_blocks`，表示查存在后实际提交给
  `batch_put_from` 的数据量，以及 `mooncake_skipped_blocks`（已存在、跳过写入）。
  全部已存在时 write bytes / blocks 均为 0，没有 write 时间段。提交量不等于
  失败时已成功写入的量；失败/未完成记录不能用于成功搬运带宽统计。
- 数据量未知时字段缺失，不填 0。没有新增 GPU 查询、tensor 拷贝或按 block 遍历；
  Mooncake 复用原有过滤循环的计数，随 worker_done 一起写出。

GDR 合批的多个 `worker_done` 共享 `physical`、`volume` 和执行时间，必须按
`(host, run, physical)` 去重后汇总。Full/SWA 等独立 worker 的字节可分别统计；
D2H 和 H2REMOTE 是不同传输阶段，不能相加后当作一次 D2H 的字节数。
不直接记录 token 数：不同 pool/region 的 block 粒度和尾块有效 token 数可能不同，
block 数乘模型 block size 不一定等于有效业务 token 数。

单次有效带宽（十进制 GB/s；时间戳单位 ns）：

```text
D2H GB/s = payload_bytes / (d2h_host_ready - d2h_backend_begin)
Mooncake GB/s = mooncake_write_bytes / (mooncake_write_end - mooncake_write_begin)
```

仅对成功且分母大于 0 的完整记录计算。D2H 分母包括后端准备和等待，并非纯 DMA
时间；Mooncake 分母是 SDK 写入调用时间。多次执行的平均带宽应使用
`sum(bytes) / sum(duration_ns)`，不要直接平均每次 GB/s；若要统计并行 worker 的
整体吞吐，则使用测量窗口的墙钟跨度作为分母，不能累加重叠区间当作墙钟耗时。

## 哪里会排队

1. `planned → graph_launch`：等待 slot mapping/上层 launch，也含调用方处理，不能都算 scheduler 排队。
2. `graph_launch → scheduler_enqueue`：transfer manager IPC/处理；多节点只比较同一 host 上的事件。
3. `scheduler_enqueue → scheduler_admit`：引擎 graph 队列。
4. `scheduler_admit → op_ready`：依赖等待及 scheduler 处理，例如 H2REMOTE 等 D2H；不是一个独立网络写队列。
5. `op_ready → op_dispatch`：scheduler 分派阶段。
6. `worker_submit → worker_start`：pipe/worker 队列及已接收批次准备；`received_ns` 进一步拆分 pipe 等待与接收后等待。
7. GDR 专用执行线程、QP credit、CQ，以及 Mooncake SDK/服务端内部也可能等待；外层时间戳不能反推出 WQE 或服务端队列深度。
8. `d2h_host_ready → gpu_release_eligible → gpu_unlock_notified`：完成 IPC、跨 rank 汇合、上层轮询/通知延迟。

同一个 Mooncake worker 的 REMOTE2H 可以阻塞 H2REMOTE，因此开启后采集所有方向
worker 操作的提交/执行区间；只输出 PUT 会漏掉这些排队来源。

## SGLang 完成回收诊断：事件驱动版

目标是定位 `D2H 数据就绪 → 完成消息消费 → GPU 源节点解锁` 的延迟。
全部由原来的 `put_trace.enabled/output_file` 管理，无额外开关。

已移除 scheduler、task API、cache drain 的逐调用 begin/end，也没有按秒心跳或
定时汇总。无任务时不读诊断时钟、不写诊断日志。重复轮询只更新内存统计，
不改变原来的轮询频率、collective、线程或 GPU 同步行为。

每个 PUT 的关联链路如下：

1. 原有 `store_created/request_created`、D2H worker points、`op_completed`。
2. `store_poll_first`：每个 task/rank 首次被 connector 轮询时只写一次。
3. `completion_consumed`：task manager 已取得并合并完成消息，准备执行回调。
4. 原有 `gpu_release_eligible/task_end/gpu_unlock_notified`。
5. `store_poll_complete`：该 task 离开 inflight 时写一次，附带 poll_count、error_count、
   首次/最后轮询时间、最大轮询起点间隔、最长 poll 的起止时间，以及 try_wait / 跨 rank
   sync 各自最长的一次调用。一次 poll 可处理多个 task，共享耗时不能重复相加。
6. `gpu_unlock_begin/done`：SGLang 真正执行 dec_lock_ref 前后；不等同于物理释放显存。

异常时每个 task/rank 最多额外写一次 `store_poll_error`，后续异常只累加 error_count。
没有匹配源节点时保留 `gpu_unlock_untracked`。日志量随任务/传输事件数增长，
不随空闲时间、scheduler tick 或 pending poll 次数增长；不是整个文件固定大小。
内存统计随 inflight task 完成清除，取消/reset 移除的 task 会在后续轮询时清除统计。

先比较 `op_completed → store_poll_first`，判断是否迟迟没轮询；再比较
`completion_consumed → gpu_release_eligible → gpu_unlock_notified → gpu_unlock_done`。
利用完成时附带的最大间隔和 try_wait/sync 最慢调用，区分轮询稀疏与调用阻塞。
不会记录每次 scheduler 内部函数，因此不能仅靠这份 trace 指认某个 forward 内部算子。
任务未完成、进程被杀或某次调用永久卡住时，可能只有 first 而无 complete，
内存中的后续统计不会写出；不能将缺少 complete 解释为成功或精确死锁位置。

两边源码都需要部署：`special-platform-FlexKV/FlexKV` 与 `sglang-special-platform`。
SGLang 的 scheduler.py 已撤掉旧 tracing 装饰器；flexkv_put_trace.py 只保留无操作
兼容入口，旧 scheduler 若仍引用它也不会再刷日志。FlexKV 若使用编译后的 .so，
必须按原流程重新编译后重启，单独更新 .py 不会替换已有编译产物。

在 FlexKV 根目录生成报告：

```bash
python3 scripts/report_put_trace.py /tmp/flexkv/put-trace.jsonl > /tmp/flexkv/request-report.txt
python3 scripts/report_completion_trace.py /tmp/flexkv/put-trace.jsonl > /tmp/flexkv/completion-report.txt
```

请保存完整原始 JSONL；第二个脚本也兼容旧版 span 日志。
CPU 回归测试包含：pending task 连续轮询 10000 次，只产生一条 first 事件；
第 10001 次完成后才输出统计，原有 collective 每次仍正常执行。

```bash
python3 -B tests/test_put_trace_standalone.py
SGLANG_SOURCE_ROOT=/path/to/sglang-special-platform python3 -B tests/test_completion_trace_standalone.py
```

## 后处理的计数口径

报告按 `request` 汇总。`ticket` 是一个逻辑 worker 操作，`queue` 是特定引擎进程的
特定 worker；不能把所有 worker 的积压加起来当成一个串行队列。

- `Q/R/U`：在本次 worker_submit 时，同队列此前已提交的其他逻辑 op 中，等待执行、正在执行、缺少完整执行区间的数量。
- `prior_request_groups`：以上 op 对应的不同 graph 请求组数量。新记录中的
  GET/PREFETCH 也有 request 标签；旧记录缺少标签时按 ticket 计数。已合并图计为一组，
  不能当成精确业务请求数。
- `other_active`：到达某执行阶段时，同 host/run/direction 中其他物理执行仍在该阶段的数量（跨 worker）；GDR 合批按 physical 去重。
- `unknown`：该范围内先前提交但尚无完整日志的物理操作数，无法判断其实际阶段。
- 区间右端点排除：结束时间等于到达时间不再计为积压；相同到达时间不视为先前请求。
- 计数是到达时的现存量，不代表所有这些操作都一定排在当前操作前面（尤其有合批时）。

worker 阶段只在操作结束后合并写入一条 JSON，运行时不计算这些计数。worker 队列
计数采用离线时间线扫描；阶段计数采用排序端点查找。

## 多 TP 的记录粒度

追踪整组请求的控制链路，不为每个 TP rank 重复输出一套生命周期：

- `store_created` / `gpu_unlock_notified` 只由对应同步域的 sync leader 输出。
- D2H 在 Python TP-group worker 外层打点；同一 WorkerKey 的本机 TP GPU 由一个
  `tpGPUCPUTransferWorker` 管理，原生后端内部再分发到各 rank。单节点 TP=8、单一
  worker 分支的一次 D2H op 只有一份 worker 阶段记录，不会因 TP=8 变成八份。
- 当前 RegionBatch 路径使用 `submit(..., True)`，在 GDR fence 和 stream 等待后返回。
  `d2h_host_ready` 反映该 worker 所负责各 rank 都完成后的观察时间；CE / GDR 均如此。
- 引擎 `op_completed` 再收齐该 op 所有 worker 副本，随后由任务管理器判断
  `gpu_release_eligible`，connector 记录完成通知。保留这些汇合点可以观察整条关键路径。

因此不需要另设 `tp_rank: 0` 过滤。只记录某个 rank 自己的 DMA 完成会漏掉其他 rank
更慢的部分，不能代替整组 D2H 完成。现有追踪不增加 GPU 同步，但也无法指出哪个
rank 最慢；需要定位这种不均衡时，再启用原生 GDR diagnostics 或 profiler。
GDR `summary/timing/probe` 是另一套设备级诊断，可能按 GPU 输出，不受本追踪的粒度约束。

跨节点 TP、PP 或 Full/SWA 存在多个独立 worker 分支时，保留各自的 queue/ticket 和
阶段记录。这些分支的执行、排队不同，不是重复的 TP 生命周期；不能只取一支推算
所有队列。每台机器写本地文件，分析时一起传给报告脚本，跨主机时钟限制仍然适用。

## 合批、跨 rank 和失败

- Full/SWA 用 `pool` 区分；跨 worker 的副本保留 request 和 parent_op，并有各自 ticket/queue，
  不是 TP-group worker 内部每个 GPU rank 各一份。
- GDR 合并多个 op 时，各 op 仍有单独 worker_done，但共享 physical ID 和执行时间。
  **不能相加这些执行时长或把它们当作多个并行物理 DMA。**
- KVTask 层合批时，新的 planned 携带 children request 列表；原请求报告指向共享批次。
- `store_created/gpu_unlock_notified` 通过 run+task 映射；若多个 task manager 造成 ID 歧义，
  报告明确显示 unlinked，而不猜测请求归属。内核传输关联本身使用独立 request key，不依赖该映射。
- 传输失败仍输出 worker_done(status=failed)。如果进程强制退出，可能只有 submit，没有 done；
  这类操作保留 unknown，不能解释成零耗时、仍然排队或成功。
- 仅在成功 task_end 分支输出 gpu_release_eligible；失败不意味着已安全满足提前解锁条件。

## 开销和验证范围

默认关闭。关闭时不读取阶段时钟、不附加 graph/op 诊断元数据、不输出诊断日志；
只剩入口的布尔分支/函数调用。开启后每个 worker 操作收集少量整数时间戳，结束后序列化
一次；没有新进程、轮询线程、共享计数锁、GPU 同步或按 block 打点。
完整文件记录写入时使用线程锁和进程间文件锁；高层生命周期事件另有写入，
因此文件 I/O 开销仍会影响测试，不能宣称零开销。

CPU 测试：

```bash
python3 -B tests/test_put_trace_standalone.py
python3 -B tests/gdr/test_backend_coexistence.py
```

测试覆盖 disabled、请求/副本关联、合批共享区间、失败完成上报、实际 batch_put 钩子、
全部 key 命中、缺失/重复日志、计数边界，以及索引算法与参考算法的一致性。
本地验证不代表已经完成 在一些特殊机型和场景下/CE/GDR/Mooncake 硬件性能及端到端验证。

## 三个 batch benchmark 脚本

在 FlexKV 根目录运行：

```bash
PUT_TRACE=1 bash batch_run_ce_kernel_tp8.sh
PUT_TRACE=1 bash batch_run_dsv4_transfer_modes.sh
bash batch_run_put_get_e2e.sh
```

默认 `PUT_TRACE=0`；可用 `PUT_TRACE=1 DRY_RUN=1` 检查命令和路径。每个 arm / CPU
碎片 / GPU 碎片测试目录直接追加写入 `put-trace.jsonl`，例如
`<最终 OUT>/CE_pagesegnone/put-trace.jsonl`；最终 OUT 仍沿用脚本原有后缀规则。
直接运行 Python 时可用 `--put-trace-file /absolute/path/trace.jsonl` 指定文件名。
IPC probe 不输出此追踪。此文件由 Python 直接写入，不依赖脚本现有的控制台日志 tee。

这两个脚本绕过生产 worker / scheduler，直接调用 RegionBatch 后端；因此输出
`event=benchmark_transfer`，不伪造请求创建、调度排队、GPU 解锁或 Mooncake 阶段。
生产 YAML 的 `put_trace` 不控制这两个脚本。每次物理提交记录一次，覆盖整组 TP，
合批后不会为每个逻辑请求或每张 GPU 重复写。记录包括：

- `arm/direction/profile/layout/tp`、碎片参数、`blocks_per_request/concurrency`。
- `round`（从 0 起）和 `warmup`；预热也记录，正式带宽分析过滤 `warmup=false`。
- `prepare_begin_ns/prepare_end_ns`：整轮操作及请求描述符准备的共享时间段，不是每个
  op 独立准备耗时；同轮多次提交不能累加此区间。
- `transfer_begin_ns/transfer_end_ns`：本次物理提交到原生同步完成的 CPU 观察时间；
  H2D 包含原生逐层完成等待，D2H 使用 `submit(..., True)`。
- `volume.blocks/payload_bytes/logical_payload_bytes`：采用 benchmark 原有带宽字节口径；
  shared D2H 写一份，H2D 广播按 TP 数累计，非 shared 按 TP 数累计。不含协议开销。
- `status`：原生提交/完成调用成功或失败。字节校验结果仍以 benchmark 最终结果为准。

JSON 序列化和文件写入在该轮计时结束后执行，计时区间内仅增加时钟读取和内存记录；
开启追踪仍可能影响后续轮次的缓存/调度，不能宣称没有测量扰动。
`report_put_trace.py` 当前面向生产请求链路，不解析 `benchmark_transfer`；benchmark
原有 `raw.jsonl/summary.csv` 仍提供整体带宽，新增逐次记录供后续脚本分析。

第三个脚本 `batch_run_put_get_e2e.sh` 不走 RegionBatch 微基准捷径，而是启动真实
`KVManager`、TP GPU 注册进程、KVTask scheduler 和 transfer workers。每轮在计时前
生成独立 token 请求并把可校验数据写入 GPU，然后执行：

1. PUT request/planning → scheduler/worker queue → D2H → 可选 Mooncake write → task end；
2. 清零 GPU；Mooncake 模式还会清空本地 CPU tier，再执行 REMOTE2H prefetch；
3. GET request/planning → scheduler/worker queue → H2D → task end，并逐字节值校验 GPU。

Host-only 示例：

```bash
BLOCKS=4,16 CONCURRENCY=1,4 ROUNDS=10 bash batch_run_put_get_e2e.sh
```

Mooncake 示例：

```bash
STORAGE_MODES='host mooncake' \
MOONCAKE_CONFIG=/absolute/path/mooncake-store.json \
bash batch_run_put_get_e2e.sh
```

每个 `<storage>_<arm>` cell 输出 `raw.jsonl`、`summary.csv`、`summary.json`、
`request-trace.jsonl` 和 `request-report.txt`。`raw.jsonl` 的预热记录带
`warmup=true`，汇总只使用正式轮次。`request-trace.jsonl` 保留预热请求，便于排查
首次初始化；完整分段报告用 `request-report.txt`。
