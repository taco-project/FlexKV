# FlexKV stress benchmark 设计

## 目标与边界

`flexkv_stress` 用合成 KV tensor 持续驱动 FlexKV 的 prefix match、GPU/CPU/SSD
load/store、layerwise、SWA、并发和异步接口，重点验证长时间运行中的正确性、稳定性和
有效载荷吞吐。它不模拟模型计算，也不把 token 生成时间计入 FlexKV 传输性能。

CPU stub 用于无加速卡环境下的控制流回归。它会真实分配 CPU PyTorch tensor，并验证
TP/DP 路由、prefix、PUT/GET、readback 和 byte pattern，但不覆盖 CUDA/ROCm IPC、原生
kernel、eventfd 或 SSD，因此不能替代硬件压力测试。

## 执行链路

每轮先按 seed 生成多轮会话。每个 turn 依次执行：

1. `get_match`：查询已经缓存的 block prefix，并与本地 `PrefixOracle` 对照。
2. `launch_get`：把命中 block 加载到模拟推理引擎的 GPU pool。
3. 写入确定性的 block pattern，执行 `put_match`，只提交未命中的完整 block。
4. `launch_put`：把新 KV 写入 FlexKV。
5. `launch_readback`：立即重新加载刚写入的序列；抽样请求会先清零目标，再逐 byte 验证。
6. 按配置周期执行 async API 或强制 SSD reload probe。

多个 DP rank 由独立 `ManagerDriver` 驱动；会话按 `conversation_id % dp_size` 路由。TP 和
CP worker 共享同一个 DP manager，并通过各自的设备注册信息处理对应 slice。

加速卡模式下，每个物理 rank 由一个独立 OS process 驱动且只绑定一张 `cuda:N` 卡。
CPU stub 为了轻量和可调试性使用进程内 virtual worker；它验证 worker 数、rank 路由和
KV 内容，但不验证多进程/CUDA IPC 隔离。

## TP/CP 拓扑

`model.tp_size` 与 SGLang 的 composite TP world size 对齐，表示每个 DP shard 的物理
worker 数。q-split CP 从这个 world 内划分 rank，不额外相乘：

```text
kv_tp_size     = tp_size / cp_size
workers_per_dp = tp_size = kv_tp_size × cp_size
required_gpus  = dp_size × workers_per_dp
```

因此 TP=4、CP=2 时只启动四个 worker，按 CP-major 顺序映射为
`(tp0,cp0), (tp1,cp0), (tp0,cp1), (tp1,cp1)`。同一个 KV-TP rank 在不同 q-split
CP rank 上持有相同的全量 KV slice；CPU/SSD cache 只保存唯一 KV-TP slice，不因 CP
副本数扩容。`tp_size` 必须能被 `cp_size` 整除。这与 SGLang adapter 中
`attn_tp_size = sglang_tp_size / (attn_dp_size × attn_cp_size)` 的处理一致；stress 配置的
`tp_size` 是每个 DP shard 的 composite world，因此 DP 仍作为外层独立副本相乘。

## 模型与物理布局

模型 preset 只描述 KV 几何，不加载权重。每个 layer group 独立声明 layer、head size、
dtype 和压缩比。主 KV host page 的字节数为所有 group 的物理 footprint 之和，并包含
全部 TP slice：

```text
main_page_bytes = KV_TP × Σ(layers × kv_dim × page_tokens/compress_ratio
                            × kv_heads × head_size × dtype_bytes)
```

DSv4 还带独立的 SWA pool。一个 SWA slot 是一个完整 page：

```text
swa_page_bytes = page_tokens × swa_layers × swa_head_size   # uint8
```

SWA 的 launch 参数是 token-index slot mapping，而不是 page id。benchmark 为每个 SWA
page 传入连续的 `page_tokens` 个 slot；FlexKV 再将其折叠为 SWA slot id。这个约定必须与
普通 KV slot mapping 保持一致，否则 page 1 以后会错误地落到 page 0。

## 正确性判定

一轮成功需要同时满足：

- 实际 prefix hit 与 oracle 一致（允许按配置设置 block 容差）；
- 所有 GET/PUT/readback transfer 返回 `SUCCESS`；
- 被抽样的 byte pattern 完全一致；
- async/SSD probe（启用时）成功。

运行**永不因为校验失败而提前中断**。byte mismatch、hit 偏差、失败的 transfer 都会把
对应请求计为失败、写入按需创建的 `errors.csv`，并拉低所在窗口的
`byte_validation_accuracy` / `request_success_rate`，但测试会继续跑完整个 workload，
以便一次性暴露所有问题，而不是停在第一个错误上。校验结果只影响进程**最终退出码**：任一
场景的 `request_success_rate` 低于 `minimum_success_rate`、出现任何 byte mismatch、或发生
运行时异常，退出码即为 `1`，否则为 `0`。warmup 轮的失败同样只记录并计入退出码，不中断
后续测量场景。

**GPU 池尺寸与批量回读**：`model.gpu_blocks_per_rank` 是模拟推理引擎的 GPU KV 池块数。
批量回读校验会把一个 batch 内所有在途 turn 的数据载入一段**连续**的 GPU slot；当这段长度
超过池容量时会绕回，导致同 batch 内不同 turn 的 slot 相互覆盖，被抽样的 turn 便会读到别的
turn 的字节，**报出假的 byte-validation 失败**（仅在 `batch_size>1` 且命中抽样时发生）。因此
`gpu_blocks_per_rank` 必须 ≥ 一个 batch 的最坏在途块数，约
`batch_size × (system_prompt_blocks + turns_max × (max_input_blocks + max_output_blocks))`。
配置低于该值时启动会打 WARNING。空载（`batch_size=1`）不受影响，所以「unloaded 全绿、loaded
零星 byte 失败」这一模式通常是 GPU 池偏小，而非 FlexKV 传输 bug。

## 双模式与计时口径

`latency_hit` 模式会**依次**跑两个 profile，两者共用同一套 workload、各自跑满
`duration_seconds`（或 `rounds`）。因此一次 `latency_hit` 的墙上时间约为
`2 × duration_seconds`：

- **`unloaded`（空载）**：强制 `batch_size=1`、`max_inflight_per_dp=1`，任一时刻只有一个
  请求在途。测的是**单请求最优延迟**——没有排队、没有争用时 match/load/save 各自要多久，
  是延迟的下界基线。
- **`loaded`（满载）**：使用配置里的 `concurrency.batch_size` 与 `max_inflight_per_dp`
  并发打流。测的是**并发压力下的延迟与吞吐**——排队、锁争用、传输带宽饱和都会体现在这里，
  更接近线上高负载表现。

把两个场景对照，就能看出并发把延迟抬高了多少、把吞吐（QPS）提升了多少。

计时口径：match 是 `get_match`；load 是 `launch_get + wait(completely=True)`；save 从
`put_match` 开始，到 `launch_put + wait` 结束。readback 只判定正确性，不进入这三段时延。

命中和精度公式为：

```text
hit_ratio          = actual_hit_tokens / query_tokens
hit_exact_rate     = 在容差内的请求数 / 请求数
hit_token_accuracy = 1 - Σ|actual-expected| / Σquery_tokens
byte_accuracy      = passed_samples / validation_samples
```

`bandwidth` 独立报告四条 path：`gpu_to_cpu_save`、`cpu_to_gpu_load`、
`gpu_to_ssd_save_e2e` 和 `ssd_to_gpu_reload_e2e`。SSD 两项是端到端指标；现有接口不提供
可靠的纯 CPU→SSD 分段，因此不从总时间中反推。每个并发档同时达到最小时长、最少
operation 和可选目标 payload 后停止；match、workload 生成和 readback 不进入传输计时。

## 带宽与容量口径

`transfer_bytes` 是 FlexKV launch 的逻辑有效载荷，不是 PCIe/ROCm profiler 读到的总线
transaction bytes。主 KV 按去重后的物理 page 数乘 `main_page_bytes`；SWA 按 SWA page
数乘 `swa_page_bytes`。因此该口径包含压缩后的 group 和 SWA 数据，但不包含协议头、对齐
padding、重试或设备驱动开销。

对于 q-split CP，该口径统计 cache 中的唯一 payload，不重复计算广播到多个 CP GPU 的
副本；若需要 PCIe/ROCm aggregate traffic，必须以硬件 profiler 为准。

所有容量使用十进制 GB（`bytes / 1e9`），带宽使用十进制 GB/s（不是 Gb/s），时延使用
ms，比例使用 `[0,1]`。DP/CP 广播副本不重复计入逻辑 cache payload；硬件 aggregate
traffic 仍应由系统 profiler 测量。

## 输出设计

每次运行固定生成四个文件：

- `summary.csv`：给人查看的最终结论，按 mode 使用不同 schema；
- `metrics.csv`：每个统计窗口的同名指标和资源趋势；
- `summary.json`：schema `1.0` 的正式机器接口；
- `effective_config.yaml`：完全展开、可复现的运行配置；

只有发生错误时才创建 `errors.csv`，不生成 turn/operation/resource 中间 CSV。CPU stub
也生成完整结果，但 `performance_valid=false`。

结果目录使用 `YYYYmmdd_HHMMSS_PID`，避免同一秒启动的任务相互覆盖。
Reporter 使用固定大小、可合并的对数 histogram 汇总 p50/p95/p99，不在内存中保留长跑
observation，内存占用不随运行时长增长。CSV 与 JSON 从同一语义汇总对象生成。

## 指标字段说明

`summary.csv` 是每个场景跑完后的最终汇总，`metrics.csv` 是同名字段的逐窗口趋势（多出
`window_id` / `window_started_at` / `window_duration_s` 三个前缀列），`summary.json` 是把
同一批字段按语义分组后的机器接口。字段随 `mode` 不同：`latency_hit` 与 `bandwidth` 各有
一套。下面按分组解释。

### 通用标识与拓扑（两种 mode 共有）

| 字段 | 含义 |
| --- | --- |
| `run_id` | 运行 ID（`时间戳_PID`），也是结果目录名 |
| `mode` | `latency_hit` 或 `bandwidth` |
| `scenario` | 场景名。`latency_hit` 下为 `unloaded`/`loaded`；`bandwidth` 下为 path 名（`gpu_to_cpu_save` 等） |
| `backend` | `cuda` / `rocm` / `cpu_stub` |
| `performance_valid` | 性能数字是否可信；CPU stub 下为 `false`（SVG 顶部会加醒目水印） |
| `model` / `architecture` | preset 名与架构标识 |
| `composite_tp` | SGLang composite TP world size（每个 DP shard 的物理 worker 数） |
| `kv_tp` | KV/attention TP 宽度 = `composite_tp / cp` |
| `cp` / `dp` | q-split CP 宽度 / DP 副本数 |
| `gpu_count` | 实际使用的 GPU 数 = `dp × workers_per_dp` |
| `page_tokens` | 每个 block 的 token 数（`tokens_per_block`） |
| `main_page_gb` | 主 KV 单个 host page 的字节数（十进制 GB，含全部 KV-TP slice） |
| `cpu_cache_gb` / `ssd_cache_gb` | CPU / SSD cache 容量 |
| `layerwise` / `ssd_enabled` / `swa` | 是否启用 fused layerwise 传输 / SSD tier / SWA pool |

### workload 参数（回显配置，让结果自解释）

`conversations_per_round`、`turns_min`、`turns_max`、`system_prompt_blocks`、
`first_input_blocks`、`added_input_blocks`、`output_blocks`、`partial_block_tokens`、
`shared_system_prompt`、`read_after_put` 均直接回显对应的 `conversation.*` 配置。此外每行还带：

| 字段 | 含义 |
| --- | --- |
| `batch_size` | 该场景每批提交的请求数 |
| `concurrency` | 该场景每 DP 的最大在途请求数（`max_inflight_per_dp`） |

### `latency_hit` 专有：延迟 / 命中 / 正确性 / 资源

| 字段 | 含义 |
| --- | --- |
| `requests` | 该场景累计请求（turn）数 |
| `qps` | `requests / 累计计时秒数` |
| `match_p50/p95/p99_ms` | `get_match`（prefix 查询）延迟分位 |
| `load_p50/p95/p99_ms` | `launch_get + wait`（H2D 读取命中 block）延迟分位 |
| `save_p50/p95/p99_ms` | `put_match → launch_put + wait`（D2H 写入新 KV）延迟分位 |
| `hit_ratio` | `actual_hit_tokens / query_tokens`，实际 prefix 命中占查询 token 的比例 |
| `hit_exact_rate` | 命中量落在容差内的请求占比（命中 block 数是否与 oracle 精确一致） |
| `hit_token_accuracy` | `1 - Σ\|actual-expected\| / Σquery_tokens`，命中 token 数的整体准确度 |
| `request_success_rate` | success 请求 / 总请求（match+get+put+byte 校验全过才算 success） |
| `byte_validation_accuracy` | 被抽样请求中 byte pattern 完全一致的比例；`1.0` = 传输零损坏 |
| `gpu_memory_gb` | 采样到的 GPU 显存占用峰值（所有 worker 之和） |
| `rss_gb` | 主机 RSS（加速卡模式为各 worker 进程之和） |
| `ssd_used_gb` | 本 run SSD cache 的实际磁盘占用。每个 run 独占 `<ssd_cache_dir>/<run_id>/` 子目录，故只统计本 run 自己的 .bin（不含同目录下历史 run 残留）；按实际分配块（`st_blocks×512`）计，而非稀疏文件的 apparent size，避免虚高。run 结束（含异常/中断）在 `finally` 里删除该子目录，不再攒盘。 |

命中与精度公式见上一节「双模式与计时口径」。

### `bandwidth` 专有：吞吐 / 延迟 / 正确性 / 资源

标识、拓扑、workload、`batch_size`/`concurrency`、资源字段与上面一致，差异部分：

| 字段 | 含义 |
| --- | --- |
| `payload_gb` | 该 path×并发档累计的逻辑有效载荷（十进制 GB，去重后的物理 page） |
| `operations` | 计入该 path 的传输 operation 数 |
| `duration_s` | 该 path 累计的**活跃传输时间**（各 operation latency 之和，**非**墙上时间） |
| `throughput_gb_s` | `payload_gb / duration_s`（十进制 GB/s） |
| `latency_p50/p95/p99_ms` | 单次传输 operation 的延迟分位 |
| `operation_success_rate` | 成功 operation / 总 operation |

四条 path：`gpu_to_cpu_save`（D2H 存）、`cpu_to_gpu_load`（H2D 载）、
`gpu_to_ssd_save_e2e`、`ssd_to_gpu_reload_e2e`（后两者为端到端口径）。

### `summary.json`（schema `1.0`）分组

同一批字段按语义嵌套为：`run`（run_id/mode/backend/started_at/duration_s）、`model`、
`topology`、`cache`、`workload`，以及 `scenarios[]`。每个 scenario 内把延迟收进
`latency_ms.{match,load,save}.{p50,p95,p99}`，命中收进 `hit.{ratio,exact_rate,token_accuracy}`，
正确性收进 `correctness.{request_success_rate,byte_validation_accuracy,validation_samples}`，
资源收进 `resources_gb.{gpu_memory,rss,ssd_used}`。

### `errors.csv`（仅出错时创建）

列为 `time` / `scenario` / `window_id` / `operation` / `error`。既然运行不再提前中断，出现问题时
这里会累积**所有**失败记录——排查时先看它，再对照 `metrics.csv` 里哪个窗口的
`byte_validation_accuracy` / `request_success_rate` 掉了。

## DSv4 验证层级

1. `--dry-run`：校验 preset、压缩比、layer group、page 大小、容量和 GPU 数量。
2. `dsv4_cpu_smoke.yaml`：在无 GPU 主机上跑通多 group + SWA 的完整控制流。
3. `glm5_cpu_smoke.yaml`：验证 composite TP=4、KV TP=2、CP=2 的重叠 rank 和异步接口。
4. `dsv4_pro.yaml` / `dsv4_flash.yaml`：在目标 CUDA/ROCm 节点验证 IPC、layerwise、
   eventfd、SSD 和真实传输带宽。

硬件运行前应确认 FlexKV extension 与当前 CUDA/ROCm 匹配、SSD 目录容量充足，并根据
机器显存调整 `gpu_blocks_per_rank` 与 `swa_num_slots`。
