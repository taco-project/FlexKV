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

`minimum_success_rate` 决定进程最终退出码，`stop_on_mismatch` 决定 byte mismatch 后是否
立即停止。运行时异常和失败操作写入按需创建的 `errors.csv`。

## 双模式与计时口径

`latency_hit` 自动运行两种 profile：`unloaded` 固定 batch/concurrency 为 1，`loaded`
使用配置值。match 是 `get_match`；load 是 `launch_get + wait(completely=True)`；save 从
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

每次运行固定生成五个文件：

- `summary.csv`：给人查看的最终结论，按 mode 使用不同 schema；
- `metrics.csv`：每个统计窗口的同名指标和资源趋势；
- `summary.json`：schema `1.0` 的正式机器接口；
- `charts.svg`：标准库生成的单文件四面板 Dashboard；
- `effective_config.yaml`：完全展开、可复现的运行配置；

只有发生错误时才创建 `errors.csv`，不生成 turn/operation/resource 中间 CSV。CPU stub
也生成完整结果，但 `performance_valid=false`，SVG 顶部显示性能数字无效的醒目标记。

结果目录使用 `YYYYmmdd_HHMMSS_PID`，避免同一秒启动的任务相互覆盖。
Reporter 使用固定大小、可合并的对数 histogram 汇总 p50/p95/p99，不在内存中保留长跑
observation，内存占用不随运行时长增长。CSV 与 JSON 从同一语义汇总对象生成。

## DSv4 验证层级

1. `--dry-run`：校验 preset、压缩比、layer group、page 大小、容量和 GPU 数量。
2. `dsv4_cpu_smoke.yaml`：在无 GPU 主机上跑通多 group + SWA 的完整控制流。
3. `glm5_cpu_smoke.yaml`：验证 composite TP=4、KV TP=2、CP=2 的重叠 rank 和异步接口。
4. `dsv4_pro.yaml` / `dsv4_flash.yaml`：在目标 CUDA/ROCm 节点验证 IPC、layerwise、
   eventfd、SSD 和真实传输带宽。

硬件运行前应确认 FlexKV extension 与当前 CUDA/ROCm 匹配、SSD 目录容量充足，并根据
机器显存调整 `gpu_blocks_per_rank` 与 `swa_num_slots`。
