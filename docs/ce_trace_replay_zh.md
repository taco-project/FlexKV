# 传输 trace 与合成回放

[English](ce_trace_replay.md)

在启动 worker **之前**设置 `FLEXKV_CE_TRACE=1`，记录原生 CPU/GPU 传输的 launch。
默认关闭，开启后不会改变传输选择或完成语义。trace 覆盖直接调用、TP 线程组及
region batch，并在 rank 共享和 region 偏移解析之后记录。

```bash
FLEXKV_CE_TRACE=1 FLEXKV_CE_TRACE_FILE=/tmp/flexkv-transfer \
FLEXKV_CE_TRACE_MAX_BLOCKS=0 your-normal-launch-command
```

每个进程写入 `/tmp/flexkv-transfer.<pid>.jsonl`，父目录必须已存在。worker 退出时会
排空自己的异步 logger；直接调用原生 API 的程序应在传输完成后调用
`flexkv.c_ext.ce_trace_shutdown()`。关闭是幂等的，但对该进程是终态。logger 使用
独立线程池，不会关闭其他 spdlog logger。应在 spawn 出的 worker 中初始化 trace；
不支持 fork 已初始化的 logger。

每条记录表示一次 **launch**，不代表成功完成。schema 1 包含进程、线程和设备身份、
方向、后端（copy engine 或 SM kernel）、tensor 布局、完整层编号、字节 stride、
region/rank 偏移、block ID、CE 配置、分析和所选路径，不包含 KV 内容或指针；空
launch 不记录。应按常规日志策略管理这些工作负载几何和 block-ID 元数据。

队列容量为 4096 条，满时丢弃最旧记录，不阻塞传输。`dropped_before` 是构造记录时
观察到的累计丢弃数；`ce_trace_shutdown()` 返回最终丢弃数并向 stderr 报告。初始化
或写入失败会禁用 trace，不使传输失败。trace 有 CPU 和文件 I/O 开销，适合诊断。

`FLEXKV_CE_TRACE_MAX_BLOCKS` 默认每条记录最多 256 个 ID；设为 0 记录全部，或设置
1 至 1,000,000 的整数。截断记录可以检查但不能回放；缺失或丢弃的 launch 无法恢复。

## 无 CUDA 检查

```bash
python -m flexkv.transfer.ce_replay /tmp/flexkv-transfer.1234.jsonl
```

每行报告从零开始的记录编号、方向/后端/路径、字节数、截断和观察到的丢弃数。未知
schema 或错误记录会报告文件和行号。

## 在 CUDA 上回放一次 launch

```bash
python -m flexkv.transfer.ce_replay /tmp/flexkv-transfer.1234.jsonl \
  --replay 0 --device 0 --max-mib 256
# 与逐 block 拷贝基线对比：
python -m flexkv.transfer.ce_replay /tmp/flexkv-transfer.1234.jsonl \
  --replay 0 --device 0 --per-block
```

回放构造确定性的合成 buffer，保留 stride、偏移、block ID 和部分层范围，并将包括
未触及区域在内的 **所有 CPU 和 GPU 字节**与拷贝参考结果比较；不会读取原应用的
内存。原始指针的 tensor 所有者会保留到 CUDA 同步完成，即使原生调用抛错也是如此。

CUDA 初始化前先检查分配上限，预算涵盖 buffer、预期值、读回数据以及 ID/指针表。
负数或未对齐的几何、溢出、截断 ID、重叠目标写、超过一百万个拷贝区间和强制
benchmark 路径都会被拒绝。仅在可信 capture 的几何确实需要时提高预算。

记录的 CE 策略会在选定设备重新执行；`--per-block` 只更改 CE 策略。输出包含
正确性以及一次包含 launch 开销的同步墙钟时间；它不是吞吐 benchmark，也不回放
分布式调度、缓存生命周期、压缩、eventfd 时序、RDMA、SSD I/O 或模型输出。

直接原生绑定还接受可选的尾部参数 `gpu_startoff_inside_chunks`、
`cpu_startoff_inside_chunks`（字节，默认 0）和 `total_num_layers`（默认 0 保留原
`num_layers` 行为）。回放传入总层数，使部分层 capture 在 SGLang 独立 K/V 指针数组
中保持正确索引。
