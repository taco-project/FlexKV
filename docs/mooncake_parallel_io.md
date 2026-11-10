# 可选的 Mooncake 共享 TransferEngine 多 Store 通道与跨请求流水 IO

仅在 `FLEXKV_MOONCAKE_PARALLEL_IO=1` 时启用。默认不启用。

实现位于 `flexkv/external/mooncake_parallel_io.py` 的 `attach_parallel_channels` 与 `install_worker_pipeline`。没有修改 SGLang、HiCache、GPU 搬运或计算路径；关闭开关时原 `MooncakeStoreTransferWorker` 初始化、MR 注册及 `run` 循环保持 AST 级等价。

## 为什么采用方案 1 + 跨请求流水

在单进程 7 线程共用同一个 `MooncakeDistributedStore` 的测试（`test-20261105-233426-mh_tvfhp`）中，7 个线程同时在 0~1 ms 内发起 `batch_put_from`，但完成时间呈约 2.1 ms 等差阶梯（5.8 ms → 8.0 ms → 10.0 ms → 13.3 ms → 16.3 ms → 19.4 ms → 21.4 ms），总 PUT 带宽仅从单路的 9.04 GB/s 升至 11.42 GB/s。根因是单个 `MooncakeDistributedStore` 实例内部串行化了搬运与 master 交互。

本实现同时解决两个问题：

1. **方案 1（1 份 `TransferEngine` + 1 份 Host MR + 8 个独立 `MooncakeDistributedStore` 通道）**：
   - 在 `MooncakeStoreTransferWorker` 进程内创建 **1 个** `TransferEngine`（`device_name=""` 自动发现 8 张 HCA），Host 内存池只在这一份 `TransferEngine` 上执行 **1 次** `register_memory`，彻底避免 8 进程方案带来的 8 倍 MR 注册开销。
   - 创建 `workers`（默认 8）个独立的 `MooncakeDistributedStore` 实例，通过 8 参数 `store.setup(session_id, metadata_server, per_lane_segment_size, local_buffer_size, protocol, device_name, master_addr, transfer_engine)` 共同挂载到该 `TransferEngine` 上。每个通道拥有独立的 `store` 对象与专属常驻线程，消除单 `store` 锁争用。
   - 若运行环境的定制 Mooncake wheel 不支持同进程多 `store` 共享同一个 `TransferEngine`，启动时自动捕获并记录 `event=shared_te_fallback`，平滑退回单 `store` 多线程通道，服务不中断。
2. **多请求并发无屏障流水（Cross-Request Pipelining）**：
   - 当高并发 prefill batch 中有多个请求同时触发 `H2REMOTE` / `REMOTE2H` 时，`MooncakeStoreTransferWorker` 的主循环不再按请求逐个阻塞等待全部页完成。
   - 主线程仅做非阻塞拆包与 key/指针计算，将每个请求的页切片按最少在途字节（Least-Loaded）立即投递到 `lane[0..7]` 的专属队列，随后立刻返回 `recv` 接收下一个请求。
   - 哪条通道先搬完手头的切片，立刻从队列取下一个请求的切片继续搬运；每个请求维护独立的 `remaining_slices` 计数，归零时由通道线程直接将 `WorkerTransferResult` 推入 `finished_ops_queue`，小请求无需等待大请求。

## 环境变量

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `FLEXKV_MOONCAKE_PARALLEL_IO` | 未设置 | 仅值 `1` 开启；`0`/未设置走原路径 |
| `FLEXKV_MOONCAKE_SHARED_TE` | `1` | 默认开启方案 1（多 `store` 共享单个 `TransferEngine` 与单份 MR）；设为 `0` 仅使用单 `store` 多线程 |
| `FLEXKV_MOONCAKE_IO_WORKERS` | `8` | 独立 Store 通道数，1～8 |
| `FLEXKV_MOONCAKE_IO_MIN_PAGES` | `8` | 每条通道最少分配页数；63 页在默认 `8` 下拆为 7 路，设为 `7` 可铺满 8 路 |
| `FLEXKV_MOONCAKE_IO_MAX_INFLIGHT_BYTES` | `536870912` | 跨请求在途总字节上限（默认 512 MiB），超出时在投递端背压等待 |
| `FLEXKV_MOONCAKE_IO_TIMEOUT_S` | `300` | 通道排空与在途配额等待超时上限 |

## 日志与验收口径

开启后 `server.log` 会打印 `[FlexKV-MC-PARALLEL]`：

- `event=ready`：包含 `"channels": "shared_te_stores"`、`"distinct_stores": 8`、`"session_id": "<ip>:<rpc_port>"`。若回退则为 `"channels": "threads"` 并前置一条 `event=shared_te_fallback`。
- `event=complete`：记录单次请求的真实搬运墙钟 `elapsed_s`、排队等待时间 `queue_wait_s`、有效带宽 `GB_s`、各通道 SDK 调用明细 `lane_sdk_calls`（含 `store_id`）及 `peak_overlapping_sdk_calls`。
