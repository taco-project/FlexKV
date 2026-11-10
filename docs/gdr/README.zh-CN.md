# GDR GPU/CPU 传输功能

GDR 是可选功能，默认关闭。它由 `RegionBatchGroup` 管理，不需要修改 SGLang，也不依赖进程级全局 verbs 引擎。配置位于 `flexkv/common/gdr_config.py`，队列和内存注册逻辑在 `csrc/gdr/transport.*`，`csrc/gdr/runtime.*` 负责接入现有的 region 接口。适配层统一使用 `kv_dim` 和 tensor stride，不包含特定模型专用的 `is_mla` 分支。

## 编译与配置

在已经安装项目构建依赖的 NVIDIA GPU/RDMA 服务器上执行：

```bash
cd FlexKV-PR/FlexKV
# 还需要 rdma-core 开发头文件和库，例如 libibverbs-dev；
# GPU peer-memory 注册（nvidia-peermem）必须可用，并且 memlock 上限要足够大。
git submodule update --init --recursive
# setup.py 使用 build/include 和 build/lib，pip install 前先生成它们。
cmake -S . -B build -DFLEXKV_ENABLE_MONITORING=OFF
cmake --build build -j
FLEXKV_ENABLE_METRICS=0 FLEXKV_ENABLE_GDR=1 python -m pip install -v --no-build-isolation -e .
python -c 'from flexkv import c_ext; print("GDR compiled:", c_ext.gdr_compiled())'
```

未设置 `FLEXKV_ENABLE_GDR=1` 时，扩展不会链接 libibverbs；如果配置里仍启用 GDR，初始化失败后如何处理由相应策略决定。GDR 要求 NVIDIA CUDA 11.3 或更高版本，GPU 和驱动也必须能保证 RDMA 写入的可见性。NIC 的端口 1 必须处于 active 状态；以太网环境还要求一个非零、类型明确的 RoCE v1/v2 GID。初始化顺序与 134ad747 相同：先尝试非 link-local 的 v2 GID，再试 link-local v2 和 v1；一组 QP 连接失败后会先销毁，再继续尝试。worker 必须使用显存分配进程最终选中的同一个 endpoint。如果无法通过 sysfs 判断拓扑关系，可用 `nic_names` 显式指定 NIC。

把[带注释的 YAML](../../tests/gdr/config.yaml)合并到现有 FlexKV 配置。设置 `gdr.enabled: true` 后，两个方向默认采用 `auto` / `avgbytes`。H2D 和 D2H 有各自独立的配置和原生选择函数。两边的初始公式相同，但不会读取对方的设置。

| `path` | 行为 |
|---|---|
| `original` | 保留原有 `use_ce_transfer_*` 判断和 CE 子路径逻辑。 |
| `ce` | 强制使用现有 CE 后端，包括其 gather/segment 逻辑。 |
| `kernel` | 强制使用现有 SM 拷贝 kernel。 |
| `gdr` | 强制使用 GDR，跳过性能阈值判断。 |
| `auto` | 用当前方向的选择器判断；未选中 GDR 时回到 `original`。 |

`avgbytes` 在平均连续 segment 大小**严格小于** `gdr_when_ce_avg_bytes_below`，且 block fragment **不大于** `gdr_max_block_fragment_bytes` 时选择 GDR。`segments` 则要求 segment 数不少于 `segment_threshold`，同时满足相同的 fragment 大小限制。这些只是从原实现沿用的经验规则，不代表适用于所有硬件的带宽交叉点，也不表示“等累计到足够字节再传输”。旧名称 `min_bytes_per_call` 和 `max_chunk_bytes` 已不再接受。

`use_dedicated_gdr_thread` 会为每块本地 GPU 增加一个常驻 GDR 执行线程，与原传输线程并存。`batch_requests` 控制 H2D 合并上限：设为 1 表示关闭；4、8、16 取代了过去分散的 gdropt/gdrbatch8 命名。只有已经收到、共享同一个计数器且彼此兼容的 layerwise 操作才会合并。控制消息、不同计数器、目标地址别名和 block 数上限都会形成批次边界。实现不会为了凑批而等待。原始 operation ID 和成败结果保持独立，每层仍会为每个原始操作产生一个 semaphore token。即使 `auto` 最终选回原后端，同一批收到的操作仍可能先被合并。SGLang 的计算 batching 既不保证这些操作一起到达，也不会阻止它们一起到达。

## 正确性与性能边界

数据通路采用 **134ad747 的 direct owner-target 设计**。显存的实际分配进程注册原始 KV storage，并通过 FlexKV 现有 tensor handle 导出地址和 rkey。每个 worker 建立自己的 RC sender QP，与 owner 端的 target QP 连接，再把导入的别名地址映射到 owner 地址；worker 绝不会注册导入的 CUDA IPC 别名。H2D 的 RDMA WRITE 直接写最终 KV，D2H 的 RDMA READ 直接读取最终 KV。**这里没有 GPU staging buffer、gather/scatter kernel，也没有额外的 GPU 内 payload 拷贝。** CUDA context 和 stream 资源仍然存在，NIC 访问也仍会占用 PCIe/HBM 带宽。

`flexkv/transfer/gdr_owner.py` 实现 owner 私有握手和 lease。tensor 在注册期间始终保持存活。endpoint 会核对 GPU PCI 标识、NIC、端口、GID、MTU 和 READ depth。在一个 worker 接收批次中第一次执行可用的 D2H 前，worker 会请求原始分配进程执行 producer fence。交换失败后，后续传输全部进入失败状态，不能误把迟到的 ACK 当成下一批的响应。这个 fence 故意放在 owner 进程，而不是 worker context 内做 CUDA 同步；它是较保守的设备级同步，评估端到端性能时必须计入成本。

每个接收批次只做一次 fence 是严格成立的，并非抽样优化。run loop 会先排空 pipe，再调用 `_prepare_received_batch`，之后才启动第一个操作。因此 fence 发出时，这一批的所有操作都已经送达；调用方也不可能要求 FlexKV 保存尚未启动写入 kernel 的 KV。owner 的 device-wide synchronization 会等待该设备上已经提交的所有工作，包括仍排在驱动队列中的工作。`maybe_fence_d2h` 还会记录发送 `R` 的时刻：如果某个操作的 `_received_ns` 更晚，就重新执行 fence，防止从 forward path 提交的保存请求错误复用一个早于自身的 fence。合并批次同样上报最晚子操作的到达时间。如果在 fence 后再增加一次 `recv()`，或者把排空 pipe 调到 fence 之后，每一批都会多一次 owner 级往返。

runtime 每次最多准备 256 个 span，并把长度受限的 WR 链分摊到配置的 QP 上；前一批 DMA 执行期间可以继续准备 descriptor。SQ/CQ 占用有上限，队列容量不足时会先回收已完成项。只有 host 和最终 GPU 两端地址都连续时，相邻 span 才会合并。H2D 必须完成 RDMA 写可见性操作后才能发送 layer-ready 通知；D2H 必须完成 READ 后才能报告 host 数据就绪。专用执行线程和可配置的 H2D 合批仍然可用。

原有的 producer/block 所有权约束继续生效：buffer 和 ID 在完成前必须保持有效，KV 目标 block 必须已经可以安全覆盖。本地 stream event 无法独自同步另一个进程。`GPUCPUTransferWorker` 会先排空并销毁 sender 资源，再关闭 lease 或释放导入映射/host 注册。暂停 GPU 时，client 会等待 worker barrier，然后关闭 owner MR，最后释放导出的 VMM handle；恢复时重新创建 endpoint 和映射。owner 在整个注册生命周期内保留 MR，不会自动淘汰仍在使用 MR 的 allocation。

### 调用方必须保证什么，以及 FlexKV 为什么无法代劳

有三条顺序关系跨越进程边界，CUDA event 本身跨不过去。这个问题并非 GDR 独有：CE 和 SM kernel 同样从 worker 进程读写这块内存；只是 NIC 的访问完全游离于 CUDA stream 顺序，也不会出现在 profile 里，所以 GDR 下更难察觉。

**H2D 的 write-after-read。** 提交操作时，目标 GPU block 必须已经没有 reader。FlexKV 无法表达“等 owner 的 attention kernel 用完这个 slot”：worker 记录的 stream event 覆盖不到另一进程的 compute stream；对 GDR 而言，本地 `cudaStreamWaitEvent` 也挡不住 NIC。SGLang 集成依靠 allocation 生命周期满足这一点：`init_load_back` 在启动计算前从 allocator 获取新 slot；运行中的请求持续持有 slot，并锁住对应 radix prefix；只有 `cache_finished_req` 才归还 slot，而 SGLang 会在该批次 completion event 同步完成后才进入那里。真正建立 happens-before 的是这次 host 侧同步，allocator 负责把关系延续下去。只要有人绕过同步后的 result path 回收 slot，或把数据载入当前 forward 仍在读取的 slot，所有后端都会一起出错。

**D2H 的 read-after-write。** 约束与上面相对称，并额外由前述 owner producer fence 兜底。源 block 在整个传输期间也必须存活；集成层会一直持有 radix lock，直到 `check_completed_stores` 排空任务。

**RDMA 写可见性。** 详见后面的 XDR 章节。NVIDIA 使用文档规定的 device-scoped flush；一些特殊机型和场景 使用 NIC 侧 read-back，再做 XMLIR device synchronization。

`on_init_failure: original` 允许在缺少 owner 元数据、注册失败或 endpoint 建立失败时回到原路径；设为 `error` 时，只要操作选中 GDR 就直接失败。实现**没有 staging fallback**。新的 host region 注册失败会令操作失败。一旦开始 post，任何不能确认完成状态的错误——部分 post 失败、CQ 错误或超时、写可见性失败——都会终止 worker，避免复用可能仍在被访问的 buffer。正常的 preflight 异常则会先排空已经 post 的 chunk，再上报失败。

这次改动把 direct address、注册和完成语义迁移到 PR 的实例级 region 接口，并没有照搬 134ad747 的全局 runtime 和实验开关。真实硬件性能和跨进程 GPU 正确性仍须在部署服务器上验证；CPU contract test 无法证明物理带宽上限。

GDR 处理未压缩的 GPU/CPU region-batch 传输，可以与 Mooncake Store L3（CPU↔REMOTE）、本地 SSD、GDS 和 NIXL GDS_MT（GPU↔SSD）共存。这些路径由不同 worker 负责，不会替换 GPU/CPU worker。GDR transport 拥有自己的 PD、CQ、QP 和 MR；Mooncake 也使用自己的注册和连接，即使两者共享同一张 NIC。无需独占 NIC 或共享 QP，但仍受硬件资源上限约束。

现有后端限制不变：GDS 需要 SSD，NIXL 需要 GDS 且 effective TP=1，GDS 不能和 KV sharing 同时使用。Mooncake REMOTE2H 仍只用于 prefetch：计算的 H2D 开始前必须先预取到 CPU，而 writeback 要等 D2H 完成后再执行 H2REMOTE。启用 GDR 不会增加压缩 GDR 通路，也不会改变这些调度规则。关闭 GDR 时，默认行为和 API 保持原状。

`PHASE=unit` 会运行 CPU-only 的共存回归 `tests/gdr/test_backend_coexistence.py` 和 native contract。它覆盖配置组合、仍然有效的 GDS/NIXL/KV-sharing 限制、GPU/CPU worker 上的 GDR attach，以及用 mock worker 进程验证生产路由。并发数据正确性和资源余量仍要在部署硬件上验证。

GDR 不会预留资源编号：`qp_count` 只是数量，lane/rank 索引只是本地 bookkeeping。local QPN 来自 `ibv_create_qp` / `ibv_create_qp_ex`，peer QPN 由 owner 握手返回；lkey/rkey 来自各 transport 自己的 `ibv_reg_mr`。端口 1、P_Key index 0、选定的 GID 和初始 PSN 0 都不会占用 QP 编号。completion ID 只在私有 CQ 内有效，控制 socket 位于唯一的 `flexkv-gdr-*` 临时目录。GDR 不会修改 NIC/GID 配置，也不会绑定 Mooncake 的 TCP 端口。`ibv_fork_init` 是进程级初始化，不是资源重置。

native `backend_resource_isolation` contract 会先在同一虚拟 NIC 上为另一后端分配 QP/MR，注册相同的 CPU buffer 和 GPU page，并让该后端的传输继续挂在自己的 CQ 上；随后让 GDR 初始化、扩展 MR、传输并退出。测试确认 GDR 只向自己分配的 sender QPN post，使用不同的 GPU rkey，忽略传给 `release` 的外部 QPN，保留外部注册和 completion，并隔离初始化失败的清理。测试通过 deferred verbs 在 ASan/UBSan 下运行生产 transport 代码，也覆盖 extended-WR 和 XDR build；它不能替代真实 Mooncake SDK 或 cuFile/NIXL 硬件测试。

## 测试与输出

无需 GPU 的 contract test 会用 deferred CUDA/verbs 模型编译生产 runtime/transport，并启用 address/undefined-behavior sanitizer。覆盖范围包括 layout/`kv_dim` 组合、rank offset、远端地址转换、乱序 completion、有界异步队列、初始化 fallback、失败关闭、合批上限、别名 barrier、结果分发，以及真实 Unix socket lease 握手。native 模型还会断言 payload 的 CUDA copy 次数和 GPU 数据 allocation 次数均为零。

其中三项专门固定跨进程顺序。XDR build 会检查：每个 write epoch 对每个执行过 WRITE 的 QP 恰好做一次 read-back；纯 READ epoch 不做 read-back；read-back 无法 post 时必须 abort（`contracts flush`），device synchronization 失败也同样 abort（`contracts visibility`）。`Context::launch` 拒绝负数 `gpu_block_stride_in_bytes`，因为批内冲突范围只用最小和最大 block ID 包围请求，只有在 `ptr_at` 随 `block_idx` 单调递增时它才是安全上界。Python 侧还会测试 `maybe_fence_d2h` 容易遗漏的情况：fence 已经开始后才到达的操作必须重新 fence；合并操作上报最晚子操作的到达时间，而不是最早时间。

另一个 contract 是保护测试本身。`tests/gdr/test_visibility.py` 只有在 consumer 的 `eventfd_read` 与读取 kernel 之间没有其他操作时才有意义；在那里加一个 `synchronize` 或临时 allocation，就可能让完全没有 flush 的机器也通过。contract suite 因此解析该文件，确认循环体只调用 `torch.ne`、`range`、`max` 和 `time.perf_counter_ns`。设备边界以上只是普通数组处理，suite 也会实际执行：marker 不能连续重复，也不能与 untouched-byte 值冲突；failure classifier 会用 numpy stand-in 喂入合成的 late/stale/unwritten/torn buffer。这样可以确保它真能区分“迟到后变正确”和“一直错误”，而不是到了最难复现问题的机器上才发现报告逻辑有错。

```bash
PHASE=unit bash tests/batch_run_gdr_modes.sh
```

硬件 smoke 会先运行 `tests/gdr/test_direct_ipc.py`：单独的 allocation-owner 进程导出真实 CUDA IPC handle，依次完成强制 direct H2D/D2H、逐层校验、owner-context producer synchronization、lease/MR 释放以及第二代 owner。通过后才运行五组对比：

```bash
TP=8 PHASE=smoke bash tests/batch_run_gdr_modes.sh
TP=8 PHASE=compare DIRECTION=both bash tests/batch_run_gdr_modes.sh
```

microbenchmark 本身采用显式的**同进程 owner 注册**，用于隔离传输吞吐；独立 IPC gate 验证跨进程协议。只有明确设置 `CHECK_IPC=0` 才会跳过 gate。

五组分别是 `CE`、`kernel`、`overallperf-origin`、`GDR`、`overallperf-withgdr`。`overallperf-origin` 关闭 GDR，并原样传递原有的方向性 CE flag；它不会再引入一套 CE/kernel heuristic，因此默认 flag 下可能与 `kernel` 重合。`overallperf-withgdr` 在此基础上加入 `auto` 判断。测试会把初始化失败策略强制设为 error，避免 GDR arm 失败后 fallback 却被误认为测试成功。

方向、layout、合批和专用线程的示例：

```bash
TP=8 DIRECTION=H2D BATCH_REQUESTS=8 bash tests/batch_run_gdr_modes.sh
TP=8 DIRECTION=D2H QP_COUNT=4 bash tests/batch_run_gdr_modes.sh
TP=8 PHASE=smoke KV_DIM=2 SHARED_KV=0 LAYOUTS=LAYERFIRST,BLOCKFIRST \
  NOTIFY_MODE=polling bash tests/batch_run_gdr_modes.sh
for batch in 1 4 8 16; do
  TP=8 DIRECTION=H2D ARMS=GDR BATCH_REQUESTS=$batch \
    bash tests/batch_run_gdr_modes.sh
done
TP=8 DIRECTION=H2D USE_DEDICATED_GDR_THREAD=0 ARMS=GDR \
  bash tests/batch_run_gdr_modes.sh
# 详细日志只在诊断时开启，最好单独跑一次。
TP=8 PHASE=smoke DIAGNOSTICS=1 bash tests/batch_run_gdr_modes.sh
```

其他参数包括：`BLOCKS=32,256`、`CONCURRENCY=1,8`、`FRAGMENT_BYTES=8192`、`LAYERS=8`、`SCATTER=0|1`、`QP_COUNT=1..16`、`SELECTOR=avgbytes|segments`、`NIC_NAMES=mlx5_0,...`、`ROUNDS`、`WARMUP`、`OUTPUT`、`PYTHON`。主 region 和较小的 SWA-like region 会覆盖不同层数。H2D 会在每层通知到达时校验 GPU 字节，也会检查包括未触碰 block 在内的完整 buffer；D2H 检查完整 host buffer。校验和 route probe 均不计入计时。

每个输出目录包含：

- `direct_ipc.log`（smoke）、`run.log`、`topology.txt`、`metadata.json`：进度和硬件/源码/配置上下文。
- `raw.jsonl`：逐步保存 case、实际 route-probe 次数、字节数、sample、配置和物理提交次数；中断后也会保留。
- `summary.csv`、`summary.md`：完整运行的对比结果。
- `failure.json`：发生异常时失败的 case。

`median_ms` / `p95_ms` 衡量完整 ready-request set，包括 native WQE 准备、提交和完成，但不含 Python plan 构建。`prepare_median_ms` 和 `with_prepare_median_ms` 也会单独显示该成本。这是使用生产 coalescer/native 接口的传输 microbenchmark，不是 SGLang 端到端延迟，也不衡量请求在 IPC pipe 中积累的时间。`aggregate_GB_s` 对参与的 GPU 传输求和：shared-KV H2D 会向每个 rank 广播，而 sharded D2H 每个字节只写一次，所以它不能解释为单条 PCIe 链路带宽。`logical_GB_s` 表示逻辑 KV payload 的推进速度。

正常集成默认关闭两项 GDR diagnostics。若还想抑制 FlexKV 的常规日志，在服务端环境设置 `FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0`。测试 wrapper 默认如此；`DIAGNOSTICS=1` 会打开 INFO 摘要和阶段计时。direct-GDR 的 submit 时间包含 CQ 等待和最终写可见性，不包含 Python owner-fence 交换；后者应通过端到端 worker benchmark 测量。raw row 会写明 `implementation=direct-owner-v1` 和 `gpu_staging_bytes=0`，旧 staging 实现的结果不能与之混用。

## TP 与线程通信

payload 始终留在 pinned host memory 和最终 GPU KV storage 中。native rank task 共享一份不可变的 request plan 和 block-ID 数组；各 rank 自己解析地址并构造 WQE。layerwise engine 消息只包含 main/SWA 的源、目标 ID，不携带 KV payload。PP sibling replication 与本地 TP 线程分发是两套机制。长连续区间可以把 ID 数组压成 run 以减少 IPC 字节，但散乱 cache block 会产生大量 run 和额外解码；没有数据证明收益前，不引入新的压缩或 shared-memory 生命周期协议。

当 `path: gdr` 且 endpoint 可用时，只包含 GDR 请求的批次现在对每个参与 rank 只做**一次专用线程 handoff**，不再按 region/layer 分别 handoff。该线程遍历共享 plan，立即开始提交有界 WQE chunk，并在 RDMA 完成后记录相同的逐层 ready marker。批次入口只有一个本地 stream dependency，没有逐 region CUDA dependency event，也没有逐层回到原线程的往返。原始 batch submission 也不再为了等待 GDR job 而启动第二轮 TP fan-out。强制 GDR 会跳过无用的连续 segment 扫描，但执行阶段仍检查传输边界和 ID。

`auto`、混合后端批次和初始化 fallback 继续通过原线程有序 dispatch，避免把 CE/kernel 调度迁到另一线程，也避免在第一次传输前增加完整 descriptor-preparation barrier。`use_dedicated_gdr_thread: false` 仍可使用：传输很小时，即使只有一次 handoff，加上额外 CPU 线程也可能得不偿失。

route-probe 输出中的 `dispatch.batch` 和 `dispatch.region` 是次数，不是 payload 字节数。打开计时后，`[GDR-DISPATCH]` 会分开显示 queue wake-up 与本地 producer-stream fence；`[GDR-DIRECT-PRODUCER]` 衡量单独的跨进程 D2H owner fence。producer request 会先发给所有 owner，再统一等 ACK。一个慢 owner/rank 仍可能拖慢完成，owner-wide CUDA synchronization 也可能干扰并行计算。若要去掉这道 fence，必须建立 allocation-owner producer-event 契约；单发一条小消息无法安全替代。

可在 GPU 服务器上做如下简短的 CPU/TP scaling 对比，默认保持安静日志：

```bash
for tp in 1 8; do
  for dedicated in 0 1; do
    TP=$tp USE_DEDICATED_GDR_THREAD=$dedicated ARMS=GDR DIRECTION=both \
      BLOCKS=8,256 CONCURRENCY=1,8 LAYERS=64 ROUNDS=5 \
      OUTPUT=result_gdr_dispatch_tp${tp}_thread${dedicated} \
      bash tests/batch_run_gdr_modes.sh
  done
done
```

应分别比较小传输和大传输的延迟、aggregate/logical bandwidth 与准备时间。TP=1 到 TP=8 同时改变物理流量、内存/NIC 竞争和 dispatch 成本，不能当成纯 IPC benchmark。queue/fence 计时请另跑一次 `DIAGNOSTICS=1`，不要用 verbose timing 的结果作为主要性能结论。

初始化/注册沿用 `flexkv_134ad747/csrc/gdr/backends/verbs_transport.cpp` 和 `flexkv_134ad747/flexkv/gdr/owner.py` 的设计：每进程一次 verbs fork 初始化、候选 GID 选择、连接失败时新建 QP、owner 握手前先把 sender 置为 INIT、endpoint 校验、仅 owner 注册 GPU 并设置 SYNC_MEMOPS，以及释放 lease/MR 前先排空并销毁 sender。PR 保留实例级 transport，没有复制旧的全局 engine table。冲突的导入别名范围会在连接前失败，一致的重叠 view 可以正常使用。legacy CUDA IPC 还保留相对 CUDA allocation 的 offset，包括 allocator suballocation，从而修复参考实现同样存在的 zero-offset 假设。这些改动不改变 WQE post、queue depth、请求合并、路径选择或专用线程调度。

### GDR 自动决定 D2H 所有权

`gdr.d2h.auto_layer_parallel: true` 为默认值。在 rank fan-out 之前，默认采用 `sharded` 所有权的 shared-KV D2H 请求可能被转换成 `layer_parallel`。转换条件是：所有 rank 的 direct 注册/握手都可用；各 rank chunk 大小相同；后端为 AUTO；去掉 byte sharding 后的完整 chunk 仍能按 D2H 策略选中 GDR。最后一条保证 auto 阈值仍有意义：完整 chunk 超过 GDR 大小上限时，仍保留原来的 sharded 路由。显式指定 CE/kernel、非 shared KV、H2D、关闭 GDR 或某个 rank 不可用时，都不会改变原所有权。因此初始化 fallback 不会在同一请求内混出 layer ownership 和 byte ownership。调用方显式设置的非 sharded 模式也会保留。

设置 `gdr.d2h.auto_layer_parallel: false` 可完全服从调用方。worker 如果检测到显式的 `FLEXKV_KV_SHARED_ACROSS_RANKS_MODE`，也会关闭自动覆盖。`tune.py --share-mode ...` 和 `probe_d2h.py --share-mode ...` 同理，确保 sharded/layer-parallel 消融实验名副其实。关闭 GDR 时的 CE/kernel 路由和 H2D 逐层通知不受影响。

### D2H shared-KV 所有权与 descriptor 查找

在 worker/SGLang 集成中，进程启动前设置 `FLEXKV_KV_SHARED_ACROSS_RANKS_MODE=layer_parallel`。它只对完全一致的 KV 副本（`num_kv_heads == 1`）启用现有 D2H layer partition；H2D 和按 head 分片的 KV 不变。每个 rank 为分配给自己的层传完整 chunk；层数太少，或 main/SWA 层数不均，可能导致负载不平衡。显式所有权仍默认 `sharded`，但前述 GDR 自动覆盖可能重分配符合条件的请求。

direct benchmark 不读取这个环境变量，请使用 `tests/gdr/tune.py --share-mode layer_parallel ...` 或 `tests/gdr/probe_d2h.py --share-mode layer_parallel ...`；两者都会为所有对比 arm 显式覆盖 request 字段。原 workload 脚本不变，所选模式会记入 tuning/probe metadata。

transport 地址解析现在使用一次 post-local host MR hint，并为最近一次 owner mapping 提供经过检查的 fast path。每个真实 span 仍会做边界检查，包括散乱/非单调地址和跨 registration 情况；一个完整 chunk 的所有 span 都解析完后才会 post WQE。layer/KV 的 host offset 在 block 循环外只计算一次。completion、credit、visibility 和 H2D 逐层通知规则不变。这样减少重复工作，同时没有引入持久 descriptor cache 或准备线程；实际 GPU 性能仍须测量。

### 传输对比中的 DeepSeek V4 Flash 几何

向 `compare_modes.py`、`tune.py` 或 `probe_d2h.py` 传入 `--profile dsv4-flash`。该 profile 固定为 43 层、KV dimension 1、shared KV、每页 256 token，与 `test_dsv4_flash_concurrency_e2e.py::Dsv4Geometry` 及其默认 YAML 一致：

| Region | 全局层 | 每层每页字节数 |
|---|---|---:|
| c4 | 2,4,...,42 | 37440 |
| c128 | 3,5,...,41 | 1728 |
| c4_indexer | 2,4,...,42 | 8448 |
| swa_kv | 0,...,42 | 149760 |
| c4_attention_state | 2,4,...,42 | 65536 |
| c4_indexer_state | 2,4,...,42 | 16384 |

KV page 包含 576 字节对齐，indexer 每 token 使用 132 字节，两个 state buffer 使用 8 项 float32 ring。H2D 会先把 region 内局部层号映射回全局层号，再发 ready 通知。D2H 提交全部六个 region；tuning/probe wrapper 的 `--share-mode` 对每个 region 生效。`--layers` 和 `--fragment` 不会改变这个固定 profile。默认 `synthetic` profile 继续使用旧几何。metadata 和 raw result 会记录所有 region 的层 ID 和字节宽度。

这些 region 合计每个 block 9,158,208 字节。建议从 `--blocks 4,16,64`（1024/4096/16384 token）开始，而不是沿用 synthetic 的 2048 block。初始化使用有界临时 buffer，但校验仍需要额外 host/GPU 内存。这是 direct transfer microbenchmark：每个 region 传相同页数，使用独立 host allocation；它不模拟 SWA eviction、请求调度或 owner RPC。

### 在一些特殊机型和场景下 / XMLIR XDR 后端

在一些特殊机型和场景下后端沿用已验证的 `../special-platform validation suite/test_xdr.c` 和 `test_xdr_torch.py` 机制，同时保留 CE-only build 和 legacy worker。安装了 `torch_xmlir` 时，`FLEXKV_GDR_BACKEND=auto` 会选择 XDR；在 在一些特殊机型和场景下编译时建议显式设置 `xdr`，避免 runtime/library 选择含糊。

allocation owner 对每个新 storage 调用一次 `cudaPointerGetAttributes`，并把其中的 **hostPointer** 注册给 verbs。这是 XDR peer mapping，不是 host bounce buffer。layout 和校验仍使用原始 XPU 指针；WQE 和导出的 IPC metadata 使用 mapped address 加经过检查的 storage offset。worker 不会注册导入的 IPC 别名。tensor 生命周期和 owner lease 不变；D2H 在 RDMA READ 前继续执行 producer fence。

#### XDR 上的 H2D 写可见性

`Transport::finish()` 在发布任何 layer-ready event 前分两步建立可见性。`submit_layerwise` 会在每次 `notifier_.record` 前通过 `fence_gdr` 调用它，因此 eventfd 一定晚于下面两步：

1. **NIC 侧 read-back。** 本 epoch 内每个执行过 WRITE 的 QP 发起一次小型 RDMA READ，读入 endpoint 创建时一次性注册的 scratch buffer。PCIe 不允许同一 completer 返回的 READ completion 越过此前已经发往该 completer 的 posted write，所以这个 READ 的 CQE 表示设备已经接收 payload。WRITE 自身的 CQE 无法证明这一点：它是写向 host memory 的 posted write，属于另一个 completer，两者没有顺序保证。该做法等价于 NVIDIA 的 `cudaDeviceFlushGPUDirectRDMAWrites`，每层每 QP 增加一次有界往返，并单独计入 `verbs.flush_reads`，不混入衡量 payload 的 `wqes`/`posts`。
2. **XMLIR `cudaDeviceSynchronize`。** 保留它是因为 `../special-platform validation suite` 已用它验证端到端路径，而且它可能还会清理 XPU 侧 cache。

不能把第 2 步单独视为充分条件。它只排空调用进程的 CUDA queue，而 NIC 写入从未进入 CUDA queue；消费 kernel 又运行在 allocation owner 进程，不在 worker。单进程 `special-platform validation suite` 无法区分两者，因此这里增加 read-back，而不是假定 device synchronization 自带所需顺序。任一步 post 或 CQE 失败都会按 payload 路径相同的 fail-closed 规则终止 worker；在设备尚未接收数据时发布 layer-ready，比直接退出更危险。

在跨进程验证 XDR 专用可见性原语之前，不要删除第 1 步。`tests/gdr/test_visibility.py` 就是这道验证门；下节还会介绍 `FLEXKV_GDR_SKIP_FLUSH`，用于验证第 1 步是否确实不可缺少。NVIDIA build 不变：继续使用 SYNC_MEMOPS 和 device-scoped flush API，不做 read-back。

#### 跨进程验证门

`test_direct_ipc.py` 不足以回答可见性问题，因为它在 worker 进程里通过 `tensor.cpu()` 检查字节。copy engine 未必与消费 kernel 使用同一条可见性路径，而且发起 RDMA 的进程也不是我们真正关心其视图的进程。

`tests/gdr/test_visibility.py` 按生产环境分配角色：父进程充当 FlexKV worker，通过 RDMA 把 layerwise H2D 数据写进 IPC 导入的 KV；每个 TP rank 都是独立 allocation-owner 进程，拥有自己的 CUDA context、eventfd 和解释器，与 SGLang 一致。owner 等待该层 fd 后，直接在主循环中启动比较，没有 consumer thread，也没有 queue handoff。

每 rank 一个进程是测试敏感性的关键。旧版本把八个 rank 放在同一进程的八个线程中；所有 eventfd 在一个紧凑循环里写出，consumer 同时变成 runnable，随后又受 GIL 串行化，导致通知到 kernel 启动的尾部拖到数百微秒。窗口越宽，RDMA write 越可能自行变得可见，测试反而越难捕捉问题。

比较使用 `torch.ne`，这是机制要求，不是实现偏好。比较属于算术操作，只能由 SM kernel 执行，并从 owner 自己的 context 读取该层全部字节。同 dtype 的 `copy_` 很可能走 DMA，而 DMA 可见性从来不是争议点；gather 在这里也不可移植，因为 XDNN 不支持 uint8/int64 `index_select`。程序会在 IPC/GDR 握手前先在设备上探测该操作，确保厂商能力缺失能立即给出明确错误。

它之所以是可见性测试，而不是普通数据测试，还依赖以下设计：

- `eventfd_read` 与 kernel 启动之间**什么也不做**。expectation tensor、mask 和逐层 view 都在启动时一次性构建，避免临时 allocation 或 `aten::select` 意外建立可见性。
- 每轮写入一个**新 marker**，owner 会提前读取相同 block，让 cache 中保留上一轮字节。缺少 flush 时，读到的是上一轮 marker，而不是无法判断的噪声。
- kernel 只记录**判定结果**，解释信息稍后再读。kernel 判错但稍后读回正确，说明数据迟到，报告为 `VISIBILITY FAILURE`；两次都错才是 corruption。这正是 XDR read-back 要防止的特征。
- 奇数 block ID 从不作为传输目标，并始终与零比较，因此越界写会变成明确错误。
- 首轮前用合成的 late/stale/unwritten/torn buffer 自检 classifier。一个不会失败的 gate 不能作为证据。

```bash
FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0 \
python tests/gdr/test_visibility.py --tp 8 --qps 4 --rounds 200 \
  --notify polling --expect-flush --nics "$NICS"
```

`--expect-flush` 会在 `verbs.flush_reads` 为零时令测试失败，因为没有实际 read-back 的通过结果无法说明 XDR 是否需要它。`--background` 会在 owner 的旁路 stream 加入计算，更接近 serving 进程，也会污染 consumer 读取的 cache。

#### 中断运行

Ctrl-C 会同时发给九个进程，而八个 owner 正持有 worker QP 仍会访问的 MR。如果 owner 立即响应信号并拆掉资源，worker 就会失去底层对象，过去因此出现八份 traceback 和 `GDR drain acknowledgement failed`。现在 owner 在分配资源前把 `SIGINT` 设为 `SIG_IGN`，关闭动作由 worker 按顺序完成。worker 打印 `interrupted; releasing owner registrations`，正常释放 lease，关闭 pipe——这本身足以唤醒阻塞在 `recv` 的 owner——随后 terminate，若无效再 kill。每次 join 都有上限，避免关闭逻辑本身永久挂住。`PR_SET_PDEATHSIG` 处理最后一种情况：worker 被直接杀死、来不及清理时，owner 也不会永远留在逐层 `select` 超时中。

如果仍无法退出，阻塞点通常在 Python 以下：native call 释放 GIL 后占住主线程，解释器无法抛出 `KeyboardInterrupt`，继续按 Ctrl-C 也不会改变。不要猜，直接抓栈：

```bash
py-spy dump --pid $(pgrep -f 'test_visibility.py' | head -1) --native
```

临时可用 Ctrl-\（`SIGQUIT`）结束进程组。

#### 如何读通知到 kernel 启动的分布

摘要按每次检查输出 `n / min / p50 / p95 / max`。这是 gate 对自身敏感度的诊断，敏感度属于单个 sample，而不是总体。eventfd 后几微秒就启动的检查很敏锐，半毫秒后才启动的检查很钝；测试能力来自分布的左端，因此重点看 `min` 和 `p50`，不要看 `max`。

XMLIR 会把一个进程的第一次 kernel 启动拖慢数十毫秒，足以独自支配普通 max。每个 owner 因而会在等待 fd 前预热 `torch.ne`；剩余延迟才是 XMLIR 每次 launch 的 dispatch 成本。数字异常时用 `--tp 1` 交叉验证。owner 是独立进程，tp1 与 tp8 本应相近；若 tp8 明显更差，原因是设备或 NIC 竞争，可从分布中看到。

#### 如何读通知与 consumer 等待时间

`layer_ready` 报告提交后各层 owner 观察时间的中位数，以及每轮内部从完成到观察的延迟。它不评价 overlap：如果计算需要前所有数据早已 ready，同样是好结果。不能因为 consumer 较慢、最后观察时间更晚，反而让分数变好。历史结果中的 `BURST`/`progressive` 来自已经废弃的 ratio 指标，不再是当前验收规则。

timeline 中用 `fd - cq_last` 看 producer 的通知延迟，用 `notified - fd` 看后续观察延迟。`fd` 在所有 rank FD 写完后记录，是聚合时间，不是每个 rank 各自的 post 时间。`consumer_wait` 是合成 consumer 的 select/read wall time，包含调度和 syscall 开销；它既不是 SGLang compute stall，也不能证明 GPU 执行 overlap。消费 eventfd 与启动检查 kernel 之间没有额外 allocation、同步或 readiness probe。

serving 性能应测 SGLang `wait_until()` 的进入/返回时间，以及需要恢复时 token 是否已经全部就绪，再与 TTFT/吞吐结合判断。数据提前 ready 时，无需强求时间上的重叠。受控的提前通知回归应故意让后续某层未完成，并确认首层仍可先通知。XMLIR on some special machine types and scenarios 会在创建 owner 前拒绝 `--notify hostfunc`，请使用 `polling`；在支持 hostfunc 的平台，纯 GDR stream-free counter 断言只适用于 polling。

##### 在一些特殊机型和场景下结果（2026-09-18）：如代码所示，通知成批涌出

43 层、tp8、200 轮、`polling`，修复前：

```text
L0=2320 L1=2415 L2=2435 ... L41=2996 L42=3014  (us after submit)
-> BURST: layer 0 at 77% of the window, leaving 23% of it to overlap,
   against the 98% an ideal 43-layer pipeline would leave
```

传输在 1314 us 时已经结束，直到 2320 us 才出现第一条通知；43 条通知随后以稳定的 14–17 us 间隔在 694 us 内集中发出。这是 poller 遍历一层时执行八次 `cudaEventQuery` 和八次 eventfd write 的成本，不代表 43 层真的每隔 16 us 才完成一层。平均而言，每层在完成后又等了 1902 us 才通知 consumer，而整个传输只用 1086 us。**这条路径上的逐层 eventfd 当时没有给 consumer 带来任何收益。**

原因与代码一致：`RegionBatchGroup::submit_layerwise` 在 `pool_->run_on_all` 返回后才调用 `notifier_.arm()`，polling 因而要等所有层 marker 都完成后才启动。之所以这样安排，是因为 notification event 来自池，之前 record 过的 event 仍呈 completed 状态；如果在新的 `record()` 前先 arm，`cudaEventQuery` 会读到旧状态并错误发布 layer-ready。hostfunc 没这个问题，但 XMLIR 不支持 `cudaLaunchHostFunc`，一些特殊机型和场景 只能 polling。

##### 修复：到达本身就是 marker，因此 poller 可以提前 arm

机制不变：每个 rank 为一层留下 completion marker，由一个 watcher 汇总，最后一个到达后发布 fd。变化只在 marker 的定义。某个 rank 的该层完全走 NIC 时，event 无法再提供额外信息：`fence_gdr(rank)` 返回已经表示 QP 排空，XDR 下也完成 flush。因此 `LayerNotifier::record` 新增 `used_stream`，为 false 时直接把 rank 到达 layer boundary 当作 marker，不再调用 `cudaEventRecord`，poller 也无需 `cudaEventQuery`。43 层 tp8 往返可各省 344 次驱动调用。只要任一 rank 使用过 stream，整层仍回到 event 路径，因为 GDR fence 无法代表驱动尚未执行的工作。

这也解决 arm 时机。真正阻止提前 arm 的不是 fan-out，而是池化 event 在重新 record 前显示已完成。新增的逐层 arrival counter 可以回答“本次传输的 marker 是否已经放好”，poller 先检查它，再接触 event，所以 `arm()` 能移到 launch 之前，层完成后即可通知。native contract 会故意对已完成的池化 event 提前 arm，验证 arrival guard 确实能阻止误通知。

最后完成该层的 rank 不再亲自 post。tp8 要做八次 `write()` syscall，约 24 us；把它们放在最慢 rank 的关键路径上，会拖延该 rank 的下一层，并逐层累积。现在 rank 只调用约 1–2 us 的 `notify_one` 就继续工作，由 poller 自己写 fd。poller 使用带旧 50 us tick 超时的 `wait_for`，即使丢失 wake 也最多损失一个 tick；使用 stream 的层仍按原方式发现。`layer_stream_free_layers()` 会报告完全不接触 CUDA event 的层数，`test_visibility.py` 要求它等于 `rounds * layers`，防止测试悄悄落回 event 路径，测出看似合理但生产并不承担的驱动延迟。

##### 一些特殊机型和场景 修复结果：通知已解决，rank 偏斜成为上限

同样是 43 层、tp8、200 轮。第 0 层从整个窗口的 77% 提前到 42%，判定从 `BURST` 变成 `progressive`；层完成到 consumer 知道的等待从 1902 us 降到约 300 us。传输本身没变：中位 rank 的第 42 层仍约在 1330 us 回收，修复前约为 1314 us。

这次也解释了此前无法归因的 1 ms：`slowest - cq_done` 平均为 **1004 us**。中位 rank 在 274 us 排空第 0 层，最慢 rank 要到 963 us。偏斜原本一直存在，只是旧表只报告跨 rank 中位数，看不到它；旧实现又必须等最慢 rank 完成全部 43 层后才能 arm。如今一层仍要等最慢 rank，因此七个 owner 会多等约 690 us。`slowest rank per layer` 用来判断慢者是否固定：固定说明要查机器不对称，来回变化则可能要重新考虑是否必须等待所有 rank。

此前八次 `write()` 还会落在最后到达的 rank 线程上，使下一层 post 比原来再晚约 5 us；现在已移到 polling 线程。另一个仍未解决的问题是逐层 fence：只有回收完 L 层 completion，才把 L+1 层交给 NIC，所以每个层边界 NIC 都会空闲。该测试稳态每层 fence 22 us、handoff 5 us，而 200 Gb/s 传 512 KiB 的线上时间约 21 us，NIC 利用率约 80%；解耦二者最多约可改善 20%。

##### 43 层结果：问题不在网络，也不在 notifier

真实模型深度下，第 0 层 `post_last - post_first` 为 **891 us**：最快 rank 在提交后 167 us 已交给 NIC，最慢 rank 到 1058 us 才完成。`cq_last - cq_first` 为 895 us，几乎完全相同，第 42 层也仍是 1553 对 1554 us。**网络没有增加任何偏斜；偏斜在首字节传输前就已存在，并被原样带到末尾。**

`fd` 也排除了 notifier：层完成到 consumer 知道的 639 us 中，18 us 用于写 eventfd，621 us 是 owner 等待被调度。第 0 层单独就占 449 us，而它不存在通知 backlog；`notify_to_launch` p50 仅 9.8 us。也就是说，进程真正运行后十微秒就能响应，但为了拿到 CPU 核却等了约 450 us。

worker 自己的线程 fan-out 同样如此：`enqueue` 只做八次 mutex acquisition 和八次 `notify_one`，被唤醒的线程进入 lambda 却相差 **553 us**；某一轮甚至有 rank 到 2573 us 才进入第 0 层，此时最快 rank 已完成全部 43 层。根因可从 `Transport::finish` 看出：completion drain 是只含 `pause`、不 yield 的裸 `ibv_poll_cq` 自旋。传输期间八个 rank 线程各占满一个 CPU 核，八个 owner 进程只能争抢剩余调度机会。

后续优化顺序因此是：

1. **CPU 竞争**：已确认，并由 `OMP_NUM_THREADS=1` 解决。
2. **eventfd `write()`**：最慢 rank 每层承担约 24 us，现已迁到 polling 线程。
3. **逐层 fence**：按上面的估计，约占传输时间的 20%。

`slowest rank per layer` 分布接近均匀（r7 19% 到 r3 4%），没有固定机器或慢 NIC；各 rank 只是没有同时启动。

##### 答案是 `OMP_NUM_THREADS=1`：流水线随即出现

`vmstat` 很直观：空闲时 runnable（`r`）只有 0–8，测试期间却达到 **876–1286**，user time 99%。八个 rank 线程和八个 owner 进程不可能产生这么多 runnable。Torch 按 CPU 核数创建 intra-op pool，OpenMP 默认自旋等待；在 384 核主机上启动九个进程，会产生数千个自旋线程，测试里的所有等待都被它们污染。

设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 后，`worker_threads` 降到 2，几乎解决整个问题，无需绑核或改代码：

| 指标 | 继承默认值 | `OMP_NUM_THREADS=1` |
|---|---:|---:|
| 第 0 层通知 | 1404 us，窗口的 48% | **258 us，8%** |
| 相比理想流水线损失 | 45% | **6%** |
| `idle` | 500 us | **29 us** |
| 其中 owner 等 CPU | 482 us | **5 us** |
| fan-out 进入偏差 | 294 us | **18 us** |
| `notify_to_launch` max | 571 us | **84 us** |
| 100 轮 wall clock | 190 s | **4.3 s** |

此时逐层 eventfd 才真正发挥作用：consumer 启动第 0 层时，92% 的传输仍在进行。瓶颈从来不是 notification path 或 poller，而是数千个 OpenMP 自旋线程。**分析这套栈的任何时间数据前都必须先设置上述变量**，并把输出头中的 `worker_threads` 当成前置条件。

剩下两项已足够干净，可以继续优化。八次 `write()` 是剩余通知成本的全部：29 us `idle` 中，`fd - cq_last` 占 24 us，owner 只等 5 us。它还恰好由已经落后的 rank 承担。起点处的 launcher 偏斜从 811 us 消失到 42 us，却以每层 32 us 累积，到第 42 层达 1396 us。不能把 1396 us 全算作 post 成本：慢 rank 在一轮内通常固定，大约一半层由它 post，因此直接成本约为 `43 × 0.5 × 24 ≈ 500 us`，剩余来自机器不对称和 consumer 与 NIC 的真实竞争。

把 write 移到既有 poller 后，无需新线程便收回这约 500 us。43 层、tp8、100 轮、`OMP_NUM_THREADS=1` 的复测如下：

| 指标 | rank 线程 post | poller post |
|---|---:|---:|
| 第 0 层通知 | 268 us | **221 us** |
| 第 42 层通知 | 3180 us | **2062 us** |
| 传输本身 | 3065 us | **1952 us** |
| 每层 `idle` | 29 us | 39 us |
| 其中 `fd - cq_last` | 24 us | 31 us |

**通知每层慢了 7 us，整个传输却快了 1113 us。** 八次 write 原本与 post rank 的下一层串行，`43 × 24 us ≈ 1032 us` 几乎解释了全部收益；付出一次唤醒换走这项工作，回报约为 150:1。观察 timeline 的 `fd - cq_last`：poller 按层顺序 post，一旦写一层所需时间超过层到达间隔，该列就会逐层增长。这里它从第 0 层 51 us 降到第 42 层 18 us，尚能跟上；但后期每层只间隔 21 us，margin 已很小。更多层或更小层会先撞上这个瓶颈。

剩下的是 AND 语义：只有全部 rank 排空后，L 层才通知所有 consumer；但 H2D 中每个 rank 只写自己 GPU 的 shard，因此 consumer r 理论上只依赖 rank r。改成逐 rank 通知可以完全去掉最慢 rank 等待，但这会改变 eventfd 的含义，属于设计决策，所以本次没有修改。

现在还能看到 rank 不对称，但它不像 NIC 问题：三次运行分别由 r7、r1、r2 长期最慢，而且自动选择 NIC 已得到理想拓扑。建议不设置 `nic_names`，让 `select_local_nic` 自动选择，避免手误。单轮内固定一个慢 rank 更像线程落点问题。`describe_laggards` 现在不会仅凭一次运行断言机器不对称，而会要求第二次数据，因为单次观察无法区分两类原因。

还要注意 overlap 并非免费：传输从 2425 us 变为 3065 us，最快 rank 每层从 22 us 变为 39 us，而纯线上时间约 21 us。consumer 真正与 NIC 并发后会争用 HBM，这既是测试目的，也是代价。测试用 `torch.ne` 扫描整层 capacity，可能比 attention 访问单层 KV 更重，因此 26% 应视为上界，不能直接当 serving 预测。也不要急于用 `taskset`：一次中间测试把八个 owner 各绑一个核后，owner 等待反而达到 **16 ms**；CUDA runtime 的驱动和 OpenMP 线程也在争同一个核。把 worker 限在 0–15 核还会让八个 rank 线程全落在同一 NUMA node，远离一半 NIC。这里 `OMP_NUM_THREADS` 这种简单限制比复杂亲和性更可靠。

#### timeline 表

`layer_ready` 把多项成本混在一起。`set_layer_timeline(True)` 会在 `submit_layerwise` 已有的 layer boundary 周围加入 worker 时间戳：该层请求全部交给 NIC lane 后记 `posted`，收完 completion 后记 `reaped`。时间源是 Linux 上对应 `CLOCK_MONOTONIC` 的 `steady_clock` 纳秒，因此无需时钟交换即可与 owner wake 时间对齐。`test_visibility.py` 会打开它并打印：

```text
layer post_first post_last cq_first cq_last   fd notified idle
L0           166       934      188     959  966     1404  446
L1           203      1027      220    1049 1056     1568  519
...
L42         1105      2521     1127    2539 2546     2952  413
```

每列都是一轮内跨 rank 的极值，因此任意两列相减，都代表某个 rank 真正经历过的等待。主要读四类问题：

- **流水线。** 比较 `post_first[L+1]` 与 `cq_first[L]`，可以看到上一层回收完才启动下一层造成的 NIC 空闲。
- **偏斜及来源。** `post_last - post_first` 是网络开始工作前的 rank 偏差，`cq_last - cq_first` 是传输后的偏差。两者比较可判断偏斜来自传输还是 launcher；fan-out 进入偏差还能区分“线程本来就晚启动”和“准时启动但首层很慢”。
- **通知。** `notified - cq_last` 以 `fd` 分成 poller 唤醒并写完该层 fd 的 `fd - cq_last`，以及 owner 等待调度的 `notified - fd`。这是两个不同问题；后者占主导时，再优化 post 路径也无济于事。poller 已提前 arm，所以 `joined` 不再限制通知，但仍限制 launcher。

instrumentation 默认关闭。它为每个（layer, rank）预分配一行，设备线程不会在被测 boundary 上加锁；开启后每层每 rank 增加两次 `clock_gettime`。

##### 单独看一轮

第一张表对所有轮次和 rank 分别取各列中位数，适合看分布，不适合还原一段真实时序。两个中位数之差仍只是“中位数之差”，而 `cq_last` 本身又是最大值，可能比任意一轮真实结果都好。第二张表因此展示同一轮：

```text
layer post_first post_last cq_first cq_last   fd notified idle
L0           177       279      217     298  305      502  204
...
```

同一行都来自同一次传输，`notified - cq_last` 才是真实发生的等待，`post_first[L+1] - cq_last[L]` 也是真实 NIC 空档。程序选择通知延迟位于中位的那轮，同时打印所有轮的范围。单轮表还能显示 backlog 的形状：中位表只给每层一个 `idle` 数字，单轮表则能看到它逐层下降，说明 consumer 早期落后、随后不断追赶。

##### 分 rank 查看

前两张表每列都跨 rank 取极值，看不到某个落后 rank 自己的节奏，因此第三张表逐 rank 展示：

```text
 rank   finished  step_p50  step_p95  step_max fence_p50
r0        1190.3      27.7      33.3      38.5      12.5
r2        1917.4      28.3     207.7     216.9      12.5
...
```

`step` 表示同一 rank 从一个 layer boundary 到下一个，包含该层 launch 和 fence。判断关键是中位数与尾部：`step_p50` 高，说明该 rank 每层都慢，应查路径、NIC 或内存带宽；`step_p50` 正常但 `step_max` 高，说明它能跟上，只是偶尔失去 CPU，应查线程放置。

比较最慢 rank 时应对比**最快值**，而不是中位数。若八个 `step_p50` 呈平滑梯度，对中位数看可能只有 1.15 倍，显得正常；对最快值却可能达到 1.59 倍，这才暴露所有 rank 都不同。另一个常见误读是：所有 rank 都有相同高 tail，通常说明 host 整体抖动，不是某个 rank 的 placement 问题。

##### 整段运行期间机器状态一致吗？

所有中位数都假设各轮来自同一分布。运行中途性能变差时，这一假设会悄悄失效：数字仍打印到三位小数，却无法代表前后任一阶段。gate 会先把运行分成两半比较；如果出现：

```text
NOT STATIONARY: the run slowed from 2395us to 3029us (1.26x) between its halves,
so every median below blends two machines and none of them is comparable to
another run. Find what changed first
```

就应停止解读后续表格，先排查其他租户、温控降频或遗留进程。拿非平稳运行与历史结果、阈值或其他后端比较，只是在测机器当时的状态。

#### read-back 是否确实不可缺少？

gate 通过只说明 read-back **有效**，不说明它**必需**。若要证明每层每 QP 的有界往返值得保留，唯一办法是移除 read-back 后观察；`FLEXKV_GDR_SKIP_FLUSH=1` 只关闭第 1 步、保留第 2 步，准确检验 device synchronization 是否已经足够：

```bash
FLEXKV_GDR_SKIP_FLUSH=1 FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0 \
python tests/gdr/test_visibility.py --tp 8 --qps 4 --rounds 200 \
  --notify polling --control-no-flush --nics "$NICS"
```

变量每进程只读取一次，默认走安全路径，启用时以 WARNING 记录。`--control-no-flush` 会断言 `verbs.flush_reads == 0`，避免拼错变量后 read-back 实际仍开启，却把通过结果误当成反证。native contract 的 `contracts noflush` 也会在开关失效时失败。

两种结果的含义不同：**失败是决定性证据**，表明 read-back 必不可少；**通过则不是**。内存顺序是架构属性，在 10 us 窗口里未观察到违例不能证明不存在。摘要会把这种运行标为 `CONTROL RUN, not a gate`，防止后来被当成绿色验收结果。

##### 在一些特殊机型和场景下结果（2026-09-17）：未观察到错误，read-back 仍保留

关闭 read-back 后运行 200 轮 × 8 层 × tp8，共 12,800 次，没有 mismatch；开关也由 `verbs.flush_reads=0`、WARNING 和 `contracts noflush` 验证生效。但这是空结果，不能支持删除。测试自身 `min=6.4us` 已是 Python consumer 启动路径下限，而 PCIe posted write 很可能远早于此排空，危险窗口可能完全低于测试分辨率。增加轮数只会增加窗口之外的 sample，不会提高敏感度。

成本也未分辨出来：有 read-back 为 85.2 s，无 read-back 为 85.5 s，均由 harness 开销主导。删除它会用无法量化的正确性余量换取无法测出的收益。NVIDIA 专门提供 `cudaDeviceFlushGPUDirectRDMAWrites` 处理同一风险；除非厂商文档明确保证 XPU GPUDirect write 在 CQE 可见后对 SM 可见，否则应保留 read-back。真正能定论的是文档，而不是继续堆样本。仍有区分力的实验是连第 2 步也关闭：若可见性实际来自 device synchronization，两步都关应失败；若仍通过，说明 gate 根本触及不到危险窗口。

所有 owner/worker 进程都应在 import torch 前设置：

```bash
export BKCL_ENABLE_XDR=1 CUDA_ENABLE_P2P_NO_UVA=1
export CUDA_DEVICE_ORDER=OAM_ID MLX5_SCATTER_TO_CQE=0
# 使用与已成功运行 special-platform validation suite 相同的 XMLIR 环境和 LD_LIBRARY_PATH。
FLEXKV_ENABLE_GDR=1 FLEXKV_GDR_BACKEND=xdr \
FLEXKV_ENABLE_KERNEL_TRANSFER=0 FLEXKV_ENABLE_METRICS=0 \
python -m pip install -v --no-deps --no-build-isolation -e .

# 按当前逻辑设备顺序自动选择 NIC：
NICS=''
FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0 \
python tests/gdr/tune.py --ipc --tp 8 --qps 4 --nics "$NICS"

FLEXKV_LOG_LEVEL=WARNING FLEXKV_TRANSFER_TRACE=0 \
python tests/gdr/tune.py --profile dsv4-flash --share-mode layer_parallel \
  --arms CE,GDR --directions H2D,D2H --tp 8 --qps 4 --nics "$NICS" \
  --blocks 4,16,64 --concurrency 1 --warmup 3 --rounds 20 \
  --output result_special_xdr
```

生产环境若显式设置 `gdr.nic_names`，必须按**逻辑设备顺序**填写；visibility/order 改映射时列表也要同步修改。显式 NIC 可绕过 XMLIR PCI bus 查询。原有 endpoint/GID 协商不变，不会隐式猜 NIC index，也不会覆盖可见性设置。做正确性验收时应强制 GDR，并关闭初始化 fallback。一些特殊机型和场景 的 `kernel` arm 不是 SM-kernel baseline，应与 CE 比较。

CPU contract 覆盖非恒等 XDR mapping、storage offset、远端 owner 地址转换、READ/WRITE 以及同步失败，但不能证明真实一些特殊机型和场景 的跨进程可见性或性能；扩大 workload 前，IPC smoke 与 `test_visibility.py` 都必须在设备上通过。

### 逐层校验超时与有界 readback

`--notification-timeout` 默认 60 秒，分别应用于每次 eventfd 通知等待和 native completion 等待。检查字节所用时间不会挤占下一次通知预算。超时信息包含 case、op、global layer、rank、token index，以及此前等待、校验和总耗时。字节 mismatch 还会报告 region/local layer、KV component、GPU/CPU block ID 和 byte offset。native 与 consumer 错误都会保留。

`--verification-chunk-mib` 默认 16，用于限制每层 readback slab，最小为一个 GPU page。copy 使用连续 GPU slice，scatter indexing 只在 CPU 上做；layer ready 后会检查每个请求字节。包括 untouched block 在内的完整校验仍然保留，大 case 仍需大量内存。该参数只限制逐层临时 copy，不限制总测试内存或完整校验时长。DSv4 batch 脚本分别用 `NOTIFICATION_TIMEOUT` 和 `VERIFICATION_CHUNK_MIB` 暴露这两个选项；二者都不改变生产传输代码。

#### timing matrix 检查什么、不检查什么

绿色 matrix 的含义比表面上窄。每个 case 只有一次传输——route probe——会做字节校验：H2D 在 layer-ready token 到达时逐 rank、逐层检查所有请求字节；两个方向之后都比较包括 untouched block 在内的完整 buffer。随后的 `--warmup` 和 `--rounds` 计时迭代不校验，以免校验进入计时。

循环后的 compare 不能弥补这一点。每轮生成相同的确定性 block pair，把相同字节写到相同地址；第 7 轮损坏若被第 8 轮正确覆盖就无法发现，某轮完全没搬数据也可能沿用上一轮正确内容。最终 compare 只能发现最后一轮仍残留的损坏。

`VERIFY_EVERY_ROUND=1`（`--verify-every-round 1`）会在每轮前恢复 baseline，并在轮后比较所有字节，能同时发现 no-op 和中间轮损坏，校验仍不计时。请使用单独的 `OUT`：reset/compare 会扰动稳态，其数据不能与纯性能运行直接比较；`summary.md` 和 `summary.csv` 的 `verified_every_round` 列都会注明。

D2H 不需要也没有位置做逐层校验：它以 `sync=true` 提交一整组请求，返回前排空所有 stream，因此完整 buffer compare 就位于完成点。这个不对称是实际语义造成的：RDMA READ completion 已经表示数据进入 host memory 且对 CPU 可见；GPUDirect WRITE 进入 device memory 的 completion 没有同等保证，所以 read-back 问题只出现在 H2D。

### 在一些特殊机型和场景下批量对比脚本

根目录两个脚本都可在本代码树内独立运行，不导入 PR 或 special-platform validation suite 的 Python 文件。`batch_run_ce_kernel_tp8.sh` 保留 synthetic workload；`batch_run_dsv4_transfer_modes.sh` 使用 43 层 DSv4 几何，默认 `KV_SHARED_MODE=layer_parallel`。两者默认比较 CE、overallperf-origin、GDR 和 overallperf-withgdr。一些特殊机型和场景 没有编译 SM transfer path，因此拒绝 `kernel`；original arm 保留 CE on some special machine types and scenarios 实现。

两个 launcher 都会在 import torch 前设置 XDR 环境默认值，并为 original 和 auto fallback 设置 `FLEXKV_USE_CE_TRANSFER_H2D=1` / `FLEXKV_USE_CE_TRANSFER_D2H=1`。显式环境变量优先；CE-only build 中，任一方向的 CE flag 设为 0 都无法使用 original path。`NIC_NAMES` 要按逻辑设备顺序设置，并使用已经验证的 XMLIR 环境。

```bash
export NIC_NAMES=''  # 自动按 PCIe 映射，不按相同编号配对 NIC
# 简短验收（默认启用 IPC preflight）：
HIT_TOKENS=1024 ROUNDS=1 WARMUP_ROUNDS=1 CPU_RUNS=1 GPU_RUNS=none \
  bash batch_run_dsv4_transfer_modes.sh
# 完整 matrix；显式区分输出目录：
OUT=result_special_dsv4 bash batch_run_dsv4_transfer_modes.sh
OUT=result_special_synthetic bash batch_run_ce_kernel_tp8.sh
```

输出沿用 PR 格式：每个 cell 的文本日志和 JSON/CSV，以及合并后的 `raw.jsonl`、`summary.csv`、`summary.md`。`DRY_RUN=1` 只打印命令，不 import torch，也不创建输出目录。shell/mock-output 检查不能证明设备正确性或性能。

### XMLIR IPC 传输

在 XMLIR 上，`TensorSharedHandle` 始终使用厂商 PyTorch 的 `reduce_tensor` 导出/重建路径，即使调用方传入 `force_direct_ipc=True`。这个 flag 不能在 在一些特殊机型和场景下强制触发 NVIDIA VMM 探测或 `cuMemGetAddressRange_v2`。storage offset、IPC mapping 复用和引用计数均由 allocator 管理。GDR owner MR/XDR metadata 独立且不变，不增加 tensor copy 或 staging。在错误逻辑设备上 rebuild 会直接失败，不会复制到新 allocation；rebuild 失败向上传播，不会返回空 tensor。NVIDIA 继续使用原有 legacy IPC/VMM 选择。

XMLIR 上生产通知默认采用 polling，与一些特殊机型和场景 测试 launcher 一致；显式 `FLEXKV_LAYERWISE_NOTIFY_MODE` 仍可覆盖。IPC smoke 在一些特殊机型和场景下 输出 `handle_types=torch_reduce`，并验证两代 owner、逐层数据以及 H2D/D2H。special-platform validation suite 的 RDMA 测试是单进程测试，真实设备上仍必须完成这些跨进程检查，其中 `test_visibility.py` 专门覆盖消费 kernel。

无论使用哪种模式，eventfd `write` 都可能失败；丢 token 会让 consumer 一直等到 `FLEXKV_LAYERWISE_WAIT_TIMEOUT_S`（默认 240 秒），然后抛出包含 rank、counter 和 layer 的 `TimeoutError`。这比永久挂死好，但 consumer 无法知道 token 丢失原因，所以 `LayerEventfdTable::post` 现在会在 short write 时记录 fd 和 `errno`。两种模式共用该路径；polling 还会把真正的 `cudaEventQuery` 错误转成 hard failure，而不是无限等待永远无法完成的层。

### 在一些特殊机型和场景下自动选择 NIC

保持 YAML 中 `gdr.nic_names: []`（默认），或在 batch launcher 中设置 `NIC_NAMES=''`。需要清除此前 export 的 NIC 列表；显式列表始终视为有意覆盖，会关闭自动选择。

后端查询每个逻辑 XMLIR 设备的 PCI BDF，统一大小写后解析 `/sys/bus/pci/devices/<BDF>`。active 的 verbs port-1 设备通过 canonical `/sys/class/infiniband/<NIC>/device` 路径逐组件比较：优先共同 PCI ancestor 最深的候选，再取路径最短者。路径缺失、没有共同 PCI root 或最佳候选并列时，初始化会给出可操作的错误；`on_init_failure` 仍决定是否回到 original path。实现不会用相同编号、最近 BDF 数字或枚举到的第一张 NIC 猜测。

IPC owner 会在注册前选择 NIC，并通过现有 lease 握手共享选中的 NIC 和 endpoint；worker 使用该 owner endpoint。本地 benchmark allocation 使用同一个选择器。选择只发生在初始化阶段，不会每次传输重新判断。INFO 日志下，`[GDR-NIC]` 会显示逻辑设备、BDF 和所选 NIC；WARNING 保持安静。

在提供的八设备 OAM 拓扑上，预期逻辑顺序为 `mlx5_bond_3,mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_7,mlx5_bond_5,mlx5_bond_4,mlx5_bond_6`。这只是回归测试 fixture，不是写死在生产代码里的映射。
