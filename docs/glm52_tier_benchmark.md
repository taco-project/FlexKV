# GLM52 无模型分层传输基准

本脚本不启动 SGLang、不加载模型、不创建 router。使用 FlexKV 的 `RegionBatchGroup` 测 GPU/Host，使用 `MooncakeStoreClient` 测 Host/Store。它测传输接口，不包含 KVManager 的 radix 匹配、工作队列、owner RPC、调度或端到端请求吞吐。

脚本：`benchmarks/benchmark_glm52_tiers.py`。SGLang 仓库的配套入口是 `scripts/flexkv/bench_tiers.sh`，加载现有 `glm52-1p.env` 及 some special machine types and scenarios CUDA 兼容库路径。只需要已编译的 FlexKV 原生扩展和已有镜像中的 Mooncake SDK；新增脚本本身无需编译。不要将本地 CPU 测试通过当作 some special machine types and scenarios 实机验收。

## 运行

在 some special machine types and scenarios容器中，确保没有模型进程同时占用要测的 GPU/内存，以便结果可比较。脚本不自动停止任何服务。

终端一，沿用原入口启动 Store master/元数据服务；已启动时复用，不要重复启动：

```bash
cd /data/kelvinlv/sglang-v0.5.10-kunlun-main-hy4/scripts/flexkv
./start_1p.sh master
```

**不启动 prefill、decode、router。** master 不是 KV 内存池提供者；测试中的 L3 客户端默认合计挂载 16 GiB Store segment，实际覆盖 JSON 中的 400 GiB，且客户端数量增加时总量不变。客户端退出前只删除本次 UUID 命名空间的测试 key，绝不调用 clear/remove_all。

终端二，先跑原始 63 页载荷的 Host/L3 对照：

```bash
cd /data/kelvinlv/sglang-v0.5.10-kunlun-main-hy4/scripts/flexkv
./bench_tiers.sh --gpus 0 --pages 63 --lanes 1,8 --rounds 5
```

然后扫描八卡和单卡 GPU/Host/L3。GPU-NIC 对应顺序来自之前这台机器的拓扑；换机器需重新核对：

```bash
./bench_tiers.sh --gpus 1,8 --pages 1,4,16,63,128,256 \
  --lanes 1,2,4,8 --rounds 5 \
  --gpu-nics mlx5_bond_3,mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_7,mlx5_bond_5,mlx5_bond_4,mlx5_bond_6
```

默认 Store 的 `device_name` 沿用 JSON。要区分“单客户端自动路由”和“多个客户端各绑定一张 NIC”，再单独跑：

```bash
./bench_tiers.sh --gpus 0 --pages 16,63,128,256 --lanes 1,2,4,8 \
  --store-nics mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
```

这指定的是客户端 NIC，不保证 Store 副本所在的段/NIC 与客户端一一对应。单机注册多个段时，SDK 的副本放置也可能改变；根据 NIC 计数、运行时记录和结果共同判断。

比较 64 MiB SDK 拆批上限和不拆批：

```bash
./bench_tiers.sh --gpus 0 --pages 63,128,256 --lanes 1 --layouts whole-pages --batch-mib 64
./bench_tiers.sh --gpus 0 --pages 63,128,256 --lanes 1 --layouts whole-pages --batch-mib 0
```

默认 `--get-concurrency 1`。需要检验 FlexKV 已有的客户端内 GET 并发能力时，在相同配置下分别加 `--get-concurrency 1` 和 `--get-concurrency 4`；不要同时改变 lanes、NIC、拆批上限后把改善归于单个因素。每个客户端的默认在途预算仍由 `FLEXKV_MOONCAKE_GET_MAX_INFLIGHT_BYTES` 控制。

CPU/NUMA 控制：可使用 `numactl --cpunodebind=0 --membind=0 ./bench_tiers.sh ...`，再用 `numactl --interleave=all ./bench_tiers.sh ...` 对照。默认不改变亲和性，单线程初始化共享 Host 池，可能集中 first-touch 在一个 NUMA；这正是必须记录和单独对照的变量。不要把该布局当作已经复现服务的 hugepage/NUMA 分配策略。

只检查配置和载荷，不导入 torch、不连接 master：

```bash
python /data/kelvinlv/FlexKV-special-platform/benchmarks/benchmark_glm52_tiers.py --plan
```

## 扫描变量与计时口径

| 项目 | 含义 |
| --- | --- |
| `--gpus 0` | 只测 Host memcpy 和 L3，无 GPU 设备操作 |
| `--gpus 1,8` | 单卡和八卡执行相同总载荷；不是八卡各搬一份 |
| `--pages` | 逻辑完整页数，每页 64 token、3,855,616 B |
| `--lanes` | 独立 spawn 进程、独立 Mooncake 客户端；不是 GPU 数，也不是自动等于 NIC 数 |
| `whole-pages` | 63 个完整页分配给各客户端；单客户端最接近当前 FlexKV Store 批次 |
| `page-shards` | 每页按连续字节范围拆给各客户端，63 页、8 lanes 时每客户端 63 个 481,952 B 对象；近似比较 rank 分片形态，但不是实际 HiCache 的层布局/适配器 |
| `HOST_MEMCPY` | 单线程 libc memmove 对照，不经过 Store；不是机器峰值内存带宽 |
| `D2H/H2D` | FlexKV 原生 region submit 至 wait_all_streams 完成；请求描述构造、初始化和校验不计时 |
| `H2REMOTE/REMOTE2H` | 主进程向所有 L3 客户端发命令至收到全部完成，包含 IPC、FlexKV 包装及 SDK 调用 |
| `client_ns` | 子进程 FlexKV batch_put/get 的调用时间；PUT 包含 exists 查询 |
| `sdk_calls` | 每个实际 batch_put_from/get_into 的开始/结束、字节和对象数，能直接看到拆批 |
| `sdk_ns` | 一个客户端从首个 SDK 子调用开始至最后一个结束的跨度；含子批间隙 |
| `sdk_busy_ns` | 子调用耗时相加，并行时会重叠，不能拿它当端到端时间 |
| `GPU_TO_L3_STAGE_SUM` | 本轮 D2H 完成时间 + PUT 完成时间之和，排除中间计数采集；不是调度流水吞吐 |
| `L3_TO_GPU_STAGE_SUM` | 本轮 GET 完成时间 + H2D 完成时间之和，排除校验等中间开销 |

四组布局为主 KV 78×41984、主 indexer 21×8448、draft KV 8×41984、draft indexer 8×8448 B/页。八卡 owner 按已验证 GLM 日志中的层分配；只向所属 GPU 搬运。这里用逐层 RegionRequest 在一个 batch 内提交，测试无 eventfd 消费者和计算重叠，也不包括生产进程间 GDR owner attach/RPC。

63 页就是 4032 token、242,903,808 B。用不同 key 重复写，必须实际进入 SDK PUT；全 exists 跳过会被判为错误。GET 前清零目标，SDK 必须返回每对象完整长度，然后比对整页 SHA256；有 GPU 时再逐区域比对恢复结果。GPU 单向测试默认检查首尾轮，可加 `--verify-every-round`。校验不计时，但仍可能影响下一轮的 CPU cache/内存状态，因此这些是有校验的合成载荷结果。

每个 GPU-count case 独立进程，避免前一案例 CUDA 状态污染；GDR 必须有实际 route counter 且无 init fallback，否则失败。可用 `--backend ce` 做另一组显式对照，不自动降级。默认每个 IPC 等待 300 秒、每个完整 GPU case 3600 秒超时；超时终止该测试创建的进程组，保留失败文件。

## 输出与判断

入口打印目录：`scripts/flexkv/logs/glm52-1p/tier-benchmark/<时间-随机串>/`。也可传 `--output /绝对目录`。拷贝整个目录即可。

- `summary.csv`：每种方向/数据量/并发的样本数、P50/P95 耗时、总字节÷总时间的 GB/s；不含 warmup。
- `gpu-*/samples.jsonl`：逐轮指标、各客户端 SDK 子批端点、NIC 计数差值。
- `runtime.json`：实际 Torch/host_buffer 路径、共享内存布局、NUMA maps、亲和性和相关环境。
- `l3-*-runtime.json`：各进程实际 Store 配置、SDK 文件路径、加载库、NUMA 和亲和性。
- `benchmark.log`、各 L3 worker log：初始化、注册、SDK 警告/错误。
- `SUCCESS`：全部案例完成；`FAILED.txt`：失败原因。部分 summary 不代表整轮通过。

NIC 数据计数是 sysfs 原始 `port_xmit_data/port_rcv_data`（通常单位为 4 octets），不是脚本已经换算好的字节；bond/物理端口可能重复计数，不要盲目求和。有其他流量时不能归因于测试。指标采集位于 GPU 传输之外，仅包围 Store 调用；读不到计数时记录为空，不猜测网络路径。

建议按以下顺序判断：

1. 同机 memcpy 很快，L3 小批次慢、大批次明显快：看 SDK 子批数和固定控制开销。
2. 取消 64 MiB 限制后 PUT 明显提高：原串行拆批是实际限制之一。
3. lanes 增加明显提高总 GB/s：单客户端的并发/提交结构是限制之一；继续对照默认 NIC 与显式 NIC。
4. page-shards 比 whole-pages 快：对象形态和每客户端批量影响明显；不能据此宣称 HiCache 同比例更快。
5. NIC 几乎不增长而 L3 能完成：支持数据未主要经过可见 NIC 的解释，但需 SDK 路由日志确认内存直拷/其他接口，不能只凭计数为零断言本地 memcpy。
6. lanes 增加仍不提高：检查 NUMA/内存带宽、SDK 本机路径与 Store 分配/元数据服务，而不是继续机械增加 GPU 数。

本地已验证参数、分片覆盖、payload 计算、SDK 短读/失败检测、并行分发顺序、超时传播及汇总口径；当前开发机没有 some special machine types and scenarios/CUDA/Mooncake 服务，真实数值待容器运行后分析。
