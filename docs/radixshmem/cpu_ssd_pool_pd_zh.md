# 基于 radixshmem 的跨节点 CPU + SSD KV 池（PD 分离）部署指南

本文介绍如何用 FlexKV + SGLang 部署：所有 prefill（P）和 decode（D）节点在集群范围内共享一个 CPU KV 池和一个 SSD KV 池。
英文版：[cpu_ssd_pool_pd.md](cpu_ssd_pool_pd.md)。

## 1. 能做什么

- **CPU 层**：每个节点的主机内存由本节点的 `radix-server`（数据模式）管理成 slot 池。任意节点上的请求都能命中其他节点
  CPU 里的前缀，并通过 RDMA 拉到本地。
- **SSD 层**：每个节点的 SSD 块存在本地文件里，由每节点第二个 `radix-server`（只做索引）索引。所有节点看到同一个全局
  SSD 索引，一个节点写入的前缀，其他节点也能读。
- **PD**：P 和 D 都接入 FlexKV。D 在请求结束时（也可以在 decode 过程中每 N 个 token）把生成的 KV 存进来，下一轮请求
  无论落到哪个 P 都能复用。
- **整段预取**：请求排队时，FlexKV 把整个已缓存前缀从本地 SSD、对端 CPU、对端 SSD 预取到本地 CPU；调度时 GPU 加载
  （H2D）只读本地 CPU。
- **SSD 全局去重写入**（可选，推荐）：写 SSD 前先查全局 SSD 索引。对端已有整段就不写；对端有前半段，就只把剩余部分写到
  那个对端的 SSD。避免同一份 KV 在多个节点的 SSD 上各存一份（例如 P 和 D 都存同一段对话）。

```
            节点 A (P)                                   节点 B (D)
 GPU ─H2D/D2H─ CPU 池 (radix-server, 数据) ◄─RDMA 拉取─► CPU 池 (radix-server, 数据)
                 │  ▲                                          │  ▲
             H2DISK│  │DISK2H                                 ...
                 ▼  │
            SSD 文件 (radix-server, 索引) ◄─NIXL 读/写─► SSD 文件 (radix-server, 索引)

 全局索引：CPU 集群（etcd + RDMA）          SSD 集群（etcd + RDMA）
```

FlexKV 日志里的传输类型：`H2D`/`D2H`（GPU 与本地 CPU）、`H2DISK`/`DISK2H`（本地 CPU 与本地 SSD）、对端 CPU 拉取
（radixshmem `pull_async`）、`PEERSSD2H`（对端 SSD -> 本地 CPU）、`H2PEERSSD`（本地 CPU -> 对端 SSD，去重写入）。

## 2. 版本

| 组件 | 分支 | 说明 |
|---|---|---|
| FlexKV | `linhu-nv/FlexKV` `linhu/rsm-ssd-pool` | 本分支 |
| SGLang | `linhu-nv/sglang` `linhu/flexkv-pd-rsm-ssd` | FlexKV PD decode 接入、decode 分段存储，以及两个修复（见第 8 节） |
| radixshmem | `feat/pd-support`（验证于 `5e32312`） | 带 RDMA 和 Mooncake 传输后端编译 |
| NIXL | 1.4 或 1.5 Python wheel，含 UCX 和 POSIX 插件 | 用于对端 SSD 读写 |

所有节点的 NIXL / Mooncake / UCX 版本要一致。

## 3. 前置条件

- 节点之间有 RDMA 网卡（InfiniBand 或 RoCE）。RoCE 要设置 GID index（radixshmem `--gid-idx 3`、Mooncake
  `MC_GID_INDEX=3`，UCX 没选对时设 `UCX_IB_GID_INDEX`）。
- 一个所有节点都能访问的 etcd。
- 每个节点有本地 NVMe 文件系统放 SSD 文件。
- 一个**所有节点共享**的目录，放 FlexKV 的 SSD manifest 和 NIXL 联系文件（`FLEXKV_PEER_SSD_MANIFEST_DIR`），比如 NFS。
  每个节点启动时在里面写几个小 JSON 文件，对端靠它们找到本节点的 SSD 文件和 NIXL 端点。没有共享文件系统时，需要在节点
  之间同步这个目录。
- 时钟同步（NTP）：NIXL 请求带截止时间。
- 编译 radixshmem（`cd radixshmem/python && pip install -e .`，RDMA、gRPC、Mooncake 依赖见其 README）和 FlexKV
  （本仓库 `pip install -e .`，需要 C++20 编译器）。

## 4. 部署

启动顺序：etcd → 每个节点的两个 radix-server → 每个节点的 SGLang（P 和 D）→ router。下面的例子是 2 个节点、每节点 4 卡：
节点 A 是 prefill（`10.0.0.10`），节点 B 是 decode（`10.0.0.20`），etcd 在 `10.0.0.10:2379`。`N` 是集群总节点数（P + D）。

### 4.1 etcd

```bash
etcd --name kv --data-dir /var/lib/etcd-kv \
  --listen-client-urls http://10.0.0.10:2379 --advertise-client-urls http://10.0.0.10:2379
```

### 4.2 radix-server（每节点两个）

两个集群用同一个 etcd，但 `--cluster-id` 不同。同一节点的两个 server 用同一个 `--node-name`，FlexKV 也用这个名字发布
manifest。

```bash
# 节点 A 上（每个节点都要起，换成各自的 --node-name / --rpc-address / 网卡）
# CPU 池：数据模式。--data-bytes 就是本节点 CPU 层容量。
# SWA 模型（如 DeepSeek-V4.x）必须设 --swa-ratio；其他模型为 0。
radix-server --name /kv_cpu --data-bytes 200G --swa-ratio 0.2 \
  --expected-min-nodes 2 --bootstrap-timeout 3600 \
  --registry etcd://10.0.0.10:2379 --cluster-id kv_cpu --node-name nodeA \
  --rpc-address 10.0.0.10 --index-dev mlx5_0 --gid-idx 3 \
  --transfer-dev mlx5_0 --transfer-backend mooncake

# SSD 索引：只做索引（不设 --data-bytes）。slot id 就是 FlexKV 的 SSD 块号。
radix-server --name /kv_ssd \
  --expected-min-nodes 2 --bootstrap-timeout 3600 \
  --registry etcd://10.0.0.10:2379 --cluster-id kv_ssd --node-name nodeA \
  --rpc-address 10.0.0.10 --index-dev mlx5_0 --gid-idx 3
```

（`radix-server` 即 `python -m radixshmem.cli`。）SSD 索引 server 的几何参数（SSD 块数、SWA slot 数）由本节点的 FlexKV
进程下发，所以 `N` 个节点的 FlexKV 都启动之后 SSD 集群才就绪。这套方案里 decode 节点的 radix-server **不要**加
`--decode`：该模式（prefill 淘汰时回写到 decode）没有和本方案一起验证过。

### 4.3 FlexKV 配置文件

```yaml
# /etc/flexkv/nodeA.yaml
cpu_cache_gb: 8              # radixshmem 模式下不生效：CPU 容量 = radix-server --data-bytes
ssd_cache_gb: 1000           # 本节点 SSD 层容量
ssd_cache_dir: /nvme/flexkv  # 本地 NVMe 目录，放 SSD 文件
swa:
  num_ssd_slots: 4096        # SSD 上的 SWA slot 数（仅 SWA 模型）
```

### 4.4 环境变量（每个 SGLang 实例）

```bash
# radixshmem 两层
export FLEXKV_ENABLE_RADIXSHMEM=1
export FLEXKV_CPU_LAYOUT=BLOCKFIRST
export FLEXKV_RADIXSHMEM_SERVER_NAME=/kv_cpu
export FLEXKV_RADIXSHMEM_SSD_SERVER_NAME=/kv_ssd
export FLEXKV_PEER_SSD_MANIFEST_DIR=/shared/flexkv-manifests

# 对端 SSD 数据面走 NIXL
export FLEXKV_REMOTE_SSD_READER=flexkv.transfer.workers.peer_ssd_nixl:make_reader
export FLEXKV_NIXL_SSD_IP=10.0.0.10        # 对端能访问到的本节点 IP
export FLEXKV_NIXL_SSD_PORT=5810           # FULL 池端口；SWA 池用 +1
export FLEXKV_NIXL_UCX_DEVICES=mlx5_0
export UCX_TLS=rc,ud,self,cuda_copy

# SSD 全局去重（推荐）
export FLEXKV_SSD_WRITE_MODE=global_dedup

# 进程 / 启动
export FLEXKV_ENABLE_MPS=0
export FLEXKV_SERVER_RECV_PORT=ipc:///tmp/flexkv_nodeA   # 同一台机器上每个实例不同
export FLEXKV_READY_TIMEOUT_S=3600                      # 首次启动慢（加载模型、集群汇合）
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export MC_GID_INDEX=3                                   # RoCE

# 仅 decode 节点（可选）：decode 过程中每生成 512 个 token 也存一次
export SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE=512
```

### 4.5 SGLang（DeepSeek-V4.1，attention DP4 + MoE EP4）

```bash
# prefill（节点 A）
python3 -m sglang.launch_server --model-path /models/DeepSeek-V4.1 --trust-remote-code \
  --tp 4 --dp 4 --enable-dp-attention --ep-size 4 \
  --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 \
  --page-size 256 --mem-fraction-static 0.93 --chunked-prefill-size 4096 \
  --no-enable-decoder-swa-bounded-replay --enable-cache-report \
  --enable-flexkv --flexkv-config-file /etc/flexkv/nodeA.yaml \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3 --disaggregation-bootstrap-port 8995 \
  --host 0.0.0.0 --port 30000

# decode（节点 B）：相同，改成 --disaggregation-mode decode，去掉 bootstrap 端口。
# decode 上开 --enable-flexkv 会自动打开 --disaggregation-decode-enable-radix-cache。
```

然后按常规方式启动 SGLang PD router，传入 prefill（`http://10.0.0.10:30000`，bootstrap 端口 8995）和 decode 端点。
其他模型用法相同，去掉 DSv4 专用参数即可。

### 4.6 检查是否生效

1. 所有 radix-server 日志显示 ready；每个 FlexKV 实例日志显示已 attach CPU 和 SSD 两个 radix-server，用 NIXL 时还会显示
   对端 SSD 端点已开始服务。
2. `ls $FLEXKV_PEER_SSD_MANIFEST_DIR` 能看到每个节点的 `<node>.json` 和 `<node>.nixl-full.json`（以及 `nixl-swa`）。
3. 向 P 发一个长 prompt，把它挤出 P 的 CPU（或换一个 P），再发一次：SGLang 报告 cached token，FlexKV 日志出现
   `DISK2H` / `PEERSSD2H` / 对端拉取，回答与第一次一致。
4. 指标：`flexkv_py_transfer_blocks_total{transfer_type=...}` 中正在使用的传输类型会增长。

## 5. 配置项

| 变量 | 默认 | 含义 |
|---|---|---|
| `FLEXKV_RADIXSHMEM_SSD_SERVER_NAME` | 空 | 本节点只做索引的 SSD radix-server 的 `--name`；radixshmem 模式下 `ssd_cache_gb > 0` 时必须设置 |
| `FLEXKV_PEER_SSD_MANIFEST_DIR` | 空 | 放 SSD manifest 和 NIXL 联系文件的共享目录；对端 SSD 必需 |
| `FLEXKV_REMOTE_SSD_READER` | 共享文件读取 | 对端 SSD 传输的 `module:factory`。`flexkv.transfer.workers.peer_ssd_nixl:make_reader` 为 NIXL；默认实现直接读对端 SSD 文件，只适用于 SSD 文件在共享文件系统上的情况 |
| `FLEXKV_NIXL_SSD_IP` / `FLEXKV_NIXL_SSD_PORT` | 自动 / 5600 | 对端访问本节点 NIXL 端点的地址（FULL 池端口；SWA 用 +1） |
| `FLEXKV_NIXL_UCX_DEVICES` | UCX 默认 | NIXL（UCX）使用的网卡 |
| `FLEXKV_NIXL_SSD_TIMEOUT_S` | 30 | 单个对端 SSD 请求的截止时间 |
| `FLEXKV_NIXL_SSD_RETRY_S` | 10 | 对端端点失败后，重试前的退避时间 |
| `FLEXKV_NIXL_SSD_DIRECT_READ` | 1 | 服务端用 O_DIRECT 读 SSD（不是 4 KiB 对齐的块读对齐区间） |
| `FLEXKV_NIXL_POSIX_URING` | true | NIXL POSIX 插件使用 io_uring |
| `FLEXKV_SSD_WRITE_MODE` | `local` | `global_dedup` 开启 SSD 全局去重 |
| `FLEXKV_SSD_DEDUP_LOCAL_FRACTION` | 0 | 按此比例让部分 SSD 写入跳过全局查询、直接写本地（分散写入压力） |
| `FLEXKV_SSD_DEDUP_PEER_BACKOFF_S` | 10 | 远程写失败后，这段时间内不再写该对端，改写本地 |
| `FLEXKV_RADIXSHMEM_SSD_READY_TIMEOUT_S` | 3600 | FlexKV 等待 SSD 集群就绪的时长 |
| `SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE` | 0 | D 每生成 N 个 token（按页对齐）也存一次；0 表示只在结束时存。SWA 模型 decode 中途只存 FULL KV |

## 6. 容量与运维

- **CPU 池大小**：CPU 池同时是 SSD 读、对端拉取、SSD 写的中转区，至少要 *并发请求数 × 每请求新增块数*。太小时 FlexKV
  会静默跳过存储和预取（不报错，但命中率会崩）。
- **重启规则**：节点的两个 radix-server 要和本节点的 SGLang/FlexKV 一起重启。单独重启 radix-server 时，它会等 FlexKV
  重新下发配置，在 FlexKV 重启之前该节点的 SSD 层一直不可用。FlexKV 崩溃后也要一起重启 radix-server（崩溃进程 pin 住的
  slot 不会被回收）。
- **首次启动**：SSD 集群要等所有节点的 FlexKV 都配置完 SSD 索引 server 才就绪；节点加载模型快慢不一时，保留 4.4 中的
  启动超时设置。
- **单机多实例**：每个实例要有独立的 `FLEXKV_SERVER_RECV_PORT`、`FLEXKV_NIXL_SSD_PORT`（每个实例占两个端口）、
  radix-server 名字，开 layerwise 时还要有独立的 `FLEXKV_LAYERWISE_EVENTFD_SOCKET`。
- **故障**：对端在截止时间内没响应会被退避；读会退回重新计算，去重写会退回本地写。NIXL 对端 SSD worker 进程死掉会自动
  重启（每个 engine 最多 8 次）。
- DeepSeek-V4.x 不支持 `--disaggregation-enable-kv-checksum`。

## 7. 验证与性能（摘要）

在 H20 节点上验证：Qwen3-8B（1P1D）、DeepSeek-V4.1（1P1D，以及两台机器跨机 2P1D，attention DP4 + MoE EP4）。场景包括
1 小时和 4 小时长跑（随机路由、会话固定路由）、20% 客户端中途取消、热门前缀突发、重启 P/D、暂停 NIXL 端点、64k-100k
长上下文、layerwise 开/关、强制 retraction、decode 分段存储、去重开/关。全部 0 个意外请求错误，needle 答案全对，命中缓存
的输出无乱码。

| 指标 | 结果 |
|---|---|
| 跨节点 SSD 重复块，local vs `global_dedup` | DeepSeek-V4.1 2P1D 60.6% -> 1.7%；Qwen3-8B 1P1D 37-41% -> 0-14% |
| SSD 写入量 | `global_dedup` 下减少 34%-60% |
| 吞吐 / TTFT / TPOT，`global_dedup` vs local | 差距 2% 以内（DeepSeek-V4.1 1P1D，并发 8-32，3 轮交替） |
| 对端 SSD 读（NIXL，跨机，1.67-9.4 MB 块） | 6-27 GB/s，随请求大小和并发变化 |
| TTFT，DeepSeek-V4.1，16k prompt（5 次中位数） | 重新计算 4572 ms；本地 CPU 189；本地 SSD 206；对端 CPU 262；对端 SSD 197 |
| TTFT，DeepSeek-V4.1，64k prompt（5 次中位数） | 重新计算 20390 ms；本地 CPU 217；本地 SSD 262；对端 CPU 222；对端 SSD 228 |

## 8. 已知限制

- 单独重启的 radix-server 要等 FlexKV 重启后才能重新加入（第 6 节）。
- 开 `global_dedup` 时，P 和 D 同时写同一段新前缀仍可能都写一份（百分之几的重复）；前缀会集中在最先存它的节点上，
  因此有一部分 SSD 读会从本地变成对端。
- SGLang 分支还修了本方案下发现的两个 SGLang 问题：分块 prefill 在 FlexKV 加载回 KV 之后被取消时 SWA 页重复释放；
  PD retract 重新 bootstrap 复用原请求的 rid 和 bootstrap room（上游问题，不开 FlexKV 也会出现）。
