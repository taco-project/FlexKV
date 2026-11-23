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
- **attention-DP 直连**（可选，`FLEXKV_DP_DIRECT=1`）：每个 DP rank 自带一个 FlexKV 引擎，直接访问 radixshmem 索引，
  不再经过每节点一个的 KVServer 排队。
- **节点内共享 SSD 文件**：本节点所有 FlexKV 引擎（直连的各 DP rank、同机多实例）共用 `ssd_cache_dir` 下同一组 SSD 文件
  和同一个 SSD 索引；FlexKV 重启后（radix-server 不重启）SSD 上的缓存仍然有效。radix-server 只管索引，不碰 SSD 文件。
- **KV 事件上报给 dynamo 路由**（可选）：CPU 池 radix-server 发布本节点的块放置事件，dynamo KV router 据此知道每个节点
  CPU 池里有哪些前缀，优先把请求路由到已持有前缀的节点（`--shared-cache-type radixshmem`）。

```
            节点 A (P)                                   节点 B (D)
 GPU ─H2D/D2H─ CPU 池 (radix-server, 数据) ◄─RDMA 拉取─► CPU 池 (radix-server, 数据)
                 │  ▲                                          │  ▲
             H2DISK│  │DISK2H                                 ...
                 ▼  │
            SSD 文件 (radix-server, 索引) ◄─NIXL 读/写─► SSD 文件 (radix-server, 索引)

 全局索引：CPU 集群（etcd + gRPC + Redis）    SSD 集群（etcd + gRPC + Redis）
```

FlexKV 日志里的传输类型：`H2D`/`D2H`（GPU 与本地 CPU）、`H2DISK`/`DISK2H`（本地 CPU 与本地 SSD）、对端 CPU 拉取
（radixshmem `pull_async`）、`PEERSSD2H`（对端 SSD -> 本地 CPU）、`H2PEERSSD`（本地 CPU -> 对端 SSD，去重写入）。

## 2. 版本

| 组件 | 分支 | 说明 |
|---|---|---|
| FlexKV | `linhu-nv/FlexKV` `linhu/rsm-ssd-pool-flexkv-owned` | 本分支（上游 main + 本方案） |
| SGLang | `linhu-nv/sglang` `linhu/flexkv-1116-xing-pd-ssdpool-kvevent`（上游 main + XingLiu1 `agent/flexkv-dsv4-main`） | FlexKV PD decode 接入、decode 分段存储、hybrid 批量 layerwise 恢复、调用方块哈希、非 layerwise 恢复前的同步，以及修复（见第 9 节） |
| radixshmem | `linhu-nv/radixshmem` `linhu/index-id`（基于 `main`） | KV 事件上报、索引 id（`NodeInfo.index_id`）；带 RDMA、gRPC 和 Mooncake 传输后端编译 |
| dynamo（可选） | `linhu-nv/dynamo` `linhu/radixshmem-shared-cache-1116`（基于 `main`） | 只在用 dynamo KV router + KV 事件时需要 |
| NIXL | 1.4 或 1.5 Python wheel，含 UCX 和 POSIX 插件 | 用于对端 SSD 读写 |

所有节点的 NIXL / Mooncake / UCX 版本要一致。

## 3. 前置条件

- 节点之间有 RDMA 网卡（InfiniBand 或 RoCE）。RoCE 要设置 GID index（Mooncake `MC_GID_INDEX=3`，UCX 没选对时设
  `UCX_IB_GID_INDEX`）。radixshmem 的索引默认走 gRPC + Redis RHT，不用 RDMA；只有改用 RDMA 索引时才需要
  `RADIXSHMEM_INDEX_DEV` / `RADIXSHMEM_INDEX_GID_IDX`（以前的 `--index-dev` / `--gid-idx` 参数已删除）。
- 一个 Redis Cluster（radixshmem 的全局路由表 RHT，`RADIXSHMEM_RHT_BACKEND=redis`，部署见 radixshmem
  `docs/` 下的 Redis Cluster 指南）。
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
# 每个 radix-server 进程都要带 Redis RHT 的环境变量（只在 rank 0 上生效，其他节点从 etcd 读）：
export RADIXSHMEM_RHT_BACKEND=redis RADIXSHMEM_REDIS_ADDRS=10.0.0.10:7001,10.0.0.10:7002,10.0.0.10:7003
radix-server --name /kv_cpu --data-bytes 200G --swa-ratio 0.2 \
  --expected-min-nodes 2 --bootstrap-timeout 3600 \
  --registry etcd://10.0.0.10:2379 --cluster-id kv_cpu --node-name nodeA \
  --rpc-address 10.0.0.10 --rpc-port 24610 \
  --transfer-dev mlx5_0 --transfer-backend mooncake \
  --kv-events-endpoint 'tcp://*:5691'            # 可选：KV 事件给 dynamo 路由（4.7）

# SSD 索引：只做索引（不设 --data-bytes）。slot id 就是 FlexKV 的 SSD 块号，SSD 文件由 FlexKV 创建。
radix-server --name /kv_ssd \
  --expected-min-nodes 2 --bootstrap-timeout 3600 \
  --registry etcd://10.0.0.10:2379 --cluster-id kv_ssd --node-name nodeA \
  --rpc-address 10.0.0.10 --rpc-port 24611
```

（`radix-server` 即 `python -m radixshmem.cli`。）SSD 索引 server 的几何参数（SSD 块数、SWA slot 数）由本节点的 FlexKV
进程下发，所以 `N` 个节点的 FlexKV 都启动之后 SSD 集群才就绪。这套方案里 decode 节点的 radix-server **不要**加
`--decode`：该模式（prefill 淘汰时回写到 decode）没有和本方案一起验证过。

### 4.3 FlexKV 配置文件

```yaml
# /etc/flexkv/nodeA.yaml
cpu_cache_gb: 8              # radixshmem 模式下不生效：CPU 容量 = radix-server --data-bytes
ssd_cache_gb: 1000           # 本节点 SSD 层容量
ssd_cache_dir: /nvme/flexkv  # 本地 NVMe 目录，放 SSD 文件（本节点所有 FlexKV 共用）
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

# layerwise 传输 + 批量恢复（推荐）
export FLEXKV_ENABLE_LAYERWISE_TRANSFER=1
export FLEXKV_BATCH_LAYERWISE_RESTORE=1
export FLEXKV_LAYERWISE_EVENTFD_SOCKET=/tmp/flexkv_lw_nodeA.sock   # 同一台机器上每个实例不同

# 进程 / 启动
export FLEXKV_SERVER_RECV_PORT=ipc:///tmp/flexkv_nodeA   # 同一台机器上每个实例不同
export FLEXKV_READY_TIMEOUT_S=3600                      # 首次启动慢（加载模型、集群汇合）
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export MC_GID_INDEX=3                                   # RoCE

# DeepSeek-V4.1：SWA 快照在 chunk 边界按 4 页对齐，配合默认打开的 decoder SWA bounded replay（吞吐更高）
export SGLANG_FLEXKV_SWA_GRID_PAGES=4

# 仅 decode 节点（可选）：decode 过程中每生成 512 个 token 也存一次
export SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE=512
```

### 4.5 SGLang（DeepSeek-V4.1，attention DP4 + MoE EP4）

```bash
# prefill（节点 A）
python3 -m sglang.launch_server --model-path /models/DeepSeek-V4.1 --trust-remote-code \
  --tp 4 --dp 4 --enable-dp-attention --ep-size 4 \
  --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 \
  --page-size 256 --mem-fraction-static 0.92 --chunked-prefill-size 4096 \
  --enable-cache-report \
  --enable-flexkv --flexkv-config-file /etc/flexkv/nodeA.yaml \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3 --disaggregation-bootstrap-port 8995 \
  --host 0.0.0.0 --port 30000

# decode（节点 B）：相同，改成 --disaggregation-mode decode、--mem-fraction-static 0.93（见第 5 节），去掉 bootstrap 端口。
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

### 4.7 可选：attention-DP 直连、KV 事件路由（dynamo）

两项可以单独开，也可以一起开。

**直连**（`FLEXKV_DP_DIRECT=1`，每个 SGLang 实例的环境变量）：attention DP 下每个 DP rank 一个 FlexKV 引擎，不再经过
每节点一个的 KVServer。一个 DP 实例跨多个节点时每个节点要跑自己的 DP rank。对端 SSD 的 NIXL 端点每个 rank 一个，从 `FLEXKV_NIXL_SSD_PORT` 起每个 rank
占两个端口（第 k 个 rank 用 `PORT+2k`、`PORT+2k+1`）。

**KV 事件路由**：需要 dynamo 分支，用 `python -m dynamo.sglang` 起 worker、`python -m dynamo.frontend` 起路由。

```bash
# CPU 池 radix-server 加 --kv-events-endpoint 'tcp://*:5691'（4.2）

# 每个 dynamo.sglang worker 的环境变量
export FLEXKV_HASH_SCHEME=sglang-sha256-i64-v1                  # FlexKV 用 SGLang 的页哈希给块命名，路由才能算出同样的 key
export DYN_RADIXSHMEM_KV_EVENTS_ENDPOINT=tcp://10.0.0.10:5691   # 本节点 CPU 池 radix-server 的事件端点（router 能连到的地址）
# 一个 attention-DP 实例跨多个节点时，改用 DYN_RADIXSHMEM_KV_EVENTS_ENDPOINTS=<按节点 rank 顺序，逗号分隔>

python3 -m dynamo.sglang <与 4.5 相同的 SGLang 参数> \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:24710","topic":"kv"}'

# 路由（frontend）
export DYN_RADIXSHMEM_REMOTE_HIT_WEIGHT=0.9   # 只在其他节点 CPU 池里的块，按本节点块的这个比例计分（默认 0.5）
python3 -m dynamo.frontend --http-port 8000 --router-mode kv --router-kv-events \
  --router-track-prefill-tokens --shared-cache-type radixshmem
```

路由日志里出现 `subscribed to radixshmem` 和 `worker nodes`，并有 `shared-cache lookup ... held=N`（N>0）就说明在用。
`FLEXKV_HASH_SCHEME` 写进了 radixshmem 的 geometry：同一个集群里所有 FlexKV 必须一致，否则 attach 时报错。SSD 索引
radix-server 不要加 `--kv-events-endpoint`（路由只看 CPU 池）。

## 5. 配置指南

分三层：先用 **最小可用配置** 跑通，再按 **推荐配置** 部署，最后按 **场景** 调整。每个开关都说明了它的代价和怎么观察效果。

### 5.1 最小可用配置

第 4 节的部署步骤本身就是最小可用配置：每节点两个 radix-server、FlexKV 配置文件、4.4 的环境变量（不含可选项）、SGLang 加 `--enable-flexkv`。跑通的标志见 4.6。

### 5.2 推荐配置

| 配置 | 推荐值 | 说明 |
|---|---|---|
| CPU 池 `radix-server --data-bytes` | 主机内存允许的范围内尽量大，且不小于 *并发请求数 × 每请求新增块数* | CPU 池同时是 SSD 读、对端拉取和 SSD 写的中转区。不够时 FlexKV 静默跳过存储和预取（不报错，命中率下降），日志会有 `skipped: CPU ... slots available` / `no CPU room to stage` |
| SWA 池占比 `--swa-ratio`（SWA 模型） | 0.2 | 会话很多、SWA 槽不足时调大 |
| SSD 容量 `ssd_cache_gb` / `swa.num_ssd_slots` | 按盘容量 / 4096 | — |
| 对端 SSD 传输 `FLEXKV_REMOTE_SSD_READER` | `flexkv.transfer.workers.peer_ssd_nixl:make_reader` | 只有 SSD 文件在共享文件系统上时才可以用默认实现 |
| NIXL 网卡 `FLEXKV_NIXL_UCX_DEVICES` / `UCX_TLS` | 与 CPU 池同 NUMA 的 RDMA 网卡 / `rc,ud,self,cuda_copy` | — |
| SSD 全局去重 `FLEXKV_SSD_WRITE_MODE` | `global_dedup` | DeepSeek-V4.1 跨节点 SSD 重复 60.6% -> 1.7%，SSD 写入 -60%，吞吐/TTFT/TPOT 差距 2% 以内 |
| 部分写入直接写本地 `FLEXKV_SSD_DEDUP_LOCAL_FRACTION` | 0 | 见下方场景调整 |
| layerwise 传输 `FLEXKV_ENABLE_LAYERWISE_TRANSFER` | 1 | 加载与计算按层重叠；同机多实例时每个实例设不同的 `FLEXKV_LAYERWISE_EVENTFD_SOCKET` |
| 批量 layerwise 恢复 `FLEXKV_BATCH_LAYERWISE_RESTORE` | 1 | 同一 batch 的多个请求合并成少量传输，而不是逐个请求串行恢复。TP>1 且命中较短时最多快 1.8 倍，其他情况持平 |
| decode 分段存储 `SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE` | 输出较短：0；agentic / 长输出（数千 token）：512-1024 | 让 D 生成的 KV 更早可被复用，结束时只存尾部；代价是 decode 期间多一些 D2H |
| 对端退避 `FLEXKV_SSD_DEDUP_PEER_BACKOFF_S` / `FLEXKV_NIXL_SSD_RETRY_S` | 10 / 10 | 一般不改 |
| SSD O_DIRECT | 自动（默认） | 按设备对齐（NVMe 一般 512 B）读写都绕过页缓存。块大小是对齐值的整数倍时生效：DeepSeek-V4.1 的 SWA 块（5990400 B）以及 TP 部署下的 FULL 块（1671680 B）本地冷读 5 -> 18-22 GB/s；attention-DP 部署下 FULL 块是 417920 B（不满足 512 对齐），仍走页缓存（对端读用对齐区间 O_DIRECT，不受影响）。排查问题时可用 `FLEXKV_NIXL_SSD_DIRECT_READ=0`、`FLEXKV_NIXL_SSD_DIRECT_ALIGN` 覆盖 |
| radixshmem CPU 层拉取后去重（`RADIXSHMEM_DISABLE_AUTO_DEDUP`） | 默认（开） | 已在默认值下验证 |
| radix-server `--decode`（prefill 淘汰写回 decode） | 不开 | 尚未与本方案一起验证 |
| FlexKV 进程模式 | 请求率高、prefill 偏重时用直连 `FLEXKV_DP_DIRECT=1`；否则默认（`dp_size>1` 时每节点一个 KVServer，各 DP rank 为 client） | 直连去掉了 KVServer 的排队：Qwen3-8B DP8 上 FlexKV 每次调用 p90 从 3-11 ms 降到 0.3 ms，prefill 偏重时吞吐 +3%~10%，decode 偏重时持平。直连和同机多实例（`FLEXKV_INSTANCE_NUM>1`）的各引擎共用本节点的 SSD 文件，`ssd_cache_gb` / `ssd_cache_dir` 要一致 |
| 路由 | 能做会话粘性就做 | 粘性路由下跨节点流量约为随机路由的 1/3，对端 SSD 读接近 0 |
| 启动超时 `FLEXKV_READY_TIMEOUT_S` / `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` / `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | 3600 | 首次启动要等所有节点就绪 |
| SGLang（DeepSeek-V4.1） | `--mem-fraction-static 0.92 --chunked-prefill-size 4096 --page-size 256`，保持 decoder SWA bounded replay 默认打开（不要加 `--no-enable-decoder-swa-bounded-replay`），`SGLANG_FLEXKV_SWA_GRID_PAGES=4` | bounded replay 对吞吐影响很大，但运行时多用约 1 GB 显存，所以 `--mem-fraction-static` 从 0.93 降到 0.92（H20 TP4）。PD 的 decode 节点例外：H20 上加载权重后剩余显存比 prefill 少约 1 GB，0.92 时 `--max-running-requests 64` 的 SWA 池放不下，启动报 `SWA pool cap ... leaves no room`，decode 用 0.93（已验证运行正常）。不要开 `--disaggregation-enable-kv-checksum`（DeepSeek-V4.x 不支持） |

### 5.3 按场景调整

- **某几个节点 SSD 写入或占用明显偏高**（热门前缀集中在最先存它的节点）：把 `FLEXKV_SSD_DEDUP_LOCAL_FRACTION` 调到 0.1-0.3，让这一比例的写入不做全局查询、直接写本地。代价是这部分会产生重复。
- **网络带宽紧张，宁可多占盘也不想多读对端 SSD**：`FLEXKV_SSD_WRITE_MODE=local`。
- **CPU 命中率低于预期**：先看日志里 CPU 槽不足的告警，加大 `--data-bytes` 或降低并发。
- **长输出 / 多轮 agentic**：打开 decode 分段存储（512-1024）。

### 5.4 怎么观察

- 传输量：`flexkv_py_transfer_blocks_total{transfer_type=...}`，关注 `DISK2H`、`PEERSSD2H`（对端 SSD 读）、`H2PEERSSD`（去重写到对端）。
- 去重统计：每分钟一行 `[FlexKV-IO] operation=ssd_dedup act=stats`，字段 `puts_skipped`（整段已在对端）、`puts_remote`（剩余部分写到对端）、`fallback_*`（退回本地写）、`remote_write_failed`。
- 命中：SGLang 的 `cached_tokens`（`--enable-cache-report`）。
- 健康：radix-server 日志的 health；FlexKV 日志中 `transfer failed`、worker 死亡或重启（带 class、pid、exitcode）。

## 6. 配置项

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
| `FLEXKV_NIXL_SSD_DIRECT_ALIGN` | 自动探测 | NIXL 端点访问 SSD 文件的 O_DIRECT 对齐（一般自动探测，NVMe 多为 512） |
| `FLEXKV_ENABLE_LAYERWISE_TRANSFER` | 0 | 按层加载 KV，与计算重叠 |
| `FLEXKV_LAYERWISE_EVENTFD_SOCKET` | — | layerwise 传输的 eventfd socket；同一台机器上每个实例不同 |
| `FLEXKV_BATCH_LAYERWISE_RESTORE` | 0 | 把同一批准入请求的 layerwise 恢复合并执行 |
| `FLEXKV_RADIXSHMEM_SSD_READY_TIMEOUT_S` | 3600 | FlexKV 等待 SSD 集群就绪的时长 |
| `SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE` | 0 | D 每生成 N 个 token（按页对齐）也存一次；0 表示只在结束时存。SWA 模型 decode 中途只存 FULL KV。PD decode 上从 prompt 末尾开始计数，第一次存储带上 prompt 和新输出 |
| `FLEXKV_DP_DIRECT` | 0 | 1 = attention DP 下每个 DP rank 一个 FlexKV 引擎（见 4.7） |
| `FLEXKV_HASH_SCHEME` | 空 | `sglang-sha256-i64-v1` = 用 SGLang 的页哈希给块命名（KV 事件路由需要） |
| `DYN_RADIXSHMEM_KV_EVENTS_ENDPOINT(S)` | 空 | dynamo worker：本节点（或按节点顺序的各节点）CPU 池 radix-server 的事件端点 |
| `DYN_RADIXSHMEM_REMOTE_HIT_WEIGHT` | 0.5 | dynamo frontend：只在其他节点上的块的计分比例 |

## 7. 容量与运维

- **CPU 池大小**：CPU 池同时是 SSD 读、对端拉取、SSD 写的中转区，至少要 *并发请求数 × 每请求新增块数*。太小时 FlexKV
  会静默跳过存储和预取（不报错，但命中率会崩）。
- **重启规则**：节点的两个 radix-server 要和本节点的 SGLang/FlexKV 一起重启。单独重启 radix-server 时，它会等 FlexKV
  重新下发配置，在 FlexKV 重启之前该节点的 SSD 层一直不可用。FlexKV 崩溃后也要一起重启 radix-server（崩溃进程 pin 住的
  slot 不会被回收）。
- **SSD 缓存跨 FlexKV 重启保留**：SSD 文件名固定（`ssd_cache_dir` 下 `flexkv_rsm_ssd_*.bin`），旁边的
  `flexkv_rsm_ssd.json` 记录这些文件属于哪个 SSD 索引（radixshmem 的 `index_id`）和文件布局。FlexKV 启动时两者都对得上
  就保留 SSD 索引；SSD radix-server 重启过、`ssd_cache_gb` / 模型变了、或文件被重建，就自动清空 SSD 索引（日志
  `clearing ... blocks of SSD index`）。
- **首次启动**：SSD 集群要等所有节点的 FlexKV 都配置完 SSD 索引 server 才就绪；节点加载模型快慢不一时，保留 4.4 中的
  启动超时设置。
- **单机多实例**：每个实例要有独立的 `FLEXKV_SERVER_RECV_PORT`、`FLEXKV_NIXL_SSD_PORT`（每个实例占两个端口）、
  radix-server 名字，开 layerwise 时还要有独立的 `FLEXKV_LAYERWISE_EVENTFD_SOCKET`。
- **故障**：对端在截止时间内没响应会被退避；读会退回重新计算，去重写会退回本地写。NIXL 对端 SSD worker 进程死掉会自动
  重启（每个 engine 最多 8 次）。
- DeepSeek-V4.x 不支持 `--disaggregation-enable-kv-checksum`。

## 8. 验证与性能（摘要）

在 H20 节点上验证：Qwen3-8B（1P1D）、DeepSeek-V4.1（1P1D，以及两台机器跨机 2P1D，attention DP4 + MoE EP4）。场景包括
1 小时和 4 小时长跑（随机路由、会话固定路由）、20% 客户端中途取消、热门前缀突发、重启 P/D、暂停 NIXL 端点、64k-100k
长上下文、layerwise 开/关（含跨机批量恢复）、强制 retraction、decode 分段存储、去重开/关。全部 0 个意外请求错误，needle 答案全对，命中缓存
的输出无乱码。


本文第 2 节的分支组合（2026-11-23，上游 main 基线）在同一类 H20 节点上复测了 DeepSeek-V4.1 两节点（TP4/DP4/EP4，
bounded replay 开、`SGLANG_FLEXKV_SWA_GRID_PAGES=4`、`--mem-fraction-static 0.92`）：直连 + FlexKV 共享 SSD 文件 + 对端 SSD +
`global_dedup` + KV 事件路由（dynamo）与 server-client 两种模式各 952 个请求，0 错误、needle 全对、命中率 78.5%；路由看到的
每节点块数与 radix-server 精确一致。只重启 FlexKV/SGLang（radix-server 不动）后 SSD 索引保留，同样的请求 480 个 0 错误、命中率
93.5%（重启前 79.8%）。PD 1P1D（layerwise 关、overlap 开；decode 节点 `--mem-fraction-static 0.93`，见第 5 节）2833 个请求，
0 错误、needle 全对、0 乱码。

| 指标 | 结果 |
|---|---|
| 跨节点 SSD 重复块，local vs `global_dedup` | DeepSeek-V4.1 2P1D 60.6% -> 1.7%；Qwen3-8B 1P1D 37-41% -> 0-14% |
| SSD 写入量 | `global_dedup` 下减少 34%-60% |
| 吞吐 / TTFT / TPOT，`global_dedup` vs local | 差距 2% 以内（DeepSeek-V4.1 1P1D，并发 8-32，3 轮交替） |
| 对端 SSD 读（NIXL，跨机，1.67-9.4 MB 块） | 6-27 GB/s，随请求大小和并发变化 |
| TTFT，DeepSeek-V4.1，16k prompt（5 次中位数） | 重新计算 4572 ms；本地 CPU 189；本地 SSD 206；对端 CPU 262；对端 SSD 197 |
| TTFT，DeepSeek-V4.1，64k prompt（5 次中位数） | 重新计算 20390 ms；本地 CPU 217；本地 SSD 262；对端 CPU 222；对端 SSD 228 |

## 9. 已知限制

- 单独重启的 radix-server 要等 FlexKV 重启后才能重新加入（第 7 节）。
- 开 `global_dedup` 时，P 和 D 同时写同一段新前缀仍可能都写一份（百分之几的重复）；前缀会集中在最先存它的节点上，
  因此有一部分 SSD 读会从本地变成对端。
- SGLang 分支还修了本方案下发现的 SGLang 问题：PD retract 重新 bootstrap 复用原请求的 rid 和 bootstrap room（上游问题，
  不开 FlexKV 也会出现）；overlap 调度下非 layerwise 恢复的 H2D 与上一批未完成的 kernel 写同一批 slot（偶发乱码，已在恢复前
  同步）；`SGLANG_LOGGING_CONFIG_PATH` 设置时 detokenizer 启动崩溃（上游 `logging.config` 未导入）。分块 prefill 取消时
  SWA 页重复释放的修复已包含在 XingLiu1 的分支里。
