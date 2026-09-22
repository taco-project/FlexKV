# radixshmem 模式配置

FlexKV 以 radixshmem 作为 CPU 层（索引 + SlotStore + 跨节点拉取）。配置分两处：

| 谁 | 载体 | 内容 |
|---|---|---|
| 运维 | `radix-server` 命令行，每节点一个进程 | 名字、SlotStore 字节预算与 SWA 占比、hugepage、传输引擎、集群成员（etcd、网卡、rank）、索引调优 |
| FlexKV | 两个环境变量 | 是否启用、attach 哪个 server |

FlexKV 只实例化 `shmradix.RadixClient`：第一个 client 把几何（每 block 的 token 数、一个 CPU block 和一个 SWA page
的字节数、SWA 窗口、slot 对齐）交给 server，server 按 `--data-bytes` 和 `--swa-ratio` 规划各池的 slot 数并发布，
每个 FlexKV 进程 attach 时把 slot 数采纳到 `CacheConfig`（`num_cpu_blocks`、`swa.num_slots`）。CPU 层容量由
`radix-server --data-bytes` 决定，`cpu_cache_gb` 在该模式下不起作用。

实现：`flexkv/server/shm_radix_bootstrap.py`（几何、attach、采纳、固定参数）。
radixshmem 侧接口见 radixshmem 仓库 `python/README.md`。

## 1. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `FLEXKV_ENABLE_RADIXSHMEM` | `0` | `1` 启用。在 `flexkv` 首次 import 前设置。 |
| `FLEXKV_RADIXSHMEM_SERVER_NAME` | `/flexkv` | attach 的 `radix-server --name`，即索引 shm 名，以 `/` 开头。gRPC 端点是 radixshmem 由名字派生的 `unix:///dev/shm/<name>.sock`。 |
| `FLEXKV_CPU_LAYOUT` | | 必须是 `BLOCKFIRST`：一个 SlotStore slot 就是一个连续的 CPU block。 |
| `FLEXKV_INSTANCE_NUM` / `FLEXKV_INSTANCE_ID` | `1` / `0` | 同一节点上多个推理引擎共享一个 radix-server 时区分实例（第 4.4 节）。 |

该模式只承担 CPU 层：`enable_ssd`、`enable_remote`、`enable_p2p_cpu`、`enable_p2p_ssd` 必须关闭，启动时校验。
跨节点复用由 radix-server 完成（etcd + RDMA），server 以集群参数启动时自动开启。

进程模型与 FlexKV 其它模式一致：`dp_size=1` 且单实例时 KVTaskEngine 在引擎进程内，TE 是它的子进程；
`dp_size>1` 或多实例时每节点一个 KVServer，DP 进程是它的 client，KVServer 里的 KVTaskEngine 和 TE attach radix-server。

## 2. 固定参数

attach 的其余参数是 `flexkv/server/shm_radix_bootstrap.py` 里的常量：

| 常量 | 值 | 含义 |
|---|---|---|
| `READY_TIMEOUT_S` | 600 | 等 server 可达且 ready 的总时长，覆盖 server 晚起、SlotStore prefault、集群 rendezvous；server 的 `--bootstrap-timeout` 不要超过它。 |
| `PREFETCH_TIMEOUT_MS` | 5000 | 一次 prefetch 拉取的服务端超时，到期后 job 以本地命中的部分完成。 |
| `PREFETCH_MAX_INFLIGHT` | 128 | 每个 KVTaskEngine 在飞的 peer 拉取上限，达到后新的 prefetch 跳过 peer 查询。 |
| `MAX_OUTSTANDING` | 256 | `RadixClient` 未领取 job 的上限。 |

## 3. 几何与 slot 数

FlexKV 交给 server 的几何（`shm_radix_bootstrap.expected_geometry` → `shmradix.Geometry`）：

| 字段 | 来源 |
|---|---|
| `block_size` | `CacheConfig.tokens_per_block` |
| `full_slot_bytes` | 按 `StorageEngine` 的 BLOCKFIRST 布局算出的一个 CPU block 字节数 |
| `swa_slot_bytes` / `swa_window_blocks` | `CacheConfig.swa` 开启时一个 SWA page 的字节数与窗口块数；未开启则没有 SWA 池 |
| `slot_align` | 不超过 4096 且整除每个池 slot 字节数的最大二次幂，使 SlotStore stride 等于 block 字节数 |
| `register_chunk_tokens` | `0`，即 server 的 `--register-chunk-tokens`（默认 4096 token）；采纳后按 `tokens // tokens_per_block`（至少 1）换算成 block 数 |

server 的规划：`swa_slots = floor(swa_ratio × data_bytes / swa_stride)`，`full_slots = (data_bytes − SWA 占用) / full_stride`。
任一池为 0、模型有 SWA 而 `--swa-ratio` 为 0，configure 时拒绝，FlexKV 报 `cannot serve FlexKV's geometry`。

**采纳**（`adopt_radix_server`，KVManager 与 KVTaskEngine 各调一次，幂等）：`pools.full.num_slots` 写进
`CacheConfig.num_cpu_blocks`，`pools.swa.num_slots` 写进 `CacheConfig.swa.num_slots`，同时记录 `register_chunk_tokens`。
日志形如 `adopted radix-server /flexkv's geometry: FULL 8605 slots ..., SWA 1024 slots ...; RHT registration chunk 4096 tokens = 64 blocks`。

**校验**（`check_geometry`，TE 等 attach 方）：server 发布的 `block_size`、各池 `slot_bytes`、SlotStore stride、SWA 窗口
须与自身布局一致，否则报错退出。slot 数是 server 的，不在校验范围。

同一 server 上的多个 client 须带相同的几何（相同模型、page size、SWA 配置）；不同的几何被 server 以 `GeometryMismatch`
拒绝，FlexKV 报 `already serves another geometry`。

## 4. 启动

### 4.1 单机

```bash
# 运维，每节点一次；nohup / systemd 皆可。有 SWA 池的模型（DSv4 等）给 --swa-ratio。
radix-server --name /flexkv --data-bytes 64G --swa-ratio 0.5

# 推理引擎侧
export FLEXKV_ENABLE_RADIXSHMEM=1
export FLEXKV_CPU_LAYOUT=BLOCKFIRST
# 不设 FLEXKV_RADIXSHMEM_SERVER_NAME 即 attach /flexkv
```

server 起来后打印 `Waiting for a client's geometry`；FlexKV 的第一个进程 attach 时交出几何，server 建区域后 ready。
server 可晚于引擎启动，FlexKV 在 `ready_timeout_s` 内重试连接。

### 4.2 多机（一个集群）

每个节点各起一个 server，相同的 `--cluster-id` 和 `--registry`；`--rpc-interface`（或 `--rpc-address`）给对端拨入的 IP，
`--node-name` 空时为 `node<ip>`：

```bash
radix-server --name /flexkv --data-bytes 64G --swa-ratio 0.5 \
  --expected-min-nodes 4 --num-rht-shards 4 --rht-slots 4 \
  --registry etcd://10.0.0.1:2379 --cluster-id prod_a \
  --rpc-interface bond0 --index-dev mlx5_bond_0 --gid-idx 3 \
  --transfer-dev mlx5_1 --transfer-dev mlx5_2 --bootstrap-timeout 600
```

集群一致的几何字段（`block_size`、池集合、每池 `slot_bytes`、SWA 窗口、`slot_align`、`register_chunk_tokens`）由第一个
拿到几何的节点发布到 etcd `radix/<cluster_id>/geometry/<node>`，其余节点采纳；各节点的 slot 数可以不同。FlexKV 侧每个节点
同一个 `FLEXKV_RADIXSHMEM_SERVER_NAME`。

### 4.3 同机多节点（测试）

两个 server 在一台机器上：不同的 `--name`、不同的 `--node-name`、`--rpc-address 127.0.0.1`。两个 FlexKV 进程各自的
`FLEXKV_RADIXSHMEM_SERVER_NAME` 指向自己的 server，并各给一个 `FLEXKV_SERVER_RECV_PORT`。

### 4.4 一节点多引擎共享一个 server

两个独立的推理引擎（各自的 FlexKV、各自的 GPU）attach 同一个 radix-server，互相命中对方存的 KV：

```bash
# 引擎 A                                      # 引擎 B
FLEXKV_INSTANCE_NUM=2 FLEXKV_INSTANCE_ID=0    FLEXKV_INSTANCE_NUM=2 FLEXKV_INSTANCE_ID=1
```

同一个 `FLEXKV_RADIXSHMEM_SERVER_NAME`。`instance_num > 1` 走 server-client 模式：`instance 0` 的 dp0 内嵌本节点的 KVServer（或用
`FLEXKV_SERVER_LAUNCH_MODE=external` 单独启动），它的 TE 等 `instance_num × gpus_per_node` 张 GPU 都注册后 ready，
所以两个引擎都要启动。两边模型 / page size / SWA 配置必须相同（第 3 节）。node-local DP（多机 DP attention）下不支持多实例。

## 5. 命名

| 对象 | 名字 |
|---|---|
| index shm | `--name`；集群模式下 radixshmem 追加 `_<node_name>`，attach 方只需 `--name` |
| SlotStore shm | `<name>_data`（`--data-name` 可改） |
| gRPC socket | `/dev/shm/<name>.sock`，FlexKV 按名字派生；server 端保持默认 `--endpoint` |
| etcd 键空间 | `radix/<cluster_id>/...` |

## 6. 报错含义

- `FLEXKV_RADIXSHMEM_SERVER_NAME` 须以 `/` 开头且无空白。
- `CacheConfig`：`FLEXKV_CPU_LAYOUT != BLOCKFIRST`；打开了 `enable_ssd`、`enable_remote`、`enable_p2p_cpu` 或 `enable_p2p_ssd`；
  SWA 开启但 `window_blocks < 1`。
- attach：600 s 内连不上 server 报 `no radix-server named ... reachable ...`（附启动命令）；server 有响应但 attach
  失败（如 `server is closed`）立即报错；server 一直在等几何或配置失败报 `not ready within ...`（附 server 当时的
  `mode` 和 `last_error`）；几何冲突见第 3 节。
