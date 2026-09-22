# radixshmem 模式配置参考

FlexKV 以 radixshmem 作为 CPU 层（索引 + SlotStore + 跨节点拉取）时，配置分两处：

| 谁 | 载体 | 内容 |
|---|---|---|
| 运维 | `radix-server` 命令行，每节点一个进程 | 名字、SlotStore 字节预算及 SWA 占比、hugepage、传输引擎、集群成员（etcd、网卡、rank）、索引调优 |
| FlexKV | 环境变量 + 一个很小的 YAML（`FLEXKV_RADIXSHMEM_CONFIG_PATH`） | 是否启用、attach 哪个 server、等待多久、prefetch 限额 |
| FlexKV 推导 | `ModelConfig` / `CacheConfig` | 几何：每 block 的 token 数、一个 CPU block 的字节数、一个 SWA page 的字节数、SWA 窗口、slot 对齐 |

FlexKV **不再创建 radix-server**。它只实例化 `shmradix.RadixClient`：第一个 client 把几何交给 server，
server 按自己的字节预算规划各池的 slot 数并发布；之后每个 FlexKV 进程 attach 时把这些 slot 数采纳到
`CacheConfig`（`num_cpu_blocks`、`swa.num_slots`）。也就是说，在这个模式下 CPU 层的容量由
`radix-server --data-bytes` 决定，`cpu_cache_gb` 只是占位。

实现：`flexkv/common/radixshmem_config.py`（YAML）、`flexkv/server/shm_radix_bootstrap.py`（几何、attach、采纳）。
radixshmem 侧的接口见 radixshmem 仓库 `python/README.md`。

---

## 1. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `FLEXKV_ENABLE_RADIXSHMEM` | `0` | 模式总开关。`1` 时 CPU 层由 radixshmem 承担，KVServer 不启动，每个 DP 进程各建一个 KVTaskEngine 并 attach 同一个 radix-server。在 `flexkv` 首次 import 前设置。 |
| `FLEXKV_RADIXSHMEM_CONFIG_PATH` | 空 | 第 2 节 YAML 的路径。为空时全部取默认值：attach 本机 `radix-server --name /flexkv`。 |

另有两个 FlexKV 通用变量在该模式下有约束：

- `FLEXKV_CPU_LAYOUT` 必须是 `BLOCKFIRST`。一个 SlotStore slot 就是一个连续的 CPU block，LAYERFIRST 给不出这个布局。
- `FLEXKV_INSTANCE_NUM` / `FLEXKV_INSTANCE_ID`：同一节点上多个推理引擎共享同一个 radix-server 和同一个 TE 时用来区分实例（第 4.4 节）。

该模式与 `enable_ssd`、`enable_remote` 互斥，启动时报错。`enable_p2p_cpu` / `enable_p2p_ssd` 也必须为 False：
跨节点复用由 radix-server 自己完成（etcd + RDMA），在它以集群参数启动时自动开启，不经过 FlexKV 的 Redis P2P 路径。

已移除的变量：`FLEXKV_RADIX_SERVER_LAUNCH_MODE`（不再有嵌入式启动）、`FLEXKV_RADIX_NODE_NAME`、
`FLEXKV_RADIX_RPC_ADDRESS`（节点身份是 `radix-server --node-name` / `--rpc-address` 的事）。

---

## 2. YAML 字段

两个段，都可省略。

### 2.1 `server`：attach 哪个 radix-server

| 键 | 默认 | 说明 |
|---|---|---|
| `name` | `/flexkv` | `radix-server --name`，即索引 shm 名。以 `/` 开头。也派生默认 socket 和 FlexKV 自己的 TE channel 前缀（第 5 节）。 |
| `endpoint` | 空 | gRPC 端点。空为 `unix:///dev/shm/<name>.sock`；server 以 `--endpoint` 改成 TCP 或别的路径时这里写同一个值。 |
| `ready_timeout_s` | `600` | 一个 FlexKV 进程等 server **可达且 ready** 的总时长。覆盖运维晚起 server、SlotStore prefault、集群 rendezvous（server 的 `--bootstrap-timeout`）。超时报错并给出启动命令。 |

### 2.2 `client`：FlexKV 侧参数

| 键 | 默认 | 说明 |
|---|---|---|
| `prefetch_timeout_ms` | `5000` | 一次 prefetch 拉取的服务端超时。到期后 job 以本地命中的部分完成。 |
| `prefetch_max_inflight` | `128` | 每个 DP 进程在飞的 peer 拉取上限，达到后新的 prefetch 跳过 peer 查询。需小于 `max_outstanding`。 |
| `max_outstanding` | `256` | `RadixClient` 未领取 job 的上限。 |

### 2.3 不再接受的段

旧格式的 `cluster` / `data` / `index` 段出现时直接报错：这些键现在都是 `radix-server` 的命令行参数（第 7 节有对照表）。

示例：

```yaml
# /etc/flexkv/radixshmem.yaml
server:
  name: /flexkv
  ready_timeout_s: 900
client:
  prefetch_timeout_ms: 5000
  prefetch_max_inflight: 128
```

---

## 3. 几何与 slot 数

FlexKV 交给 server 的几何（`shm_radix_bootstrap.expected_geometry` → `shmradix.Geometry`）：

| 字段 | 来源 |
|---|---|
| `block_size` | `CacheConfig.tokens_per_block`（sglang 的 page size） |
| `full_slot_bytes` | 按 `StorageEngine` 的 BLOCKFIRST 布局算出的一个 CPU block 字节数（每 PP 段层数 × 节点内 KV head 数 × head_size × kv_dim × dtype × tokens_per_block） |
| `swa_slot_bytes` / `swa_window_blocks` | `CacheConfig.swa` 开启时：一个 SWA page 的字节数（uint8）与窗口块数；未开启则没有 SWA 池 |
| `slot_align` | 不超过 4096 且整除每个池 slot 字节数的最大二次幂，保证 SlotStore stride 等于 block 字节数 |

server 收到几何后的规划（radixshmem 的规则）：`swa_slots = floor(swa_ratio × data_bytes / swa_stride)`，
`full_slots = (data_bytes − SWA 占用) / full_stride`。任一池算出 0 个 slot、模型有 SWA 而 `--swa-ratio` 为 0，
都在 configure 时拒绝，FlexKV 报 `cannot serve FlexKV's geometry`。

**采纳**：attach 成功后 `adopt_geometry` 把 `pools.full.num_slots` 写进 `CacheConfig.num_cpu_blocks`，
`pools.swa.num_slots` 写进 `CacheConfig.swa.num_slots`，日志形如
`adopted radix-server /flexkv's slot counts: FULL 8605 slots (cpu_cache_gb had given 1524), SWA 1024 slots`。
之后 TE 的 StorageEngine、cache engine、指标都用采纳后的值。

**校验**：每个 attach 方（KVManager、cache engine、TE）用 `check_geometry` 复核 server 发布的
`block_size`、各池 `slot_bytes`、SlotStore stride、SWA 窗口与自己的布局一致，不一致报错退出，不会静默错位传输。
slot 数不在校验范围内，它们是 server 的。

**同一 server 上的多个 client** 必须带相同的几何：相同模型、page size、SWA 配置。第二个不同的几何被 server 以
`GeometryMismatch` 拒绝，FlexKV 报 `already serves another geometry`。单机下 TP 不同的同一模型通常几何相同
（节点内 KV head 数与 TP 无关），以 `check_geometry` 为准。

---

## 4. 启动方式

### 4.1 单机

```bash
# 运维，每节点一次；nohup / systemd 皆可。DSv4 这类有 SWA 池的模型给 --swa-ratio。
radix-server --name /flexkv --data-bytes 64G --swa-ratio 0.5

# 推理引擎侧
export FLEXKV_ENABLE_RADIXSHMEM=1
export FLEXKV_CPU_LAYOUT=BLOCKFIRST
# 不设 FLEXKV_RADIXSHMEM_CONFIG_PATH 即 attach /flexkv
```

server 起来后打印 `Waiting for a client's geometry`；FlexKV 的第一个进程 attach 时把几何交过去，server 建区域后 ready，
所有进程的 `wait_ready` 返回。server 晚于引擎启动也可以：FlexKV 在 `ready_timeout_s` 内重试连接。

### 4.2 多机（一个集群）

每个节点各起一个 server，用相同的 `--cluster-id` 和 `--registry`；`--rpc-interface`（或 `--rpc-address`）给对端拨入的 IP，
`--node-name` 空时自动为 `node<ip>`：

```bash
radix-server --name /flexkv --data-bytes 64G --swa-ratio 0.5 \
  --expected-min-nodes 4 --num-rht-shards 4 --rht-slots 4 \
  --registry etcd://10.0.0.1:2379 --cluster-id prod_a \
  --rpc-interface bond0 --index-dev mlx5_bond_0 --gid-idx 3 \
  --transfer-dev mlx5_1 --transfer-dev mlx5_2 --bootstrap-timeout 600
```

集群一致的几何字段（`block_size`、池集合、每池 `slot_bytes`、SWA 窗口、`slot_align`）由第一个拿到几何的节点发布到
etcd `radix/<cluster_id>/geometry/<node>`，其余 `waiting` 的节点采纳；各节点的 slot 数可以不同（预算可以不同）。
FlexKV 侧每个节点同一份 YAML 即可。`ready_timeout_s` 要不小于 `--bootstrap-timeout`。

### 4.3 同机多节点（测试）

两个 server 在一台机器上：不同的 `--name`（或相同 name 加 `--endpoint`、`--data-name` 区分）、不同的 `--node-name`、
`--rpc-address 127.0.0.1`。两个 FlexKV 进程各用一份 YAML，`server.name` / `server.endpoint` 指向自己的 server，
并各给一个 `FLEXKV_SERVER_RECV_PORT`。

### 4.4 一节点多引擎共享一个 server

两个独立的推理引擎（各自的 FlexKV、各自的 GPU）attach 同一个 radix-server，互相命中对方存的 KV：

```bash
# 引擎 A                                      # 引擎 B
FLEXKV_INSTANCE_NUM=2 FLEXKV_INSTANCE_ID=0    FLEXKV_INSTANCE_NUM=2 FLEXKV_INSTANCE_ID=1
```

同一份 YAML。`instance 0` 的 dp0 拉起本节点唯一的 TE（`channels = instance_num × dp_size`），TE 等到
`instance_num × gpus_per_node` 张 GPU 都注册才 ready，所以两个引擎都要启动。两边模型 / page size / SWA 配置必须相同
（第 3 节）。node-local DP（多机 DP attention）路径下 `local_dp_client_id` 不带 instance，多实例暂不支持。

---

## 5. 命名派生

| 对象 | 名字 |
|---|---|
| index shm | `--name`；集群模式下 radixshmem 追加 `_<node_name>`，attach 方只需 `--name` |
| SlotStore shm | `<name>_data`（`--data-name` 可改） |
| gRPC socket | `/dev/shm/<name>.sock`（`--endpoint` 可改；YAML `server.endpoint` 跟着改） |
| etcd 键空间 | `radix/<cluster_id>/...` |
| FlexKV TE channel / ctrl | `/dev/shm/flexkv_te_ch_<te_server_id>_<k>`、`flexkv_te_ctrl_<te_server_id>`，`te_server_id` = `name` 去掉开头的 `/`（`/` 换成 `_`） |

---

## 6. 启动时校验

FlexKV 加载 YAML 时报错的情况：出现 `cluster` / `data` / `index` 段；未知段或未知键；`server.name` 不以 `/` 开头或含空白；
`server.ready_timeout_s <= 0`；`client.prefetch_max_inflight >= client.max_outstanding`；`client.prefetch_timeout_ms <= 0`。

`CacheConfig` 侧：`FLEXKV_CPU_LAYOUT != BLOCKFIRST`；打开了 `enable_ssd`、`enable_remote`、`enable_p2p_cpu` 或 `enable_p2p_ssd`；
SWA 开启但 `window_blocks < 1`。

attach 时：`ready_timeout_s` 内连不上 server 报 `no radix-server named ... reachable ...（start it with radix-server --name ...）`；
server 一直在等几何或配置失败报 `not ready within ...`（带 server 的 `mode` 和 `last_error`）；几何冲突见第 3 节。

---

## 7. 从旧版迁移

| 旧 YAML 键 | 现在 |
|---|---|
| `cluster.cluster_id` | `radix-server --cluster-id`（同时不再派生 shm 名；shm 名是 `--name`） |
| `cluster.expected_min_nodes` / `num_rht_shards` / `rht_shard_holders` / `rht_slots_per_bucket` | `--expected-min-nodes` / `--num-rht-shards` / `--rht-shard-holders` / `--rht-slots` |
| `cluster.registry` / `rpc_interface` / `rpc_port` / `settle_ms` / `bootstrap_timeout_sec` | `--registry` / `--rpc-interface` / `--rpc-port` / `--settle-ms` / `--bootstrap-timeout` |
| `cluster.index_dev` / `gid_idx` / `rht_transport` / `peer_index_transport` / `remote_op_transport` / `zmq_listen_port` | 同名 `--index-dev` 等 |
| `data.transfer_devices` / `transfer_protocol` / `transfer_ip` / `transfer_port` / `transfer_metadata` | `--transfer-dev`（可重复）/ `--transfer-protocol` / `--transfer-ip` / `--transfer-port` / `--transfer-metadata` |
| `data.prefault` / `max_inflight` / `max_pending_jobs` / `job_ttl_s` | `--no-prefault` / `--max-inflight` / `--max-pending-jobs` / `--job-ttl` |
| `index.data_pool_ratio` / `background_evict_ratio` / `max_nodes` / `register_chunk_size` | `--data-pool-ratio` / `--background-evict-ratio` / `--max-nodes` / `--register-chunk-tokens`（按 token 数） |
| `server.endpoint` | 保留：server 的 `--endpoint` 与 YAML `server.endpoint` 各写一次 |
| `server.rpc_workers` / `hugepage_path` | `--rpc-workers` / `--hugepage-path` |
| （由 FlexKV 推导的 slot 数、`data_bytes`） | slot 数由 `--data-bytes` 和 `--swa-ratio` 决定，FlexKV 采纳 |
| `FLEXKV_RADIX_SERVER_LAUNCH_MODE` | 移除，只有外部 server |
| `FLEXKV_RADIX_NODE_NAME` / `FLEXKV_RADIX_RPC_ADDRESS` | `--node-name` / `--rpc-address` |
