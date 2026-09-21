# radixshmem 模式配置参考

本文列出 FlexKV 以 radixshmem 作为 CPU 层（索引 + SlotStore + 跨节点拉取）时的全部配置项：
哪些走环境变量、哪些走 YAML、哪些由 FlexKV 自己推导而禁止手工设置。

配置分三层：

| 层 | 载体 | 内容 |
|---|---|---|
| 进程级开关 | 环境变量 `FLEXKV_RADIX_*` | 是否启用、YAML 路径、server 启动方式，以及两个仅供同机多节点测试的 per-node 覆盖 |
| 集群配置 | YAML，`FLEXKV_RADIXSHMEM_CONFIG_PATH` 指向 | 全局，所有节点逐字节相同；键名与 radixshmem 的 dataclass 字段一致 |
| 几何 | 由 `ModelConfig` / `CacheConfig` 推导 | slot 数、slot 字节数、对齐、shm 名；不可配置 |

同一个值只有一个来源。YAML 里没写的键取本文列出的默认值；没有 YAML 时全部取默认值，即单机模式。
示例文件在 `examples/radixshmem_configs/`：`radixshmem_single_node.yaml` 和 `radixshmem_multi_node.yaml`。
实现在 `flexkv/common/radixshmem_config.py`。

---

## 1. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `FLEXKV_ENABLE_RADIXSHMEM` | `0` | 模式总开关。`1` 时 CPU 层由 radixshmem 承担，KVServer 不启动，每个 DP 进程各建一个 KVTaskEngine 并 attach 共享的 radix 区域。在 `flexkv` 首次 import 前设置。 |
| `FLEXKV_RADIXSHMEM_CONFIG_PATH` | 空 | 第 2 节 YAML 的路径。为空时所有键取默认值。 |
| `FLEXKV_RADIX_SERVER_LAUNCH_MODE` | `embedded` | `embedded`：dp0 进程以子进程方式启动 radix-server；`external`：attach 运维已启动的 radix-server。 |
| `FLEXKV_RADIX_NODE_NAME` | 空 | per-node 覆盖，见 3.3。生产部署不设。 |
| `FLEXKV_RADIX_RPC_ADDRESS` | 空 | per-node 覆盖，见 3.3。设了就忽略 YAML 的 `cluster.rpc_interface`。生产部署不设。 |

另有两个 FlexKV 通用变量在该模式下有约束：

- `FLEXKV_CPU_LAYOUT` 必须是 `BLOCKFIRST`。一个 SlotStore slot 就是一个连续的 CPU block，LAYERFIRST 给不出这个布局。
- `FLEXKV_HUGETLBFS_DIR`（默认 `/mnt/hugepages`）：`server.hugepage_path` 为空且 `CacheConfig.use_hugepage_cpu_buffer` 为真时，radix 区域建在这个 hugetlbfs 挂载点下。

该模式与 `enable_ssd`、`enable_remote` 互斥，启动时报错。`enable_p2p_cpu` / `enable_p2p_ssd` 也必须为 False：
跨节点复用由 radix-server 自己完成（etcd + RDMA），在 YAML 使集群成为分布式（`expected_min_nodes > 1` 或
`num_rht_shards > 1`）时自动开启，不再经过 FlexKV 的 Redis P2P 路径。

---

## 2. YAML 字段

五个段。`cluster` / `data` / `index` / `server` 四段的键按名字直接构造 radixshmem 的
`ClusterConfig` / `DataPlaneConfig` / `IndexConfig` / `RadixServerConfig`，用
`dataclasses.fields()` 校验：未知键报错，几何键（2.6）报错。`client` 段是 FlexKV 自己的参数。

### 2.1 `cluster`（`shmradix.ClusterConfig`）

| 键 | 默认 | 说明 |
|---|---|---|
| `cluster_id` | `flexkv` | 集群命名空间。既是 etcd 键前缀 `radix/<cluster_id>/...`，也派生本机全部 shm 和 socket 名（见 4）。同一台机器上跑多个 FlexKV 实例时用不同的 `cluster_id` 区分。 |
| `expected_min_nodes` | `0` | 集群开关。`> 1` 时进入 etcd + RDMA 模式：bootstrap 等到 etcd 里登记的节点数达到 `max(expected_min_nodes, num_rht_shards)` 且稳定 `settle_ms` 后分配 rank。`world_size` 是实际观察到的节点数，可以大于该值。 |
| `registry` | `etcd://127.0.0.1:2379` | etcd 地址，集群模式必填。格式 `etcd://host:port`，多个成员在 scheme 之后用逗号或分号分隔，scheme 只写一次：`etcd://10.0.0.1:2379,10.0.0.2:2379`。见 3.4。 |
| `rpc_interface` | 空 | 解析 bootstrap IP 的网卡名，如 `bond0`（南北向管理网卡）。集群模式下必填（除非用 `FLEXKV_RADIX_RPC_ADDRESS` 覆盖）。每个节点解析出自己的 IP，节点身份自动派生为 `node<ip>`。 |
| `rpc_port` | `0` | bootstrap / XRC 监听端口，0 由系统分配。 |
| `settle_ms` | `500` | 成员集合稳定多久后开始 bootstrap。 |
| `bootstrap_timeout_sec` | `120` | rendezvous 超时。FlexKV 的 attach 超时是该值加 60 秒。 |
| `index_dev` | 空 | index 控制面（RHT 面和 peer index 面）用的 HCA，空为第一个可用设备，通常是 `mlx5_0` 即东西向计算网卡。建议指定南北向管理网卡的 HCA（如 `mlx5_bond_0`）：控制面只有小消息，计算网卡留给 KV 字节。KV 字节的 HCA 是 `data.transfer_devices`。 |
| `gid_idx` | `3` | RoCE GID 索引。 |
| `rht_transport` | `xrc` | client 到 RHT shard holder 的 QP 类型，`xrc` 或 `dc`。 |
| `peer_index_transport` | `xrc` | remote walk 读 peer index 的 QP 类型，`xrc` 或 `dc`。 |
| `remote_op_transport` | `zmq` | remote insert / query 控制面，`zmq` 或 `dc`。FlexKV 不开 remote op，该字段不生效。 |
| `num_rht_shards` | `0` | RHT 分片数。0 为每节点一片。设了必须不大于节点数。 |
| `rht_shard_holders` | `[]` | 持有分片的 rank 列表，空为 rank 0 到 `num_rht_shards - 1`。rank 由 rendezvous 后按 `node<ip>` 字典序分配。 |
| `rht_slots_per_bucket` | `4` | RHT 每 bucket 的 slot 数，取 1 / 2 / 4 / 8。1 是盲覆盖，会丢路由项。 |
| `enable_remote_insert` | `false` | 透传。 |
| `enable_remote_query` | `false` | 透传。 |
| `zmq_listen_port` | `0` | 透传。 |

`rht_transport` / `peer_index_transport` / `remote_op_transport` / `num_rht_shards` / `rht_shard_holders` /
`rht_slots_per_bucket` 只有 rank 0 的值生效，bootstrap 时经 etcd `/config` 广播给其他节点。全局 YAML
下各节点值本来相同，这一规则只是多一层保险。

禁止出现：`node_name`、`rpc_address`。它们是 per-node 值，全局 YAML 放不下；需要时走 3.3 的环境变量。

### 2.2 `data`（`shmradix.DataPlaneConfig` 的非几何字段）

| 键 | 默认 | 说明 |
|---|---|---|
| `transfer_devices` | `[]` | KV 字节传输（mooncake）用的 HCA 列表，空为 mooncake 发现的全部设备。 |
| `transfer_protocol` | `rdma` | `rdma` 或 `tcp`。 |
| `transfer_ip` | 空 | 数据面 IP，空为 rpc 地址。 |
| `transfer_port` | `0` | 0 由引擎选。 |
| `transfer_metadata` | `P2PHANDSHAKE` | mooncake 元数据服务。 |
| `prefault` | `true` | server 启动时 MAP_POPULATE 整个 SlotStore。D2H 延迟可预测，页由 server 进程的 NUMA 策略放置；大池会拉长启动时间。 |
| `max_inflight` | `256` | 传输引擎在飞 batch 数。 |
| `max_pending_jobs` | `4096` | 排队 + 运行 + 未领取的 job 上限，超过则 Submit 被拒。 |
| `job_ttl_s` | `60.0` | 未领取 job 的保留秒数。 |

禁止出现：`data_bytes`、`full_slot_bytes`、`swa_slot_bytes`、`mamba_slot_bytes`、`slot_align`、`data_name`。

### 2.3 `index`（`shmradix.IndexConfig` 的非几何字段）

| 键 | 默认 | 说明 |
|---|---|---|
| `data_pool_ratio` | `8.0` | 索引 DataPool 大小系数：`full_slots × ratio × (12 或 16)` 字节。 |
| `background_evict_ratio` | `0.05` | 后台驱逐比例，0 关闭。 |
| `max_nodes` | `0` | radix 节点池容量，0 自动。 |
| `register_chunk_size` | `4096 / tokens_per_block` | RHT 注册粒度（block 数）。FlexKV 的默认让一段覆盖 4096 个 token，与 block 大小无关（radixshmem 自身默认 128 block）。 |

禁止出现：`name`、`tokens_per_block`、`full_slots`、`swa_slots`、`swa_window_blocks`、`mamba_slots`、`evict_policy`。

### 2.4 `server`（`shmradix.RadixServerConfig` 顶层）

| 键 | 默认 | 说明 |
|---|---|---|
| `endpoint` | 空 | gRPC 端点。空为 `unix:///dev/shm/<index 名>.sock`。server 监听和 client attach 都用它。 |
| `rpc_workers` | `32` | gRPC 工作线程数。每个有在飞 job 的 client 占一个。 |
| `hugepage_path` | 空 | index 和 SlotStore 的 hugetlbfs 挂载点。空时按第 1 节 `FLEXKV_HUGETLBFS_DIR` 的规则决定。 |

### 2.5 `client`（FlexKV 侧，不传给 radixshmem）

| 键 | 默认 | 说明 |
|---|---|---|
| `prefetch_timeout_ms` | `5000` | 一次 prefetch 拉取的服务端超时。到期后 job 以本地命中的部分完成。 |
| `prefetch_max_inflight` | `128` | 每个 DP 进程在飞的 peer 拉取上限，达到后新的 prefetch 跳过 peer 查询。需小于 `max_outstanding`。 |
| `max_outstanding` | `256` | `RadixClient` 未领取 job 的上限。 |

### 2.6 由 FlexKV 推导、禁止手工设置的字段

| 字段 | 来源 |
|---|---|
| `index.name` | `/shmradix_<cluster_id>_cpu` |
| `index.tokens_per_block` | `CacheConfig.tokens_per_block` |
| `index.full_slots` | `CacheConfig.num_cpu_blocks` |
| `index.swa_slots` / `swa_window_blocks` | `CacheConfig.swa.num_slots` / `window_blocks`，SWA 未开启为 0 |
| `data.full_slot_bytes` | 按 `StorageEngine` 的 BLOCKFIRST 布局算出的一个 block 字节数 |
| `data.swa_slot_bytes` | 同上，SWA 池按 uint8 |
| `data.data_bytes` | `full_slots × full_slot_bytes + swa_slots × swa_slot_bytes` |
| `data.slot_align` | 不超过 4096 且整除每个池 slot 字节数的最大二次幂，保证 slot stride 等于 block 大小 |
| `data.data_name` | `<index.name>_data` |

这些值在 YAML 中出现时启动报错，防止和 `CacheConfig` 静默冲突。TE 进程 attach 后还会用
`check_geometry` 复核 server 端区域和 FlexKV 自己的布局一致。

---

## 3. 示例

### 3.1 单机

不设 `FLEXKV_RADIXSHMEM_CONFIG_PATH` 即可。等价于：

```yaml
cluster:
  cluster_id: flexkv
  expected_min_nodes: 0
```

无 etcd、无 RDMA 依赖。多个 DP 进程共享一个 radix-server 和一个 TE。

### 3.2 多机（全局配置，所有节点同一文件）

```yaml
# /etc/flexkv/radixshmem.yaml
cluster:
  cluster_id: prod_a
  expected_min_nodes: 4
  num_rht_shards: 4
  registry: etcd://10.0.0.1:2379
  rpc_interface: bond0            # 南北向网卡；每节点解析自己的 IP，身份为 node<ip>
  index_dev: mlx5_bond_0          # index 内部 RDMA 的 HCA，南北向网卡
  gid_idx: 3
  rht_transport: xrc
  peer_index_transport: dc
  rht_slots_per_bucket: 4
  bootstrap_timeout_sec: 120
data:
  transfer_devices: [mlx5_1, mlx5_2]   # KV 字节传输的 HCA
  prefault: true
index:
  data_pool_ratio: 8.0
server:
  rpc_workers: 32
client:
  prefetch_timeout_ms: 5000
  prefetch_max_inflight: 128
```

每个节点：

```bash
export FLEXKV_ENABLE_RADIXSHMEM=1
export FLEXKV_RADIXSHMEM_CONFIG_PATH=/etc/flexkv/radixshmem.yaml
export FLEXKV_CPU_LAYOUT=BLOCKFIRST
```

节点身份、shm 名、rank 全部自动派生，文件里没有任何 per-node 内容。

### 3.3 同机多节点（测试）

两个 radix-server 在一台机器上时，同一网卡解析出同一 IP，身份会撞。用两个 per-node 环境变量区分：

```bash
# 进程 A
FLEXKV_RADIX_NODE_NAME=r0 FLEXKV_RADIX_RPC_ADDRESS=127.0.0.1 ...
# 进程 B
FLEXKV_RADIX_NODE_NAME=r1 FLEXKV_RADIX_RPC_ADDRESS=127.0.0.1 ...
```

`FLEXKV_RADIX_RPC_ADDRESS` 设置后 FlexKV 清掉 YAML 的 `rpc_interface`（radixshmem 规则是 interface 优先，
不清会被覆盖回去）。两个进程仍共用同一份 YAML。

设置了 `FLEXKV_RADIX_NODE_NAME` 时，本机命名前缀从 `<cluster_id>` 变为 `<cluster_id>_<node_name>`（第 4 节），
两个进程的 SlotStore、socket 和 TE channel 因此互不冲突。每个进程还要各给一个 `FLEXKV_SERVER_RECV_PORT`。

### 3.4 `registry` 的填法

radixshmem 把 `registry` 去掉第一个 `://` 之前的 scheme 后，余下部分按逗号或分号切成 endpoint 列表，
交给 etcd 的 clientv3。因此：

- 单成员：`etcd://10.0.0.1:2379`。
- 多成员：`etcd://10.0.0.1:2379,10.0.0.2:2379,10.0.0.3:2379`。scheme 只写一次；写成
  `etcd://a:2379,etcd://b:2379` 会把第二个 `etcd://b:2379` 原样当作 endpoint 传下去，连接失败。
- 只支持明文连接，没有 TLS 和用户名密码的配置入口。拨号超时固定 5 秒。
- 一个 etcd 可以服务多个集群，键空间由 `cluster_id` 隔开（`radix/<cluster_id>/...`）；索引 rendezvous 和
  数据面登记（`data/<node>`）都在同一个 etcd 里。
- etcd 不只在启动时用：节点的 lease keep-alive、`/peers` watch 和数据面登记贯穿整个运行期，etcd 不可用会导致
  lease 过期、节点从集群视图中消失。生产环境用 3 成员 etcd，并把全部成员写进 `registry`。
- 每个节点必须能访问 `registry` 里的地址；默认值 `127.0.0.1:2379` 只适用于所有节点在同一台机器上的测试。
- 同一进程内 etcd 连接是全局单例，首个 `init` 的 endpoint 生效；FlexKV 里索引和数据面都用 `cluster.registry`，
  不会出现两个不同地址。

---

## 4. 命名派生

所有名字来自 `cluster.cluster_id`。记本机前缀 `local_id`：未设 `FLEXKV_RADIX_NODE_NAME` 时就是
`cluster_id`，设了则是 `<cluster_id>_<node_name>`（`RadixShmemConfig.local_id`）。

| 对象 | 名字 |
|---|---|
| etcd 键空间 | `radix/<cluster_id>/...` |
| index shm | `/shmradix_<local_id>_cpu`；集群模式下 radixshmem 再追加 `_<node_name>`，attach 方只需 base name |
| SlotStore shm | `/shmradix_<local_id>_cpu_data` |
| gRPC socket | `/dev/shm/shmradix_<local_id>_cpu.sock` |
| TE shm channel | FlexKV 内部 IPC 名，以 `local_id` 为前缀 |

---

## 5. 三套传输的区分

| 配置 | 取值 | 链路 | HCA |
|---|---|---|---|
| `cluster.rht_transport` | xrc / dc | client 向 RHT shard holder 写路由项 | `cluster.index_dev` |
| `cluster.peer_index_transport` | xrc / dc | remote walk 时对 peer 节点 index 的单边 RDMA read | `cluster.index_dev` |
| `cluster.remote_op_transport` | zmq / dc | remote insert / query 控制面，FlexKV 不使用 | zmq 走 TCP |
| `data.transfer_protocol` + `data.transfer_devices` | rdma / tcp | 两节点 SlotStore 之间的 KV 字节搬运（mooncake），即 `pull_async` 的实际拉取 | `data.transfer_devices` |

xrc 对每个目标一条 QP；dc 用一个 DC initiator 对所有目标，QP 数 O(1)，需要 mlx5。任一 index 面为 dc 时
server 建 DCT，client 两个面共用一个 DCI。FlexKV 的 `get_match` 只查本地，不走前两条；`pull_async`
在服务端规划时走 RHT 和 remote walk，随后的字节搬运走 mooncake。

---

## 6. 启动时校验

FlexKV 在加载 YAML 时检查以下条件，不满足直接报错，不等到 rendezvous：

- 未知键、2.6 的几何键、`cluster.node_name`、`cluster.rpc_address` 出现在 YAML。
- `expected_min_nodes > 1` 时 `registry` 为空，或 `rpc_interface` 与 `FLEXKV_RADIX_RPC_ADDRESS` 都为空。
- `FLEXKV_RADIX_RPC_ADDRESS=0.0.0.0`：所有节点会派生出同一个身份。
- `rht_shard_holders` 和 `transfer_devices` 接受列表或逗号分隔字符串，其他类型报错。
- `num_rht_shards > expected_min_nodes`（`expected_min_nodes` 非 0 时）。
- `rht_slots_per_bucket` 不在 {1, 2, 4, 8}。
- `rht_transport` / `peer_index_transport` 不在 {xrc, dc}；`remote_op_transport` 不在 {zmq, dc}。
- `client.prefetch_max_inflight >= client.max_outstanding`。
- `FLEXKV_CPU_LAYOUT != BLOCKFIRST`，或 `num_cpu_blocks <= 0`，或 SWA 池装不下一个 window。
- `CacheConfig` 打开了 `enable_ssd`、`enable_remote`、`enable_p2p_cpu` 或 `enable_p2p_ssd`。

TE attach 后 `check_geometry` 复核 server 端的 `tokens_per_block`、各池 slot 数、slot 字节数和 stride，
不一致则报错退出，不会静默错位传输。
