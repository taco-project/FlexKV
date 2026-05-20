# FlexKV Dist-Reuse Redis Schema 手册

> **本文目的**：列清楚 FlexKV `dist_reuse` 用到的所有 Redis key 的命名规则、字段含义、典型读写时序，方便用户做容量规划、运维诊断、故障排查。
>
> Redis 在 `dist_reuse` 中只承担两类职责：
> 1. **集群发现与心跳**：每个节点（Master / Remote）注册自己的 ZMQ 地址、Mooncake CPU buffer 指针，方便 peer instance 知道"哪个 SD 在哪台节点"。
> 2. **block 元数据广播**：每个 block 的 `(parent_hash, hash, lease_time, state)` 等元数据 publish 到 Redis，让 peer instance 的 `DistributedRadixTree` 能重建跨 instance 的索引。
>
> 跨 instance 的 KV 数据本身不走 Redis，走 Mooncake P2P RDMA。
>
> 关于 sd_key 格式与 dist_reuse 整体原理见
> [`dist_reuse_with_cp_pp_multinode_tp_simplified.md`](./dist_reuse_with_cp_pp_multinode_tp_simplified.md)。

---

## 0. sd_key 速记

```
<sd_key> = "<model_id>:ppn<pp_node_idx>/<pp_node_count>:tpn<tp_node_idx>/<tp_node_count>:nsa<0|1>"
```

例：

```
c3a2f91d0bcdef01:ppn0/1:tpn0/1:nsa0       — 单机 PP=1 部署
c3a2f91d0bcdef01:ppn0/2:tpn0/1:nsa0       — 跨节点 PP=2 第 0 节点
c3a2f91d0bcdef01:ppn1/2:tpn0/1:nsa0       — 跨节点 PP=2 第 1 节点
c3a2f91d0bcdef01:ppn0/1:tpn0/2:nsa0       — 跨机 TP=2 第 0 节点
c3a2f91d0bcdef01:ppn0/1:tpn1/2:nsa0       — 跨机 TP=2 第 1 节点
```

不变量：`pp_node_count × tp_node_count == nnodes`，即 **SD 数量 = 物理节点数**。

逻辑 db：由 `CacheConfig.flexkv_redis_db` 指定（默认 0，建议生产环境用独立的 db，如 15）。

---

## 1. 命名空间一览

| 命名空间 | 作用域 | key 数量 |
|---|---|---|
| `sd:<sd_key>:*` | 每个 SD 独立 | 每实例 `pp_node_count × tp_node_count = nnodes` 份 |
| `flexkv:instance:<instance_id>:*` | 每个 FlexKV 实例独立 | 跨 SD 共享 |
| `global:node_id` | 全局（跨实例） | 单条计数器 |
| `flexkv_node_id_updated:<sd_key>` | Pub/Sub channel | 每 SD 一个（非 key） |

---

## 2. SD 维度 key（每节点 1 份，每实例 `nnodes` 份）

### 2.1 `sd:<sd_key>:node:<node_id>` — 节点心跳

| 属性 | 值 |
|---|---|
| 类型 | **Hash + TTL** |
| TTL | `CacheConfig.instance_session_ttl_seconds`（默认 8s） |
| 维护方 | `RedisNodeInfo._heartbeat_worker` 以 TTL/3 频率发 `EXPIRE` |
| 生命周期 | 进程启动时 `register_node` 创建；`atexit` / `SIGINT` 时 `unregister_node` |

**Hash 字段**：

| 字段 | 类型 | 含义 |
|---|---|---|
| `node_id` | int | 从 `global:node_id` INCR 取到（全局唯一） |
| `ip` / `local_ip` | str | 本节点监听 IP |
| `uuid` | str | 进程 UUID（防同 IP 重启后留下"鬼节点"） |
| `status` | str | `"active"` |
| `timestamp` | int | 注册时的 Unix 时间戳（秒） |
| `sd_key` | str | 冗余存本 SD 的序列化形式，便于运维排查 |

---

### 2.2 `sd:<sd_key>:meta:<node_id>` — 节点地址元信息

| 属性 | 值 |
|---|---|
| 类型 | **Hash** |
| TTL | 无（生命周期跟随 `node:<id>` 的 TTL） |
| 维护方 | `RedisMeta.regist_node_meta(...)` |

**Hash 字段**：

| 字段 | 含义 |
|---|---|
| `node_id` | int |
| `addr` | 节点 IP |
| `zmq_addr` | `tcp://ip:port`；Master 派发 `TransferOpGraph` 时使用 |
| `cpu_buffer_ptr` | Mooncake P2P 读取的 CPU block 池首地址 |
| `ssd_buffer_ptr` | SSD block 池首地址（如启用 SSD） |

---

### 2.3 `sd:<sd_key>:buffer:<node_id>:<buffer_ptr>` — Mooncake 注册缓冲区

| 属性 | 值 |
|---|---|
| 类型 | **Hash** |
| TTL | 无 |
| 维护方 | `RedisMeta.regist_buffer([(ptr, size), ...])` |

**Hash 字段**：

| 字段 | 含义 |
|---|---|
| `buffer_size` | int，buffer 字节数 |
| 自定义字段 | 可扩展 `rdma_port` / `nic_name` 等 Mooncake 附加信息 |

---

### 2.4 `sd:<sd_key>:block:<node_id>:<hash_hex>` — Block 元信息

> **最热的 key，数量最多**（量级：每 SD 1k~100k）

| 属性 | 值 |
|---|---|
| 类型 | **Hash** |
| TTL | 无（生命周期由 `lt`/`state` 管理） |
| 维护方 | C++ `RedisMetaChannel::publish` / `update_block_state_batch` / `delete_blockmeta_batch` |

**Hash 字段**（固定 6 个）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ph` | int64 | parent hash（构造 radix 链） |
| `pb` | int64 | parent block_node_id |
| `nid` | uint32 | 写入者 node_id |
| `hash` | int64 | 自身 hash |
| `lt` | uint32 | lease time（续租时间戳） |
| `state` | int | 0=READY / 1=EVICTED |

**全局 SCAN pattern**：`sd:<sd_key>:block:*`（由 `RedisMetaChannel::list_all_block_keys` 使用）。

---

### 2.5 `sd:<sd_key>:aggregate:<prefix_hash>` — 跨 SD 聚合标记（未实现）

| 属性 | 值 |
|---|---|
| 类型 | **未启用**，且当前没有构造器实现 |

预留 Redis key 命名空间，供未来把 `MasterCoordinator` 的跨 SD 聚合状态持久化到 Redis（用于 Master 重启恢复）。现阶段 `AggregateRadixTree` 只在内存。如需启用，应在 `SharingDomainNamespace` 上重新实现 key 构造器。

---

### 2.6 `sd:<sd_key>:pcfs:<node_id>` — PCFS 文件节点索引

| 属性 | 值 |
|---|---|
| 类型 | **List** |
| 维护方 | `RedisMeta.add_pcfs_file_nodeids` / `load_pcfs_file_nodeids` |
| 含义 | 记录本节点能读到的 PCFS 文件对应的 node_id 列表（用于 3rd remote） |

---

## 3. Instance 维度 key（每实例共享一份）

### 3.1 `flexkv:instance:<instance_id>:session` — 实例会话（故障检测）

| 属性 | 值 |
|---|---|
| 类型 | **JSON string + TTL** |
| TTL | `CacheConfig.instance_session_ttl_seconds`（默认 8s） |
| 维护方 | `RedisSessionClient.register` / `renew` / `unregister` |
| 读取方 | `FailureDetector.poll_once()`（peer instance 跨实例扫描） |

**JSON payload**：

```json
{
  "instance_id": "<instance_id>",
  "epoch": "<monotonic uuid>",
  "master_zmq_addr": "tcp://ip:port",
  "node_ids": [123, 124, 125, ...],
  "mooncake_addrs_by_sd": {"<sd_key_str>": "tcp://ip:port", ...}
}
```

**故障判定**：
- Peer 观察到 key 消失（TTL 到期）→ 触发 `on_peer_lost(peer_instance_id)`
- Peer 观察到 `epoch` 字段变化 → 视为重启事件
- **即使 session 漏报**，数据面的 Mooncake P2P 失败会兜底（通过 `FailureReportMsg`）

---

### 3.2 `flexkv:instance:<instance_id>:sd_nodes` — 实例 SD→节点映射

| 属性 | 值 |
|---|---|
| 类型 | **Hash** |
| TTL | 无 |
| 维护方 | Master 启动时 `RedisMeta.register_instance_sd_nodes(instance_id, sd_to_nid)` 写入一次 |
| 读取方 | 其他实例的 `DistributedRadixTree.remote_tree_refresh` |

**Hash 字段**（field = sd_key 字符串，value = 该 SD 所在节点的 node_id）：

跨节点 PP=2 的例子：

```
"c3a2f91d0bcdef01:ppn0/2:tpn0/1:nsa0"  ->  50      # 第 0 节点
"c3a2f91d0bcdef01:ppn1/2:tpn0/1:nsa0"  ->  51      # 第 1 节点
```

跨机 TP=2 的例子：

```
"c3a2f91d0bcdef01:ppn0/1:tpn0/2:nsa0"  ->  60      # 第 0 节点
"c3a2f91d0bcdef01:ppn0/1:tpn1/2:nsa0"  ->  61      # 第 1 节点
```

> 在 sd_key 不变量 `pp_node_count × tp_node_count == nnodes` 的约束下，每个 sd_key 对应**唯一的物理节点**。同节点上的 cp_rank>0 worker 不在这里出现（CP 不进 sd_key），CPU pool 内容由 sync_leader 那一份代表。

---

## 4. 全局 key

### 4.1 `global:node_id` — 全局计数器

| 属性 | 值 |
|---|---|
| 类型 | **String 计数器（INCR）** |
| 作用域 | **所有 SD / 所有实例共用** |
| 维护方 | `RedisNodeInfo.register_node` 里 `INCR global:node_id` |

`node_id` 全局唯一保证 `BlockMeta.nid` 在 Redis 跨 SD 查询时不会歧义。

---

### 4.2 `flexkv_node_id_updated:<sd_key_str>` — Pub/Sub channel

| 属性 | 值 |
|---|---|
| 类型 | **Pub/Sub channel**（非 key） |
| 作用域 | 每 SD 一个 |
| 用途 | SD 内其他节点订阅此 channel，实时得到"新节点加入"事件 |

---

## 5. 读写时序速查

### 5.1 节点启动（Master 或 Remote 都走这条）

```
INCR global:node_id                              → 取到 nid
HSET sd:<sd>:node:<nid> ip=... uuid=... status=active sd_key=<sd_str> ...
EXPIRE sd:<sd>:node:<nid> <ttl>
HSET sd:<sd>:meta:<nid> addr=... zmq_addr=... cpu_buffer_ptr=...
HSET sd:<sd>:buffer:<nid>:<ptr> buffer_size=...  (1 次 per buffer)
PUBLISH flexkv_node_id_updated:<sd> <nid>         → 通知同 SD 其他节点
```

`<sd>` 形如 `c3a2:ppn0/2:tpn0/1:nsa0`。

### 5.2 Master 收齐 Remote ready 后（启动最后一步）

```
HSET flexkv:instance:<id>:sd_nodes
     ppn0/2:tpn0/1:nsa0  nid_master
     ppn1/2:tpn0/1:nsa0  nid_remote_pp
     ...
SET flexkv:instance:<id>:session <json> EX <ttl>  → 启动心跳线程
```

### 5.3 KVCache PUT（block 就绪）

```
HSET sd:<sd>:block:<nid>:<hash_hex>
     ph=... pb=... nid=... hash=... lt=... state=0
```

PUT 阶段每个 block D2H 完成后由 Master 通过 `post_complete_callback` 触发上述 publish。

### 5.4 跨 SD 聚合（多 SD 部署下）

跨 SD 协调本身**不读写 Redis**——它走 ZMQ + `TransferOpGraph` 派发链路。Redis 只承担 §5.3 的 block 元数据 publish + 启动期的 ready handshake。

```
Master 端 (kvtask.py::_launch_task)
  for handle in transfer_handles:          # master in-proc + N 个 remote handle
      handle.submit(transfer_graph, ...)   # 同一份 graph 广播给所有 SD

Remote 端 (transfer_manager.py::_handle_submit)
  按 target_node_ids 过滤掉不归本节点的 op
  rebind 把 op.pp_rank 改写到本地
  TransferEngine.submit(graph)             # 提交本地执行

Worker 端：
  - D2H clone：完成后回 CompletedOp(sd_key, contributing_node_id)
  - PEERH2H clone：按 src_block_node_ids 分组 → get_node_meta(peer_node)
                  → mooncake.transfer_sync_read → 完成后回 CompletedOp

Master 端 polling worker
  收到 CompletedOp(sd_key=...) → MasterCoordinator.mark_sd_ready(...)
```

### 5.5 远端 radix 重建（`DistributedRadixTree.remote_tree_refresh`）

```
HGETALL flexkv:instance:<peer>:sd_nodes         → sd_key → nid map
for sd in map:
  SCAN sd:<sd>:block:*
  pipeline HMGET (ph pb nid hash lt state)  batch 500
```

接口 `RedisMetaChannel::list_all_block_keys` / `load_metas_by_keys(batch_size)` 已就位。

---

## 6. 典型部署下的 key 量级估算

以 `CP=8, 跨节点 PP=2, tp_node_count=1`（共 2 个物理节点）为例：

| key 种类 | 量级 |
|---|---|
| `sd:*:node:*` | **2 条**（每节点 1 个 SD） |
| `sd:*:meta:*` | 2 条 |
| `sd:*:buffer:*` | 2~6 条（取决于每 SD 注册的 buffer 数） |
| `sd:*:block:*` | **1k~100k/SD × 2 SD ≈ 2k~200k**（主要数据） |
| `flexkv:instance:*:*` | 2 条 per 实例（session + sd_nodes） |
| `global:node_id` | 1 条 |

**单实例峰值估算**：block key 约 **20 万量级**，远低于 Redis 单实例百万级的舒适区。

> CP 维度由于不进 sd_key 而被折叠到同一 SD，相同物理资源下 SD 数量大幅减少（直接等于物理节点数 `nnodes`）。

---

## 7. 运维清单

### 7.1 清空本实例所有 key（推荐）

```bash
# 前提：CacheConfig.flexkv_redis_db = 15（建议 FlexKV 独占一个 db）
redis-cli -n 15 FLUSHDB
```

### 7.2 只清某个实例（实例级隔离）

```bash
# 清 instance 级 key
redis-cli --scan --pattern "flexkv:instance:<id>:*" | xargs redis-cli DEL

# 清该实例下所有 SD 的 key（先拿到 sd_key list）
for sd in $(redis-cli HKEYS flexkv:instance:<id>:sd_nodes); do
  redis-cli --scan --pattern "sd:${sd}:*" | xargs redis-cli DEL
done
```

### 7.3 清某个 SD 的所有 key

```bash
redis-cli --scan --pattern "sd:<sd_key>:*" | xargs redis-cli DEL
# 例：redis-cli --scan --pattern "sd:c3a2*:ppn0/2:tpn0/1:nsa0:*" | xargs redis-cli DEL
```

### 7.4 诊断：看某个实例的健康度

```bash
# 1. session 是否活着（TTL > 0）
redis-cli TTL flexkv:instance:<id>:session

# 2. 有多少个 SD 已注册（应当等于 nnodes）
redis-cli HLEN flexkv:instance:<id>:sd_nodes

# 3. 所有 SD 是否都有节点在线
for sd in $(redis-cli HKEYS flexkv:instance:<id>:sd_nodes); do
  nid=$(redis-cli HGET flexkv:instance:<id>:sd_nodes "$sd")
  echo -n "SD=$sd node=$nid node_ttl="
  redis-cli TTL "sd:$sd:node:$nid"
done
```

---

## 8. 常见问题

**Q1：我想让 FlexKV 用独立 db 不影响其他服务。**
设置 `CacheConfig.flexkv_redis_db = 15`，所有 FlexKV key 都落在 db=15；运维 `redis-cli -n 15 FLUSHDB` 一把清。
Python 端和 C++ 端都会真实发 `SELECT <db>`，详见 `flexkv/common/dist_reuse/failure_detector.py::make_redis_client_from_cache_config`。

**Q2：block key 太多导致 SCAN 卡。**
`RedisMetaChannel::list_all_block_keys` 用全局 SCAN（按 `sd:<sd_key>:block:*` 模式扫描）+ 大批量 pipeline (batch=500) 加载 metadata，避免逐 node 单条 round-trip。

**Q3：TTL 过期了，但数据仍被读出？**
Redis TTL 到期不保证立刻被后台清理（惰性失效 + 定期扫描两种策略叠加）。
FlexKV 在 `_cleanup_stale_nodes_by_ip` 里用 `uuid` 字段区分同 IP 重启前后的节点，避免误读老数据。

**Q4：sd_key 中没有 `pp_rank` 字段，怎么知道某个节点上具体跑哪个 PP rank？**
sd_key 描述的是"节点 KV 物理切片形态"，并不直接编码 PP rank。具体 PP rank 由 sglang launcher 通过启动参数决定，FlexKV 只关心"本节点 KV 物理切片对应的 ppn{idx}/{count}"。

**Q5：跨节点 PP=2 实例与单机 PP=1 实例之间能 P2P 复用 KV 吗？**
不能。跨节点 PP=2 节点 0 的 sd_key 是 `ppn0/2:tpn0/1`，单机 PP=1 的 sd_key 是 `ppn0/1:tpn0/1`，二者 `pp_node_count` 不同，sd_key 字符串不相等 → 不在同一共享域。物理上前者 CPU pool 只装前半 layer，后者装完整 L 层 layer，block 物理 size 也不兼容。
