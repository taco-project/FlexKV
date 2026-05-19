# FlexKV Dist-Reuse Redis Schema 手册

> 所有 key 都走 `sd:<sd_key>:*` 或 `flexkv:instance:*` 两套命名空间（一刀切新布局）。
>
> `<sd_key>` 的序列化形式（**简化方案，CP 不进 sd_key**）：
> `<model_id>:pp<rank>/<size>:tpn<idx>/<count>:nsa<0|1>`
>
> 例：`c3a2f91d0bcdef01:pp0/2:tpn0/1:nsa0`
>
> 逻辑 db：由 `CacheConfig.flexkv_redis_db` 指定（默认 0）。

---

## 1. 命名空间一览

| 命名空间 | 作用域 | 配置 key |
|---|---|---|
| `sd:<sd_key>:*` | 每个 SD 独立 | 每实例有 `pp_size × tp_node_count` 份（CP 不再进 SD） |
| `flexkv:instance:<instance_id>:*` | 每个 FlexKV 实例独立 | 跨 SD 共享 |
| `global:node_id` | 全局（跨实例） | 单条计数器 |
| `flexkv_node_id_updated:<sd_key>` | Pub/Sub channel | 非 key，每 SD 一个 |

---

## 2. SD 维度 key（每 SD 独立，共 `pp_size × tp_node_count` 份）

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
| `node_id` | int | 从 `global:node_id` INCR 取到（所有 SD / 实例共享，全局唯一） |
| `ip` / `local_ip` | str | 本节点监听 IP |
| `uuid` | str | 进程 UUID（防同 IP 重启后留下"鬼节点"） |
| `status` | str | `"active"` |
| `timestamp` | int | 注册时的 Unix 时间戳（秒） |
| `sd_key` | str | 冗余存本 SD 序列化形式，便于运维排查 |

---

### 2.2 `sd:<sd_key>:meta:<node_id>` — 节点地址元信息

| 属性 | 值 |
|---|---|
| 类型 | **Hash** |
| TTL | 无（生命周期跟随 `node:<id>` 的 TTL；节点死了 meta 保留是 ok 的） |
| 维护方 | `RedisMeta.regist_node_meta(...)` |

**Hash 字段**：

| 字段 | 含义 |
|---|---|
| `node_id` | int |
| `addr` | str，节点 IP |
| `zmq_addr` | str，`tcp://ip:port`；Master 发 coord 指令时使用 |
| `cpu_buffer_ptr` | int，Mooncake P2P 读取的 CPU block 池首地址 |
| `ssd_buffer_ptr` | int，SSD block 池首地址（如有） |

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

**最热的 key，数量最多（量级：每 SD 1k~100k）**

| 属性 | 值 |
|---|---|
| 类型 | **Hash** |
| TTL | 无（生命周期由 `lt`/`state` 管理，由 C++ `RedisMetaChannel` 读写） |
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

> **注**：早期方案曾在 leaf block 上额外写 `cp_map` 字段，用于 CP > 1 时跨 cp_rank 的 leaf 映射；简化方案中 CP 不进 sd_key（各 cp_rank 的 KV pool 物理对等，由 attention all-gather 保证），`cp_map` 已经移除。

**全局 SCAN pattern**：`sd:<sd_key>:block:*`（由 `RedisMetaChannel::list_all_block_keys` 使用，替代逐 node SCAN）。

---

### 2.5 `sd:<sd_key>:aggregate:<prefix_hash>` — 跨 SD 聚合标记（预留）

| 属性 | 值 |
|---|---|
| 类型 | **未启用**（`SharingDomainNamespace.aggregate_key(...)` 已提供构造器） |

用途：文档 §4.7 保留。未来把 `MasterCoordinator` 的跨 SD 聚合状态持久化到 Redis（用于 Master 重启恢复）。现阶段 `AggregateRadixTree` 只在内存。

---

### 2.6 `sd:<sd_key>:pcfs:<node_id>` — PCFS 文件节点索引

| 属性 | 值 |
|---|---|
| 类型 | **List** |
| 维护方 | `RedisMeta.add_pcfs_file_nodeids` / `load_pcfs_file_nodeids` |
| 含义 | 记录本节点能读到的 PCFS 文件对应的 node_id 列表（用于 3rd remote） |

---

## 3. Instance 维度 key（跨 SD，每实例共享一份）

### 3.1 `flexkv:instance:<instance_id>:session` — 实例会话（Layer-1 故障检测）

| 属性 | 值 |
|---|---|
| 类型 | **JSON string + TTL** |
| TTL | `CacheConfig.instance_session_ttl_seconds`（默认 8s） |
| 维护方 | `RedisSessionClient.register` / `renew` / `unregister` |
| 读取方 | `FailureDetector.poll_once()`（跨实例扫描） |

**JSON payload**：

```json
{
  "instance_id": "<instance_id>",
  "epoch": "<monotonic uuid>",
  "master_zmq_addr": "tcp://ip:port",
  "node_ids": [123, 124, 125, ...],
  "mooncake_addrs_by_sd": {"sd_key_str": "tcp://ip:port", ...}
}
```

**故障判定**：
- Peer 观察到 key 消失（TTL 到期）→ 触发 `on_peer_lost(peer_instance_id)`
- Peer 观察到 `epoch` 字段变化（重启）→ 触发 `on_peer_seen` + 视为重启事件
- **即使 Layer-1 漏报**，Layer-2（数据面 Mooncake 失败）会兜底

---

### 3.2 `flexkv:instance:<instance_id>:sd_nodes` — 实例 SD→节点映射

| 属性 | 值 |
|---|---|
| 类型 | **Hash** |
| TTL | 无 |
| 维护方 | Master 启动时 `RedisMeta.register_instance_sd_nodes(instance_id, sd_to_nid)` 写入一次 |
| 读取方 | 其他实例的 `DistributedRadixTree.remote_tree_refresh`（Phase 1-F 里会用 `load_instance_sd_nodes` 批量拉取） |

**Hash 字段**（field = SD key 字符串，value = 该 SD 所在节点的 node_id）：

```
"c3a2f91d0bcdef01:pp0/2:tpn0/1:cp0/4:nsa0"  ->  50
"c3a2f91d0bcdef01:pp1/2:tpn0/1:cp0/4:nsa0"  ->  51
"c3a2f91d0bcdef01:pp0/2:tpn0/1:cp1/4:nsa0"  ->  60
...
```

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

## 5. 读写时序（Cheat Sheet）

### 5.1 节点启动（任何 SD 节点）

```
INCR global:node_id                              → 取到 nid
HSET sd:<sd>:node:<nid> ip=... uuid=... status=active ...
EXPIRE sd:<sd>:node:<nid> <ttl>
HSET sd:<sd>:meta:<nid> addr=... zmq_addr=... cpu_buffer_ptr=...
HSET sd:<sd>:buffer:<nid>:<ptr> buffer_size=...  (1 次 per buffer)
PUBLISH flexkv_node_id_updated:<sd> <nid>         → 通知同 SD 其他节点
```

### 5.2 Master 收齐 Remote ready 后（启动最后一步）

```
HSET flexkv:instance:<id>:sd_nodes
     sd_key_str_1 nid_1
     sd_key_str_2 nid_2
     ...
SET flexkv:instance:<id>:session <json> EX <ttl>  → 启动心跳线程
```

### 5.3 KVCache PUT（block 就绪）

```
HSET sd:<sd>:block:<nid>:<hash_hex>
     ph=... pb=... nid=... hash=... lt=... state=0
```

> 简化方案中 CP 不进 sd_key，所有 cp_rank 的 KV 内容由 attention all-gather 保证物理对等，
> 因此 leaf block 不再写 `cp_map` 字段。

### 5.4 跨 SD 聚合（Phase D，Master 发起）

跨 SD 协调**不走独立的 ZMQ 协议**，统一在 `TransferOpGraph` 上挂多 op 并通过现有 `_launch_task` 广播链路完成。Redis 在这条路径上**不被读/写**——
它只承担 §5.3 的 block 元数据 publish + bootstrap 期的 ready handshake。

```
Master 端 (kvtask.py::_launch_task)
  for handle in transfer_handles:          # master in-proc + N 个 remote handle
      handle.submit(transfer_graph, ...)   # 同一份 graph 广播

Remote 端 (transfer_manager.py::_handle_submit)
  _filter_graph_by_target_node_ids(graph)  # 丢弃 target_node_ids 不含本节点的 op
  _rebind_graph_to_local_worker(graph)     # 把 op.pp_rank 改写到本地
  TransferEngine.submit(graph)             # 提交本地执行

Worker 端（D2H 或 PEERH2H）：
  - D2H clone：完成后回 CompletedOp(sd_key, contributing_node_id, success=True)
  - PEERH2H clone：按 src_block_node_ids 分组 → get_node_meta(peer_node)
                  → mooncake.transfer_sync_read → 完成后回 CompletedOp

Master 端 polling worker (TransferManagerMultiNodeHandle._polling_worker)
  收到 CompletedOp(sd_key=...) → _completion_sink(completed_op)
                              → GlobalCacheEngine._on_peer_sd_completed_op
                              → MasterCoordinator.mark_sd_ready(...)
```

具体的 op 字段约定见
`proposal_unify_with_graph_dispatch_2026-05-15.md` §6.4 + §11；
PUT 路径见 §6.3，GET 路径见 §6.2。

> 历史注记：早期版本曾用独立的 `CoordQueryMsg → BlockIndex → CoordQueryAckMsg → CoordGetCmdMsg → CoordGetAckMsg` / `CoordPutCmdMsg → CoordPutAckMsg`
> 协议层；Phase D-4（2026-05-15）将其整体废弃，合并到 graph 派发链路。
> 详细背景见 `proposal_unify_with_graph_dispatch_2026-05-15.md` §1-§3。

### 5.5 远端 radix 重建（`DistributedRadixTree.remote_tree_refresh`）

**现状**（逐 SD 逐 node SCAN，未做 1-F 优化）：
```
for sd in peer_sds:
  SCAN sd:<sd>:node:*
  for nid in node_ids:
    SCAN sd:<sd>:block:<nid>:*
    pipeline HMGET (ph pb nid hash lt state)  ×  block 数
```

**目标**（Phase 1-F）：
```
HGETALL flexkv:instance:<peer>:sd_nodes         → sd_key → nid map
for sd in map:
  SCAN sd:<sd>:block:*  (全局，不再按 nid)
  pipeline HMGET (ph pb nid hash lt state)  batch 500
```

---

## 6. 典型部署下的 key 量级估算

以 `CP=4, PP=2, tp_node_count=1` 的实例为例：

| key 种类 | 量级 |
|---|---|
| `sd:*:node:*` | 8 条（每 SD 1 个） |
| `sd:*:meta:*` | 8 条 |
| `sd:*:buffer:*` | 8~24 条（取决于每 SD 注册的 buffer 数） |
| `sd:*:block:*` | **1k~100k/SD × 8 SD ≈ 8k~800k**（主要数据） |
| `flexkv:instance:*:*` | 2 条 per 实例（session + sd_nodes） |
| `global:node_id` | 1 条 |

**CP=8 大实例峰值**：block key 约 **10 万量级**，远低于 Redis 单实例百万级的舒适区。

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

# 清该实例下所有 SD 的 key（需要先拿到 sd_key list）
for sd in $(redis-cli HKEYS flexkv:instance:<id>:sd_nodes); do
  redis-cli --scan --pattern "sd:${sd}:*" | xargs redis-cli DEL
done
```

### 7.3 清某个 SD 的所有 key

```bash
redis-cli --scan --pattern "sd:<sd_key>:*" | xargs redis-cli DEL
```

### 7.4 诊断：看某个实例的健康度

```bash
# 1. session 是否活着（TTL > 0）
redis-cli TTL flexkv:instance:<id>:session

# 2. 有多少个 SD 已注册
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
Phase 1-F 的优化会把"逐 node SCAN + 逐 node pipeline"改成"全局 SCAN + 大批量 pipeline (batch=500)"，显著减少 round-trip。接口 `RedisMetaChannel::list_all_block_keys` / `load_metas_by_keys(batch_size)` 已就位。

**Q3：TTL 过期了，但数据仍被读出？**
Redis TTL 到期不保证立刻被后台清理（惰性失效 + 定期扫描两种策略叠加）。
FlexKV 在 `_cleanup_stale_nodes_by_ip` 里用 `uuid` 字段区分同 IP 重启前后的节点，避免误读老数据。
