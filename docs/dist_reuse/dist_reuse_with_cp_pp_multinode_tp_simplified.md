# FlexKV 分布式 KVCache 共享原理

> **本文目的**：给用户讲清楚 FlexKV 的 `dist_reuse`（跨实例 KVCache 共享）是怎么工作的——什么样的实例之间能共享 KV、Master/Remote 各自做什么、跨实例 P2P 是怎么发起的。
>
> 阅读对象：FlexKV 用户、运维、上层框架（如 sglang）的接入开发者。

---

## 1. 一句话概括

**两个 FlexKV 实例只有在"KV 物理切片形态完全一致"时才能直接 P2P 复用对方的 KV——所谓"切片形态一致"，就是节点级别的 layer 段相同、KV head 段相同、模型 layout 相同。**

由此引申出三个关键概念：

| 概念 | 含义 |
|---|---|
| **共享域**（Sharing Domain，SD） | 一组"KV 切片形态完全一致"的节点的集合。同 SD 节点之间可直接 P2P 互拷 block。 |
| **sd_key** | 共享域的字符串身份。两个节点 sd_key 相同 ⇔ 处于同一共享域 ⇔ 可 P2P。 |
| **Master / Remote** | 一个 FlexKV 实例由 1 个 Master + 多个 Remote 组成。Master 是控制面唯一事实来源，Remote 只搬数据。 |

下面分别展开。

---

## 2. 哪些维度会影响"切片形态"

```
┌──────────────────┬──────────────┬──────────────────────────┐
│ 维度              │ 切的是什么    │ 是否影响节点 KV 物理形态？│
├──────────────────┼──────────────┼──────────────────────────┤
│ 跨节点 PP         │ layer 维度    │ ✅ 影响——节点持有的 layer 段不同 │
│ 跨机 TP           │ KV head 维度  │ ✅ 影响——节点持有的 KV head 段不同 │
│ CP（普通 / NSA）  │ 序列计算量    │ ❌ 不影响——attention all-gather 后各 cp_rank 的 KV pool bit-wise 一致 │
│ 模型 / dtype      │ 整个 layout   │ ✅ 影响——layout 不同，block 物理 size 不同 │
│ NSA vs 非 NSA     │ block layout  │ ✅ 影响——NSA 多了一份 indexer K cache buffer │
└──────────────────┴──────────────┴──────────────────────────┘
```

### 关于 CP 的关键事实

CP（Context Parallelism）只是把序列拆给不同 cp_rank 算 query；attention 层做了 all-gather 之后，**每个 cp_rank 自己的 KV pool 写入的都是完整全序列的 KV，且各 cp_rank 之间 bit-wise 一致**。

代码事实：
- 普通 CP：`flashattention_backend.py::cp_allgather_and_save_kv_cache` 完成 all-gather
- NSA CP：`deepseek_v2.py::rebuild_cp_kv_cache` 在 attention 之前做 all-gather；`nsa_indexer.py:1333-1347` 额外对 indexer K 做 `cp_all_gather_rerange_output`
- KV head 切分由 `attn_tp_size` 单独承担（`parallel_state.py:1860-1862`），CP 不切 head

所以 CP 维度**不参与共享域划分**——同一节点上所有 cp_rank 的 KV pool 物理对等，跨实例 cp=i ↔ cp=j 直接互拷在数据层面合法。

---

## 3. sd_key 格式与含义

### 3.1 序列化形式

```
<model_id>:ppn<pp_node_idx>/<pp_node_count>:tpn<tp_node_idx>/<tp_node_count>:nsa<0|1>
```

例：

```
c3a2f91d0bcdef01:ppn0/1:tpn0/1:nsa0       — 单机 PP=1 部署
c3a2f91d0bcdef01:ppn0/2:tpn0/1:nsa0       — 跨节点 PP=2 的第 0 节点
c3a2f91d0bcdef01:ppn1/2:tpn0/1:nsa0       — 跨节点 PP=2 的第 1 节点
c3a2f91d0bcdef01:ppn0/1:tpn0/2:nsa0       — 跨机 TP=2 的第 0 节点
c3a2f91d0bcdef01:ppn0/1:tpn1/2:nsa0       — 跨机 TP=2 的第 1 节点
c3a2f91d0bcdef01:ppn0/1:tpn0/1:nsa1       — 单机 PP=1 NSA 模型
```

### 3.2 字段含义

| 字段 | 含义 |
|---|---|
| `model_id` | 模型 + 数值精度 + page_size 等的指纹。同模型同配置才能复用。 |
| `pp_node_idx` | 本节点是 PP 维度上的第几台**物理节点**（0 起）。 |
| `pp_node_count` | PP 维度跨了几台**物理节点**。 |
| `tp_node_idx` | 本节点是 TP 维度上的第几台**物理节点**（0 起）。 |
| `tp_node_count` | TP 维度跨了几台**物理节点**。 |
| `nsa<flag>` | 是否 NSA 模型（NSA 与非 NSA block 物理 layout 不同，必须隔离）。 |

### 3.3 字段派生公式

```python
pp_node_count = max(min(pp_size, nnodes), 1)           # PP 维度跨了几台节点
pp_node_idx   = pp_rank // max(pp_size // nnodes, 1)   # 本节点在 PP 维度的位置

tp_node_count = nnodes_per_tp_group                    # TP 维度跨了几台节点
tp_node_idx   = tp_rank // tp_size_per_node            # 本节点在 TP 维度的位置
```

### 3.4 不变量

```
pp_node_count × tp_node_count == nnodes
```

也就是说：**共享域数量 = 物理节点数**——每个物理节点对应唯一一个 sd_key，同实例不同节点之间 sd_key 不同。

### 3.5 sd_key 的核心语义

> **sd_key 相同 ⇒ 节点 KV 物理切片形态完全相同 ⇒ 节点间 block 可直接 P2P。**

注意 sd_key 中**不包含 cp_rank**——这是对 §2"CP 不影响切片形态"事实的直接体现。同一节点上不同 cp_rank 的 worker 都属于同一个 sd_key，CPU pool 共用一份。

---

## 4. 共享域数量的几个典型例子

| 部署 | nnodes | pp_size | tp_size | sd_key 集合 | SD 数量 |
|------|---|---|---|---|---|
| 单机 PP=1 TP=1/2/4/8 | 1 | 1 | 1~8 | `ppn0/1:tpn0/1` | **1** |
| 跨节点 PP=2（每节点 PP=1） | 2 | 2 | 任意 | `ppn0/2:tpn0/1` + `ppn1/2:tpn0/1` | **2** |
| 跨机 TP=16（PP=1, 2 节点） | 2 | 1 | 16 | `ppn0/1:tpn0/2` + `ppn0/1:tpn1/2` | **2** |
| 跨节点 PP=2 × 跨机 TP=2（4 节点） | 4 | 2 | 16 | 4 个 ppn × tpn 笛卡尔积 | **4** |

CP=4 / CP=8 不会增加 SD 数量，但 CP 内部的多张 GPU 仍要通过 H2D 接收同一份 CPU pool 的数据。

---

## 5. Master / Remote 架构

### 5.1 角色分工

```
┌──────────────────────────────────────────────────────────┐
│  Master (pp_node_idx=0, tp_node_idx=0, cp_rank=0)        │
│                                                          │
│  控制面（唯一事实来源）：                                  │
│  - KVManager + CacheEngine（唯一的 LocalRadixTree）       │
│  - 跨 SD 聚合层 radix（多 SD 时的 fully-ready 判定）       │
│  - get_match / put_match / insert / evict 决策            │
│  - block 级 refcount（保护在途 block）                    │
│  - Redis 元数据同步                                       │
│  - 跨 instance 的 remote hit 判定                         │
│                                                          │
│  决策后通过两条通道下发：                                  │
│  - 维度内（CP / TP / PP）：sglang 现有 broadcast / scatter  │
│  - 跨 SD：FlexKV TransferOpGraph 派发                     │
├──────────────────────────────────────────────────────────┤
│  Remote (pp_node_idx>0 / tp_node_idx>0)                  │
│                                                          │
│  数据面：                                                 │
│  - TransferManagerOnRemote + RedisMeta + Mooncake         │
│  - 接收 Master 派发的 TransferOpGraph                     │
│  - 本地过滤出归本节点的 op，执行 GPU↔CPU / P2P 传输       │
│  - 完成后通过 CompletedOp 回报                            │
│  - 不维护任何 radix 索引，不做任何缓存决策                  │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Remote 的两种类型

按 sd_key 维度区分：

| Remote 类型 | 触发条件 | 持有的 KV 切片 |
|---|---|---|
| **PP-Remote** | `pp_node_idx > 0` | layer 切片的后半段 |
| **TP-Remote** | `tp_node_idx > 0` | KV head 切片的后半段 |

> CP 维度的 cp_rank > 0 worker 在 dist_reuse 视角下与 Master 处于同一 SD（CP 不进 sd_key）。它们仍然要把 GPU buffer 注册到 FlexKV 以便接收 H2D 指令，但**不向 Redis 单独注册 SD 节点身份**——CPU pool 中数据由 sync_leader（cp_rank=0）那一份代表整组。

### 5.3 同实例 Master 如何统一控制 Remote

```
sglang sync_leader (cp_rank=0)
        │
        ▼
   Master KVManager      ─── TransferOpGraph 派发 ──►   PP-Remote / TP-Remote
   (本节点 ZMQ 进程)                                      (其他节点 ZMQ 进程)
```

- Master 在内部有一个 in-process handle、对每个 Remote 各有一个 ZMQ handle
- 一次 `_launch_task` 把同一份 `TransferOpGraph` 同时投递给所有 handle
- 每个 handle 收到后：
  1. 按 op 上的 `target_node_ids` 过滤出归自己执行的 op
  2. 对归自己的 op 调用本地 TransferEngine 执行
  3. 完成后回报 `CompletedOp`（带 sd_key + contributing_node_id）
- Master 收齐所有 `CompletedOp` → 标记任务完成

---

## 6. 跨实例 KV 共享流程

下面以两个 FlexKV 实例（同模型同配置）之间的 KV 复用为例，说明 dist_reuse 是怎么发起的。

### 6.1 启动期：互相注册到 Redis

每个实例启动时：

1. Master 和每个 Remote 各自在 Redis 上 `INCR global:node_id` 拿到全局唯一的 `node_id`
2. 在自己 sd_key 命名空间下注册 `sd:<sd_key>:node:<node_id>` 心跳 key（带 TTL）
3. 在 `sd:<sd_key>:meta:<node_id>` 写入自己的 ZMQ 地址、Mooncake CPU buffer 指针等
4. Master 收齐所有 Remote ready 后，把"sd_key → node_id"映射汇总到 `flexkv:instance:<instance_id>:sd_nodes`

这之后，**任何一个 instance 都能通过扫描 Redis 知道其他 instance 在每个 SD 上的 node_id**。

### 6.2 PUT：把推理产生的 KV 存到分布式视角

```
1. 推理结束 → sync_leader 在全序列 token_ids 上做 put_match
2. Master 决策：哪些 block 需要 D2H 落到 CPU pool
3. Master 构造 TransferOpGraph：
   每个 SD 各挂一个 D2H op，target_node_ids 指向该 SD 自己的节点
4. 通过 _launch_task 把同一份 graph 派发给所有 handle
5. 每个 SD（master in-proc 或 remote）执行各自归属的 D2H op
6. D2H 完成 → 触发 post_complete_callback：
   - 把本批 block 的元数据 publish 到 sd:<sd_key>:block:<nid>:<hash> （Redis）
   - 给 Master 回 CompletedOp(sd_key, contributing_node_id)
7. Master 收齐所有 SD 的 CompletedOp → mark_sd_ready
   全 SD ready → 该前缀进入 fully-ready，对外可被 reuse
```

PUT 的副作用：本实例的 KV 元数据通过 Redis 让其他 instance 可见。

### 6.3 GET：从其他实例拉取已有的 KV

```
1. 新请求到达 → sync_leader 在全序列 token_ids 上做 get_match
2. 命中本实例 fully-ready 前缀 → 直接 reuse（local hit）
3. 若 miss，查 DistributedRadixTree（基于其他 instance 同步过来的 Redis 元数据）：
   - 命中 → 知道哪个 peer instance 的哪个 node_id 持有该 block
4. Master 构造跨实例 GET TransferOpGraph，每个 SD 各挂一个 PEERH2H op：
   src_block_node_ids = [peer_inst.该 SD 上的 node_id]
   target_node_ids    = [self.该 SD 上的 node_id]
5. 通过 _launch_task 派发给所有 handle
6. 每个 SD 上的 worker：
   - 按 src_block_node_ids 分组
   - 从 Redis 查到 peer_node 的 zmq_addr / Mooncake addr
   - 发起 Mooncake transfer_sync_read，把 peer 的 CPU block 直接 RDMA 到本节点 CPU pool
7. 全部 op 完成 → Master 触发后续 H2D，把 CPU pool 的内容刷到 GPU
8. CP 维度通过 sglang 现有 broadcast/scatter 把 H2D 结果分发到所有 cp_rank 的 GPU
```

GET 的关键点：
- 跨实例 P2P 完全走 Mooncake RDMA，不经过 Master 中转
- 单 Node 匹配约束（§7.4）保证一次 match 结果只来自一个 peer instance，避免多 peer 并发导致的复杂度

---

## 7. 几个关键设计简化

### 7.1 跨 SD 协调走统一的 graph 派发链路

Master 不为 dist_reuse 单独引入协议层；所有跨 SD 协调（PUT/GET）都表达成 `TransferOpGraph` 上挂多个带节点身份标签的 op，复用现有的跨节点 PP / 跨机 TP graph 派发链路。

`TransferOp` 上的两个关键字段：

| 字段 | 含义 |
|---|---|
| `src_block_node_ids` | 每个 src block 来自哪个 distributed_node_id（worker 内部按 peer 分组发 RDMA） |
| `target_node_ids` | 这条 op 归哪些 SD 的节点执行；Remote 在 `_handle_submit` 阶段按 `target_node_ids` 过滤掉不属于自己的 op |
| `post_complete_callback` | op 完成后在 master 进程上下文执行的回调（如 redis publish） |

`pp_rank` 字段保持原义（路由到本地 PP worker），跟 `target_node_ids` 正交。

### 7.2 CompletedOp 携带 sd_key 标签

每个 op 完成后，Remote 通过 result_socket 把 sd_key + contributing_node_id 跟 `CompletedOp` 一起发回 Master。Master polling worker 用 sd_key 路由到对应的 SD-ready 处理逻辑（如 `mark_sd_ready`）。

### 7.3 跨 SD 聚合一致性

> 仅 SD 数量 > 1 时（`nnodes > 1`）涉及。

一个请求的"完整 KV reuse"要求**所有共享域都命中**（缺一不可）。

Master 维护一个**跨 SD 聚合层 radix tree**：每个 block 的状态从 "ready / not-ready" 扩展为 "ready on SD(0) / ready on SD(1) / ... / fully ready"。只有 fully ready 的 block 才对外表现为"可 reuse"。

```
PUT 流程：所有 SD 都通过 CompletedOp 回报后才标记 fully ready
GET 流程：只对 fully ready 的前缀返回 hit
EVICT 流程：Master 单方面 evict（跳过 refcount > 0 的 block），不通知 Remote
```

**Master 单方面 evict 的合理性**：Master 的 radix tree 是唯一的索引——Master evict 后不再有任何请求会去读那些 block，Remote 上的孤儿数据不影响正确性，最终会被新数据自然覆盖。

### 7.4 DistributedRadixTree 单 Node 匹配约束

`DistributedRadixTree::match_prefix` 一次匹配的所有 block 限定来自单个 peer Node（同一个 `node_id`）。
- 匹配过程中锁定第一个有效 block 的 `node_id`，后续 block `node_id` 不同则停止匹配
- 命中率影响极小（同一请求的 KV 通常整体写入到同一 Node）
- 让跨 SD GET 可以直接确定唯一 peer instance，构图大幅简化

---

## 8. 故障模型

基于"共命运"假设：同 instance 的 Master 和所有 Remote 在进程生命周期上共命运，任一 rank crash 会导致整 instance 秒级内全部退出。

故障只剩两类：

1. **同 instance 整体退出/重启**：
   - Master 维护 `flexkv:instance:<id>:session` 这个 TTL key + epoch
   - peer instance 的 `FailureDetector` 观察到 key 消失或 epoch 变化 → 批量 invalidate

2. **跨 instance 链路故障**（Mooncake P2P read 失败）：
   - Worker 通过 `FailureReportMsg` 异步上报到 Master
   - Master 单前缀 invalidate + fallback 到正常 prefill

---

## 9. 部署形态速查

### 9.1 端口拓扑

跟跨机 TP/PP 现有部署一致——`FLEXKV_MASTER_HOST` + `FLEXKV_MASTER_PORTS=5556,5557,5558`，所有 Remote 通过 ZMQ identity 区分。**不需要 per-SD 端口、不需要 sglang launcher 做 endpoint 发现**。

### 9.2 共享域数量上限

```
SD 数量 = pp_node_count × tp_node_count = nnodes
```

当前部署上限 `nnodes ≤ 2`，所以 SD 数量 ≤ 2；保留对未来 4 节点（`pp_node_count=2 × tp_node_count=2`）的扩展支持。

### 9.3 跨实例配对规则

只有满足下面**所有条件**的两个实例之间才能 P2P 复用 KV：

- 同 `model_id`（模型 / dtype / page_size 一致）
- 同 `pp_node_count` 和 `tp_node_count`（节点切片维度一致）
- 同 `is_nsa`（block 物理 layout 一致）

且具体配对发生在节点级别——`inst1.ppn=i:tpn=j` 只与 `inst2.ppn=i:tpn=j` 互拷，节点身份必须严格对齐。

---

## 10. 一张图总结

```
                        Instance 1                                    Instance 2
┌─────────────────────────────────────────┐    ┌─────────────────────────────────────────┐
│                                         │    │                                         │
│  Master (ppn=0, tpn=0, cp=0 = sync_leader)│  │  Master (ppn=0, tpn=0, cp=0)           │
│  ┌─────────────────────────────────┐    │    │  ┌─────────────────────────────────┐    │
│  │ KVManager + CacheEngine         │    │    │  │ KVManager + CacheEngine         │    │
│  │ + LocalRadixTree (唯一索引)      │    │    │  │ + LocalRadixTree                │    │
│  │ + 跨 SD 聚合层 radix             │◄───┼────┼──┤ + 跨 SD 聚合层 radix             │    │
│  │ + RedisMeta + Mooncake          │    │    │  │ + RedisMeta + Mooncake          │    │
│  └────────────┬────────────────────┘    │    │  └─────────────────────────────────┘    │
│               │ TransferOpGraph 派发     │    │                                         │
│               │                          │    │                                         │
│               ▼                          │    │                                         │
│  PP-Remote (ppn=1) ◄── Mooncake P2P ──►│    │  PP-Remote (ppn=1)                     │
│  TP-Remote (tpn=1) ◄── Mooncake P2P ──►│    │  TP-Remote (tpn=1)                     │
│                                         │    │                                         │
│  cp=1..N-1（同 SD，仅 GPU 注册）         │    │  cp=1..N-1                              │
└─────────────────────────────────────────┘    └─────────────────────────────────────────┘

同 sd_key 的节点之间通过 Mooncake P2P 互拷 KV block
（如 inst1.ppn=1 ↔ inst2.ppn=1，但 inst1.ppn=0 ↮ inst2.ppn=1）
跨 SD 不允许 P2P（layer / KV head 切片不同）
CP 维度不参与 SD 划分；各 cp_rank 的 KV pool 由 attention all-gather 保证 bit-wise 一致
Master 是控制面唯一事实来源，Remote 只搬数据
跨 SD 协调统一通过 TransferOpGraph 派发完成
```

---

## 附录：sd_key 字段速查

```
sd_key 文本格式：
    <model_id>:ppn<pp_node_idx>/<pp_node_count>:tpn<tp_node_idx>/<tp_node_count>:nsa<0|1>

字段含义：
    model_id          —— 模型 + dtype + page_size 的指纹
    pp_node_idx       —— 本节点是 PP 维度的第几台节点（0 起）
    pp_node_count     —— PP 维度跨了几台物理节点
    tp_node_idx       —— 本节点是 TP 维度的第几台节点（0 起）
    tp_node_count     —— TP 维度跨了几台物理节点
    nsa<flag>         —— 是否 NSA 模型（NSA 与非 NSA 必须隔离）

不变量：
    pp_node_count × tp_node_count == nnodes

派生：
    pp_node_count   = max(min(pp_size, nnodes), 1)
    pp_node_idx     = pp_rank // max(pp_size // nnodes, 1)
    tp_node_count   = nnodes_per_tp_group
    tp_node_idx     = tp_rank // tp_size_per_node

示例：
    单机 PP=1 TP=8                        c3a2:ppn0/1:tpn0/1:nsa0
    跨节点 PP=2（每节点 PP=1）节点 0      c3a2:ppn0/2:tpn0/1:nsa0
    跨节点 PP=2（每节点 PP=1）节点 1      c3a2:ppn1/2:tpn0/1:nsa0
    跨机 TP=16（PP=1）节点 0              c3a2:ppn0/1:tpn0/2:nsa0
    跨机 TP=16（PP=1）节点 1              c3a2:ppn0/1:tpn1/2:nsa0
    NSA 单机 PP=1 TP=8                    c3a2:ppn0/1:tpn0/1:nsa1
```
