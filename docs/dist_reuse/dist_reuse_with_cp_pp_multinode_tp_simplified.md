# FlexKV 分布式 KVCache 重用（CP / PP / 跨机 TP）—— 核心原理精简版

> **本版本**（2026-05-15 重写）：跨 SD 协调统一走 graph 派发链路，废弃独立的 `coord_query / coord_get / coord_put` 协议层。详细背景见 `proposal_unify_with_graph_dispatch_2026-05-15.md`。
>
> 本文档是 `dist_reuse_with_cp_pp_multinode_tp.md` 的精简版，只阐述核心原理和关键设计决策，省略代码级细节、Redis key 布局、C++ 接口变更、部署脚本等实施层面的内容。完整实施细节请参考原文档。

---

## 1. 问题本质：一句话概括

FlexKV 的 `dist_reuse`（跨实例 KVCache P2P 重用）建立在一个隐含假设上：**每个 FlexKV 实例都是一个完整的、对等的 KV Cache 副本**。而 PP / 跨机 TP 打破了这个假设——它们把一个实例的 KV Cache 按 layer / KV head 维度切成了不对等的碎片。

CP 的情况则需要按代码事实严格区分：

| 维度 | 切的是什么 | 是否影响 P2P 对等性 |
|------|-----------|---------------------|
| PP（PP=2） | layer 维度（每个 PP rank 持有一半 layer） | ✅ 影响——PP rank 之间不对等 |
| 跨机 TP（跨 2 节点） | KV head 维度（每个节点持有一半 head） | ✅ 影响——TP 节点之间不对等 |
| **CP（CP=4/8）** | **只切 query 计算量；attention 层做 all-gather 后每个 cp_rank 本地 pool 写入完整全序列 KV，bit-wise 一致** | **❌ 不影响——所有 cp_rank 物理 KV 内容完全相同** |

CP 的 all-gather 同时发生在普通 attention（`flashattention_backend.py` 的 `cp_allgather_and_save_kv_cache`）和 NSA（`deepseek_v2.py:rebuild_cp_kv_cache` + `nsa_indexer.py` 的 indexer K all-gather）。**两种 CP 模式下，各 cp_rank 的 KV pool 内容都对等**——CP 不切 head，head 切分完全由 `attn_tp_size` 承担（`parallel_state.py:1860-1862`）。

| 场景 | layer_range | kv_head_range | seq_range | 各 rank pool 内容 | 能否跨实例 P2P？ |
|------|------------|---------------|-----------|-------------------|-----------------|
| 单机 TP（无 PP/CP） | 全部 | 全部（TP 聚合后） | 全部 | — | ✅ 对等 |
| 跨机 PP（PP=2） | 一半 layers | 全部 | 全部 | PP rank 间不同 | ❌ 不对等 |
| 跨机 TP（跨 2 节点） | 全部 | 一半 KV heads | 全部 | TP 节点间不同 | ❌ 不对等 |
| **CP（普通/NSA）** | **全部** | **全部** | **全部（all-gather 后）** | **bit-wise 相同** | **✅ 对等** |

**核心矛盾**：`dist_reuse` 是"对等 + 完整"的共享模型；PP / 跨机 TP 真正打破了对等假设；CP 看似切了序列，但 attention all-gather 后每台机本地的 KV pool 又被还原成完整且对等的副本。本方案的简化也基于这一事实。

---

## 2. 解决方案：四个核心原则

### 原则一：共享域（Sharing Domain）

**只有 KV 切片完全一致的节点之间才允许 P2P 重用。**

定义"共享域"为满足以下条件的一组节点：
- layer 集合相同（由 `pp_rank` 决定）
- KV head 切片相同（由 `tp_node_idx` 决定）
- model 和 dtype/page_size 等基本参数一致

CP 维度**不参与 SD 切分**（关键简化）：因为 attention all-gather 后所有 cp_rank 的 pool 内容物理对等，跨 instance 的 cp=i ↔ cp=j 直接互拷在数据上是合法的；让所有 cp_rank 落在同一 SD 内可以最大化命中率并消除多 SD barrier。

共享域 ID：
```
sd_key = "{model}:pp{rank}/{size}:tpn{idx}/{count}:nsa{flag}"
```

`nsa{flag}` 仅用于隔离 NSA 与非 NSA 模型（block 物理 layout 不同），不引入 NSA 特有的 cp 切分维度。

**共享域数量** = `PP × tp_node_count`，本方案约束下最大 `2 × 2 = 4`（与 CP 大小无关）。

### 原则二：统一的 Master/Remote 架构 + Master 是唯一事实来源

**所有控制面决策集中在 Master，所有 Remote 只做数据面搬运；Master 拥有唯一的 radix tree，Remote 不维护任何索引。**

通过分析 sglang 的实现，CP / PP / 跨机 TP 三个维度在控制面/数据面分工上**完全同构**：

```
┌──────────────────────────────────────────────────────────┐
│  Master (pp_rank=0, tp_node_idx=0, cp_rank=0)            │
│                                                          │
│  控制面（唯一事实来源）：                                  │
│  - KVManager + CacheEngine（唯一的 LocalRadixTree）       │
│  - 跨 SD 聚合层 radix（PP / 跨机 TP 维度的聚合）           │
│  - get_match / put_match / insert / evict 决策            │
│  - block 级 refcount（保护在途 block）                    │
│  - Redis 元数据同步                                       │
│  - 跨 instance 的 remote hit 判定                         │
│                                                          │
│  决策后通过两条通道下发：                                  │
│  - 维度内（CP / TP / PP）：sglang 现有 broadcast / scatter  │
│  - 跨 SD（PP / 跨机 TP）：FlexKV TransferOpGraph 派发      │
├──────────────────────────────────────────────────────────┤
│  Remote (cp_rank>0 / tp_node_idx>0 / pp_rank>0)          │
│                                                          │
│  数据面：                                                 │
│  - TransferManagerOnRemote + RedisMeta + Mooncake         │
│  - 接收 Master 派发的 TransferOpGraph                     │
│  - 本地 group 过滤出归本节点的 op，执行 GPU↔CPU / P2P 传输 │
│  - 完成后通过 CompletedOp 回报                            │
│  - 不维护任何 radix 索引，不做任何缓存决策                  │
│                                                          │
│  三种 Remote 行为完全一致，仅 sharing_domain_key 维度不同   │
└──────────────────────────────────────────────────────────┘
```

**为什么能这样简化？** 因为 sglang 中 CP 的 `sync_leader` 机制已经把控制面集中在 `cp_rank=0`：

```python
# sglang flexkv_connector.py 核心逻辑
self.is_sync_leader = (self.cp_rank if self.cp_size > 1 else self.tp_rank) == 0

# 只有 sync_leader 才创建 KVManager
if self.is_sync_leader:
    self.kv_manager = KVManager(...)
```

跨机 TP 和跨机 PP 也是同样的模式——只有 Master 节点创建 `KVManager`，Remote 节点只做数据搬运。因此三个维度的 dist_reuse 实现可以完全统一。

### 原则三：Block 物理形态在共享域内统一

一个 block 的物理 size 由 `(num_local_layers, local_num_kv_heads, head_size, tokens_per_block, dtype)` 决定。共享域的定义保证了域内所有节点的这些参数完全一致，因此域内的 block 可以直接 P2P 传输。

跨共享域的 block 物理形态不同（比如 PP0 的 block 只含前半 layers，PP1 的 block 只含后半 layers），不能互拷。

CP 维度不影响 block 物理 size——`attn_tp_size = tp_size // attn_dp_size // attn_cp_size`，head 切分由 `attn_tp_size` 单独承担，CP 维度只切 query 不切 KV head。

### 原则四：跨 SD 协调走统一的 graph 派发链路（关键简化点）

**Master 不为 dist_reuse 单独引入协议层；所有跨 SD 协调都表达成 `TransferOpGraph` 上挂多个带节点身份标签的 op，复用现有跨机 TP / PP 的 graph 派发与回执通道。**

具体如何映射，见 §6。

---

## 3. CP 的简化处理

> 本节是相比早期方案的关键简化。基于 sglang + FlexKV 代码事实：CP attention 层做 all-gather 后，所有 cp_rank 的本地 KV pool 内容 bit-wise 一致；CP 不切 KV head；NSA CP 也走相同的 all-gather 路径（额外覆盖 indexer K）。

### 3.1 CP 不进 SD key

各 cp_rank 的 pool 物理对等，跨 instance 的 cp=i ↔ cp=j 互拷在数据层面合法；将 CP 折叠到同一 SD 后，跨 instance reuse 不再要求"必须同 cp_rank 配对"，命中率提升、SD barrier 数量大幅减少。

### 3.2 sync_leader 单点查询 + scatter

控制面层面，CP 维度上的所有 `get_match` / `put_match` 都由 `sync_leader (cp_rank=0)` 在**全序列 token_ids** 上完成，结果通过 sglang 现有的 `scatter` / `broadcast_pyobj` 同步到其他 cp_rank。这与现有 sglang `flexkv_connector` 行为一致，无需改动。

### 3.3 D2H / H2D 由谁做

| 操作 | 控制面发起者 | 数据面执行者 |
|------|--------------|-------------|
| `get_match` / `put_match` | sync_leader 一个进程 | KVManager 内部 |
| **H2D**（CPU → 各 cp_rank GPU） | sync_leader 提交 slot_mappings（含所有 cp_rank） | FlexKV transfer worker，每个 GPU 一个 worker，并发写入 |
| **D2H**（GPU → CPU） | sync_leader 提交单一 cp_rank 的 slot_mapping | 任一 cp_rank 的 transfer worker（pool 内容相同，挑 sync_leader 那张最简单） |
| **跨 instance P2P (RDMA)** | sync_leader（控制） | Mooncake/RDMA 引擎在 sync_leader CPU pool 上完成 |

D2H 只需读 sync_leader 一份是因为各 cp_rank 内容物理相同——这是 CP 折叠到同一 SD 的直接受益。

### 3.4 NSA CP 与非 NSA CP 的统一

代码事实：

| 维度 | 普通 CP | NSA CP |
|------|---------|--------|
| Attention KV all-gather 调用位置 | `flashattention_backend.py` 的 `cp_allgather_and_save_kv_cache` | `deepseek_v2.py:rebuild_cp_kv_cache`（在 attn backend 之前） |
| Indexer K all-gather | — | `nsa_indexer.py:1333-1347` 调用 `cp_all_gather_rerange_output` |
| 各 cp_rank pool 是否对等 | 是（bit-wise 一致） | 是（bit-wise 一致） |
| 是否切 KV head | 否 | 否 |

两种 CP 在 dist_reuse 视角下行为同构，**`is_nsa` 仅作为模型 layout 隔离维度进 sd_key**（NSA 模型多一份 indexer K cache buffer，block 物理 layout 与非 NSA 不同），不引入 cp_rank 隔离或额外协同 D2H 路径。

> **命名说明**：旧版本曾用 `is_nsa_cp` 这一字段名（沿用 sglang `enable_nsa_prefill_context_parallel`），但语义上这个 flag 表达的是"模型是 NSA 模型"，与是否启用 CP 无关——即使 `cp_size=1`，NSA 模型也有 indexer K layout。本方案在 sd_key 序列化形式中保留 `nsa{0|1}` 字段，**Python 字段建议命名为 `is_nsa`** 以反映真实语义。下文统一使用 `is_nsa`。

### 3.5 Block hash 语义

由于 sync_leader 在全序列 token_ids 上做 match，block hash 的输入就是全序列分块——不需要 `cp_map` 之类的跨 cp_rank 映射结构。同一条 prompt 在两个 instance 的 sync_leader 上得到的 block hash 序列完全一致 → 直接可 P2P。

### 3.6 跨 instance reuse 的对齐方式

`hit_length` 由 sync_leader 在全序列上算出后通过 scatter 同步给所有 cp_rank：
```
hit_length_global = sync_leader.kv_manager.get_match(token_ids=full_seq)
```
不存在"按 cp_rank 取 min"的聚合需要。

---

## 4. DistributedRadixTree 单 Node 匹配约束

### 4.1 设计决策

**`DistributedRadixTree::match_prefix` 的一次匹配结果限定只能来自单个 peer Node（即单个 `node_id`）。**

现有实现中，`RefRadixTree` 在 `remote_tree_refresh()` 时会将多个 peer node 的 BlockMeta merge 到同一棵树中，导致一次 `match_prefix` 的结果中不同 block 可能属于不同的 `node_id`。本方案将其简化为：匹配过程中锁定第一个有效 block 的 `node_id`，后续 block 如果 `node_id` 不同则停止匹配。

### 4.2 简化收益

| 维度 | 简化前（多 Node 匹配） | 简化后（单 Node 匹配） |
|------|----------------------|----------------------|
| `CMatchResult` | 返回 per-block 的 `block_node_ids` tensor | 返回单个 `matched_node_id`（uint32） |
| `MatchResultAccel` | `block_node_ids` + `matched_node_ids` 数组 | `matched_node_id: Optional[int]`（单值） |
| Worker 传输 | 需要按 node 分组并行拉取 | 单 node 直接拉取 |
| 索引 merge | 跨 Node chain compression | 同一 `CRadixNode` 内 block 属于同一 Node |
| Lease 管理 | 一个 Node lease 过期级联影响跨 Node 匹配 | 故障域最小 |
| 跨 SD GET op 构图 | 可能涉及多个 peer instance | 直接确定唯一的 peer instance |

### 4.3 对命中率的影响

**影响极小**。原因：
- Radix Tree 是 prefix tree，同一条匹配路径上的 block 大概率来自同一个 Instance（同一个请求的 KV cache 是整体写入的）
- 跨 Node 的情况只在 chain compression 合并了不同 Node 的 block 时出现，实际中非常少见
- 选择"最长连续单 Node 匹配"几乎不会损失匹配长度

### 4.4 实现要点

```cpp
// RefRadixTree::match_prefix 中的单 Node 约束
uint32_t matched_node_id = UINT32_MAX;  // 尚未确定

// 匹配过程中：
auto bnis = current_node->get_block_node_ids();
for (int i = 0; i < matched; ++i) {
    uint32_t block_nid = (*bnis)[i];
    if (matched_node_id == UINT32_MAX) {
        matched_node_id = block_nid;  // 锁定第一个 node_id
    } else if (block_nid != matched_node_id) {
        matched = i;  // 遇到不同 node_id，停止匹配
        break;
    }
    pb_out[pb_write++] = pbs[i];
}
```

### 4.5 对跨 SD GET 流程的简化

由于 `matched_node_id` 是单个值，Master 可以直接确定唯一的 peer instance：

```
1. match_prefix() 返回 (physical_blocks, matched_node_id)
2. 从 matched_node_id → sd_nodes_cache → peer_instance_id（唯一确定）
3. 从 sd_nodes_cache[peer_instance_id] → 各 SD 的 node_id
4. 构图：每个 SD 一个 PEERH2H op，src_block_node_ids=[各自 SD 的 peer node_id]
5. 走 _launch_task 广播给所有 transfer_handle
6. 每个 SD 的 worker 自己 group + 发起 Mooncake read
```

Worker 层的传输从多 node 分组并行退化为单 node 直接读取，代码路径大幅简化。

---

## 5. 跨 SD 聚合一致性

> CP 折叠到同一 SD 后，本节只针对 PP / 跨机 TP 维度的多 SD。当前部署约束下 SD 数量 ≤ 4。

一个请求的"完整 KV reuse"要求**所有共享域都命中**（缺一不可）。

**问题**：Master 在自己的 SD 看到 remote hit，但其他 SD 可能 miss → 不能 reuse，否则 GPU 上的 KV 是半拼接的错误数据。

**解决方案**：Master 维护一个**跨 SD 聚合层 radix tree**——每个 block 的状态从 "ready / not-ready" 扩展为 "ready on SD(0) / ready on SD(1) / ... / fully ready"。只有 **fully ready** 的 block 才对外表现为"可 reuse"。

```
insert 流程：所有 SD 都通过 CompletedOp 回报后才标记 fully ready
match 流程：只对 fully ready 的前缀返回 hit
evict 流程：Master 单方面 evict（跳过 refcount > 0 的 block），不通知 Remote
```

聚合层 radix 的 `ready_sds` 字段是 `Dict[sd_key_str, contributing_node_id]`——对每条 prefix 记录"哪个 SD 的哪个 distributed_node_id 持有这份数据"，便于跨 SD GET 时直接构图。

### 5.1 Eviction 与引用计数

**核心设计决策：Master 单方面 evict，不广播给 Remote。**

- Master 对每个 block 维护 `refcount`：`get_match` 选中时 +1，所有 SD 完成后 -1
- evict 时跳过 `refcount > 0` 的 block
- Remote 上被 evict 的 block 成为"孤儿数据"，等待后续被新数据自然覆盖
- 不需要两阶段 eviction 协议，大幅简化实现

**为什么不需要通知 Remote？** 因为 Master 的 radix tree 是唯一的索引——Master evict 后不再有任何请求会去读那些 block，Remote 上的孤儿数据不影响正确性。

**也因此不需要 `coord_query`**——Master 不可能问 Remote "你那边还有没有 block X"，因为 Master 的 radix 自己就是唯一答案；如果 Master 的 radix 里有，那就一定有。

### 5.2 故障模型

基于 sglang 的"共命运"假设（同 instance 的 Master 和所有 Remote 在进程生命周期上共命运，任一 rank crash 会导致整 instance 秒级内全部退出），故障只剩两类：

1. **同 instance 整体退出/重启**：通过 Redis session epoch 检测，peer instance 批量 invalidate
2. **跨 instance 链路故障**：Mooncake P2P read 失败时即时通过 `FailureReportMsg` 上报 + 单前缀 invalidate + fallback 到正常 prefill

`FailureReportMsg` 跟 graph 派发链路正交——它是数据面失败的异步上报通道，不影响协调路径的设计。

---

## 6. 跨 SD 协调流程：基于 graph 派发的统一实现

> 本节是相比早期版本的**核心重构**。早期版本 §6 描述的 `coord_query → coord_get_cmd → coord_get_ack` / `coord_put_cmd → coord_put_ack` 三阶段协议层在新方案中**整体废弃**，跨 SD 协调改成在统一 `TransferOpGraph` 上挂多个 op + 复用现有跨机 TP/PP 的 graph 派发链路完成。

### 6.1 复用的现有基础设施

FlexKV 跨机 TP/PP 已有的 graph 派发链路：

```python
# Master 端 (kvtask.py::_launch_task)
for handle in self.transfer_handles:                     # master in-proc + N 个 remote handle
    handle.submit(transfer_graph, task_end_op_id=...)    # 同一份 graph 广播

# Remote 端 (transfer_manager.py::_handle_submit)
def _handle_submit(self, graph, task_end_op_id):
    self._rebind_graph_to_local_worker(graph)            # 把 op.pp_rank 改写到本地
    self.submit(graph)                                    # 提交本地 TransferEngine

# Worker 端 (transfer/worker.py::PEER2CPUTransferWorker)
src_block_node_ids = transfer_op.src_block_node_ids
groups = group_blocks_by_node_and_segment(...)           # 按 node_id 分组
for node_id, segment in groups.items():
    peer_node_info = self.get_node_meta(node_id)         # 从 Redis 查 zmq_addr / engine_addr
    # 发起 Mooncake RDMA 到这个 peer
```

**这套机制已在生产 GET 路径（single-SD `TransferType.PEERH2H`）验证可用**——dist_reuse multi-SD 协调只是把它扩展到多 SD。

### 6.2 跨 SD GET（从其他 instance 拉 KV）

```
1. 请求到达 → sync_leader 在全序列 token_ids 上 match
2. 命中 fully_ready 前缀（fully_ready 是在 PUT 阶段已经通过 §6.3 完整建立的）
3. 从 aggregate_radix.ready_sds[sd_key] 取出每个 SD 的 contributing_node_id
4. Master 构造统一 GET TransferOpGraph，每个 SD 挂一个 PEERH2H op：
   op[sd_0]: src_block_node_ids=[peer_inst.sd0_node],
             target_node_ids=[master_node_id],
             src=peer_cpu_blocks, dst=local_cpu_blocks
   op[sd_1]: src_block_node_ids=[peer_inst.sd1_node],
             target_node_ids=[sd_1_node_id],
             src=peer_cpu_blocks, dst=local_cpu_blocks
   ...
5. 走 _launch_task 广播给所有 transfer_handle
6. 每个 handle 收到 graph：
   - master in-proc：TransferEngine 执行 target_node_ids 含 master_node_id 的 op
   - peer SD remote：_rebind 后 TransferEngine 执行 target_node_ids 含本节点 self_node_id 的 op
7. 每个 op 的 worker 内部：
   - 按 src_block_node_ids 分组（单 Node 匹配约束保证只有一个 group）
   - get_node_meta(peer_node) → mooncake.transfer_sync_read
   - 完成后回报 CompletedOp
8. Master 收齐所有 CompletedOp → 触发后续 H2D 阶段（CP 维度的 scatter 不变）
9. 任一 op 失败 → FailureReportMsg → invalidate 前缀 + fallback 到正常 prefill
```

### 6.3 跨 SD PUT（推理完成后存 KV）

```
1. 推理完成 → sync_leader 在全序列上做 put_match
2. Master 在 _notify_master_sd_ready 阶段构造统一 PUT TransferOpGraph，每个 SD 挂一个 D2H op：
   op[master_sd]: target_node_ids=[master_node_id],
                  src=gpu_block_ids, dst=local_cpu_block_ids,
                  block_hashes=[...],
                  post_complete_callback=publish_to_redis
   op[peer_sd_1]: target_node_ids=[peer_sd_1_node_id], 同上
   op[peer_sd_2]: target_node_ids=[peer_sd_2_node_id], 同上
3. 走 _launch_task 广播给所有 transfer_handle
4. 每个 SD 的 TransferEngine 执行归本节点的 D2H op（gpu_block_ids 在 mirror 假设下跟 master 一致）
5. 每个 op 完成 → 触发 post_complete_callback → redis_meta.publish_batch
   （publish 在 master 进程上下文做：每个 op 完成时 master 收到 CompletedOp，回调里 publish 自己的那批；
    避免 worker 进程持有 RedisMeta 引用）
6. Master 收齐 CompletedOp → mark_sd_ready(prefix_hash, sd_key, block_ids,
                                            node_id=op.contributing_node_id)
7. 全 SD ready → aggregate radix 翻 fully_ready → 该 prefix 进入"可 reuse"状态
```

### 6.4 op 字段约定

`TransferOp` 在现有字段基础上引入两个语义清晰的标签字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `src_block_node_ids: np.ndarray`（已有） | `node_id` 数组 | 每个 src block 来自哪个 distributed_node_id（用于 worker 内部按 peer 分组发起 RDMA） |
| `target_node_ids: np.ndarray`（新增） | `node_id` 数组 | 这条 op 归哪些 SD 的节点执行；Remote 在 `_handle_submit` 阶段按 `target_node_ids` 过滤掉不属于自己的 op |
| `post_complete_callback: Optional[Callable]`（新增） | callable | op 完成后在 master 进程上下文执行的回调（如 `redis_meta.publish_batch`） |

`pp_rank` 字段保持原义（路由到本地 PP worker），跟 `target_node_ids` 正交：`target_node_ids` 决定"哪个 SD 处理这条 op"，`pp_rank` 决定"该 SD 内部哪个 PP worker 处理"。

### 6.5 CompletedOp 携带 sd_key 标签

`_handle_submit` 在 rebind 后给 graph 上每个 op 打一个 `_sd_key` 标签（哪个 handle dispatch 它），op 完成后 `result_socket` 把 sd_key 跟 `CompletedOp` 一起发回 master。

Master 端 polling worker 用 sd_key 路由到 `master_coord.mark_sd_ready(prefix_hash, sd_key, block_ids, node_id=completed_op.contributing_node_id)`。

这跟早期版本里 `RemoteCoordHandler._tag_ack` 给 `CoordPutAckMsg` 加 `_tagged_sd_key` 的语义对称——只是把 tag 从独立协议消息搬到了 `CompletedOp`。

---

## 7. Remote 节点的扩展

现有的 `TransferManagerOnRemote` 需要扩展，使其成为所在共享域的"一等公民"：

| 现有能力 | 新增能力 |
|---------|---------|
| 接收 Master 的传输指令（graph 派发） | 初始化 `RedisMeta`，注册到自己的共享域 |
| 执行 GPU↔CPU 传输 | 初始化 Mooncake，注册本地 CPU buffer |
| 回报完成状态（CompletedOp） | 接受/发起跨 instance 的 Mooncake P2P |
| `_rebind_graph_to_local_worker` 改 op.pp_rank | 在 rebind 之前按 `target_node_ids` 过滤 op |
| — | `pending_dict`：缓存 Master 的 graph，等本机 slot_mapping 到达后执行 |

**Remote 类型按维度区分**（CP 不再单独构成 Remote 角色）：
- **PP-Remote**（`pp_rank > 0`）：跨 PP 的 layer 切片
- **TP-Remote**（`tp_node_idx > 0`）：跨机 TP 的 KV head 切片
- **CP-rank 节点**（`cp_rank > 0`）：在 dist_reuse 视角下与 Master 处于同一 SD（CP 不进 sd_key），但仍需要本机的 transfer worker 注册到 FlexKV 以做并发 H2D；它们**不是 SD-Remote**，无需独立 RedisMeta（CPU pool 中数据由 sync_leader 那一份代表整组）。

> 实施层面 cp_rank > 0 的 worker 仍走 `TransferManagerOnRemote` 路径以共享代码，但只承担"GPU 注册 + 接收 Master 协同 H2D 指令"职责，不向 Redis 单独注册 SD 节点身份。

**Remote 不再需要的能力**（早期版本 §6 协议层 → 新方案下废弃）：
- `RemoteCoordHandler` / `BlockIndex` / `on_query` / `on_get_cmd` / `on_put_cmd` —— 全部废弃，统一走 `_handle_submit` 处理 `TransferOpGraph`
- 独立的 ZMQ 协议路由表（`_COORD_INBOUND_TYPES` / `_COORD_ACK_TYPES`）—— 废弃，因为协调消息整体改成 `TransferOpGraph`

---

## 8. 部署约束与简化

### 8.1 设计范围

| 维度 | 支持取值 |
|------|---------|
| CP | 1, 4, 8 |
| PP | 1, 2 |
| TP 跨机节点数 | 1, 2 |
| Prefill 跨节点数 | ≤ 2 |

**关键简化**：由于 prefill 跨节点数 ≤ 2，`PP=2` 和 `tp_node_count=2` 不会同时生效。这意味着：
- 跨机 Remote 物理节点最多 1 个
- 跨机 ZMQ 连接最多 1 条
- 不需要同时处理 PP + TP 两条跨机通道的交叉

### 8.2 共享域数量

由于 CP 不进 sd_key，SD 数量只随 PP × tp_node_count 变化：

| 配置 | SD 数量 |
|------|--------|
| PP=1, TP 不跨机, CP=任意 | 1（退化为现有 dist_reuse） |
| PP=2, TP 不跨机, CP=任意 | 2 |
| PP=1, TP 跨 2 机, CP=任意 | 2 |
| PP=2, TP 跨 2 机（未来） | 4 |

CP=4/8 不增加 SD 数量，但 CP 内部仍有 4/8 张 GPU 通过 H2D 并发接收同一份 CPU pool 的数据。

### 8.3 端口拓扑

跟跨机 TP/PP 现有部署完全一致——`FLEXKV_MASTER_HOST` + `FLEXKV_MASTER_PORTS=5556,5557,5558`，所有 Remote 通过 ZMQ identity 区分。**不需要 per-SD 端口、不需要 sglang launcher 做 endpoint 发现**。

---

## 9. 实施路径

### 9.1 已完成阶段

```
Phase 0: 共享域抽象（公共前置）
  ├── SharingDomainKey + Redis key 布局（不含 cp_rank）           ✅
  ├── PP-Remote / TP-Remote 节点的 RedisMeta/Mooncake 初始化     ✅
  ├── Master 跨 SD 聚合层 radix（仅 PP / 跨机 TP 维度）           ✅
  ├── DistributedRadixTree 单 Node 匹配约束（§4）                 ✅
  ├── KVTaskManager N 路 handle + barrier（N ≤ 4）                ✅
  └── 回归验证：CP=1, PP=1, TP 不跨机（退化为现有 dist_reuse）    ✅

Phase 1: 单 SD 跨实例 P2P + multi-SD aggregate radix 数据结构
  ├── ready_sds: Dict[sd_key, contributing_node_id]               ✅
  ├── MasterCoordinator.self_node_id 缓存                        ✅
  ├── single-SD cross-instance reuse e2e（s3 跨机）              ✅
  └── multi-SD aggregate radix 数据结构 + 单测                   ✅
```

### 9.2 方案 D 重构 + multi-SD（代码层已完成，2026-05-18）

```
Phase D-1: TransferOp 加 target_node_ids + sd_key tag         ✅ 已完成
  ├── flexkv/common/transfer.py: TransferOp 加 target_node_ids / block_hashes
  ├── transfer_manager.py: _handle_submit 按 target_node_ids 过滤
  ├── CompletedOp 加 sd_key + contributing_node_id 字段
  └── 单测：multi-SD graph dispatch 路由（test_d3_filter_and_get_clones.py）

Phase D-2: PUT 路径切到 graph 派发（§6.3）                     ✅ 已完成
  ├── cache_engine.py::_notify_master_sd_ready 重写为注册 _pending_put_batches
  ├── _maybe_attach_multi_sd_d2h_ops 给 D2H op 加 N 个 per-SD clone
  ├── publish 走 master 进程 op_callback_dict（不需要 worker 端 RedisMeta）
  ├── 删除 _make_d2h_and_publish_closure / _d2h_and_publish
  └── 单测：multi-SD PUT e2e（test_cache_engine_dist_reuse_gate.py
            TestPeerSdCompletionSink）

Phase D-3: GET 路径切到 graph 派发（§6.2）                     ✅ 已完成
  ├── GlobalCacheEngine._maybe_attach_multi_sd_peerh2h_ops:
  │     给主 SD 的 PEERH2H op 加 N 个 per-SD clone（每个挂
  │     target_node_ids=[peer_sd_nid]，src_block_node_ids 从
  │     AggregateRadixTree.ready_sds 反查）
  ├── _filter_graph_inplace_by_target_node_ids 提取为模块级工具，master
  │     in-proc handle 与 remote handle 共用同一过滤逻辑
  ├── transfer_graph 上挂 op_h2d → 每个 PEERH2H clone 的依赖（H2D 等
  │     全部 peer SD 完成后才发射）
  └── 单测：tests/test_d3_filter_and_get_clones.py

Phase D-4: 协议层 + 死代码清理                                  ✅ 已完成
  ├── 删除 CoordQueryMsg / CoordGetCmdMsg / CoordPutCmdMsg + ack 数据类
  ├── 删除 CoordinationCoordinator / RemoteCoordHandler / BlockIndex
  ├── 删除 _handle_coord_message / _make_d2h_and_publish_closure /
  │     ingest_coord_ack / _coord_get_cross_sd
  ├── 删除 attach_dist_reuse 的 deprecated coord_dispatcher 参数（2026-05-18）
  └── 单测重写：test_coord_protocol.py / test_cache_engine_dist_reuse_gate.py
```

**剩余工作**：multi-SD 真硬件 e2e（`benchmarks/sglang_flexkv_e2e/s4_multi_sd_pp2.sh`）。
代码路径就绪，缺真硬件 harness fixture——单 SD 真硬件回归（`s3` 系列）已 PASS。

### 9.3 后续阶段

```
Phase E: 单维度跨机 CP（CP=4 → CP=8，SD 数量保持为 1）
Phase F: 两维度组合（当前部署上限）
  ├── PP=2 + CP（SD 数量 2）
  └── 跨机 TP + CP（SD 数量 2）
Phase G: 三维度全开（预留，不在当前计划）
```

详细方案 D 的 step-by-step 实施细节、风险点、可回滚锚点见
`proposal_unify_with_graph_dispatch_2026-05-15.md`。

---

## 10. 一张图总结

```
                        Instance 1                                    Instance 2
┌─────────────────────────────────────────┐    ┌─────────────────────────────────────────┐
│                                         │    │                                         │
│  Master (pp=0, tpn=0, cp=0 = sync_leader)│   │  Master (pp=0, tpn=0, cp=0)            │
│  ┌─────────────────────────────────┐    │    │  ┌─────────────────────────────────┐    │
│  │ KVManager + CacheEngine         │    │    │  │ KVManager + CacheEngine         │    │
│  │ + 唯一的 LocalRadixTree          │    │    │  │ + 唯一的 LocalRadixTree          │    │
│  │ + 聚合层 radix                  │◄───┼────┼──┤ + 聚合层 radix                  │    │
│  │   (ready_sds: Dict[sd, node_id])│    │    │  │ + RedisMeta + Mooncake          │    │
│  │ + RedisMeta + Mooncake          │    │    │  └─────────────────────────────────┘    │
│  └────────────┬────────────────────┘    │    │                                         │
│               │ TransferOpGraph 派发     │    │                                         │
│               │ (复用跨机 TP/PP 通道)     │    │                                         │
│               ▼                          │    │                                         │
│  PP-Remote (pp=1)  ◄── Mooncake P2P ──►│    │  PP-Remote (pp=1)                      │
│  TP-Remote (tpn=1) ◄── Mooncake P2P ──►│    │  TP-Remote (tpn=1)                      │
│       ▲                                 │    │                                         │
│       │ _handle_submit:                 │    │                                         │
│       │   1. 按 target_node_ids 过滤    │    │                                         │
│       │   2. _rebind 改 op.pp_rank       │    │                                         │
│       │   3. submit 本地 TransferEngine  │    │                                         │
│       │ Worker:                         │    │                                         │
│       │   按 src_block_node_ids 分组     │    │                                         │
│       │   get_node_meta → 发 RDMA       │    │                                         │
│       │ 完成后回 CompletedOp(sd_key, nid)│    │                                         │
│                                         │    │                                         │
│  cp=1..N-1（同 SD，仅 GPU 注册）         │    │  cp=1..N-1                              │
└─────────────────────────────────────────┘    └─────────────────────────────────────────┘

PP / 跨机 TP 维度的同 SD 节点之间通过 Mooncake P2P 传输 KV block（如 inst1.pp=1 ↔ inst2.pp=1）
跨 PP / 跨 TP node 不允许 P2P（layer / KV head 切片不同）
CP 维度不参与 SD 划分；各 cp_rank 的 KV pool 内容由 attention all-gather 保证 bit-wise 相同
Master 统一决策（唯一事实来源），Remote 只搬数据
跨 SD 协调走 TransferOpGraph 派发（不引入独立协议层）
```

---

## 附录：与早期版本的关键差异

| 维度 | 早期版本（已废弃） | 本版本 |
|------|------------------|--------|
| 跨 SD 协调通道 | 独立的 `CoordQueryMsg / CoordGetCmdMsg / CoordPutCmdMsg` 协议层 | 复用 `TransferOpGraph` 派发链路 |
| Remote 端索引 | `BlockIndex`（hash → local_cpu_block_id）| 无（违反"Master 唯一事实来源"原则） |
| Remote 端协议 handler | `RemoteCoordHandler.on_query/on_get_cmd/on_put_cmd` | 无（统一走 `_handle_submit`） |
| Master 端协调器 | `CoordinationCoordinator.coord_query/coord_get/coord_put` | 无（统一走 `_launch_task`） |
| ack 通道 | `CoordXxxAckMsg` 携带 `_tagged_sd_key` | `CompletedOp` 携带 sd_key + contributing_node_id |
| D2H + Redis publish | `_d2h_and_publish` 闭包（200 行） | D2H op + `post_complete_callback` |
| 端口拓扑 | 计划 per-SD 独立端口（需要 sglang launcher 配合发现） | 跟跨机 TP/PP 一致，单组共享端口 |
| sglang 改动需求 | 需要 launcher 端做 endpoint 发现 | 0 改动 |
| 协议消息总数 | 8（RemoteReady / Query / QueryAck / Get / GetAck / Put / PutAck / FailureReport） | 3（RemoteReady / FailureReport / 在 CompletedOp 上扩展 sd_key tag） |
