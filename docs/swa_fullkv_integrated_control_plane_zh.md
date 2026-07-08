# SWA 与 Full-KV 控制面融合设计

本文梳理 `GlobalCacheEngine` 里 Full-KV 与 SWA 的 GET/PUT 控制面关系，并说明为什么这次把 SWA 的 graph append 融入 `_get_impl_local()` / `_get_impl_global()` / `_put_impl_local()` / `_put_impl_global()`，而不是继续保留独立的 `_swa_get_slots()` / `_swa_put_slots()` 或多个 SWA match 包装函数。

## 背景问题

之前的结构把 Full-KV 和 SWA 拆成两段：

1. `_get_impl_*()` / `_put_impl_*()` 先完成 Full-KV 的 match、分段、分配、插入、Full-KV graph 构建。
2. 外层 `get()` / `put()` 再从 `_get_impl_*()` / `_put_impl_*()` 的返回值里解包一些元数据。
3. `_swa_get_slots()` / `_swa_put_slots()` 再基于这些元数据单独解析 SWA slot。
4. 外层 `get()` / `put()` 再调用 `SWACacheManager.build_get_chain()` / `build_put_chain()` 把 SWA ops 追加回同一个 graph。

这个结构的问题是：SWA 明明依赖的是同一轮 Full-KV match 和同一批 Full-KV store/load 节点，却被拆到了外层 wrapper 再做一次“二次解析”。结果是 `_get_impl_*()` 的返回 tuple 越来越长，GET 路径里还需要 `_resolve_swa_get_source` 这类中间封包/解包逻辑，PUT 路径里也有 `_swa_put_slots()` 这种看起来和 Full-KV store 节点割裂的步骤。

更顺的边界应该是：

- `match_*()` 只负责索引匹配，返回 per-tier Full-KV `MatchResult`。
- `_get_impl_*()` / `_put_impl_*()` 负责把一次请求的 transfer plan 建完整，Full-KV 和 SWA peer ops 都在这里追加。
- 外层 `get()` / `put()` 只负责通用收尾：加 virtual op、计算 return mask、锁住 Full-KV 节点、返回 callback。

## Full-KV 原本怎么做

Full-KV 的 GET/PUT 不是单纯 “match 后立刻返回 graph”。它的职责分三层：

1. **Match 层**  
   `match_local_accel()` / `match_all_accel()` 查询 CPU/SSD/REMOTE 各 tier 的 radix index，返回 per-tier `MatchResult`。这些结果包含：
   - `num_ready_matched_blocks` / `num_matched_blocks`
   - `physical_blocks`
   - `last_ready_node` / `last_node`
   - P2P 场景里的 `matched_pos` / `matched_node_ids`
   - SWA 复用所需的 `last_swa_node` / `swa_hit_blocks`

2. **Plan 层**  
   `_get_impl_local()` / `_get_impl_global()` 和 `_put_impl_local()` / `_put_impl_global()` 根据 match result 做真实计划：
   - 裁剪 `[block_mask_start:block_mask_end]`
   - 计算 CPU hit、SSD hit、REMOTE hit 的 fragment
   - 为缺口分配 CPU/SSD/REMOTE block
   - 插入 unready Full-KV node
   - 构建 Full-KV `TransferOpGraph`
   - 生成 per-op ready callback
   - 返回 `node_to_unlock`、`buffer_to_free`、`num_gpu_blocks_to_transfer` 等收尾需要的信息

3. **Wrapper 收尾层**  
   外层 `get()` / `put()` 做所有请求通用的事情：
   - `add_virtual_op_for_multiple_finished_ops()`
   - 计算 `return_mask`
   - 对 `node_to_unlock` 里的 Full-KV node 加锁
   - 返回 `_transfer_callback()`，在 transfer 完成后 unlock/set_ready/recycle

所以 Full-KV 的核心不是 “match 函数直接产 graph”，而是 `match_*()` 产出结构化 match metadata，`_get_impl_*()` / `_put_impl_*()` 消费这些 metadata 构建 graph。

## 新边界

这次重构后，SWA 也采用同样边界：

- `_resolve_swa_read_source()` 是内部 per-tier resolver，不再暴露成一个看起来像独立匹配流程的公开入口。
- `_resolve_swa_read_source()` 可以接收同 tier 的 Full-KV `match_result`，直接复用 `last_swa_node` / `swa_hit_blocks`，避免重复 walk radix tree。
- `_get_impl_*()` 在 Full-KV fragment 和 graph 构建过程中，按同一套 CPU/SSD/REMOTE tier 顺序内联生成 SWA GET peer ops。
- `_put_impl_*()` 在 Full-KV store node 插入时通过 `insert(..., swa_store=True)` 让 tier engine 自己完成 SWA slot 分配和挂载，然后直接用这些 node 上的 SWA slot 生成 SWA PUT peer ops。
- 外层 `get()` / `put()` 不再知道 SWA slot 如何解析，也不再调用单独的 `_swa_get_slots()` / `_swa_put_slots()`。

对应的数据返回也从长 tuple 改成两个 plan dataclass：

- `GetTransferPlan`
- `PutTransferPlan`

这样 wrapper 层只消费 plan，不需要知道 plan 内部 Full-KV/SWA 是怎么拼出来的。

## GET 时序

```text
get()
  -> _get_impl_local() / _get_impl_global()
      -> match_local_accel() / match_all_accel()
           -> per-tier Full-KV MatchResult
              - physical_blocks
              - last_ready_node / last_node
              - last_swa_node / swa_hit_blocks
      -> if swa_aware:
           _clamp_end_to_swa()
      -> slice Full-KV fragments
      -> allocate CPU staging blocks when SSD/REMOTE hit needs H2D
      -> insert unready CPU/SSD nodes when Full-KV backfill is possible
      -> build Full-KV ops
           CPU hit:        H2D
           SSD hit:        DISK2H -> H2D
           REMOTE hit:     REMOTE2H -> H2D
      -> follow the same tier order to build SWA ops
           -> engine._resolve_swa_read_source(
                sequence_meta,
                upper_bound_blocks=full_hit_blocks,
                match_result=tier_match_result,
                lock_for_load=True,
              )
           -> CPU SWA hit:
                SWA H2D
                callback releases source node SWA pin
           -> SSD/REMOTE SWA hit:
                allocate transient CPU SWA staging slot
                SWA DISK2H/REMOTE2H -> SWA H2D
                callback releases source node SWA pin
                callback frees transient CPU SWA staging slot
      -> return GetTransferPlan
  -> add_virtual_op_for_multiple_finished_ops()
  -> compute return_mask
  -> lock Full-KV nodes in plan.node_to_unlock
  -> return graph, mask, transfer callback, op callbacks, task_end_op_id
```

关键点：

- SWA 选择源 tier 的优先级仍然是 CPU -> SSD -> REMOTE，和 Full-KV GET 的读取偏好一致。
- SWA GET source 必须在 graph 里 pin 住，因为 match 后到 H2D 完成前，node-mounted SWA slot 可能被 SWA-LRU 回收。
- SSD/REMOTE SWA GET 需要一个 CPU SWA staging slot，因为最终 SWA H2D 的源端是 CPU SWA slot；这个 staging slot 不挂在 radix node 上，H2D 完成后直接 free 回 CPU SWA pool。

## PUT 时序

```text
put()
  -> _put_impl_local() / _put_impl_global()
      -> match_local_accel() / match_all_accel()
           -> per-tier Full-KV MatchResult
      -> skip already cached CPU prefix
      -> allocate CPU/SSD/REMOTE Full-KV blocks for missing suffix
      -> build Full-KV ops
           D2H
           D2H -> H2DISK
           D2H -> H2REMOTE
      -> insert unready Full-KV store nodes
           -> CPU insert(..., swa_store=True)
                -> _alloc_swa_slot()
                -> _mount_swa_slot(cpu_store_node, slot)
           -> if CPU SWA store succeeds:
                SSD/REMOTE insert(..., swa_store=True)
                -> _alloc_swa_slot()
                -> _mount_swa_slot(tier_store_node, tier_slot)
      -> append SWA PUT peer ops from the store nodes' SWA slots
           SWA D2H
           SWA D2H -> SWA H2DISK
           SWA D2H -> SWA H2REMOTE
      -> return PutTransferPlan
  -> add_virtual_op_for_multiple_finished_ops()
  -> compute return_mask
  -> lock Full-KV nodes in plan.node_to_unlock
  -> return graph, mask, transfer callback, op callbacks, task_end_op_id
```

关键点：

- PUT 的 SWA slot 挂载跟 Full-KV store node 插入放在同一个 tier engine 调用里完成，因为 SWA 是 node-mounted 的：没有 store node，就没有可挂载的 SWA entry。
- SWA 写穿只写到 Full-KV 同样写到的 tier，保持 “SWA 是 Full-KV 子集” 的约束。
- Lower tier SWA 写穿依赖 CPU SWA store 成功。CPU SWA slot 是 SWA D2H 的目标，也是 H2DISK/H2REMOTE 的源；如果 CPU SWA pool 满且无法驱逐，就只保留 Full-KV store，不写穿 lower-tier SWA。
- P2P/层级索引目前不暴露完整 node-mounted SWA 挂载能力，所以它的 `insert(..., swa_store=True)` 是 no-op，`_get_mounted_swa_slot()` 返回 miss，不会构造不完整的 SWA 写穿图。

## 为什么还需要 match_result

用户直觉上会问：能不能一个 SWA match/resolve 一次拿到所有信息，GET 的时候再解析建图？

现在的答案是：SWA 信息确实是在同一次 Full-KV match 里带出来的，但 graph 不应该由 match 层直接建。

`MatchResult` 仍然需要存在，原因是：

1. **避免重复 walk radix tree**  
   Full-KV match 已经沿 radix tree 找到了 ready prefix。SWA 的 `last_swa_node` / `swa_hit_blocks` 是同一轮 match 里顺手产出的，`_resolve_swa_read_source(..., match_result=...)` 直接复用它。

2. **保证 SWA 是 Full-KV 子集**  
   SWA hit 必须被 Full-KV hit 的 upper bound clamp。常见路径下 `swa_hit_blocks <= upper_bound_blocks`，直接复用 match result。极少数场景里，如果 match result 记录的 SWA node 超过当前可用 Full-KV bound，`_resolve_swa_read_source()` 会 fallback 到 clamped probe，避免把 SWA 指到不可复用的前缀之后。

3. **match 层不应该有副作用**  
   GET 的 SWA 复用需要 pin source node；SSD/REMOTE 还需要分配 transient CPU staging slot。PUT 的 SWA store 需要在 `insert(..., swa_store=True)` 里分配 slot 并挂载到刚插入的 Full-KV store node。
   这些都是控制面副作用，不适合放进 `match_local_accel()` / `match_all_accel()`。

所以更准确的模型是：

```text
match_*()
  -> 返回 Full-KV MatchResult，其中携带 SWA metadata

_get_impl_*() / _put_impl_*()
  -> 消费 MatchResult
  -> 决定 Full-KV fragments / store nodes
  -> 决定 SWA pin / slot / staging / write-through
  -> 构建完整 TransferPlan
```

## 为什么还需要 lock

SWA 的 lock 只在 GET load 路径需要，目的不是为了 match 本身，而是为了保护 “match 到的 SWA slot 在异步 H2D 完成前不能被回收”。

GET CPU source：

```text
_resolve_swa_read_source(..., lock_for_load=True)
  -> inc_swa_lock_ref(source_node)
  -> append SWA H2D
  -> SWA H2D callback
       -> dec_swa_lock_ref(source_node)
```

GET SSD/REMOTE source：

```text
_resolve_swa_read_source(..., lock_for_load=True)
  -> inc_swa_lock_ref(source_tier_node)
  -> allocate transient CPU SWA staging slot
  -> append SWA DISK2H/REMOTE2H -> SWA H2D
  -> SWA H2D callback
       -> dec_swa_lock_ref(source_tier_node)
       -> free transient CPU SWA staging slot
```

PUT 不需要 `lock_for_load`，因为它不是从已有 SWA slot 读取；它是在新 Full-KV store node 上挂载新分配的 SWA slot。PUT 的完成顺序由 graph 的 virtual barrier 和 Full-KV transfer callback 统一收尾。

## 删除和保留

删除的旧入口：

- 旧的 SWA locked wrapper
- 旧的 SWA from-result wrapper
- `_swa_get_slots()`
- `_swa_put_slots()`

保留并收敛后的内部入口：

- `_resolve_swa_read_source(sequence_meta, upper_bound_blocks, match_result=None, lock_for_load=False)`

新增和调整的内生接口：

- `GetTransferPlan`
- `PutTransferPlan`
- `_alloc_swa_slot()`：从 SWA host pool 分配一个 slot；pool 满时触发 `_evict_swa_slots()`，由同一棵 radix tree 选 SWA-LRU。
- `_free_unmounted_swa_slot()`：只回收没有挂在任何 radix node 上的 slot，例如 GET 的 transient CPU staging slot。
- `_drain_unmounted_swa_slots()`：把 radix tree 在 split/merge/evict/replace 时摘下来的 slot drain 回 host pool。
- `_mount_swa_slot()`：engine 侧的内部挂载入口，底层调用 radix tree 的 `set_swa(node, slot)`。
- `_reserve_swa_tail_slot()`：`insert(..., swa_store=True)` 使用的组合动作，负责 alloc + mount + drain。
- `insert(..., swa_store=True)`：Full-KV node 插入后同步挂载 SWA tail slot。
- `_get_mounted_swa_slot()`：从 store node 读取刚挂载的 SWA slot，用于构造 SWA PUT graph。
- `_build_op_callback_dict()`

## Engine 和 radix tree 的职责边界

`CacheEngineAccel` 和 `CacheEngine` 现在通过 `_NodeMountedSWAEngineMixin` 共享 SWA 控制面 helper。这个 mixin 只处理 engine 层应该知道的事情：

- SWA host pool 的 slot 分配和 unmounted slot 回收。
- 在 Full-KV insert 返回 store node 后，把 SWA slot 挂到该 node。
- GET 时把 Full-KV `MatchResult` 里的 SWA metadata 解析成 read source，并按需 pin source node。

radix tree 仍然是 SWA 位置和结构一致性的唯一 owner：

- `set_swa(node, slot)`：把 slot 挂到 node 的 trailing page，并维护 SWA-LRU。
- `promote_swa(node)`：GET hit 后更新 SWA-LRU。
- `evict_swa(...)`：按 SWA-LRU 选择可驱逐 node。
- `drain_freed_swa_slots()`：把 split/merge/full-evict/SWA-evict 摘下来的 slot 交回 engine drain。

因此 engine 里不再保留公开转发式的 SWA helper；底层挂载能力只保留在 radix tree。如果要检查 node-mounted 适配是否完整，只看两层契约：

- match 结果必须带 `last_swa_node` / `swa_hit_blocks`。
- radix tree 所有会改变 node trailing page 所属关系的路径，都必须调用 `record_freed_swa_slot()` 或 remount SWA slot，并让 engine drain。

## 当前结构总结

现在 SWA 和 Full-KV 的关系是：

- 同一轮 Full-KV match 产出 Full-KV metadata 和 SWA metadata。
- 同一层 `_get_impl_*()` / `_put_impl_*()` 消费这些 metadata，并沿 Full-KV fragment / insert 生命周期构建完整 transfer plan。
- SWA peer ops 和 Full-KV ops 在同一个 `TransferOpGraph` 里，用同一个 virtual barrier 归并完成条件。
- 外层 `get()` / `put()` 不再处理 SWA slot 解包，只处理通用 request 收尾。
- Full-KV eviction、`recycle()`、SWA-only eviction 都会 drain tree 里 pending freed SWA slots；`SWAHostPool.free()` 对重复 free 做防御，避免 free-list 重复入栈。

这个结构比之前更接近 Full-KV 的原始分层，也把 SWA 的副作用限制在真正需要建 plan 的地方。
