# SWA Match API 梳理

## 结论

当前只保留一个 SWA 匹配入口：

```python
engine.match_swa(
    sequence_meta,
    upper_bound_blocks,
    match_result=mr,        # 可选：复用已有 Full-KV match 结果
    lock_for_load=True,     # 可选：GET H2D 前 pin 住 SWA 源 node
    return_node=True,       # 可选：返回 node handle，供完成回调释放 pin
)
```

返回值：

- 默认返回 `(swa_hit_blocks, slot_id)`。
- `return_node=True` 或 `lock_for_load=True` 时返回 `(swa_hit_blocks, slot_id, node)`。
- miss 返回 `0, -1` 或 `0, -1, None`。

旧的 `match_swa_locked()` / `match_swa_from_result()` 没有外部调用，已经删除。它们不是新的语义，只是把同一个 `match_swa()` 行为拆成了两个额外名字：一个代表 `lock_for_load=True`，一个代表 `match_result=...`。

## 先看 Full-KV 怎么做

先区分两个名字相近但层级不同的 match：

- `KVManager.get_match()` / `KVTaskEngine.get_match()` 是对外任务 API。它会用 fake `slot_mapping` 创建一个 `UNREADY` GET task，并在创建任务时调用 `GlobalCacheEngine.get()` 把 graph 先建好；后续 `launch()` 只 late-bind 真实 GPU block ids，不再重新 match。
- `GlobalCacheEngine.match_local()` / `match_all()` / `match_local_accel()` / `match_all_accel()` 是 cache engine 内部的 per-tier radix match 聚合。它们返回 CPU/SSD/REMOTE 的 `MatchResult`，只服务本次 `GlobalCacheEngine.get()` 构图。

所以看起来像“get_match 之后 get 又查了一遍”，实际不是两个阶段重复查。`get_match()` 的实现本身就是通过 `GlobalCacheEngine.get()` 完成 match + graph build；`launch()` 只是把 graph 里 placeholder GPU blocks 替换成真实 slot mapping。

Full-KV 的控制面不是“match 直接返回 graph”，而是分三层。现在 GET 路径把中间结果显式收成 `FullKVGetPlan`，避免上层继续解长 tuple：

```text
match_*() -> MatchResult
_get_impl_*() -> FullKVGetPlan
get() -> resolve SWA source, append SWA graph, lock nodes, attach callbacks
_put_impl_*() -> TransferOpGraph + node_to_unlock + callbacks
put() -> append SWA store graph, lock nodes, attach callbacks
```

`match_*()` 只做索引层前缀匹配。它返回 `MatchResult`，里面有：

- `num_ready_matched_blocks` / `physical_blocks`：这次能直接复用的 Full-KV block。
- `last_ready_node`：已经 ready 的匹配边界 node。GET 会把它放进 `node_to_unlock`，在 graph 执行期间 lock 住，避免被 Full-KV eviction 回收。
- `last_node`：用于 `take(..., protected_node=last_node)`，分配 staging/new blocks 时保护当前匹配路径。
- `last_node_matched_length` 等插入所需信息。

`_get_impl_local()` / `_get_impl_global()` 使用这些 `MatchResult` 来做三件事：

- 切出 CPU/SSD/REMOTE 各 tier 已命中的 physical blocks，构建 H2D、DISK2H、REMOTE2H、H2DISK 等 Full-KV op。
- 如果需要把 lower tier 数据 staged 回 CPU/SSD，会 `take()` 新 block，并用 `insert(..., is_ready=False, match_result=...)` 把即将落地的数据先挂到 radix tree 上。
- 产出 `node_to_unlock` 和 `op_node_to_ready`：前者由最终 transfer callback 统一 unlock/set_ready，后者在某个具体 op 完成时把新插入 node 标成 ready。

`FullKVGetPlan` 承接这些 GET 元数据：

- `transfer_graph` / `finished_ops_ids`：Full-KV 已经建好的 graph 和 barrier 终点。
- `node_to_unlock` / `op_node_to_ready`：Full-KV node 生命周期。
- `buffer_to_free`：未插入 radix tree 的临时 staging block。
- `num_gpu_blocks_to_transfer`：本次 Full-KV GET 最终会搬到 GPU 的 block 数。
- `tier_match_results`：CPU/SSD/REMOTE 同一轮 Full-KV match 结果，供 SWA source resolver 复用。

`get()` / `put()` 在 graph 返回给执行层之前，会对 `node_to_unlock` 里的 Full-KV node 调 `lock_node()`；graph 完成后 `_transfer_callback()` 再 `unlock()` 并 `set_ready()`。这就是 Full-KV 的生命周期：match 只返回元数据，lock 保护异步传输期，callback 释放。

## SWA 为什么也需要 result

SWA 是 node-mounted 的：SWA slot 挂在 Full-KV radix node 上，不是独立索引。Full-KV `match_prefix()` 在同一次 radix 前缀匹配里已经顺手产出了：

- `last_swa_node`：匹配路径上最深的、ready 的、挂着 live SWA slot 的 node。
- `swa_hit_blocks`：这个 SWA window 对应的 prefix block 数。

所以 SWA slot resolution 应该优先复用 Full-KV `MatchResult`，原因有三个：

- 避免 `_resolve_swa_get_source()` 为每个 tier 再走一次 radix tree。
- 保证 Full-KV hit 和 SWA hit 来自同一轮 match 视图，降低控制面时序漂移。
- `_clamp_end_to_swa()` 已经用同一批 match result 算 `usable = min(full_hit, swa_hit)`，后面 `_resolve_swa_get_source()` 继续用这批 result 才一致。

但不能无条件直接拿 `match_result.last_swa_node`。`upper_bound_blocks` 是当前请求真正可复用的 Full-KV 上界；如果已有 match result 里的 `swa_hit_blocks > upper_bound_blocks`，说明那个 SWA node 在本次 usable prefix 之外。此时不能简单 `min(swa_hit, upper_bound)`，因为 slot/node 仍然指向更深的旧 node，会读错 window。正确做法是对截断到 `upper_bound_blocks` 的 prefix 做一次 clamped probe，找 bound 内最深的 SWA node。

## SWA 为什么也需要 lock

Full-KV 的 `lock_node()` 保护的是 Full-KV node/physical blocks，防止异步 transfer 期间被 Full-KV eviction 回收。SWA 也有同样问题，但保护对象不同：

- SWA H2D 读的是 SWA host-pool slot。
- 这个 slot 挂在 radix node 上。
- SWA-only eviction 可以在 Full-KV node 还存在时单独释放 SWA slot。

因此 `_resolve_swa_get_source()` 一旦决定某个 tier 的 SWA slot 会作为 GET 源，就必须在构图阶段对源 node 增加 `swa_lock_ref`。SWA H2D 完成后，`_swa_release_load_lock()` 再释放这个 pin。这样 SWA-only eviction 在异步 H2D 读 slot 期间不会把 slot 回收并复用给别的 window。

这和 Full-KV 的 `node_to_unlock` 是同一个生命周期模式：

```text
Full-KV: node_to_unlock -> lock_node() -> H2D/DISK2H/... -> _transfer_callback() unlock()
SWA    : match_swa(lock_for_load=True) -> inc_swa_lock_ref()
         -> SWA H2D/DISK2H/... -> _swa_release_load_lock() dec_swa_lock_ref()
```

## GET 时序

CPU 命中 SWA 的 GET：

```text
GlobalCacheEngine.get()
  -> _get_impl_local/_get_impl_global()
       -> match_local_accel()/match_all_accel()
            -> per-tier Full-KV MatchResult
       -> build Full-KV transfer graph
       -> return FullKVGetPlan(graph, callbacks, tier_match_results, ...)
  -> _resolve_swa_get_source(full_plan.tier_match_results, full_hit_blocks)
       -> CPU first
       -> cpu_engine.match_swa(
              match_result=cpu_match,
              upper_bound_blocks=full_hit_blocks,
              lock_for_load=True,
              return_node=True)
            -> reuse last_swa_node/swa_hit_blocks when inside bound
            -> otherwise clamped probe
            -> promote SWA LRU
            -> inc_swa_lock_ref()
       -> return SWAGetSource(CPU slot + pinned node)
  -> SWACacheManager.build_get_chain()
       -> add SWA H2D op into full_plan.graph
  -> get()
       -> add virtual op over full_plan.finished_ops_ids + SWA H2D
       -> lock full_plan.node_to_unlock
       -> attach SWA H2D callback
  -> execution completes
       -> SWA callback: dec_swa_lock_ref()
       -> Full-KV callback: unlock()/set_ready()
```

SSD/REMOTE 命中 SWA、CPU 没有 SWA 的 GET：

```text
_resolve_swa_get_source()
  -> CPU match_swa miss
  -> SSD/REMOTE match_swa(..., lock_for_load=True, return_node=True)
       -> pin source-tier SWA node
  -> allocate transient CPU SWA staging slot
  -> build DISK2H/REMOTE2H -> H2D SWA chain
  -> H2D callback:
       -> release source-tier SWA lock
       -> free transient CPU staging slot
```

这里 staging slot 不挂 radix node，只是 SWA H2D 的临时 CPU source，所以完成后直接 free。

`SWAGetSource` 是 GET 侧一次性解析出来的 SWA source plan：

- `hit_blocks` / `source_tier`：这次 SWA window 从哪个 tier 命中。
- `gpu_slots` / `cpu_slots` / `ssd_slots` / `remote_slots`：直接传给 `SWACacheManager.build_get_chain()` 的 slot ids。
- `lock_node`：被 `match_swa(lock_for_load=True)` pin 住的源 node。
- `staging_slot`：SSD/REMOTE staging 到 CPU 时临时分配的 CPU SWA slot，H2D 完成后释放。

## PUT 时序

SWA PUT 不需要 match lock，因为它不是读取旧 SWA slot，而是在 Full-KV store node 上写入新的 trailing window：

```text
GlobalCacheEngine.put()
  -> _put_impl_local/_put_impl_global()
       -> Full-KV match
       -> allocate/insert Full-KV store nodes
       -> return node_to_unlock
  -> _swa_put_slots(..., node_to_unlock)
       -> allocate CPU SWA slot
       -> set_swa(cpu_store_node, slot)
       -> optionally allocate SSD/REMOTE SWA slots when Full-KV also stores there
       -> set_swa(lower_tier_store_node, tier_slot)
  -> SWACacheManager.build_put_chain()
       -> add SWA D2H and optional H2DISK/H2REMOTE write-through ops
  -> put()
       -> lock Full-KV node_to_unlock
  -> callbacks
       -> Full-KV unlock/set_ready
```

PUT 的保护主要依赖 Full-KV 已有 `node_to_unlock`；SWA slot 是新分配并挂到 store node 上的，不存在“异步读取旧 slot 前被 eviction 回收”的问题。

## 为什么不是 CacheEngine.match_swa 一次返回所有信息

如果“一次返回所有信息”指的是 GET 需要的 source tier、slot、lock node、staging slot，那么这个入口应该是 `GlobalCacheEngine._resolve_swa_get_source()`，不是单 tier 的 `CacheEngine.match_swa()`。

原因是 `CacheEngine.match_swa()` 属于单 tier 索引层，只知道“这个 tier 的 radix tree 上有没有可复用 SWA node”。它不知道：

- GPU slot mapping。
- CPU/SSD/REMOTE 的优先级。
- 本次 Full-KV hit 的 global usable bound。
- CPU staging slot 是否能分配成功。
- SWA GPU placeholder 和 launch-time late bind。
- op dependency、finished op、ready callback、unlock callback。

这些都是 Global GET 的请求级上下文。因此现在的边界是：

```text
CacheEngine.match_swa()              -> 单 tier SWA node/slot 解析
GlobalCacheEngine._resolve_swa_get_source() -> 多 tier GET source plan
SWACacheManager.build_get_chain()    -> 根据 source plan 追加 graph ops
```

这样既保留了“GET 一次解析出所有 SWA 建图信息”的清晰度，又不把单 tier radix match 和 transfer graph 绑死。

## 当前 API 边界

- `match_swa()` 是唯一入口。
- `match_result` 参数表示“复用 Full-KV 同一轮 match 的 SWA metadata”。
- `lock_for_load` 参数表示“这个 slot 会被异步 SWA load 读取，需要 pin 到完成回调”。
- `return_node` 参数表示“调用方需要拿 node handle 交给完成回调”。
- `_resolve_swa_get_source()` 是唯一生产 GET 调用方：它同时需要 `match_result`、`lock_for_load=True`、`return_node=True`，并返回结构化 `SWAGetSource`。
