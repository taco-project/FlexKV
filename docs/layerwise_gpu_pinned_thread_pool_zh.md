# Layerwise GPU 绑定线程池改造方案

> **状态（2026-07-22）**：Phase 1 已在 `csrc/layerwise.{h,cpp}` 落地（跳过 Phase 0）。
> 热路径（single-group / multi-group / SWA / HOSTFUNC / POLLING）均经 per-GPU pinned worker 提交。

## 1. 改造原因

### 1.1 问题

`LayerwiseTransferGroup` 在 GPU H2D 热路径上由**调用线程**反复执行 `cudaSetDevice`：

- multi-group：对每个 orig layer，循环顺序为 `member → GPU`，每层 `cudaSetDevice` 次数 ≈ `members × num_gpus`
- SWA sidecar（uniform / multi-group）同样按 GPU 循环切 device
- POLLING：`cudaEventRecord` 与 `event_polling_loop` 查询时也会反复 `cudaSetDevice`

`cudaSetDevice` 在驱动侧有不可忽略的 context 切换成本。层数多、TP 度高、DSv4 multi-group（main members + SWA members）叠加后，单次 layerwise GET 可累积数百到上千次 SetDevice，成为 CPU 侧提交延迟的主要来源之一。

### 1.2 已有解法（TP 路径）

`TPTransferThreadGroup` 已用 **per-GPU 常驻线程池**解决同类问题：

```text
构造时：每个 GPU 起一条 worker → cudaSetDevice(device) 只做一次
热路径：enqueue_for_gpu(i, task) → worker 在已绑定 device 上跑 kernel / CE
同步：  futures.get() 等待本批提交完成
```

关键实现见 `csrc/tp_transfer_thread_group.cpp`（构造线程池 + `enqueue_for_gpu`）。

### 1.3 为何 layerwise 也适用

两边的 GPU 侧工作同构：

| | TPTransferThreadGroup | LayerwiseTransferGroup |
|--|----------------------|------------------------|
| 工作内容 | 在各 GPU stream 上 launch transfer | 同左（+ SWA + hostfunc/event） |
| 当前开销 | 已消除重复 SetDevice | 主线程按层 × member × GPU 反复 SetDevice |
| 差异 | 一次调用覆盖一段 layer，整批 sync | **必须按 orig 顺序推进**，靠 eventfd 通知层完成 |

结论：**可以复用「per-GPU 绑定 + enqueue」机制**；不能原样嵌套整个 `TPTransferThreadGroup`（其 API 是 bulk sync，且不含 per-layer HOSTFUNC/POLLING），需要在 layerwise 内引入同构的 GPU pinned pool，并保持层序语义。

---

## 2. 改造方案

### 2.1 目标与非目标

**目标**

1. Layerwise H2D 热路径上，`cudaSetDevice` 从「每层 × members × GPUs」降为「每个 GPU worker 生命周期内一次」。
2. 保持现有层完成语义：HOSTFUNC eventfd / POLLING eventfd 顺序与计数不变。
3. 单 group 与 multi-group、有/无 SWA、HOSTFUNC/POLLING 均可走同一套提交模型。

**非目标（本阶段不做）**

- 不跨 orig layer 乱序并行提交（避免层完成通知语义复杂化）。
- 不改变 Python API、`LayerwiseTransferOp` 图语义、SSD/REMOTE 路径。
- 不强制把 `TPTransferThreadGroup` 与 layerwise 合并成一个类（可后续再抽公共 `GpuPinnedThreadPool`）。

### 2.2 总体架构

```text
LayerwiseTransferGroup
  ├─ 构造：创建 GpuPinnedWorkers[num_gpus]
  │         每个 worker: cudaSetDevice(id) 一次 → 循环取队列任务
  ├─ 每层（主线程顺序 for work_origs）:
  │     for d in GPUs:
  │       enqueue(d):  // 在已绑定 device 上执行
  │         for member in main_members[orig]:
  │           transfer_kv_blocks(..., streams_[d], sync=false)
  │         if swa: launch SWA members for GPU d on streams_[d]
  │         if HOSTFUNC: 在本 GPU 对应 stream 上挂齐 slots 的 hostfunc
  │         if POLLING:  cudaEventRecord(poll_event[d], streams_[d])
  │     wait futures  // 只等「本层已提交」，不等 GPU 算完
  └─ 层真正完成：仍由 HOSTFUNC / POLLING 写 eventfd（与现网一致）
```

关键不变量：

1. **层循环仍在主线程**，保证 `work_origs` 顺序与 NVTX / next-layer 逻辑不变。
2. **层内跨 GPU 并行提交**；同一 GPU 上多个 member 串行 launch（与 stream 顺序一致）。
3. `futures.get()` ≠ GPU 完成；GPU 完成仍靠现有 notify 路径。

### 2.3 分阶段落地

#### Phase 0（可选快赢，可与 Phase 1 并行或先行）

**循环换序**：把 `for member: for gpu: SetDevice` 改为 `for gpu: SetDevice; for member: launch`。

- 改动：`layerwise_transfer_multi_group`、`launch_swa_mg_h2d_layer_`、单 group 同类循环。
- 效果：SetDevice 次数除以 `members`（DSv4 约 2–3×），无新线程模型风险。
- 若直接上 Phase 1，Phase 0 可跳过（线程池下 SetDevice 已常数化）。

#### Phase 1（主方案）：Layerwise 内嵌 per-GPU 线程池

**新增成员（示意）**

```cpp
using GpuTask = std::function<void()>;
std::vector<std::thread> gpu_workers_;
std::vector<std::queue<GpuTask>> gpu_queues_;
std::vector<std::mutex> gpu_mtxs_;
std::vector<std::condition_variable> gpu_cvs_;
std::atomic<bool> stop_gpu_pool_{false};

std::future<void> enqueue_for_gpu(int gpu_idx, GpuTask task);
```

**构造**：在现有 `streams_` / `events_` 创建之后启动 worker（与 TP 相同：`cudaSetDevice` 一次后 wait queue）。

**析构顺序**（重要）：

1. 停 POLLING 线程（若有）
2. `stop_gpu_pool_ = true`，notify + join GPU workers
3. 再 destroy stream / event / handler

**热路径改造点**

| 函数 | 改造要点 |
|------|----------|
| `layerwise_transfer`（单 group） | 每 batch：对每 GPU enqueue「launch + hostfunc/event」；主线程 wait futures |
| `layerwise_transfer_multi_group` | 每 orig：对每 GPU enqueue「main members + SWA + hostfunc/event」 |
| `launch_swa_h2d_layer_` / `launch_swa_mg_h2d_layer_` | 改为「按 GPU 的 task 片段」或内联进上层 enqueue task，避免主线程再 SetDevice |
| `event_polling_loop` | 优先：每 GPU 由本 GPU worker 查询；或主线程 `cudaEventQuery` 且去掉不必要的 SetDevice（event query 通常可不切 device，需实测确认） |

**HOSTFUNC 计数**

保持：

```text
expected_count = slots_per_gpu * num_gpus_
slots_per_gpu  = members_this_layer + swa_slots
```

仅把 `cudaLaunchHostFunc` 的调用点挪到对应 GPU worker、对应 `streams_[d]` 上；回调逻辑与 eventfd 不变。

**POLLING**

`cudaEventRecord` 放进该 GPU 的 layer task 末尾；polling 线程只负责 query + 推进 batch，尽量不再 SetDevice。

### 2.4 与 TP 代码关系

推荐两步走：

1. **先在 `LayerwiseTransferGroup` 内复制/精简 TP 的 queue+worker 模式**（改动面可控，便于单独回归）。
2. **稳定后再抽公共 `GpuPinnedThreadPool`**，供 `TPTransferThreadGroup` 与 layerwise 共用，避免两套逻辑漂移。

不建议第一版直接让 layerwise 持有一个 `TPTransferThreadGroup`：接口、stream 所有权、HOSTFUNC/SWA 语义都不匹配。

### 2.5 风险与约束

| 风险 | 缓解 |
|------|------|
| 跨层并行破坏 eventfd 顺序 | 主线程严格按层 enqueue + wait submit |
| 析构竞态 | 固定 stop 顺序：poll → gpu pool → CUDA 资源 |
| CE / staging 线程安全 | 与 TP 一致：按 device 隔离；task 内只碰本 GPU 资源 |
| 单 GPU 收益小 | 仍走同一代码路径，保持实现统一 |
| 线程唤醒开销 | 每层每 GPU 一次 enqueue；相对 SetDevice 通常可忽略，可用微基准验证 |

### 2.6 测试与验证

1. **功能**：现有 layerwise / multi-group / SWA fuse / HOSTFUNC & POLLING 单测与集成测。
2. **正确性**：首层 eventfd 仍最早到达；`expected_count` 与层完成次数一致。
3. **性能**（建议加轻量 NVTX / 计数）：
   - 统计单次 GET 的 `cudaSetDevice` 次数（应用层计数或 CUPTI）
   - 统计主线程「层提交延迟」：从进入层循环到本层 futures ready
   - 对比改造前后 wall time（固定 layers / TP / block 数）

---

## 3. 预期结果

### 3.1 行为不变

- Python / TransferEngine 接口不变。
- 层完成通知时序语义不变（首层仍可最早 ready）。
- HOSTFUNC / POLLING 两种 notify 模式均可工作。
- SWA fuse、multi-group members 数据布局不变。

### 3.2 性能预期

以典型 multi-group GET 粗算（仅主路径 H2D 提交侧）：

| 指标 | 改造前 | Phase 0（换序） | Phase 1（线程池） |
|------|--------|-----------------|-------------------|
| `cudaSetDevice` 次数 | `≈ L × M × G`（+ SWA / poll） | `≈ L × G` | `≈ G`（构造期） |
| 例：L=60, M=2, G=8 | ~960（仅 main） | ~480 | ~8 |

其中 `L`=工作层数，`M`=每层 member 数，`G`=`num_gpus`。

定性预期：

1. **CPU 提交路径明显变短**：尤其 DSv4（M≥2）+ 高 TP + SWA sidecar。
2. **首层 ready 延迟下降**：减少 SetDevice 排队后，第一批 kernel/hostfunc 更早挂上 stream。
3. **GPU 拷贝带宽本身未必显著上升**（瓶颈若在 PCIe/CE）；主要收益在 **host 调度与首层 TTFB**。
4. 与 TP 路径对齐后，layerwise 不再是「唯一仍在热路径疯狂 SetDevice」的模块，后续调优更可预期。

### 3.3 成功标准（建议）

1. 功能回归：layerwise 相关单测 / 集成测全绿。
2. 热路径：单次 multi-group GET 中，worker 启动后 **不再出现** 周期性 `cudaSetDevice`（构造/析构除外）。
3. 性能：在约定 benchmark（例如 8 GPU、DSv4 multi-group + SWA fuse）上，**层提交 CPU 耗时**相对 baseline 有可测下降（目标量级：与 SetDevice 次数降幅同方向，具体百分比以实测为准）。

---

## 4. 建议排期

| 步骤 | 内容 | 产出 |
|------|------|------|
| A | 评审本方案，确认 Phase 0 是否跳过 | 决策 |
| B | Phase 1：pool + multi-group 热路径 | 可跑 patch |
| C | 单 group + SWA + POLLING 对齐 | 全路径覆盖 |
| D | 微基准 + 回归 | 数据与结论 |
| E（可选） | 抽取公共 `GpuPinnedThreadPool`，TP/layerwise 共用 | 去重 |

---

## 5. 一句话总结

**原因**：layerwise 热路径重复 `cudaSetDevice` 成本高，TP 路径已用 per-GPU 绑定线程池证明可行。  
**方案**：在 `LayerwiseTransferGroup` 引入同构 GPU pinned pool；主线程按层顺序 enqueue，层内跨 GPU 并行提交，notify 语义保持不变。  
**预期**：SetDevice 从 `O(L×M×G)` 降到 `O(G)`，降低层提交与首层 ready 延迟，功能与图语义不变。
