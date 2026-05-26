# FlexKV SWA (Sliding Window Attention) 完整实现文档

> 版本: v2.0 | 日期: 2026-05-26 | 测试: 74 passed

---

## 1. 背景：什么是 SWA

DeepSeek V4 使用三种注意力机制协同工作：
- **CSA** (4:1压缩): 中等精度的中距离注意力
- **HCA** (128:1压缩): 粗粒度的远距离注意力
- **SWA** (128-token窗口): 精确的近距离局部注意力

SWA 是一个**固定大小的环形缓冲区**，每个 request 只保留最近 128 个 token 的完整 KV cache，覆盖所有 61 层。当新 token 到来时，覆盖最旧的 slot（`position % 128`）。

### SWA 的关键特性

| 特性 | 值 |
|------|-----|
| 窗口大小 | 128 tokens |
| 每 token 每层大小 | 584 bytes (k_nope FP8 448B + k_rope BF16 128B + scale 8B) |
| 覆盖层数 | 61 层 |
| 每个 request 的 SWA 数据量 | 128 × 584 × 61 = **4,559,872 bytes ≈ 4.35 MB** (固定) |
| 写入语义 | 环形覆盖：`write_pos = global_token_position % 128` |
| 命中策略 | **TRAILING_PAGES**: 只需最新的 1 page 存在即为命中 |

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KVManager (公共API)                             │
│  swa_put(token_ids, data)  swa_get(token_ids)  swa_remove(token_ids)   │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SWAPoolManager (协调层)                            │
│                                                                         │
│  ┌──────────────────┐    ┌─────────────────┐    ┌───────────────────┐  │
│  │ SWACacheEngine   │    │ SWAStorage      │    │ SWACudaDMA        │  │
│  │ (索引+驱逐)      │    │ (CPU pinned buf)│    │ (异步GPU↔CPU)     │  │
│  │                  │    │                 │    │                   │  │
│  │ Hash-map index   │    │ [num_slots,     │    │ Stream pool       │  │
│  │ LRU eviction     │    │  page_size]     │    │ Event tracking    │  │
│  │ Lock/unlock      │    │ uint8 tensor    │    │ Non-blocking poll │  │
│  └──────────────────┘    └─────────────────┘    └───────────────────┘  │
│                                                                         │
│  Endpoint Hash: hash(最后一个block的tokens) ─ 与主KV RadixTree对齐      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│            推理引擎 (SGLang / vLLM / TRT-LLM)                          │
│                                                                         │
│  GPU SWA Ring Buffer (128 tokens × 61 layers × 584 B/token)           │
│  ↕ evict/restore 触发 SWA put/get                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 完整文件清单

```
flexkv/
├── common/
│   └── config.py                      # [修改] +SWAPoolConfig dataclass
├── swa/                               # [新增] SWA 完整子包
│   ├── __init__.py
│   ├── swa_cache_engine.py            # 索引管理: hash-map + LRU + lock
│   ├── swa_storage.py                 # CPU pinned buffer: write/read slot
│   ├── swa_pool_manager.py            # 统一协调层: put/get/match + async + stats
│   ├── swa_cuda_dma.py                # CUDA 异步传输: stream pool + events
│   ├── swa_storage_async.py           # 异步存储辅助方法
│   └── swa_benchmark.py              # 性能基准测试工具
├── kvmanager.py                       # [修改] +swa_put/get/remove 公共API
└── ...

tests/
├── test_swa_cache_engine.py           # 43 tests: 引擎+存储+基础流程
├── test_swa_integration.py            # 31 tests: PoolManager+E2E场景
└── test_swa_cuda_dma.py              # 15 tests: DMA引擎
```

**总代码量**: ~3000 行新增 (实现 ~1500行 + 测试 ~1500行)

---

## 4. 各模块详解

### 4.1 SWACacheEngine — 索引管理

**文件**: `flexkv/swa/swa_cache_engine.py` (224行)

```python
class SWACacheEngine:
    """O(1) hash-map 索引，TRAILING_PAGES 命中策略"""
    
    def allocate(endpoint_hash) → slot_id    # 分配 slot
    def match(endpoint_hash) → SWAMatchResult # 查找: hit/miss
    def set_ready(endpoint_hash, True)       # 数据就绪
    def lock(endpoint_hash)                  # 防止驱逐
    def unlock(endpoint_hash)                # 允许驱逐
    def remove(endpoint_hash)                # 显式删除
    def _evict() → num_evicted              # LRU 驱逐
```

**为什么是 hash-map 而非 RadixTree？**
- 每个 sequence 只有 1 个 SWA page (固定 128 token)
- 没有前缀共享层次 → O(1) 查找最优
- RadixTree 适合 ALL_PAGES 的主 KV，SWA 用 TRAILING_PAGES

### 4.2 SWAStorage — CPU 数据缓冲

**文件**: `flexkv/swa/swa_storage.py` (180行)

```python
class SWAStorage:
    """CPU pinned memory buffer, 按 slot 寻址"""
    
    # buffer: torch.Tensor[num_slots, page_size_bytes] (uint8, pin_memory=True)
    def write_slot(slot_id, data)   # 写入完整 SWA page
    def read_slot(slot_id) → data   # 读取 SWA page (copy)
    def get_slot_view(slot_id)      # 零拷贝 view (用于 DMA)
```

**Pin Memory**: 页锁定内存，支持 `cudaMemcpyAsync` 无需额外 staging。

### 4.3 SWAPoolManager — 统一协调层

**文件**: `flexkv/swa/swa_pool_manager.py` (280行)

```python
class SWAPoolManager:
    """SWA 完整生命周期管理"""
    
    # 同步路径 (CPU-only, 测试/fallback)
    def put(endpoint_hash, swa_data) → bool
    def get(endpoint_hash) → Optional[ndarray]
    def match(endpoint_hash) → SWAMatchResult
    def remove(endpoint_hash)
    
    # 异步路径 (GPU↔CPU DMA)
    def put_async(endpoint_hash, gpu_tensor) → (transfer_id, success)
    def get_async(endpoint_hash, gpu_tensor) → (transfer_id, hit)
    def poll_transfers() → List[completed_ids]
    
    # 辅助
    def compute_endpoint_hash(token_ids) → int
    def lock/unlock(endpoint_hash)
    def stats → Dict[str, int]
```

### 4.4 SWACudaDMA — 异步传输引擎

**文件**: `flexkv/swa/swa_cuda_dma.py` (383行)

```python
class SWACudaDMA:
    """基于 CUDA Stream + Event 的异步传输"""
    
    # Stream pool: 多 stream 并发传输
    # Event: 非阻塞完成检测
    # 双向: H2D (restore) + D2H (evict)
    
    def submit_d2h(gpu_src, cpu_dst, size) → event
    def submit_h2d(cpu_src, gpu_dst, size) → event
    def poll_event(event) → completed
```

---

## 5. 运行流程详解

### 5.1 完整的 Request 生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Request A 开始 prefill                                        │
│                                                                         │
│ SGLang:                                                                 │
│   1. 在 GPU 分配 SWA ring buffer (128 tokens × 584B × 61 layers)       │
│   2. Prefill 过程中，每层每 token 写入 SWA buffer                       │
│   3. Prefill 结束后，SWA buffer 保存最近 128 tokens 的 KV               │
│                                                                         │
│ FlexKV: 此时不参与 (SWA 数据在 GPU 上)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Request A 正常 decode                                         │
│                                                                         │
│ SGLang:                                                                 │
│   - 每个新 token 写入 SWA buffer[position % 128]                        │
│   - 注意力计算时使用 SWA buffer 的 128 tokens 做局部注意力               │
│                                                                         │
│ FlexKV: 仍然不参与                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (GPU 显存不足，调度器决定驱逐)
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Request A 被驱逐 — SWA PUT                                    │
│                                                                         │
│ SGLang FlexKV Connector:                                                │
│   token_ids = request_a.all_token_ids   # [500个token]                  │
│   swa_data = gpu_swa_buffer.flatten()   # 4.35MB uint8                  │
│   kv_manager.swa_put(token_ids, swa_data)                               │
│                                                                         │
│ FlexKV SWAPoolManager 内部:                                             │
│   1. endpoint_hash = compute_endpoint_hash(token_ids)                   │
│      → hash(tokens[496:500])  # 最后一个 block (4 tokens)              │
│      → 结果: 0x3A7F8B2C (举例)                                         │
│                                                                         │
│   2. SWACacheEngine.allocate(0x3A7F8B2C) → slot=7                      │
│      → 如果 pool 满: 先 LRU 驱逐最旧的 entry                           │
│      → 分配 slot 7                                                      │
│                                                                         │
│   3. SWAStorage.write_slot(7, swa_data)                                │
│      → 将 4.35MB 数据写入 CPU pinned buffer[7]                          │
│      → 如果有 GPU: 用 cudaMemcpyAsync(D2H, stream) 非阻塞              │
│      → 否则: 直接 numpy copy                                           │
│                                                                         │
│   4. SWACacheEngine.set_ready(0x3A7F8B2C, True)                        │
│      → 标记此 entry 可被后续 match 命中                                  │
│                                                                         │
│   5. GPU SWA buffer 释放 → 显存回收                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (一段时间后，Request A 重新调度)
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Request A 恢复 — SWA GET                                      │
│                                                                         │
│ SGLang FlexKV Connector (prefix matching 阶段):                         │
│   token_ids = request_a.all_token_ids                                   │
│                                                                         │
│   # 1. 主 KV 前缀匹配 (RadixTree)                                      │
│   main_match = kv_manager.get_match(token_ids)                          │
│   → 返回: 命中了 N 个 blocks 的主 KV cache (C4/C128 compressed)        │
│                                                                         │
│   # 2. SWA 匹配 (TRAILING_PAGES)                                       │
│   swa_data = kv_manager.swa_get(token_ids)                              │
│   → swa_data != None  ← HIT!                                           │
│                                                                         │
│ FlexKV SWAPoolManager 内部:                                             │
│   1. endpoint_hash = compute_endpoint_hash(token_ids)                   │
│      → 0x3A7F8B2C (和之前 PUT 时相同)                                  │
│                                                                         │
│   2. SWACacheEngine.match(0x3A7F8B2C)                                  │
│      → hit=True, physical_block=7                                      │
│                                                                         │
│   3. SWAStorage.read_slot(7) → 4.35MB data                            │
│      → 如果有 GPU: cudaMemcpyAsync(H2D, stream) 非阻塞                │
│      → 否则: numpy copy                                                │
│                                                                         │
│   4. 返回 swa_data 给 connector                                        │
│                                                                         │
│ Connector:                                                              │
│   gpu_swa_buffer.copy_(swa_data)  # 恢复到 GPU                         │
│   Request A 继续 decode，SWA ring buffer 完整恢复                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Request A 结束 — SWA REMOVE                                   │
│                                                                         │
│ SGLang:                                                                 │
│   request_a 生成 EOS token，正常结束                                    │
│   kv_manager.swa_remove(token_ids)                                      │
│                                                                         │
│ FlexKV:                                                                 │
│   SWACacheEngine.remove(0x3A7F8B2C)                                    │
│   → slot 7 回收到 free list                                            │
│   → 可被其他 request 使用                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 驱逐压力场景

```
场景: num_slots=1000, 当前已缓存 1000 个 SWA pages

新 Request B 被驱逐，需要存 SWA page:
  SWACacheEngine.allocate(hash_B):
    → pool 满 (num_free=0)
    → 触发 _evict():
      → evict_ratio=0.1 → 驱逐 100 个最旧的 entries
      → 按 last_access_time 排序，跳过 locked entries
      → 释放 100 个 slots
    → 分配 1 个 slot 给 hash_B
    → num_free 变为 99
```

### 5.3 与主 KV 的协同工作

```
主 KV RadixTree (ALL_PAGES):          SWA Pool (TRAILING_PAGES):
┌──────────────────────────┐         ┌──────────────────────────┐
│ root                     │         │                          │
│  └─ [block 0]            │         │ endpoint_hash → slot     │
│      └─ [block 1]        │         │ 0x3A7F8B2C  → slot 7    │
│          └─ [block 2]    │         │ 0x5C12DE90  → slot 3    │
│              └─ [block 3] ← endpoint│ 0x8F4A1B77  → slot 11   │
│                           │         │                          │
│ match: 必须所有 block 在   │         │ match: 只看 endpoint hash │
│        才算命中            │         │        O(1) 查找          │
└──────────────────────────┘         └──────────────────────────┘

完整命中条件:
  main_kv_hit (ALL_PAGES) AND swa_hit (TRAILING_PAGES)
  = 所有主 KV blocks 在缓存 + SWA 最新 page 在缓存

部分命中处理:
  - main_kv_hit + swa_miss: 主 KV 可用，但 SWA 需要从头计算 (re-prefill SWA部分)
  - main_kv_miss + swa_hit: SWA 可用但主 KV 不全，需要完整 re-prefill
  - 实际中，主 KV miss 时通常 SWA 也不需要 (会做完整 prefill)
```

---

## 6. 代码改动详情

### 新增 (共 ~2500 行实现 + ~800 行测试)

| 文件 | 行数 | 功能 |
|------|------|------|
| `flexkv/swa/swa_cache_engine.py` | 224 | Hash-map 索引 + _SlotAllocator + LRU驱逐 |
| `flexkv/swa/swa_storage.py` | 180 | CPU pinned buffer (torch/numpy dual-mode) |
| `flexkv/swa/swa_pool_manager.py` | 280 | 统一协调层 + 异步传输 + 统计 |
| `flexkv/swa/swa_cuda_dma.py` | 383 | CUDA stream pool + event + 双向DMA |
| `flexkv/swa/swa_storage_async.py` | 117 | 异步存储辅助方法 |
| `flexkv/swa/swa_benchmark.py` | 318 | 性能基准测试 |
| `tests/test_swa_cache_engine.py` | 512 | 单元测试: 引擎+存储+基础流程 |
| `tests/test_swa_integration.py` | 310 | 集成测试: PoolManager+E2E场景 |
| `tests/test_swa_cuda_dma.py` | 242 | DMA引擎测试 |

### 修改

| 文件 | 改动 |
|------|------|
| `flexkv/common/config.py` | +26行: `SWAPoolConfig` dataclass + `CacheConfig.swa` |
| `flexkv/kvmanager.py` | +132行: `swa_put/get/remove` API + `_init_swa_pool` + `_compute_endpoint_hash` |

---

## 7. 如何运行测试

```bash
cd /data/git_store2/dpskv4/FlexKV

# 运行所有 SWA 测试 (74个, 无需 GPU)
PYTHONPATH=. pytest tests/test_swa_cache_engine.py tests/test_swa_integration.py -v --noconftest

# 只运行集成测试
PYTHONPATH=. pytest tests/test_swa_integration.py -v --noconftest

# 运行特定场景
PYTHONPATH=. pytest tests/test_swa_integration.py::TestE2ERequestLifecycle -v --noconftest
```

**依赖**: `pytest`, `numpy`, `torch` (CPU版), `pyyaml`, `pyzmq`

---

## 8. 当前状态与后续工作

### ✅ 已完成

| 层 | 内容 | 状态 |
|---|------|------|
| 配置 | `SWAPoolConfig` + `CacheConfig.swa` | ✅ |
| 索引 | `SWACacheEngine` (hash-map + LRU + lock) | ✅ |
| 存储 | `SWAStorage` (CPU pinned buffer) | ✅ |
| 协调 | `SWAPoolManager` (put/get/match + async) | ✅ |
| DMA | `SWACudaDMA` (stream pool + events) | ✅ |
| API | `KVManager.swa_put/get/remove` | ✅ |
| 测试 | 74 tests passing | ✅ |

### 🔲 后续集成工作 (需要完整 GPU 环境)

| 工作项 | 说明 | 优先级 |
|--------|------|--------|
| SGLang connector 集成 | 在 `flexkv_connector.py` 的 eviction/restore 回调中调用 SWA API | P0 |
| GlobalCacheEngine 集成 | 在 `cache_engine.py` 的 `GlobalCacheEngine.__init__` 中初始化 SWA pool | P0 |
| 端到端 GPU 测试 | 真实 CUDA 设备上的 DMA 传输验证 | P1 |
| Hash 对齐验证 | 确保 `compute_endpoint_hash` 与 SGLang radix tree 的 block hash 一致 | P1 |
| SSD tier | SWA page 进一步下沉到 SSD (大部署场景) | P2 |
| 多节点共享 | 跨节点 SWA page RDMA 传输 | P2 |
