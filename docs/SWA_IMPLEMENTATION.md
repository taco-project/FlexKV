# FlexKV SWA (Sliding Window Attention) 实现文档

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
| 每个 request 的 SWA 数据量 | 128 × 584 × 61 ≈ **4.35 MB** (固定) |
| 写入语义 | 环形覆盖：`write_pos = global_token_position % 128` |
| 命中策略 | **TRAILING_PAGES**: 只需最新的 1 page 存在即为命中 |

### 为什么 SWA 不需要历史

超过 128 token 之前的信息已被 CSA/HCA 捕获。SWA 只负责"对最近 128 token 的精确局部注意力"。因此：
- 驱逐时直接丢弃旧数据
- 恢复时只需加载最新的 1 page (~4.35 MB)
- 与主 KV cache 的 ALL_PAGES 策略完全不同

---

## 2. 当前实现的范围和定位

### 本次实现了什么

本次实现是 FlexKV 对 SWA 的**CPU 缓存层支持**，即：

```
推理引擎 (SGLang)              FlexKV SWA Pool
┌───────────────────┐         ┌──────────────────────────────┐
│ GPU SWA Ring Buf  │ ──PUT──→│ SWACacheEngine (索引)        │
│ (128 tokens/req)  │         │   endpoint_hash → slot       │
│                   │ ←─GET── │ SWAStorage (CPU pinned buf)  │
│                   │         │   slot → 4.35 MB page data   │
└───────────────────┘         └──────────────────────────────┘
```

**核心场景**：当 GPU 显存不够需要驱逐 request 时，将其 SWA page 快照保存到 CPU；当 request 重新调度时，从 CPU 恢复 SWA page 到 GPU。

### 本次没有实现什么（需要后续扩展）

| 未实现 | 说明 | 需要时机 |
|--------|------|----------|
| GPU↔CPU DMA 传输 | 当前只有 CPU buffer 的读写，没有集成 CUDA stream 异步拷贝 | 与推理引擎集成时 |
| 与 SGLang FlexKV Connector 的对接 | connector 调用 swa_put/swa_get 的胶水代码 | 集成测试时 |
| 多节点 SWA page 共享 | 跨节点的 SWA page RDMA 传输 | 分布式部署时 |
| SSD 层存储 | SWA page 进一步下沉到 SSD | 大规模部署时 |
| CSA/HCA 压缩 KV 的多 Pool 管理 | 本次只做 SWA，不涉及 C4/C128 | 后续 V4 全量支持 |

---

## 3. 代码改动详情

### 3.1 新增文件

#### `flexkv/swa/swa_cache_engine.py` (224行)

**作用**：SWA page 的索引和生命周期管理。

**核心类**：
```python
class SWACacheEngine:
    """基于 hash-map 的 SWA 缓存引擎，实现 TRAILING_PAGES 命中策略。"""
    
    def match(endpoint_hash) → SWAMatchResult    # O(1) 查找
    def allocate(endpoint_hash) → slot_id        # 分配 slot，满时 LRU 驱逐
    def set_ready(endpoint_hash, True)           # 标记数据传输完成
    def lock/unlock(endpoint_hash)               # 锁定防止驱逐
    def remove(endpoint_hash)                    # 显式删除
```

**设计选择**：
- **为什么用 hash-map 而不是 RadixTree？** 每个 sequence 只有 1 个 SWA page（最新窗口），没有前缀共享层次结构。O(1) 查找最优。
- **为什么用 endpoint_hash 作为 key？** 对应主 KV RadixTree 中 sequence 最后一个 block 的 hash，唯一标识一个 sequence 的"当前位置"。
- **驱逐策略**：LRU（最近最少访问的先驱逐），被锁定的 page 不会被驱逐。

#### `flexkv/swa/swa_storage.py` (180行)

**作用**：SWA page 的实际数据存储。

**核心类**：
```python
class SWAStorage:
    """CPU pinned memory buffer，按 slot 寻址。"""
    
    # buffer shape: [num_slots, page_size_bytes]
    def write_slot(slot_id, data)  # 写入一个完整的 SWA page
    def read_slot(slot_id) → data  # 读取一个完整的 SWA page
    def get_slot_view(slot_id)     # 零拷贝 view（用于 DMA）
```

**设计选择**：
- **Pinned memory**：使用 `torch.zeros(..., pin_memory=True)` 分配页锁定内存，后续可直接用于 GPU DMA 传输而无需额外 staging buffer。
- **Flat layout**：所有 slot 等大小，简单高效，无碎片。

#### `tests/test_swa_cache_engine.py` (512行)

43 个单元测试，覆盖：
- 配置（page size 计算）
- 缓存引擎（分配、匹配、LRU驱逐、锁、重置）
- TRAILING_PAGES 语义验证
- 存储（读写隔离、地址计算）
- 端到端流程（put→get 完整循环、驱逐压力测试）

### 3.2 修改文件

#### `flexkv/common/config.py` (+26行)

新增 `SWAPoolConfig` 数据类：
```python
@dataclass
class SWAPoolConfig:
    enabled: bool = False
    window_size: int = 128
    bytes_per_token_per_layer: int = 584
    num_swa_layers: int = 61
    num_slots: int = 1000
    evict_ratio: float = 0.1
    
    @property
    def page_size_bytes(self) -> int:
        return self.window_size * self.bytes_per_token_per_layer * self.num_swa_layers
```

在 `CacheConfig` 中添加：
```python
swa: Optional[SWAPoolConfig] = None
```

#### `flexkv/kvmanager.py` (+132行)

在 `KVManager` 公共 API 中添加三个方法：
```python
def swa_put(token_ids, swa_data) → bool      # 保存 SWA page 到 CPU
def swa_get(token_ids) → Optional[np.ndarray] # 从 CPU 恢复 SWA page
def swa_remove(token_ids)                      # 删除 SWA page
```

加上内部初始化和 hash 计算方法。

---

## 4. 运行流程示例

### 场景：Request 被驱逐然后恢复

```
时间线：

1. Request A 正在 decode (已生成 500 tokens)
   - GPU 上有 SWA ring buffer：保存 token 373~500 (最近128个)
   
2. GPU 显存不足，调度器决定驱逐 Request A
   ┌─────────────────────────────────────────────────────────┐
   │ SGLang connector 调用:                                  │
   │   kv_manager.swa_put(                                   │
   │       token_ids=[...500个token...],  # 用于计算hash     │
   │       swa_data=gpu_swa_buffer.cpu()   # 4.35MB数据      │
   │   )                                                     │
   │                                                         │
   │ FlexKV 内部:                                            │
   │   1. 计算 endpoint_hash = hash(最后一个block的tokens)    │
   │   2. SWACacheEngine.allocate(endpoint_hash) → slot=7    │
   │   3. SWAStorage.write_slot(7, swa_data)                 │
   │   4. SWACacheEngine.set_ready(endpoint_hash, True)      │
   └─────────────────────────────────────────────────────────┘
   
3. GPU SWA buffer 被释放，显存回收

4. 一段时间后，Request A 重新被调度
   ┌─────────────────────────────────────────────────────────┐
   │ SGLang connector 调用:                                  │
   │   data = kv_manager.swa_get(                            │
   │       token_ids=[...500个token...]                      │
   │   )                                                     │
   │                                                         │
   │ FlexKV 内部:                                            │
   │   1. 计算 endpoint_hash = hash(最后一个block的tokens)    │
   │   2. SWACacheEngine.match(endpoint_hash)                │
   │      → hit=True, physical_block=7                       │
   │   3. SWAStorage.read_slot(7) → 4.35MB data             │
   │   4. 返回数据给 connector                               │
   │                                                         │
   │ Connector 将数据拷贝回 GPU SWA buffer                    │
   │ Request A 继续 decode                                   │
   └─────────────────────────────────────────────────────────┘
```

### 场景：缓存满时的 LRU 驱逐

```
SWA Pool (num_slots=1000):
  slot 0: Request X (last_access=10:00:01) ← 最旧
  slot 1: Request Y (last_access=10:00:02)
  ...
  slot 999: Request Z (last_access=10:05:30) ← 最新

新 Request W 需要缓存 SWA page：
  → Pool 满，触发 LRU 驱逐
  → 驱逐 evict_ratio=10% → 100 个最旧的 slot
  → slot 0~99 被回收
  → Request W 获得空闲 slot
```

---

## 5. 你的顾虑："代码较少，似乎没有办法撑起来"

你的直觉是对的。当前实现是 SWA 支持的**第一层基础设施**，它提供了：

✅ **已有的**：
- SWA page 生命周期管理（分配、匹配、驱逐、锁定）
- CPU 端数据存储
- TRAILING_PAGES 命中策略的正确实现
- 公共 API 供 connector 调用
- 完整的单元测试保证正确性

❌ **还需要补充才能端到端运行的**：

| 缺失部分 | 工作量 | 说明 |
|----------|--------|------|
| **CUDA 异步传输** | ~200行 | 用 `cudaMemcpyAsync` + CUDA stream 做 GPU↔CPU DMA，替代当前的同步 `read_slot/write_slot` |
| **SGLang connector 集成** | ~100行 | 在 `flexkv_connector.py` 中的 `_evict_request()` 和 `_restore_request()` 调用 `swa_put/swa_get` |
| **与主 KV eviction 联动** | ~50行 | 驱逐主 KV 时同步驱逐 SWA page，恢复时同步恢复 |
| **endpoint_hash 对齐** | ~30行 | 确保 hash 计算与 SGLang radix tree 的 block hash 一致 |

### 后续推荐开发顺序

```
Phase 1 (当前完成): SWA 核心引擎 + CPU 存储 + 单元测试
Phase 2 (下一步):   CUDA DMA 传输集成（需要 GPU 环境）
Phase 3:           SGLang connector 对接
Phase 4:           端到端集成测试
```

---

## 6. 文件清单

```
flexkv/
├── common/
│   └── config.py                  # [修改] +SWAPoolConfig
├── swa/                           # [新增] SWA 子包
│   ├── __init__.py
│   ├── swa_cache_engine.py        # 核心：索引 + 驱逐 + 匹配
│   └── swa_storage.py             # CPU buffer 存储
├── kvmanager.py                   # [修改] +swa_put/get/remove API
└── ...

tests/
└── test_swa_cache_engine.py       # [新增] 43个单元测试
```

总代码量：**1074 行新增** (实现 404行 + API 158行 + 测试 512行)
