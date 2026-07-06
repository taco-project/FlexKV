# FlexKV 缓存驱逐策略指南

本文档介绍 FlexKV 支持的所有缓存驱逐策略，包括其行为、配置方式和适用场景。

---

## 概述

FlexKV 是位于 GPU 之下的多级缓存层（CPU 内存 / SSD / 远端存储），GPU KV cache 本身仍由推理引擎管理，不在 FlexKV 的驱逐范围内。当某个缓存层（CPU/SSD/远端）空间不足或利用率超过阈值时，FlexKV 会从**该层对应的 Radix Tree** 中驱逐（移除）缓存块以腾出空间。**驱逐策略**决定了哪些缓存块优先被移除。

---

## 多级缓存包含关系

FlexKV 的多级缓存采用近似 **inclusive（包含式）/ write-through** 的模型：在 `put` 阶段，数据会先写入 CPU 缓存；如果 SSD 或远端存储启用且可用，FlexKV 会继续尝试把对应层级尚未命中的 block 补写到这些更低层级中。因此，同一份 KV block 通常会在多个缓存层级中保留副本。

基于这个模型，某一层发生驱逐时，FlexKV 只会从**当前层级**的索引中移除对应 block，并回收该层的物理 block 空间；驱逐本身不会触发向下一层的写回、迁移或额外传输。例如，CPU 层驱逐不会主动发起 CPU→SSD 传输，SSD 层驱逐也不会主动发起 SSD→远端传输。

需要注意的是，这种包含关系是**尽力而为**的：如果某个下层缓存未启用、被临时策略跳过、空间不足或传输失败，那么该层可能没有对应副本。此时上层驱逐后，后续请求只能从其他已有层级命中，或者发生缓存未命中。

---

## 支持的驱逐策略

| 策略 | 说明 | 驱逐顺序 |
|------|------|----------|
| `lru` | 最近最少使用 | 优先驱逐**最久未被访问**的节点 |
| `lfu` | 最不经常使用 | 优先驱逐**命中次数最少**的节点；次数相同时按 LRU 排序 |
| `slru` | 分段最近最少使用 | 将节点分为试用段和保护段，优先驱逐试用段中**最久未访问**的节点 |
| `fifo` | 先进先出 | 优先驱逐**最早插入**的节点 |
| `mru` | 最近最多使用 | 优先驱逐**最近刚被访问**的节点 |
| `filo` | 先进后出 | 优先驱逐**最近插入**的节点 |

### 默认策略

默认驱逐策略为 `lru`，适用于大多数推理服务场景。

---

## 配置方式

### 通过配置文件

```yml
eviction_policy: lru
```

或使用 JSON 格式：

```json
{
  "eviction_policy": "lru"
}
```

### 通过环境变量

```bash
export FLEXKV_EVICTION_POLICY=lru
```

可选值：`lru`、`lfu`、`slru`、`fifo`、`mru`、`filo`。

---

## 驱逐相关配置项

除了 `eviction_policy` 本身，FlexKV 还提供以下配置项来控制驱逐行为。所有配置均支持通过环境变量或配置文件设置。

| 配置项 | 环境变量 | 类型 | 默认值 | 说明 |
|--------|---------|------|--------|------|
| `eviction_policy` | `FLEXKV_EVICTION_POLICY` | str | `lru` | 驱逐策略，可选值见上方[支持的驱逐策略](#支持的驱逐策略) |
| `evict_start_threshold` | `FLEXKV_EVICT_START_THRESHOLD` | float | `0.7` | 触发主动驱逐的缓存利用率阈值。当缓存占用比例达到该值时，FlexKV 开始主动驱逐节点。例如 `0.7` 表示缓存占用达到 70% 时即开始驱逐；设为 `1.0` 则仅在缓存满时才驱逐 |
| `evict_ratio` | `FLEXKV_EVICT_RATIO` | float | `0.05` | 每次驱逐的最小淘汰比例。例如 `0.05` 表示每次至少淘汰总 block 数的 5%，以减少频繁小量驱逐带来的开销。设为 `0.0` 则仅淘汰满足当前需求所需的最少 block 数 |
| `hit_reward_seconds` | `FLEXKV_HIT_REWARD_SECONDS` | int | `0` | 命中奖励秒数，仅对 LRU 策略生效。每次缓存命中时向节点的有效访问时间叠加指定秒数（可累积），使命中越多的节点越难被驱逐。设为 `0` 时为标准 LRU 行为。注意：该参数对其他策略（LFU/SLRU/FIFO/MRU/FILO）不生效 |
| `slru_protected_threshold` | `FLEXKV_SLRU_PROTECTED_THRESHOLD` | int | `2` | SLRU 策略的保护阈值。节点命中次数达到该值后进入保护段，不易被驱逐。仅对 SLRU 策略生效 |

### 配置示例

**环境变量方式：**
```bash
export FLEXKV_EVICTION_POLICY=lru
export FLEXKV_EVICT_START_THRESHOLD=0.7
export FLEXKV_EVICT_RATIO=0.05
export FLEXKV_HIT_REWARD_SECONDS=10
```

**配置文件方式（yml）：**
```yml
eviction_policy: lru
evict_start_threshold: 0.7
evict_ratio: 0.05
hit_reward_seconds: 10
```

### 驱逐触发机制

FlexKV 在以下任一条件满足时触发驱逐：

1. **缓存利用率超过阈值**：当前缓存利用率 ≥ `evict_start_threshold`
2. **空闲 block 不足**：当前请求所需的 block 数 > 可用空闲 block 数

触发驱逐后，实际驱逐的 block 数量取以下三者的最大值：
- 满足当前请求所需的最少 block 数
- 将利用率降回阈值以下所需的 block 数
- `evict_ratio × 总 block 数`（最小批量淘汰）

---

## 各策略详解

### LRU（最近最少使用）

- **优先级值**：`last_access_time`
- **行为**：最长时间未被访问的节点最先被驱逐。
- **适用场景**：通用推理服务，最近使用的前缀大概率会被再次使用。

#### LRU 增强：`hit_reward_seconds`

FlexKV 提供了可选的**命中奖励**机制，为 LRU 增加频率感知能力。当配置了 `hit_reward_seconds`（默认值 `0`，即标准 LRU 行为）时，每次缓存命中都会向节点的有效访问时间额外叠加指定的秒数（可累积），从而使命中次数越多的节点越难被驱逐。

**配置方式**：
```bash
export FLEXKV_HIT_REWARD_SECONDS=10
```
或在配置文件中：
```yml
hit_reward_seconds: 10
```

### LFU（最不经常使用）

- **优先级值**：`(hit_count, last_access_time)`
- **行为**：缓存命中次数最少的节点最先被驱逐。命中次数相同时，最久未被访问的节点优先驱逐。
- **适用场景**：访问模式高度倾斜的场景，例如某些 system prompt 或前缀被大量复用。

### FIFO（先进先出）

- **优先级值**：`creation_time`
- **行为**：最早插入的节点最先被驱逐，不考虑访问模式。
- **适用场景**：缓存新鲜度比复用频率更重要的场景，或需要可预测驱逐顺序的场景。

### MRU（最近最多使用）

- **优先级值**：`-last_access_time`
- **行为**：最近刚被访问的节点最先被驱逐，是 LRU 的反向策略。
- **适用场景**：最近访问的内容不太可能被再次使用的场景（例如单次查询、唯一前缀）。

### FILO（先进后出）

- **优先级值**：`-creation_time`
- **行为**：最近插入的节点最先被驱逐，是 FIFO 的反向策略。
- **适用场景**：较旧的缓存内容更有价值、需要更长时间保留的场景。

### SLRU（分段最近最少使用）

- **优先级值**：`(is_protected, last_access_time)`
- **行为**：将缓存节点逻辑上分为两个段：
  - **Probationary（试用段）**：`hit_count < protected_threshold`（默认 2）的节点。新插入的节点默认进入此段。
  - **Protected（保护段）**：`hit_count >= protected_threshold` 的节点。被多次访问的节点自动晋升到此段。
  
  驱逐时优先淘汰 Probationary 段中的节点；同段内按 LRU 规则（最久未访问的先淘汰）。Protected 段的节点只有在 Probationary 段全部被淘汰后才会被驱逐。
- **适用场景**：混合工作负载场景，既有高频复用的 system prompt，又有大量一次性查询。SLRU 能有效抵抗缓存扫描污染（cache scan pollution），避免大量一次性请求冲刷掉高频缓存。
- **注意**：SLRU 同段内按 `last_access_time` 排序，因此 `hit_reward_seconds`（仅作用于 `grace_time`）对 SLRU 不生效。

**配置方式**：
```bash
export FLEXKV_EVICTION_POLICY=slru
export FLEXKV_SLRU_PROTECTED_THRESHOLD=2  # 可选，默认值为 2
```
或在配置文件中：
```yml
eviction_policy: slru
slru_protected_threshold: 2
```
