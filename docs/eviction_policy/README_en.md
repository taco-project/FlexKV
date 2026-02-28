# FlexKV Cache Eviction Policy Guide

This document describes all cache eviction policies supported by FlexKV, including their behavior, configuration, and applicable scenarios.

---

## Overview

When GPU KV cache memory is full, FlexKV evicts (removes) cached blocks from the Radix Tree to free space for new requests. The **eviction policy** determines which cached blocks are removed first.

---

## Supported Eviction Policies

| Policy | Description | Eviction Order |
|--------|-------------|----------------|
| `lru` | Least Recently Used | Evicts the **least recently accessed** nodes first |
| `lfu` | Least Frequently Used | Evicts the **least frequently accessed** nodes first; ties broken by LRU |
| `fifo` | First In First Out | Evicts the **oldest inserted** nodes first |
| `mru` | Most Recently Used | Evicts the **most recently accessed** nodes first |
| `filo` | First In Last Out | Evicts the **most recently inserted** nodes first |

### Default Policy

The default eviction policy is `lru`, which is suitable for most inference serving scenarios.

---

## Configuration

### Via Configuration File

```yml
eviction_policy: lru
```

Or using JSON:

```json
{
  "eviction_policy": "lru"
}
```

### Via Environment Variable

```bash
export FLEXKV_EVICTION_POLICY=lru
```

Supported values: `lru`, `lfu`, `fifo`, `mru`, `filo`.

---

## Eviction-Related Configuration

In addition to `eviction_policy` itself, FlexKV provides the following configuration options to control eviction behavior. All options can be set via environment variables or configuration files.

| Option | Environment Variable | Type | Default | Description |
|--------|---------------------|------|---------|-------------|
| `eviction_policy` | `FLEXKV_EVICTION_POLICY` | str | `lru` | Eviction policy, see [Supported Eviction Policies](#supported-eviction-policies) for available values |
| `evict_start_threshold` | `FLEXKV_EVICT_START_THRESHOLD` | float | `0.7` | Cache utilization threshold to trigger proactive eviction. When cache usage reaches this ratio, FlexKV begins proactively evicting nodes. For example, `0.7` means eviction starts when cache is 70% full; setting to `1.0` means eviction only occurs when cache is completely full |
| `evict_ratio` | `FLEXKV_EVICT_RATIO` | float | `0.05` | Minimum eviction ratio per eviction round. For example, `0.05` means at least 5% of total blocks are evicted each round, reducing overhead from frequent small evictions. Setting to `0.0` evicts only the minimum blocks needed to satisfy the current request |
| `hit_reward_seconds` | `FLEXKV_HIT_REWARD_SECONDS` | int | `0` | Hit reward in seconds, only effective for the LRU policy. Each cache hit adds the specified seconds to the node's effective access time (stackable), making frequently hit nodes harder to evict. Set to `0` for standard LRU behavior |

### Configuration Examples

**Via environment variables:**
```bash
export FLEXKV_EVICTION_POLICY=lru
export FLEXKV_EVICT_START_THRESHOLD=0.7
export FLEXKV_EVICT_RATIO=0.05
export FLEXKV_HIT_REWARD_SECONDS=10
```

**Via configuration file (yml):**
```yml
eviction_policy: lru
evict_start_threshold: 0.7
evict_ratio: 0.05
hit_reward_seconds: 10
```

### Eviction Trigger Mechanism

FlexKV triggers eviction when either of the following conditions is met:

1. **Cache utilization exceeds threshold**: current cache utilization ≥ `evict_start_threshold`
2. **Insufficient free blocks**: blocks needed by the current request > available free blocks

Once eviction is triggered, the actual number of evicted blocks is the maximum of:
- Minimum blocks needed to satisfy the current request
- Blocks needed to bring utilization back below the threshold
- `evict_ratio × total blocks` (minimum batch eviction)

---

## Policy Details

### LRU (Least Recently Used)

- **Priority Value**: `last_access_time`
- **Behavior**: Nodes that have not been accessed for the longest time are evicted first.
- **Best For**: General-purpose serving workloads where recently used prefixes are likely to be reused.

#### LRU Enhancement: `hit_reward_seconds`

FlexKV provides an optional **hit reward** mechanism to enhance LRU with frequency awareness. When `hit_reward_seconds` is configured (default `0`, i.e., standard LRU behavior), each cache hit adds extra seconds to the node's effective access time (stackable), making frequently hit nodes harder to evict.

**Configuration**:
```bash
export FLEXKV_HIT_REWARD_SECONDS=10
```
Or in a config file:
```yml
hit_reward_seconds: 10
```

### LFU (Least Frequently Used)

- **Priority Value**: `(hit_count, last_access_time)`
- **Behavior**: Nodes with the fewest cache hits are evicted first. When two nodes have the same hit count, the one accessed least recently is evicted first.
- **Best For**: Workloads with highly skewed access patterns where some system prompts or prefixes are reused far more than others.

### FIFO (First In First Out)

- **Priority Value**: `creation_time`
- **Behavior**: Nodes that were inserted earliest are evicted first, regardless of access patterns.
- **Best For**: Scenarios where cache freshness is more important than reuse frequency, or when you want predictable eviction order.

### MRU (Most Recently Used)

- **Priority Value**: `-last_access_time`
- **Behavior**: Nodes that were accessed most recently are evicted first. This is the reverse of LRU.
- **Best For**: Workloads where recently accessed items are unlikely to be reused soon (e.g., one-shot queries with unique prefixes).

### FILO (First In Last Out)

- **Priority Value**: `-creation_time`
- **Behavior**: The most recently inserted nodes are evicted first. This is the reverse of FIFO.
- **Best For**: Scenarios where older cached content is more valuable and should be preserved longer.
