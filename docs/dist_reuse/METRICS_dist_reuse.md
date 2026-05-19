# Dist-Reuse 监控指标接入清单

> 配套 `KNOWN_ISSUE_p2p_refcount_2026-05-14.md` §4 的具体落地。
> 本文档列出**已经在代码里埋点**的指标 vs **需要 C++ build 才能加**的指标。

## 启用方式

```bash
export FLEXKV_ENABLE_METRICS=1
export FLEXKV_PY_METRICS_PORT=8080  # 默认 8080
# PROMETHEUS_MULTIPROC_DIR 会自动创建为 /tmp/flexkv_prom_<parent_pid>
# 如需自定义可显式 export
```

启动 FlexKV 后访问 `http://<host>:8080/metrics` 即可看到所有指标。

## ⚠️ 多进程聚合（已自动启用）

FlexKV 的数据面 worker（`PEER2CPUTransferWorker` 等）跑在 `mp.Process`
子进程里。**朴素的 `prometheus_client` 不会跨进程聚合**——子进程递增的
counter 不会出现在父进程的 `/metrics` 端点。

本项目已通过 prometheus_client 官方的 `PROMETHEUS_MULTIPROC_DIR` +
`MultiProcessCollector` 机制解决：

- `flexkv/metrics/collector.py::_bootstrap_multiproc_dir` 在 import
  `prometheus_client` **之前**自动创建一个 multiproc dir，并设置环境变量
- `flexkv/metrics/server.py::start_metrics_server` 检测到 multiproc dir 后
  使用 `MultiProcessCollector(registry, path=multiproc_dir)` 包装 registry
- 所有子进程通过继承父进程的 env，自动写入同一 multiproc dir
- HTTP `/metrics` 端点每次被 scrape 时实时聚合所有进程的样本

**集成测试**：父进程 + spawn 子进程各自递增计数器，从 `/metrics` 端点
读出聚合值正确（5/5 断言通过，见 `_test_metrics_multiproc.py`）。

**运维注意**：
- multiproc dir 默认在 `/tmp/flexkv_prom_<parent_pid>`，pid 不复用就不
  会冲突；docker 容器重启自动清掉
- 如运行很久（pid 漂移），mmap 文件可能累积（每个 worker pid 一个文件，
  ~几十 KB），可定期清理过时 pid 的文件
- 子进程崩溃不会自动清理它的 mmap 文件，但 `MultiProcessCollector`
  会继续读这些样本作为"最后已知值"——不影响监控正确性，只占少量磁盘

---

## 已就绪指标（Python 侧）

### 1. `flexkv_py_dist_reuse_peer_mooncake_read_seconds` (Histogram)

**含义**：peer 端 mooncake `transfer_sync_read` 调用延迟（含失败路径）。

**埋点位置**：`flexkv/transfer/worker.py::PEER2CPUTransferWorker._batch_transfer_impl`
（PEERH2H 分支，包了 mooncake 调用全程）

**告警规则**（PromQL）：

```promql
# P99 mooncake_read 延迟 > 500ms 持续 5 分钟
histogram_quantile(0.99,
    rate(flexkv_py_dist_reuse_peer_mooncake_read_seconds_bucket[5m])
) > 0.5
```

KNOWN_ISSUE §4.2：剩余 lease 缓冲 < 10x 时风险升高。

---

### 2. `flexkv_py_dist_reuse_peer_mooncake_read_failures_total` (Counter, label=reason)

**含义**：peer 端 mooncake_read 失败计数，按 reason 分类：
- `mooncake_error`：mooncake 返回 ret != 0
- `zero_byte_transfer`：data_lens 全 0（这正是 2026-05-14 P0 bug 的特征）

**埋点位置**：同上。

**告警规则**：

```promql
# 失败率 > 0.1%
sum(rate(flexkv_py_dist_reuse_peer_mooncake_read_failures_total[5m]))
/
(
  sum(rate(flexkv_py_dist_reuse_peer_mooncake_read_failures_total[5m]))
  + sum(rate(flexkv_py_dist_reuse_peer_mooncake_read_success_total[5m]))
) > 0.001
```

任何 `reason="zero_byte_transfer"` 出现都应**立即 P0 oncall** —— P0 bug 复发或回归。

---

### 3. `flexkv_py_dist_reuse_peer_mooncake_read_success_total` (Counter)

**含义**：peer 端 mooncake_read 成功计数。`#2` 的分母。

**埋点位置**：同 #1。

---

### 4. master 端 CPU pool 利用率（已有指标派生）

利用率不需要新指标，PromQL 直接算：

```promql
# CPU pool 利用率 (KNOWN_ISSUE §4.1 入口指标)
1 - (
  flexkv_py_mempool_free_blocks{device="cpu"}
  /
  flexkv_py_mempool_total_blocks{device="cpu"}
)
```

**告警规则**：

```promql
# 利用率 > 95% 持续 60s — 可能进入场景 D（KNOWN_ISSUE §2）
(
  1 - (
    flexkv_py_mempool_free_blocks{device="cpu"}
    /
    flexkv_py_mempool_total_blocks{device="cpu"}
  )
) > 0.95
```

---

## 待 C++ build 后才能加的指标

> 这两个指标**必须在 C++ 内部计数**才能准确，Python 侧无法窥探到。
> 等下一次 FlexKV 容器 build 时一起加。

### 5. `flexkv_py_dist_reuse_lease_meta_nullptr_total` (Counter, label=device)

**Python collector 已就绪**（`record_dist_reuse_lease_nullptr`），缺 C++ 侧的 trigger。

**待加位置**：`csrc/dist/local_radix_tree.cpp::publish_node_blocks` 的
`set_lease_meta(nullptr)` 分支（约 L164-167）。

**预期改动**：

```cpp
// csrc/dist/local_radix_tree.cpp L164-167
if ((current_block_count + new_node->size()) > (max_num_blocks - swap_block_threshold)) {
  new_node->set_lease_meta(nullptr);

  // [METRICS] expose to Python via a stats counter accessor
  this->_metrics_lease_nullptr_count += new_node->size();
}
```

加 Python 侧采集（在 `GlobalCacheEngine._update_mempool_metrics` 同节奏轮询）：

```python
# flexkv/cache/cache_engine.py — 新增方法
def _update_dist_reuse_metrics(self):
    if self._metrics_collector is None:
        return
    for device_type, engine in self.cache_engines.items():
        if hasattr(engine, '_radix_tree_stats'):
            stats = engine._radix_tree_stats()  # 新 API
            self._metrics_collector.record_dist_reuse_lease_nullptr(
                DEVICE_TYPE[device_type].lower(),
                stats.lease_nullptr_count_delta
            )
```

**告警规则**（生效后）：

```promql
# 任何 lease_meta=nullptr 都是 CRITICAL
increase(flexkv_py_dist_reuse_lease_meta_nullptr_total[1m]) > 0
```

KNOWN_ISSUE §5 trigger #1：必须立即升级到方案 A/B。

---

### 6. `flexkv_py_dist_reuse_about_to_evict_total` (Counter, label=device)

**Python collector 已就绪**（`record_dist_reuse_about_to_evict`），缺 C++ 侧的 trigger。

**待加位置**：`csrc/dist/local_radix_tree.cpp::evict` L612-650
（fresh-branch 加入 `about_to_evict_q` 时计数）。

**告警规则**（生效后）：

```promql
# 健康比例：fresh 标记 / 真实 evict <= 1
# 持续 > 10:1 说明 master 在死撑 evict
sum(rate(flexkv_py_dist_reuse_about_to_evict_total[5m]))
/
sum(rate(flexkv_py_evicted_blocks_total[5m]))
> 10
```

---

## 业务侧仍需自建的指标

### 7. `cross_instance_hit_text_garbage_rate`

**这个不能由 FlexKV 自动采集**，必须**业务层（sglang）加抽样工具**：

```
对跨实例命中的请求，按 1% 抽样：
1. 完整跑一遍 prefill（不命中 cache）
2. 比较 token-id 序列与原命中结果
3. 不一致比例 > 0.01% → P0 oncall
```

KNOWN_ISSUE §5 trigger #2：业务层观察到生成质量退化。

---

## Grafana Dashboard 推荐 panel

| Panel | 数据源 | 阈值 |
|---|---|---|
| CPU pool 利用率 | `1 - free/total` | warning > 80%, critical > 95% |
| 跨实例 mooncake_read P99 | histogram_quantile(0.99, ...) | warning > 500ms |
| 跨实例 mooncake_read 失败率 | failures / (failures + success) | warning > 0.1%, critical > 1% |
| zero_byte_transfer 计数 | `failures_total{reason="zero_byte_transfer"}` | critical > 0 |
| lease_nullptr 计数（待 C++ build） | `lease_meta_nullptr_total` | critical > 0 |
| fresh/expired evict 比 | `about_to_evict / evicted_blocks` | warning > 5, critical > 10 |

---

## 测试

### 验证 metrics 启动

```bash
# 容器内
export FLEXKV_ENABLE_METRICS=1
export FLEXKV_PY_METRICS_PORT=8080
python3 -c '
from flexkv.metrics import init_global_collector
c = init_global_collector()
print("collector enabled:", c.enabled)
'

# 应输出：
# [FlexKV PyMetrics] Prometheus metrics collector initialized
# collector enabled: True
```

### 触发并查看指标

```bash
# 跑一次 P2P 跨实例 e2e（按你昨天的双机 harness）
# 然后：
curl -s http://localhost:8080/metrics | grep dist_reuse

# 应能看到：
# flexkv_py_dist_reuse_peer_mooncake_read_seconds_bucket{le="..."} ...
# flexkv_py_dist_reuse_peer_mooncake_read_success_total ...
```

---

## 升级路线

1. **立即可做**（本次落盘）：5 个 Python 指标已埋好；CPU pool 利用率走 PromQL 派生
2. **下次 C++ build 时**：加 `lease_meta_nullptr` + `about_to_evict` 的 C++ counter
3. **业务立项时**：sglang 侧加抽样工具采 `cross_instance_hit_text_garbage_rate`

---

*文档生成时间：2026-05-14*
