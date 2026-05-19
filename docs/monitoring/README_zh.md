# FlexKV Prometheus Metrics 文档

FlexKV 集成了基于 [Prometheus](https://prometheus.io/) 的运行时指标监控框架，覆盖 Python 和 C++ 两层关键路径。该框架以**零侵入**方式嵌入 FlexKV 运行时——用户只需设置环境变量 `FLEXKV_ENABLE_METRICS=1`，即可在应用运行期间自动收集缓存命中、内存池状态、数据传输等核心指标，并通过标准 HTTP 端点暴露给 Prometheus 进行采集和可视化（Grafana）。

---

## 一、配置说明

### 1.1 环境变量

| 环境变量 | 默认值 | 描述 |
|---|---|---|
| `FLEXKV_ENABLE_METRICS` | `0` | 启用指标收集（设为 `1` 启用，默认禁用） |
| `FLEXKV_PY_METRICS_PORT` | `8080` | Python 指标 HTTP 服务端口 |
| `FLEXKV_CPP_METRICS_PORT` | `8081` | C++ 指标 HTTP 服务端口 |
| `PROMETHEUS_MULTIPROC_DIR` | *(自动)* | `prometheus_client` 多进程样本文件目录。FlexKV 会在多个 Python 进程（sglang TP/PP worker、KVManager 子进程、transfer worker）中分别写入采样数据，HTTP server 在抓取时通过 `MultiProcessCollector` 聚合。未设置时 collector 会自动初始化一个可写临时目录；建议在长时间运行场景中显式指向 tmpfs 路径（如 `/dev/shm/flexkv_prom`）。 |

### 1.2 配置方式

```bash
# Enable FlexKV metrics collection
export FLEXKV_ENABLE_METRICS=1

# Custom ports (optional)
export FLEXKV_PY_METRICS_PORT=8080
export FLEXKV_CPP_METRICS_PORT=8081
```

---

## 二、指标总览

### 2.1 Python 运行时指标 (`flexkv_py_*`)

Python 指标由 `GlobalCacheEngine` 在 `cache_engine.py` 中记录，通过 `FlexKVMetricsCollector` 收集。

| 指标名称 | 类型 | 标签 | 描述 |
|---|---|---|---|
| `flexkv_py_cache_hit_blocks_total` | Counter | `device` | 缓存命中的 blocks 总数 |
| `flexkv_py_cache_miss_blocks_total` | Counter | - | 缓存未命中的 blocks 总数（所有层级均未命中） |
| `flexkv_py_transfer_blocks_total` | Counter | `transfer_type`, `operation` | 传输的 blocks 总数 |
| `flexkv_py_transfer_ops_total` | Counter | `transfer_type`, `operation` | 传输操作次数 |
| `flexkv_py_mempool_total_blocks` | Gauge | `device` | 内存池总 blocks |
| `flexkv_py_mempool_free_blocks` | Gauge | `device` | 内存池空闲 blocks |
| `flexkv_py_evicted_blocks_total` | Counter | `device` | 驱逐的 blocks 总数 |
| `flexkv_py_allocated_blocks_total` | Counter | `device` | 分配的 blocks 总数 |
| `flexkv_py_allocation_failures_total` | Counter | `mode` | 资源分配失败次数 |

---

### 2.2 C++ 运行时指标 (`flexkv_cpp_*`)

C++ 指标由 `MetricsManager` 单例管理，主要在 RadixTree 缓存操作和数据传输中埋点。

| 指标名称 | 类型 | 标签 | 描述 |
|---|---|---|---|
| `flexkv_cpp_transfer_ops_total` | Counter | `type`, `direction` | C++ 层数据传输操作次数 |
| `flexkv_cpp_transfer_bytes_total` | Counter | `type`, `direction` | C++ 层数据传输字节总数 |
| `flexkv_cpp_cache_ops_total` | Counter | `operation` | RadixTree 缓存操作次数 |
| `flexkv_cpp_cache_blocks_total` | Counter | `operation` | RadixTree 缓存操作涉及的 blocks 数 |

---

### 2.3 跨实例复用指标 (`flexkv_py_dist_reuse_*`)

这组指标观测**分布式 KV-cache 复用**路径（master / peer 实例通过
Redis-meta 协调 + Mooncake P2P CPU 拉取），是 lease 安全机制的核心信号 ——
用于保证跨实例读取不会与 master 端 evict 发生竞争。这 5 个指标与现有
`flexkv_py_*` 共同暴露在 Python 指标端点（`/metrics`，端口
`FLEXKV_PY_METRICS_PORT`）。

| 指标名称 | 类型 | 标签 | 严重级别 | 描述 |
|---|---|---|---|---|
| `flexkv_py_dist_reuse_lease_meta_nullptr_total` | Counter | `device` | **CRITICAL** | Master 端因池容量超过 `swap_block_threshold` 而以 `lease_meta=nullptr` 插入的 block 数。这类 block 立即可被 evict，破坏了 lease 保护的 P2P 安全性 — **生产环境出现任何正值都应立即告警 oncall**。 |
| `flexkv_py_dist_reuse_about_to_evict_total` | Counter | `device` | **WARN** | 进入 *fresh*-branch evict 路径的 block 数（lease 仍有效，但池压力强行需要这个槽位）。与 `flexkv_py_evicted_blocks_total` 一起计算 `fresh / expired` evict 比值 — 持续 > 10 表示 master 在与 evict 压力对抗，lease 安全余量正在收缩。 |
| `flexkv_py_dist_reuse_peer_mooncake_read_seconds` | Histogram | — | **OPS** | Peer 端 `mooncake.transfer_sync_read` 调用耗时（P2P CPU 拉取 master 实例数据）。Bucket：`0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0` 秒。**P99 > 500ms** 表示 lease 余量正逼近耗尽。 |
| `flexkv_py_dist_reuse_peer_mooncake_read_failures_total` | Counter | `reason` | **CRITICAL** | Peer 端 mooncake 读取失败计数。`reason` 取值：`mooncake_error`（非零返回）、`zero_byte_transfer`（2026-05-14 修复的 P0 bug 症状：ret==0 但实际未传输字节）、`node_meta_missing`（peer 节点发现失败）、`timeout`。**持续失败率 > 0.1% 应告警 oncall**。 |
| `flexkv_py_dist_reuse_peer_mooncake_read_success_total` | Counter | — | — | Peer 端 mooncake 读取成功计数，作为 `_failures_total` 计算失败率时的分母。 |

**埋点接入状态（截至本 commit）：**

| 指标 | 生产侧调用位置 | 说明 |
|---|---|---|
| `flexkv_py_dist_reuse_peer_mooncake_read_seconds` | `flexkv/transfer/worker.py`（`PEER2CPUTransferWorker`） | 端到端已接入。 |
| `flexkv_py_dist_reuse_peer_mooncake_read_failures_total` | 同上 | 端到端已接入。 |
| `flexkv_py_dist_reuse_peer_mooncake_read_success_total` | 同上 | 端到端已接入。 |
| `flexkv_py_dist_reuse_lease_meta_nullptr_total` | *（collector 钩子已就绪，业务侧 trigger 待补）* | Python 辅助方法 `record_dist_reuse_lease_nullptr` 已在 `flexkv/metrics/collector.py` 就绪；C++ master 端 evict 路径的对应触发逻辑见内部计划文档 `docs/dist_reuse/METRICS_dist_reuse.md`。该 trigger 落地前指标值会一直为 `0`。 |
| `flexkv_py_dist_reuse_about_to_evict_total` | *（collector 钩子已就绪，业务侧 trigger 待补）* | 状态同上（对应方法 `record_dist_reuse_about_to_evict`）。 |

> 上述两个未接入指标也会照常暴露（值恒为 `0`），便于 Prometheus 抓取配置和
> Grafana 面板提前接入；C++ trigger 落地后会自动开始上报非零值。
> **不要把这两个指标的 `0` 值理解为「系统健康」** — 它们的 `0` 当前表示
> 「埋点尚未接入」。

---

## 三、监控组件部署说明

### 3.1 目录结构

```
FlexKV/monitoring/
├── docker-compose.yml         # Prometheus + Grafana container orchestration
├── prometheus.yml             # Prometheus scrape configuration
└── grafana/
    ├── dashboards/
    │   └── flexkv-demo.json   # Grafana pre-built dashboard
    └── provisioning/
        ├── dashboards/
        │   └── dashboards.yml # Dashboard auto-load configuration
        └── datasources/
            └── prometheus.yml # Datasource auto-configuration
```

### 3.2 快速部署

```bash
# 0. Install Python dependency
pip3 install prometheus_client

# 1. Start FlexKV application with monitoring enabled
export FLEXKV_ENABLE_METRICS=1
python your_flexkv_app.py

# 2. Start Prometheus + Grafana services
cd <path-to-FlexKV>/monitoring
docker compose up -d

# 3. Stop Prometheus + Grafana services
cd <path-to-FlexKV>/monitoring
docker compose stop

# 4. Fully clean up Prometheus + Grafana services
cd <path-to-FlexKV>/monitoring
docker compose down -v
```

### 3.3 访问服务

| 服务 | 地址 | 说明 |
|---|---|---|
| Python Metrics | `http://localhost:8080/metrics` | Python 运行时指标端点 |
| C++ Metrics | `http://localhost:8081/metrics` | C++ 运行时指标端点 |
| Prometheus | `http://localhost:9090` | 指标查询界面 |
| Grafana | `http://localhost:3000` | 可视化仪表板 |

**快速验证指标端点：**

```bash
# Verify Python metrics endpoint
curl -s http://localhost:8080/metrics | grep flexkv_py_

# Verify C++ metrics endpoint
curl -s http://localhost:8081/metrics | grep flexkv_cpp_

# Verify the new cross-instance reuse metrics (must be present even if value=0)
curl -s http://localhost:8080/metrics | grep flexkv_py_dist_reuse_
```

### 3.5 多进程抓取说明

FlexKV 的 Python 控制面运行在多个进程中（sglang scheduler、每个
`WorkerKey` 一个 transfer-engine 子进程、后台 KVManager）。
`prometheus_client` 在 `PROMETHEUS_MULTIPROC_DIR` 中按进程写入采样文件，
HTTP server 在每次抓取时通过 `MultiProcessCollector` 聚合这些采样。

* 若未显式设置 `PROMETHEUS_MULTIPROC_DIR`，collector 会自动初始化一个
  可写临时目录，因此单进程场景（如单测）无需额外配置。
* 长期运行的部署建议把 `PROMETHEUS_MULTIPROC_DIR` 指向 tmpfs 路径
  （如 `/dev/shm/flexkv_prom`），既避免磁盘磨损，也能保证容器重启时
  目录被清理。
* HTTP server 会输出 Prometheus 标准的
  `Content-Type: text/plain; version=0.0.4` 响应头，Prometheus 端只需配置
  `<host>:<FLEXKV_PY_METRICS_PORT>/metrics` 抓取地址即可，不需要额外
  scrape 参数。

### 3.6 dist_reuse 指标推荐 PromQL 告警

以下规则可作为 Prometheus / Alertmanager 的起点配置：

```yaml
groups:
- name: flexkv_dist_reuse
  rules:
  # CRITICAL — 任何 nullptr lease 插入都意味着安全保证已失效。
  - alert: FlexKVDistReuseLeaseMetaNullptr
    expr: increase(flexkv_py_dist_reuse_lease_meta_nullptr_total[5m]) > 0
    for: 1m
    labels: { severity: critical }
    annotations:
      summary: "FlexKV master inserted lease_meta=nullptr blocks (device={{ $labels.device }})"

  # CRITICAL — peer mooncake_read 失败率 > 0.1% 持续 5 分钟。
  - alert: FlexKVDistReusePeerReadFailureRate
    expr: |
      sum by (reason) (rate(flexkv_py_dist_reuse_peer_mooncake_read_failures_total[5m]))
        /
      sum (rate(flexkv_py_dist_reuse_peer_mooncake_read_success_total[5m]) +
           rate(flexkv_py_dist_reuse_peer_mooncake_read_failures_total[5m]))
        > 0.001
    for: 5m
    labels: { severity: critical }
    annotations:
      summary: "FlexKV peer mooncake_read failure rate > 0.1% (reason={{ $labels.reason }})"

  # OPS — peer mooncake_read P99 > 500ms。
  - alert: FlexKVDistReusePeerReadP99High
    expr: histogram_quantile(0.99, sum by (le) (rate(flexkv_py_dist_reuse_peer_mooncake_read_seconds_bucket[5m]))) > 0.5
    for: 5m
    labels: { severity: warning }
    annotations:
      summary: "FlexKV peer mooncake_read P99 > 500ms (lease margin shrinking)"

  # WARN — fresh / expired evict 比值持续 > 10。
  - alert: FlexKVDistReuseEvictPressure
    expr: |
      sum by (device) (rate(flexkv_py_dist_reuse_about_to_evict_total[5m]))
        /
      clamp_min(sum by (device) (rate(flexkv_py_evicted_blocks_total[5m])), 1e-9)
        > 10
    for: 10m
    labels: { severity: warning }
    annotations:
      summary: "FlexKV master eviction is fighting lease pressure (device={{ $labels.device }})"
```

### 3.4 访问 Grafana 仪表板

1. 打开浏览器访问 `http://localhost:3000`
2. 使用默认账号登录：用户名 `admin`，密码 `admin`
3. 进入 **Dashboards → FlexKV Demo** 查看预置仪表板

**预置仪表板包含以下典型面板：**

| 面板 | 说明 |
|---|---|
| Cache Hit/Miss Rate | Python 层缓存命中/未命中速率 |
| Memory Pool Blocks | Python 层内存池块数统计 |
| C++ Cache Operations Rate | C++ 层缓存操作速率 |
| C++ Transfer Throughput | C++ 层数据传输吞吐量 |

> 用户可以按需创建自定义面板并添加和配置 PromQL 查询语句。
