# FlexKV Prometheus Metrics Documentation

FlexKV integrates a [Prometheus](https://prometheus.io/)-based runtime metrics monitoring framework, covering critical paths in both the Python and C++ layers. The framework is embedded in the FlexKV runtime in a **zero-intrusion** manner — users simply set the environment variable `FLEXKV_ENABLE_METRICS=1` to automatically collect core metrics such as cache hits, memory pool status, and data transfers during application runtime, exposing them via standard HTTP endpoints for Prometheus scraping and Grafana visualization.

---

## 1. Configuration

### 1.1 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLEXKV_ENABLE_METRICS` | `0` | Enable metrics collection (set to `1` to enable, disabled by default) |
| `FLEXKV_PY_METRICS_PORT` | `8080` | Python metrics HTTP server port |
| `FLEXKV_CPP_METRICS_PORT` | `8081` | C++ metrics HTTP server port |
| `PROMETHEUS_MULTIPROC_DIR` | *(auto)* | Directory for `prometheus_client` per-process sample files. Required when FlexKV runs across multiple Python processes (sglang TP/PP workers, KVManager subprocess, transfer workers). The collector auto-bootstraps a writable temp directory if unset; explicitly set it to a tmpfs path to override. |

### 1.2 Configuration

```bash
# Enable FlexKV metrics collection
export FLEXKV_ENABLE_METRICS=1

# Custom ports (optional)
export FLEXKV_PY_METRICS_PORT=8080
export FLEXKV_CPP_METRICS_PORT=8081
```

---

## 2. Metrics Reference

### 2.1 Python Runtime Metrics (`flexkv_py_*`)

Python metrics are recorded by `GlobalCacheEngine` in `cache_engine.py` and collected via `FlexKVMetricsCollector`.

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_py_cache_hit_blocks_total` | Counter | `device` | Total number of cache-hit blocks |
| `flexkv_py_cache_miss_blocks_total` | Counter | - | Total number of cache-miss blocks (missed at all levels) |
| `flexkv_py_transfer_blocks_total` | Counter | `transfer_type`, `operation` | Total number of transferred blocks |
| `flexkv_py_transfer_ops_total` | Counter | `transfer_type`, `operation` | Number of transfer operations |
| `flexkv_py_mempool_total_blocks` | Gauge | `device` | Total blocks in memory pool |
| `flexkv_py_mempool_free_blocks` | Gauge | `device` | Free blocks in memory pool |
| `flexkv_py_evicted_blocks_total` | Counter | `device` | Total number of evicted blocks |
| `flexkv_py_allocated_blocks_total` | Counter | `device` | Total number of allocated blocks |
| `flexkv_py_allocation_failures_total` | Counter | `mode` | Number of allocation failures |

---

### 2.2 C++ Runtime Metrics (`flexkv_cpp_*`)

C++ metrics are managed by the `MetricsManager` singleton, primarily instrumented in RadixTree cache operations and data transfers.

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_cpp_transfer_ops_total` | Counter | `type`, `direction` | C++ layer data transfer operation count |
| `flexkv_cpp_transfer_bytes_total` | Counter | `type`, `direction` | C++ layer total transferred bytes |
| `flexkv_cpp_cache_ops_total` | Counter | `operation` | RadixTree cache operation count |
| `flexkv_cpp_cache_blocks_total` | Counter | `operation` | Blocks involved in RadixTree cache operations |

---

### 2.3 Cross-instance Reuse Metrics (`flexkv_py_dist_reuse_*`)

These metrics observe the **distributed KV-cache reuse** path (master / peer
instances coordinated through Redis-meta + Mooncake P2P CPU pulls). They are
the primary signals for the lease-based safety guarantee that protects
cross-instance reads from racing master-side eviction.  The 5 metrics live
alongside the existing `flexkv_py_*` set on the Python metrics endpoint
(`/metrics`, port `FLEXKV_PY_METRICS_PORT`).

| Metric Name | Type | Labels | Severity | Description |
|---|---|---|---|---|
| `flexkv_py_dist_reuse_lease_meta_nullptr_total` | Counter | `device` | **CRITICAL** | Master-side blocks inserted with `lease_meta=nullptr` because the pool exceeded `swap_block_threshold`. Such blocks become evictable immediately and break the lease-based P2P safety guarantee — **any positive value in production should page oncall**. |
| `flexkv_py_dist_reuse_about_to_evict_total` | Counter | `device` | **WARN** | Blocks marked `ABOUT_TO_EVICT` (the *fresh*-branch evict path: lease still valid but the slot was needed anyway). Used together with `flexkv_py_evicted_blocks_total` to compute the `fresh / expired` evict ratio — sustained ratio > 10 means master is fighting eviction pressure and the lease-based safety margin is shrinking. |
| `flexkv_py_dist_reuse_peer_mooncake_read_seconds` | Histogram | — | **OPS** | Latency of peer-side `mooncake.transfer_sync_read` calls (P2P CPU pull from a master instance). Buckets: `0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0` seconds. **P99 > 500 ms** means the remaining lease window is shrinking toward exhaustion. |
| `flexkv_py_dist_reuse_peer_mooncake_read_failures_total` | Counter | `reason` | **CRITICAL** | Peer-side mooncake read failures. The `reason` label is one of `mooncake_error` (non-zero ret), `zero_byte_transfer` (the P0-bug symptom from 2026-05-14: ret==0 but no bytes moved), `node_meta_missing` (peer discovery breakdown), `timeout`. **Sustained failure rate > 0.1% warrants oncall**. |
| `flexkv_py_dist_reuse_peer_mooncake_read_success_total` | Counter | — | — | Peer-side mooncake read successes. Used as the denominator when computing the failure rate from `_failures_total`. |

**Instrumentation status (as of this commit):**

| Metric | Production call site | Notes |
|---|---|---|
| `flexkv_py_dist_reuse_peer_mooncake_read_seconds` | `flexkv/transfer/worker.py` (`PEER2CPUTransferWorker`) | Wired end-to-end. |
| `flexkv_py_dist_reuse_peer_mooncake_read_failures_total` | same as above | Wired end-to-end. |
| `flexkv_py_dist_reuse_peer_mooncake_read_success_total` | same as above | Wired end-to-end. |
| `flexkv_py_dist_reuse_lease_meta_nullptr_total` | *(collector hook ready, business-side trigger pending)* | The Python helper `record_dist_reuse_lease_nullptr` is ready in `flexkv/metrics/collector.py`; the C++ master-side eviction path that should call it is tracked in `docs/dist_reuse/METRICS_dist_reuse.md`. The metric will read `0` until that hook lands. |
| `flexkv_py_dist_reuse_about_to_evict_total` | *(collector hook ready, business-side trigger pending)* | Same status as above (`record_dist_reuse_about_to_evict`). |

> The two pending metrics are intentionally exposed (with value `0`) so
> Prometheus scrape configs and Grafana panels can be wired ahead of
> time; they will start emitting non-zero values automatically once the
> C++ trigger lands. **Do not interpret a `0` as "system healthy"** — a
> `0` value on these two specifically means "not yet instrumented".

---

## 3. Monitoring Stack Deployment

### 3.1 Directory Structure

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

### 3.2 Quick Deploy

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

### 3.3 Service Access

| Service | URL | Description |
|---|---|---|
| Python Metrics | `http://localhost:8080/metrics` | Python runtime metrics endpoint |
| C++ Metrics | `http://localhost:8081/metrics` | C++ runtime metrics endpoint |
| Prometheus | `http://localhost:9090` | Metrics query interface |
| Grafana | `http://localhost:3000` | Visualization dashboards |

**Quick endpoint verification:**

```bash
# Verify Python metrics endpoint
curl -s http://localhost:8080/metrics | grep flexkv_py_

# Verify C++ metrics endpoint
curl -s http://localhost:8081/metrics | grep flexkv_cpp_

# Verify the new cross-instance reuse metrics (must be present even if value=0)
curl -s http://localhost:8080/metrics | grep flexkv_py_dist_reuse_
```

### 3.5 Multiprocess Scrape Notes

FlexKV runs the Python control plane across several processes (the
sglang scheduler, one transfer-engine subprocess per `WorkerKey`, and a
background KVManager).  `prometheus_client` writes one sample file per
process into `PROMETHEUS_MULTIPROC_DIR`; the metrics HTTP server then
aggregates them on every scrape via `MultiProcessCollector`.

* The collector auto-bootstraps a writable temp dir if
  `PROMETHEUS_MULTIPROC_DIR` is unset, so a single-process workflow
  (e.g. unit tests) needs no setup.
* For long-running deployments, point `PROMETHEUS_MULTIPROC_DIR` at a
  tmpfs path (e.g. `/dev/shm/flexkv_prom`) to avoid disk wear and to
  ensure the directory is wiped on container restart.
* The HTTP server writes the standard `Content-Type:
  text/plain; version=0.0.4` header expected by Prometheus; no extra
  scrape config is needed beyond pointing Prometheus at
  `<host>:<FLEXKV_PY_METRICS_PORT>/metrics`.

### 3.6 Recommended PromQL alerts for the dist_reuse metrics

Use these as a starting point in Prometheus / Alertmanager:

```yaml
groups:
- name: flexkv_dist_reuse
  rules:
  # CRITICAL — any nullptr lease insert means the safety guarantee is broken.
  - alert: FlexKVDistReuseLeaseMetaNullptr
    expr: increase(flexkv_py_dist_reuse_lease_meta_nullptr_total[5m]) > 0
    for: 1m
    labels: { severity: critical }
    annotations:
      summary: "FlexKV master inserted lease_meta=nullptr blocks (device={{ $labels.device }})"

  # CRITICAL — peer mooncake_read failure rate > 0.1% sustained for 5m.
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

  # OPS — peer mooncake_read P99 > 500ms.
  - alert: FlexKVDistReusePeerReadP99High
    expr: histogram_quantile(0.99, sum by (le) (rate(flexkv_py_dist_reuse_peer_mooncake_read_seconds_bucket[5m]))) > 0.5
    for: 5m
    labels: { severity: warning }
    annotations:
      summary: "FlexKV peer mooncake_read P99 > 500ms (lease margin shrinking)"

  # WARN — fresh / expired evict ratio > 10 sustained.
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

### 3.4 Accessing Grafana Dashboards

1. Open your browser and navigate to `http://localhost:3000`
2. Log in with default credentials: username `admin`, password `admin`
3. Go to **Dashboards → FlexKV Demo** to view the pre-built dashboard

**Pre-built dashboard panels:**

| Panel | Description |
|---|---|
| Cache Hit/Miss Rate | Python layer cache hit/miss rate |
| Memory Pool Blocks | Python layer memory pool block statistics |
| C++ Cache Operations Rate | C++ layer cache operation rate |
| C++ Transfer Throughput | C++ layer data transfer throughput |

> Users can create custom panels and configure PromQL queries as needed.
