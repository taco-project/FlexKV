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
| `FLEXKV_TRANSFER_TRACE` | `0` | Log per-transfer timing (`[XFER]` lines). Logs only; does not affect metrics |

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
| `flexkv_py_transfer_bytes_total` | Counter | `transfer_type`, `operation` | Total bytes transferred |
| `flexkv_py_mempool_total_blocks` | Gauge | `device` | Total blocks in memory pool |
| `flexkv_py_mempool_free_blocks` | Gauge | `device` | Free blocks in memory pool |
| `flexkv_py_evicted_blocks_total` | Counter | `device` | Total number of evicted blocks |
| `flexkv_py_allocated_blocks_total` | Counter | `device` | Total number of allocated blocks |
| `flexkv_py_allocation_failures_total` | Counter | `mode` | Number of allocation failures |
| `flexkv_py_transfer_wait_duration_seconds` | Histogram | `transfer_type` | Time the op waited on the worker before the transfer (received → started) |
| `flexkv_py_transfer_xfer_duration_seconds` | Histogram | `transfer_type` | Time in the transfer itself (started → returned) |
| `flexkv_py_transfer_e2e_duration_seconds` | Histogram | `transfer_type` | End-to-end time (submit → completion observed), including both |

> The three duration metrics are bucketed 0.5ms – 30s and labelled by
> `transfer_type`. Timing follows `FLEXKV_ENABLE_METRICS` and covers successful
> transfers only.

---

The collector also exposes `flexkv_py_joint_prefetch_outcomes_total` (label
`outcome`) and `flexkv_py_joint_prefetch_swa_slot_alloc_failed_total` for
Mooncake Full+SWA prefetch. These describe joint-prefetch outcomes, not all
requests or every prefetch failure.

#### Chunked prefetch metrics (`flexkv_prefetch_*`)

When chunked prefetch and metrics are enabled, the Python exporter records:

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_prefetch_sessions_total` | Counter | `policy`, `reason`, `outcome` | Sessions counted once after all claimed chunks drain |
| `flexkv_prefetch_loaded_tokens_total` | Counter | `policy` | Published L3 tokens in the reusable prefix at terminal while its lease is valid |
| `flexkv_prefetch_elapsed_seconds` | Histogram | `policy`, `reason` | Time from admission to terminal |
| `flexkv_prefetch_drain_seconds` | Histogram | `policy`, `reason` | Time from seal to terminal |

For example, the p95 drain time per policy in seconds is:

```promql
histogram_quantile(0.95, sum by (le, policy) (rate(flexkv_prefetch_drain_seconds_bucket[5m])))
```

An early release, reset or shutdown invalidates the result lease; such a session
adds no loaded tokens. Unknown custom policies and stop reasons are labelled
`other`. These counters do not measure H2D completion or model consumption, and
the legacy transfer and joint-prefetch counters do not cover all chunked
prefetch completions. Reserved/pinned bytes, inflight/state and runtime health
gauges are not yet provided by these metrics.


---

### 2.2 C++ Runtime Metrics (`flexkv_cpp_*`)

C++ metrics are managed by the `MetricsManager` singleton, primarily instrumented in RadixTree cache operations and data transfers.

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_cpp_cache_ops_total` | Counter | `operation` | RadixTree cache operation count |
| `flexkv_cpp_cache_blocks_total` | Counter | `operation` | Blocks involved in RadixTree cache operations |

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
```

### 3.4 Accessing Grafana Dashboards

1. Open your browser and navigate to `http://localhost:3000`
2. Log in with default credentials: username `admin`, password `admin`
3. Go to **Dashboards → FlexKV Demo** to view the pre-built dashboard

**Pre-built dashboard panels:**

| Section | Panel | Description |
|---|---|---|
| Python Runtime Metrics | Cache Hit/Miss Rate | Cache hit/miss rate |
| Python Runtime Metrics | Memory Pool Blocks | Memory pool block statistics |
| Python Runtime Metrics | Transfer Throughput | Data transfer throughput |
| C++ Runtime Metrics | Cache Operations Rate | Cache operation rate |
| C++ Runtime Metrics | Cache Blocks Rate | Cache blocks operation rate |

> Users can create custom panels and configure PromQL queries as needed.
