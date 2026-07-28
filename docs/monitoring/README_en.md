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
| `PROMETHEUS_MULTIPROC_DIR` | - | Optional. Directory for `prometheus_client` multiprocess aggregation. Set it (before process start) to export metrics recorded in spawned subprocesses — currently the layerwise startup/submit metrics recorded in the TransferManager process. Metrics recorded in the main process (cache/SWA/transfer/pool) are exported without it. |

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

#### DSV4 Metrics (SWA / Cache Query / Layerwise)

Metrics for DeepSeek-V4-style deployments: sliding-window attention (SWA) cache, per-tier match readiness, and layerwise transfer startup/dispatch. Recorded by `GlobalCacheEngine` (main process) and `TransferEngine` (TransferManager subprocess; layerwise metrics require `PROMETHEUS_MULTIPROC_DIR` to be aggregated into the HTTP endpoint).

**SWA:**

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_py_swa_query_total` | Counter | `result` (`hit`/`miss`) | SWA-aware GET queries by result |
| `flexkv_py_swa_hit_blocks_total` | Counter | - | Blocks served from an SWA read source on hits |
| `flexkv_py_swa_slot_used` | Gauge | `device` | SWA host-pool slots currently in use, per tier |
| `flexkv_py_swa_slot_total` | Gauge | `device` | SWA host-pool total slots, per tier |
| `flexkv_py_swa_slot_alloc_failed_total` | Counter | `device` | SWA slot allocation failures (pool still full after one eviction retry) |
| `flexkv_py_swa_evicted_slots_total` | Counter | `device`, `reason` (`pool_full`/`cascade`) | SWA slots freed: `pool_full` = SWA-LRU eviction under pool pressure; `cascade` = detached by radix-tree structural changes |
| `flexkv_py_swa_transfer_bytes_total` | Counter | `operation` | Bytes transferred by SWA (`is_swa`) ops after completion |

**Cache query / readiness:**

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_py_cache_query_total` | Counter | `result` (`full`/`partial`/`miss`) | GET queries by coverage of the queried window |
| `flexkv_py_cache_match_blocks_total` | Counter | `device`, `ready` (`true`/`false`) | Blocks matched in the radix tree per tier, split by data readiness. `ready=true` is immediately reusable ("effective reuse"); `ready=false` is matched but still in flight |

**Layerwise:**

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `flexkv_py_layerwise_workers_ready` | Gauge | - | Layerwise workers that finished initialization (eventfd handshake) |
| `flexkv_py_layerwise_workers_expected` | Gauge | - | Layerwise workers expected at startup |
| `flexkv_py_layerwise_submit_seconds` | Histogram | - | Time to fan out a LAYERWISE op to all sibling workers (successful fan-outs only) |
| `flexkv_py_layerwise_submit_total` | Counter | `status` (`ok`/`error`) | LAYERWISE op fan-out count by status |

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
