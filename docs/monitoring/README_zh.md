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
