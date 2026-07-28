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
| `PROMETHEUS_MULTIPROC_DIR` | - | 可选。`prometheus_client` 多进程聚合目录。在进程启动前设置后，可汇聚 spawn 子进程中记录的指标（目前是 TransferManager 进程中的 layerwise 启动/提交指标）。主进程指标（缓存/SWA/传输/内存池）不依赖此变量 |

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
| `flexkv_py_transfer_bytes_total` | Counter | `transfer_type`, `operation` | 传输字节总数 |
| `flexkv_py_mempool_total_blocks` | Gauge | `device` | 内存池总 blocks |
| `flexkv_py_mempool_free_blocks` | Gauge | `device` | 内存池空闲 blocks |
| `flexkv_py_evicted_blocks_total` | Counter | `device` | 驱逐的 blocks 总数 |
| `flexkv_py_allocated_blocks_total` | Counter | `device` | 分配的 blocks 总数 |
| `flexkv_py_allocation_failures_total` | Counter | `mode` | 资源分配失败次数 |

#### DSV4 指标（SWA / 缓存查询 / Layerwise）

面向 DeepSeek-V4 形态部署的指标：滑动窗口注意力（SWA）缓存、分层匹配就绪状态、layerwise 传输启动/派发。由 `GlobalCacheEngine`（主进程）和 `TransferEngine`（TransferManager 子进程；layerwise 指标需设置 `PROMETHEUS_MULTIPROC_DIR` 才会汇聚到 HTTP 端点）记录。

**SWA：**

| 指标名称 | 类型 | 标签 | 描述 |
|---|---|---|---|
| `flexkv_py_swa_query_total` | Counter | `result`（`hit`/`miss`） | SWA 感知 GET 查询次数，按结果分类 |
| `flexkv_py_swa_hit_blocks_total` | Counter | - | SWA 读源命中服务的 blocks 总数 |
| `flexkv_py_swa_slot_used` | Gauge | `device` | SWA host-pool 已用 slot 数（按层级） |
| `flexkv_py_swa_slot_total` | Gauge | `device` | SWA host-pool 总 slot 数（按层级） |
| `flexkv_py_swa_slot_alloc_failed_total` | Counter | `device` | SWA slot 分配失败次数（淘汰重试后池仍满） |
| `flexkv_py_swa_evicted_slots_total` | Counter | `device`, `reason`（`pool_full`/`cascade`） | SWA slot 释放数：`pool_full` = 池满触发 SWA-LRU 淘汰；`cascade` = radix 树结构变化级联释放 |
| `flexkv_py_swa_transfer_bytes_total` | Counter | `operation` | SWA（`is_swa`）op 完成后统计的传输字节数 |

**缓存查询 / 就绪状态：**

| 指标名称 | 类型 | 标签 | 描述 |
|---|---|---|---|
| `flexkv_py_cache_query_total` | Counter | `result`（`full`/`partial`/`miss`） | GET 查询按查询窗口覆盖度分类的次数 |
| `flexkv_py_cache_match_blocks_total` | Counter | `device`, `ready`（`true`/`false`） | radix 树命中的 blocks 按层级与数据就绪状态拆分。`ready=true` 即可直接复用量（有效复用）；`ready=false` 为已匹配但仍在传输中 |

**Layerwise：**

| 指标名称 | 类型 | 标签 | 描述 |
|---|---|---|---|
| `flexkv_py_layerwise_workers_ready` | Gauge | - | 已完成初始化（eventfd 握手）的 layerwise worker 数 |
| `flexkv_py_layerwise_workers_expected` | Gauge | - | 启动时预期的 layerwise worker 数 |
| `flexkv_py_layerwise_submit_seconds` | Histogram | - | 将 LAYERWISE op 扇出到全部 sibling worker 的耗时（仅统计成功扇出） |
| `flexkv_py_layerwise_submit_total` | Counter | `status`（`ok`/`error`） | LAYERWISE op 扇出次数（按状态） |

---

### 2.2 C++ 运行时指标 (`flexkv_cpp_*`)

C++ 指标由 `MetricsManager` 单例管理，主要在 RadixTree 缓存操作和数据传输中埋点。

| 指标名称 | 类型 | 标签 | 描述 |
|---|---|---|---|
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

| 分区 | 面板 | 说明 |
|---|---|---|
| Python Runtime Metrics | Cache Hit/Miss Rate | 缓存命中/未命中速率 |
| Python Runtime Metrics | Memory Pool Blocks | 内存池块数统计 |
| Python Runtime Metrics | Transfer Throughput | 数据传输吞吐量 |
| C++ Runtime Metrics | Cache Operations Rate | 缓存操作速率 |
| C++ Runtime Metrics | Cache Blocks Rate | 缓存块操作速率 |

> 用户可以按需创建自定义面板并添加和配置 PromQL 查询语句。
