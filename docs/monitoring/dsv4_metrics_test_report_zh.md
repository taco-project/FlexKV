# DSV4 指标功能测试记录

| 项目 | 内容 |
|---|---|
| 被测功能 | DSV4 可观测性指标（SWA / 缓存查询就绪拆分 / Layerwise），共 13 个 `flexkv_py_*` 新指标 |
| 代码分支 | `dsv4-metrics-on-main`（基于 `main` @ `3ecd83f`） |
| 提交 | `270ab1c` feat(metrics): add DSV4 metrics infrastructure<br>`6746948` feat(metrics): instrument DSV4 SWA/cache-query/layerwise hook points |
| 测试日期 | 2026-07-26 |
| 测试环境 | 8×NVIDIA H20（使用 0-3 卡），CUDA 13.0，Python 3.12（venv `/data/workspace/dsv4-flexkv`），模型 `/data1/models/deepseek-ai/DeepSeek-V4-Flash` |

## 结果汇总

| # | 测试项 | 方式 | 结果 |
|---|---|---|---|
| 1 | 新增指标单元测试 | `pytest tests/test_metrics_dsv4.py` | 13/13 通过 |
| 2 | 回归（metrics 关闭，默认） | 相关测试集 | 254 通过，5 失败（均为 main 自带上游测试漂移，与本改动无关，见末节） |
| 3 | 回归（metrics 开启） | 相关测试集 | 217/217 通过 |
| 4 | 多进程聚合冒烟 | spawn 子进程 + `MultiProcessCollector` | 聚合值正确（3+1=4） |
| 5 | 实机部署（sglang + DSV4-Flash tp4） | 真实推理请求 + `/metrics` 抓取 | 服务稳定，SWA/传输指标实时变化 |
| 6 | 引擎级命中路径 | kvtask 测试 + multiproc 聚合 | 命中类指标准确触发（`cache_query{full}`、`cache_match{ready=true}`、`swa_query{hit}`） |
| 7 | 关闭路径零副作用 | 单测 disabled 用例 + 关闭态全量回归 | 所有 `record_*` 为 no-op，行为与改造前一致 |
| 8 | 单进程端点输出 | 独立进程起 8080 + curl | 修复后全部 13 个新指标准确输出（见第 8 节） |

---

## 1. 单元测试

```
$ python -m pytest tests/test_metrics_dsv4.py -v

tests/test_metrics_dsv4.py::TestSWAMetrics::test_swa_query_and_hit_blocks PASSED [  7%]
tests/test_metrics_dsv4.py::TestSWAMetrics::test_swa_slot_stats_and_failures PASSED [ 15%]
tests/test_metrics_dsv4.py::TestSWAMetrics::test_swa_evicted_by_reason PASSED [ 23%]
tests/test_metrics_dsv4.py::TestSWAMetrics::test_swa_transfer_bytes PASSED [ 30%]
tests/test_metrics_dsv4.py::TestCacheEngineSWAHooks::test_slot_gauge_follows_alloc_and_free PASSED [ 38%]
tests/test_metrics_dsv4.py::TestCacheEngineSWAHooks::test_alloc_failure_counts_after_eviction_retry PASSED [ 46%]
tests/test_metrics_dsv4.py::TestCacheEngineSWAHooks::test_evict_and_cascade_counted PASSED [ 53%]
tests/test_metrics_dsv4.py::TestCacheQueryMetrics::test_query_results PASSED [ 61%]
tests/test_metrics_dsv4.py::TestCacheQueryMetrics::test_match_blocks_ready_split PASSED [ 69%]
tests/test_metrics_dsv4.py::TestLayerwiseMetrics::test_workers_progress PASSED [ 76%]
tests/test_metrics_dsv4.py::TestLayerwiseMetrics::test_submit_count_and_latency PASSED [ 84%]
tests/test_metrics_dsv4.py::TestDisabledPath::test_all_record_methods_noop PASSED [ 92%]
tests/test_metrics_dsv4.py::TestDisabledPath::test_engine_hooks_noop PASSED [100%]

============================== 13 passed in 0.13s ==============================
```

覆盖：SWA 查询/命中块数、slot 水位 gauge、slot 分配失败、按原因淘汰（pool_full/cascade）、SWA 传输字节、缓存查询三态、ready/not-ready 拆分、layerwise 进度与提交时延、关闭态全 no-op（含引擎挂点）。

## 2. 回归测试（metrics 关闭，默认路径）

```
$ python -m pytest tests/test_metrics_dsv4.py tests/test_cache_engine.py \
    tests/test_kvtask_lifecycle.py tests/test_batch_kvtask.py \
    tests/test_swa_control_plane.py tests/test_swa_peer_op.py \
    tests/test_swa_state_sidecars.py tests/test_config_swa_multi_group.py -q

FAILED tests/test_swa_state_sidecars.py::test_swa_multi_layer_false_keeps_sidecar_h2d_as_predecessor
FAILED tests/test_config_swa_multi_group.py::test_swa_multi_layer_defaults_to_enabled
FAILED tests/test_config_swa_multi_group.py::test_swa_multi_layer_env_override[0-False]
FAILED tests/test_config_swa_multi_group.py::test_swa_multi_layer_env_override[1-True]
FAILED tests/test_config_swa_multi_group.py::test_swa_multi_layer_rejects_non_boolean_config_value
5 failed, 254 passed in 10.51s
```

5 个失败均为 **main 分支自带的上游测试漂移**：用例引用已不存在的 `swa_multi_layer` 配置属性（`UserConfig` 现仅有 `swa_multi_group`）。在未应用本改动的干净 main 上同样失败，与本 feature 无关。

## 3. 回归测试（metrics 开启）

```
$ FLEXKV_ENABLE_METRICS=1 python -m pytest tests/test_metrics_dsv4.py \
    tests/test_cache_engine.py tests/test_swa_control_plane.py -q

217 passed in 2.66s
```

## 4. 多进程聚合冒烟

模拟生产形态：spawn 子进程（worker 角色，不绑端口）记录 3 次计数，主进程记录 1 次，`MultiProcessCollector` 聚合：

```
$ PROMETHEUS_MULTIPROC_DIR=/tmp/prom_test FLEXKV_ENABLE_METRICS=1 python test_multiproc.py

[FLEXKV] INFO [collector.py:263] Prometheus metrics collector initialized   # 子进程
[FLEXKV] INFO [collector.py:263] Prometheus metrics collector initialized   # 主进程
aggregated swa_query hit = 4.0
multiprocess aggregation OK
```

## 5. 实机部署验证

启动命令（关键点：`--moe-runner-backend marlin` 必需，当前 sglang 的 `auto` 不解析，否则 fp4 权重落到 fp8 triton kernel 报 `Hidden size mismatch`）：

```
FLEXKV_ENABLE_METRICS=1 PROMETHEUS_MULTIPROC_DIR=/tmp/flexkv_prom \
python -m sglang.launch_server \
  --model-path /data1/models/deepseek-ai/DeepSeek-V4-Flash \
  --tp-size 4 --trust-remote-code --page-size 256 \
  --mem-fraction-static 0.9 --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 4 --chunked-prefill-size 8192 \
  --disable-cuda-graph --moe-runner-backend marlin \
  --enable-flexkv --flexkv-config-file /data/flexkv-dsv4-test/flexkv_config.yaml
```

启动日志确认：

```
[FlexKV PyMetrics] Initialized successfully, exposing metrics at http://127.0.0.1:8080/metrics
The server is fired up and ready to roll!
```

发送 2 次相同长 prompt 请求（均 200 正常返回；上一版分支无 keepalive 修复时首请求即 `tp_group_transfer failed: illegal memory access`，main 已修复）。

请求后 `curl -s http://127.0.0.1:8080/metrics` 抓取（非零项）：

```
flexkv_py_swa_query_total{result="miss"} 2.0
flexkv_py_swa_slot_total{device="cpu"} 2048.0
flexkv_py_swa_slot_used{device="cpu"} 2.0
flexkv_py_swa_transfer_bytes_total{operation="put"} 1.59616e+07
flexkv_py_transfer_ops_total{operation="put",transfer_type="D2H"} 2.0
flexkv_py_transfer_blocks_total{operation="put",transfer_type="D2H"} 3.0
flexkv_py_transfer_bytes_total{operation="put",transfer_type="D2H"} 2.39424e+07
```

说明：SWA 查询 miss（首次无缓存）、slot 池 2/2048 使用中、SWA PUT 约 16MB、D2H 下沉 2 次/3 块/约 24MB，指标与流量行为一致。

## 6. 引擎级命中路径验证

实机洪泛难以可靠挤出 GPU 前缀（DSV4 hybrid 池 full-KV 仅 10%），改用 kvtask 级测试驱动真实引擎 GET/PUT 往返：

```
$ FLEXKV_ENABLE_METRICS=1 PROMETHEUS_MULTIPROC_DIR=/tmp/prom_rt \
  python -m pytest tests/test_batch_kvtask.py tests/test_kvtask_lifecycle.py -q

22 passed in 6.86s
```

`MultiProcessCollector` 聚合读取（非零项）：

```
flexkv_py_cache_query_total {'result': 'full'} = 7.0
flexkv_py_cache_match_blocks_total {'device': 'cpu', 'ready': 'true'} = 20.0
flexkv_py_cache_hit_blocks_total {'device': 'cpu'} = 20.0
flexkv_py_swa_query_total {'result': 'hit'} = 3.0
flexkv_py_swa_hit_blocks_total {} = 12.0
flexkv_py_swa_slot_used {'device': 'cpu'} = 1.0
flexkv_py_swa_slot_total {'device': 'cpu'} = 256.0
flexkv_py_transfer_ops_total {'operation': 'get', 'transfer_type': 'H2D'} = 1.0
flexkv_py_transfer_ops_total {'operation': 'unknown', 'transfer_type': 'D2H'} = 1.0
flexkv_py_transfer_bytes_total {'operation': 'get', 'transfer_type': 'H2D'} = 8388608.0
flexkv_py_transfer_bytes_total {'operation': 'unknown', 'transfer_type': 'D2H'} = 8388608.0
flexkv_py_mempool_total_blocks {'device': 'cpu'} = 4096.0
flexkv_py_mempool_free_blocks {'device': 'cpu'} = 4092.0
flexkv_py_allocated_blocks_total {'device': 'cpu'} = 16.0
```

一致性校验：新指标 `cache_match_blocks_total{ready=true}`（20）与存量指标 `cache_hit_blocks_total{cpu}`（20）数值一致，"有效复用量"口径正确；H2D(get) 与 D2H(put) 字节数一致（8388608 B）。

## 7. 关闭路径零副作用

- 单测 `TestDisabledPath`：metrics 关闭时全部 `record_*`（含 Histogram `observe`）为 no-op 且不抛异常；
- 关闭态全量回归（见第 2 节）与改造前基线一致。

## 8. 单进程端点输出（含一次实测暴露问题的修复记录）

验证中单进程模式（未设 `PROMETHEUS_MULTIPROC_DIR`）曾暴露问题：8080 端点无任何 `flexkv_py_*` 输出。根因：collector 支持 registry 注入的实现将 `registry=None` 显式传给指标构造函数，而 prometheus_client 在**显式** `registry=None` 时完全跳过注册（与"不传该参数默认注册到全局 REGISTRY"语义不同）。此前未暴露的原因：单测均注入独立 registry；多进程模式数值写 mmap 文件、`MultiProcessCollector` 直接读文件聚合，不依赖注册。

修复（commit `fix(metrics)`）：未注入 registry 时回退到 prometheus 默认 `REGISTRY`；新增回归用例 `TestDefaultRegistry::test_metrics_land_in_default_registry`（覆盖默认 REGISTRY 路径，且通过前后清理做到与套件执行顺序无关）。

修复后单进程独立验证：

```
$ FLEXKV_ENABLE_METRICS=1 python <standalone_server_script> &
$ curl -s http://127.0.0.1:8080/metrics | grep ^flexkv_py_

flexkv_py_cache_match_blocks_total{device="cpu",ready="true"} 20.0
flexkv_py_cache_match_blocks_total{device="cpu",ready="false"} 3.0
flexkv_py_cache_query_total{result="full"} 1.0
flexkv_py_layerwise_submit_seconds_count 1.0
flexkv_py_layerwise_submit_seconds_sum 0.004
flexkv_py_layerwise_submit_total{status="ok"} 1.0
flexkv_py_swa_evicted_slots_total{device="cpu",reason="pool_full"} 2.0
flexkv_py_swa_hit_blocks_total 12.0
flexkv_py_swa_query_total{result="hit"} 1.0
flexkv_py_swa_slot_total{device="cpu"} 256.0
flexkv_py_swa_slot_used{device="cpu"} 1.0
...（13 个新指标全部正常输出，含 0 值 label 组合）
```

修复后测试：单测 14/14 通过（新增 1 例回归），metrics 开启回归 218 通过。

---

## 已知问题与说明

1. **上游测试漂移（非本改动引入）**：`tests/test_config_swa_multi_group.py` 4 例、`tests/test_swa_state_sidecars.py` 1 例引用已删除的 `swa_multi_layer` 配置属性，在干净 main 上即失败。
2. **`CPURemoteTransferWorker.launch_transfer` 缺 `return True`**（上游 main 已确认存在）：remote op 永不回报完成、任务等满 20s 假超时。本部署未启用 remote tier 不受影响；做 remote IO 指标前需先修。
3. **layerwise 指标**（workers_ready/expected、submit_*）产生在 TransferManager 子进程，需设置 `PROMETHEUS_MULTIPROC_DIR` 才能经 8080 端点聚合暴露；本次部署形态（layerwise=False）下恒为 0，属预期。
4. **多进程端口仲裁**：各进程均尝试绑定 8080，先到先得，其余探测后跳过；sglang 会重命名进程，不可依赖进程名区分主/子。
