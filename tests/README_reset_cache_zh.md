# KV cache reset 功能测试指南

[English](README_reset_cache.md)

本文测试 `reset_cache` 路径：vLLM 的
`reset_prefix_cache(reset_connector=True)`（verl 在每次更新权重后调用）向 FlexKV
传播 reset，丢弃旧权重产生的 KV。

涉及代码：

- `flexkv/kvtask.py`：`KVTaskEngine.reset_cache()` 排空在飞任务并重置各缓存层。
- `flexkv/kvmanager.py`：`KVManager.reset()`，覆盖进程内和 server-client 路径。
- `flexkv/server/{request,client,server}.py`：`ResetRequest` IPC 往返。
- `flexkv/integration/vllm/vllm_v1_adapter.py`：connector 的 `reset_cache()`。
- vLLM 仓库的 `.../v1/flexkv_connector.py`：wrapper 转发 `reset_cache()`。

## 两级测试

| 级别 | 文件 | 环境 | 验证内容 |
| --- | --- | --- | --- |
| A | `test_reset_cache.py` | 1 张 GPU（纯 no-op 用例不需要） | 清空 FlexKV 树、释放 mempool，IPC 往返有效 |
| B | `test_reset_cache_vllm_e2e.py` | 1 张 GPU、vLLM≥0.13 和小模型 | verl 的调用到达 FlexKV 并使缓存失效 |

## 运行

```bash
# A 级；test_reset_empty_is_noop 不需要 GPU
pytest -s tests/test_reset_cache.py

# B 级；也可以使用本地已有的其他小模型
export FLEXKV_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct
pytest -s tests/test_reset_cache_vllm_e2e.py
```

## B 级中通过 FlexKV 侧 spy 检查调用

vLLM connector-stats API 随版本变化；B 级行为测试无法读取 external-hit 指标时会
跳过。若需直接确认 reset 到达 FlexKV，可以在 conftest 或测试中增加：

```python
import flexkv.integration.vllm.vllm_v1_adapter as adapter

def test_reset_calls_flexkv(monkeypatch, llm):
    calls = {"n": 0}
    orig = adapter.FlexKVSchedulerConnector.reset_cache
    def spy(self):
        calls["n"] += 1
        return orig(self)
    monkeypatch.setattr(adapter.FlexKVSchedulerConnector, "reset_cache", spy)

    llm.reset_prefix_cache(reset_connector=True)
    assert calls["n"] >= 1
```

connector 运行在 scheduler 进程中。离线 `LLM` 的 enforce-eager 单进程模式下上述
monkeypatch 才适用；多进程引擎需要在 scheduler 进程安装 spy，或通过共享计数验证。

## 各断言的意义

1. `test_reset_empty_is_noop`：空缓存 reset 幂等，快速返回。
2. `test_reset_after_put_clears_match`：原先命中的前缀 reset 后不再命中，mempool 释放。
3. `test_reset_cache_server_client`：server-client IPC reset 路径有效。
4. B 级 smoke：调用无异常返回。
5. B 级行为测试或 spy：reset 实际到达 FlexKV，而非调用基类空实现。

## 手动检查（不使用 pytest）

```python
from flexkv.kvmanager import KVManager
from flexkv.common.config import ModelConfig, CacheConfig
import torch
m = ModelConfig(num_layers=4, num_kv_heads=8, head_size=128,
                dtype=torch.float16, kv_dim=2, tp_size=1, dp_size=1)
c = CacheConfig(tokens_per_block=16, enable_cpu=True, enable_ssd=False, num_cpu_blocks=1024)
kv = KVManager(model_config=m, cache_config=c, dp_client_id=0); kv.start()
kv.reset()          # 空缓存应快速返回且不抛错
kv.shutdown()
```
