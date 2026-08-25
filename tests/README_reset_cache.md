# KV-cache reset feature — test guide

This covers the `reset_cache` path that lets vLLM's
`reset_prefix_cache(reset_connector=True)` (what verl calls after every weight
update) propagate into FlexKV so stale-weight KV is dropped.

Code changes under test:
- `flexkv/kvtask.py` — `KVTaskEngine.reset_cache()` (drain in-flight → all-tier reset)
- `flexkv/kvmanager.py` — `KVManager.reset()` (in-proc + server-client)
- `flexkv/server/{request,client,server}.py` — `ResetRequest` IPC round-trip
- `flexkv/integration/vllm/vllm_v1_adapter.py` — connector `reset_cache()`
- (vLLM repo) `.../v1/flexkv_connector.py` — wrapper forwards `reset_cache()`

## Two levels

| Level | File | Needs | Proves |
|---|---|---|---|
| A | `test_reset_cache.py` | 1 GPU (pure-noop case: none) | FlexKV drops tree + frees mempool; IPC round-trip works |
| B | `test_reset_cache_vllm_e2e.py` | 1 GPU + vLLM≥0.13 + small model | verl's real call reaches FlexKV and invalidates it |

## Run

```bash
# Level A (fastest; test_reset_empty_is_noop needs no GPU)
pytest -s tests/test_reset_cache.py

# Level B (end-to-end through vLLM)
export FLEXKV_TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct   # or any small model you have
pytest -s tests/test_reset_cache_vllm_e2e.py
```

## Recommended robust assertion for Level B: a FlexKV-side spy

vLLM's connector-stats API moves between versions, so the behavioral test in
Level B falls back to `pytest.skip` if it can't read external-hit metrics. The
**most reliable** way to prove "reset actually reached FlexKV" is to spy on the
adapter method directly. Add this to a conftest or the test:

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
    assert calls["n"] >= 1   # proves the wire reached FlexKV, not the base no-op
```

Note: the connector runs in the scheduler process. With the offline `LLM` in
enforce-eager, single-process mode the monkeypatch applies; if you move to a
multi-process engine the spy must be installed in the scheduler process (or
assert via a counter FlexKV writes to a file / shared value).

## What each assertion buys you

1. `test_reset_empty_is_noop` — idempotency & cheap short-circuit (§2.5: verl
   calls reset up to 3× per weight update).
2. `test_reset_after_put_clears_match` — the core correctness claim: a prefix
   that hit before reset misses after, and the mempool is fully freed.
3. `test_reset_cache_server_client` — the IPC path works (no more
   "clear_cache is not supported in server client mode" no-op).
4. Level B smoke — verl's exact call returns without error (wire is connected).
5. Level B behavioral / spy — reset genuinely reaches FlexKV (not §3 silent
   false-success).

## Manual sanity check (no pytest)

```python
from flexkv.kvmanager import KVManager
from flexkv.common.config import ModelConfig, CacheConfig
import torch
m = ModelConfig(num_layers=4, num_kv_heads=8, head_size=128,
                dtype=torch.float16, use_mla=False, tp_size=1, dp_size=1)
c = CacheConfig(tokens_per_block=16, enable_cpu=True, enable_ssd=False, num_cpu_blocks=1024)
kv = KVManager(model_config=m, cache_config=c, dp_client_id=0); kv.start()
kv.reset()          # must not raise, must be near-instant on empty cache
kv.shutdown()
```
