""" End-to-end vLLM + FlexKV cache-reset test.

Exercises the EXACT call verl makes to clear_kv_cache after a weight update: `reset_prefix_cache(reset_connector=True)`

Using a vLLM engine configured with the FlexKV connector.

vllm:prefix_cache_hits / _queries            -> local GPU
vllm:external_prefix_cache_hits / _queries   -> KV connector (FlexKV)

We diff the EXTERNAL counters around each generate() to get per-run hits.
(reset_prefix_cache does NOT zero these counters, so we must diff, not read absolute values.)

Test protocol (why it proves the reset works):
    warm  : generate                           -> FlexKV gets populated
    CLEAR : reset_prefix_cache(reset_connector=True) [+ reset_mm_cache]  (verl call)
    run1  : generate     -> external hits ~= 0   (FlexKV was cleared)
    run2  : generate     -> external hits  > 0   (run1 re-populated FlexKV)

Run:
    FLEXKV_CPU_CACHE_GB=64 pytest -s tests/test_reset_cache_vllm_e2e.py
"""

import os
import time

import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
pytest.importorskip("flexkv")

# Default matches the user's `vllm serve` command; override with FLEXKV_TEST_MODEL.
MODEL = "/raid/model/Qwen3-8B"
MAX_MODEL_LEN = "8192"

# Prometheus counter names (vllm/v1/metrics/loggers.py).
_EXT_HITS = "vllm:external_prefix_cache_hits"
_EXT_QUERIES = "vllm:external_prefix_cache_queries"

def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def _counter(llm, name):
    """Read a cumulative Prometheus counter value (0 if absent)."""
    for m in llm.get_metrics():
        if m.name == name:
            return getattr(m, "value", 0)
    return 0


def _verl_clear(llm):
    """Reproduce verl's clear_kv_cache() (vllm_async_server.py:809-820).

    The llm fixture already skips vLLM < 0.13.0, so reset_connector is always
    forwarded and reset_mm_cache always exists here.
    """
    ok = llm.reset_prefix_cache(reset_connector=True)
    llm.reset_mm_cache()
    # reset_encoder_cache is multimodal-only; skipped for a text model.
    return ok


def generate(llm, prompts, sp):
    """Generate the whole batch once; return per-run external-counter deltas.

    Returns (ext_hits, ext_queries) as counter deltas around the generate()
    call. Sleeps afterwards so FlexKV's async put has time to land before the
    next run reads it back.
    """
    eh0, eq0 = _counter(llm, _EXT_HITS), _counter(llm, _EXT_QUERIES)
    llm.generate(prompts, sp)
    # time.sleep(5)  # let FlexKV's async put/get drain before measuring next run
    return (_counter(llm, _EXT_HITS) - eh0,
            _counter(llm, _EXT_QUERIES) - eq0)


def _reset_local_prefix_cache(llm, retries=50, delay=0.1):
    """Reset ONLY the local GPU prefix cache (FlexKV kept), tolerating FlexKV's
    async D2H put.

    Unlike the verl clear (reset_connector=True), this path keeps FlexKV, so we
    must NOT free GPU blocks out from under an in-flight save. After generate()
    returns, FlexKV's put may still be draining and vLLM holds those blocks under
    delayed free, so reset_prefix_cache() returns False until the saves land and
    the engine's next steps free the blocks. Poll until the reset succeeds rather
    than sleeping a fixed interval.
    """
    for _ in range(retries):
        if llm.reset_prefix_cache() is not False:
            return True
        time.sleep(delay)
    return False


@pytest.fixture(scope="module")
def flexkv_config():
    """Ensure FlexKV has a CPU-cache config.

    FlexKV reads either FLEXKV_CPU_CACHE_GB (simplest) or FLEXKV_CONFIG_PATH
    (yaml/json). If the user set neither, default to a small CPU cache via the
    env-var route (per docs/vllm_adapter/README_en.md).
    """
    if os.environ.get("FLEXKV_CONFIG_PATH") or os.environ.get("FLEXKV_CPU_CACHE_GB"):
        return
    os.environ["FLEXKV_CPU_CACHE_GB"] = "8"


@pytest.fixture(scope="module")
def llm(flexkv_config):
    _skip_if_no_gpu()
    from vllm import LLM
    from vllm.config import KVTransferConfig

    # Mirror the user's `vllm serve` command as closely as the offline LLM API
    # allows (same model, TP, prefix caching, chunked prefill, FlexKV connector).
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        trust_remote_code=True,           # Qwen3 needs this
        max_model_len=MAX_MODEL_LEN,      # 8192, matches --max_model_len
        max_num_seqs=128,                 # matches --max-num-seqs
        max_num_batched_tokens=8192,      # matches --max-num-batched-tokens
        gpu_memory_utilization=0.5,       # matches --gpu-memory-utilization
        enable_prefix_caching=True,       # matches --enable-prefix-caching (required)
        enable_chunked_prefill=True,      # matches --enable-chunked-prefill
        disable_log_stats=False,          # required: get_metrics() asserts otherwise
        kv_transfer_config=KVTransferConfig(
            kv_connector="FlexKVConnectorV1",
            kv_role="kv_both",
        ),
    )
    yield llm
    del llm


# A batch of long prompts so each spans many KV blocks and actually gets
# offloaded to FlexKV. `turn` is the fixed dataset index so re-sending the same
# batch produces identical prompts (a prerequisite for prefix-cache hits).
_DATASET = [
    "你有什么爱好？",
    "你有什么特长？",
    "你有什么兴趣？",
    "你有什么梦想？",
    "你有什么愿望？",
    "你有什么期待？",
    "你有什么计划？",
    "你有什么遗憾？",
]
PROMPTS = [f"[{turn}] {prompt * 800}" for turn, prompt in enumerate(_DATASET)]


def test_reset_prefix_cache_reset_connector_succeeds(llm):
    """Smoke: the exact verl call returns without error (wire is connected)."""
    from vllm import SamplingParams

    llm.generate(PROMPTS, SamplingParams(max_tokens=8, temperature=0))
    # time.sleep(5)  # drain FlexKV's async D2H put -> quiesced boundary before reset
    ok = _verl_clear(llm)
    # Scheduler treats only a literal False as failure.
    assert ok is not False


def test_flexkv_populates_before_asserting_reset(llm):
    """Sanity: with FlexKV warm but the LOCAL cache cleared, a re-run must hit
    FlexKV.

    The local GPU prefix cache would otherwise serve the whole prefix and
    starve the connector of queries (observed as ext=0/32). Clearing ONLY the
    local cache (reset_connector NOT set) forces the next run to miss locally
    and query FlexKV for the full prefix. If external hits stay 0 here, the
    connector isn't really working and the reset test below is meaningless.
    """
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=8, temperature=0)
    generate(llm, PROMPTS, sp)                       # populate FlexKV (+ local)
    # Clear LOCAL only (keep FlexKV) so the next run can't be served from GPU.
    # Poll: FlexKV's async put may still hold GPU blocks under delayed free.
    assert _reset_local_prefix_cache(llm), "local prefix cache reset failed"
    eh, eq = generate(llm, PROMPTS, sp)              # cold local -> full ext query
    assert eh > 0, (
        f"no external FlexKV hits with warm FlexKV + cold local (hits={eh}/{eq}); "
        f"FlexKV not engaged — check connector mount / get-put matching"
    )


def test_reset_invalidates_flexkv_hits(llm):
    """Behavioral: verl's clear must drop FlexKV so the next A does NOT hit it.

    Protocol: warm -> CLEAR -> run1 (expect ~0 ext hits) -> run2 (expect > run1).
    """
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=8, temperature=0)

    def _rate(h, q):
        return h / max(q, 1)

    # warm: make sure A is in FlexKV before we clear.
    ehw, eqw = generate(llm, PROMPTS, sp)

    # CLEAR: exactly what verl's clear_kv_cache does (drops LOCAL + FlexKV).
    _verl_clear(llm)

    # run1: local was just cleared -> full ext query; FlexKV also cleared -> ~0
    # hits. This is the run that proves the reset reached FlexKV.
    eh1, eq1 = generate(llm, PROMPTS, sp)

    # run2: run1 refilled FlexKV, but it ALSO refilled the local GPU cache, which
    # would serve the whole prefix and starve the connector (ext=0/32). Clear
    # LOCAL only (keep FlexKV) so run2 is forced to query FlexKV for the full
    # prefix and can actually register hits.
    # Poll: FlexKV's async put may still hold GPU blocks under delayed free.
    assert _reset_local_prefix_cache(llm), "local prefix cache reset failed"
    eh2, eq2 = generate(llm, PROMPTS, sp)

    print(
        f"[reset-test] FlexKV hit-rate  "
        f"warm={_rate(ehw, eqw):.2f} ({ehw}/{eqw})  "
        f"run1={_rate(eh1, eq1):.2f} ({eh1}/{eq1})  "
        f"run2={_rate(eh2, eq2):.2f} ({eh2}/{eq2})"
    )

    rate1 = _rate(eh1, eq1)
    # Assert 1: cleared cache does not serve stale external hits.
    assert rate1 < 0.1, (
        f"external FlexKV hit-rate after clear is {rate1:.2f} (hits={eh1}/{eq1}); "
        f"reset did NOT propagate to FlexKV (stale KV still served)"
    )
    # Assert 2: cache is functional again afterwards (proves we measured a real
    # cache, not a permanently-dead one).
    assert eh2 > eh1, (
        f"external hits did not recover after refill (run1={eh1}, run2={eh2}); "
        f"FlexKV may not be working at all"
    )
