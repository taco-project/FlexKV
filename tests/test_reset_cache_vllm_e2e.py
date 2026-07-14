"""Level B: end-to-end vLLM + FlexKV cache-reset test.

Exercises the EXACT call verl makes after a weight update
(vllm_async_server.py clear_kv_cache: `reset_prefix_cache(reset_connector=True)`)
against a real vLLM engine configured with the FlexKV connector, and asserts —
via vLLM's own Prometheus counters — that FlexKV's EXTERNAL prefix cache is
actually invalidated (not the silent base-class no-op).

Why counters and not RequestOutput.num_cached_tokens:
    num_cached_tokens = local_gpu_hits + external(FlexKV)_hits, summed — it
    cannot isolate FlexKV. vLLM keeps them as SEPARATE cumulative counters:
        vllm:prefix_cache_hits / _queries            -> local GPU
        vllm:external_prefix_cache_hits / _queries   -> KV connector (FlexKV)
    We diff the EXTERNAL counters around each generate() to get per-run hits.
    (reset_prefix_cache does NOT zero these counters, so we must diff, not read
    absolute values.)

Test protocol (why it proves the reset works):
    warm  : generate(A)                      -> FlexKV gets populated
    CLEAR : reset_prefix_cache(reset_connector=True) [+ reset_mm_cache]  (verl call)
    run1  : generate(A)  -> external hits ~= 0   (FlexKV was cleared)
    run2  : generate(A)  -> external hits  > 0   (run1 re-populated FlexKV)
    Assert: run1 external hit-rate is ~0 AND run2 > run1.
    If the reset did NOT reach FlexKV, run1 would already hit -> test fails.

Requirements:
  - 1 GPU
  - vLLM >= 0.13.0 (older versions don't forward reset_connector)
  - FlexKV installed/importable (with the reset_cache changes in this branch)
  - A small model (default: Qwen/Qwen2.5-0.5B-Instruct)
  - FlexKV cache config: set FLEXKV_CPU_CACHE_GB (e.g. 8) or FLEXKV_CONFIG_PATH.
    If neither is set, this test defaults FLEXKV_CPU_CACHE_GB=8.

Run:
    FLEXKV_CPU_CACHE_GB=8 pytest -s tests/test_reset_cache_vllm_e2e.py
"""
import os

import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
pytest.importorskip("flexkv")

from packaging import version

# Default matches the user's `vllm serve` command; override with FLEXKV_TEST_MODEL.
MODEL = os.environ.get("FLEXKV_TEST_MODEL", "/raid/model/Qwen3-8B")
MAX_MODEL_LEN = int(os.environ.get("FLEXKV_TEST_MAX_MODEL_LEN", "8192"))

# Prometheus counter names (vllm/v1/metrics/loggers.py).
_EXT_HITS = "vllm:external_prefix_cache_hits"
_EXT_QUERIES = "vllm:external_prefix_cache_queries"
_LOCAL_HITS = "vllm:prefix_cache_hits"


def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def _skip_if_vllm_too_old():
    if version.parse(vllm.__version__) < version.parse("0.13.0"):
        pytest.skip(f"vLLM {vllm.__version__} < 0.13.0 does not forward reset_connector")


def _counter(llm, name):
    """Read a cumulative Prometheus counter value (0 if absent)."""
    for m in llm.get_metrics():
        if m.name == name:
            return getattr(m, "value", 0)
    return 0


def _reset_kwargs():
    """Mirror verl's _RESET_PREFIX_CACHE_KWARGS gating (vllm_async_server.py:59-62)."""
    kw = {}
    if version.parse(vllm.__version__) >= version.parse("0.13.0"):
        kw["reset_connector"] = True
    return kw


def _verl_clear(llm):
    """Reproduce verl's clear_kv_cache() (vllm_async_server.py:809-820)."""
    ok = llm.reset_prefix_cache(**_reset_kwargs())
    if version.parse(vllm.__version__) >= version.parse("0.9.0"):
        llm.reset_mm_cache()
    # reset_encoder_cache is multimodal-only; skipped for a text model.
    return ok


def _run_A(llm, prompt, sp):
    """Generate once; return per-run deltas (ext_hits, ext_queries, local_hits)."""
    eh0, eq0, lh0 = (_counter(llm, _EXT_HITS),
                     _counter(llm, _EXT_QUERIES),
                     _counter(llm, _LOCAL_HITS))
    llm.generate([prompt], sp)
    return (_counter(llm, _EXT_HITS) - eh0,
            _counter(llm, _EXT_QUERIES) - eq0,
            _counter(llm, _LOCAL_HITS) - lh0)


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
    _skip_if_vllm_too_old()
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


# A long prompt so it spans many KV blocks and actually gets offloaded to FlexKV.
_PROMPT_A = "Tell me about the history of the Roman Empire in detail. " * 40


def test_reset_prefix_cache_reset_connector_succeeds(llm):
    """Smoke: the exact verl call returns without error (wire is connected)."""
    from vllm import SamplingParams

    llm.generate([_PROMPT_A], SamplingParams(max_tokens=8, temperature=0))
    ok = _verl_clear(llm)
    # Scheduler treats only a literal False as failure.
    assert ok is not False


def test_flexkv_populates_before_asserting_reset(llm):
    """Sanity: without any reset, re-running A must produce EXTERNAL hits.

    If this fails, FlexKV isn't actually engaged (prompt too short / connector
    not mounted), and the reset test below would be meaningless. This guards
    against a false-pass where '0 hits after reset' is really '0 hits ever'.
    """
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=8, temperature=0)
    _run_A(llm, _PROMPT_A, sp)                 # populate FlexKV
    llm.reset_prefix_cache(**_reset_kwargs())  # clear LOCAL only-ish; also clears FlexKV
    # After clear FlexKV is empty; this run repopulates it (few/no ext hits)...
    _run_A(llm, _PROMPT_A, sp)
    # ...and THIS run should hit FlexKV (no reset in between).
    eh, eq, _ = _run_A(llm, _PROMPT_A, sp)
    assert eh > 0, (
        f"no external FlexKV hits even without reset (hits={eh}/{eq}); FlexKV not "
        f"engaged — check prompt length / connector mount before trusting reset test"
    )


def test_reset_invalidates_flexkv_hits(llm):
    """Behavioral: verl's clear must drop FlexKV so the next A does NOT hit it.

    Protocol: warm -> CLEAR -> run1 (expect ~0 ext hits) -> run2 (expect > run1).
    """
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=8, temperature=0)

    # warm: make sure A is in FlexKV before we clear.
    _run_A(llm, _PROMPT_A, sp)

    # CLEAR: exactly what verl's clear_kv_cache does.
    _verl_clear(llm)

    # run1: right after clear -> FlexKV empty -> external hits ~= 0.
    eh1, eq1, _ = _run_A(llm, _PROMPT_A, sp)
    # run2: no clear in between -> run1 refilled FlexKV -> external hits climb.
    eh2, eq2, _ = _run_A(llm, _PROMPT_A, sp)

    print(f"[reset-test] run1 ext={eh1}/{eq1}  run2 ext={eh2}/{eq2}")

    rate1 = eh1 / max(eq1, 1)
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
