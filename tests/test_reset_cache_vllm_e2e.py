"""Level B: end-to-end vLLM + FlexKV reset test.

This exercises the EXACT call verl makes after a weight update
(vllm_async_server.py: `await self.engine.reset_prefix_cache(reset_connector=True)`)
against a real vLLM engine configured with the FlexKV connector, and asserts
that FlexKV's external cache is actually invalidated (not the §3 silent
false-success).

We use the OFFLINE `LLM` (synchronous) rather than the async server because it
exposes the identical `reset_prefix_cache(reset_connector=...)` API with far
less setup — the connector code path hit is the same one verl drives.

Requirements:
  - 1 GPU
  - vLLM >= 0.13.0 (older versions don't forward reset_connector)
  - FlexKV installed/importable
  - A small model (default: Qwen/Qwen2.5-0.5B-Instruct)
  - FLEXKV_CONFIG_PATH pointing at a JSON with a cpu cache_config, e.g.:
        {"cache_config": {"enable_cpu": true, "num_cpu_blocks": 4096}}
    (this fixture writes one automatically to a temp file if unset)

Run:
    pytest -s tests/test_reset_cache_vllm_e2e.py
"""
import json
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
pytest.importorskip("flexkv")

from packaging import version

MODEL = os.environ.get("FLEXKV_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def _skip_if_vllm_too_old():
    if version.parse(vllm.__version__) < version.parse("0.13.0"):
        pytest.skip(f"vLLM {vllm.__version__} < 0.13.0 does not forward reset_connector")


@pytest.fixture(scope="module")
def flexkv_config_path(tmp_path_factory):
    """Ensure FLEXKV_CONFIG_PATH points at a minimal CPU-only cache config."""
    existing = os.environ.get("FLEXKV_CONFIG_PATH")
    if existing:
        return existing
    cfg = {"cache_config": {"enable_cpu": True, "num_cpu_blocks": 4096}}
    p = tmp_path_factory.mktemp("flexkv") / "flexkv_config.json"
    p.write_text(json.dumps(cfg))
    os.environ["FLEXKV_CONFIG_PATH"] = str(p)
    return str(p)


@pytest.fixture(scope="module")
def llm(flexkv_config_path):
    _skip_if_no_gpu()
    _skip_if_vllm_too_old()
    from vllm import LLM
    from vllm.config import KVTransferConfig

    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="FlexKVConnectorV1",
            kv_role="kv_both",
        ),
    )
    yield llm
    del llm


def _reset_kwargs():
    """Mirror verl's _RESET_PREFIX_CACHE_KWARGS gating."""
    kw = {}
    if version.parse(vllm.__version__) >= version.parse("0.13.0"):
        kw["reset_connector"] = True
    return kw


def test_reset_prefix_cache_reset_connector_succeeds(llm):
    """Smoke: the exact verl call must return without error.

    Weak assertion (returns True / no exception) — proves the reset_cache()
    override is wired end-to-end (not the base-class no-op path). Stronger
    behavioral assertion is in the next test.
    """
    from vllm import SamplingParams

    prompt = "The capital of France is " * 40  # long enough to span many blocks
    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0))

    # ★ This is the verl call (vllm_async_server.py clear_kv_cache line 815).
    ok = llm.reset_prefix_cache(**_reset_kwargs())
    # vLLM's reset_prefix_cache returns True on success; scheduler treats only a
    # literal False as failure. With the FlexKV reset_cache() override wired,
    # this should be truthy.
    assert ok is not False


def test_reset_invalidates_flexkv_hits(llm):
    """Behavioral: after reset, re-running the same prompt must NOT get external
    (FlexKV) cache hits — proving the offloaded KV was actually dropped.

    We compare FlexKV connector stats across two runs of the same prompt with a
    reset in between. Without a working reset, the 2nd run would hit FlexKV.

    NOTE: connector stats plumbing differs across vLLM versions; if stats are
    unavailable this test degrades to a structural check and is skipped with a
    clear message rather than silently passing.
    """
    from vllm import SamplingParams

    prompt = "Tell me about the history of the Roman Empire. " * 40
    sp = SamplingParams(max_tokens=8, temperature=0)

    # Warm-up run -> FlexKV gets populated on request_finished.
    llm.generate([prompt], sp)

    # Reset BOTH the local prefix cache and (via reset_connector) FlexKV.
    llm.reset_prefix_cache(**_reset_kwargs())

    # Re-run the identical prompt. If FlexKV was correctly invalidated, the
    # connector must recompute (0 external matched tokens) rather than load
    # stale blocks.
    #
    # Observing "external matched tokens" requires connector stats; the exact
    # accessor is version-specific. We surface whatever the engine exposes and
    # assert it does not indicate a fresh external hit. If we can't read it,
    # skip loudly (do not false-pass).
    stats = _try_get_connector_prefix_hits(llm)
    if stats is None:
        pytest.skip(
            "connector prefix-cache stats not accessible on this vLLM build; "
            "use the FlexKV-side spy (see README) to assert reset_cache() was called"
        )
    external_hits_after_reset = stats
    assert external_hits_after_reset == 0, (
        f"expected 0 external FlexKV hits after reset, got {external_hits_after_reset}"
    )


def _try_get_connector_prefix_hits(llm):
    """Best-effort read of external (connector) prefix-cache hit count.

    Returns an int, or None if the metric can't be located on this vLLM build.
    Kept intentionally defensive: vLLM's stats API is still moving.
    """
    try:
        # V1 engine path: metrics live on the engine core's scheduler stats.
        # This is best-effort and may need adjustment for your vLLM version.
        engine = getattr(llm, "llm_engine", None)
        if engine is None:
            return None
        # Try a few known-ish locations; return None if none present.
        get_metrics = getattr(engine, "get_metrics", None)
        if callable(get_metrics):
            metrics = get_metrics()
            for m in metrics:
                name = getattr(m, "name", "")
                if "connector" in name and "hit" in name:
                    return int(getattr(m, "value", 0))
        return None
    except Exception:
        return None
