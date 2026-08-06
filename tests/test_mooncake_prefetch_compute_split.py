"""M0: mooncake REMOTE2H is prefetch-only; compute GET stays local."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from flexkv.cache.cache_engine import (
    CacheStrategy,
    GetTransferPlan,
    GlobalCacheEngine,
    resolve_get_cache_strategy,
)
from flexkv.common.config import assert_mooncake_prefetch_ready
from flexkv.common.transfer import TransferOpGraph


pytestmark = pytest.mark.unit


def _bare_engine(*, mooncake: bool) -> GlobalCacheEngine:
    engine = GlobalCacheEngine.__new__(GlobalCacheEngine)
    engine._cache_tree_lock = threading.RLock()
    engine.use_mooncake_store_backend = mooncake
    engine.tokens_per_block = 1
    engine.cache_config = SimpleNamespace(
        enable_remote=True,
        tokens_per_block=1,
    )
    engine.cache_engines = {}
    engine._metrics_collector = None
    engine._check_input = lambda *args, **kwargs: None
    engine.slot_mapping_to_block_ids = (
        lambda slot_mapping, tpb: np.arange(len(slot_mapping), dtype=np.int64)
    )
    return engine


@pytest.mark.parametrize(
    ("mooncake", "strategy", "expect_ignore_remote"),
    [
        # Compute retrieve under mooncake: force local tiers only.
        (True, CacheStrategy(), True),
        # Prefetch (ignore_gpu): may still pull REMOTE2H.
        (True, CacheStrategy(ignore_gpu=True, ignore_gds=True), False),
        # Explicit ignore_remote stays ignore_remote.
        (True, CacheStrategy(ignore_remote=True), True),
        # Non-mooncake compute: remote still allowed.
        (False, CacheStrategy(), False),
        # Non-mooncake prefetch: unchanged.
        (False, CacheStrategy(ignore_gpu=True), False),
    ],
)
def test_resolve_get_cache_strategy(mooncake, strategy, expect_ignore_remote):
    resolved = resolve_get_cache_strategy(mooncake, strategy)
    assert resolved.ignore_remote is expect_ignore_remote
    assert resolved.ignore_gpu is strategy.ignore_gpu
    assert resolved.ignore_gds is strategy.ignore_gds
    assert resolved.ignore_ssd is strategy.ignore_ssd


def test_mooncake_compute_get_routes_to_local_not_global():
    """GPU-bound GET with mooncake must not enter _get_impl_global."""
    engine = _bare_engine(mooncake=True)
    local_calls = []
    global_calls = []
    empty_plan = GetTransferPlan.empty()

    def _local(*args, **kwargs):
        local_calls.append(kwargs.get("temp_cache_strategy") or args[5])
        return empty_plan

    def _global(*args, **kwargs):
        global_calls.append(True)
        return empty_plan

    engine._get_impl_local = _local
    engine._get_impl_global = _global
    engine.cache_engines = {}

    token_ids = np.arange(4, dtype=np.int64)
    token_mask = np.ones(4, dtype=np.bool_)
    slot_mapping = np.arange(4, dtype=np.int64)

    graph, return_mask, *_ = engine.get(
        request_id=1,
        token_ids=token_ids,
        token_mask=token_mask,
        slot_mapping=slot_mapping,
        dp_client_id=0,
        temp_cache_strategy=CacheStrategy(),
    )

    assert len(local_calls) == 1
    assert local_calls[0].ignore_remote is True
    assert global_calls == []
    assert isinstance(graph, TransferOpGraph)
    assert return_mask.shape == token_mask.shape


def test_mooncake_prefetch_get_still_allows_global():
    """Prefetch (ignore_gpu) under mooncake may still use _get_impl_global."""
    engine = _bare_engine(mooncake=True)

    local_calls = []
    global_calls = []
    empty_plan = GetTransferPlan.empty()

    def _local(*args, **kwargs):
        local_calls.append(True)
        return empty_plan

    def _global(*args, **kwargs):
        strategy = kwargs.get("temp_cache_strategy") or args[5]
        global_calls.append(strategy)
        return empty_plan

    engine._get_impl_local = _local
    engine._get_impl_global = _global
    engine.cache_engines = {}

    token_ids = np.arange(4, dtype=np.int64)
    graph, *_ = engine.get(
        request_id=2,
        token_ids=token_ids,
        token_mask=np.ones(4, dtype=np.bool_),
        slot_mapping=np.arange(4, dtype=np.int64),
        dp_client_id=0,
        temp_cache_strategy=CacheStrategy(ignore_gpu=True, ignore_gds=True),
    )

    assert local_calls == []
    assert len(global_calls) == 1
    assert global_calls[0].ignore_remote is False
    assert isinstance(graph, TransferOpGraph)


def test_cache_config_mooncake_enables_remote(tmp_path, monkeypatch):
    from flexkv.common.config import CacheConfig

    cfg_path = tmp_path / "mooncake.json"
    cfg_path.write_text("{}")
    monkeypatch.delenv("FLEXKV_USE_MOONCAKE_STORE_BACKEND", raising=False)

    cfg = CacheConfig(
        tokens_per_block=16,
        enable_cpu=True,
        use_mooncake_store_backend=True,
        mooncake_store_config_path=str(cfg_path),
    )
    assert cfg.enable_remote is True


def test_cache_config_enable_remote_from_mooncake_env(tmp_path, monkeypatch):
    """Env-only mooncake enable must also flip enable_remote (ordering bug)."""
    from flexkv.common.config import CacheConfig

    cfg_path = tmp_path / "mooncake.json"
    cfg_path.write_text("{}")
    monkeypatch.setenv("FLEXKV_USE_MOONCAKE_STORE_BACKEND", "1")
    monkeypatch.setenv("FLEXKV_MOONCAKE_STORE_CONFIG_PATH", str(cfg_path))

    # Default CacheConfig() path used by FlexKVConfig.field(default_factory=...)
    cfg = CacheConfig()
    assert cfg.use_mooncake_store_backend is True
    assert cfg.enable_remote is True


def test_assert_mooncake_prefetch_ready_rejects_without_prefetch():
    cache_config = SimpleNamespace(use_mooncake_store_backend=True)
    with pytest.raises(RuntimeError, match="requires prefetch"):
        assert_mooncake_prefetch_ready(cache_config, prefetch_enabled=False)


def test_assert_mooncake_prefetch_ready_ok_when_prefetch_on():
    assert_mooncake_prefetch_ready(
        SimpleNamespace(use_mooncake_store_backend=True),
        prefetch_enabled=True,
    )
    assert_mooncake_prefetch_ready(
        SimpleNamespace(use_mooncake_store_backend=False),
        prefetch_enabled=False,
    )
