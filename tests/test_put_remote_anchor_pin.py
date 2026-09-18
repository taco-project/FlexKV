"""The ordinary (non-Mooncake) REMOTE prefix must stay pinned for a PUT.

``remote_cache_engine.take(..., protected_node=...)`` only guards the matched
remote prefix *while this plan allocates*. Once planning returns, the plan's
staging is published relative to that prefix: ``staged_start_block`` is the
number of blocks the plan matched. If a concurrent allocation evicts the prefix
before H2REMOTE lands, ``_commit_deferred_insert`` fresh-rematches, sees
``current_blocks < staged_start_block``, and throws away a suffix this graph
wrote successfully -- and the CPU-full-hit early return can keep a retry from
ever rewriting it.

CPU and SSD anchors were already pinned for the plan's lifetime; REMOTE was not.
This is the reviewer's repro for that gap: a four-block remote pool holding a
two-block prefix, a PUT extending it to four, an unrelated two-block allocation
in between, then completion.
"""
import numpy as np
import pytest

from flexkv import c_ext
from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import (
    CacheConfig,
    GLOBAL_CONFIG_FROM_ENV,
    ModelConfig,
)
from flexkv.common.transfer import DeviceType

pytestmark = pytest.mark.unit

TPB = 16
NUM_REMOTE = 4

# (A stubbed c_ext module has no __file__.)
_HAS_REAL_C_EXT = getattr(c_ext, "__file__", None) is not None

INDEX_ACCEL_MODES = [
    pytest.param(False, id="python-index"),
    pytest.param(True, id="accel-index",
                 marks=pytest.mark.skipif(not _HAS_REAL_C_EXT,
                                          reason="requires compiled c_ext")),
]


def _locks(node) -> int:
    """Lock refcount of a radix node, whichever index built it.

    The pure-Python ``RadixNode`` exposes ``lock_cnt`` as an attribute; the
    accel index's ``c_ext.CRadixNode`` only binds the ``get_lock_cnt()``
    accessor.
    """
    getter = getattr(node, "get_lock_cnt", None)
    return getter() if getter is not None else node.lock_cnt


def _tokens(num_blocks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 50000, num_blocks * TPB, dtype=np.int64)


def _seq(tokens: np.ndarray) -> SequenceMeta:
    return SequenceMeta(token_ids=tokens, tokens_per_block=TPB)


@pytest.fixture(params=INDEX_ACCEL_MODES)
def engine(request, monkeypatch):
    """CPU + an ordinary remote tier backed by a plain ``CacheEngine``.

    ``CacheConfig.__post_init__`` derives ``enable_remote`` from
    ``enable_3rd_remote`` / mooncake, and both of those route the remote tier to
    a different engine class (hierarchical / key-addressed). Flip the flag after
    construction to get the plain block-addressed remote tier this test is about.
    """
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "index_accel", request.param)
    cache_config = CacheConfig(tokens_per_block=TPB,
                               enable_cpu=True,
                               enable_ssd=False,
                               num_cpu_blocks=64,
                               num_remote_blocks=NUM_REMOTE)
    cache_config.enable_remote = True
    model_config = ModelConfig(num_layers=2, num_kv_heads=2, head_size=8,
                               tp_size=1)
    return GlobalCacheEngine(cache_config, model_config)


def test_remote_prefix_survives_a_concurrent_allocation_during_put(engine):
    remote = engine.remote_cache_engine
    full = _tokens(4, seed=7)
    prefix = full[:2 * TPB]

    # A two-block remote prefix already exists.
    seeded = remote.take(2)
    assert remote.insert(_seq(prefix), seeded) is not None
    assert remote.match(_seq(full)).num_matched_blocks == 2

    # Plan a PUT that extends it to four blocks. It stages the two-block
    # suffix, so the remote pool is now full.
    _graph, mask, callback, op_callbacks, _end = engine.put(
        request_id=1,
        token_ids=full,
        token_mask=np.ones_like(full, dtype=bool),
        slot_mapping=np.arange(full.size, dtype=np.int64),
        dp_client_id=0,
    )
    assert mask.any()
    assert DeviceType.REMOTE in callback.keywords["node_to_unlock"], \
        "the matched remote prefix must be owned by the plan"
    assert remote.mempool.num_free_blocks == 0

    # An unrelated allocation arrives mid-flight. The prefix is pinned and the
    # staging is allocated, so there is nothing it may take.
    stolen = remote.take(2, strict=False)
    assert len(stolen) == 0, "the pinned remote prefix must not be evicted"
    assert remote.match(_seq(full)).num_matched_blocks == 2

    for op_callback in op_callbacks.values():
        op_callback()
    callback()

    assert remote.match(_seq(full)).num_matched_blocks == 4, \
        "the written suffix must publish on top of the surviving prefix"
    assert remote.index.total_cached_blocks() == 4
    assert remote.mempool.num_used_blocks == 4


def test_put_releases_the_remote_pin_when_it_completes(engine):
    remote = engine.remote_cache_engine
    full = _tokens(4, seed=8)
    prefix = full[:2 * TPB]
    anchor = remote.insert(_seq(prefix), remote.take(2))
    assert anchor is not None
    locked_before = _locks(anchor)

    _graph, _mask, callback, op_callbacks, _end = engine.put(
        request_id=1,
        token_ids=full,
        token_mask=np.ones_like(full, dtype=bool),
        slot_mapping=np.arange(full.size, dtype=np.int64),
        dp_client_id=0,
    )
    assert _locks(anchor) == locked_before + 1

    for op_callback in op_callbacks.values():
        op_callback()
    callback()

    assert _locks(anchor) == locked_before, "completion must drop the pin"


def test_aborted_put_releases_the_remote_pin_and_its_staging(engine):
    remote = engine.remote_cache_engine
    full = _tokens(4, seed=9)
    prefix = full[:2 * TPB]
    anchor = remote.insert(_seq(prefix), remote.take(2))
    assert anchor is not None
    locked_before = _locks(anchor)

    _graph, _mask, callback, _op_callbacks, _end = engine.put(
        request_id=1,
        token_ids=full,
        token_mask=np.ones_like(full, dtype=bool),
        slot_mapping=np.arange(full.size, dtype=np.int64),
        dp_client_id=0,
    )
    assert _locks(anchor) == locked_before + 1
    assert remote.mempool.num_free_blocks == 0

    callback.abort()

    assert _locks(anchor) == locked_before, "abort must drop the pin too"
    assert remote.mempool.num_free_blocks == NUM_REMOTE - 2, \
        "unwritten remote staging must go back to the pool"
    assert remote.match(_seq(full)).num_matched_blocks == 2
    assert remote.index.total_cached_blocks() == 2
