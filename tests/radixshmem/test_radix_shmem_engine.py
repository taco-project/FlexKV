"""Tests for the radixshmem CPU tier: the engine on a radix-server, the data
plane FlexKV maps as its CPU pool, the planners, and the peer pull.
Skipped if `shmradix` is missing.

Four parts, one file:

  Part 1 — `CacheEngineRadixShmem` semantics against an in-process
    `shmradix.RadixServer` (index + SlotStore, started the way the operator's
    `radix-server` is: a name and a byte budget; FlexKV's client brings the
    geometry): take / insert / match / recycle, insert-after-transfer
    publication, lock vs. eviction, SWA windows, and the fact that a standalone
    region has no peer to prefetch from.
  Part 1b — the data plane: the SlotStore pool viewed as FlexKV's CPU tensor,
    the exact-stride geometry FlexKV derives from its configuration and hands
    to the server, the slot counts it adopts back, and a client that arrives
    before its server.
  Part 2 — `GlobalCacheEngine.get()/put()` planning on the radixshmem backend,
    driven by synthetic matches (no region): the local GET is one H2D, the
    prefetch plan carries a `pull_async` job, the PUT arms the deferred insert;
    plus `KVTaskEngine` completing a job-backed prefetch task.
  Part 3 — a real 2-node radixshmem cluster over RDMA in two spawned
    processes: node 0 publishes a prefix with bytes, node 1 prefetches it (the
    server pulls the bytes) and then matches it locally, byte for byte.

Import-time notes:
  * Parts 1 and 3 use a duck-typed fake `SequenceMeta` and side-load
    `radix_shmem_engine.py`, so they do not pull in `flexkv.c_ext` (CUDA).
  * Part 2 needs the real `GlobalCacheEngine` (and therefore `c_ext`), so it
    imports it lazily and skips instead of breaking collection for Parts 1/3.
  * Part 3 needs an ACTIVE RDMA device, a shmradix built WITH RDMA + etcd +
    mooncake, and an etcd (FLEXKV_TEST_RADIX_REGISTRY, or an `etcd` binary on
    PATH to start a private one); it is gated behind FLEXKV_RUN_RADIX_PEER_TEST=1.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import glob
import importlib.util
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import shmradix
except ImportError as exc:
    # Not importorskip: a shmradix built before the current API (stale `_core.so`
    # next to a newer `__init__.py`) raises ImportError rather than
    # ModuleNotFoundError, and pytest >= 8.2 only skips on the latter — which
    # would abort collection for the whole suite instead of skipping this file.
    pytest.skip(f"shmradix unusable ({exc}); rebuild the extension",
                allow_module_level=True)

for _name in ("RadixServer", "ServerConfig", "Geometry", "RadixClient"):
    if not hasattr(shmradix, _name):
        pytest.skip(f"shmradix lacks {_name}: needs the RadixServer/RadixClient surface",
                    allow_module_level=True)


def _load_module_direct(name: str, path: str):
    """Load a module by file path, bypassing parent package __init__.

    `flexkv/cache/__init__.py` imports `flexkv.c_ext`, which links libcudart.
    Side-load `radix_shmem_engine` directly so the test runs on CPU-only hosts.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` looks up the module in sys.modules during class
    # construction; register before exec_module.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


_FLEXKV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_engine_mod = _load_module_direct(
    "_radix_shmem_engine_test",
    os.path.join(_FLEXKV_ROOT, "flexkv", "cache", "radix_shmem_engine.py"),
)
CacheEngineRadixShmem = _engine_mod.CacheEngineRadixShmem
# The planner only duck-types the match, so the side-loaded class is as good as
# the one `flexkv.cache.cache_engine` imports — and it needs no c_ext.
ShmRadixMatch = _engine_mod.ShmRadixMatch

# Pure-Python (no c_ext): the bootstrap (server config, geometry, attach) and
# the transfer enums.
from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV  # noqa: E402
from flexkv.common.transfer import TransferType  # noqa: E402
from flexkv.server import shm_radix_bootstrap as bootstrap  # noqa: E402

FULL = shmradix.ComponentType.FULL
_SWA = shmradix.ComponentType.SWA
SLOT_BYTES = 256      # bytes of one test block in the SlotStore


@dataclass
class FakeSeq:
    """Duck-typed SequenceMeta for the engine API: only `block_hashes` and
    `gen_hashes()` are read by `CacheEngineRadixShmem`."""
    block_hashes: np.ndarray
    tokens_per_block: int = 4

    @property
    def num_blocks(self) -> int:
        return len(self.block_hashes)

    def gen_hashes(self) -> None:
        # already populated
        pass


def _hashes(seed: int, num: int) -> np.ndarray:
    """Deterministic, distinct int64 hashes."""
    rng = np.random.default_rng(seed)
    return rng.integers(low=1, high=2**62, size=num, dtype=np.int64)


def _sweep_region(name: str, data_name: str | None = None) -> None:
    """Drop the shm objects and the socket a previous run may have left."""
    base = name.lstrip("/").replace("/", "_")
    paths = [f"/dev/shm/{base}", f"/dev/shm/{base}.sock",
             f"/dev/shm/{(data_name or name + '_data').lstrip('/')}"]
    for root in ("/dev/hugepages",):
        paths.append(f"{root}/{base}")
    for path in paths:
        with contextlib.suppress(FileNotFoundError, IsADirectoryError):
            os.remove(path)


def _server_budget(blocks: int, swa_slots: int = 0, slot_bytes: int = SLOT_BYTES):
    """(data_bytes, swa_ratio) that make a data-mode server plan exactly
    ``blocks`` FULL and ``swa_slots`` SWA slots of ``slot_bytes`` each (the
    stride equals the slot bytes, see ``slot_align_for``)."""
    data_bytes = (blocks + swa_slots) * slot_bytes
    swa_ratio = (swa_slots * slot_bytes) / data_bytes if swa_slots else 0.0
    return data_bytes, swa_ratio


def _server_config(name: str, blocks: int, tokens_per_block: int,
                   swa_slots: int = 0, window_blocks: int = 0,
                   slot_bytes: int = SLOT_BYTES):
    """A standalone data-mode radix-server the way the operator starts it (a
    name and a byte budget, nothing about the model) plus the Geometry
    FlexKV's client brings. Budget and ratio are chosen so the counts the
    server plans equal the test's ``blocks`` / ``swa_slots``."""
    align = bootstrap.slot_align_for(slot_bytes)
    data_bytes, swa_ratio = _server_budget(blocks, swa_slots, slot_bytes)
    cfg = shmradix.ServerConfig(name=name, data_bytes=data_bytes, swa_ratio=swa_ratio,
                                slot_align=align, prefault=False)
    geo = shmradix.Geometry(block_size=tokens_per_block, full_slot_bytes=slot_bytes,
                            swa_slot_bytes=slot_bytes if swa_slots else 0,
                            swa_window_blocks=window_blocks if swa_slots else 0,
                            slot_align=align)
    return cfg, geo


class _Env:
    """Owns the in-process servers and engines a test creates; closes them in
    reverse order at teardown (engine first, it maps the server's regions)."""

    def __init__(self) -> None:
        self._stack = []

    def server(self, cfg):
        """Start the server: ``waiting`` until a client brings the geometry."""
        _sweep_region(cfg.name, cfg.resolved_data_name)
        server = shmradix.RadixServer(cfg).start()
        self._stack.append(server.close)
        return server

    def engine(self, name: str, **kwargs) -> CacheEngineRadixShmem:
        engine = CacheEngineRadixShmem(name, **kwargs)
        self._stack.append(engine.close)
        return engine

    def make(self, name: str, blocks: int = 2000, tokens_per_block: int = 4, **engine_kwargs):
        cfg, geo = _server_config(name, blocks, tokens_per_block)
        server = self.server(cfg)
        engine = self.engine(name, geometry=geo, num_total_blocks=blocks,
                             tokens_per_block=tokens_per_block, **engine_kwargs)
        return engine, server

    def close(self) -> None:
        while self._stack:
            with contextlib.suppress(Exception):
                self._stack.pop()()


@pytest.fixture
def env():
    saved = bootstrap.READY_TIMEOUT_S
    bootstrap.READY_TIMEOUT_S = 60.0     # a broken test fails instead of waiting ten minutes
    e = _Env()
    try:
        yield e
    finally:
        e.close()
        bootstrap.READY_TIMEOUT_S = saved


# =============================================================================
# Part 1 — local engine semantics on a standalone radix-server
# =============================================================================


def test_take_insert_match_recycle(env):
    engine, _server = env.make("/cers_basic")

    seq = FakeSeq(block_hashes=_hashes(seed=1, num=4))
    # Initial match: nothing.
    r = engine.match(seq)
    assert r.num_matched_blocks == 0
    r.release()

    # take 4 slots and insert.
    slots = engine.take(num_required_blocks=4)
    assert len(slots) == 4
    engine.insert(seq, slots, num_insert_blocks=4)

    # Match should now hit all 4 blocks, all of them this node's slots.
    r2 = engine.match(seq)
    assert r2.num_matched_blocks == 4
    assert r2.local_slots.size == 4
    np.testing.assert_array_equal(np.sort(r2.local_slots), np.sort(slots))
    r2.release()

    # Recycle a fresh allocation; tree-attached slots are not affected.
    free_slots = engine.take(num_required_blocks=2)
    engine.recycle(free_slots)


def test_engine_adopts_the_servers_register_chunk(env):
    """The RHT registration chunk is whatever the radix-server was started
    with (--register-chunk-tokens); the engine carries it in tokens and in
    FlexKV blocks, converted the way radixshmem does."""
    engine, _server = env.make("/cers_chunk")           # radixshmem's default: 4096 tokens
    assert engine.register_chunk_tokens == 4096
    assert engine.register_chunk_blocks == 4096 // 4    # tokens_per_block 4
    name = f"/cers_chunk{os.getpid()}"
    cfg, geo = _server_config(name, blocks=64, tokens_per_block=4)
    env.server(dataclasses.replace(cfg, register_chunk_tokens=64))
    engine2 = env.engine(name, geometry=geo, num_total_blocks=64, tokens_per_block=4)
    assert engine2.register_chunk_tokens == 64 and engine2.register_chunk_blocks == 16


def test_take_clamps_to_the_pool(env):
    """radixshmem refuses an allocation larger than the pool (allocate_slots
    raises ValueError); the engine clamps instead, so a planner asking for more
    than exists gets what exists and nothing it took before leaks."""
    engine, _server = env.make(f"/cers_clamp{os.getpid()}", blocks=64)
    slots = engine.take(num_required_blocks=64 + 7)
    assert 0 < len(slots) <= 64
    engine.recycle(slots)
    assert engine.take(num_required_blocks=0).size == 0
    # the SWA pool is all-or-none: an oversize window comes back short and the
    # planner stores Full KV only
    swa_engine, _ = _make_swa_engine(env, f"/cers_clamp_swa{os.getpid()}", blocks=64,
                                     swa_slots=SWA_W, window_blocks=SWA_W)
    got = swa_engine.take(num_required_blocks=SWA_W + 3, component=_SWA)
    assert len(got) <= SWA_W
    swa_engine.recycle(got, component=_SWA)


def test_insert_publishes_immediately(env):
    """There is no ready bit: being in the tree IS being servable.

    Insert runs after the transfer on this backend, so a matched block is
    complete by construction — there is no flag to withhold a span with, and a
    single insert() is the whole publication.
    """
    engine, _server = env.make("/cers_unready")

    seq = FakeSeq(block_hashes=_hashes(seed=2, num=6))
    slots = engine.take(num_required_blocks=6)
    engine.insert(seq, slots, num_insert_blocks=6)

    r = engine.match(seq)
    assert r.num_matched_blocks == 6
    r.release()


def test_recycle_returns_staged_slots(env):
    """Slots whose transfer never landed are only reachable through recycle().

    They were never attached to the tree, so no query finds them and eviction
    cannot reclaim them — without recycle() they are lost for the life of the
    region.
    """
    engine, _server = env.make("/cers_recycle")

    before = engine.num_free_blocks
    slots = engine.take(num_required_blocks=5)
    assert engine.num_free_blocks == before - 5
    engine.recycle(slots)
    assert engine.num_free_blocks == before

    # And nothing was published on the way through.
    seq = FakeSeq(block_hashes=_hashes(seed=22, num=5))
    r = engine.match(seq)
    assert r.num_matched_blocks == 0
    r.release()


def test_eviction_reclaims_inserted(env):
    """A published span is immediately LRU-evictable.

    insert() runs after the transfer, so the span it attaches has no reader and
    takes no ref — nothing has to be released to make it reclaimable.
    """
    engine, _server = env.make("/cers_evict", blocks=2000)

    seq = FakeSeq(block_hashes=_hashes(seed=4, num=1500))
    s1 = engine.take(num_required_blocks=1500)
    engine.insert(seq, s1, num_insert_blocks=1500)
    # insert() reports nothing, so check the span landed rather than let the
    # eviction assert below pass on an empty tree.
    published = engine.match(seq)
    assert published.num_matched_blocks == 1500
    published.release()
    assert engine.num_free_blocks == 500
    # Allocate enough new blocks that eviction is forced (need > current free 500).
    s2 = engine.take(num_required_blocks=1500)
    assert len(s2) > 500


def test_pinned_match_survives_eviction_pressure(env):
    """The query pin (`lock=True`) is what keeps a matched prefix out of the
    evictor's reach until `release()`."""
    engine, _server = env.make("/cers_pin", blocks=2000)
    seq = FakeSeq(block_hashes=_hashes(seed=5, num=1500))
    engine.insert(seq, engine.take(1500), num_insert_blocks=1500)

    pinned = engine.match(seq)
    assert pinned.num_matched_blocks == 1500
    # 500 free; everything else is pinned, so the take comes up short.
    short = engine.take(num_required_blocks=1500)
    assert len(short) == 500
    engine.recycle(short)
    pinned.release()
    evicting = engine.take(num_required_blocks=1500)
    assert len(evicting) == 1500
    engine.recycle(evicting)


def test_standalone_region_has_no_peer(env):
    """A single-node region: peer reuse is off and prefetch has nothing to pull."""
    engine, _server = env.make("/cers_local_only", peer_enabled=True)
    seq = FakeSeq(block_hashes=_hashes(seed=12, num=3))
    slots = engine.take(num_required_blocks=3)
    engine.insert(seq, slots, num_insert_blocks=3)

    assert engine.is_distributed is False
    assert engine.peer_enabled is False            # asked for, but world_size == 1
    assert bootstrap.radix_cluster_rank(engine.client) == 0
    assert engine.prefetch(seq) is None
    result = engine.match(seq)
    assert result.num_matched_blocks == 3
    assert result.finalize is not None
    result.release()
    # release() is what drops the query's pin, and it is idempotent.
    assert result.finalize is None
    result.release()


def test_local_range_intersects_the_window():
    """`local_range` bounds the range on BOTH sides, by slicing alone.

    This is the accessor the GET and PUT planners lean on instead of clamping
    the match end themselves, so the contract is that a hit stopping short of
    the window contributes nothing and one running past the window end is
    trimmed to it.
    """
    match = ShmRadixMatch(
        num_matched_blocks=4,
        local_slots=np.arange(40, 44, dtype=np.int64),
    )
    # Wholly inside the hit.
    assert match.local_range(1, 3).tolist() == [41, 42]
    # Hit runs PAST the window end -> trimmed to the window.
    assert match.local_range(0, 2).tolist() == [40, 41]
    # Window runs past the hit -> trimmed to the hit, no error.
    assert match.local_range(2, 99).tolist() == [42, 43]
    # Hit stops short of the window start -> nothing of it is ours.
    assert match.local_range(4, 9).tolist() == []
    assert match.local_range(7, 9).tolist() == []
    # Empty and inverted windows name no block; full window is the whole hit.
    assert match.local_range(2, 2).tolist() == []
    assert match.local_range(3, 1).tolist() == []
    assert match.local_range(0, 4).tolist() == [40, 41, 42, 43]


SWA_W = 8  # == flexkv.common.config.RADIX_SWA_WINDOW_BLOCKS, literal on purpose:
           # a drive-by change to the constant should fail here, visibly.
JOINT_MASK = (_engine_mod.COMPONENT_MASK_FULL |
              _engine_mod.COMPONENT_MASK_SWA)


def _make_swa_engine(env, name: str, blocks: int = 2000, swa_slots: int = 64,
                     tokens_per_block: int = 16, window_blocks: int = SWA_W):
    """A single region carrying the SWA component, and an engine that knows it."""
    from flexkv.common.config import SWAPoolConfig
    cfg, geo = _server_config(name, blocks, tokens_per_block,
                              swa_slots=swa_slots, window_blocks=window_blocks)
    server = env.server(cfg)
    engine = env.engine(
        name, geometry=geo, num_total_blocks=blocks, tokens_per_block=tokens_per_block,
        swa_config=SWAPoolConfig(enabled=True, num_slots=swa_slots,
                                 window_blocks=window_blocks))
    return engine, server


def _publish_full(engine, seq, num_blocks: int) -> np.ndarray:
    slots = engine.take(num_blocks)
    engine.insert(seq, slots, num_insert_blocks=num_blocks)
    return slots


def _publish_swa(engine, seq, path_end: int,
                 window_blocks: int = SWA_W) -> np.ndarray:
    k = min(path_end, window_blocks)
    slots = engine.take(k, component=_SWA)
    assert len(slots) == k, "SWA pool unexpectedly short in test setup"
    engine.insert(seq, slots, num_insert_blocks=path_end, component=_SWA)
    return slots


def test_swa_window_invisible_until_published_then_joint_hit(env):
    """With `common_hit=20` the Full slots cover [0, 20) and the SWA slots cover
    [12, 20) -- and before insert(SWA), the joint query matches NOTHING even
    though Full alone matches 20."""
    engine, _server = _make_swa_engine(env, "/cers_swa_basic")
    seq = FakeSeq(block_hashes=_hashes(41, 20), tokens_per_block=16)

    _publish_full(engine, seq, 20)

    match = engine.match(seq)
    assert match.num_matched_blocks == 20
    assert len(match.swa_slots) == 0 and match.swa_start == 0
    match.release()
    match = engine.match(seq, component_mask=JOINT_MASK)
    assert match.num_matched_blocks == 0
    assert len(match.swa_slots) == 0
    match.release()

    swa_slots = _publish_swa(engine, seq, path_end=20)

    match = engine.match(seq, component_mask=JOINT_MASK)
    assert match.num_matched_blocks == 20           # joint common hit
    assert match.local_slots.size == 20             # Full covers [0, 20)
    assert match.swa_start == 12                    # max(0, 20 - 8)
    assert len(match.swa_slots) == SWA_W            # window covers [12, 20)
    assert sorted(match.swa_slots.tolist()) == sorted(swa_slots.tolist())
    match.release()


def test_swa_short_path_window_starts_at_zero(env):
    """A path shorter than W publishes a window over the whole path: k=n slots,
    swa_start=0 -- the `k = min(path_end, W)` boundary."""
    engine, _server = _make_swa_engine(env, "/cers_swa_short")
    seq = FakeSeq(block_hashes=_hashes(42, 5), tokens_per_block=16)

    _publish_full(engine, seq, 5)
    _publish_swa(engine, seq, path_end=5)

    match = engine.match(seq, component_mask=JOINT_MASK)
    assert match.num_matched_blocks == 5
    assert match.swa_start == 0
    assert len(match.swa_slots) == 5
    match.release()


def test_swa_take_is_all_or_none_and_the_query_pin_protects_the_window(env):
    """`allocate_slots(k, SWA)` returns k slots or NOTHING. With the pool sized
    to exactly one window, a joint match's pin keeps that window un-evictable
    (empty take); releasing the pin frees it for eviction (full take)."""
    engine, _server = _make_swa_engine(env, "/cers_swa_allornone", swa_slots=SWA_W)
    seq = FakeSeq(block_hashes=_hashes(43, 20), tokens_per_block=16)

    _publish_full(engine, seq, 20)
    _publish_swa(engine, seq, path_end=20)

    match = engine.match(seq, component_mask=JOINT_MASK)
    assert len(match.swa_slots) == SWA_W
    empty = engine.take(SWA_W, component=_SWA)
    assert len(empty) == 0                          # all pinned -> all or none
    match.release()

    evicted = engine.take(SWA_W, component=_SWA)
    assert len(evicted) == SWA_W                    # pin gone -> window evictable
    engine.recycle(evicted, component=_SWA)


def test_swa_insert_without_full_path_is_benign_and_recycles(env):
    """FULL_PATH_MISSING (Full path evicted/absent under a pending SWA publish)
    must cost the window, not the task: insert() warns, radixshmem auto-recycles
    the whole batch, and the pool is whole again."""
    engine, _server = _make_swa_engine(env, "/cers_swa_orphan", swa_slots=SWA_W)
    seq = FakeSeq(block_hashes=_hashes(44, 20), tokens_per_block=16)

    swa_slots = engine.take(SWA_W, component=_SWA)
    assert len(swa_slots) == SWA_W
    # No Full path published: refused with FULL_PATH_MISSING, not raised.
    engine.insert(seq, swa_slots, num_insert_blocks=20, component=_SWA)

    again = engine.take(SWA_W, component=_SWA)
    assert len(again) == SWA_W                      # auto-recycled, none leaked
    engine.recycle(again, component=_SWA)


def test_swa_window_blocks_one_stores_a_single_slot_window(env):
    """W comes from the region's config, not a constant: window_blocks=1 (the
    SGLang DSv4 shape, window inside one page) publishes one-slot windows."""
    engine, _server = _make_swa_engine(env, "/cers_swa_w1", window_blocks=1)
    seq = FakeSeq(block_hashes=_hashes(45, 20), tokens_per_block=16)

    _publish_full(engine, seq, 20)
    _publish_swa(engine, seq, path_end=20, window_blocks=1)

    match = engine.match(seq, component_mask=JOINT_MASK)
    assert match.num_matched_blocks == 20
    assert match.swa_start == 19                    # max(0, 20 - 1)
    assert len(match.swa_slots) == 1
    match.release()


# =============================================================================
# Part 1b — the data plane: SlotStore as the CPU pool, geometry, server process
# =============================================================================


def test_slot_store_pool_is_the_cpu_pool(env):
    """The FULL pool of the server's SlotStore is FlexKV's CPU buffer: slot id
    == block index, stride == block bytes, and a second attach by name (what a
    transfer worker does) sees the same bytes."""
    torch = pytest.importorskip("torch")
    from flexkv.storage.allocator import SlotStoreTensorHandle, slot_store_pool_tensor

    engine, _server = env.make("/cers_store", blocks=64)
    store = engine.client.store
    pool = store.pool(FULL)
    assert int(pool.num_slots) == 64
    assert int(pool.slot_bytes) == SLOT_BYTES          # exact stride, no padding

    slots = engine.take(3)
    for i, slot in enumerate(slots):
        engine.client.slot_view(int(slot))[:] = bytes([i + 1]) * SLOT_BYTES

    tensor = slot_store_pool_tensor(store, FULL, torch.uint8, 64 * SLOT_BYTES)
    assert tensor.shape == (64 * SLOT_BYTES,)
    for i, slot in enumerate(slots):
        block = tensor[int(slot) * SLOT_BYTES:(int(slot) + 1) * SLOT_BYTES]
        assert block.unique().tolist() == [i + 1]

    # A typed view (fp16) over the same pool: 64 blocks x SLOT_BYTES/2 elements.
    typed = slot_store_pool_tensor(store, FULL, torch.float16, 64 * SLOT_BYTES // 2)
    assert typed.dtype == torch.float16 and typed.numel() == 64 * SLOT_BYTES // 2

    handle = SlotStoreTensorHandle(data_name=store.name,
                                   hugepage_path=engine.client.info.hugepage_path,
                                   kind=int(FULL), num_elements=64 * SLOT_BYTES,
                                   dtype=torch.uint8)
    worker_view = handle.get_tensor()               # re-attached by name
    first = int(slots[0])
    assert worker_view[first * SLOT_BYTES:(first + 1) * SLOT_BYTES].unique().tolist() == [1]
    # And writes through the worker's view are what the owner reads.
    worker_view[first * SLOT_BYTES] = 200
    assert bytes(engine.client.slot_view(first)[:1]) == b"\xc8"
    engine.recycle(slots)


def test_slot_align_keeps_the_stride_exact():
    """slot_align is the largest power of two <= 4096 dividing every pool's
    slot bytes, so radixshmem's round-up leaves the stride == slot bytes."""
    align = bootstrap.slot_align_for
    assert align(2359296) == 4096          # Qwen3-8B block: 2^18 x 9
    assert align(149760 * 61) == 256       # 9135360 = 2^8 x 35685
    assert align(12345) == 1               # odd -> byte stride
    assert align(4096 * 7, 1024 * 3) == 1024
    assert align(0, 8192) == 4096          # absent pools do not constrain


def _configs(num_cpu_blocks: int = 64, swa_slots: int = 0):
    torch = pytest.importorskip("torch")
    from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
    model_config = ModelConfig(num_layers=2, num_kv_heads=4, head_size=64,
                               dtype=torch.float16, tp_size=1, dp_size=1)
    cache_config = CacheConfig(tokens_per_block=16, enable_cpu=True, enable_ssd=False,
                               num_cpu_blocks=num_cpu_blocks)
    if swa_slots:
        cache_config.swa = SWAPoolConfig(enabled=True, num_slots=swa_slots,
                                         num_swa_layers=1, bytes_per_token_per_layer=64,
                                         window_blocks=SWA_W)
    return model_config, cache_config


def test_expected_geometry_mirrors_the_storage_engine_layout():
    """One FULL slot is one CPU block exactly as StorageEngine lays it out:
    2 layers x 2 (K,V) x 16 tokens x 4 heads x 64 x fp16 = 32768 B; one SWA
    slot is one SWA page: 1 layer x 16 tokens x 64 B. Counts are not part of
    it: the server plans them from its budget."""
    model_config, cache_config = _configs(num_cpu_blocks=64, swa_slots=16)
    geo = bootstrap.expected_geometry(model_config, cache_config)
    assert geo.tokens_per_block == 16 and geo.full_slot_bytes == 32768
    assert geo.has_swa and geo.swa_slot_bytes == 1024 and geo.swa_window_blocks == SWA_W
    assert geo.slot_align == 1024                  # gcd power of two of 32768 and 1024
    spec = geo.to_shmradix().to_dict()
    assert spec["block_size"] == 16 and spec["slot_align"] == 1024
    assert spec["pools"]["full"] == {"slot_bytes": 32768, "num_slots": 0}
    assert spec["pools"]["swa"] == {"slot_bytes": 1024, "num_slots": 0, "window_blocks": SWA_W}
    # Without SWA the pool is absent from what the server is asked for.
    model_config, cache_config = _configs(num_cpu_blocks=64)
    spec = bootstrap.expected_geometry(model_config, cache_config).to_shmradix().to_dict()
    assert set(spec["pools"]) == {"full"}


def test_register_chunk_is_the_servers_unless_pinned():
    """FlexKV brings no RHT registration chunk of its own: the geometry carries
    0 and the server's --register-chunk-tokens decides. A pinned value travels
    verbatim. Tokens become blocks by radixshmem's rule: tokens // block_size,
    at least 1, 0 = unaligned."""
    model_config, cache_config = _configs(num_cpu_blocks=64)
    geo = bootstrap.expected_geometry(model_config, cache_config)
    assert geo.register_chunk_tokens == 0
    assert geo.to_shmradix().to_dict()["register_chunk_tokens"] == 0
    assert "register_chunk" not in geo.describe()
    pinned = dataclasses.replace(geo, register_chunk_tokens=2048)
    assert pinned.to_shmradix().to_dict()["register_chunk_tokens"] == 2048
    assert "register_chunk_tokens=2048" in pinned.describe()
    assert bootstrap.register_chunk_blocks(4096, 16) == 256
    assert bootstrap.register_chunk_blocks(4096, 4) == 1024
    assert bootstrap.register_chunk_blocks(100, 64) == 1
    assert bootstrap.register_chunk_blocks(0, 16) == 0


def test_client_brings_the_geometry_and_adopts_the_counts(env):
    """The operator's server knows only its budget; FlexKV's first client hands
    it the slot shape, the server plans the counts, `check_geometry` verifies
    the regions and `adopt_geometry` takes the counts into CacheConfig. A
    different geometry is refused, not papered over."""
    model_config, cache_config = _configs(num_cpu_blocks=64, swa_slots=16)
    geo = bootstrap.expected_geometry(model_config, cache_config)
    name = f"/geo{os.getpid()}"
    data_bytes = 64 * geo.full_slot_bytes + 16 * geo.swa_slot_bytes
    env.server(shmradix.ServerConfig(name=name, data_bytes=data_bytes,
                                     swa_ratio=16 * geo.swa_slot_bytes / data_bytes,
                                     register_chunk_tokens=2048,  # the operator's, not FlexKV's
                                     prefault=False))       # slot_align comes with the geometry
    client = bootstrap.attach_radix_client(name, geometry=geo, timeout_s=60)
    try:
        bootstrap.check_geometry(client, geo, "test")
        pools = client.geometry["pools"]
        assert pools["full"]["num_slots"] == 64 and pools["swa"]["num_slots"] == 16
        assert int(client.store.pool(FULL).slot_bytes) == geo.full_slot_bytes    # exact stride
        assert int(client.store.pool(_SWA).slot_bytes) == geo.swa_slot_bytes
        # cpu_cache_gb's placeholders give way to the server's counts; the RHT
        # registration chunk comes along: 2048 tokens = 128 blocks of 16
        assert client.geometry["register_chunk_tokens"] == 2048
        cache_config.num_cpu_blocks, cache_config.swa.num_slots = 7, 3
        assert bootstrap.adopt_geometry(cache_config, client, "test") == {
            "full": 64, "swa": 16, "register_chunk_tokens": 2048, "register_chunk_blocks": 128}
        assert cache_config.num_cpu_blocks == 64 and cache_config.swa.num_slots == 16
        # a chunk FlexKV pinned differently from the server's fails closed too
        with pytest.raises(ValueError, match="register_chunk_tokens"):
            bootstrap.check_geometry(
                client, dataclasses.replace(geo, register_chunk_tokens=4096), "test")
        assert bootstrap.radix_cluster_rank(client) == 0 and client.info.world_size == 1
        # the connectors' prefetch gate asks the server the same question
        assert bootstrap.radix_server_is_distributed(name, timeout_s=30) is False
        # another expectation against the same regions fails closed
        cache_config.tokens_per_block = 32
        with pytest.raises(ValueError, match="tokens_per_block"):
            bootstrap.check_geometry(
                client, bootstrap.expected_geometry(model_config, cache_config), "test")
        # ...and a second client bringing another geometry is refused by the server
        for other in (dataclasses.replace(geo, tokens_per_block=32),
                      dataclasses.replace(geo, register_chunk_tokens=4096)):
            with pytest.raises(ValueError, match="another geometry"):
                bootstrap.attach_radix_client(name, geometry=other, timeout_s=60)
    finally:
        client.close()


def test_attach_waits_for_a_late_server(env):
    """The operator may start the radix-server after the engine: the attach
    retries until the socket answers, brings the geometry and waits for
    ready. No server at all fails with the command that starts one."""
    name = f"/late{os.getpid()}"
    cfg, geo = _server_config(name, blocks=32, tokens_per_block=4)
    _sweep_region(cfg.name, cfg.resolved_data_name)
    started = threading.Event()

    def _start_later():
        time.sleep(1.5)
        env._stack.append(shmradix.RadixServer(cfg).start().close)
        started.set()

    threading.Thread(target=_start_later, daemon=True).start()
    t0 = time.monotonic()
    client = bootstrap.attach_radix_client(name, geometry=geo, timeout_s=60)
    try:
        assert started.is_set() and time.monotonic() - t0 >= 1.0
        assert client.info.mode == "ready" and client.info.data_plane
        assert int(client.mempool_total()) == 32
        assert bootstrap.radix_cluster_rank(client) == 0
    finally:
        client.close()
    with pytest.raises(TimeoutError, match="radix-server --name"):
        bootstrap.attach_radix_client(f"/nobody{os.getpid()}", geometry=geo, timeout_s=2)


# =============================================================================
# Part 1c — FLEXKV_RADIXSHMEM_SERVER_NAME, the one radixshmem-mode setting
# (shm_radix_bootstrap.radix_server_name); everything else about the attach
# is a fixed default.
# =============================================================================


def test_radix_server_name_follows_the_env(monkeypatch):
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "radixshmem_server_name", "/flexkv")
    assert bootstrap.radix_server_name() == "/flexkv"
    assert bootstrap.default_endpoint("/flexkv") == "unix:///dev/shm/flexkv.sock"
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "radixshmem_server_name", "/prod/kv")
    assert bootstrap.radix_server_name() == "/prod/kv"
    assert bootstrap.default_endpoint("/prod/kv") == "unix:///dev/shm/prod_kv.sock"
    # the fixed defaults stay consistent with each other
    assert bootstrap.READY_TIMEOUT_S > 0 and bootstrap.PREFETCH_TIMEOUT_MS > 0
    assert 0 < bootstrap.PREFETCH_MAX_INFLIGHT < bootstrap.MAX_OUTSTANDING


@pytest.mark.parametrize("name", ["kv", "/a b", "/"])
def test_radix_server_name_rejects_malformed_names(monkeypatch, name):
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "radixshmem_server_name", name)
    with pytest.raises(ValueError, match="FLEXKV_RADIXSHMEM_SERVER_NAME"):
        bootstrap.radix_server_name()


# =============================================================================
# Part 2 — planning on the radixshmem backend (`RadixShmemCacheEngine._plan_get`,
# `_plan_prefetch`, `_plan_put`, and the abort path of their handles), plus
# KVTaskEngine's handling of a job-backed prefetch task. Synthetic matches, no
# region.
# =============================================================================

TOKENS_PER_BLOCK = 16


class FakeJob:
    """Stand-in for `shmradix.PullJob`: what `_plan_prefetch` reads on return
    (`local_hit`, `planned_hit`) and what `KVTaskEngine` polls."""

    def __init__(self, local_hit: int, planned_hit: int, job_id: int = 7):
        self.local_hit = local_hit
        self.planned_hit = planned_hit
        self.job_id = job_id
        self.cancelled = False
        self._result = None

    def done(self) -> bool:
        return self._result is not None

    def wait(self, timeout=None):
        if self.cancelled:
            raise RuntimeError("cancelled")
        if self._result is None:
            raise TimeoutError("still running")
        return self._result

    def cancel(self) -> None:
        self.cancelled = True

    def complete(self, common_hit: int, remote_blocks: int, remote_bytes: int = 0,
                 source_rank: int = 1) -> None:
        self._result = SimpleNamespace(common_hit=common_hit, remote_blocks=remote_blocks,
                                       remote_bytes=remote_bytes, source_rank=source_rank,
                                       finalize=lambda: None)


def _global_cache_engine():
    """Build a `RadixShmemCacheEngine` whose CPU tier is a plain `CacheEngineAccel`.

    The radixshmem planners run for real; only the tier is swapped (no region
    needed) and its tree side is then stubbed by `_force_radixshmem`.

    Imported here rather than at module scope: `flexkv.cache.__init__` pulls in
    `flexkv.c_ext` (libcudart), which Parts 1 and 3 deliberately avoid.
    """
    try:
        import torch

        from flexkv.cache.cache_engine import GlobalCacheEngine
        from flexkv.cache.radix_shmem_planner import RadixShmemCacheEngine
        from flexkv.common.config import CacheConfig, ModelConfig
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"GlobalCacheEngine unavailable (needs CUDA + flexkv.c_ext): {exc}")

    class _PlannerOnAccelTier(RadixShmemCacheEngine):
        def _build_cpu_cache_engine(self, cache_config, event_collector):
            return GlobalCacheEngine._build_cpu_cache_engine(
                self, cache_config, event_collector)

    model_config = ModelConfig(
        num_layers=2, num_kv_heads=4, head_size=64,
        dtype=torch.float16, tp_size=1, dp_size=1,
    )
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        enable_cpu=True, enable_ssd=False, enable_remote=False,
        num_cpu_blocks=256,
    )
    return _PlannerOnAccelTier(cache_config, model_config)


def _local_match(slots, finalize=None) -> ShmRadixMatch:
    slots = np.asarray(slots, dtype=np.int64)
    return ShmRadixMatch(num_matched_blocks=len(slots), local_slots=slots, finalize=finalize)


def _force_radixshmem(engine, cpu_result: ShmRadixMatch, *, prefetch_job=None,
                      peer_enabled=None) -> None:
    """Stub the tree side of a `_global_cache_engine()`.

    The tier keeps its real mempool (so `take` returns honest slot ids) but the
    tree side is faked: the synthetic match names no real prefix, and the tier
    here is a `CacheEngineAccel`, whose `insert` signature is a different one.
    Records what a planner published (`engine.inserted_pools`) and what it
    handed back (`engine.aborted_slots`), and the prefetch calls it made
    (`engine.prefetch_calls`).
    """
    engine._match_cpu = (  # type: ignore[method-assign]
        lambda *args, **kwargs: cpu_result
    )
    tier = engine.cpu_cache_engine
    tier.peer_enabled = (prefetch_job is not None) if peer_enabled is None else peer_enabled
    prefetch_calls = []

    def _prefetch(sequence_meta, **kwargs):
        prefetch_calls.append(kwargs)
        return prefetch_job

    tier.prefetch = _prefetch                       # type: ignore[attr-defined]
    inserted = []
    aborted = []

    def _insert(sequence_meta, physical_block_ids, num_insert_blocks,
                component=None, _sink=inserted):
        _sink.append((num_insert_blocks, np.asarray(physical_block_ids)))

    def _recycle(physical_block_ids, component=None,
                 _orig=tier.recycle, _sink=aborted):
        _sink.append(np.asarray(physical_block_ids))
        _orig(np.asarray(physical_block_ids))

    tier.insert = _insert                           # type: ignore[method-assign]
    tier.recycle = _recycle                         # type: ignore[method-assign]
    engine.inserted_pools = inserted                # type: ignore[attr-defined]
    engine.aborted_slots = aborted                  # type: ignore[attr-defined]
    engine.prefetch_calls = prefetch_calls          # type: ignore[attr-defined]


def _fake_request(num_blocks: int, base: int = 0):
    """(token_ids, token_mask, slot_mapping) for a fully-masked `num_blocks` window.

    `base` offsets the token ids so two requests name distinct sequences."""
    num_tokens = num_blocks * TOKENS_PER_BLOCK
    token_ids = np.arange(base, base + num_tokens, dtype=np.int64)
    token_mask = np.ones(num_tokens, dtype=np.bool_)
    # GPU blocks 1000.. so they can't be confused with CPU slot ids.
    slot_mapping = (
        np.repeat(np.arange(1000, 1000 + num_blocks), TOKENS_PER_BLOCK)
        * TOKENS_PER_BLOCK
        + np.tile(np.arange(TOKENS_PER_BLOCK), num_blocks)
    ).astype(np.int64)
    return token_ids, token_mask, slot_mapping


def _ops_by_type(graph):
    ops = {}
    for op in graph._op_map.values():
        ops.setdefault(op.transfer_type, []).append(op)
    return ops


def _run_get(engine, num_blocks: int, cpu_result: ShmRadixMatch, *, prefetch=False,
             prefetch_job=None, peer_enabled=None):
    """Call get() through the radixshmem planners with a forced match result."""
    from flexkv.cache.cache_engine import DEFAULT_CACHE_STRATEGY
    _force_radixshmem(engine, cpu_result, prefetch_job=prefetch_job,
                      peer_enabled=peer_enabled)
    token_ids, token_mask, slot_mapping = _fake_request(num_blocks)
    strategy = copy.deepcopy(DEFAULT_CACHE_STRATEGY)
    if prefetch:
        strategy.ignore_gpu = True
        strategy.ignore_gds = True
    graph, return_mask, callback, _op_cbs, _end = engine.get(
        request_id=1,
        token_ids=token_ids,
        token_mask=token_mask,
        slot_mapping=slot_mapping,
        dp_client_id=0,
        temp_cache_strategy=strategy,
    )
    engine.get_callback = callback                  # type: ignore[attr-defined]
    return graph, _ops_by_type(graph), return_mask


def test_local_hit_plans_one_h2d_and_releases_the_pin():
    """A local CPU hit is exactly one H2D read straight from the hit's slots;
    the match pin lives until the graph completes."""
    engine = _global_cache_engine()
    released = []
    cpu_slots = np.arange(40, 44, dtype=np.int64)
    free_before = engine.cpu_cache_engine.mempool.num_free_blocks
    graph, ops, return_mask = _run_get(
        engine, 4, _local_match(cpu_slots, finalize=lambda: released.append(1)))
    assert set(ops) == {TransferType.H2D}
    h2d = ops[TransferType.H2D][0]
    np.testing.assert_array_equal(h2d.src_block_ids, cpu_slots)
    np.testing.assert_array_equal(h2d.dst_block_ids, np.arange(1000, 1004))
    assert return_mask.sum() == 4 * TOKENS_PER_BLOCK
    # No staging taken, nothing to publish, nothing to give back.
    assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before
    assert released == []
    engine.get_callback()                           # type: ignore[attr-defined]
    assert released == [1]
    assert engine.inserted_pools == []              # type: ignore[attr-defined]
    assert engine.aborted_slots == []               # type: ignore[attr-defined]


def test_partial_hit_restores_the_prefix_only():
    """The hit ends inside the window: H2D covers the hit, the mask says so,
    and nothing past it is planned (the miss is recomputed)."""
    engine = _global_cache_engine()
    cpu_slots = np.arange(20, 22, dtype=np.int64)
    _graph, ops, return_mask = _run_get(engine, 4, _local_match(cpu_slots))
    h2d = ops[TransferType.H2D][0]
    np.testing.assert_array_equal(h2d.src_block_ids, cpu_slots)
    np.testing.assert_array_equal(h2d.dst_block_ids, np.arange(1000, 1002))
    assert bool(return_mask[:2 * TOKENS_PER_BLOCK].all())
    assert not bool(return_mask[2 * TOKENS_PER_BLOCK:].any())


def test_miss_is_an_empty_plan_with_the_pin_dropped():
    engine = _global_cache_engine()
    released = []
    graph, ops, return_mask = _run_get(
        engine, 4, _local_match([], finalize=lambda: released.append(1)))
    assert ops == {}
    assert not bool(return_mask.any())
    assert released == [1]                          # dropped at plan time
    engine.get_callback()                           # type: ignore[attr-defined]
    assert engine.get_callback.prefetch_job is None  # type: ignore[attr-defined]


def test_prefetch_starts_a_peer_pull():
    """A prefetch on a clustered tier is `RadixClient.pull_async`: the plan has no
    ops, the job rides on the callback handle, and the mask is the planned pull
    [local hit, planned hit)."""
    engine = _global_cache_engine()
    job = FakeJob(local_hit=1, planned_hit=4)
    graph, ops, return_mask = _run_get(
        engine, 4, _local_match(np.arange(20, 21)), prefetch=True, prefetch_job=job)
    assert ops == {}
    callback = engine.get_callback                  # type: ignore[attr-defined]
    assert callback.prefetch_job is job
    assert (callback.prefetch_local_hit_blocks, callback.prefetch_planned_hit_blocks) == (1, 4)
    assert not bool(return_mask[:TOKENS_PER_BLOCK].any())
    assert bool(return_mask[TOKENS_PER_BLOCK:4 * TOKENS_PER_BLOCK].all())
    # Full|SWA mask only when the request is SWA-aware; plain prefetch is FULL.
    (call,) = engine.prefetch_calls                 # type: ignore[attr-defined]
    assert call["component_mask"] == _engine_mod.COMPONENT_MASK_FULL
    assert call["query_end"] == 4
    assert call["timeout_ms"] == bootstrap.PREFETCH_TIMEOUT_MS


def test_prefetch_without_peers_is_an_empty_plan():
    engine = _global_cache_engine()
    _graph, ops, return_mask = _run_get(
        engine, 4, _local_match(np.arange(20, 22)), prefetch=True, peer_enabled=False)
    assert ops == {}
    assert not bool(return_mask.any())
    assert engine.get_callback.prefetch_job is None  # type: ignore[attr-defined]
    assert engine.prefetch_calls == []              # type: ignore[attr-defined]


def test_prefetch_backpressure_skips_the_peer_walk():
    """Too many pulls in flight: no pull_async, so the client never blocks."""
    engine = _global_cache_engine()
    limit = bootstrap.PREFETCH_MAX_INFLIGHT
    engine._prefetch_jobs = [FakeJob(0, 4) for _ in range(limit)]   # none done
    _graph, ops, return_mask = _run_get(
        engine, 4, _local_match([]), prefetch=True, prefetch_job=FakeJob(0, 4))
    assert ops == {} and not bool(return_mask.any())
    assert engine.prefetch_calls == []              # type: ignore[attr-defined]
    assert engine.get_callback.prefetch_job is None  # type: ignore[attr-defined]


def _bare_task_engine(cache_config):
    """A `KVTaskEngine` without transfer handles: enough of the task table for
    the job polling paths (the same `__new__` trick the fallback managers use)."""
    from flexkv.kvtask import KVTaskEngine
    mgr = KVTaskEngine.__new__(KVTaskEngine)
    mgr.cache_config = cache_config
    mgr.tasks = {}
    mgr.prefetch_jobs = {}
    mgr.graph_to_task = {}
    mgr.transfer_handles = []
    mgr.uncompleted_ops = {}
    mgr.uncompleted_op_results = {}
    mgr.uncompleted_graphs = {}
    mgr.required_completed_count = 0
    return mgr


def _prefetch_task(task_id, job, num_blocks, local_hit, planned_hit):
    from flexkv.common.transfer import TransferOpGraph
    from flexkv.kvtask import KVTask, TaskStatus, TaskType
    n = num_blocks * TOKENS_PER_BLOCK
    return KVTask(
        task_id=task_id, task_type=TaskType.PREFETCH, task_end_op_id=-1,
        task_end_op_finished=False, status=TaskStatus.RUNNING,
        token_ids=np.arange(n), slot_mapping=np.zeros(n, dtype=np.int64),
        token_mask=np.ones(n, dtype=np.bool_),
        graph=TransferOpGraph.create_empty_graph(),
        return_mask=np.zeros(n, dtype=np.bool_), callback=None, op_callback_dict={},
        prefetch_job=job, prefetch_local_hit_blocks=local_hit,
        prefetch_planned_hit_blocks=planned_hit)


def test_task_engine_completes_a_prefetch_from_its_job():
    """The PREFETCH task has an empty graph; `_update_tasks` polls the job and,
    once done, reports the pulled range and completes the task."""
    try:
        from flexkv.common.config import CacheConfig
        from flexkv.kvtask import TaskStatus
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"kvtask unavailable (needs flexkv.c_ext): {exc}")
    mgr = _bare_task_engine(CacheConfig(tokens_per_block=TOKENS_PER_BLOCK, num_cpu_blocks=64))

    job = FakeJob(local_hit=1, planned_hit=4)
    task = _prefetch_task(1, job, num_blocks=4, local_hit=1, planned_hit=4)
    mgr.tasks[1] = task
    mgr.prefetch_jobs[1] = job
    mgr._process_empty_graph(1)                      # job pending: stays RUNNING
    mgr._poll_prefetch_jobs()
    assert task.status == TaskStatus.RUNNING and 1 in mgr.prefetch_jobs

    job.complete(common_hit=4, remote_blocks=3, remote_bytes=3 * SLOT_BYTES)
    mgr._poll_prefetch_jobs()
    assert task.status == TaskStatus.COMPLETED
    assert 1 not in mgr.prefetch_jobs
    assert not bool(task.return_mask[:TOKENS_PER_BLOCK].any())
    assert bool(task.return_mask[TOKENS_PER_BLOCK:].all())

    # A shortfall (transfer refused, evicted before publish) narrows the mask.
    job2 = FakeJob(local_hit=1, planned_hit=4)
    task2 = _prefetch_task(2, job2, num_blocks=4, local_hit=1, planned_hit=4)
    mgr.tasks[2] = task2
    mgr.prefetch_jobs[2] = job2
    job2.complete(common_hit=2, remote_blocks=1)
    mgr._process_empty_graph(2)                      # done at first look
    assert task2.status == TaskStatus.COMPLETED
    assert bool(task2.return_mask[TOKENS_PER_BLOCK:2 * TOKENS_PER_BLOCK].all())
    assert not bool(task2.return_mask[2 * TOKENS_PER_BLOCK:].any())

    # Nothing pulled at all: an empty mask, still a completed (not failed) task.
    job3 = FakeJob(local_hit=1, planned_hit=4)
    task3 = _prefetch_task(3, job3, num_blocks=4, local_hit=1, planned_hit=4)
    mgr.tasks[3] = task3
    mgr.prefetch_jobs[3] = job3
    job3.complete(common_hit=1, remote_blocks=0)
    mgr._poll_prefetch_jobs()
    assert task3.status == TaskStatus.COMPLETED
    assert not bool(task3.return_mask.any())


def test_task_engine_cancel_hands_the_job_back():
    """Cancelling a job-backed prefetch cancels the job (the pull finishes in
    the background) and never touches it again."""
    try:
        from flexkv.common.config import CacheConfig
        from flexkv.kvtask import TaskStatus
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"kvtask unavailable (needs flexkv.c_ext): {exc}")
    mgr = _bare_task_engine(CacheConfig(tokens_per_block=TOKENS_PER_BLOCK, num_cpu_blocks=64))
    job = FakeJob(local_hit=0, planned_hit=4)
    task = _prefetch_task(5, job, num_blocks=4, local_hit=0, planned_hit=4)
    mgr.tasks[5] = task
    mgr.prefetch_jobs[5] = job
    mgr._cancel_task(5)
    assert job.cancelled
    assert task.status == TaskStatus.CANCELLED
    assert 5 not in mgr.prefetch_jobs and 5 not in mgr.tasks
    job.complete(common_hit=4, remote_blocks=4)
    mgr._poll_prefetch_jobs()                        # nothing left to do


# ---- PUT planning ----

def _run_put(engine, num_blocks: int, cpu_result: ShmRadixMatch):
    """Call put() through `_plan_put` with a forced match result."""
    _force_radixshmem(engine, cpu_result)
    token_ids, token_mask, slot_mapping = _fake_request(num_blocks)
    graph, return_mask, callback, _op_cbs, _end = engine.put(
        request_id=2,
        token_ids=token_ids,
        token_mask=token_mask,
        slot_mapping=slot_mapping,
        dp_client_id=0,
    )
    engine.put_callback = callback                   # type: ignore[attr-defined]
    return graph, _ops_by_type(graph), return_mask


def test_put_with_no_match_stores_the_whole_window():
    """Cold start: nothing cached, so D2H covers every block and the span publishes."""
    engine = _global_cache_engine()
    graph, ops, return_mask = _run_put(engine, num_blocks=4,
                                       cpu_result=_local_match([]))

    op_d2h = ops[TransferType.D2H][0]
    assert op_d2h.src_block_ids.size == 4          # nothing skipped
    assert op_d2h.dst_block_ids.size == 4
    assert bool(return_mask.all())

    engine.put_callback()                           # graph completion
    # One insert, of the whole window, ending at block 4.
    assert [(n, len(s)) for n, s in engine.inserted_pools] == [(4, 4)]
    assert engine.aborted_slots == []


def test_put_skips_the_cached_prefix():
    """A partial CPU match: D2H moves only the blocks past it, and they publish."""
    engine = _global_cache_engine()
    released = []
    cached = np.arange(20, 23, dtype=np.int64)      # 3 of 5 blocks already in CPU
    graph, ops, return_mask = _run_put(
        engine, num_blocks=5,
        cpu_result=_local_match(cached, finalize=lambda: released.append(1)))

    op_d2h = ops[TransferType.D2H][0]
    assert op_d2h.src_block_ids.size == 2
    # GPU blocks are 1000.., so the skipped prefix is visible in the ids read.
    assert op_d2h.src_block_ids.tolist() == [1003, 1004]
    # The staged slots are fresh — never the ones the match already holds.
    assert not set(op_d2h.dst_block_ids.tolist()) & set(cached.tolist())
    # Only the newly stored blocks come back as stored.
    assert not bool(return_mask[:3 * TOKENS_PER_BLOCK].any())
    assert bool(return_mask[3 * TOKENS_PER_BLOCK:].all())

    assert released == []                           # pinned while D2H runs
    engine.put_callback()
    # The span ends at block 5, and carries the 2 new blocks only.
    assert [(n, len(s)) for n, s in engine.inserted_pools] == [(5, 2)]
    assert engine.aborted_slots == []
    assert released == [1]                          # released after the publish


def test_put_with_fully_cached_window_does_nothing():
    """A match covering the window ends the PUT: nothing to store, nothing to arm."""
    engine = _global_cache_engine()
    released = []
    graph, ops, return_mask = _run_put(
        engine, num_blocks=3,
        cpu_result=_local_match(np.arange(20, 23), finalize=lambda: released.append(1)))
    assert ops == {}
    assert not bool(return_mask.any())
    assert engine.inserted_pools == []
    assert engine.aborted_slots == []
    assert released == [1]


# ---- abort: the plan was cancelled before its graph launched ----

def test_get_abort_drops_the_pin():
    """A cancelled GET never runs its H2D; abort releases the match pin, and a
    late completion on the consumed handle does nothing more."""
    engine = _global_cache_engine()
    released = []
    _run_get(engine, 4, _local_match(np.arange(40, 44),
                                     finalize=lambda: released.append(1)))
    assert released == []
    engine.get_callback.abort()                     # type: ignore[attr-defined]
    assert released == [1]
    engine.get_callback()                           # type: ignore[attr-defined]
    assert released == [1]
    assert engine.inserted_pools == []              # type: ignore[attr-defined]


def test_put_planning_failure_returns_the_taken_slots_and_drops_the_pin():
    """An exception after the FULL slots were taken (here the SWA take blows up)
    must not leak them or the match pin: the planner recycles, releases and
    re-raises. Before this, a server planning fewer SWA slots than the window
    leaked FULL slots on every PUT."""
    engine = _global_cache_engine()
    released = []
    _force_radixshmem(engine, _local_match(np.arange(20, 22),
                                           finalize=lambda: released.append(1)))
    tier = engine.cpu_cache_engine
    free_before = tier.mempool.num_free_blocks
    real_take = tier.take

    def _take(num_required_blocks, component=None, **kwargs):
        if component is not None:
            raise RuntimeError("SWA pool exploded")
        return real_take(num_required_blocks=num_required_blocks, **kwargs)

    tier.take = _take                                            # type: ignore[method-assign]
    engine.swa_op_constructor = SimpleNamespace(enabled=True)    # type: ignore[assignment]
    engine.cache_config.swa = SimpleNamespace(window_blocks=2)
    token_ids, token_mask, slot_mapping = _fake_request(5)
    with pytest.raises(RuntimeError, match="SWA pool exploded"):
        engine.put(request_id=3, token_ids=token_ids, token_mask=token_mask,
                   slot_mapping=slot_mapping, dp_client_id=0)
    assert tier.mempool.num_free_blocks == free_before          # the 3 FULL slots came back
    assert [len(s) for s in engine.aborted_slots] == [3]         # type: ignore[attr-defined]
    assert released == [1]                                       # pin dropped
    assert engine.inserted_pools == []                           # type: ignore[attr-defined]


def test_put_abort_returns_the_staged_slots_and_drops_the_pin():
    """A cancelled PUT never runs its D2H: nothing is published, the staged
    slots go back to the mempool and the match pin is released."""
    engine = _global_cache_engine()
    released = []
    free_before = engine.cpu_cache_engine.mempool.num_free_blocks
    _graph, ops, _mask = _run_put(
        engine, num_blocks=5,
        cpu_result=_local_match(np.arange(20, 23), finalize=lambda: released.append(1)))
    staged = ops[TransferType.D2H][0].dst_block_ids
    assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before - 2
    assert released == []

    engine.put_callback.abort()                     # type: ignore[attr-defined]
    assert engine.inserted_pools == []              # type: ignore[attr-defined]
    assert [s.tolist() for s in engine.aborted_slots] == [staged.tolist()]
    assert engine.cpu_cache_engine.mempool.num_free_blocks == free_before
    assert released == [1]
    engine.put_callback()                           # type: ignore[attr-defined]
    assert engine.inserted_pools == []              # consumed: no late publish


def test_prefetch_abort_leaves_the_job_alone():
    """Cancelling a job-backed prefetch is KVTaskEngine's business (it cancels
    the job); the plan itself holds nothing to roll back."""
    engine = _global_cache_engine()
    job = FakeJob(local_hit=0, planned_hit=4)
    _run_get(engine, 4, _local_match([]), prefetch=True, prefetch_job=job)
    engine.get_callback.abort()                     # type: ignore[attr-defined]
    assert not job.cancelled
    assert engine.aborted_slots == []               # type: ignore[attr-defined]


# =============================================================================
# Part 2b — SWA planning on a real region (needs c_ext for GlobalCacheEngine)
# =============================================================================

SWA_ENV_BLOCKS = 64


@contextlib.contextmanager
def _swa_global_engine(swa_slots: int = 2 * SWA_W,
                       num_blocks: int = SWA_ENV_BLOCKS,
                       window_blocks: int = SWA_W):
    """A `RadixShmemCacheEngine` on a real radix-server with the SWA component.

    `cache_config.swa` + `enable_swa_transfer` turn on `swa_op_constructor`, and
    the same config drives the bootstrap, so this also covers the
    shm_radix_bootstrap side of the design. The pool is small on purpose --
    pin-release is asserted through exact take() counts.
    """
    try:
        import torch
        from flexkv.cache.radix_shmem_planner import RadixShmemCacheEngine
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"RadixShmemCacheEngine unavailable (needs CUDA + flexkv.c_ext): {exc}")

    from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig

    server_name = f"/swaplanner{os.getpid()}"
    saved = {"enable_radixshmem": GLOBAL_CONFIG_FROM_ENV.enable_radixshmem,
             "radixshmem_server_name": GLOBAL_CONFIG_FROM_ENV.radixshmem_server_name}
    GLOBAL_CONFIG_FROM_ENV.enable_radixshmem = True
    GLOBAL_CONFIG_FROM_ENV.radixshmem_server_name = server_name

    server = None
    engine = None
    try:
        cache_config = CacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            enable_cpu=True, enable_ssd=False, enable_remote=False,
            num_cpu_blocks=num_blocks,
        )
        cache_config.swa = SWAPoolConfig(enabled=True, num_slots=swa_slots,
                                         num_swa_layers=1,
                                         bytes_per_token_per_layer=64,
                                         window_blocks=window_blocks)
        cache_config.enable_swa_transfer = True
        model_config = ModelConfig(num_layers=2, num_kv_heads=4, head_size=64,
                                   dtype=torch.float16,
                                   tp_size=1, dp_size=1)
        # The operator's server: a budget sized so the planned counts are the
        # test's; the planner's client brings the geometry.
        geo = bootstrap.expected_geometry(model_config, cache_config)
        data_bytes = num_blocks * geo.full_slot_bytes + swa_slots * geo.swa_slot_bytes
        cfg = shmradix.ServerConfig(name=server_name, data_bytes=data_bytes,
                                    swa_ratio=swa_slots * geo.swa_slot_bytes / data_bytes,
                                    prefault=False)
        _sweep_region(cfg.name, cfg.resolved_data_name)
        server = shmradix.RadixServer(cfg).start()
        engine = RadixShmemCacheEngine(cache_config, model_config)
        assert engine.swa_op_constructor.enabled, \
            "SWA gate should be on: enable_swa_transfer + radixshmem swa_enabled"
        yield engine
    finally:
        if engine is not None and engine.cpu_cache_engine is not None:
            engine.cpu_cache_engine.close()
        if server is not None:
            server.close()
        for name, value in saved.items():
            setattr(GLOBAL_CONFIG_FROM_ENV, name, value)


def _split_swa(ops_of_type):
    full = [op for op in ops_of_type if not getattr(op, "is_swa", False)]
    swa = [op for op in ops_of_type if getattr(op, "is_swa", False)]
    return full, swa


def _real_seq(token_ids):
    from flexkv.common.block import SequenceMeta
    return SequenceMeta(token_ids=np.asarray(token_ids).copy(),
                        tokens_per_block=TOKENS_PER_BLOCK)


def test_put_then_get_swa_roundtrip_on_real_region():
    """PUT: the graph carries a 20-block Full D2H plus an 8-slot is_swa D2H,
    both on the task-end barrier; before the completion callback a joint query
    sees nothing; the callback publishes insert(FULL) then insert(SWA) and
    releases the query. GET(swa_aware): one graph with a 20-block Full H2D plus
    the 8-slot SWA H2D, both on the barrier; the query pin lives until the
    callback and is gone after it."""
    with _swa_global_engine() as engine:
        cpu = engine.cpu_cache_engine
        num_total = SWA_ENV_BLOCKS

        # Record the publish order without breaking the real inserts.
        published = []
        real_insert = cpu.insert

        def _recording_insert(*args, **kwargs):
            published.append(kwargs.get("component"))
            return real_insert(*args, **kwargs)

        cpu.insert = _recording_insert              # type: ignore[method-assign]

        token_ids, token_mask, slot_mapping = _fake_request(20)
        graph, put_mask, put_cb, _op_cbs, put_end = engine.put(
            request_id=7, token_ids=token_ids, token_mask=token_mask,
            slot_mapping=slot_mapping, dp_client_id=0)

        full_d2h, swa_d2h = _split_swa(_ops_by_type(graph)[TransferType.D2H])
        assert len(full_d2h) == 1 and len(swa_d2h) == 1
        assert full_d2h[0].dst_block_ids.size == 20
        assert swa_d2h[0].src_block_ids.size == SWA_W
        assert swa_d2h[0].dst_block_ids.size == SWA_W
        put_end_preds = set(graph._op_map[put_end].predecessors)
        assert {full_d2h[0].op_id, swa_d2h[0].op_id} <= put_end_preds
        assert bool(put_mask.all())

        pending = cpu.match(_real_seq(token_ids), component_mask=JOINT_MASK)
        assert pending.num_matched_blocks == 0
        pending.release()

        put_cb()                                    # graph completion
        assert published == [shmradix.ComponentType.FULL, _SWA]

        after = cpu.match(_real_seq(token_ids), component_mask=JOINT_MASK)
        assert after.num_matched_blocks == 20
        assert after.swa_start == 12
        assert len(after.swa_slots) == SWA_W
        after.release()

        graph, get_mask, get_cb, _op_cbs, get_end = engine.get(
            request_id=8, token_ids=token_ids, token_mask=token_mask,
            slot_mapping=slot_mapping, dp_client_id=0, swa_aware=True)
        assert int(get_mask.sum()) == 20 * TOKENS_PER_BLOCK

        full_h2d, swa_h2d = _split_swa(_ops_by_type(graph)[TransferType.H2D])
        assert len(full_h2d) == 1 and len(swa_h2d) == 1
        assert full_h2d[0].src_block_ids.size == 20
        assert swa_h2d[0].src_block_ids.tolist() == after.swa_slots.tolist()
        get_end_preds = set(graph._op_map[get_end].predecessors)
        assert {full_h2d[0].op_id, swa_h2d[0].op_id} <= get_end_preds

        held = cpu.take(num_total)
        assert len(held) == num_total - 20
        cpu.recycle(held)

        get_cb()                                    # Full H2D + SWA H2D done
        drained = cpu.take(num_total)
        assert len(drained) == num_total
        cpu.recycle(drained)


def test_put_degrades_to_full_only_when_the_swa_pool_is_exhausted():
    """An empty all-or-none SWA take drops the SWA leg -- no is_swa op, no SWA
    staged insert -- and the Full plan proceeds untouched."""
    with _swa_global_engine(swa_slots=SWA_W) as engine:  # exactly one window
        cpu = engine.cpu_cache_engine

        tok_a, mask_a, sm_a = _fake_request(10)
        _graph, _mask, put_cb_a, _cbs, _end = engine.put(
            request_id=11, token_ids=tok_a, token_mask=mask_a,
            slot_mapping=sm_a, dp_client_id=0)
        put_cb_a()
        pin = cpu.match(_real_seq(tok_a), component_mask=JOINT_MASK)
        assert len(pin.swa_slots) == SWA_W

        tok_b, mask_b, sm_b = _fake_request(10, base=1_000_000)
        graph, put_mask, put_cb_b, _cbs, _end = engine.put(
            request_id=12, token_ids=tok_b, token_mask=mask_b,
            slot_mapping=sm_b, dp_client_id=0)
        full_d2h, swa_d2h = _split_swa(_ops_by_type(graph)[TransferType.D2H])
        assert len(full_d2h) == 1 and swa_d2h == []
        assert bool(put_mask.all())
        put_cb_b()
        pin.release()

        full_only = cpu.match(_real_seq(tok_b))
        assert full_only.num_matched_blocks == 10
        full_only.release()
        joint = cpu.match(_real_seq(tok_b), component_mask=JOINT_MASK)
        assert joint.num_matched_blocks == 0
        joint.release()


def test_get_without_swa_aware_stays_full_only_on_swa_region():
    with _swa_global_engine() as engine:
        token_ids, token_mask, slot_mapping = _fake_request(12)
        _graph, _mask, put_cb, _cbs, _end = engine.put(
            request_id=21, token_ids=token_ids, token_mask=token_mask,
            slot_mapping=slot_mapping, dp_client_id=0)
        put_cb()

        graph, get_mask, get_cb, _cbs, _end = engine.get(
            request_id=22, token_ids=token_ids, token_mask=token_mask,
            slot_mapping=slot_mapping, dp_client_id=0)
        assert int(get_mask.sum()) == 12 * TOKENS_PER_BLOCK
        full_h2d, swa_h2d = _split_swa(_ops_by_type(graph)[TransferType.H2D])
        assert len(full_h2d) == 1 and swa_h2d == []
        get_cb()


def _drive_put(engine, token_ids, token_mask, slot_mapping, request_id):
    """put() + immediate completion; returns (graph, return_mask)."""
    graph, return_mask, cb, _op_cbs, _end = engine.put(
        request_id=request_id, token_ids=token_ids, token_mask=token_mask,
        slot_mapping=slot_mapping, dp_client_id=0)
    cb()
    return graph, return_mask


def test_reput_of_a_fully_cached_prefix_is_an_early_return():
    with _swa_global_engine() as engine:
        tok, mask, sm = _fake_request(10)
        _drive_put(engine, tok, mask, sm, request_id=41)
        graph, return_mask = _drive_put(engine, tok, mask, sm, request_id=42)
        assert _ops_by_type(graph) == {}
        assert not bool(return_mask.any())


def test_put_extension_releases_a_nonempty_match_pin_after_both_publishes():
    with _swa_global_engine() as engine:
        cpu = engine.cpu_cache_engine
        tok10, mask10, sm10 = _fake_request(10)
        _drive_put(engine, tok10, mask10, sm10, request_id=51)

        tok20, mask20, sm20 = _fake_request(20)     # same first 10 blocks
        graph, return_mask, cb, _op_cbs, _end = engine.put(
            request_id=52, token_ids=tok20, token_mask=mask20,
            slot_mapping=sm20, dp_client_id=0)
        full_d2h, swa_d2h = _split_swa(_ops_by_type(graph)[TransferType.D2H])
        assert full_d2h[0].dst_block_ids.size == 10  # only the extension moves
        assert len(swa_d2h) == 1                     # window rides along
        held = cpu.take(SWA_ENV_BLOCKS)
        assert len(held) == SWA_ENV_BLOCKS - 20
        cpu.recycle(held)

        cb()                                        # FULL publish, SWA publish, release
        drained = cpu.take(SWA_ENV_BLOCKS)
        assert len(drained) == SWA_ENV_BLOCKS       # pin gone, all evictable
        cpu.recycle(drained)


def test_swa_get_of_a_shorter_prefix_misses():
    with _swa_global_engine() as engine:
        tok, mask, sm = _fake_request(20)
        _drive_put(engine, tok, mask, sm, request_id=61)

        short = 12 * TOKENS_PER_BLOCK
        graph, get_mask, get_cb, _op_cbs, _end = engine.get(
            request_id=62, token_ids=tok[:short], token_mask=mask[:short],
            slot_mapping=sm[:short], dp_client_id=0, swa_aware=True)
        assert int(get_mask.sum()) == 0
        assert _ops_by_type(graph) == {}
        get_cb()

        graph, get_mask, get_cb, _op_cbs, _end = engine.get(
            request_id=63, token_ids=tok[:short], token_mask=mask[:short],
            slot_mapping=sm[:short], dp_client_id=0)
        assert int(get_mask.sum()) == short
        get_cb()


def test_short_path_put_and_get_use_k_smaller_than_w():
    with _swa_global_engine() as engine:
        cpu = engine.cpu_cache_engine
        tok, mask, sm = _fake_request(5)
        graph, _ = _drive_put(engine, tok, mask, sm, request_id=71)
        _full_d2h, swa_d2h = _split_swa(_ops_by_type(graph)[TransferType.D2H])
        assert swa_d2h[0].src_block_ids.size == 5

        joint = cpu.match(_real_seq(tok), component_mask=JOINT_MASK)
        assert (joint.num_matched_blocks, joint.swa_start,
                len(joint.swa_slots)) == (5, 0, 5)
        joint.release()

        graph, get_mask, get_cb, _op_cbs, _end = engine.get(
            request_id=72, token_ids=tok, token_mask=mask,
            slot_mapping=sm, dp_client_id=0, swa_aware=True)
        assert int(get_mask.sum()) == 5 * TOKENS_PER_BLOCK
        _full_h2d, swa_h2d = _split_swa(_ops_by_type(graph)[TransferType.H2D])
        assert swa_h2d[0].src_block_ids.size == 5
        get_cb()


def test_planner_uses_configured_window_blocks():
    with _swa_global_engine(window_blocks=1) as engine:
        cpu = engine.cpu_cache_engine
        tok, mask, sm = _fake_request(10)
        graph, _ = _drive_put(engine, tok, mask, sm, request_id=81)
        _full_d2h, swa_d2h = _split_swa(_ops_by_type(graph)[TransferType.D2H])
        assert swa_d2h[0].src_block_ids.size == 1

        joint = cpu.match(_real_seq(tok), component_mask=JOINT_MASK)
        assert (joint.num_matched_blocks, joint.swa_start,
                len(joint.swa_slots)) == (10, 9, 1)
        joint.release()

        graph, get_mask, get_cb, _op_cbs, _end = engine.get(
            request_id=82, token_ids=tok, token_mask=mask,
            slot_mapping=sm, dp_client_id=0, swa_aware=True)
        assert int(get_mask.sum()) == 10 * TOKENS_PER_BLOCK
        _full_h2d, swa_h2d = _split_swa(_ops_by_type(graph)[TransferType.H2D])
        assert swa_h2d[0].src_block_ids.size == 1
        get_cb()


# =============================================================================
# Part 3 — opt-in two-node radixshmem cluster over RDMA: prefetch pulls a peer's
# blocks (index walk over RDMA, server-side RDMA READ of the SlotStore bytes),
# then a local match finds them and the bytes are the writer's.
#
# Two spawned processes each run a data-mode RadixServer (distinct data names
# and sockets on one host) and a `CacheEngineRadixShmem` attached to it.
# Gated behind FLEXKV_RUN_RADIX_PEER_TEST=1; needs an ACTIVE RDMA device, a
# shmradix built with RDMA + etcd + mooncake, and an etcd
# (FLEXKV_TEST_RADIX_REGISTRY, or `etcd` on PATH for a private one).
# =============================================================================

PEER_BLOCKS = 1170
PEER_SLOT_BYTES = 65536


def _active_rdma_devices() -> list:
    """RDMA devices with an ACTIVE port, in FLEXKV_TEST_RDMA_DEVICES order."""
    def _has_active_port(device: str) -> bool:
        for state in glob.glob(f"/sys/class/infiniband/{device}/ports/*/state"):
            with contextlib.suppress(OSError):
                with open(state) as f:
                    if "ACTIVE" in f.read():
                        return True
        return False

    requested = [d for d in os.getenv("FLEXKV_TEST_RDMA_DEVICES", "").split(",") if d]
    candidates = requested or sorted(
        os.path.basename(p) for p in glob.glob("/sys/class/infiniband/*"))
    return [d for d in candidates if _has_active_port(d)]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def cluster():
    """(rdma device, etcd registry): skip when the RDMA prerequisites are
    absent; start a private etcd when none is configured."""
    if os.getenv("FLEXKV_RUN_RADIX_PEER_TEST") != "1":
        pytest.skip("set FLEXKV_RUN_RADIX_PEER_TEST=1 to run the RDMA test")
    devices = _active_rdma_devices()
    if not devices:
        pytest.skip("no ACTIVE RDMA device found")
    try:
        from shmradix import _data
        if not hasattr(_data, "DataPlaneRegistry"):
            pytest.skip("shmradix built without etcd (no DataPlaneRegistry)")
    except ImportError:
        pytest.skip("shmradix built without the _data extension")
    registry = os.getenv("FLEXKV_TEST_RADIX_REGISTRY", "")
    proc = None
    workdir = None
    if not registry:
        etcd = shutil.which("etcd")
        if not etcd:
            pytest.skip("set FLEXKV_TEST_RADIX_REGISTRY or put etcd on PATH")
        client_port, peer_port = _free_port(), _free_port()
        workdir = tempfile.mkdtemp(prefix="flexkv_radix_etcd_")
        proc = subprocess.Popen(
            [etcd, "--name", "t", "--data-dir", os.path.join(workdir, "data"),
             "--listen-client-urls", f"http://127.0.0.1:{client_port}",
             "--advertise-client-urls", f"http://127.0.0.1:{client_port}",
             "--listen-peer-urls", f"http://127.0.0.1:{peer_port}",
             "--initial-advertise-peer-urls", f"http://127.0.0.1:{peer_port}",
             "--initial-cluster", f"t=http://127.0.0.1:{peer_port}"],
            stdout=open(os.path.join(workdir, "etcd.log"), "w"),
            stderr=subprocess.STDOUT)
        registry = f"etcd://127.0.0.1:{client_port}"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            with contextlib.suppress(OSError):
                with socket.create_connection(("127.0.0.1", client_port), timeout=0.5):
                    break
            time.sleep(0.2)
        else:
            proc.kill()
            pytest.skip("private etcd did not come up")
    try:
        yield devices[0], registry
    finally:
        if proc is not None:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(10)
            shutil.rmtree(workdir, ignore_errors=True)


def _peer_pattern(block: int, writer: int) -> bytes:
    return bytes([(block * 7 + writer * 131 + 3) % 251 + 1]) * PEER_SLOT_BYTES


def _node_main(rank, prefix, cluster_id, registry, rdma_dev, ready, done, output,
               local_head_blocks=0):
    """One node: a data-mode RadixServer (in-process) plus the FlexKV engine."""
    try:
        # Two servers on one host: distinct names, so radixshmem derives
        # distinct sockets (cluster names are per node; the geometry is shared).
        name = f"{prefix}_r{rank}"
        data_name = f"{prefix}_data_r{rank}"
        _sweep_region(name, data_name)
        cluster_kwargs = dict(
            expected_min_nodes=2, registry=registry, cluster_id=cluster_id,
            node_name=f"r{rank}", rpc_address="0.0.0.0", index_dev=rdma_dev,
            gid_idx=int(os.getenv("FLEXKV_TEST_RADIX_GID_IDX", "3")),
            bootstrap_timeout_sec=60, rht_slots_per_bucket=4)
        cfg = shmradix.ServerConfig(
            name=name, data_bytes=PEER_BLOCKS * PEER_SLOT_BYTES, slot_align=4096,
            data_name=data_name, prefault=False, transfer_devices=[rdma_dev],
            cluster=shmradix.ClusterConfig(**cluster_kwargs),
        )
        server = shmradix.RadixServer(cfg).start()      # waiting: the engine's geometry starts the rendezvous
        bootstrap.READY_TIMEOUT_S = 180.0                # this process only: the rendezvous may take a while
        engine = CacheEngineRadixShmem(
            name, geometry=shmradix.Geometry(block_size=16, full_slot_bytes=PEER_SLOT_BYTES,
                                             slot_align=4096),
            num_total_blocks=PEER_BLOCKS, tokens_per_block=16, peer_enabled=True)
        if not engine.peer_enabled:
            raise RuntimeError("engine did not see a distributed region")
        cluster_rank = bootstrap.radix_cluster_rank(engine.client)

        hashes = np.arange(26, dtype=np.uint64) * 104729 + 101
        query_hashes = hashes[:-1]

        def _seq(block_hashes):
            return FakeSeq(block_hashes=block_hashes.view(np.int64), tokens_per_block=16)

        if rank == 0:
            sequence = _seq(hashes)
            slots = engine.take(num_required_blocks=len(hashes))
            assert len(slots) == len(hashes)
            for i, slot in enumerate(slots):
                engine.client.slot_view(int(slot))[:] = _peer_pattern(i, writer=0)
            # insert() publishes and, with peer_enabled, flushes the RHT so the
            # reader can route to us.
            engine.insert(sequence, slots, num_insert_blocks=len(hashes))
            output.put({"writer_rank": cluster_rank})
            ready.set()
            if not done.wait(120):
                raise TimeoutError("reader did not complete")
        else:
            if local_head_blocks > 0:
                head = hashes[:local_head_blocks]
                head_slots = engine.take(num_required_blocks=local_head_blocks)
                assert len(head_slots) == local_head_blocks
                for i, slot in enumerate(head_slots):
                    engine.client.slot_view(int(slot))[:] = _peer_pattern(i, writer=1)
                engine.insert(_seq(head), head_slots, num_insert_blocks=local_head_blocks)
            if not ready.wait(120):
                raise TimeoutError("writer did not publish")
            # The writer's RHT publication is asynchronous: prefetch until the
            # pull brings the whole prefix home.
            expect = len(query_hashes)
            result = None
            job = None
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                job = engine.prefetch(_seq(query_hashes), timeout_ms=20000)
                result = job.wait(60)
                if result.common_hit >= expect:
                    break
                time.sleep(0.05)
            if result is None or result.common_hit < expect:
                raise AssertionError(
                    f"prefetch reached {result.common_hit if result else None} blocks, "
                    f"expected {expect}")
            match = engine.match(_seq(query_hashes))
            bad = []
            for i, slot in enumerate(match.local_slots.tolist()):
                writer = 1 if i < local_head_blocks else 0
                if bytes(engine.client.slot_view(int(slot))) != _peer_pattern(i, writer):
                    bad.append(i)
            output.put({
                "num_matched": match.num_matched_blocks,
                "job_local_hit": int(job.local_hit),
                "job_planned_hit": int(job.planned_hit),
                "remote_blocks": int(result.remote_blocks),
                "remote_bytes": int(result.remote_bytes),
                "source_rank": int(result.source_rank),
                "bad_blocks": bad,
                "reader_rank": cluster_rank,
            })
            match.release()
            done.set()
        engine.close()
        server.close()
    except Exception:
        output.put({"error": traceback.format_exc(), "rank": rank})
        ready.set()
        done.set()


def _run_two_nodes(registry, rdma_dev, local_head_blocks=0):
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    done = ctx.Event()
    output = ctx.Queue()
    prefix = f"/shmradix_peer_test_{os.getpid()}_{local_head_blocks}"
    cluster_id = f"flexkv-peer-test-{os.getpid()}-{local_head_blocks}"

    processes = [
        ctx.Process(target=_node_main,
                    args=(rank, prefix, cluster_id, registry, rdma_dev, ready, done,
                          output, local_head_blocks))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=240)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    messages = []
    while not output.empty():
        messages.append(output.get())
    errors = [message for message in messages if "error" in message]
    assert not errors, errors
    assert all(process.exitcode == 0 for process in processes)
    return (next(m for m in messages if "num_matched" in m),
            next(m for m in messages if "writer_rank" in m))


def test_prefetch_pulls_a_peer_prefix_over_rdma(cluster):
    """Node 1 holds nothing: the prefetch pulls all 25 blocks off node 0 and
    the local match then serves them with node 0's bytes."""
    rdma_dev, registry = cluster
    reader, writer = _run_two_nodes(registry, rdma_dev)
    assert reader["job_local_hit"] == 0
    assert reader["job_planned_hit"] == 25
    assert reader["remote_blocks"] == 25
    assert reader["remote_bytes"] == 25 * PEER_SLOT_BYTES
    assert reader["source_rank"] == writer["writer_rank"]
    assert reader["num_matched"] == 25
    assert reader["bad_blocks"] == []


def test_prefetch_extends_a_local_prefix_over_rdma(cluster):
    """Node 1 holds blocks 0-9 itself: the prefetch pulls only 10-24, and the
    match serves the head with node 1's bytes and the tail with node 0's."""
    rdma_dev, registry = cluster
    reader, writer = _run_two_nodes(registry, rdma_dev, local_head_blocks=10)
    assert reader["job_local_hit"] == 10
    assert reader["job_planned_hit"] == 25
    assert reader["remote_blocks"] == 15
    assert reader["source_rank"] == writer["writer_rank"]
    assert reader["num_matched"] == 25
    assert reader["bad_blocks"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
