# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests: KVTask state machine + SWA load-lock lifecycle at the engine level.

Two concerns, both cheap (no TransferManager subprocess, no GPU):

1. The ``KVTask`` dataclass state machine: status transitions, is_completed,
   the new ``swa_slot_mapping`` field, and shed_heavy_resources.

2. The SWA load-lock lifecycle on ``GlobalCacheEngine``: a GET pins the matched
   CPU SWA node while building the GET plan, and the SWA H2D completion callback
   releases it (``_swa_release_load_lock``), leaving the
   node cached (dec_swa_lock_ref, NOT dec_swa_lock_only). This documents that
   the SWA lock follows the SAME lifecycle as the full-KV node lock (both taken
   in get()/put(), both released via the op/transfer callbacks) — so a fresh
   GET after a completed GET re-locks cleanly with no residual pin.

Requires flexkv.c_ext for concern 2 (production CacheEngineAccel); concern 1 is
pure Python.
"""
import numpy as np
import pytest
import torch

from flexkv.kvtask import KVTask, TaskType, TaskStatus

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# 1. KVTask state machine (pure, no engine)                                    #
# --------------------------------------------------------------------------- #

def _make_task(**over):
    base = dict(
        task_id=1, task_type=TaskType.GET, task_end_op_id=0,
        task_end_op_finished=False, status=TaskStatus.UNREADY,
        token_ids=np.arange(4, dtype=np.int64),
        slot_mapping=np.arange(4, dtype=np.int64),
        token_mask=np.ones(4, dtype=np.int64),
        graph=None, return_mask=np.zeros(4, dtype=np.bool_),
        callback=None, op_callback_dict={},
    )
    base.update(over)
    return KVTask(**base)


def test_task_defaults_and_swa_field():
    t = _make_task()
    assert t.status == TaskStatus.UNREADY
    assert t.swa_slot_mapping is None          # new field defaults None
    assert not t.is_completed()


def test_task_swa_slot_mapping_settable():
    sm = np.arange(16, dtype=np.int64)
    t = _make_task(swa_slot_mapping=sm)
    assert t.swa_slot_mapping is sm


@pytest.mark.parametrize("status,done", [
    (TaskStatus.UNREADY, False),
    (TaskStatus.READY, False),
    (TaskStatus.RUNNING, False),
    (TaskStatus.COMPLETED, True),
    (TaskStatus.CANCELLED, True),
    (TaskStatus.FAILED, True),
])
def test_is_completed_matrix(status, done):
    assert _make_task(status=status).is_completed() is done


def test_shed_heavy_resources_keeps_status():
    t = _make_task(status=TaskStatus.COMPLETED,
                   token_ids=np.arange(4, dtype=np.int64))
    t.shed_heavy_resources()
    assert t.graph is None and t.token_ids is None
    assert t.slot_mapping is None and t.token_mask is None
    assert t.callback is None
    # status/return_mask survive so wait() can still report
    assert t.status == TaskStatus.COMPLETED
    assert t.return_mask is not None


# --- M15/M16: SWA-aware get's core arithmetic (pure, mirrors kvtask logic) --- #

def _usable_and_swa_mask(full_hit, swa_hit, num_tokens, tpb):
    """Replicate the SWA-aware get arithmetic so the contract is pinned without
    standing up a KVTaskEngine (which needs a GPU TransferManager subprocess).
    usable=min(full,swa) clamps the full transfer; the trailing SWA window is
    the last block of the clamped hit."""
    usable = min(full_hit, swa_hit)
    full_mask = np.ones(num_tokens, dtype=np.bool_)
    truncated = full_mask.copy()
    truncated[usable * tpb:] = False
    swa_mask = np.zeros(num_tokens, dtype=np.bool_)
    if swa_hit > 0:
        swa_mask[(swa_hit - 1) * tpb: swa_hit * tpb] = True
    return usable, truncated, swa_mask


@pytest.mark.parametrize("full_hit,swa_hit,exp_usable", [
    (10, 6, 6),   # M15.1: SWA shorter than full -> clamp full to SWA
    (6, 6, 6),    # M15.4: SWA covers whole full hit -> no truncation loss
    (10, 0, 0),   # M15.3: no SWA hit -> usable 0 -> empty full graph
])
def test_get_swa_usable_is_min(full_hit, swa_hit, exp_usable):
    tpb, num_tokens = 16, 10 * 16
    usable, truncated, _ = _usable_and_swa_mask(full_hit, swa_hit, num_tokens, tpb)
    assert usable == exp_usable
    # The full transfer is shaped to exactly `usable` blocks (truncate-before-build).
    assert int(truncated.sum()) == exp_usable * tpb


@pytest.mark.parametrize("swa_hit", [1, 3, 6])
def test_get_swa_trailing_window_is_one_block(swa_hit):
    """M16.1: SWA is page-granular, so return_mask_swa
    marks exactly ONE block, ending at swa_hit."""
    tpb, num_tokens = 16, 6 * 16
    _, _, swa_mask = _usable_and_swa_mask(6, swa_hit, num_tokens, tpb)
    assert int(swa_mask.sum()) == tpb                       # exactly one block
    assert swa_mask[(swa_hit - 1) * tpb: swa_hit * tpb].all()  # the trailing one


def test_get_swa_no_hit_empty_window_no_underflow():
    """M16.2: swa_hit=0 -> return_mask_swa all-zero, and the (swa_hit-1) index is
    never evaluated (no negative-index wraparound marking the last block)."""
    tpb, num_tokens = 16, 6 * 16
    _, _, swa_mask = _usable_and_swa_mask(6, 0, num_tokens, tpb)
    assert not swa_mask.any()


# --------------------------------------------------------------------------- #
# 2. SWA load-lock lifecycle on the engine (needs c_ext)                       #
# --------------------------------------------------------------------------- #

c_ext = pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.debug import flexkv_logger

flexkv_logger.set_level("OFF")
TPB = 16


def _engine():
    mc = ModelConfig(num_layers=4, num_kv_heads=1, head_size=128,
                     use_mla=True, dtype=torch.bfloat16, tp_size=1, dp_size=1)
    cc = CacheConfig(tokens_per_block=TPB, enable_cpu=True, enable_ssd=False,
                     enable_remote=False, num_cpu_blocks=4096)
    cc.swa = SWAPoolConfig(
        enabled=True,
        num_slots=256,
        num_swa_layers=1,
        bytes_per_token_per_layer=64,
    )
    cc.enable_swa_transfer = True
    return GlobalCacheEngine(cc, mc)


def _tokens(base):
    rs = np.random.RandomState(base)
    return rs.randint(0, 30000, size=4 * TPB, dtype=np.int64)


def _put(eng, tok):
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(1, tok, mask, sm, dp_client_id=0)
    for c in op_cb.values():
        c()
    cb()


def test_get_pins_then_releases_swa_lock():
    """GET pins the matched CPU SWA node; the SWA H2D callback releases it
    (dec_swa_lock_ref -> node stays cached, lock back to 0)."""
    eng = _engine()
    tok = _tokens(21)
    _put(eng, tok)

    graph, _rm, cb, op_cb, end_id = eng.get(
        request_id=2, token_ids=tok, token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=np.arange(tok.shape[0], dtype=np.int64), dp_client_id=0,
        swa_aware=True)
    swa_h2d = [o for o in graph._op_map.values() if o.is_swa][0]

    # After building the GET graph, the matched CPU SWA node is pinned.
    sm = SequenceMeta(token_ids=tok, tokens_per_block=TPB); sm.gen_hashes()
    mr = eng.cpu_cache_engine.match(sm)
    assert mr.swa_hit_blocks == 4
    # The node carries the load pin (>=1) taken while building the GET plan.
    node = mr.last_swa_node
    assert node is not None
    assert node.swa_lock_ref >= 1

    # Complete the ops: the SWA H2D callback releases the pin.
    for c in op_cb.values():
        c()
    cb()
    node2 = eng.cpu_cache_engine.index.match_prefix(
        torch.from_numpy(sm.block_hashes[:4]).to(torch.int64), 4, False).last_swa_node
    assert node2.swa_lock_ref == 0, "SWA load lock not released on H2D completion"
    # slot still live (dec_swa_lock_ref keeps the cache, unlike dec_swa_lock_only)
    assert node2.has_swa()


def test_repeated_get_relocks_cleanly():
    """Two sequential GETs of the same prefix each pin+release with no residual
    (proves the release is paired, not leaking across requests)."""
    eng = _engine()
    tok = _tokens(22)
    _put(eng, tok)
    sm = SequenceMeta(token_ids=tok, tokens_per_block=TPB); sm.gen_hashes()
    bh = torch.from_numpy(sm.block_hashes[:4]).to(torch.int64)

    for req in (2, 3):
        _g, _rm, cb, op_cb, _e = eng.get(
            request_id=req, token_ids=tok,
            token_mask=np.ones_like(tok, dtype=np.int64),
            slot_mapping=np.arange(tok.shape[0], dtype=np.int64), dp_client_id=0,
            swa_aware=True)
        for c in op_cb.values():
            c()
        cb()
        node = eng.cpu_cache_engine.index.match_prefix(bh, 4, False).last_swa_node
        assert node.swa_lock_ref == 0, f"residual SWA lock after GET {req}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
