# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Level 2 — SWA light-attach to the full-KV path (single-pass match).

Node-mount makes ``radix.match_prefix`` return the full prefix AND the deepest
in-range SWA node in ONE forward pass. Level 2 removes the redundant standalone
match that ``swa_align`` used to run: a SWA-aware GET must resolve
``usable = min(full_hit, swa_hit)`` and build the (clamped) graph from a SINGLE
per-tier match, not two.

These tests pin the Level 2 contract at the GlobalCacheEngine layer (no GPU /
KVTaskEngine needed — the match redundancy lives entirely in the engine):

  RED (the point of Level 2):
    * a SWA-aware get triggers exactly ONE round of per-tier full-KV matching
      (today's swa_align + get() path does TWO — this is what we remove).
  Behavior-equivalence (must stay green after the fold):
    * the full-KV transfer is still clamped to usable = min(full, swa).
  Guardrail (must stay green — plain path untouched):
    * a NON-SWA get on an SWA-enabled cache is never clamped and matches once.

Requires flexkv.c_ext (production CacheEngineAccel / CRadixTreeIndex).
"""
import numpy as np
import pytest
import torch

pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.debug import flexkv_logger

flexkv_logger.set_level("OFF")

pytestmark = pytest.mark.smoke

TPB = 16


def _model_config():
    return ModelConfig(
        num_layers=4, num_kv_heads=1, head_size=128,
        use_mla=True, dtype=torch.bfloat16, tp_size=1, dp_size=1,
    )


def _cache_config():
    cc = CacheConfig(
        tokens_per_block=TPB,
        enable_cpu=True, enable_ssd=False, enable_remote=False,
        num_cpu_blocks=4096,
    )
    cc.swa = SWAPoolConfig(
        enabled=True, num_slots=256, window_size=TPB,
        num_swa_layers=1, bytes_per_token_per_layer=64,
    )
    cc.enable_swa_transfer = True
    return cc


def _tokens(n_blocks, base):
    rs = np.random.RandomState(base)
    return rs.randint(0, 30000, size=n_blocks * TPB, dtype=np.int64)


def _complete(op_cb, cb):
    for c in op_cb.values():
        c()
    cb()


def _put(eng, tok, req=1):
    mask = np.ones_like(tok, dtype=np.int64)
    sm = np.arange(tok.shape[0], dtype=np.int64)
    _g, _rm, cb, op_cb, _e = eng.put(req, tok, mask, sm, dp_client_id=0)
    _complete(op_cb, cb)


class _MatchCounter:
    """Wrap the engine's per-tier match entry points and count invocations.

    CPU-only config routes through ``match_local_accel``; we also wrap
    ``match_all_accel`` so the same counter works if a tier config changes.
    Each call = one round of per-tier radix matching.
    """

    def __init__(self, eng):
        self.eng = eng
        self.n = 0
        self._orig_local = eng.match_local_accel
        self._orig_all = eng.match_all_accel

    def __enter__(self):
        def local(*a, **k):
            self.n += 1
            return self._orig_local(*a, **k)

        def allm(*a, **k):
            self.n += 1
            return self._orig_all(*a, **k)

        self.eng.match_local_accel = local
        self.eng.match_all_accel = allm
        return self

    def __exit__(self, *exc):
        self.eng.match_local_accel = self._orig_local
        self.eng.match_all_accel = self._orig_all
        return False


# =========================================================================== #
# RED — the redundant-match removal (Level 2 core)                            #
# =========================================================================== #

def test_swa_aware_get_matches_once():
    """A single SWA-aware GET must trigger exactly ONE round of full-KV matching.

    Today the get_match_swa path does two rounds: swa_align() runs a full
    per-tier match to compute usable, then get() -> _get_impl_* runs the SAME
    match again to build the graph. Level 2 folds usable computation into
    _get_impl_*; get(swa_aware=True) must own the single match.

    RED until Level 2 lands: get() has no swa_aware parameter (TypeError), and
    even routed through it the current code would match once in get() while
    swa_align matched separately.
    """
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tok = _tokens(4, base=31)
    _put(eng, tok)

    with _MatchCounter(eng) as mc:
        eng.get(request_id=2, token_ids=tok,
                token_mask=np.ones_like(tok, dtype=np.int64),
                slot_mapping=np.arange(tok.shape[0], dtype=np.int64),
                dp_client_id=0, swa_aware=True)
    assert mc.n == 1, (
        f"SWA-aware get matched {mc.n} rounds; Level 2 requires exactly 1 "
        "(no separate swa_align pass)")


def test_swa_aware_get_clamps_full_to_usable():
    """The folded path must still clamp the full-KV transfer to usable =
    min(full_hit, swa_hit) — the behavior swa_align+truncate used to provide.

    With SWA on the full stored tail, full_hit == swa_hit == 4, so return_mask
    covers all 4 blocks. This pins that the fold does not over- or under-clamp.
    """
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tok = _tokens(4, base=32)
    _put(eng, tok)

    graph, return_mask, cb, op_cb, end_id = eng.get(
        request_id=2, token_ids=tok,
        token_mask=np.ones_like(tok, dtype=np.int64),
        slot_mapping=np.arange(tok.shape[0], dtype=np.int64),
        dp_client_id=0, swa_aware=True)
    # usable = min(4, 4) = 4 -> all 4 blocks resident, SWA H2D present.
    assert int(return_mask.sum()) == 4 * TPB
    swa = [o for o in graph._op_map.values() if getattr(o, "is_swa", False)]
    assert len(swa) == 1, "SWA H2D must still attach to the same graph"
    _complete(op_cb, cb)


# =========================================================================== #
# Guardrail — the plain (non-SWA) path must be untouched                      #
# =========================================================================== #

def test_plain_get_on_swa_cache_matches_once_and_unclamped():
    """A plain get() (swa_aware defaults False) on an SWA-enabled cache must
    match exactly once and must NOT be clamped by any SWA window. This is the
    critical guardrail: Level 2 touches the shared _get_impl_* hot path."""
    eng = GlobalCacheEngine(_cache_config(), _model_config())
    tok = _tokens(4, base=33)
    _put(eng, tok)

    with _MatchCounter(eng) as mc:
        graph, return_mask, cb, op_cb, end_id = eng.get(
            request_id=2, token_ids=tok,
            token_mask=np.ones_like(tok, dtype=np.int64),
            slot_mapping=np.arange(tok.shape[0], dtype=np.int64),
            dp_client_id=0)
    assert mc.n == 1, f"plain get matched {mc.n} rounds; must be 1"
    # full 4-block hit, unclamped by SWA.
    assert int(return_mask.sum()) == 4 * TPB
    _complete(op_cb, cb)


# =========================================================================== #
# kvtask dispatch — get_match_swa's main path drops swa_align (single pass)   #
# =========================================================================== #

def test_get_match_swa_main_path_no_swa_align():
    """KVTaskEngine.get_match_swa on the main path (mask has 1s) must NOT call
    swa_align (the removed redundant match) and must drive exactly ONE
    _get_match_impl with swa_aware=True. Exercised via the real unbound method on
    a fake self — no GPU/TransferManager subprocess, mirroring the pure-logic
    tests in test_kvtask_lifecycle.py."""
    import types
    from flexkv.kvtask import KVTaskEngine

    calls = {"swa_align": 0, "get_match_impl": [], "update": 0, "trace": 0}
    tok = np.arange(4 * TPB, dtype=np.int64)

    class _FakeCacheEngine:
        tokens_per_block = TPB
        def swa_align(self, *a, **k):
            calls["swa_align"] += 1
            return 4, 4

    def _fake_get_match_impl(token_ids, slot_mapping, **kw):
        calls["get_match_impl"].append(kw.get("swa_aware"))
        # emulate a clamped return_mask_full covering all 4 blocks (usable=4)
        rm = np.zeros(token_ids.shape[0], dtype=np.bool_)
        rm[:4 * TPB] = True
        return 7, rm

    fake = types.SimpleNamespace(
        cache_engine=_FakeCacheEngine(),
        _update_tasks=lambda timeout=0: calls.__setitem__("update", calls["update"] + 1),
        _get_match_impl=_fake_get_match_impl,
        tracer=types.SimpleNamespace(
            trace_request=lambda **k: calls.__setitem__("trace", calls["trace"] + 1)),
    )
    full_mask = np.ones(4 * TPB, dtype=np.bool_)

    tid, rm_full, rm_swa = KVTaskEngine.get_match_swa(
        fake, tok, full_mask=full_mask, dp_client_id=0)

    assert calls["swa_align"] == 0, "main path must NOT call swa_align (Level 2)"
    assert calls["get_match_impl"] == [True], \
        "must drive exactly one _get_match_impl with swa_aware=True"
    assert tid == 7
    assert int(rm_full.sum()) == 4 * TPB
    # SWA window = trailing single block of the clamped full hit.
    assert int(rm_swa.sum()) == TPB
    assert rm_swa[3 * TPB:4 * TPB].all()


def test_get_match_swa_only_path_keeps_single_swa_align():
    """The SWA-only branch (full_mask all-0: GPU already holds full prefix) keeps
    ONE swa_align probe — there is no Full-KV match to ride, so this single match
    is necessary, not redundant. It must still report the trailing SWA window."""
    import types
    from flexkv.kvtask import KVTaskEngine

    calls = {"swa_align": 0, "get_match_impl": 0}
    tok = np.arange(4 * TPB, dtype=np.int64)

    class _FakeCacheEngine:
        tokens_per_block = TPB
        def swa_align(self, *a, **k):
            calls["swa_align"] += 1
            return 4, 4   # full_hit=4 (resident), swa_hit=4

    def _fake_get_match_impl(token_ids, slot_mapping, **kw):
        calls["get_match_impl"] += 1
        return 9, np.zeros(token_ids.shape[0], dtype=np.bool_)  # empty full graph

    fake = types.SimpleNamespace(
        cache_engine=_FakeCacheEngine(),
        _update_tasks=lambda timeout=0: None,
        _get_match_impl=_fake_get_match_impl,
        tracer=types.SimpleNamespace(trace_request=lambda **k: None),
    )
    full_mask = np.zeros(4 * TPB, dtype=np.bool_)   # all-0: SWA-only

    tid, rm_full, rm_swa = KVTaskEngine.get_match_swa(
        fake, tok, full_mask=full_mask, dp_client_id=0)

    assert calls["swa_align"] == 1, "SWA-only path resolves via one swa_align"
    assert tid == 9
    assert int(rm_full.sum()) == 0, "no Full-KV transfer when GPU holds full prefix"
    # SWA window still reported (trailing block at swa_hit=4).
    assert int(rm_swa.sum()) == TPB
    assert rm_swa[3 * TPB:4 * TPB].all()
