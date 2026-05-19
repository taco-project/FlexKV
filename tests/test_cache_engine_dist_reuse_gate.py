"""Phase D-2/D-4 unit tests for ``_sharing_domain_gate_get`` +
``_notify_sd_ready_on_put`` + ``_notify_master_sd_ready`` +
``_on_peer_sd_completed_op``.

These methods are grafted onto the huge ``GlobalCacheEngine`` class,
which is itself impossible to import here (it drags in torch + c_ext
+ redis_meta).  We use a lean ``_CacheEngineGateStub`` that mirrors
the real method bodies in spirit, plus source-level greps to guard
the real file from drift.

After Phase D-4 (proposal_unify_with_graph_dispatch_2026-05-15.md):
the per-SD ZMQ ``coord_put`` broadcast is gone — peer SD acks now
arrive as ``CompletedOp(sd_key, contributing_node_id)`` through the
graph-dispatch path, and ``_notify_master_sd_ready`` only does the
self-SD mark plus a pending-batch registration consumed by
``_on_peer_sd_completed_op``.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flexkv.common.dist_reuse.aggregate_radix import AggregateRadixTree  # noqa: E402
from flexkv.common.dist_reuse.master_coordinator import MasterCoordinator  # noqa: E402
from flexkv.common.dist_reuse.sharing_domain import SharingDomainKey  # noqa: E402


# ---------------------------------------------------------------------------
# Master fixture helpers
# ---------------------------------------------------------------------------
def _mk_single_sd_master() -> MasterCoordinator:
    sd = SharingDomainKey(
        model_id="test-model", pp_rank=0, pp_size=1,
        tp_node_idx=0, tp_node_count=1, is_nsa=False,
    )
    mc = MasterCoordinator(self_sd=sd, instance_id="inst-master")
    mc.expect_remotes(0)
    return mc


def _mk_multi_sd_master(total_sd: int = 2) -> MasterCoordinator:
    sd = SharingDomainKey(
        model_id="test-model", pp_rank=0, pp_size=total_sd,
        tp_node_idx=0, tp_node_count=1, is_nsa=False,
    )
    mc = MasterCoordinator(self_sd=sd, instance_id="inst-master")
    mc.expect_remotes(total_sd - 1)
    return mc


# ---------------------------------------------------------------------------
# MasterCoordinator.total_sd_count property
# ---------------------------------------------------------------------------
def test_master_total_sd_count_single():
    mc = _mk_single_sd_master()
    assert mc.total_sd_count == 1


def test_master_total_sd_count_multi():
    mc = _mk_multi_sd_master(total_sd=2)
    assert mc.total_sd_count == 2

    mc4 = _mk_multi_sd_master(total_sd=4)
    assert mc4.total_sd_count == 4


# ---------------------------------------------------------------------------
# Stubbed cache engine — mirrors the real ``GlobalCacheEngine`` shape we test.
# ---------------------------------------------------------------------------
class _CacheEngineGateStub:
    def __init__(self, master_coord=None) -> None:
        self._master_coord = master_coord
        self._pending_put_lock = threading.Lock()
        self._pending_put_batches: Dict[int, list] = {}

    @property
    def has_dist_reuse(self) -> bool:
        return self._master_coord is not None

    def _sharing_domain_gate_get(
        self,
        *,
        sequence_meta,
        return_mask,
        block_start_idx: int,
        num_gpu_blocks_to_transfer: int,
    ) -> bool:
        if not self.has_dist_reuse:
            return True
        if self._master_coord is None:
            return True
        try:
            total_sd = int(getattr(self._master_coord, "total_sd_count", 1))
        except Exception:
            total_sd = 1
        if total_sd <= 1:
            return True
        if num_gpu_blocks_to_transfer <= 0:
            return True
        try:
            sequence_meta.gen_hashes()
        except Exception:
            return True
        terminal_block_idx = block_start_idx + num_gpu_blocks_to_transfer - 1
        if terminal_block_idx >= sequence_meta.block_hashes.shape[0]:
            return True
        try:
            prefix_hash = int(sequence_meta.block_hashes[terminal_block_idx].item())
        except Exception:
            return True
        try:
            entry = self._master_coord.match_fully_ready(prefix_hash)
        except Exception:
            return True
        return entry is not None

    def _notify_sd_ready_on_put(
        self,
        *,
        sequence_meta,
        inserted_block_ids,
        block_start_idx: int,
        num_blocks_inserted: int,
    ) -> None:
        if not self.has_dist_reuse:
            return
        if self._master_coord is None:
            return
        if num_blocks_inserted <= 0:
            return
        try:
            sequence_meta.gen_hashes()
        except Exception:
            return
        terminal_idx = block_start_idx + num_blocks_inserted - 1
        if terminal_idx < 0 or terminal_idx >= sequence_meta.block_hashes.shape[0]:
            return
        try:
            prefix_hash = int(sequence_meta.block_hashes[terminal_idx].item())
        except Exception:
            return
        block_ids_list = []
        try:
            if inserted_block_ids is not None:
                block_ids_list = [int(b) for b in inserted_block_ids]
        except Exception:
            block_ids_list = []
        self._notify_master_sd_ready(
            prefix_hash=prefix_hash,
            block_ids=block_ids_list,
        )

    def _notify_master_sd_ready(
        self,
        prefix_hash: int,
        block_ids: list,
    ) -> None:
        """Phase D-2 stub: self-SD mark + pending PUT batch registration.

        No coord_put, no peer-SD broadcast — those acks now arrive
        asynchronously via ``_on_peer_sd_completed_op``.
        """
        if self._master_coord is None:
            return
        try:
            self_node_id = int(getattr(self._master_coord, "self_node_id", -1))
        except Exception:
            self_node_id = -1
        try:
            self._master_coord.mark_sd_ready(
                prefix_hash=int(prefix_hash),
                sd_key_str=self._master_coord.self_sd.serialize(),
                block_ids=list(block_ids) if block_ids is not None else [],
                node_id=self_node_id,
            )
        except Exception:
            pass

        # Multi-SD: register a pending PUT batch for the
        # _completion_sink to consume on peer-SD CompletedOp arrival.
        try:
            total_sd = int(getattr(self._master_coord, "total_sd_count", 1))
        except Exception:
            total_sd = 1
        if total_sd <= 1:
            return
        try:
            block_ids_list = [int(b) for b in (block_ids or [])]
        except Exception:
            block_ids_list = []
        with self._pending_put_lock:
            self._pending_put_batches[int(prefix_hash)] = block_ids_list

    def _on_peer_sd_completed_op(self, completed_op) -> None:
        """Phase D-2 stub: route peer-SD CompletedOp into mark_sd_ready."""
        if self._master_coord is None:
            return
        sd_key = getattr(completed_op, "sd_key", "") or ""
        if not sd_key:
            return
        if sd_key == self._master_coord.self_sd.serialize():
            return
        if not getattr(completed_op, "success", True):
            return
        node_id = int(getattr(completed_op, "contributing_node_id", -1))
        with self._pending_put_lock:
            pending_snapshot = list(self._pending_put_batches.items())
        for prefix_hash, block_ids_list in pending_snapshot:
            try:
                self._master_coord.mark_sd_ready(
                    prefix_hash=int(prefix_hash),
                    sd_key_str=sd_key,
                    block_ids=block_ids_list,
                    node_id=node_id,
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# SequenceMeta helper — avoid pulling torch just to build one.
# ---------------------------------------------------------------------------
class _FakeSeqMeta:
    """Mimics the two attributes we touch: ``gen_hashes()`` + ``block_hashes``."""
    def __init__(self, block_hashes):
        self.block_hashes = np.array(block_hashes, dtype=np.int64)
        self._hashed = True

    def gen_hashes(self):
        self._hashed = True


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------
class TestGateGet:
    def test_no_dist_reuse_passes(self):
        eng = _CacheEngineGateStub(master_coord=None)
        assert eng._sharing_domain_gate_get(
            sequence_meta=_FakeSeqMeta([1, 2, 3]),
            return_mask=None,
            block_start_idx=0,
            num_gpu_blocks_to_transfer=3,
        ) is True

    def test_single_sd_passes(self):
        mc = _mk_single_sd_master()
        try:
            eng = _CacheEngineGateStub(master_coord=mc)
            assert eng._sharing_domain_gate_get(
                sequence_meta=_FakeSeqMeta([1, 2, 3]),
                return_mask=None,
                block_start_idx=0,
                num_gpu_blocks_to_transfer=3,
            ) is True
        finally:
            mc.shutdown()

    def test_multi_sd_not_ready_rejects(self):
        mc = _mk_multi_sd_master(total_sd=2)
        try:
            eng = _CacheEngineGateStub(master_coord=mc)
            # No SDs marked ready — gate must reject.
            assert eng._sharing_domain_gate_get(
                sequence_meta=_FakeSeqMeta([0xAA, 0xBB, 0xCC]),
                return_mask=None,
                block_start_idx=0,
                num_gpu_blocks_to_transfer=3,
            ) is False
        finally:
            mc.shutdown()

    def test_multi_sd_fully_ready_passes(self):
        mc = _mk_multi_sd_master(total_sd=2)
        try:
            eng = _CacheEngineGateStub(master_coord=mc)
            # Mark every SD ready for the terminal prefix hash.
            self_sd_str = mc.self_sd.serialize()
            peer_sd = next(p for p in mc.self_sd.enumerate_peers() if p != mc.self_sd)
            peer_sd_str = peer_sd.serialize()
            mc.mark_sd_ready(prefix_hash=0xCC, sd_key_str=self_sd_str, block_ids=[1, 2, 3])
            mc.mark_sd_ready(prefix_hash=0xCC, sd_key_str=peer_sd_str, block_ids=[1, 2, 3])
            assert eng._sharing_domain_gate_get(
                sequence_meta=_FakeSeqMeta([0xAA, 0xBB, 0xCC]),
                return_mask=None,
                block_start_idx=0,
                num_gpu_blocks_to_transfer=3,
            ) is True
        finally:
            mc.shutdown()

    def test_zero_blocks_passes(self):
        mc = _mk_multi_sd_master()
        try:
            eng = _CacheEngineGateStub(master_coord=mc)
            assert eng._sharing_domain_gate_get(
                sequence_meta=_FakeSeqMeta([1]),
                return_mask=None,
                block_start_idx=0,
                num_gpu_blocks_to_transfer=0,
            ) is True
        finally:
            mc.shutdown()


# ---------------------------------------------------------------------------
# _notify_sd_ready_on_put → _notify_master_sd_ready
# ---------------------------------------------------------------------------
class TestNotifySdReadyOnPut:
    def test_no_dist_reuse_is_noop(self):
        eng = _CacheEngineGateStub(master_coord=None)
        # Must not raise.
        eng._notify_sd_ready_on_put(
            sequence_meta=_FakeSeqMeta([1, 2]),
            inserted_block_ids=[10, 20],
            block_start_idx=0,
            num_blocks_inserted=2,
        )

    def test_self_sd_marked(self):
        mc = _mk_single_sd_master()
        try:
            mc.mark_sd_ready = MagicMock(return_value=True)
            eng = _CacheEngineGateStub(master_coord=mc)
            eng._notify_sd_ready_on_put(
                sequence_meta=_FakeSeqMeta([0xAA, 0xBB]),
                inserted_block_ids=[10, 20],
                block_start_idx=0,
                num_blocks_inserted=2,
            )
            mc.mark_sd_ready.assert_called_once()
            kwargs = mc.mark_sd_ready.call_args.kwargs
            assert kwargs["prefix_hash"] == 0xBB
            assert kwargs["sd_key_str"] == mc.self_sd.serialize()
            assert kwargs["block_ids"] == [10, 20]
        finally:
            mc.shutdown()


# ---------------------------------------------------------------------------
# Phase D-2: _on_peer_sd_completed_op marks peer SD ready
# ---------------------------------------------------------------------------
class TestPeerSdCompletionSink:
    def test_self_sd_completed_op_ignored(self):
        mc = _mk_multi_sd_master(total_sd=2)
        try:
            eng = _CacheEngineGateStub(master_coord=mc)
            # Register a pending PUT batch first.
            eng._notify_master_sd_ready(prefix_hash=0xABC, block_ids=[1, 2])
            mc.mark_sd_ready = MagicMock()
            # Self-SD CompletedOp must be a no-op (already marked above).
            self_sd_str = mc.self_sd.serialize()
            self_co = SimpleNamespace(
                sd_key=self_sd_str,
                contributing_node_id=99,
                success=True,
            )
            eng._on_peer_sd_completed_op(self_co)
            mc.mark_sd_ready.assert_not_called()
        finally:
            mc.shutdown()

    def test_peer_sd_completed_op_marks_ready(self):
        mc = _mk_multi_sd_master(total_sd=2)
        try:
            eng = _CacheEngineGateStub(master_coord=mc)
            # Register a pending PUT batch.
            eng._notify_master_sd_ready(prefix_hash=0xABC, block_ids=[10, 20])
            assert eng._pending_put_batches[0xABC] == [10, 20]

            mc.mark_sd_ready = MagicMock(return_value=True)
            peer_sd = next(p for p in mc.self_sd.enumerate_peers() if p != mc.self_sd)
            peer_sd_str = peer_sd.serialize()
            peer_co = SimpleNamespace(
                sd_key=peer_sd_str,
                contributing_node_id=42,
                success=True,
            )
            eng._on_peer_sd_completed_op(peer_co)
            mc.mark_sd_ready.assert_called_once()
            kwargs = mc.mark_sd_ready.call_args.kwargs
            assert kwargs["prefix_hash"] == 0xABC
            assert kwargs["sd_key_str"] == peer_sd_str
            assert kwargs["block_ids"] == [10, 20]
            assert kwargs["node_id"] == 42
        finally:
            mc.shutdown()

    def test_failed_completed_op_ignored(self):
        mc = _mk_multi_sd_master(total_sd=2)
        try:
            eng = _CacheEngineGateStub(master_coord=mc)
            eng._notify_master_sd_ready(prefix_hash=0xABC, block_ids=[1])
            mc.mark_sd_ready = MagicMock()
            peer_co = SimpleNamespace(
                sd_key="peer-sd-key",
                contributing_node_id=42,
                success=False,
            )
            eng._on_peer_sd_completed_op(peer_co)
            mc.mark_sd_ready.assert_not_called()
        finally:
            mc.shutdown()

    def test_empty_sd_key_ignored(self):
        mc = _mk_multi_sd_master(total_sd=2)
        try:
            eng = _CacheEngineGateStub(master_coord=mc)
            eng._notify_master_sd_ready(prefix_hash=0xABC, block_ids=[1])
            mc.mark_sd_ready = MagicMock()
            co = SimpleNamespace(sd_key="", contributing_node_id=42, success=True)
            eng._on_peer_sd_completed_op(co)
            mc.mark_sd_ready.assert_not_called()
        finally:
            mc.shutdown()


# ---------------------------------------------------------------------------
# Source-level guards — keep the real file in lock-step with this stub.
# ---------------------------------------------------------------------------
def test_source_has_gate_and_notify_methods():
    path = REPO_ROOT / "flexkv" / "cache" / "cache_engine.py"
    src = path.read_text()
    required = [
        "def _sharing_domain_gate_get(",
        "def _notify_sd_ready_on_put(",
        "def _notify_master_sd_ready(",
        "def _on_peer_sd_completed_op(",
        "def is_evictable(",
        "has_dist_reuse",
        "_master_coord",
        "_pending_put_batches",
    ]
    missing = [m for m in required if m not in src]
    assert not missing, (
        f"GlobalCacheEngine source-guard: missing required tokens {missing}"
    )


def test_source_put_callback_carries_sd_notify_kwargs():
    """Defensive: the PUT-path callback must thread ``sd_notify_kwargs``
    so that the dist_reuse fully_ready signal reaches the aggregate.
    Empirically this is tied to the line that builds
    ``sd_notify_kwargs`` and the one that passes it to
    ``_transfer_callback``.
    """
    path = REPO_ROOT / "flexkv" / "cache" / "cache_engine.py"
    src = path.read_text()
    assert "sd_notify_kwargs = {" in src
    assert "is_put=True" in src
    assert "sd_notify_kwargs=sd_notify_kwargs" in src


def test_source_master_total_sd_count_property_present():
    path = REPO_ROOT / "flexkv" / "common" / "dist_reuse" / "master_coordinator.py"
    src = path.read_text()
    assert "def total_sd_count" in src


def test_source_notify_master_sd_ready_uses_pending_put_registry():
    """Phase D-4 source guard: ``_notify_master_sd_ready`` must
    populate ``_pending_put_batches`` for multi-SD deployments
    (replacing the old ``coord_put`` broadcast)."""
    path = REPO_ROOT / "flexkv" / "cache" / "cache_engine.py"
    src = path.read_text()
    assert "def _notify_master_sd_ready" in src
    assert "_pending_put_batches[" in src, (
        "_notify_master_sd_ready must register pending PUT batches "
        "for graph-dispatch peer-SD ack consumption"
    )
    # Negative check: the old coord_put route AND the deprecated
    # ``_coord_dispatcher`` field must be gone (Phase D-4 cleanup).
    assert "self._coord_dispatcher" not in src, (
        "_coord_dispatcher field was supposed to be removed in the "
        "Phase D-4 cleanup pass"
    )
    assert ".coord_put(" not in src, (
        "coord_put broadcast was supposed to be deleted in Phase D-4"
    )
