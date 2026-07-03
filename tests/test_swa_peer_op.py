"""SWA peer-op transfer-graph scenarios — the data-plane interface.

The transfer graph is the interface between the control plane (this code) and
the data plane (the SWA worker). These tests build the graph for the full matrix
of scenarios and (a) assert its shape/deps/flags, (b) dump a readable view so the
data-plane side can eyeball exactly what it will receive.

Representation (aligned with the data plane): SWA ops are plain ``TransferOp``
with ``is_swa=True`` and a STANDARD ``transfer_type`` (H2D / D2H / DISK2H /
H2DISK / REMOTE2H / H2REMOTE). They live in the SAME graph as the full-KV ops; a
VIRTUAL barrier joins the reported ops. SWA ops carry SWA-pool slot ids and are
kept OUT of the graph's _gpu_transfer_op_id (so set_gpu_blocks can't clobber them
with full-KV block ids).

Tier dependencies mirror the full-KV graph:
  GET  load:  SWA H2D depends on SWA DISK2H / REMOTE2H staging ops (only H2D reported)
  PUT  store: SWA H2DISK / H2REMOTE depend on SWA D2H, fire-and-forget (only D2H reported)

Sandbox note: SWACacheManager imports only flexkv.common.* (no torch/c_ext), but
flexkv.common.block pulls torch, so run in the real env: pytest tests/test_swa_peer_op.py
"""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

if 'flexkv.c_ext' not in sys.modules:
    sys.modules['flexkv.c_ext'] = MagicMock()

import numpy as np
import pytest

from flexkv.common.transfer import (
    TransferOpGraph, TransferOp, TransferType, DeviceType,
)
from flexkv.cache.transfer_pattern import add_virtual_op_for_multiple_finished_ops
from flexkv.cache.swa_cache_engine import SWACacheManager, SWAMatchResult

pytestmark = pytest.mark.unit


# --- graph dump (data-plane interface view) ----------------------------------

def dump_graph(graph: TransferOpGraph) -> str:
    """Render a graph as the data plane sees it: one line per op with type,
    is_swa, src/dst slot ids, and predecessor op_ids."""
    lines = []
    for op_id in sorted(graph._op_map):
        op = graph._op_map[op_id]
        tag = "SWA" if getattr(op, "is_swa", False) else "FULL"
        src = list(np.asarray(op.src_block_ids).tolist())
        dst = list(np.asarray(op.dst_block_ids).tolist())
        preds = sorted(op.predecessors)
        lines.append(
            f"  op#{op_id} [{tag}] {op.transfer_type.name} "
            f"src={src} dst={dst} deps={preds}"
            + (f" swa_key={op.swa_key}" if getattr(op, 'is_swa', False) else "")
        )
    return "\n".join(lines)


def _print_scenario(name, graph, end_op_id):
    print(f"\n=== {name} === (task_end_op={end_op_id})\n{dump_graph(graph)}")


def _swa_ops(graph):
    return [op for op in graph._op_map.values() if getattr(op, "is_swa", False)]


def _full_ops(graph):
    return [op for op in graph._op_map.values()
            if not getattr(op, "is_swa", False)
            and op.transfer_type != TransferType.VIRTUAL]


# --- stubs -------------------------------------------------------------------

def _mgr(enabled=True, cpu=True, ssd=False, remote=False):
    engines = {}
    if cpu:
        engines[DeviceType.CPU] = SimpleNamespace(swa_enabled=True)
    if ssd:
        engines[DeviceType.SSD] = SimpleNamespace(swa_enabled=True)
    if remote:
        engines[DeviceType.REMOTE] = SimpleNamespace(swa_enabled=True)
    gce = SimpleNamespace(
        cache_engines=engines,
        cache_config=SimpleNamespace(enable_swa_transfer=enabled),
    )
    return SWACacheManager(gce)


def _full_h2d(graph, n=2):
    """A stand-in full-KV H2D op (the main-KV terminal op for GET)."""
    op = TransferOp(
        graph_id=graph.graph_id, transfer_type=TransferType.H2D,
        src_block_ids=np.arange(100, 100 + n, dtype=np.int64),
        dst_block_ids=np.arange(200, 200 + n, dtype=np.int64),
    )
    graph.add_transfer_op(op)
    return op


def _full_d2h(graph, n=2):
    op = TransferOp(
        graph_id=graph.graph_id, transfer_type=TransferType.D2H,
        src_block_ids=np.arange(200, 200 + n, dtype=np.int64),
        dst_block_ids=np.arange(100, 100 + n, dtype=np.int64),
    )
    graph.add_transfer_op(op)
    return op


SWA_GPU = np.array([1, 2], dtype=np.int64)
SWA_CPU = np.array([11, 12], dtype=np.int64)
SWA_SSD = np.array([21, 22], dtype=np.int64)
SWA_REMOTE = np.array([31, 32], dtype=np.int64)


# ============================================================================
# GET scenarios (load) — the transfer graph the data plane receives
# ============================================================================

def test_get_full_only_no_swa():
    """No SWA needed: graph is just the full-KV H2D (barrier collapses to it)."""
    mgr = _mgr(enabled=True)
    g = TransferOpGraph()
    full = _full_h2d(g)
    swa_id = mgr.build_get_chain(g, gpu_slot_ids=np.array([], dtype=np.int64),
                                 cpu_slot_ids=np.array([], dtype=np.int64))
    finished = [full.op_id] + ([swa_id] if swa_id is not None else [])
    g, end = add_virtual_op_for_multiple_finished_ops(g, finished, 0)
    _print_scenario("GET full-only", g, end)
    assert swa_id is None
    assert len(_swa_ops(g)) == 0
    assert end == full.op_id


def test_get_full_plus_swa_cpu():
    """SWA resident on CPU: full H2D + SWA H2D (no staging), both into barrier."""
    mgr = _mgr(enabled=True)
    g = TransferOpGraph()
    full = _full_h2d(g)
    swa_id = mgr.build_get_chain(g, gpu_slot_ids=SWA_GPU, cpu_slot_ids=SWA_CPU, swa_key=7)
    g, end = add_virtual_op_for_multiple_finished_ops(g, [full.op_id, swa_id], 0)
    _print_scenario("GET full + SWA(CPU)", g, end)
    swa = _swa_ops(g)
    assert len(swa) == 1 and swa[0].transfer_type == TransferType.H2D
    assert swa[0].is_swa and swa[0].swa_key == 7
    assert len(swa[0].predecessors) == 0          # CPU-resident: no staging dep
    # barrier waits for BOTH full and SWA
    barrier = g._op_map[end]
    assert barrier.transfer_type == TransferType.VIRTUAL
    assert {full.op_id, swa_id} <= barrier.predecessors


def test_get_full_plus_swa_ssd_staging():
    """SWA from SSD: SWA H2D depends on SWA DISK2H (mirrors full DISK2H->H2D)."""
    mgr = _mgr(enabled=True, ssd=True)
    g = TransferOpGraph()
    full = _full_h2d(g)
    swa_id = mgr.build_get_chain(g, gpu_slot_ids=SWA_GPU, cpu_slot_ids=SWA_CPU,
                                 ssd_slot_ids=SWA_SSD, swa_key=7)
    g, end = add_virtual_op_for_multiple_finished_ops(g, [full.op_id, swa_id], 0)
    _print_scenario("GET full + SWA(SSD staging)", g, end)
    swa_h2d = g._op_map[swa_id]
    assert swa_h2d.transfer_type == TransferType.H2D and swa_h2d.is_swa
    staging = [op for op in _swa_ops(g) if op.op_id != swa_id]
    assert len(staging) == 1 and staging[0].transfer_type == TransferType.DISK2H
    assert staging[0].is_swa
    assert staging[0].op_id in swa_h2d.predecessors


def test_get_full_plus_swa_ssd_and_remote_staging():
    """SWA from SSD+REMOTE: SWA H2D depends on BOTH SWA DISK2H and SWA REMOTE2H."""
    mgr = _mgr(enabled=True, ssd=True, remote=True)
    g = TransferOpGraph()
    full = _full_h2d(g)
    swa_id = mgr.build_get_chain(g, gpu_slot_ids=SWA_GPU, cpu_slot_ids=SWA_CPU,
                                 ssd_slot_ids=SWA_SSD, remote_slot_ids=SWA_REMOTE, swa_key=7)
    g, end = add_virtual_op_for_multiple_finished_ops(g, [full.op_id, swa_id], 0)
    _print_scenario("GET full + SWA(SSD+REMOTE staging)", g, end)
    swa_h2d = g._op_map[swa_id]
    dep_types = {g._op_map[p].transfer_type for p in swa_h2d.predecessors}
    assert dep_types == {TransferType.DISK2H, TransferType.REMOTE2H}
    assert all(g._op_map[p].is_swa for p in swa_h2d.predecessors)


def test_get_swa_only():
    """GPU has full, only the trailing SWA window slid out: SWA-only graph."""
    mgr = _mgr(enabled=True)
    g, swa_id = mgr.build_swa_only_graph(TransferType.H2D, SWA_CPU, SWA_GPU, swa_key=9)
    g, end = add_virtual_op_for_multiple_finished_ops(g, [swa_id], 0)
    _print_scenario("GET SWA-only", g, end)
    assert len(_full_ops(g)) == 0
    assert len(_swa_ops(g)) == 1
    assert end == swa_id            # sole op, no VIRTUAL needed


# ============================================================================
# PUT scenarios (store) — write-through fire-and-forget
# ============================================================================

def test_put_full_plus_swa_cpu_only():
    """CPU-only store: full D2H + SWA D2H, both reported; no write-through op."""
    mgr = _mgr(enabled=True)
    g = TransferOpGraph()
    full = _full_d2h(g)
    swa_id = mgr.build_put_chain(g, gpu_slot_ids=SWA_GPU, cpu_slot_ids=SWA_CPU, swa_key=7)
    g, end = add_virtual_op_for_multiple_finished_ops(g, [full.op_id, swa_id], 0)
    _print_scenario("PUT full + SWA(CPU)", g, end)
    swa = _swa_ops(g)
    assert len(swa) == 1 and swa[0].transfer_type == TransferType.D2H and swa[0].is_swa


def test_put_swa_writethrough_ssd_remote_fire_and_forget():
    """Write-through: SWA H2DISK/H2REMOTE depend on SWA D2H but are NOT reported."""
    mgr = _mgr(enabled=True, ssd=True, remote=True)
    g = TransferOpGraph()
    full = _full_d2h(g)
    swa_d2h_id = mgr.build_put_chain(g, gpu_slot_ids=SWA_GPU, cpu_slot_ids=SWA_CPU,
                                     ssd_slot_ids=SWA_SSD, remote_slot_ids=SWA_REMOTE, swa_key=7)
    finished = [full.op_id, swa_d2h_id]   # only D2H ops reported
    g, end = add_virtual_op_for_multiple_finished_ops(g, finished, 0)
    _print_scenario("PUT full + SWA write-through(SSD+REMOTE)", g, end)
    wt = [op for op in _swa_ops(g) if op.op_id != swa_d2h_id]
    assert {o.transfer_type for o in wt} == {TransferType.H2DISK, TransferType.H2REMOTE}
    for o in wt:
        assert o.is_swa and swa_d2h_id in o.predecessors   # depend on SWA D2H
    # write-through ops are fire-and-forget: NOT predecessors of the barrier
    barrier = g._op_map[end]
    assert all(o.op_id not in barrier.predecessors for o in wt)
    assert swa_d2h_id in barrier.predecessors


# ============================================================================
# gate + routing invariants
# ============================================================================

def test_gate_off_produces_no_swa_ops():
    """enable_swa_transfer=False: build chains add nothing; graph is full-only."""
    mgr = _mgr(enabled=False, ssd=True, remote=True)
    g = TransferOpGraph()
    full = _full_h2d(g)
    assert mgr.build_get_chain(g, SWA_GPU, SWA_CPU, ssd_slot_ids=SWA_SSD) is None
    assert len(_swa_ops(g)) == 0
    assert len(g._op_map) == 1


def test_swa_ops_tracked_in_both_lists_but_preserved_by_single_arg_set_gpu_blocks():
    """Unified GPU-transfer model (PR#191 integration): SWA H2D/D2H ops ARE in
    _gpu_transfer_op_id (so set_gpu_blocks(gpu, swa_gpu) can bind them) AND in
    _swa_gpu_transfer_op_id (so set_swa_gpu_blocks / the kvtask two-call path can
    bind them). The safety property is no longer 'excluded from the list' but
    'single-arg set_gpu_blocks(gpu) never overwrites an SWA op's slot' — verified
    in test_set_gpu_blocks_does_not_touch_swa_slots below."""
    mgr = _mgr(enabled=True, ssd=True)
    g = TransferOpGraph()
    full = _full_h2d(g)        # full H2D IS a gpu transfer op
    swa_id = mgr.build_get_chain(g, SWA_GPU, SWA_CPU, ssd_slot_ids=SWA_SSD)
    assert full.op_id in g._gpu_transfer_op_id
    assert full.op_id not in g._swa_gpu_transfer_op_id
    # SWA GPU-touching ops (H2D/D2H) are tracked in BOTH lists now.
    for op in _swa_ops(g):
        if op.transfer_type in (TransferType.H2D, TransferType.D2H):
            assert op.op_id in g._gpu_transfer_op_id
            assert op.op_id in g._swa_gpu_transfer_op_id


def test_set_gpu_blocks_does_not_touch_swa_slots():
    """After set_gpu_blocks, SWA op GPU slots are untouched (still SWA-pool ids)."""
    mgr = _mgr(enabled=True)
    g = TransferOpGraph()
    full = _full_h2d(g, n=2)
    swa_id = mgr.build_get_chain(g, gpu_slot_ids=SWA_GPU, cpu_slot_ids=SWA_CPU)
    g.set_gpu_blocks(np.array([900, 901], dtype=np.int64))
    # full H2D dst rebound to the new gpu blocks; SWA H2D dst still SWA_GPU
    assert g._op_map[full.op_id].dst_block_ids.tolist() == [900, 901]
    np.testing.assert_array_equal(g._op_map[swa_id].dst_block_ids, SWA_GPU)


# ============================================================================
# multi-tier match (SWAMatchResult)
# ============================================================================

class _FakeSWANode:
    """A fake node-mounted SWA source: reports a fixed (hit, slot, key)."""
    def __init__(self, hit_blocks, slot, key):
        self._hit, self._slot, self._key = hit_blocks, slot, key

    def match(self, block_hashes, upper_bound_blocks, lock_for_load=False):
        hit = min(self._hit, int(upper_bound_blocks))
        return (hit, self._slot, self._key) if hit > 0 else (0, -1, -1)


def _engine_for(idx):
    return SimpleNamespace(
        swa_enabled=True,
        match_swa=lambda seq, upper_bound_blocks, lock_for_load=False:
            idx.match(None, upper_bound_blocks, lock_for_load),
    )


def _mgr_tiers(cpu=None, ssd=None, remote=None):
    engines = {}
    if cpu is not None:
        engines[DeviceType.CPU] = _engine_for(cpu)
    if ssd is not None:
        engines[DeviceType.SSD] = _engine_for(ssd)
    if remote is not None:
        engines[DeviceType.REMOTE] = _engine_for(remote)
    gce = SimpleNamespace(cache_engines=engines,
                          cache_config=SimpleNamespace(enable_swa_transfer=True))
    return SWACacheManager(gce)


class _Seq:
    def gen_hashes(self):
        pass


def test_match_cpu_preferred_then_ssd_then_remote():
    mgr = _mgr_tiers(cpu=_FakeSWANode(0, -1, -1),
                     ssd=_FakeSWANode(2, 22, 222),
                     remote=_FakeSWANode(5, 33, 333))
    r = mgr.match_prefix(_Seq(), 6, 6, 6)
    assert r.cpu_hit_blocks == 0
    assert r.ssd_hit_blocks == 2 and r.ssd_slot == 22
    assert r.remote_hit_blocks == 5 and r.remote_slot == 33
    assert r.best_hit_blocks == 5


def test_match_per_tier_clamp_to_full_hit():
    mgr = _mgr_tiers(cpu=_FakeSWANode(0, -1, -1),
                     ssd=_FakeSWANode(9, 22, 222),
                     remote=_FakeSWANode(9, 33, 333))
    r = mgr.match_prefix(_Seq(), cpu_full_hit_blocks=0,
                         ssd_full_hit_blocks=2, remote_full_hit_blocks=3)
    assert r.ssd_hit_blocks == 2 and r.remote_hit_blocks == 3


def test_match_no_cpu_index_empty():
    gce = SimpleNamespace(cache_engines={}, cache_config=SimpleNamespace(enable_swa_transfer=True))
    r = SWACacheManager(gce).match_prefix(_Seq(), 4)
    assert r.best_hit_blocks == 0 and r.cpu_slot == -1


def test_match_propagates_per_tier_keys():
    """Per-tier hit/slot/key propagate so the data plane can drive
    set_ready/unlock for each tier; best_hit_blocks is the cross-tier max."""
    mgr = _mgr_tiers(cpu=_FakeSWANode(1, 10, 100),
                     ssd=_FakeSWANode(2, 22, 222),
                     remote=_FakeSWANode(5, 33, 333))
    r = mgr.match_prefix(_Seq(), 6, 6, 6)
    assert r.cpu_hit_blocks == 1 and r.cpu_slot == 10 and r.cpu_key == 100
    assert r.ssd_key == 222 and r.remote_key == 333
    assert r.best_hit_blocks == 5        # cross-tier max (REMOTE here)


def test_match_forwards_upper_bound_and_lock_per_tier():
    """Each tier's match_swa is called with THAT tier's Full-KV hit as the upper
    bound (SWA subset of Full) and the caller's lock_for_load is forwarded."""
    calls = {}

    def _engine(tier):
        def _match(seq, upper_bound_blocks, lock_for_load=False):
            calls[tier] = (upper_bound_blocks, lock_for_load)
            return (1, 0, 0)
        return SimpleNamespace(swa_enabled=True,
                               match_swa=_match)

    engines = {DeviceType.CPU: _engine("cpu"),
               DeviceType.SSD: _engine("ssd"),
               DeviceType.REMOTE: _engine("remote")}
    gce = SimpleNamespace(cache_engines=engines,
                          cache_config=SimpleNamespace(enable_swa_transfer=True))
    SWACacheManager(gce).match_prefix(_Seq(), cpu_full_hit_blocks=3,
                                      ssd_full_hit_blocks=5,
                                      remote_full_hit_blocks=7,
                                      lock_for_load=True)
    assert calls["cpu"] == (3, True)
    assert calls["ssd"] == (5, True)
    assert calls["remote"] == (7, True)
