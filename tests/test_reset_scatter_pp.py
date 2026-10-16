"""``reset()`` must not scatter across PP stages.

sglang delivers flush_cache (hence ``RadixCache.reset`` -> connector
``reset()``) to its ``is_rank_zero`` scheduler only: ``pp_rank == 0 and
attn_tp_rank == 0 and attn_cp_rank == 0``. Stages 1..N therefore never run
``reset()`` and never post a matching recv, so a blocking PP scatter there
isends into the void and hangs ``wait()`` forever -- taking the whole
pipeline down with it (pp0 stops producing microbatches, pp1..N park in
``_pp_recv_pyobj_from_prev_stage``).

Observed on GLM-5.3 PP4/DP2 (attn_tp == attn_cp == 1, so the PP hop was the
*only* live scatter axis): the server came up, served warmup, then wedged the
instant the benchmark issued its first flush_cache.

Covers: the PP hop is skipped, the TP/CP hops still run, the live gloo
topology does not deadlock, and a genuinely unmatched blocking scatter now
fails loudly instead of hanging.
"""
import multiprocessing as mp
import inspect
import os
import socket
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from flexkv.integration.sglang.comm import FlexKVComm, FlexKVScatterChannel

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comm(pp_size, attn_tp_size=1, attn_cp_size=1, pp_rank=0,
          attn_tp_rank=0, attn_cp_rank=0, world_rank=0):
    """A FlexKVComm with routing tables installed but no torch.distributed.

    Mirrors the real ``__init__`` arithmetic (stride / stage_base) rather
    than inventing a layout, so the rank tables under test are the ones the
    connector would really use.
    """
    comm = FlexKVComm.__new__(FlexKVComm)
    comm.world_rank = world_rank
    comm.pp_size = pp_size
    comm.attn_tp_size = attn_tp_size
    comm.attn_cp_size = attn_cp_size
    comm.pp_rank = pp_rank
    comm.attn_tp_rank = attn_tp_rank
    comm.attn_cp_rank = attn_cp_rank
    comm.is_pp_stage_leader = attn_tp_rank == 0 and attn_cp_rank == 0
    comm.is_pp_leader = pp_rank == 0 and comm.is_pp_stage_leader
    comm.is_sync_leader = comm.is_pp_leader
    comm.is_cp_leader = attn_cp_rank == 0
    comm.is_tp_leader = attn_tp_rank == 0

    stride = attn_tp_size * attn_cp_size
    local_offset = attn_cp_rank * attn_tp_size + attn_tp_rank
    stage_base = world_rank - local_offset
    sglang_tp = stride  # one attn-DP replica
    comm._pp_stage_leader_ranks = [
        stage_base + (s - pp_rank) * sglang_tp for s in range(pp_size)
    ]
    comm._pp_stage_member_ranks = list(range(stage_base, stage_base + stride))
    comm._cp_leader_ranks = (
        [stage_base + cp * attn_tp_size for cp in range(attn_cp_size)]
        if attn_cp_size > 1 else []
    )
    comm._tp_group_ranks = (
        list(range(stage_base, stage_base + attn_tp_size))
        if attn_tp_size > 1 else []
    )
    comm._world_cpu_group = None
    comm.attn_tp_cpu_group = None
    comm.attn_cp_cpu_group = None
    comm._async_works = []
    comm._reap_high = 64
    comm._reap_calls = 0
    comm._reap_stuck_total = 0
    comm._reap_drained_total = 0
    comm._blocking_scatter_timeout_s = 120.0
    comm.needs_sync = pp_size > 1 or attn_tp_size > 1 or attn_cp_size > 1
    comm.needs_sync_without_pp = attn_tp_size > 1 or attn_cp_size > 1
    return comm


RESET_CHANNELS = [FlexKVScatterChannel.RESET, FlexKVScatterChannel.STORE_RESET]


# ---------------------------------------------------------------------------
# needs_sync_without_pp: is there anyone left once PP is dropped?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pp,tp,cp,expect_sync,expect_no_pp", [
    # The deadlocking topology: PP is the only axis, so reset must not sync.
    (4, 1, 1, True, False),
    # TP/CP present -> those ranks do run reset(), so they still need it.
    (4, 2, 1, True, True),
    (4, 1, 2, True, True),
    (1, 2, 1, True, True),
    # Nothing to talk to at all.
    (1, 1, 1, False, False),
])
def test_needs_sync_without_pp(pp, tp, cp, expect_sync, expect_no_pp):
    comm = _comm(pp, attn_tp_size=tp, attn_cp_size=cp)
    assert comm.needs_sync is expect_sync
    assert comm.needs_sync_without_pp is expect_no_pp


# ---------------------------------------------------------------------------
# skip_pp drops exactly the PP hop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channel", RESET_CHANNELS)
def test_skip_pp_sends_nothing_in_pure_pp(channel, monkeypatch):
    """PP4 + attn_tp=attn_cp=1: skip_pp must make scatter a no-op.

    This is the exact case that hung: without skip_pp the leader isends to
    ranks 1..3 and blocks in wait() because nobody there calls reset().
    """
    comm = _comm(4, pp_rank=0, world_rank=0)
    isend = MagicMock(name="isend")
    monkeypatch.setattr(comm, "_isend", isend)
    monkeypatch.setattr(comm, "_recv", MagicMock(side_effect=AssertionError))

    payload = {"ok": True, "error": ""}
    assert comm.scatter(
        payload, channel=channel, blocking=True, skip_pp=True) == payload
    isend.assert_not_called()


@pytest.mark.parametrize("channel", RESET_CHANNELS)
def test_without_skip_pp_the_leader_would_send_to_every_stage(channel, monkeypatch):
    """Guards the diagnosis: the old call really did fan out over PP."""
    comm = _comm(4, pp_rank=0, world_rank=0)
    works = [MagicMock(name="w0"), MagicMock(name="w1")]
    isend = MagicMock(return_value=works)
    monkeypatch.setattr(comm, "_isend", isend)

    comm.scatter({"ok": True, "error": ""}, channel=channel, blocking=True)

    assert [c.args[0] for c in isend.call_args_list] == [1, 2, 3]
    # And it blocked on each send -- the hang.
    assert works[0].wait.call_count == 3


@pytest.mark.parametrize("channel", RESET_CHANNELS)
def test_skip_pp_keeps_the_tp_hop(channel, monkeypatch):
    """attn_tp=2: those ranks *do* run reset(), so TP must still fan out."""
    comm = _comm(4, attn_tp_size=2, pp_rank=0, attn_tp_rank=0, world_rank=0)
    isend = MagicMock(return_value=[MagicMock()])
    monkeypatch.setattr(comm, "_isend", isend)

    comm.scatter({"ok": True, "error": ""}, channel=channel,
                 blocking=True, skip_pp=True)

    # Only the TP sibling (rank 1), never the PP leaders (2, 4, 6).
    assert [c.args[0] for c in isend.call_args_list] == [1]


@pytest.mark.parametrize("channel", RESET_CHANNELS)
def test_skip_pp_follower_still_receives_from_tp_leader(channel, monkeypatch):
    """A TP follower must keep reading the leader's verdict, not skip it."""
    comm = _comm(4, attn_tp_size=2, pp_rank=0, attn_tp_rank=1, world_rank=1)
    verdict = {"ok": False, "error": "drain failed"}
    monkeypatch.setattr(comm, "_recv", MagicMock(return_value=verdict))
    monkeypatch.setattr(
        comm, "_isend", MagicMock(side_effect=AssertionError("follower sent")))

    got = comm.scatter({"ok": True, "error": ""}, channel=channel,
                       blocking=True, skip_pp=True)
    assert got == verdict


def test_skip_pp_does_not_leak_into_the_hot_path(monkeypatch):
    """Default scatter (e.g. LOOKUP) must still use PP: those callers run
    on every stage leader, so dropping the hop there would break lookup."""
    comm = _comm(4, pp_rank=0, world_rank=0)
    isend = MagicMock(return_value=[MagicMock()])
    monkeypatch.setattr(comm, "_isend", isend)

    comm.scatter({}, channel=FlexKVScatterChannel.LOOKUP)

    assert [c.args[0] for c in isend.call_args_list] == [1, 2, 3]


# ---------------------------------------------------------------------------
# A stalled blocking scatter must fail loudly, not hang
# ---------------------------------------------------------------------------

def test_blocking_scatter_timeout_names_the_problem(monkeypatch):
    comm = _comm(4, pp_rank=0, world_rank=0)
    stuck = MagicMock()
    stuck.wait.side_effect = RuntimeError("gloo timeout")
    monkeypatch.setattr(comm, "_isend", MagicMock(return_value=[stuck]))

    with pytest.raises(RuntimeError, match="scatter\\(blocking=True\\) stalled"):
        comm.scatter({"ok": True, "error": ""},
                     channel=FlexKVScatterChannel.RESET, blocking=True)
    stuck.wait.assert_called_once()
    assert stuck.wait.call_args.kwargs["timeout"] == timedelta(seconds=120)


def test_blocking_scatter_timeout_can_be_disabled(monkeypatch):
    comm = _comm(4, pp_rank=0, world_rank=0)
    comm._blocking_scatter_timeout_s = 0.0
    work = MagicMock()
    monkeypatch.setattr(comm, "_isend", MagicMock(return_value=[work]))

    comm.scatter({"ok": True, "error": ""},
                 channel=FlexKVScatterChannel.RESET, blocking=True)
    # 0 restores the legacy unbounded wait(): no timeout kwarg at all.
    assert work.wait.call_args.kwargs == {}


def test_non_blocking_scatter_is_untouched(monkeypatch):
    comm = _comm(4, pp_rank=0, world_rank=0)
    work = MagicMock()
    monkeypatch.setattr(comm, "_isend", MagicMock(return_value=[work]))

    comm.scatter({}, channel=FlexKVScatterChannel.LOOKUP, blocking=False)

    work.wait.assert_not_called()
    assert len(comm._async_works) == 3


# ---------------------------------------------------------------------------
# The real connector.reset() call sites
# ---------------------------------------------------------------------------
# The checks above verify that ``scatter(skip_pp=True)`` behaves. These verify
# that ``reset()`` actually *asks* for it -- reverting either call site to the
# old ``needs_sync`` / no-skip_pp form must fail here, otherwise the fix is
# only tested in the abstract.

class _RecordingSyncCtx:
    """Stands in for FlexKVComm, recording how reset() calls scatter."""

    def __init__(self, *, needs_sync=True, needs_sync_without_pp=False,
                 is_sync_leader=True):
        self.needs_sync = needs_sync
        self.needs_sync_without_pp = needs_sync_without_pp
        self.is_sync_leader = is_sync_leader
        self.calls = []

    def scatter(self, data, *, channel, blocking=False, skip_pp=False):
        self.calls.append(
            {"channel": channel, "blocking": blocking, "skip_pp": skip_pp}
        )
        return data


def _connector_for_reset(sync_ctx):
    """A FlexKVConnector with only what reset() touches."""
    from flexkv.integration.sglang.connector import FlexKVConnector

    c = FlexKVConnector.__new__(FlexKVConnector)
    c._sync_ctx = sync_ctx
    c.kv_manager = None            # no tasks to drain
    c._chunked_prefetch = False
    c._launched_load_tids = []
    c._launched_load_contexts = {}
    c._pending_lookups = {}
    c._pending_lookup_contexts = {}
    c._prefetch_contexts = {}
    c._ongoing_prefetches = {}
    c._prefetch_sessions = {}
    c._prefetch_result_sessions = {}
    c._prefetch_loaded_spans = {}
    c._inflight_loads = {}
    c._inflight_stores = {}
    c._inflight_store_contexts = {}
    c._completed_layerwise = {}
    c.layer_done_counter = None
    return c


def test_reset_never_scatters_over_pp_in_pure_pp_topology():
    """PP4, attn_tp=attn_cp=1: reset() must issue no scatter at all.

    needs_sync is True (pp_size > 1) -- the old gate -- so a regression that
    goes back to ``needs_sync`` would scatter here and hang in production.
    """
    ctx = _RecordingSyncCtx(needs_sync=True, needs_sync_without_pp=False)
    _connector_for_reset(ctx).reset()
    assert ctx.calls == []


def test_reset_scatters_with_skip_pp_when_tp_peers_exist():
    """attn_tp>1: both reset payloads go out, and both must skip the PP hop."""
    ctx = _RecordingSyncCtx(needs_sync=True, needs_sync_without_pp=True)
    _connector_for_reset(ctx).reset()

    assert [c["channel"] for c in ctx.calls] == [
        FlexKVScatterChannel.RESET,
        FlexKVScatterChannel.STORE_RESET,
    ]
    for call in ctx.calls:
        assert call["skip_pp"] is True, call
        assert call["blocking"] is True, call


def test_reset_propagates_a_follower_failure():
    """skip_pp must not swallow the leader's verdict on TP followers."""
    class _Failing(_RecordingSyncCtx):
        def scatter(self, data, *, channel, blocking=False, skip_pp=False):
            super().scatter(data, channel=channel, blocking=blocking,
                            skip_pp=skip_pp)
            return {"ok": False, "error": "drain failed upstream"}

    ctx = _Failing(needs_sync=True, needs_sync_without_pp=True,
                   is_sync_leader=False)
    with pytest.raises(RuntimeError, match="drain failed upstream"):
        _connector_for_reset(ctx).reset()


# ---------------------------------------------------------------------------
# End-to-end on real gloo: pp0-only reset must not wedge the pipeline
# ---------------------------------------------------------------------------

def _reset_worker(rank, world, port, tp, pp, q):
    """Reproduce the live topology: only pp_rank 0 calls the reset scatter."""
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        import torch.distributed as dist

        from flexkv.integration.sglang.comm import (
            FlexKVComm,
            FlexKVScatterChannel,
        )

        dist.init_process_group("gloo", rank=rank, world_size=world,
                                timeout=timedelta(seconds=60))
        my_pp, tp_rank = divmod(rank, tp)
        my_group = None
        for t in range(tp):
            g = dist.new_group([s * tp + t for s in range(pp)])
            if t == tp_rank:
                my_group = g
        rank_info = SimpleNamespace(
            model_config=SimpleNamespace(
                pp_size=pp, attn_tp_size=1, attn_cp_size=1, attn_dp_size=tp),
            pp_rank=my_pp, attn_tp_rank=0, attn_cp_rank=0, pp_size_per_node=pp,
        )
        comm = FlexKVComm(rank_info=rank_info, world_rank=rank,
                          pp_group=my_group, world_group=dist.group.WORLD)

        # Exactly what connector.reset() now does, on exactly the ranks
        # sglang's is_rank_zero would route flush_cache to.
        did_scatter = False
        if my_pp == 0:
            if comm.needs_sync_without_pp:
                comm.scatter(
                    {"ok": True, "error": ""},
                    channel=FlexKVScatterChannel.RESET,
                    blocking=True,
                    skip_pp=True,
                )
                did_scatter = True

        # Proof the pipeline is still usable afterwards: a collective that
        # every rank joins. A leaked PP isend would desync this.
        dist.barrier()
        q.put((rank, "ok", my_pp, did_scatter))
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        q.put((rank, f"ERROR {type(e).__name__}: {e}", -1, False))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_pp_only_reset_does_not_deadlock_live():
    """PP4 x DP2, attn_tp=1 -- the GLM-5.3 prefill topology that hung."""
    tp, pp = 2, 4
    world = tp * pp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_reset_worker, args=(r, world, port, tp, pp, q))
             for r in range(world)]
    for p in procs:
        p.start()
    try:
        # Generous but finite: the bug's signature is "never returns".
        results = [q.get(timeout=120) for _ in range(world)]
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()

    for rank, status, _, _ in results:
        assert status == "ok", f"rank {rank}: {status}"
    # Nobody scattered: attn_tp == attn_cp == 1 leaves no peer once PP is
    # dropped, which is precisely why the PP isend had no receiver.
    assert all(not did for _, _, _, did in results)

# NOTE: a companion test that drove the *pre-fix* call (no skip_pp) against
# real gloo to watch it stall was dropped on purpose. Provoking an unmatched
# isend leaves 8 ranks tearing down a half-finished gloo rendezvous, which
# aborts with a C++ stack trace and destabilises the rest of the suite when
# run under load. The stall path is covered deterministically by
# ``test_blocking_scatter_timeout_names_the_problem`` and the PP fan-out it
# would have exercised by
# ``test_without_skip_pp_the_leader_would_send_to_every_stage``.


# ===========================================================================
# Per-request blocking collectives: all_reduce_min / barrier must skip PP.
# ===========================================================================
# A second deadlock, same family, different call: the layerwise H2D load runs
# inside every stage's get_new_batch_prefill and calls all_reduce_min(local_ok)
# as an "all stages registered before pp0 launches" barrier. But the scheduler
# forwards a request to stage N+1 only after stage N's get_new_batch_prefill
# returns, so a blocking cross-stage reduction inside it waits for stages that
# cannot arrive until it returns -- circular wait. Observed live on GLM-5.3
# PP4/DP2 the instant FlexKV produced its first cache hit (DP1 PP0 parked in
# _pp_all_reduce_min_p2p -> recv, pp1..3 in _pp_recv_pyobj_from_prev_stage).
#
# Unlike reset(), this path DOES run on every stage, and producer_id must stay
# globally consistent (only pp0 launches the fan-out load with one counter_id,
# broadcast via the non-blocking LAYERWISE_MANIFEST scatter). So the fix is not
# "don't communicate" but "communicate only within the stage": skip_pp keeps
# the TP/CP reduction and drops the deadlocking PP hop.

def test_all_reduce_min_skip_pp_drops_pp_hop_in_pure_pp(monkeypatch):
    comm = _comm(4, pp_rank=0, world_rank=0)
    pp = MagicMock(name="_pp_all_reduce_min_p2p")
    bcast = MagicMock(name="_bcast_to_stage_members")
    monkeypatch.setattr(comm, "_pp_all_reduce_min_p2p", pp)
    monkeypatch.setattr(comm, "_bcast_to_stage_members", bcast)

    # Pure PP (tp==cp==1): skip_pp leaves nothing to do, value passes through.
    assert comm.all_reduce_min(1, skip_pp=True) == 1
    assert comm.all_reduce_min(0, skip_pp=True) == 0
    pp.assert_not_called()
    bcast.assert_not_called()


def test_all_reduce_min_without_skip_pp_would_do_pp_rendezvous(monkeypatch):
    """Guards the diagnosis: the old call really did a blocking PP hop."""
    comm = _comm(4, pp_rank=0, world_rank=0)
    pp = MagicMock(name="_pp_all_reduce_min_p2p")
    bcast = MagicMock(name="_bcast_to_stage_members")
    monkeypatch.setattr(comm, "_pp_all_reduce_min_p2p", pp)
    monkeypatch.setattr(comm, "_bcast_to_stage_members", bcast)

    comm.all_reduce_min(1)  # default skip_pp=False

    pp.assert_called_once()        # leader-side blocking rendezvous
    bcast.assert_called_once()     # ...then stage broadcast


def test_all_reduce_min_skip_pp_keeps_tp_reduction(monkeypatch):
    """attn_tp>1: those ranks are in lockstep, the TP MIN must still run."""
    import flexkv.integration.sglang.comm as comm_mod

    comm = _comm(4, attn_tp_size=2, pp_rank=0, attn_tp_rank=0, world_rank=0)
    comm.attn_tp_cpu_group = object()
    pp = MagicMock(name="_pp_all_reduce_min_p2p")
    monkeypatch.setattr(comm, "_pp_all_reduce_min_p2p", pp)
    monkeypatch.setattr(comm, "_bcast_to_stage_members", MagicMock())
    allreduce = MagicMock(name="dist.all_reduce")
    monkeypatch.setattr(comm_mod.dist, "all_reduce", allreduce)

    comm.all_reduce_min(5, skip_pp=True)

    allreduce.assert_called_once()   # TP reduction happened
    pp.assert_not_called()           # PP hop skipped


def test_barrier_skip_pp_drops_pp_hop(monkeypatch):
    comm = _comm(4, pp_rank=0, world_rank=0)
    pp = MagicMock(name="_pp_barrier_p2p")
    bcast = MagicMock(name="_bcast_to_stage_members")
    monkeypatch.setattr(comm, "_pp_barrier_p2p", pp)
    monkeypatch.setattr(comm, "_bcast_to_stage_members", bcast)

    comm.barrier(skip_pp=True)
    pp.assert_not_called()
    bcast.assert_not_called()

    comm.barrier()  # default keeps the PP hop
    pp.assert_called_once()
    bcast.assert_called_once()


# ---------------------------------------------------------------------------
# The real connector call sites for the layerwise load / sync load
# ---------------------------------------------------------------------------

class _RecordingLayerwiseCtx:
    """Records all_reduce_min / barrier / scatter calls from the load paths."""

    def __init__(self, *, needs_sync=True, needs_sync_without_pp=False,
                 is_sync_leader=True):
        self.needs_sync = needs_sync
        self.needs_sync_without_pp = needs_sync_without_pp
        self.is_sync_leader = is_sync_leader
        self.should_send_slot_mapping_to_remote = False
        self.world_rank = 0
        self.all_reduce_calls = []
        self.barrier_calls = []
        self.scatter_calls = []

    def all_reduce_min(self, value, *, skip_pp=False):
        self.all_reduce_calls.append({"value": value, "skip_pp": skip_pp})
        return value

    def barrier(self, *, skip_pp=False):
        self.barrier_calls.append({"skip_pp": skip_pp})

    def scatter(self, data, *, channel, blocking=False, skip_pp=False):
        self.scatter_calls.append(
            {"channel": channel, "blocking": blocking, "skip_pp": skip_pp}
        )
        return data


def test_layerwise_all_reduce_min_uses_skip_pp():
    """Every all_reduce_min reachable from start_load_kv_layerwise's preflight
    must pass skip_pp=True. Reverting connector.py:928/948 to the plain call
    (or the needs_sync gate) must fail here."""
    import flexkv.integration.sglang.connector as conn_mod

    src = inspect.getsource(conn_mod.FlexKVConnector.start_load_kv_layerwise)
    # Textual guard is intentional: executing the full layerwise path needs a
    # live KVManager + counter. We assert the call shape the fix guarantees.
    assert "all_reduce_min(local_ok, skip_pp=True)" in src
    assert "all_reduce_min(cleanup_ok, skip_pp=True)" in src
    assert "needs_sync_without_pp" in src
    # The manifest scatter must stay a PP broadcast (producer_id flows to every
    # stage); it must NOT have been turned into skip_pp.
    assert "LAYERWISE_MANIFEST" in src
    mani = src.split("LAYERWISE_MANIFEST")[1][:160]
    assert "skip_pp" not in mani, "manifest scatter must keep its PP hop"


def test_sync_load_barrier_uses_skip_pp():
    """retrieve_kv's post-load barrier must skip PP (same dormant deadlock)."""
    import flexkv.integration.sglang.connector as conn_mod

    src = inspect.getsource(conn_mod.FlexKVConnector.retrieve_kv)
    assert "barrier(skip_pp=True)" in src
    assert "needs_sync_without_pp" in src


# ---------------------------------------------------------------------------
# End-to-end on real gloo: pp0-first layerwise preflight must not wedge.
# ---------------------------------------------------------------------------

def _layerwise_sync_worker(rank, world, port, tp, pp, q):
    """Reproduce the hit path's cross-stage ordering: pp0 reaches the preflight
    collective first (and would block on a PP rendezvous), pp1..N arrive much
    later. With skip_pp the stages never rendezvous across PP, so pp0 proceeds
    regardless of how far behind the downstream stages are.
    """
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        import time as _time

        import torch.distributed as dist

        from flexkv.integration.sglang.comm import FlexKVComm

        dist.init_process_group("gloo", rank=rank, world_size=world,
                                timeout=timedelta(seconds=60))
        my_pp, tp_rank = divmod(rank, tp)
        my_group = None
        for t in range(tp):
            g = dist.new_group([s * tp + t for s in range(pp)])
            if t == tp_rank:
                my_group = g
        rank_info = SimpleNamespace(
            model_config=SimpleNamespace(
                pp_size=pp, attn_tp_size=1, attn_cp_size=1, attn_dp_size=tp),
            pp_rank=my_pp, attn_tp_rank=0, attn_cp_rank=0, pp_size_per_node=pp,
        )
        comm = FlexKVComm(rank_info=rank_info, world_rank=rank,
                          pp_group=my_group, world_group=dist.group.WORLD)

        # Pipeline skew: downstream stages are intentionally late, exactly as
        # when pp0 processes get_new_batch_prefill before forwarding the req.
        _time.sleep(my_pp * 0.5)

        # The connector's preflight agreement, as rewritten.
        if comm.needs_sync_without_pp:
            comm.all_reduce_min(1, skip_pp=True)
        q.put((rank, "ok", my_pp))
        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        q.put((rank, f"ERROR {type(e).__name__}: {e}", -1))


def test_pp_skewed_layerwise_preflight_does_not_deadlock_live():
    """PP4 x DP2, attn_tp=1, downstream stages lagging -- must not wedge."""
    tp, pp = 2, 4
    world = tp * pp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_layerwise_sync_worker,
                         args=(r, world, port, tp, pp, q))
             for r in range(world)]
    for p in procs:
        p.start()
    try:
        results = [q.get(timeout=120) for _ in range(world)]
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()

    for rank, status, *_ in results:
        assert status == "ok", f"rank {rank}: {status}"

