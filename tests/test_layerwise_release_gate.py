"""Cross-stage release gate for the layerwise producer ring.

Live failure (GLM-5.3, PP4, 3-slot ring): pp0 runs up to pp_size-1 batches
ahead of the last stage, so it wrapped around and relaunched slot 2 for a new
batch while PP3 was still consuming slot 2's previous batch. The worker posted
the new batch's 20 per-layer signals into PP3's slot-2 eventfds; PP3 then hit
its last layer, its release-time drain swallowed all 20 (``drained=20``), and
PP3 blocked forever on the new batch at ``layer=0``.

Fix: pp0 only reuses a slot once every rank of the replica has published a
release of that slot's previous batch (``LayerwiseReleaseBoard``). It skips
busy slots instead of waiting, and when none is free the restore is refused
on every rank alike, so the tokens are recomputed.
"""

import inspect
import logging

import pytest

from flexkv.integration.sglang import comm as comm_mod
from flexkv.integration.sglang.comm import (
    FlexKVLayerDoneCounter,
    LayerwiseReleaseBoard,
    resolve_layerwise_num_counters,
)

NUM_LAYERS = 3
LEADER, FOLLOWER = 0, 6


class _DictStore:
    """Minimal stand-in for a torch.distributed Store."""

    def __init__(self):
        self.data = {}
        self.reads = 0

    def set(self, key, value):
        self.data[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        self.reads += 1
        return self.data[key]

    def check(self, keys):
        return all(k in self.data for k in keys)

    def multi_get(self, keys):
        self.reads += 1
        return [self.data[k] for k in keys]


def _post(counter, slot):
    """The worker finishing every layer of one transfer on ``slot``."""
    import os

    for fd in counter.events[slot].load_event_fds:
        os.write(fd, (1).to_bytes(8, "little"))


def _make_pair(store=None, gated=True, num_counters=3):
    store = store if store is not None else _DictStore()
    peers = [LEADER, FOLLOWER]
    leader_board = follower_board = None
    if gated:
        leader_board = LayerwiseReleaseBoard(
            store, leader_rank=LEADER, my_rank=LEADER, peer_ranks=peers
        )
        follower_board = LayerwiseReleaseBoard(
            store, leader_rank=LEADER, my_rank=FOLLOWER, peer_ranks=peers
        )
    leader = FlexKVLayerDoneCounter(
        NUM_LAYERS, num_counters, world_rank=LEADER, release_board=leader_board
    )
    follower = FlexKVLayerDoneCounter(
        NUM_LAYERS, num_counters, world_rank=FOLLOWER, release_board=follower_board
    )
    for c in (leader, follower):
        c.wait_timeout_s = 0.3
    return leader, follower, store


def _leader_open_and_run(leader, follower, generation, task_id):
    """pp0: select, launch (worker posts to every stage), run its own forward."""
    slot = leader.update_producer()
    leader.note_batch_opened(slot, generation)
    leader.register_task(task_id, slot)
    leader.set_consumer(task_id)
    _post(leader, slot)
    _post(follower, slot)
    for layer in range(NUM_LAYERS):
        leader.wait_until(layer)
    return slot


def _follower_register(follower, slot, generation, task_id):
    follower.register_task_with_explicit_counter_id(
        task_id, slot, generation=generation
    )
    follower.set_consumer(task_id)


def test_ungated_ring_reproduces_the_live_lost_wakeup():
    """Without the gate the 3-slot ring loses the follower's wakeup."""
    leader, follower, _ = _make_pair(gated=False)

    s0 = _leader_open_and_run(leader, follower, 0, 100)
    _follower_register(follower, s0, 0, 100)
    follower.wait_until(0)
    follower.wait_until(1)  # last stage mid-forward on batch 0

    s1 = _leader_open_and_run(leader, follower, 1, 101)
    s2 = _leader_open_and_run(leader, follower, 2, 102)
    s3 = _leader_open_and_run(leader, follower, 3, 103)
    assert s3 == s0, "ring wrapped onto the slot the follower still uses"

    follower.wait_until(2)  # release drains batch 3's signals
    for slot, gen, task in ((s1, 1, 101), (s2, 2, 102)):
        _follower_register(follower, slot, gen, task)
        for layer in range(NUM_LAYERS):
            follower.wait_until(layer)

    _follower_register(follower, s3, 3, 103)
    with pytest.raises(TimeoutError):
        follower.wait_until(0)


def test_gate_refuses_reuse_until_follower_releases_then_no_wakeup_is_lost():
    leader, follower, _ = _make_pair(gated=True)

    s0 = _leader_open_and_run(leader, follower, 0, 100)
    _follower_register(follower, s0, 0, 100)
    follower.wait_until(0)
    follower.wait_until(1)

    s1 = _leader_open_and_run(leader, follower, 1, 101)
    s2 = _leader_open_and_run(leader, follower, 2, 102)

    # Every slot is still owned by the lagging stage: refuse, don't block.
    with pytest.raises(RuntimeError, match="released on every rank"):
        leader.update_producer()

    follower.wait_until(2)  # releases batch 0 on slot 0 and publishes it

    s3 = _leader_open_and_run(leader, follower, 3, 103)
    assert s3 == s0

    for slot, gen, task in ((s1, 1, 101), (s2, 2, 102), (s3, 3, 103)):
        _follower_register(follower, slot, gen, task)
        for layer in range(NUM_LAYERS):
            follower.wait_until(layer)  # must not time out
    assert follower.consumer_index == -1


def test_gate_skips_a_busy_slot_and_takes_the_next_released_one():
    leader, follower, _ = _make_pair(gated=True)

    s0 = _leader_open_and_run(leader, follower, 0, 100)
    s1 = _leader_open_and_run(leader, follower, 1, 101)
    s2 = _leader_open_and_run(leader, follower, 2, 102)
    # Follower finished batches 1 and 2 but batch 0 is still in its forward.
    _follower_register(follower, s0, 0, 100)
    follower.wait_until(0)
    # Follower-side order is enforced elsewhere; here only the publishes matter.
    follower.release_board.publish(s1, 1)
    follower.release_board.publish(s2, 2)

    assert leader.update_producer() == s1


def test_slot_never_used_needs_no_release():
    leader, follower, store = _make_pair(gated=True)
    assert leader.update_producer() == 0
    assert store.reads == 0


def test_release_is_published_on_rollback_and_reset():
    leader, follower, store = _make_pair(gated=True)
    _follower_register(follower, 1, 7, 200)
    follower.rollback_prepared_transfer(200, 1, transfer_added=True)
    assert store.data[f"flexkv/lw_release/{LEADER}/{FOLLOWER}/1"] == b"7"

    _follower_register(follower, 2, 8, 201)
    follower.reset()
    assert store.data[f"flexkv/lw_release/{LEADER}/{FOLLOWER}/2"] == b"8"


def test_leader_never_publishes_and_caches_released_peers():
    store = _DictStore()
    board = LayerwiseReleaseBoard(
        store, leader_rank=LEADER, my_rank=LEADER, peer_ranks=[LEADER, 2, 4]
    )
    board.publish(0, 5)
    assert store.data == {}

    assert board.laggards(0, 5) == [(2, -1), (4, -1)]
    store.set(f"flexkv/lw_release/{LEADER}/2/0", "5")
    store.set(f"flexkv/lw_release/{LEADER}/4/0", "6")
    assert board.laggards(0, 5) == []
    reads = store.reads
    assert board.laggards(0, 5) == []
    assert store.reads == reads, "satisfied peers must not be re-read"


def test_board_works_on_a_real_torch_store():
    torch_dist = pytest.importorskip("torch.distributed")
    store = torch_dist.HashStore()
    leader_board = LayerwiseReleaseBoard(
        store, leader_rank=LEADER, my_rank=LEADER, peer_ranks=[LEADER, FOLLOWER]
    )
    follower_board = LayerwiseReleaseBoard(
        store, leader_rank=LEADER, my_rank=FOLLOWER, peer_ranks=[LEADER, FOLLOWER]
    )
    assert leader_board.laggards(3, 0) == [(FOLLOWER, -1)]
    follower_board.publish(3, 0)
    assert leader_board.laggards(3, 0) == []
    assert leader_board.laggards(3, 1) == [(FOLLOWER, 0)]


def test_publish_failure_does_not_break_the_forward(caplog):
    class _Broken(_DictStore):
        def set(self, key, value):
            raise ConnectionError("store down")

    board = LayerwiseReleaseBoard(
        _Broken(), leader_rank=LEADER, my_rank=FOLLOWER, peer_ranks=[FOLLOWER]
    )
    with caplog.at_level(logging.ERROR):
        board.publish(0, 1)
    assert "publish failed" in caplog.text


@pytest.mark.parametrize(
    "env, pp_size, expected, warns",
    [
        (None, 1, 16, False),
        (None, 4, 16, False),
        (None, 8, 32, False),
        ("24", 4, 24, False),
        ("16", 4, 16, False),
        ("8", 4, 16, True),
        ("20", 8, 32, True),
        ("1", 1, 16, True),
        ("abc", 4, 16, True),
    ],
)
def test_resolve_layerwise_num_counters(monkeypatch, caplog, env, pp_size, expected, warns):
    if env is None:
        monkeypatch.delenv("FLEXKV_LAYERWISE_NUM_COUNTERS", raising=False)
    else:
        monkeypatch.setenv("FLEXKV_LAYERWISE_NUM_COUNTERS", env)
    with caplog.at_level(logging.WARNING, logger=comm_mod.logger.name):
        assert resolve_layerwise_num_counters(pp_size) == expected
    assert (f"reset to {expected}" in caplog.text) == warns


def test_counter_default_ring_is_not_the_old_triple_buffer(monkeypatch):
    monkeypatch.delenv("FLEXKV_LAYERWISE_NUM_COUNTERS", raising=False)
    assert FlexKVLayerDoneCounter(NUM_LAYERS).num_counters >= 16


def test_connector_wires_generation_and_board():
    from flexkv.integration.sglang import connector as conn_mod

    src = inspect.getsource(conn_mod.FlexKVConnector.start_load_kv_layerwise)
    assert "note_batch_opened" in src
    assert "generation=generation" in src
    init_src = inspect.getsource(conn_mod.FlexKVConnector.__init__)
    assert "build_layerwise_release_board" in init_src
    assert "resolve_layerwise_num_counters" in init_src
    assert comm_mod.build_layerwise_release_board is not None
