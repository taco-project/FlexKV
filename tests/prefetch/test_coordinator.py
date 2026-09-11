"""Deterministic policy/ownership tests; no GPU, native extension, or sleeps."""

from dataclasses import replace
import random
from types import SimpleNamespace as NS

import pytest

from flexkv.prefetch.coordinator import PrefetchCoordinator
from flexkv.prefetch.types import PrefetchOptions, PrefetchHandle


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Backend:
    def __init__(self, start=0, target=20):
        self.start, self.target = start, target
        self.sent = []
        self.commits = []
        self.discards = []
        self.releases = []
        self.allocations = set()
        self.next_graph = 0
        self.metadata_ready = True
        self.reserve_hook = lambda: None
        self.capacity = True

    def begin(self, tokens, namespace, client_id, options):
        return object()

    def cancel_query(self, context):
        pass

    def result_end(self, context, committed):
        return committed

    def resolve(self, context):
        return (self.start, self.target) if self.metadata_ready else None

    def reserve(self, context, begin, target, options, available_bytes):
        self.reserve_hook()
        if not self.capacity or available_bytes < 1:
            return None
        end = min(target, begin + options.chunk_max_blocks, begin + available_bytes)
        chunk = NS(
            begin=begin,
            end=end,
            nbytes=end - begin,
            graph=NS(graph_id=self.next_graph),
            done=False,
            success=end - begin,
        )
        self.next_graph += 1
        self.allocations.add(chunk.graph.graph_id)
        return chunk

    def submit_batch(self, chunks):
        self.sent.extend(chunks)

    def complete(self, chunk, completion):
        chunk.done = completion.op_id == -1
        if hasattr(completion, "success"):
            chunk.success = completion.success

    def commit(self, context, chunk):
        self.allocations.remove(chunk.graph.graph_id)
        self.commits.append(chunk)
        return chunk.begin + chunk.success

    def discard(self, context, chunk):
        self.allocations.remove(chunk.graph.graph_id)
        self.discards.append(chunk)

    def release(self, context):
        assert context not in self.releases
        self.releases.append(context)


def setup(policy="wait_complete", **kwargs):
    clock, backend = Clock(), Backend()
    coordinator = PrefetchCoordinator(backend, clock=clock, max_reserved_bytes=20)
    options = PrefetchOptions(
        policy=policy, chunk_max_blocks=4, timeout_budget_s=5, **kwargs
    )
    handle = coordinator.start(list(range(20)), options)
    return coordinator, backend, clock, handle


def finish(coordinator, chunk, success=None):
    completion = NS(graph_id=chunk.graph.graph_id, op_id=-1)
    if success is not None:
        completion.success = success
    coordinator.on_completion(completion)


def test_wakeup_tracks_active_work_then_result_expiry():
    c, b, clock, h = setup()
    assert c.next_wakeup(0.002) == 0.002
    for _ in range(10):
        c.tick()
        for chunk in list(b.sent):
            finish(c, chunk)
    assert c.snapshot(h).terminal
    assert c.next_wakeup(0.002) == c.result_ttl_s
    clock.now = c.result_ttl_s
    assert c.next_wakeup(0.002) == 0
    c.tick()
    assert c.snapshot(h).state == "expired"
    assert c.next_wakeup(0.002) is None
    assert len(b.releases) == 1


def test_stopped_session_keeps_polling_until_inflight_drains():
    c, b, clock, h = setup("timeout")
    c.tick()
    clock.now = 5
    c.tick()
    assert c.snapshot(h).stop_reason == "deadline"
    assert c.next_wakeup(0.002) == 0.002
    finish(c, b.sent[0])
    c.tick()
    assert c.snapshot(h).terminal
    assert c.next_wakeup(0.002) == c.result_ttl_s


@pytest.mark.parametrize("policy", ["wait_complete", "timeout", "best_effort"])
def test_no_poll_needed_to_finish_all_chunks(policy):
    c, b, clock, h = setup(policy)
    for _ in range(10):
        c.tick()
        for chunk in list(b.sent):
            finish(c, chunk)
    s = c.snapshot(h)
    assert s.terminal and s.outcome == "complete"
    assert s.loaded_tokens == 20 and s.l3_loaded_spans == ((0, 20),)
    assert not b.allocations and not c.graphs and c.reserved_bytes == 0


@pytest.mark.parametrize(
    "policy,stops",
    [("wait_complete", False), ("timeout", False), ("best_effort", True)],
)
def test_demand_policy_and_drain(policy, stops):
    c, b, clock, h = setup(policy)
    c.tick()
    c.demand([h])
    c.tick()
    assert (c.snapshot(h).stop_reason == "demand") is stops
    assert not c.snapshot(h).terminal
    for chunk in b.sent:
        finish(c, chunk)
    c.tick()
    if stops:
        assert c.snapshot(h).terminal and c.snapshot(h).loaded_tokens == 4
        assert len(b.sent) == 1


def test_timeout_freezes_all_claimed_chunks_including_worker_queue():
    c, b, clock, h = setup("timeout")
    c.tick()
    c.tick()
    assert len(b.sent) == 2
    clock.now = 5
    c.tick()
    assert c.snapshot(h).sealed_submit_seq == 2
    finish(c, b.sent[1])
    c.tick()
    assert not c.snapshot(h).terminal and not b.commits
    finish(c, b.sent[0])
    c.tick()
    assert c.snapshot(h).terminal
    assert c.snapshot(h).loaded_tokens == 8
    assert len(b.sent) == 2


def test_timeout_between_reserve_and_claim_rolls_back():
    c, b, clock, h = setup("timeout")
    b.reserve_hook = lambda: setattr(clock, "now", 6)
    c.tick()
    assert not b.sent and not b.allocations
    assert c.snapshot(h).terminal and c.snapshot(h).stop_reason == "deadline"


@pytest.mark.parametrize("policy", ["timeout", "best_effort"])
def test_stop_during_metadata_has_no_buffer_to_drain(policy):
    c, b, clock, h = setup(policy)
    b.metadata_ready = False
    c.tick()
    clock.now = 6
    c.demand([h])
    c.tick()
    assert c.snapshot(h).terminal and c.snapshot(h).loaded_tokens == 0
    b.metadata_ready = True
    c.tick()
    assert not b.sent


def test_partial_second_chunk_does_not_count_success_after_hole():
    c, b, clock, h = setup(max_inflight_chunks=3)
    c.tick()
    c.tick()
    c.tick()
    finish(c, b.sent[2])
    c.tick()
    finish(c, b.sent[1], success=1)
    c.tick()
    assert not b.commits
    finish(c, b.sent[0])
    c.tick()
    assert c.snapshot(h).terminal and c.snapshot(h).loaded_tokens == 5
    assert [x.begin for x in b.commits] == [0, 4]
    assert [x.begin for x in b.discards] == [8]
    assert not b.allocations


def test_partial_read_reports_time_spent_draining_inflight(caplog):
    c, b, clock, h = setup()
    c.tick()
    c.tick()
    clock.now = 1
    finish(c, b.sent[0], success=1)
    c.tick()
    assert not c.snapshot(h).terminal
    clock.now = 3
    with caplog.at_level("INFO", logger="flexkv.prefetch.coordinator"):
        finish(c, b.sent[1])
        c.tick()
    assert c.snapshot(h).terminal and c.snapshot(h).loaded_tokens == 1
    assert "reason=partial_read" in caplog.text
    assert "drain_ms=2000.000" in caplog.text
    assert len(b.sent) == 2 and not b.allocations


def test_abort_release_does_not_recycle_inflight():
    c, b, clock, h = setup()
    c.tick()
    c.tick()
    c.release(h)
    c.release(h)
    assert len(b.allocations) == 2 and not b.releases
    for chunk in b.sent:
        finish(c, chunk)
    c.tick()
    assert c.snapshot(h).terminal and not c.snapshot(h).lease_valid
    assert len(b.releases) == 1 and not b.allocations


def test_reset_discards_old_epoch_data_after_drain():
    c, b, clock, h = setup()
    c.tick()
    c.tick()
    c.stop_all("reset")
    with pytest.raises(RuntimeError, match="closed"):
        c.start([1])
    for chunk in b.sent:
        finish(c, chunk)
    c.tick()
    assert not b.commits and len(b.discards) == 2
    assert c.snapshot(h).outcome == "aborted"
    assert c.snapshot(h).loaded_tokens == 0


def test_active_resource_records_never_expire():
    c, b, clock, h = setup()
    c.tick()
    clock.now = 100000
    c.reap()
    assert h in c.sessions and b.allocations
    c.stop(h)
    finish(c, b.sent[0])
    c.tick()
    clock.now += 61
    c.reap()
    assert h not in c.sessions and len(b.releases) == 1


def test_global_credit_and_fairness():
    clock, backend = Clock(), Backend(target=12)
    c = PrefetchCoordinator(backend, clock=clock, max_reserved_bytes=8)
    hs = [c.start([1] * 12, PrefetchOptions(chunk_max_blocks=4)) for _ in range(2)]
    c.tick()
    c.tick()
    assert c.reserved_bytes == 8
    assert [c.snapshot(h).submitted_chunks for h in hs] == [1, 1]
    assert len(backend.sent) == 2


def test_capacity_terminal_no_deadlock():
    c, b, clock, h = setup()
    b.capacity = False
    c.tick()
    assert c.snapshot(h).terminal and c.snapshot(h).stop_reason == "capacity"


def test_stale_handle_and_late_completion():
    c, b, clock, h = setup()
    with pytest.raises(KeyError):
        c.snapshot(PrefetchHandle("old", h.session_id))
    c.tick()
    c.stop(h)
    finish(c, b.sent[0])
    c.tick()
    assert not c.on_completion(NS(graph_id=b.sent[0].graph.graph_id, op_id=-1))


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_timeout(value):
    with pytest.raises(ValueError):
        PrefetchOptions(timeout_budget_s=value).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(policy="waite_complete"),
        dict(chunk_max_blocks=0),
        dict(max_inflight_chunks=-1),
        dict(chunk_max_blocks=True),
        dict(candidate_start_token=-2),
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PrefetchOptions(**kwargs).validate()


@pytest.mark.parametrize("seed", range(50))
def test_randomized_completion_stop_schedules(seed):
    randomizer = random.Random(seed)
    c, b, clock, h = setup(max_inflight_chunks=3)
    stopped = False
    for _ in range(100):
        c.tick()
        if not stopped and randomizer.random() < 0.08:
            c.stop(h)
            stopped = True
            frozen = len(b.sent)
        pending = [x for x in b.sent if not x.done]
        if pending:
            chunk = randomizer.choice(pending)
            finish(c, chunk, randomizer.randint(0, chunk.end - chunk.begin))
        if stopped:
            assert len(b.sent) == frozen
        assert c.reserved_bytes <= 20
        if c.snapshot(h).terminal:
            break
    c.tick()
    s = c.snapshot(h)
    assert s.terminal and not b.allocations and c.reserved_bytes == 0
    cursor = 0
    for chunk in b.commits:
        assert chunk.begin == cursor
        cursor += chunk.success
    assert s.loaded_tokens == cursor


def test_expired_result_is_safe_empty_terminal_and_tombstones_are_bounded():
    clock, backend = Clock(), Backend(target=0)
    c = PrefetchCoordinator(backend, clock=clock, max_sessions=2, result_ttl_s=1)
    handles = []
    for _ in range(20):
        h = c.start([1])
        handles.append(h)
        c.tick()
        clock.now += 2
        c.reap()
        snapshot = c.snapshot(h)
        assert snapshot.terminal and snapshot.outcome == "expired"
        assert not snapshot.lease_valid and snapshot.loaded_tokens == 0
        c.demand([h])
        c.release(h)
    assert len(c.expired) == 8 and not c.sessions
    with pytest.raises(KeyError):
        c.snapshot(handles[0])
