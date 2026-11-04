"""Scheduler-side prefetch/held-GET handoff and late-admission ownership."""

from array import array
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from flexkv.common.request import KVResponseStatus
from flexkv.integration.sglang.comm import FlexKVScatterChannel
from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.prefetch.types import PrefetchOptions

pytestmark = pytest.mark.unit


def _resolved(value=None):
    future = Future()
    future.set_result(value)
    return future


class _Manager:
    """Controllable admissions with observable source leases and held-GET pins."""

    def __init__(self, *, chunked=True, delay_admission=False, delay_progress=False):
        self.chunked = chunked
        self.delay_admission = delay_admission
        self.delay_progress = delay_progress
        self.progress_batches = []
        self.admissions = []
        self.lookups = []
        self.cancellations = []
        self.leases = set()
        self.pins = set()
        self.events = []
        self.prefetch_control_async = Mock(side_effect=self._control)
        self.get_match_async = Mock(side_effect=self._get_async)
        self.cancel_async = Mock(side_effect=self._cancel_async)
        self.get_match = Mock(return_value=(91, np.array([False, False, True, True])))
        self.cancel = Mock(side_effect=self._cancel)

    def _control(self, action, **kwargs):
        self.events.append((action, get_ident()))
        if action in {"start", "legacy_start"}:
            handle = len(self.admissions) + 1
            future = Future()
            self.admissions.append((handle, future))
            if not self.delay_admission:
                self.admit(handle)
            return future
        if action in {"progress", "legacy_progress"}:
            handles = kwargs["handles"] if self.chunked else kwargs["task_ids"]
            if self.chunked:
                snapshot = SimpleNamespace(
                    terminal=True, lease_valid=True, loaded_tokens=2,
                    l3_loaded_spans=((2, 4),),
                )
            else:
                snapshot = SimpleNamespace(
                    status=KVResponseStatus.SUCCESS,
                    return_mask=np.array([False, False, True, True]),
                )
            future = Future()
            self.progress_batches.append(({handle: snapshot for handle in handles}, future))
            if not self.delay_progress:
                self.complete_progress(len(self.progress_batches) - 1)
            return future
        if action in {"release", "legacy_release"}:
            handle = kwargs["handle"] if self.chunked else kwargs["task_ids"][0]
            self.leases.remove(handle)
            return _resolved()
        assert action == "drained"
        return _resolved(not self.leases)

    def admit(self, handle):
        self.leases.add(handle)
        self.admissions[handle - 1][1].set_result(handle)

    def complete_progress(self, index=0, *, terminal=True):
        snapshots, future = self.progress_batches[index]
        if not terminal:
            assert self.chunked
            for snapshot in snapshots.values():
                snapshot.terminal = False
        future.set_result(snapshots)

    def _get_async(self, **kwargs):
        future = Future()
        self.lookups.append(future)
        return future

    def complete_lookup(self, index=0, *, task_id=41, hit=True):
        mask = np.array([False, False, hit, hit])
        if hit and task_id >= 0:
            self.pins.add(task_id)
        self.lookups[index].set_result((task_id, mask))

    def _cancel_async(self, task_ids):
        self.events.append(("cancel_async", get_ident()))
        future = Future()
        self.cancellations.append((task_ids, future))
        return future

    def complete_cancel(self, index=0):
        task_ids, future = self.cancellations[index]
        self.pins.difference_update(task_ids)
        self.events.append(("cancel_complete", get_ident()))
        future.set_result(None)

    def _cancel(self, task_ids):
        self.pins.difference_update(task_ids)


def _connector(manager, *, leader=True, scatter=None):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 2
    connector._prefetch_enabled = connector._queued_prefetch = True
    connector._chunked_prefetch = manager.chunked
    connector._chunked_prefetch_options = PrefetchOptions(policy="timeout")
    connector._swa_kv_pool = object()
    connector.kv_manager = manager
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=leader, needs_sync=scatter is not None, scatter=scatter,
    )
    for name in (
        "_prefetch_sessions", "_prefetch_result_sessions", "_prepared_lookups",
        "_prefetch_loaded_tokens", "_prefetch_loaded_spans", "_pending_lookups",
        "_pending_lookup_contexts", "_prefetch_contexts", "_ongoing_prefetches",
        "_prefetch_planned_tokens", "_inflight_loads", "_launched_load_contexts",
        "_inflight_stores", "_inflight_store_contexts",
    ):
        setattr(connector, name, {})
    connector._completed_layerwise = []
    connector._launched_load_tids = []
    connector.layer_done_counter = None
    connector._new_op_context = Mock(side_effect=lambda *a, **kw: SimpleNamespace(
        task_id=-1, operation="lookup",
    ))
    connector._log_cache_op = Mock()
    return connector


def _prepare(connector, rid="r", *, tokens=None, namespace=None):
    tokens = array("q", [1, 2, 3, 4]) if tokens is None else tokens
    connector.prefetch_async(
        rid, tokens, namespace=namespace, candidate_start_token=2,
    )
    assert connector.check_prefetch_progress(rid) is False
    connector._poll_queued_prefetch()
    return tokens


@pytest.mark.parametrize("chunked", [True, False])
def test_pending_match_does_not_gate_prefetch_ready_or_become_a_miss(chunked):
    manager = _Manager(chunked=chunked)
    connector = _connector(manager)
    tokens = _prepare(connector, namespace=["tenant", "salt"])
    assert connector.check_prefetch_progress("r") is True
    assert connector.pop_prefetch_loaded_tokens("r") == 2
    assert manager.leases == {1}
    connector.release_pending("r")
    assert "r" in connector._prepared_lookups

    entered = Event()
    original_result = connector._prepared_lookups["r"].future.result

    def wait_for_match(*args, **kwargs):
        entered.set()
        return original_result(*args, **kwargs)

    connector._prepared_lookups["r"].future.result = wait_for_match
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            connector.lookup_kv, tokens, torch.tensor([False, False, True, True]),
            rid="r", namespace=["tenant", "salt"],
        )
        try:
            assert entered.wait(2)
            assert not result.done()
            assert manager.leases == {1} and not manager.pins
        finally:
            manager.complete_lookup()
        assert result.result(timeout=2) == (41, 2)
    assert connector._pending_lookups == {"r": 41}
    assert not connector._prepared_lookups and not manager.leases
    connector.release_pending("r")
    assert not manager.pins


@pytest.mark.parametrize("changed", ["prefix", "mask_hole", "tokens", "salt", "empty_namespace"])
def test_changed_complete_query_retires_old_held_get_and_uses_normal_lookup(changed):
    manager = _Manager()
    connector = _connector(manager)
    namespace = None if changed == "empty_namespace" else ["tenant", "salt"]
    tokens = _prepare(connector, namespace=namespace)
    manager.complete_lookup()
    assert connector.check_prefetch_progress("r")
    mask = np.array([False, False, True, True])
    if changed == "prefix":
        mask[2] = False
    elif changed == "mask_hole":
        mask[:] = [False, True, False, True]  # Same count, different query.
    elif changed == "tokens":
        tokens[0] = 1001  # Same length, different multimodal/retraction identity.
    elif changed == "salt":
        namespace[1] = "other-salt"
    else:
        namespace = []

    assert connector.lookup_kv(tokens, mask, rid="r", namespace=namespace) == (91, 2)
    manager.get_match.assert_called_once()
    manager.cancel_async.assert_called_once_with([41])
    assert not connector._prepared_lookups
    assert manager.leases == {1} and manager.pins == {41}
    manager.complete_cancel()
    assert not manager.pins and not manager.leases
    assert connector._pending_lookups == {"r": 91}
    assert [event[0] for event in manager.events][-3:] == [
        "cancel_async", "cancel_complete", "release",
    ]


def test_query_snapshot_is_immutable_and_reused_without_another_token_copy():
    manager = _Manager()
    connector = _connector(manager)
    tokens = array("q", [1, 2, 3, 4])
    namespace = ["tenant", "salt"]
    connector.prefetch_async("r", tokens, namespace=namespace, candidate_start_token=2)
    start_kwargs = manager.prefetch_control_async.call_args.kwargs
    snapshot = start_kwargs["token_ids"]
    assert snapshot.dtype == np.int64 and snapshot.flags.c_contiguous
    assert not snapshot.flags.writeable
    assert not np.shares_memory(snapshot, np.asarray(tokens))
    tokens[0] = 999
    namespace[1] = "changed"
    assert connector.check_prefetch_progress("r") is False
    assert connector.check_prefetch_progress("r") is True
    query = manager.get_match_async.call_args.kwargs
    assert query["token_ids"] is snapshot
    assert not query["token_mask"].flags.writeable
    np.testing.assert_array_equal(snapshot, [1, 2, 3, 4])
    assert query["namespace"] == start_kwargs["namespace"] == ["tenant", "salt"]
    manager.complete_lookup()
    assert connector.check_prefetch_progress("r")
    assert connector.lookup_kv(
        array("q", [1, 2, 3, 4]), query["token_mask"], rid="r",
        namespace=["tenant", "salt"],
    ) == (41, 2)
    manager.get_match.assert_not_called()
    connector.release_pending("r")


def test_one_progress_pass_prepares_every_terminal_prefetch():
    manager = _Manager()
    connector = _connector(manager)
    connector.prefetch_async("a", np.arange(4), candidate_start_token=2)
    connector.prefetch_async("b", np.arange(4) + 10, candidate_start_token=2)
    assert connector.check_prefetch_progress("a") is False
    assert connector.check_prefetch_progress("a") is True
    assert manager.get_match_async.call_count == 2
    assert len(manager.lookups) == 2 and manager.leases == {1, 2}
    for index, rid in enumerate(("a", "b")):
        manager.complete_lookup(index, task_id=41 + index)
        assert connector.check_prefetch_progress(rid)
        connector.cancel_prefetch(rid)
        manager.complete_cancel(index)
    assert not manager.leases and not manager.pins


@pytest.mark.parametrize("stage", ["pending", "completed", "published", "published_pending"])
@pytest.mark.parametrize("reset", [False, True], ids=["abort", "reset"])
def test_abort_and_reset_wait_for_late_get_cleanup_before_releasing_source(stage, reset):
    manager = _Manager()
    connector = _connector(manager)
    _prepare(connector)
    if stage not in ("pending", "published_pending"):
        manager.complete_lookup()
    if stage.startswith("published"):
        assert connector.check_prefetch_progress("r")
    retired = Event()
    retire = connector._retire_prefetch_admission

    def observe_retirement(*args):
        retire(*args)
        retired.set()

    connector._retire_prefetch_admission = observe_retirement
    with ThreadPoolExecutor(max_workers=1) as pool:
        reset_result = pool.submit(connector.reset) if reset else None
        if not reset:
            connector.cancel_prefetch("r")
        assert retired.wait(2)
        assert manager.leases == {1}
        assert not manager.lookups[0].cancelled()
        if reset:
            assert not reset_result.done()
        if stage in ("pending", "published_pending"):
            manager.cancel_async.assert_not_called()
            # This completion runs outside the scheduler/reset thread.
            manager.complete_lookup()
        manager.cancel_async.assert_called_once_with([41])
        assert manager.leases == {1} and manager.pins == {41}
        manager.complete_cancel()
        if reset:
            reset_result.result(timeout=2)
        else:
            connector._drain_queued_prefetch()
    assert not manager.leases and not manager.pins
    assert not connector._prefetch_sessions and not connector._prepared_lookups
    assert not connector._prefetch_result_sessions
    manager.cancel.assert_not_called()
    assert [name for name, _ in manager.events].index("cancel_complete") < [
        name for name, _ in manager.events
    ].index("release")


def test_abort_before_prefetch_admission_retires_the_late_source_without_get():
    manager = _Manager(delay_admission=True)
    connector = _connector(manager)
    connector.prefetch_async("r", np.arange(4), candidate_start_token=2)
    assert connector.check_prefetch_progress("r") is False
    connector.cancel_prefetch("r")
    assert not manager.admissions[0][1].cancelled()
    manager.admit(1)
    connector._drain_queued_prefetch()
    manager.get_match_async.assert_not_called()
    manager.cancel_async.assert_not_called()
    assert not manager.leases


@pytest.mark.parametrize("failed", [False, True], ids=["zero_hit", "failed_admission"])
def test_aborted_empty_or_failed_get_releases_source_without_cancelling_completed_task(failed):
    manager = _Manager()
    connector = _connector(manager)
    _prepare(connector)
    connector.cancel_prefetch("r")
    if failed:
        manager.lookups[0].set_exception(RuntimeError("GET admission failed"))
    else:
        manager.complete_lookup(hit=False)
    connector._drain_queued_prefetch()
    manager.cancel_async.assert_not_called()
    assert not manager.leases and not manager.pins


def test_failed_get_cancellation_keeps_source_and_fails_reset_closed():
    manager = _Manager()
    connector = _connector(manager)
    _prepare(connector)
    manager.complete_lookup()
    assert connector.check_prefetch_progress("r")
    connector.cancel_prefetch("r")
    manager.cancellations[0][1].set_exception(RuntimeError("cancel failed"))
    with pytest.raises(RuntimeError, match="FlexKV prefetch reset failed: cancel failed"):
        connector.reset()
    assert manager.leases == {1} and manager.pins == {41}
    assert "release" not in [name for name, _ in manager.events]


class _Mailbox:
    def __init__(self):
        self.messages = deque()
        self.scheduler = get_ident()
        self.channels = []

    def leader(self, payload, *, channel, **kwargs):
        assert get_ident() == self.scheduler
        self.messages.append((channel, payload.copy()))
        self.channels.append(channel)
        return payload

    def follower(self, payload, *, channel, **kwargs):
        assert get_ident() == self.scheduler
        expected, result = self.messages.popleft()
        assert channel == expected
        return result


@pytest.mark.parametrize("submission_error", [False, True])
def test_get_failure_keeps_lookup_error_behavior_and_follower_collective_order(submission_error):
    mailbox = _Mailbox()
    manager = _Manager()
    leader = _connector(manager, scatter=mailbox.leader)
    follower_manager = _Manager()
    follower = _connector(follower_manager, leader=False, scatter=mailbox.follower)
    if submission_error:
        manager.get_match_async.side_effect = RuntimeError("GET admission failed")
    for connector in (leader, follower):
        connector.prefetch_async("r", np.arange(4), candidate_start_token=2)
    for connector in (leader, follower):
        assert connector.check_prefetch_progress("r") is False
    if not submission_error:
        for connector in (leader, follower):
            assert connector.check_prefetch_progress("r") is True
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(
                manager.lookups[0].set_exception, RuntimeError("GET admission failed"),
            ).result(timeout=2)
        assert not mailbox.messages  # Future completion performed no collective.
    for connector in (leader, follower):
        assert connector.check_prefetch_progress("r") is True
        connector.release_pending("r")
    assert manager.leases == {1}
    for connector in (leader, follower):
        assert connector.lookup_kv(
            np.arange(4), np.array([False, False, True, True]), rid="r",
        ) == (-1, 0)
    manager.get_match.assert_not_called()
    manager.cancel_async.assert_not_called()
    assert leader._log_cache_op.call_args.args[1:3] == ("complete", "failed")
    assert leader._log_cache_op.call_args.kwargs["error"] == "GET admission failed"
    assert not mailbox.messages and not manager.leases
    assert mailbox.channels[0] == FlexKVScatterChannel.PREFETCH_START
    assert mailbox.channels[-1] == FlexKVScatterChannel.LOOKUP
    assert not follower_manager.prefetch_control_async.mock_calls
    assert not follower_manager.get_match_async.mock_calls
    assert not follower_manager.get_match.mock_calls


@pytest.mark.parametrize("chunked", [True, False])
def test_progress_completion_prepares_get_while_scheduler_is_paused(chunked):
    manager = _Manager(chunked=chunked, delay_progress=True)
    connector = _connector(manager)
    tokens = array("q", [1, 2, 3, 4])
    connector.prefetch_async("r", tokens, candidate_start_token=2)
    assert connector.check_prefetch_progress("r") is False
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(manager.complete_progress).result(timeout=2)
        # No scheduler poll between progress completion and GET completion.
        pool.submit(manager.complete_lookup).result(timeout=2)
    assert manager.leases == {1} and manager.pins == {41}
    assert connector.check_prefetch_progress("r") is True
    assert connector.lookup_kv(
        tokens, np.array([False, False, True, True]), rid="r",
    ) == (41, 2)
    assert not manager.leases
    connector.release_pending("r")
    assert not manager.pins


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("reset", [False, True], ids=["abort", "reset"])
def test_retirement_waits_for_inflight_progress_and_its_possible_get(terminal, reset):
    manager = _Manager(delay_progress=True)
    connector = _connector(manager)
    connector.prefetch_async("r", np.arange(4), candidate_start_token=2)
    assert connector.check_prefetch_progress("r") is False
    retired = Event()
    retire = connector._retire_prefetch_admission

    def observe_retirement(*args):
        retire(*args)
        retired.set()

    connector._retire_prefetch_admission = observe_retirement
    with ThreadPoolExecutor(max_workers=1) as pool:
        reset_result = pool.submit(connector.reset) if reset else None
        if not reset:
            connector.cancel_prefetch("r")
        assert retired.wait(2)
        assert manager.leases == {1} and not manager.pins
        manager.complete_progress(terminal=terminal)
        if terminal:
            assert manager.leases == {1}
            manager.complete_lookup()
            assert manager.pins == {41} and manager.leases == {1}
            manager.complete_cancel()
        if reset:
            reset_result.result(timeout=2)
        else:
            connector._drain_queued_prefetch()
    assert not manager.leases and not manager.pins


def test_nonterminal_progress_does_not_publish_a_miss_or_lose_later_get():
    manager = _Manager(delay_progress=True)
    connector = _connector(manager)
    tokens = np.arange(4)
    connector.prefetch_async("r", tokens, candidate_start_token=2)
    assert connector.check_prefetch_progress("r") is False
    manager.complete_progress(terminal=False)
    assert connector.check_prefetch_progress("r") is False
    assert manager.leases == {1} and not manager.pins
    manager.complete_progress(1)
    manager.complete_lookup()
    assert connector.check_prefetch_progress("r") is True
    assert connector.lookup_kv(
        tokens, np.array([False, False, True, True]), rid="r",
    ) == (41, 2)
    assert not manager.leases
    connector.release_pending("r")
    assert not manager.pins
