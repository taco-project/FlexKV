"""Focused tests for completion-time Mooncake radix publication."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import (
    CacheEngine,
    DeferredCacheInsert,
    DeferredPublishResult,
    GlobalCacheEngine,
    MooncakeLoadResult,
)
from flexkv.common.block import SequenceMeta
from flexkv.common.transfer import DeviceType
from dataclasses import replace


pytestmark = pytest.mark.unit


def _engine():
    return CacheEngine(
        device_type=DeviceType.CPU,
        num_total_blocks=32,
        tokens_per_block=1,
        evict_ratio=0.1,
    )


def _manager(engine):
    manager = object.__new__(GlobalCacheEngine)
    manager.cache_engines = {DeviceType.CPU: engine}
    return manager


def _sequence(tokens):
    return SequenceMeta(
        token_ids=np.asarray(tokens, dtype=np.int64),
        tokens_per_block=1,
    )


def _pending(
    engine,
    sequence,
    block_results,
    staged_start,
    requested_end,
    remote_start=None,
):
    physical_blocks = engine.take(requested_end - staged_start)
    return DeferredCacheInsert(
        device_type=DeviceType.CPU,
        sequence_meta=sequence,
        physical_blocks=physical_blocks,
        staged_start_block=staged_start,
        remote_start_block=(
            staged_start if remote_start is None else remote_start),
        requested_end_block=requested_end,
        load_result=MooncakeLoadResult(block_results=block_results),
    )


def test_deferred_insert_publishes_all_successful_blocks():
    engine = _engine()
    sequence = _sequence([1, 2, 3, 4])
    pending = _pending(
        engine, sequence, (True, True, True, True), 0, 4)

    _manager(engine)._commit_deferred_insert(pending)

    match = engine.match(sequence)
    assert match.num_matched_blocks == 4
    assert match.num_ready_matched_blocks == 4
    assert engine.index.total_unready_blocks() == 0
    assert engine.mempool.num_used_blocks == 4


def test_deferred_insert_stops_at_first_failed_block():
    engine = _engine()
    sequence = _sequence([1, 2, 3])
    pending = _pending(engine, sequence, (True, False, True), 0, 3)

    _manager(engine)._commit_deferred_insert(pending)

    match = engine.match(sequence)
    assert match.num_matched_blocks == 1
    assert match.num_ready_matched_blocks == 1
    assert engine.index.total_unready_blocks() == 0
    assert engine.mempool.num_used_blocks == 1


@pytest.mark.parametrize(("block_results", "expected_ready"), [
    ((True, False), 3),
    ((False, True), 2),
])
def test_deferred_insert_keeps_local_staging_before_remote_failure(
    block_results,
    expected_ready,
):
    """SSD-loaded staging before the remote span remains publishable."""
    engine = _engine()
    sequence = _sequence([1, 2, 3, 4])
    first_block = engine.take(1)
    engine.insert(
        sequence, first_block, num_insert_blocks=1, is_ready=True)
    pending = _pending(
        engine,
        sequence,
        block_results,
        staged_start=1,
        remote_start=2,
        requested_end=4,
    )

    _manager(engine)._commit_deferred_insert(pending)

    match = engine.match(sequence)
    assert match.num_ready_matched_blocks == expected_ready
    assert engine.index.total_unready_blocks() == 0
    assert engine.mempool.num_used_blocks == expected_ready


def test_deferred_insert_rematches_after_concurrent_save_split():
    """A save may split and extend the original match before GET completes."""
    engine = _engine()
    manager = _manager(engine)
    cached = _sequence([1, 2, 3, 4])
    loading = _sequence([1, 2, 30, 40])
    concurrent_save = _sequence([1, 2, 30, 50])

    engine.insert(cached, engine.take(4), is_ready=True)
    pending = _pending(engine, loading, (True, True), 2, 4)

    # The save splits the old [1,2,3,4] node and adds [30,50]. The GET's fresh
    # match is now three blocks, so its reserved block for 30 is redundant.
    save_match = engine.match(concurrent_save)
    assert save_match.num_matched_blocks == 2
    engine.insert(
        concurrent_save,
        engine.take(2),
        is_ready=True,
        match_result=save_match,
    )

    manager._commit_deferred_insert(pending)

    cached_match = engine.match(cached)
    loading_match = engine.match(loading)
    save_match = engine.match(concurrent_save)
    assert cached_match.num_ready_matched_blocks == 4
    assert loading_match.num_ready_matched_blocks == 4
    assert save_match.num_ready_matched_blocks == 4
    assert loading_match.physical_blocks[2] == save_match.physical_blocks[2]
    assert loading_match.physical_blocks[3] == pending.physical_blocks[1]
    assert engine.index.total_cached_blocks() == 7
    assert engine.index.total_unready_blocks() == 0
    assert engine.mempool.num_used_blocks == 7


def test_deferred_insert_recycles_staging_rejected_by_radix_guard():
    engine = _engine()
    sequence = _sequence([1, 2])
    pending = _pending(engine, sequence, (True, True), 0, 2)
    staged = pending.physical_blocks.copy()
    engine.insert = Mock(side_effect=RuntimeError(
        "radix insert conflict: target child already exists; "
        "rematch before retrying"))

    _manager(engine)._commit_deferred_insert(pending)

    assert engine.mempool._free_mask[staged].all()
    assert engine.mempool.num_used_blocks == 0


def test_deferred_insert_records_publish_result_on_success():
    engine = _engine()
    sequence = _sequence([1, 2, 3, 4])
    publish_result = DeferredPublishResult()
    pending = _pending(
        engine, sequence, (True, True, True, True), 0, 4)
    pending = replace(pending, publish_result=publish_result)

    _manager(engine)._commit_deferred_insert(pending)

    assert publish_result.published_remote_blocks == 4
    assert publish_result.failed is False
    assert publish_result.reason == "ok"


def test_deferred_insert_records_zero_publish_on_conflict():
    engine = _engine()
    sequence = _sequence([1, 2, 3])
    publish_result = DeferredPublishResult()
    pending = _pending(engine, sequence, (True, True, True), 0, 3)
    pending = replace(pending, publish_result=publish_result)
    staged = pending.physical_blocks.copy()
    engine.insert = Mock(side_effect=RuntimeError(
        "radix insert conflict: target child already exists; "
        "rematch before retrying"))

    node = _manager(engine)._commit_deferred_insert(pending)

    assert node is None
    assert publish_result.published_remote_blocks == 0
    assert publish_result.failed is False
    assert publish_result.reason == "insert_conflict"
    assert engine.mempool._free_mask[staged].all()


def test_callback_records_publish_failure_when_commit_raises():
    manager = object.__new__(GlobalCacheEngine)
    manager._cache_tree_lock = threading.RLock()
    manager._commit_deferred_insert = Mock(
        side_effect=RuntimeError("injected commit failure"))
    manager.cpu_cache_engine = SimpleNamespace(
        unlock=Mock(),
        set_ready=Mock(),
        recycle=Mock(),
    )
    publish_result = DeferredPublishResult()
    pending = SimpleNamespace(
        device_type=DeviceType.CPU,
        publish_result=publish_result,
    )

    manager._transfer_callback(
        node_to_unlock={DeviceType.CPU: (object(), 0)},
        buffer_to_free=None,
        deferred_inserts=[pending],
    )

    assert publish_result.published_remote_blocks == 0
    assert publish_result.failed is True
    assert publish_result.reason == "callback_error"


def test_callback_cleans_up_old_anchor_when_deferred_commit_fails():
    manager = object.__new__(GlobalCacheEngine)
    manager._cache_tree_lock = threading.RLock()
    manager._commit_deferred_insert = Mock(
        side_effect=RuntimeError("injected commit failure"))
    manager.cpu_cache_engine = SimpleNamespace(
        unlock=Mock(),
        set_ready=Mock(),
        recycle=Mock(),
    )
    anchor = object()
    buffer = np.asarray([7, 8], dtype=np.int64)
    pending = SimpleNamespace(device_type=DeviceType.CPU)

    manager._transfer_callback(
        node_to_unlock={DeviceType.CPU: (anchor, 2)},
        buffer_to_free={DeviceType.CPU: buffer},
        deferred_inserts=[pending],
    )

    manager._commit_deferred_insert.assert_called_once_with(pending)
    manager.cpu_cache_engine.unlock.assert_called_once_with(anchor)
    manager.cpu_cache_engine.set_ready.assert_called_once_with(anchor, True, 2)
    assert manager.cpu_cache_engine.recycle.call_count == 1
    recycled = manager.cpu_cache_engine.recycle.call_args.args[0]
    np.testing.assert_array_equal(recycled, buffer)
