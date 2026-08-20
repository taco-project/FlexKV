"""Concurrency tests for completion-time Mooncake PUT publication."""

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


pytest.importorskip("flexkv.c_ext")

from flexkv.cache.cache_engine import (
    CacheEngine,
    CacheStrategy,
    GlobalCacheEngine,
)
from flexkv.cache.radixtree import MatchResult
from flexkv.common.block import SequenceMeta
from flexkv.common.transfer import DeviceType, TransferType
from flexkv.common.type import MatchResultAccel


pytestmark = pytest.mark.unit


class _MooncakeRemoteStub:
    """Key-addressed remote tier: block ids and radix mutation are no-ops."""

    def __init__(self, kv_hit: int = 0):
        self.kv_hit = kv_hit

    def match(self, sequence_meta):
        hit = min(self.kv_hit, sequence_meta.num_blocks)
        return MatchResultAccel(
            num_ready_matched_blocks=hit,
            num_matched_blocks=hit,
            kv_matched_blocks=hit,
            physical_blocks=np.arange(hit, dtype=np.int64),
            matched_pos="global",
        )

    def take(self, num_required_blocks, protected_node=None, strict=True):
        del protected_node, strict
        return np.zeros(num_required_blocks, dtype=np.int64)

    def recycle(self, physical_blocks):
        del physical_blocks


def _sequence(tokens):
    return SequenceMeta(
        token_ids=np.asarray(tokens, dtype=np.int64),
        tokens_per_block=1,
    )


def _manager(remote_hit: int = 0, enable_ssd: bool = False):
    manager = object.__new__(GlobalCacheEngine)
    manager._cache_tree_lock = threading.RLock()
    manager._metrics_collector = None
    manager.tokens_per_block = 1
    manager.index_accel = False
    manager.use_mooncake_store_backend = True
    manager.enable_kv_sharing = False
    manager.cache_config = SimpleNamespace(
        tokens_per_block=1,
        enable_cpu=True,
        enable_ssd=enable_ssd,
        enable_remote=True,
        enable_p2p_cpu=False,
        enable_p2p_ssd=False,
    )
    manager.cpu_cache_engine = CacheEngine(
        device_type=DeviceType.CPU,
        num_total_blocks=32,
        tokens_per_block=1,
        evict_ratio=0.1,
    )
    manager.ssd_cache_engine = (
        CacheEngine(
            device_type=DeviceType.SSD,
            num_total_blocks=32,
            tokens_per_block=1,
            evict_ratio=0.1,
        )
        if enable_ssd else None
    )
    manager.remote_cache_engine = _MooncakeRemoteStub(remote_hit)
    manager.cache_engines = {
        DeviceType.CPU: manager.cpu_cache_engine,
        DeviceType.REMOTE: manager.remote_cache_engine,
    }
    if manager.ssd_cache_engine is not None:
        manager.cache_engines[DeviceType.SSD] = manager.ssd_cache_engine
    manager.swa_op_constructor = SimpleNamespace(enabled=False)
    manager.match_all = lambda sequence_meta, temp_cache_strategy: (
        manager.cpu_cache_engine.match(sequence_meta),
        (manager.ssd_cache_engine.match(sequence_meta)
         if manager.ssd_cache_engine is not None else MatchResult()),
        manager.remote_cache_engine.match(sequence_meta),
    )
    manager._empty_put_return = lambda request_id: pytest.fail(
        f"unexpected empty PUT plan for request {request_id}")
    return manager


def _plan(manager, request_id, sequence, gpu_blocks=None):
    if gpu_blocks is None:
        gpu_blocks = np.arange(sequence.num_blocks, dtype=np.int64) + 100
    return manager._put_impl_global(
        request_id=request_id,
        sequence_meta=sequence,
        block_mask_start=0,
        block_mask_end=sequence.num_blocks,
        gpu_block_ids=np.asarray(gpu_blocks, dtype=np.int64),
        temp_cache_strategy=CacheStrategy(ignore_ssd=True),
        dp_client_id=0,
    )


def _find_op(plan_or_graph, transfer_type):
    graph = getattr(plan_or_graph, "transfer_graph", plan_or_graph)
    return next(
        op for op in graph._op_map.values()
        if op.transfer_type == transfer_type and not op.is_swa
    )


def _seed_ready(engine, sequence):
    blocks = engine.take(sequence.num_blocks)
    node = engine.insert(sequence, blocks, is_ready=True)
    assert node is not None
    return blocks, node


def _start_put(manager, request_id, sequence, gpu_block_start):
    token_mask = np.ones(sequence.num_blocks, dtype=np.int64)
    slot_mapping = np.arange(
        gpu_block_start,
        gpu_block_start + sequence.num_blocks,
        dtype=np.int64,
    )
    graph, _, callback, _, _ = manager.put(
        request_id=request_id,
        token_ids=sequence.token_ids.copy(),
        token_mask=token_mask,
        slot_mapping=slot_mapping,
        dp_client_id=0,
    )
    return graph, callback


def test_cpu_ssd_duplicate_put_fresh_rematch_recycles_loser_after_split():
    manager = _manager(enable_ssd=True)
    target = _sequence([1, 2, 3, 4])
    existing_branch = _sequence([1, 2, 9])
    cpu_seed, cpu_anchor = _seed_ready(
        manager.cpu_cache_engine, existing_branch)
    ssd_seed, ssd_anchor = _seed_ready(
        manager.ssd_cache_engine, existing_branch)
    initial_cpu_locks = cpu_anchor.lock_cnt
    initial_ssd_locks = ssd_anchor.lock_cnt

    first_graph, first_callback = _start_put(manager, 1, target, 100)
    first_cpu = _find_op(first_graph, TransferType.D2H).dst_block_ids.copy()
    first_ssd = _find_op(
        first_graph, TransferType.H2DISK).dst_block_ids.copy()
    second_graph, second_callback = _start_put(manager, 2, target, 200)
    second_cpu = _find_op(second_graph, TransferType.D2H).dst_block_ids.copy()
    second_ssd = _find_op(
        second_graph, TransferType.H2DISK).dst_block_ids.copy()

    for callback in (first_callback, second_callback):
        locked = callback.keywords["node_to_unlock"]
        assert locked[DeviceType.CPU][0] is cpu_anchor
        assert locked[DeviceType.CPU][1] == 0
        assert locked[DeviceType.SSD][0] is ssd_anchor
        assert locked[DeviceType.SSD][1] == 0
        pending_tiers = {
            pending.device_type
            for pending in callback.keywords["deferred_inserts"]
        }
        assert pending_tiers == {DeviceType.CPU, DeviceType.SSD}
    assert cpu_anchor.lock_cnt == initial_cpu_locks + 2
    assert ssd_anchor.lock_cnt == initial_ssd_locks + 2

    # The second completion owns both tails and splits the pre-existing branch.
    second_callback()
    assert cpu_anchor.size() == 1
    assert ssd_anchor.size() == 1
    assert cpu_anchor.lock_cnt == initial_cpu_locks + 1
    assert ssd_anchor.lock_cnt == initial_ssd_locks + 1

    # The first completion must rematch both trees and recycle both duplicate
    # staging allocations without disturbing the winning path.
    first_callback()
    assert cpu_anchor.lock_cnt == initial_cpu_locks
    assert ssd_anchor.lock_cnt == initial_ssd_locks

    expected_by_tier = (
        (manager.cpu_cache_engine, cpu_seed, first_cpu, second_cpu),
        (manager.ssd_cache_engine, ssd_seed, first_ssd, second_ssd),
    )
    for engine, seed, loser, winner in expected_by_tier:
        target_match = engine.match(target)
        branch_match = engine.match(existing_branch)
        np.testing.assert_array_equal(
            target_match.physical_blocks,
            np.concatenate((seed[:2], winner)),
        )
        np.testing.assert_array_equal(branch_match.physical_blocks, seed)
        assert target_match.num_ready_matched_blocks == target.num_blocks
        assert branch_match.num_ready_matched_blocks == existing_branch.num_blocks
        assert engine.mempool._free_mask[loser].all()
        assert not engine.mempool._free_mask[winner].any()
        assert engine.mempool.num_used_blocks == engine.index.total_cached_blocks()
        assert engine.index.total_unready_blocks() == 0


@pytest.mark.parametrize("finish_second_first", [False, True])
def test_duplicate_puts_publish_one_owner_and_recycle_the_loser(
    finish_second_first,
):
    manager = _manager()
    sequence = _sequence([1, 2, 3])
    first = _plan(manager, 1, sequence)
    second = _plan(manager, 2, sequence)
    first_blocks = first.deferred_inserts[0].physical_blocks.copy()
    second_blocks = second.deferred_inserts[0].physical_blocks.copy()

    assert manager.cpu_cache_engine.index.total_cached_blocks() == 0
    assert manager.cpu_cache_engine.mempool.num_used_blocks == 6

    winner, loser = ((second, first) if finish_second_first
                     else (first, second))
    manager._commit_deferred_insert(winner.deferred_inserts[0])
    manager._commit_deferred_insert(loser.deferred_inserts[0])

    expected = second_blocks if finish_second_first else first_blocks
    duplicate = first_blocks if finish_second_first else second_blocks
    match = manager.cpu_cache_engine.match(sequence)
    np.testing.assert_array_equal(match.physical_blocks, expected)
    assert match.num_ready_matched_blocks == 3
    assert manager.cpu_cache_engine.mempool.num_used_blocks == 3
    assert manager.cpu_cache_engine.mempool._free_mask[duplicate].all()
    assert not manager.cpu_cache_engine.mempool._free_mask[expected].any()


@pytest.mark.parametrize("finish_second_first", [False, True])
def test_overlapping_puts_fresh_rematch_and_split_without_leaking(
    finish_second_first,
):
    manager = _manager()
    first_sequence = _sequence([1, 2])
    second_sequence = _sequence([1, 3])
    first = _plan(manager, 1, first_sequence)
    second = _plan(manager, 2, second_sequence)
    first_blocks = first.deferred_inserts[0].physical_blocks.copy()
    second_blocks = second.deferred_inserts[0].physical_blocks.copy()

    ordered = (second, first) if finish_second_first else (first, second)
    for plan in ordered:
        manager._commit_deferred_insert(plan.deferred_inserts[0])

    first_match = manager.cpu_cache_engine.match(first_sequence)
    second_match = manager.cpu_cache_engine.match(second_sequence)
    expected_shared = second_blocks[0] if finish_second_first else first_blocks[0]
    assert first_match.physical_blocks.tolist() == [
        expected_shared, first_blocks[1]]
    assert second_match.physical_blocks.tolist() == [
        expected_shared, second_blocks[1]]
    assert manager.cpu_cache_engine.index.total_cached_blocks() == 3
    assert manager.cpu_cache_engine.index.total_unready_blocks() == 0
    assert manager.cpu_cache_engine.mempool.num_used_blocks == 3


def test_put_never_uses_an_unready_local_prefix_as_mooncake_source():
    manager = _manager()
    sequence = _sequence([1, 2, 3])
    unready = manager.cpu_cache_engine.take(1)
    manager.cpu_cache_engine.insert(
        sequence, unready, num_insert_blocks=1, is_ready=False)

    plan = _plan(manager, 1, sequence)
    d2h = _find_op(plan, TransferType.D2H)
    h2remote = _find_op(plan, TransferType.H2REMOTE)

    # All three blocks come from this request's GPU data. The pre-existing
    # unready block must not appear in the upload source.
    assert len(d2h.dst_block_ids) == 3
    np.testing.assert_array_equal(h2remote.src_block_ids, d2h.dst_block_ids)
    assert int(unready[0]) not in set(h2remote.src_block_ids.tolist())
    assert manager.cpu_cache_engine.index.total_cached_blocks() == 1


def test_ready_cpu_hit_retries_a_missing_mooncake_upload_without_d2h():
    manager = _manager(remote_hit=0)
    sequence = _sequence([1, 2, 3])
    initial = _plan(manager, 1, sequence)
    manager._commit_deferred_insert(initial.deferred_inserts[0])
    cached = manager.cpu_cache_engine.match(sequence).physical_blocks.copy()

    retry = _plan(manager, 2, sequence)
    op_types = {
        op.transfer_type for op in retry.transfer_graph._op_map.values()
    }
    assert TransferType.D2H not in op_types
    assert TransferType.H2REMOTE in op_types
    np.testing.assert_array_equal(
        _find_op(retry, TransferType.H2REMOTE).src_block_ids,
        cached,
    )
    assert retry.deferred_inserts == []
    assert retry.node_to_unlock[DeviceType.CPU][1] == 0


def test_duplicate_put_recycles_its_swa_slot_without_replacing_the_winner():
    manager = _manager()
    sequence = _sequence([1, 2])
    engine = manager.cpu_cache_engine
    freed_slots = []
    engine.swa_pool = SimpleNamespace(
        free=lambda slot: freed_slots.append(int(slot)))

    first = _plan(manager, 1, sequence).deferred_inserts[0]
    second = _plan(manager, 2, sequence).deferred_inserts[0]
    first = replace(first, swa_slot=7)
    second = replace(second, swa_slot=8)

    winner = manager._commit_deferred_insert(first)
    manager._commit_deferred_insert(second)

    assert winner.has_swa()
    assert winner.swa_host_slot == 7
    assert freed_slots == [8]


def test_p2p_publication_happens_once_for_the_inserted_ready_node():
    manager = _manager()
    sequence = _sequence([1, 2])
    engine = manager.cpu_cache_engine
    engine.local_index = SimpleNamespace(insert_and_publish=Mock())

    first = _plan(manager, 1, sequence).deferred_inserts[0]
    second = _plan(manager, 2, sequence).deferred_inserts[0]
    first = replace(first, publish_to_peer=True)
    second = replace(second, publish_to_peer=True)

    inserted = manager._commit_deferred_insert(first)
    manager._commit_deferred_insert(second)

    engine.local_index.insert_and_publish.assert_called_once_with(inserted)
    assert engine.match(sequence).num_ready_matched_blocks == 2
