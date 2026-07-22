# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for CPU allocation in the local GDS GET planner.

These tests exercise graph construction only. They do not initialize a GDS
worker, require cuFile, or perform a real SSD-to-GPU transfer.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from flexkv.cache.cache_engine import CacheStrategy, GlobalCacheEngine
from flexkv.common.transfer import TransferType
from flexkv.common.type import MatchResultAccel


pytestmark = pytest.mark.unit


class _FakeNode:
    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks

    def size(self) -> int:
        return self._num_blocks


class _FakeCPUCacheEngine:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.take_requests = []
        self.recycled = []
        self.inserted = []

    def take(self, num_required_blocks, protected_node=None, strict=False):
        del protected_node, strict
        self.take_requests.append(num_required_blocks)
        num_allocated = min(num_required_blocks, self.capacity)
        return np.arange(1000, 1000 + num_allocated, dtype=np.int64)

    def recycle(self, physical_blocks):
        self.recycled.append(physical_blocks.copy())

    def insert(self, sequence_meta, physical_blocks, num_insert_blocks=None,
               is_ready=False, match_result=None):
        del sequence_meta, is_ready, match_result
        self.inserted.append(physical_blocks.copy())
        size = len(physical_blocks) if num_insert_blocks is None else num_insert_blocks
        return _FakeNode(size)


def _match(block_ids):
    blocks = np.asarray(block_ids, dtype=np.int64)
    node = _FakeNode(len(blocks)) if len(blocks) > 0 else None
    return MatchResultAccel(
        num_ready_matched_blocks=len(blocks),
        num_matched_blocks=len(blocks),
        last_ready_node=node,
        last_node=node,
        physical_blocks=blocks,
        matched_pos="local",
    )


def _build_local_get(*, enable_gds, cpu_blocks, ssd_blocks, gpu_blocks,
                     cpu_capacity):
    cpu_cache = _FakeCPUCacheEngine(capacity=cpu_capacity)
    engine = GlobalCacheEngine.__new__(GlobalCacheEngine)
    engine.cache_config = SimpleNamespace(
        enable_cpu=True,
        enable_ssd=True,
        enable_gds=enable_gds,
        enable_p2p_cpu=False,
    )
    engine.cpu_cache_engine = cpu_cache
    engine.index_accel = False
    engine._metrics_collector = None
    engine.match_local = lambda sequence_meta, strategy: (
        _match(cpu_blocks),
        _match(ssd_blocks),
    )

    plan = engine._get_impl_local(
        request_id=1,
        sequence_meta=object(),
        block_mask_start=0,
        block_mask_end=len(gpu_blocks),
        gpu_block_ids=np.asarray(gpu_blocks, dtype=np.int64),
        temp_cache_strategy=CacheStrategy(),
        dp_client_id=0,
    )
    return plan, cpu_cache


def _only_op(plan, transfer_type):
    matching = [
        op for op in plan.transfer_graph._op_map.values()
        if op.transfer_type == transfer_type
    ]
    assert len(matching) == 1
    return matching[0]


def test_ssd_only_gds_get_does_not_require_cpu_capacity():
    plan, cpu_cache = _build_local_get(
        enable_gds=True,
        cpu_blocks=[],
        ssd_blocks=[40, 41],
        gpu_blocks=[100, 101],
        cpu_capacity=0,
    )

    assert cpu_cache.take_requests == []
    assert cpu_cache.inserted == []
    assert cpu_cache.recycled == []
    assert plan.num_gpu_blocks_to_transfer == 2

    gds_op = _only_op(plan, TransferType.DISK2D)
    np.testing.assert_array_equal(gds_op.src_block_ids, [40, 41])
    np.testing.assert_array_equal(gds_op.dst_block_ids, [100, 101])
    assert not any(
        op.transfer_type == TransferType.DISK2H
        for op in plan.transfer_graph._op_map.values()
    )


def test_mixed_gds_get_allocates_no_cpu_blocks_for_ssd_fragment():
    plan, cpu_cache = _build_local_get(
        enable_gds=True,
        cpu_blocks=[10],
        ssd_blocks=[30, 40, 41],
        gpu_blocks=[100, 101, 102],
        cpu_capacity=0,
    )

    assert cpu_cache.take_requests == []

    h2d_op = _only_op(plan, TransferType.H2D)
    np.testing.assert_array_equal(h2d_op.src_block_ids, [10])
    np.testing.assert_array_equal(h2d_op.dst_block_ids, [100])

    gds_op = _only_op(plan, TransferType.DISK2D)
    np.testing.assert_array_equal(gds_op.src_block_ids, [40, 41])
    np.testing.assert_array_equal(gds_op.dst_block_ids, [101, 102])


def test_non_gds_get_still_allocates_cpu_staging_blocks():
    plan, cpu_cache = _build_local_get(
        enable_gds=False,
        cpu_blocks=[],
        ssd_blocks=[40, 41],
        gpu_blocks=[100, 101],
        cpu_capacity=2,
    )

    assert cpu_cache.take_requests == [2]

    disk2h_op = _only_op(plan, TransferType.DISK2H)
    h2d_op = _only_op(plan, TransferType.H2D)
    np.testing.assert_array_equal(disk2h_op.dst_block_ids, [1000, 1001])
    np.testing.assert_array_equal(h2d_op.src_block_ids, [1000, 1001])
    assert disk2h_op.op_id in h2d_op.predecessors
