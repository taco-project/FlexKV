"""Initial prefix admission distinguishes retained leases from inflight credit."""

from concurrent.futures import Future
from threading import RLock
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from flexkv.prefetch.mooncake_planner import MooncakeChunkPlanner, MooncakePrefetchContext
from flexkv.prefetch.types import PrefetchCapacityExhausted

pytestmark = pytest.mark.unit

MiB = 1024 * 1024


def planner_with_prefix(prefix_blocks=224):
    node = NS(parent=None, size=lambda: prefix_blocks)
    cpu = Mock()
    cpu.match.return_value = NS(num_matched_blocks=prefix_blocks, last_node=node)
    planner = MooncakeChunkPlanner.__new__(MooncakeChunkPlanner)
    planner.cache = NS(cpu_cache_engine=cpu, _cache_tree_lock=RLock())
    planner.engine = NS(_prefetch=NS(reserved_bytes=512 * MiB))
    planner.block_size = 1
    planner.block_bytes = 8 * MiB
    planner.swa_slot_bytes = 0
    planner.max_pinned_bytes = 2048 * MiB
    planner.pinned_bytes = 0
    tokens = list(range(prefix_blocks + 1))
    sequence = NS(token_ids=tokens, block_hashes=tokens)
    future = Future()
    future.set_result((sequence, len(tokens)))
    context = MooncakePrefetchContext(tokens=tokens, namespace=None, client_id=0, future=future)
    return planner, context, cpu


def test_initial_prefix_waits_for_peer_io_then_retries_same_context():
    planner, context, cpu = planner_with_prefix()
    # 1792 MiB resident prefix + 512 MiB peer reservation exceeds 2 GiB.
    assert planner.resolve(context) is None
    assert context.pinned_bytes == 0 and context.node is None
    assert not context.cancel_query.is_set()
    cpu.lock_node.assert_not_called()

    # A released peer drains and returns its reservation/result lease.
    planner.engine._prefetch.reserved_bytes = 0
    assert planner.resolve(context) == (224, 225)
    assert planner.pinned_bytes == context.pinned_bytes == 1792 * MiB
    cpu.lock_node.assert_called_once()
    planner.release(context)
    assert planner.pinned_bytes == 0


@pytest.mark.parametrize("reserved", [0, 512 * MiB])
def test_intrinsically_oversized_prefix_does_not_wait_for_peers(reserved):
    planner, context, cpu = planner_with_prefix(prefix_blocks=257)
    planner.engine._prefetch.reserved_bytes = reserved
    with pytest.raises(PrefetchCapacityExhausted):
        planner.resolve(context)
    assert context.pinned_bytes == 0
    cpu.lock_node.assert_not_called()


def test_retained_result_leases_are_not_treated_as_temporary_credit():
    planner, context, cpu = planner_with_prefix()
    planner.pinned_bytes = 512 * MiB
    with pytest.raises(PrefetchCapacityExhausted):
        planner.resolve(context)
    cpu.lock_node.assert_not_called()


def test_publication_keeps_boolean_capacity_fallback():
    planner, context, cpu = planner_with_prefix(prefix_blocks=257)
    context.sequence = context.future.result()[0]
    # Publication must keep its existing partial-prefix fallback, rather than
    # raising an admission exception after data has already entered the tree.
    assert planner._pin_prefix(context, 257, reserved_to_release=512 * MiB) is False
    cpu.lock_node.assert_not_called()
