import numpy as np
import pytest

from flexkv.cache.radixtree import RadixNode, RadixTreeIndex


pytestmark = pytest.mark.unit


def _node(block_hash, physical_block, grace_time, parent):
    node = RadixNode(
        block_hashes=np.array([block_hash], dtype=np.int64),
        physical_blocks=np.array([physical_block], dtype=np.int64),
        is_ready=True,
        lock_cnt=0,
        grace_time=grace_time,
        parent=parent,
    )
    parent.children[node.head_hash()] = node
    # P2 caches the ancestor-block offset on each node; pre-P2 trees simply
    # ignore this dynamically-added attribute and continue using the walk.
    parent_offset = getattr(parent, "block_offset", 0)
    node.block_offset = (
        0 if parent.is_root() else parent_offset + parent.size())
    return node


def test_dynamic_fallback_prefers_all_non_sink_work_before_sink_parents():
    idx = RadixTreeIndex(tokens_per_block=1, sink_block_count=1)

    parent_a = _node(10, 100, 0.1, idx.root_node)
    child_a = _node(11, 101, 1.0, parent_a)
    parent_b = _node(20, 200, 0.2, idx.root_node)
    child_b = _node(21, 201, 2.0, parent_b)
    idx.leaf_nodes = {
        child_a.head_hash(): child_a,
        child_b.head_hash(): child_b,
    }

    evicted, _ = idx.evict(3)

    # Both offset=1 non-sink leaves must be consumed before either offset=0
    # sink parent. After they are exhausted, fallback may evict the oldest sink
    # parent to satisfy the remaining demand.
    assert evicted.tolist() == [101, 201, 100]
    assert parent_b.parent is idx.root_node


def test_all_sink_fallback_uses_raw_eviction_policy_priority():
    idx = RadixTreeIndex(tokens_per_block=1, sink_block_count=1)
    newer = _node(10, 100, 2.0, idx.root_node)
    older = _node(20, 200, 1.0, idx.root_node)
    idx.leaf_nodes = {
        newer.head_hash(): newer,
        older.head_hash(): older,
    }

    evicted, _ = idx.evict(1)

    assert evicted.tolist() == [200]
