import pytest
import torch

from flexkv.cache.hie_cache_engine import HierarchyLRCacheEngine
from flexkv.cache.radix_remote import DistributedRadixTree, LocalRadixTree
from flexkv.cache.redis_meta import RedisMetaChannel, dist_available


class _FakeIndex:
    def __init__(self, events, name, reset_error=None, start_result=True):
        self.events = events
        self.name = name
        self.reset_error = reset_error
        self.start_result = start_result
        self.started = True

    def stop(self):
        self.events.append(f"{self.name}_stop")
        self.started = False

    def reset(self):
        self.events.append(f"{self.name}_reset")
        if self.reset_error is not None:
            raise self.reset_error

    def start(self, channel):
        self.events.append((f"{self.name}_start", channel))
        self.started = self.start_result
        return self.start_result


class _FakeMempool:
    def __init__(self, events):
        self.events = events

    def reset(self):
        self.events.append("mempool_reset")


class _FakeChannel:
    def __init__(self, events, barrier_ready=None, has_blocks=None, error=None):
        self.events = events
        self.barrier_ready = list(barrier_ready or [True])
        self.has_blocks = list(has_blocks or [False])
        self.error = error

    def begin_reset_barrier(self, ttl_ms):
        self.events.append(("begin_reset_barrier", ttl_ms))
        if self.error is not None:
            raise self.error
        return 7

    def mark_reset_barrier_arrival(self, epoch, ttl_ms):
        self.events.append(("mark_reset_barrier_arrival", epoch, ttl_ms))
        return True

    def is_reset_barrier_ready(self, epoch):
        self.events.append(("is_reset_barrier_ready", epoch))
        if len(self.barrier_ready) > 1:
            return self.barrier_ready.pop(0)
        return self.barrier_ready[0]

    def has_any_block_keys(self):
        self.events.append("scan_global_blocks")
        if self.error is not None:
            raise self.error
        if len(self.has_blocks) > 1:
            return self.has_blocks.pop(0)
        return self.has_blocks[0]

    def finish_reset_barrier(self, epoch):
        self.events.append(("finish_reset_barrier", epoch))
        return True


def _fake_hierarchy_engine(local_index, remote_index, mempool, local_channel, remote_channel):
    engine = object.__new__(HierarchyLRCacheEngine)
    engine.local_index = local_index
    engine.remote_index = remote_index
    engine.mempool = mempool
    engine.local_ch = local_channel
    engine.remote_ch = remote_channel
    engine.device_type = "test"
    engine.reset_barrier_timeout_ms = 100
    engine.reset_barrier_poll_ms = 1
    return engine


def test_hierarchy_reset_orders_local_cleanup_before_mempool_reuse():
    events = []
    local_channel = _FakeChannel(events)
    remote_channel = object()
    local_index = _FakeIndex(events, "local")
    remote_index = _FakeIndex(events, "remote")
    engine = _fake_hierarchy_engine(
        local_index,
        remote_index,
        _FakeMempool(events),
        local_channel,
        remote_channel,
    )

    engine.reset()

    assert events == [
        "local_stop",
        "remote_stop",
        ("begin_reset_barrier", 1000),
        "local_reset",
        "remote_reset",
        ("mark_reset_barrier_arrival", 7, 1000),
        ("is_reset_barrier_ready", 7),
        "scan_global_blocks",
        ("finish_reset_barrier", 7),
        "mempool_reset",
        ("remote_start", remote_channel),
        ("local_start", local_channel),
    ]
    assert local_index.started
    assert remote_index.started


def test_hierarchy_reset_does_not_reuse_mempool_after_redis_failure():
    events = []
    reset_error = RuntimeError("redis cleanup failed")
    local_index = _FakeIndex(events, "local", reset_error=reset_error)
    remote_index = _FakeIndex(events, "remote")
    engine = _fake_hierarchy_engine(
        local_index,
        remote_index,
        _FakeMempool(events),
        _FakeChannel(events),
        object(),
    )

    with pytest.raises(RuntimeError, match="redis cleanup failed"):
        engine.reset()

    assert events == [
        "local_stop",
        "remote_stop",
        ("begin_reset_barrier", 1000),
        "local_reset",
    ]
    assert not local_index.started
    assert not remote_index.started


def test_hierarchy_reset_waits_for_all_nodes_before_reusing_blocks():
    events = []
    local_channel = _FakeChannel(
        events,
        barrier_ready=[False, True, True],
        has_blocks=[True, True, False],
    )
    engine = _fake_hierarchy_engine(
        _FakeIndex(events, "local"),
        _FakeIndex(events, "remote"),
        _FakeMempool(events),
        local_channel,
        object(),
    )

    engine.reset()

    assert events.count("scan_global_blocks") == 3
    assert events.index("mempool_reset") > max(
        index for index, event in enumerate(events) if event == "scan_global_blocks"
    )


def test_hierarchy_reset_does_not_reuse_blocks_when_global_cleanup_times_out():
    events = []
    local_index = _FakeIndex(events, "local")
    remote_index = _FakeIndex(events, "remote")
    engine = _fake_hierarchy_engine(
        local_index,
        remote_index,
        _FakeMempool(events),
        _FakeChannel(events, barrier_ready=[True], has_blocks=[True]),
        object(),
    )
    engine.reset_barrier_timeout_ms = 1

    with pytest.raises(RuntimeError, match="physical blocks were not reset"):
        engine.reset()

    assert "mempool_reset" not in events
    assert not any(
        isinstance(event, tuple) and event[0].endswith("_start") for event in events
    )
    assert not local_index.started
    assert not remote_index.started


def test_hierarchy_reset_stops_remote_if_local_restart_fails():
    events = []
    local_index = _FakeIndex(events, "local", start_result=False)
    remote_index = _FakeIndex(events, "remote")
    engine = _fake_hierarchy_engine(
        local_index,
        remote_index,
        _FakeMempool(events),
        _FakeChannel(events),
        object(),
    )

    with pytest.raises(RuntimeError, match="Failed to restart local radix tree"):
        engine.reset()

    assert not local_index.started
    assert not remote_index.started


@pytest.mark.skipif(not dist_available(), reason="FlexKV was built without distributed cache support")
def test_local_radix_reset_releases_leases_and_discards_pending_publish():
    tree = LocalRadixTree(
        tokens_per_block=4,
        max_num_blocks=8,
        swap_block_threshold=0,
    )
    physical_blocks = torch.tensor([0, 1], dtype=torch.int64)
    block_hashes = torch.tensor([11, 22], dtype=torch.int64)

    node = tree.insert(physical_blocks, block_hashes, num_blocks=2, num_insert_blocks=2)
    assert node is not None
    assert tree.lease_pool_free_size() < tree.lease_pool_capacity()

    assert tree._c.insert_and_publish(node)
    assert tree.pending_queue_size() == 1

    tree.reset()

    assert tree.total_cached_blocks() == 0
    assert tree.pending_queue_size() == 0
    assert tree.lease_pool_free_size() == tree.lease_pool_capacity()

    node = tree.insert(physical_blocks, block_hashes, num_blocks=2, num_insert_blocks=2)
    assert node is not None
    assert tree.lease_pool_free_size() < tree.lease_pool_capacity()


@pytest.mark.skipif(not dist_available(), reason="FlexKV was built without distributed cache support")
def test_local_radix_reset_keeps_tree_and_leases_when_redis_delete_fails():
    tree = LocalRadixTree(
        tokens_per_block=4,
        max_num_blocks=8,
        swap_block_threshold=0,
    )
    physical_blocks = torch.tensor([0, 1], dtype=torch.int64)
    block_hashes = torch.tensor([33, 44], dtype=torch.int64)
    node = tree.insert(physical_blocks, block_hashes, num_blocks=2, num_insert_blocks=2)
    assert node is not None

    channel = RedisMetaChannel(
        host="127.0.0.1",
        port=1,
        node_id=7,
        local_ip="127.0.0.1",
        blocks_key="RESET_TEST",
    )
    tree.set_meta_channel(channel)
    assert tree._c.insert_and_publish(node)

    cached_blocks = tree.total_cached_blocks()
    free_leases = tree.lease_pool_free_size()

    with pytest.raises(RuntimeError, match="Redis block metadata"):
        tree.reset()

    assert tree.pending_queue_size() == 0
    assert tree.total_cached_blocks() == cached_blocks
    assert tree.lease_pool_free_size() == free_leases


@pytest.mark.skipif(not dist_available(), reason="FlexKV was built without distributed cache support")
def test_distributed_radix_reset_leaves_an_empty_index():
    tree = DistributedRadixTree(
        tokens_per_block=4,
        max_num_blocks=8,
        node_id=1,
    )

    tree.reset()

    assert tree.is_empty()
