from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flexkv.external.mooncake_parallel_io import ParallelStoreChannels


class Engine:
    instances = []

    def __init__(self):
        self.register_memory = Mock(return_value=0)
        self.unregister_memory = Mock(return_value=0)
        self.instances.append(self)

    def initialize(self, host, metadata, protocol, device):
        self.device = device
        return 0

    def get_rpc_port(self):
        return 20000 + len(self.instances)

    def get_engine(self):
        return self


class Store:
    instances = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def setup(self, *args):
        self.engine = args[-1]
        self.segment_size = args[2]
        return 0

    def batch_get_into(self, keys, pointers, sizes):
        self.calls.append((keys, pointers, sizes))
        return [size if key != "short" else size - 1 for key, size in zip(keys, sizes)]

    def batch_put_from(self, keys, pointers, sizes):
        self.calls.append((keys, pointers, sizes))
        return [0] * len(keys)


@pytest.fixture
def channels(monkeypatch):
    Engine.instances = []
    Store.instances = []
    monkeypatch.setenv("FLEXKV_MOONCAKE_IO_WORKERS", "2")
    monkeypatch.setenv("FLEXKV_MOONCAKE_IO_MIN_PAGES", "1")
    monkeypatch.setenv("FLEXKV_MOONCAKE_IO_MAX_INFLIGHT_BYTES", "16")
    monkeypatch.delenv("FLEXKV_MOONCAKE_IO_LANE_DEVICES", raising=False)
    config = SimpleNamespace(local_hostname="127.0.0.1:5000", metadata_server="P2PHANDSHAKE",
                             protocol="rdma", device_name="nic0", master_addr="127.0.0.1:5001",
                             global_segment_size=33)
    opened = []

    def create():
        instance = ParallelStoreChannels(config, store_cls=Store, engine_cls=Engine)
        opened.append(instance)
        return instance

    yield create
    for instance in opened:
        instance.close()


def test_shared_engine_and_exact_results(channels):
    channels = channels()
    assert len(Engine.instances) == 1
    assert Store.instances[0].engine is Store.instances[1].engine
    assert sum(store.segment_size for store in Store.instances) == 33
    assert channels.batch_get_into(["one", "short", "three"], [100, 108, 116], [8, 8, 8]) == [8, 7, 8]
    assert all(sum(sizes) <= 16 for store in Store.instances for _, _, sizes in store.calls)


def test_mr_registered_once_per_engine(channels):
    channels = channels()
    assert channels.register_buffer(4096, 8192) == 0
    assert channels.register_buffer(4096, 8192) == 0
    Engine.instances[0].register_memory.assert_called_once_with(4096, 8192)
    channels.close()
    assert channels.unregister_buffer(4096) == 0
    Engine.instances[0].unregister_memory.assert_called_once_with(4096)


def test_lane_binding_uses_each_named_device(channels, monkeypatch):
    monkeypatch.setenv("FLEXKV_MOONCAKE_IO_LANE_DEVICES", "nic0,nic1")
    channels = channels()
    assert [engine.device for engine in Engine.instances] == ["nic0", "nic1"]
    channels.register_buffer(4096, 8192)
    channels.unregister_buffer(4096)
    for engine in Engine.instances:
        engine.register_memory.assert_called_once_with(4096, 8192)
        engine.unregister_memory.assert_called_once_with(4096)


def test_oversized_object_rejected_before_any_call(channels):
    channels = channels()
    with pytest.raises(ValueError, match="budget"):
        channels.batch_get_into(["small", "large"], [100, 108], [8, 17])
    assert not any(store.calls for store in Store.instances)


def test_close_prevents_new_io(channels):
    channels = channels()
    channels.close()
    with pytest.raises(RuntimeError, match="closed"):
        channels.batch_put_from(["key"], [4096], [8])


def test_sdk_failure_drains_other_lanes(channels):
    channels = channels()
    completed = []

    def fail(*args):
        raise RuntimeError("SDK failure")

    def finish(*args):
        completed.append(True)
        return [8]

    Store.instances[0].batch_get_into = fail
    Store.instances[1].batch_get_into = finish
    with pytest.raises(RuntimeError, match="SDK failure"):
        channels.batch_get_into(["a", "b"], [100, 108], [8, 8])
    assert completed == [True]


def test_independent_requests_complete_out_of_order(channels):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    channel = channels()
    entered, release = Event(), Event()

    def slow(keys, pointers, sizes):
        entered.set()
        assert release.wait(3)
        return sizes

    Store.instances[0].batch_get_into = slow
    with ThreadPoolExecutor(max_workers=2) as callers:
        first = callers.submit(channel.batch_get_into, ["slow"], [100], [8])
        try:
            assert entered.wait(2)
            second = callers.submit(channel.batch_get_into, ["fast"], [108], [8])
            assert second.result(timeout=2) == [8]
            assert not first.done()
        finally:
            release.set()
        assert first.result(timeout=2) == [8]
    assert channel._inflight_bytes == 0


def test_completed_lane_refills_without_wave_barrier(channels):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    channel = channels()
    slow_started, third_started, release = Event(), Event(), Event()

    def dispatch(keys, pointers, sizes):
        if keys == ["slow"]:
            slow_started.set()
            assert release.wait(3)
        if keys == ["third"]:
            third_started.set()
        return sizes

    for store in Store.instances:
        store.batch_get_into = dispatch
    with ThreadPoolExecutor(max_workers=1) as callers:
        result = callers.submit(channel.batch_get_into,
                                ["slow", "fast", "third"], [100, 108, 116], [8] * 3)
        try:
            assert slow_started.wait(2)
            assert third_started.wait(2)
            assert not result.done()
        finally:
            release.set()
        assert result.result(timeout=2) == [8, 8, 8]


def test_byte_budget_shared_across_get_and_put(channels):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    channel = channels()
    active, peak = 0, 0
    lock, full, release = Lock(), Event(), Event()

    def dispatch(keys, pointers, sizes, put=False):
        nonlocal active, peak
        with lock:
            active += sum(sizes)
            peak = max(peak, active)
            if active == 16:
                full.set()
        assert release.wait(3)
        with lock:
            active -= sum(sizes)
        return [0] * len(sizes) if put else sizes

    for store in Store.instances:
        store.batch_get_into = dispatch
        store.batch_put_from = lambda k, p, s: dispatch(k, p, s, put=True)
    with ThreadPoolExecutor(max_workers=3) as callers:
        futures = [callers.submit(channel.batch_get_into, ["a"], [100], [8]),
                   callers.submit(channel.batch_put_from, ["b"], [108], [8]),
                   callers.submit(channel.batch_get_into, ["c"], [116], [8])]
        try:
            assert full.wait(2)
            assert peak == 16
            assert channel._inflight_bytes <= 16
        finally:
            release.set()
        for future in futures:
            future.result(timeout=2)
    assert peak <= 16
    assert channel._inflight_bytes == 0


def test_partial_executor_submission_failure_drains(channels, monkeypatch):
    from threading import Event

    channel = channels()
    completed = Event()
    submit = channel._executor.submit
    calls = 0

    def submit_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("submit rejected")
        return submit(*args, **kwargs)

    def finish(keys, pointers, sizes):
        completed.set()
        return sizes

    Store.instances[0].batch_get_into = finish
    monkeypatch.setattr(channel._executor, "submit", submit_once)
    with pytest.raises(RuntimeError, match="submit rejected"):
        channel.batch_get_into(["a", "b"], [100, 108], [8, 8])
    assert completed.is_set()
    assert channel._inflight_bytes == 0
    assert channel._active_calls == 0


def test_unregister_refuses_live_destination(channels):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    channel = channels()
    entered, release = Event(), Event()
    channel.register_buffer(4096, 8192)

    def slow(keys, pointers, sizes):
        entered.set()
        assert release.wait(3)
        return sizes

    Store.instances[0].batch_get_into = slow
    with ThreadPoolExecutor(max_workers=1) as callers:
        future = callers.submit(channel.batch_get_into, ["a"], [4096], [8])
        try:
            assert entered.wait(2)
            with pytest.raises(RuntimeError, match="Drain"):
                channel.unregister_buffer(4096)
            Engine.instances[0].unregister_memory.assert_not_called()
        finally:
            release.set()
        future.result(timeout=2)
    assert channel.unregister_buffer(4096) == 0


def test_remote_pipeline_uses_main_loop_when_disabled(monkeypatch):
    from flexkv.transfer.workers.remote import CPURemoteTransferWorker
    from flexkv.transfer.workers.runtime import TransferWorkerBase

    worker = CPURemoteTransferWorker.__new__(CPURemoteTransferWorker)
    main_run = Mock(return_value="main")
    monkeypatch.setattr(TransferWorkerBase, "run", main_run)
    assert worker.run() == "main"
    main_run.assert_called_once()


def test_remote_pipeline_reports_fast_op_before_slow():
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    from threading import Event
    import time

    from flexkv.transfer.worker_op import WorkerTransferResult
    from flexkv.transfer.workers.remote import CPURemoteTransferWorker

    worker = CPURemoteTransferWorker.__new__(CPURemoteTransferWorker)
    worker._remote_pending = {}
    worker._remote_timeout = 3
    worker.finished_ops_queue = Queue()
    release = Event()

    def slow():
        assert release.wait(3)
        return WorkerTransferResult(1, (True,)), None

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(slow)
        second = executor.submit(lambda: (WorkerTransferResult(2, (True, False)), None))
        worker._remote_pending[first] = (SimpleNamespace(transfer_op_id=1, valid_block_num=1), time.monotonic())
        worker._remote_pending[second] = (SimpleNamespace(transfer_op_id=2, valid_block_num=2), time.monotonic())
        try:
            second.result(timeout=2)
            worker._poll_remote()
            result, success, _ = worker.finished_ops_queue.get_nowait()
            assert success and result.transfer_op_id == 2
            assert result.block_results == (True, False)
            assert first in worker._remote_pending
        finally:
            release.set()
        worker._drain_remote()
        assert worker.finished_ops_queue.get_nowait()[0].transfer_op_id == 1


def test_sdk_base_exception_retains_dma_credit(channels, monkeypatch):
    from flexkv.external.mooncake_parallel_io import UnknownMooncakeDMA

    channel = channels()
    channel.register_buffer(4096, 8192)

    def interrupted(*args):
        raise SystemExit("SDK interruption")

    def fatal(reason):
        raise UnknownMooncakeDMA(reason)

    Store.instances[0].batch_get_into = interrupted
    monkeypatch.setattr(channel, "_fatal", fatal)
    try:
        with pytest.raises(UnknownMooncakeDMA):
            channel.batch_get_into(["a"], [4096], [8])
        assert channel._active_calls == 1
        assert channel._inflight_bytes == 8
        assert len(channel._free_lanes) == 1
        with pytest.raises(RuntimeError, match="Drain"):
            channel.unregister_buffer(4096)
        Engine.instances[0].unregister_memory.assert_not_called()
        with pytest.raises(UnknownMooncakeDMA):
            channel.batch_get_into(["b"], [4104], [8])
    finally:
        channel._executor.shutdown(wait=True)
        monkeypatch.setattr(channel, "close", Mock())


def test_timeout_does_not_unregister_unknown_destination(channels, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from flexkv.external.mooncake_parallel_io import UnknownMooncakeDMA

    channel = channels()
    channel.timeout = 0.02
    channel.register_buffer(4096, 8192)
    entered, release = Event(), Event()

    def slow(keys, pointers, sizes):
        entered.set()
        assert release.wait(3)
        return sizes

    def fatal(reason):
        raise UnknownMooncakeDMA(reason)

    Store.instances[0].batch_get_into = slow
    monkeypatch.setattr(channel, "_fatal", fatal)
    try:
        with ThreadPoolExecutor(max_workers=1) as callers:
            future = callers.submit(channel.batch_get_into, ["a"], [4096], [8])
            try:
                assert entered.wait(2)
                with pytest.raises(UnknownMooncakeDMA):
                    future.result(timeout=2)
                assert channel._active_calls == 1
                with pytest.raises(RuntimeError, match="Drain"):
                    channel.unregister_buffer(4096)
                Engine.instances[0].unregister_memory.assert_not_called()
            finally:
                release.set()
    finally:
        channel._executor.shutdown(wait=True)
        monkeypatch.setattr(channel, "close", Mock())


def test_remote_unknown_result_never_publishes_completion(monkeypatch):
    from concurrent.futures import Future
    from queue import Queue
    from flexkv.external.mooncake_parallel_io import UnknownMooncakeDMA
    from flexkv.transfer.workers.remote import CPURemoteTransferWorker

    worker = CPURemoteTransferWorker.__new__(CPURemoteTransferWorker)
    future = Future()
    future.set_exception(UnknownMooncakeDMA("DMA still active"))
    worker._remote_pending = {future: (SimpleNamespace(transfer_op_id=7), 0)}
    worker.finished_ops_queue = Queue()

    def fatal(reason):
        raise UnknownMooncakeDMA(reason)

    monkeypatch.setattr(worker, "_remote_fatal", fatal)
    with pytest.raises(UnknownMooncakeDMA):
        worker._publish_remote(future)
    assert future in worker._remote_pending
    assert worker.finished_ops_queue.empty()


def test_executor_queued_then_raised_never_launches_unowned_dma(channels, monkeypatch):
    channel = channels()
    submit = channel._executor.submit
    orphaned = []

    def accept_then_fail(*args, **kwargs):
        orphaned.append(submit(*args, **kwargs))
        raise RuntimeError("thread startup failed after queue admission")

    monkeypatch.setattr(channel._executor, "submit", accept_then_fail)
    with pytest.raises(RuntimeError, match="queue admission"):
        channel.batch_get_into(["a"], [4096], [8])
    for future in orphaned:
        assert future.result(timeout=2) is None
    assert not any(store.calls for store in Store.instances)
    assert channel._active_calls == 0
    assert channel._inflight_bytes == 0
    assert channel._credits == {}


def test_client_channels_do_not_take_legacy_whole_get_lock(channels):
    from threading import local
    from flexkv.external.mooncake_store_utils import MooncakeStoreClient

    channel = channels()
    client = MooncakeStoreClient.__new__(MooncakeStoreClient)
    client._use_parallel_channels = True
    client._get_closed = False
    client._get_lock = object()
    client._config = SimpleNamespace(max_get_batch_bytes=16)
    client._channel_stats = local()
    client.zero_copy_get_impl = channel.batch_get_into
    assert client.batch_get(["one", "short"], [4096, 4104], [8, 8]) == [True, False]
    assert client._channel_stats.transferred_bytes == 8


def test_client_put_bandwidth_excludes_existing_and_failed_blocks():
    from threading import local
    from flexkv.external.mooncake_store_utils import MooncakeStoreClient

    client = MooncakeStoreClient.__new__(MooncakeStoreClient)
    client._use_parallel_channels = True
    client._is_setup = True
    client._channel_stats = local()
    client.batch_exists_impl = Mock(return_value=[1, 0, 0])
    client.zero_copy_put_impl = Mock(return_value=[0, -1])
    assert client.batch_put(["old", "new", "failed"], [4096, 4104, 4112], [8] * 3) == [True, True, False]
    assert client._channel_stats.transferred_bytes == 8
    client.zero_copy_put_impl.assert_called_once_with(["new", "failed"], [4104, 4112], [8, 8])


def test_remote_queued_then_raised_never_starts_backend(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    from flexkv.transfer.workers.remote import CPURemoteTransferWorker

    worker = CPURemoteTransferWorker.__new__(CPURemoteTransferWorker)
    op = SimpleNamespace(transfer_op_id=1, valid_block_num=1)
    worker._remote_pending = {}
    worker._remote_capacity = 2
    worker._remote_timeout = 3
    worker.finished_ops_queue = Queue()
    worker.transfer_conn = SimpleNamespace(poll=lambda **kwargs: True, recv=lambda: op)
    worker._execute_remote = Mock()
    orphaned = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        submit = executor.submit

        def accept_then_fail(*args):
            orphaned.append(submit(*args))
            raise RuntimeError("queued then rejected")

        worker._remote_executor = executor
        monkeypatch.setattr(executor, "submit", accept_then_fail)
        with pytest.raises(RuntimeError, match="queued then rejected"):
            worker.run()
        for future in orphaned:
            assert future.result(timeout=2) is None
    worker._execute_remote.assert_not_called()
    assert worker._remote_pending == {}
    assert worker.finished_ops_queue.get_nowait() == (1, False, None)


def test_cp_node_keys_preserve_default_and_full_layer_isolation():
    from flexkv.external.mooncake_store_keys import PoolKind, build_key

    assert build_key("h", PoolKind.KV) == "h_FlexKV"
    assert build_key("h", PoolKind.SWA, pp_rank=1, pp_size=2) == "h_FlexKV_swa_pp_rank_1_of_2"
    keys = [build_key("h", PoolKind.KV, cp_nodes=2, cp_node_rank=node,
                      total_layers=8, node_layer_end=8) for node in range(2)]
    assert keys == ["h_FlexKV_cpnode_0_of_2", "h_FlexKV_cpnode_1_of_2"]
    assert build_key("h", PoolKind.KV, cp_nodes=2, cp_node_rank=1,
                     pp_rank=1, pp_size=2) == "h_FlexKV_cpnode_1_of_2_pp_rank_1_of_2"
    with pytest.raises(ValueError, match="shard set"):
        build_key("h", PoolKind.KV, cp_nodes=2, cp_node_rank=2)


def _cp_query_engine(cp_nodes=2, node=0):
    from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine

    engine = MooncakeStoreCacheEngine.__new__(MooncakeStoreCacheEngine)
    engine.cp_nodes = cp_nodes
    engine.cp_node_rank = node
    engine.pp_rank = 0
    engine.pp_size = 1
    engine.node_layer_start = 0
    engine.node_layer_end = 8
    engine.total_layers = 8
    engine._swa_enabled = False
    return engine


def test_cp_query_and_data_keys_are_identical_for_each_physical_node():
    from flexkv.common.transfer import TransferType
    from flexkv.external.mooncake_store_keys import PoolKind
    from flexkv.transfer.backends import MooncakeStoreBackend

    for node in range(2):
        config = SimpleNamespace(mooncake_store_cp_nodes=2, mooncake_store_cp_node_rank=node,
                                 mooncake_store_node_layer_end=8, mooncake_store_total_layers=8)
        backend = MooncakeStoreBackend(config, PoolKind.KV)
        backend._cpu_buffer = SimpleNamespace(data_ptr=lambda: 4096)
        backend.block_size_bytes = 8
        op = SimpleNamespace(transfer_type=TransferType.REMOTE2H,
                             mooncake_store_block_hashes=["a", "b"])
        pointers, sizes, data_keys = backend._keys_and_ptrs(op, [0, 1], [2, 3])
        engine = _cp_query_engine(node=node)
        query_keys = [engine._build_pool_key(h, PoolKind.KV) for h in ("a", "b")]
        assert data_keys == query_keys
        assert pointers == [4112, 4120] and sizes == [8, 8]
        assert all(f"cpnode_{node}_of_2" in key for key in data_keys)


def test_cp_match_requires_all_fragments_in_one_query():
    engine = _cp_query_engine()
    query = Mock(return_value=[1, 1, 1, 0, 1, 1])
    engine.mooncake_store_client = SimpleNamespace(batch_exists_impl=query)
    assert engine._joint_match_length(["a", "b", "c"]) == (1, 1, 0)
    query.assert_called_once_with([
        "a_FlexKV_cpnode_0_of_2", "a_FlexKV_cpnode_1_of_2",
        "b_FlexKV_cpnode_0_of_2", "b_FlexKV_cpnode_1_of_2",
        "c_FlexKV_cpnode_0_of_2", "c_FlexKV_cpnode_1_of_2"])


def test_cp_short_metadata_reply_cannot_become_false_hit():
    engine = _cp_query_engine()
    engine.mooncake_store_client = SimpleNamespace(batch_exists_impl=Mock(return_value=[1]))
    with pytest.raises(RuntimeError, match="result count"):
        engine._joint_match_length(["a"])


def test_cp_swa_snapshot_layout_is_not_silently_enabled():
    engine = _cp_query_engine()
    engine._swa_enabled = True
    with pytest.raises(NotImplementedError, match="full-KV"):
        engine._joint_match_length(["a"])
