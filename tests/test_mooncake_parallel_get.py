"""Bounded SDK concurrency and destination ownership; no live Store is needed."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from flexkv.external.mooncake_store_utils import MooncakeStoreClient, MooncakeStoreConfig
from flexkv.transfer.backends import MooncakeStoreBackend

pytestmark = pytest.mark.unit


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FLEXKV_MOONCAKE_GET_CONCURRENCY", "2")
    monkeypatch.setenv("FLEXKV_MOONCAKE_GET_MAX_INFLIGHT_BYTES", str(2 * 12288))
    monkeypatch.setattr(MooncakeStoreClient, "setup", lambda self: setattr(self, "_is_setup", True))
    monkeypatch.setattr(MooncakeStoreClient, "check_server", lambda self: None)
    monkeypatch.setattr(MooncakeStoreClient, "warm_up", lambda self: None)
    c = MooncakeStoreClient(MooncakeStoreConfig(max_get_batch_bytes=12288))
    c._is_setup = True
    yield c
    c.shutdown_get_workers()


def test_parallel_window_keeps_result_order_and_padded_byte_bound(client):
    # Each one-byte object consumes one 4 KiB page plus 8 KiB SDK overhead.
    # A barrier requires two simultaneous calls; a third cannot enter this window.
    barrier = Barrier(2)
    calls = []

    def get(keys, ptrs, sizes):
        assert len(keys) == 1
        calls.append(keys[0])
        barrier.wait(timeout=5)
        return [0 if keys[0] == "b" else sizes[0]]

    client.zero_copy_get_impl = get
    assert client.batch_get(list("abcd"), [1, 2, 3, 4], [1] * 4) == [True, False, True, True]
    assert set(calls[:2]) == {"a", "b"}
    assert set(calls[2:]) == {"c", "d"}


def test_parallel_padding_bound_does_not_reinterpret_raw_sdk_batch_limit(client):
    client._config.max_get_batch_bytes = 8
    client._get_max_inflight_bytes = 4 * 12288
    barrier = Barrier(2)
    calls = []

    def get(keys, pointers, sizes):
        assert sum(sizes) <= 8
        calls.append(list(keys))
        barrier.wait(timeout=5)
        return sizes

    client._store = SimpleNamespace(batch_get_into=get)
    assert client.batch_get(list("abcd"), [1, 2, 3, 4], [4] * 4) == [True] * 4
    assert sorted(calls) == [["a", "b"], ["c", "d"]]


@pytest.mark.parametrize("failure", ["sdk", "result_count", "submit"])
def test_every_accepted_call_drains_before_error_escapes(client, failure):
    entered, release, failed = Event(), Event(), Event()
    executor = client._get_executor

    def get(keys, ptrs, sizes):
        if keys == ["a"] and failure != "submit":
            assert entered.wait(5)
            failed.set()
            if failure == "sdk":
                raise RuntimeError("injected SDK failure")
            return []
        entered.set()
        assert release.wait(5)
        return sizes

    class RejectSecondSubmission:
        count = 0

        def submit(self, *args):
            self.count += 1
            if self.count == 2:
                assert entered.wait(5)
                failed.set()
                raise RuntimeError("injected submission failure")
            return executor.submit(*args)

        def shutdown(self, **kwargs):
            executor.shutdown(**kwargs)

    if failure == "submit":
        client._get_executor = RejectSecondSubmission()
    client.zero_copy_get_impl = get
    with ThreadPoolExecutor(max_workers=1) as caller:
        result = caller.submit(client.batch_get, list("ab"), [1, 2], [1, 1])
        try:
            assert failed.wait(5)
            assert not result.done(), "caller released buffers before the accepted SDK call drained"
        finally:
            release.set()
        with pytest.raises(RuntimeError):
            result.result(timeout=5)


def test_closed_client_never_performs_lazy_setup(client, monkeypatch):
    client.shutdown_get_workers()
    client._is_setup = False
    monkeypatch.setattr(client, "_ensure_setup", lambda: pytest.fail("setup after shutdown"))
    with pytest.raises(RuntimeError, match="closed"):
        client.batch_get(["a"], [1], [1])


def test_query_client_stays_serial(monkeypatch):
    monkeypatch.setenv("FLEXKV_MOONCAKE_GET_CONCURRENCY", "4")
    monkeypatch.setattr(MooncakeStoreClient, "setup", lambda self: setattr(self, "_is_setup", True))
    client = MooncakeStoreClient(MooncakeStoreConfig(), query_only=True)
    try:
        assert client._get_concurrency == 1
        assert client._get_executor is None
    finally:
        client.shutdown_get_workers()


def test_backend_drains_gets_before_unregistering_destination_regions():
    events = []

    def unregister(pointer):
        assert events and events[0] == "drained"
        events.append(pointer)

    backend = MooncakeStoreBackend.__new__(MooncakeStoreBackend)
    backend.mooncake_client = SimpleNamespace(
        shutdown_get_workers=lambda: events.append("drained"), unregister_buffer=unregister
    )
    backend._registered_regions = [(1, 4096), (2, 4096)]
    backend.shutdown()
    assert events == ["drained", 2, 1]
    assert backend._registered_regions == []
