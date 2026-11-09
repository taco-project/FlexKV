"""Chunked-prefetch metrics use the terminal, lease-protected prefix."""

from functools import partial
import threading
from types import SimpleNamespace as NS
from urllib.request import urlopen

import pytest

from flexkv.metrics import collector as metrics_module
from flexkv.metrics import server as metrics_server
from flexkv.prefetch.coordinator import PrefetchCoordinator
from flexkv.prefetch.runtime import TaskRuntime
from flexkv.prefetch.types import PrefetchOptions
from test_coordinator import Backend, Clock, finish


@pytest.fixture
def metrics(monkeypatch):
    prometheus = pytest.importorskip("prometheus_client")
    registry = prometheus.CollectorRegistry()
    monkeypatch.setattr(metrics_module, "Counter", partial(prometheus.Counter, registry=registry))
    monkeypatch.setattr(metrics_module, "Gauge", partial(prometheus.Gauge, registry=registry))
    monkeypatch.setattr(metrics_module, "Histogram", partial(prometheus.Histogram, registry=registry))
    monkeypatch.setattr(metrics_module, "_should_enable_metrics", lambda: True)
    monkeypatch.setattr(metrics_module, "_auto_start_metrics_server", lambda: None)
    monkeypatch.setattr(metrics_module, "_global_collector", None)
    collector = metrics_module.init_global_collector()
    assert metrics_module.init_global_collector() is collector
    return collector, registry


def sample(registry, name, **labels):
    return registry.get_sample_value(name, labels)


def test_terminal_metrics_wait_for_drain_and_count_only_reusable_prefix(metrics, monkeypatch):
    prometheus = pytest.importorskip("prometheus_client")
    collector, registry = metrics
    clock, backend = Clock(), Backend()
    # Model a Full+SWA checkpoint that protects only the first completed chunk.
    backend.result_end = lambda context, committed: min(4, committed)
    coordinator = PrefetchCoordinator(backend, clock=clock, metrics=collector)
    handle = coordinator.start(
        list(range(20)), PrefetchOptions(policy="best_effort", chunk_max_blocks=4)
    )
    coordinator.tick()
    coordinator.tick()
    clock.now = 1
    coordinator.demand([handle])
    finish(coordinator, backend.sent[1])
    clock.now = 2
    coordinator.tick()
    labels = dict(policy="best_effort", reason="demand", outcome="partial")
    assert sample(registry, "flexkv_prefetch_sessions_total", **labels) is None

    finish(coordinator, backend.sent[0])
    clock.now = 3
    coordinator.tick()
    assert coordinator.snapshot(handle).loaded_tokens == 4
    assert sample(registry, "flexkv_prefetch_sessions_total", **labels) == 1
    assert sample(registry, "flexkv_prefetch_loaded_tokens_total", policy="best_effort") == 4
    assert sample(registry, "flexkv_prefetch_elapsed_seconds_sum", policy="best_effort", reason="demand") == 3
    assert sample(registry, "flexkv_prefetch_drain_seconds_sum", policy="best_effort", reason="demand") == 2

    coordinator.release(handle)
    coordinator.release(handle)
    coordinator.tick()
    assert sample(registry, "flexkv_prefetch_sessions_total", **labels) == 1
    assert not backend.allocations

    # Exercise the existing localhost exporter with the collector's registry.
    started = []
    start_http_server = prometheus.start_http_server

    def start_and_capture(*args, **kwargs):
        result = start_http_server(*args, **kwargs)
        started.append(result)
        return result

    monkeypatch.setattr(prometheus, "REGISTRY", registry)
    monkeypatch.setattr(prometheus, "start_http_server", start_and_capture)
    monkeypatch.setattr(metrics_server, "_server_started", False)
    try:
        assert metrics_server.start_metrics_server(port=0)
        httpd, thread = started[0]
        assert isinstance(thread, threading.Thread)
        with urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/metrics", timeout=5) as response:
            body = response.read().decode()
        assert "flexkv_prefetch_sessions_total" in body
        assert "flexkv_prefetch_loaded_tokens_total" in body
        assert "flexkv_prefetch_elapsed_seconds_sum" in body
        assert "flexkv_prefetch_drain_seconds_sum" in body
    finally:
        if started:
            httpd, thread = started[0]
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)


def test_reset_does_not_count_unleased_tokens_or_finalize_twice(metrics):
    collector, registry = metrics
    clock, backend = Clock(), Backend()
    coordinator = PrefetchCoordinator(backend, clock=clock, metrics=collector)
    handle = coordinator.start(list(range(20)), PrefetchOptions(chunk_max_blocks=4))
    coordinator.tick()
    finish(coordinator, backend.sent[0])
    coordinator.tick()
    assert coordinator.snapshot(handle).loaded_tokens == 4
    coordinator.stop(handle, "reset")
    labels = dict(policy="wait_complete", reason="reset", outcome="aborted")
    assert sample(registry, "flexkv_prefetch_sessions_total", **labels) is None
    finish(coordinator, backend.sent[1])
    coordinator.tick()
    coordinator.release(handle)
    coordinator.tick()
    assert sample(registry, "flexkv_prefetch_sessions_total", **labels) == 1
    assert sample(registry, "flexkv_prefetch_loaded_tokens_total", policy="wait_complete") == 0


def test_control_thread_records_an_unpolled_terminal_session(metrics):
    collector, registry = metrics
    coordinator = PrefetchCoordinator(Backend(target=0), clock=Clock(), metrics=collector)

    class Engine:
        transfer_handles = []
        _prefetch = coordinator

        def _update_tasks(self, timeout):
            pass

        def _reap_completed_tasks(self):
            pass

        def _next_runtime_wakeup(self, poll_s):
            return self._prefetch.next_wakeup(poll_s)

    runtime = TaskRuntime(Engine())
    runtime.start()
    labels = dict(policy="wait_complete", reason="complete", outcome="empty")
    try:
        runtime.call(coordinator.start, [1])
        with runtime.changed:
            assert runtime.changed.wait_for(
                lambda: sample(registry, "flexkv_prefetch_sessions_total", **labels) == 1,
                timeout=2,
            )
    finally:
        runtime.stop()


def test_custom_labels_are_bounded(metrics):
    prometheus = pytest.importorskip("prometheus_client")
    collector, registry = metrics
    collector.record_prefetch_terminal("tenant-123", "request-456", "custom", 5, 1.0, 0.5)
    assert sample(registry, "flexkv_prefetch_sessions_total", policy="other", reason="other", outcome="other") == 1
    text = prometheus.generate_latest(registry).decode()
    assert "tenant-123" not in text and "request-456" not in text


def test_disabled_metrics_do_not_register_or_record(monkeypatch):
    prometheus = pytest.importorskip("prometheus_client")
    registry = prometheus.CollectorRegistry()
    monkeypatch.setattr(metrics_module, "Counter", partial(prometheus.Counter, registry=registry))
    monkeypatch.setattr(metrics_module, "Gauge", partial(prometheus.Gauge, registry=registry))
    monkeypatch.setattr(metrics_module, "Histogram", partial(prometheus.Histogram, registry=registry))
    monkeypatch.setattr(metrics_module, "_should_enable_metrics", lambda: False)
    monkeypatch.setattr(metrics_module, "_auto_start_metrics_server", lambda: None)
    collector = metrics_module.FlexKVMetricsCollector()
    collector.record_prefetch_terminal("timeout", "deadline", "empty", 0, 1.0, 0.0)
    assert not list(registry.collect())


def test_metrics_failure_does_not_prevent_retiring_a_session(caplog):
    def fail(*args):
        raise RuntimeError("export failed")

    backend = Backend()
    coordinator = PrefetchCoordinator(backend, clock=Clock(), metrics=NS(record_prefetch_terminal=fail))
    handle = coordinator.start([1])
    coordinator.tick()
    coordinator.release(handle)
    assert backend.allocations and not backend.releases
    finish(coordinator, backend.sent[0])
    coordinator.tick()
    assert coordinator.snapshot(handle).terminal
    assert len(backend.releases) == 1 and not backend.allocations
    assert "Failed to record chunked prefetch metrics" in caplog.text
