"""CPU regressions for shutdown device ownership and real daemon threads.

Load worker/engine methods from their source AST to avoid importing the native
transfer extension. The extracted production bodies are executed unchanged;
CUDA calls are mocked, so these tests do not claim GPU runtime validation.
"""

import ast
import errno
import queue
import threading
import time
import types
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple, Union

import pytest
import torch

from flexkv.transfer import cuda_sync


pytestmark = pytest.mark.unit
TRANSFER_DIR = Path(__file__).resolve().parents[1] / "flexkv" / "transfer"


@pytest.fixture
def cuda(monkeypatch):
    tls = threading.local()
    tls.device = 6
    calls = []

    def select(device):
        tls.device = device
        calls.append(("select", device, threading.current_thread()))

    def sync(device=None):
        actual = getattr(tls, "device", 0)
        assert device == actual
        calls.append(("sync", device, threading.current_thread()))

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", select)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: getattr(tls, "device", 0))
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return types.SimpleNamespace(tls=tls, calls=calls, sync=sync)


@pytest.mark.parametrize("devices", [[3], [0], [3, 5, 3]])
def test_drain_selects_each_device_in_its_own_thread(cuda, devices):
    assert cuda_sync.synchronize_cuda_devices(devices, 2.0, name="test")
    assert sorted(d for action, d, _ in cuda.calls if action == "sync") == sorted(set(devices))
    for device in set(devices):
        calls = [(action, thread) for action, d, thread in cuda.calls if d == device]
        assert [action for action, _ in calls] == ["select", "sync"]
        assert calls[0][1] is calls[1][1]
        assert calls[0][1] is not threading.current_thread()
        assert calls[0][1].daemon
    assert torch.cuda.current_device() == 6


@pytest.mark.parametrize("initialized, devices", [(False, [3]), (True, [])])
def test_drain_does_not_initialize_cuda_for_no_work(cuda, monkeypatch, initialized, devices):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    assert cuda_sync.synchronize_cuda_devices(devices, 2.0, name="test")
    assert cuda.calls == []


@pytest.mark.parametrize("operation", ["set_device", "synchronize"])
def test_device_failure_is_reported_and_other_device_still_drains(cuda, monkeypatch, operation):
    original = getattr(torch.cuda, operation)

    def fail(device):
        if device == 3:
            raise RuntimeError("injected device failure")
        return original(device)

    monkeypatch.setattr(torch.cuda, operation, fail)
    assert not cuda_sync.synchronize_cuda_devices([3, 5], 2.0, name="test")
    assert [d for action, d, _ in cuda.calls if action == "sync"] == [5]


def test_hung_device_does_not_block_other_device_or_extend_deadline(cuda, monkeypatch):
    release = threading.Event()
    healthy_done = threading.Event()
    hung_entered = threading.Event()
    hung_exited = threading.Event()

    def sync(device=None):
        if device == 3:
            hung_entered.set()
            release.wait()
            hung_exited.set()
        else:
            cuda.sync(device)
            healthy_done.set()

    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    started = time.monotonic()
    try:
        assert not cuda_sync.synchronize_cuda_devices([3, 5], 0.1, name="test")
        assert time.monotonic() - started < 2.0
        assert hung_entered.is_set()
        assert healthy_done.wait(1.0)
    finally:
        release.set()
        assert hung_exited.wait(2.0)


def test_thread_start_failure_is_an_incomplete_drain(cuda, monkeypatch):
    def fail_start(thread):
        raise RuntimeError("cannot start drain thread")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    assert not cuda_sync.synchronize_cuda_devices([3], 2.0, name="test")
    assert cuda.calls == []


def test_negative_device_cannot_silently_sync_default_device(cuda):
    assert not cuda_sync.synchronize_cuda_devices([-1, 3], 2.0, name="test")
    assert [d for action, d, _ in cuda.calls if action == "sync"] == [3]


@pytest.fixture
def source_api():
    namespace = {
        "torch": torch, "threading": threading, "Any": Any, "List": List,
        "Optional": Optional, "Set": Set, "Tuple": Tuple, "Union": Union,
        "TensorSharedHandle": object,
        "flexkv_logger": cuda_sync.flexkv_logger,
        "synchronize_cuda_devices": cuda_sync.synchronize_cuda_devices,
        "_undrained_host_regions": [],
    }
    worker_tree = ast.parse((TRANSFER_DIR / "worker.py").read_text())
    base = next(n for n in worker_tree.body if isinstance(n, ast.ClassDef) and n.name == "TransferWorkerBase")
    method_names = {
        "__new__", "_ensure_cuda_device", "_import_tensor_handles",
        "_register_host_tensor", "_drain_cuda_bounded", "shutdown",
    }
    functions = [n for n in worker_tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"ensure_cuda_device", "import_tensor_handles"}]
    base.bases = []
    base.body = [n for n in base.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
    module = ast.fix_missing_locations(ast.Module(body=functions + [base], type_ignores=[]))
    exec(compile(module, str(TRANSFER_DIR / "worker.py"), "exec"), namespace)
    engine_tree = ast.parse((TRANSFER_DIR / "transfer_engine.py").read_text())
    engine = next(n for n in engine_tree.body if isinstance(n, ast.ClassDef) and n.name == "TransferEngine")
    engine.body = [n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown"]
    helpers = [n for n in engine_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_te_bounded_cuda_sync"]
    module = ast.fix_missing_locations(ast.Module(body=helpers + [engine], type_ignores=[]))
    import contextlib
    import os

    namespace.update(contextlib=contextlib, os=os)
    exec(compile(module, str(TRANSFER_DIR / "transfer_engine.py"), "exec"), namespace)
    return namespace


def test_worker_tracks_tp_devices_even_when_ipc_import_fails(cuda, source_api):
    worker = source_api["TransferWorkerBase"]()

    def import_handle(device, fails=False):
        def get_tensor():
            assert device in worker._cuda_device_ids
            if fails:
                raise RuntimeError("IPC import failed")
            return object()

        return types.SimpleNamespace(device=torch.device("cuda", device), get_tensor=get_tensor)

    worker._import_tensor_handles([import_handle(3)])
    with pytest.raises(RuntimeError, match="IPC import failed"):
        worker._import_tensor_handles([import_handle(5, fails=True)])
    assert worker._cuda_device_ids == {3, 5}
    cuda.calls.clear()
    assert worker._drain_cuda_bounded(worker_id=17, timeout_s=2.0)
    assert sorted(d for action, d, _ in cuda.calls if action == "sync") == [3, 5]


def test_cpu_host_registration_tracks_registration_device(cuda, source_api):
    worker = source_api["TransferWorkerBase"]()
    worker.worker_id = 17
    source_api["cudaHostRegister"] = lambda tensor, **kwargs: None
    worker._register_host_tensor(torch.empty(1), "cpu")
    assert worker._cuda_device_ids == {6}
    assert worker._drain_cuda_bounded(17, 2.0)
    assert [d for action, d, _ in cuda.calls if action == "sync"] == [6]


@pytest.mark.parametrize("device", [None, -1, torch.device("cpu")])
def test_non_cuda_binding_does_not_add_device(cuda, source_api, device):
    worker = source_api["TransferWorkerBase"]()
    worker._ensure_cuda_device(device)
    assert worker._cuda_device_ids == set()
    assert cuda.calls == []


@pytest.mark.parametrize("drained", [True, False])
def test_worker_unpins_only_after_successful_drain_and_is_idempotent(source_api, drained):
    worker = source_api["TransferWorkerBase"]()
    worker.worker_id = 17
    worker._host_registered.extend([(object(), "first"), (object(), "second")])
    worker._op_buffer_pinned = True
    original_regions = list(worker._host_registered)
    events = []
    worker._drain_cuda_bounded = lambda *args, **kwargs: events.append("drain") or drained
    source_api["safe_cuda_host_unregister"] = lambda tensor, label: events.append(label)

    worker.shutdown()
    worker.shutdown()

    if drained:
        assert events == ["drain", "worker=17 second", "worker=17 first"]
        assert worker._host_registered == []
        assert not worker._op_buffer_pinned
        assert source_api["_undrained_host_regions"] == []
    else:
        assert events == ["drain"]
        assert worker._host_registered == original_regions
        assert worker._op_buffer_pinned
        del worker
        assert source_api["_undrained_host_regions"] == [original_regions]


def test_engine_shutdown_drains_registered_main_and_swa_devices(cuda, source_api, monkeypatch):
    engine = source_api["TransferEngine"]()
    engine._running = True
    engine.shutdown_write_fd = 123
    engine.shutdown_read_fd = 124
    engine._scheduler_thread = types.SimpleNamespace(join=lambda **kwargs: None)
    engine._collect_worker_handles = lambda: []
    engine._shutdown_worker_handles = lambda handles: None
    engine.finished_ops_queue = queue.Queue()
    engine.gpu_handle_groups = {"tp": [types.SimpleNamespace(gpu_device_id=d) for d in [3, 5]]}
    engine._swa_gpu_handles = {"swa": [types.SimpleNamespace(gpu_device_id=d) for d in [5, 7, None]]}

    def closed(*args):
        raise OSError(errno.EBADF, "closed test pipe")

    monkeypatch.setattr(source_api["os"], "write", closed)
    monkeypatch.setattr(source_api["os"], "close", closed)
    engine.shutdown()
    assert sorted(d for action, d, _ in cuda.calls if action == "sync") == [3, 5, 7]
    assert torch.cuda.current_device() == 6
