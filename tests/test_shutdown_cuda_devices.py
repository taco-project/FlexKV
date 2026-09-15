"""CPU regressions for shutdown device ownership and real daemon threads.

Load worker/engine methods from their source AST to avoid importing the native
transfer extension. The extracted production bodies are executed unchanged;
CUDA calls are mocked, so these tests do not claim GPU runtime validation.
"""

import ast
import errno
import gc
import queue
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest
import torch

from flexkv.transfer import cuda_sync


pytestmark = pytest.mark.unit
TRANSFER_DIR = Path(__file__).resolve().parents[1] / "flexkv" / "transfer"
RUNTIME_PY = TRANSFER_DIR / "workers" / "runtime.py"
ENGINE_PY = TRANSFER_DIR / "transfer_engine.py"


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


def _extract(path, namespace, class_name, method_names, function_names, extra_nodes=()):
    """Exec just the named class methods / module functions from one source file."""
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in function_names]
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.bases = []
    cls.decorator_list = []
    cls.body = [n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name in method_names]
    module = ast.Module(body=list(extra_nodes) + functions + [cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def source_api():
    namespace = {
        "torch": torch, "threading": threading, "gc": gc, "Any": Any, "Dict": Dict,
        "List": List, "Optional": Optional, "Set": Set, "Tuple": Tuple, "Union": Union,
        "TensorSharedHandle": object,
        "ABC": object,
        "flexkv_logger": cuda_sync.flexkv_logger,
        "synchronize_cuda_devices": cuda_sync.synchronize_cuda_devices,
        "_undrained_host_regions": [],
    }
    _extract(
        RUNTIME_PY, namespace,
        class_name="TransferWorkerBase",
        method_names={
            "__new__", "_ensure_cuda_device", "_import_tensor_handles",
            "_register_host_tensor", "_pin_op_buffer", "_release_imported_gpu_tensors",
            "_drain_cuda_bounded", "shutdown",
        },
        function_names={"ensure_cuda_device", "import_tensor_handles"},
    )
    import contextlib
    import os

    namespace.update(contextlib=contextlib, os=os)
    _extract(
        ENGINE_PY, namespace,
        class_name="TransferEngine",
        method_names={"shutdown"},
        function_names={"_te_bounded_cuda_sync"},
    )
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


def test_every_handle_device_is_recorded_not_only_the_first(cuda, source_api):
    """``get_tensor`` binds each handle's own device, so recording only the
    first handle would leave the rest of a mixed-device list undrained."""
    worker = source_api["TransferWorkerBase"]()
    handles = [
        types.SimpleNamespace(device=torch.device("cuda", d), get_tensor=lambda: object())
        for d in (3, 5)
    ]
    worker._import_tensor_handles(handles)
    assert worker._cuda_device_ids == {3, 5}


def test_cpu_only_worker_does_not_claim_the_default_device(cuda, source_api):
    """A CPU/SSD/remote worker pins host memory but owns no GPU.

    ``_register_host_tensor`` must not record ``current_device()``: that would
    make every CPU-side worker wait out the full 30s drain budget on GPU 0 --
    and then permanently retain its pinned host memory -- whenever GPU 0 is
    wedged by some unrelated process.
    """
    worker = source_api["TransferWorkerBase"]()
    worker.worker_id = 17
    worker.op_buffer_tensor = torch.empty(1)
    worker._op_buffer_pinned = False
    source_api["cudaHostRegister"] = lambda tensor, **kwargs: None
    worker._pin_op_buffer()
    assert worker._host_registered == [(worker.op_buffer_tensor, "op_buffer")]
    assert worker._cuda_device_ids == set()
    assert worker._drain_cuda_bounded(17, 2.0)
    assert cuda.calls == []


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
    source_api["close_all_cuda_ipc_handles"] = lambda: events.append("close_ipc")

    worker.shutdown()
    worker.shutdown()

    if drained:
        assert events == ["drain", "worker=17 second", "worker=17 first", "close_ipc"]
        assert worker._host_registered == []
        assert not worker._op_buffer_pinned
        assert source_api["_undrained_host_regions"] == []
    else:
        # No unregister, and no IPC mapping teardown either: a timeout does not
        # prove the DMA stopped, and closing a mapping under a live DMA is the
        # same fault as unpinning under one.
        assert events == ["drain"]
        assert worker._host_registered == original_regions
        assert worker._op_buffer_pinned
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


def test_engine_shutdown_survives_a_partially_failed_init(cuda, source_api, monkeypatch):
    """``shutdown`` runs on the error path of ``__init__`` too, before the
    handle dicts exist -- deriving device ids must not raise there."""
    engine = source_api["TransferEngine"]()
    engine._running = True
    engine.shutdown_write_fd = 123
    engine.shutdown_read_fd = 124
    engine._scheduler_thread = types.SimpleNamespace(join=lambda **kwargs: None)
    engine._collect_worker_handles = lambda: []
    engine._shutdown_worker_handles = lambda handles: None
    engine.finished_ops_queue = queue.Queue()

    monkeypatch.setattr(source_api["os"], "write", lambda *a: None)
    monkeypatch.setattr(source_api["os"], "close", lambda *a: None)
    engine.shutdown()
    assert cuda.calls == []
