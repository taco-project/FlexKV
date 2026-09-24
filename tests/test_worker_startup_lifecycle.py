"""CPU startup and allocation-failure regressions; no GPU allocation is made."""

import multiprocessing as mp
import os
import signal
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

import flexkv.storage.allocator as allocator
import flexkv.transfer_manager as tm
from flexkv.integration.sglang.connector import FlexKVConnector


def _initialization_failure(handle):
    def fail(*args):
        assert signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
        raise RuntimeError("injected initialization failure")

    tm.TransferManager = fail
    handle._process_worker(
        handle.model_config,
        handle.cache_config,
        handle.command_child_conn,
        handle.result_child_conn,
        handle.gpu_register_port,
        handle.ready_event,
        handle.start_event,
        handle.error_child_conn,
    )


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires fork")
def test_child_initialization_error_reaches_parent_and_is_reaped():
    handle = tm.TransferManagerInterProcessHandle(NS(tp_size=1, dp_size=1), NS(), "/tmp/unused")
    handle.process = mp.get_context("fork").Process(target=_initialization_failure, args=(handle,))
    handle.process.start()
    handle.error_child_conn.close()
    try:
        deadline = time.monotonic() + 5
        error = None
        while time.monotonic() < deadline:
            try:
                handle.is_ready()
            except RuntimeError as exc:
                error = str(exc)
                break
            time.sleep(0.01)
        assert error and "injected initialization failure" in error
        handle.process.join(timeout=3)
        assert handle.process.exitcode not in (None, 0)
    finally:
        handle.shutdown()
    assert handle.process is None
    assert handle.error_parent_conn.closed and handle.command_child_conn.closed


@pytest.mark.parametrize("previous", [None, "true", "false"])
def test_spawn_wait_deadline_restores_mpi_environment(monkeypatch, previous):
    handle = tm.TransferManagerInterProcessHandle(NS(tp_size=1, dp_size=1), NS(), "/tmp/unused")
    handle._start_process = lambda: None
    monkeypatch.setenv("FLEXKV_WORKER_SPAWN_TIMEOUT_S", "0.01")
    if previous is None:
        monkeypatch.delenv("MPI4PY_RC_INITIALIZE", raising=False)
    else:
        monkeypatch.setenv("MPI4PY_RC_INITIALIZE", previous)
    with pytest.raises(TimeoutError, match="did not start"):
        handle.start()
    assert os.environ.get("MPI4PY_RC_INITIALIZE") == previous


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_spawn_timeout_does_not_spawn(monkeypatch, timeout):
    handle = tm.TransferManagerInterProcessHandle.__new__(tm.TransferManagerInterProcessHandle)
    handle._start_process = Mock()
    monkeypatch.setenv("FLEXKV_WORKER_SPAWN_TIMEOUT_S", timeout)
    with pytest.raises(ValueError, match="finite and positive"):
        handle.start()
    handle._start_process.assert_not_called()


def test_health_checked_after_ready_and_during_wait():
    handle = tm.TransferManagerInterProcessHandle(NS(tp_size=1, dp_size=1), NS(), "/tmp/unused")
    handle.ready_event.set()
    handle.error_child_conn.send("post-ready failure")
    with pytest.raises(RuntimeError, match="post-ready failure"):
        handle.is_ready()
    with pytest.raises(RuntimeError, match="post-ready failure"):
        handle.wait(timeout=0)


def test_connector_ready_wait_has_deadline(monkeypatch):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.kv_manager = Mock()
    connector.kv_manager.is_ready.return_value = False
    connector._label = "test"
    monkeypatch.setenv("FLEXKV_READY_TIMEOUT_S", "0.01")
    with pytest.raises(TimeoutError, match="did not become ready"):
        connector._wait_kv_manager_ready(poll_interval=0.001)


@pytest.mark.parametrize("fail_wrap", [False, True])
def test_hugepage_failure_removes_owned_file_and_closes_fd(tmp_path, monkeypatch, fail_wrap):
    path = tmp_path / "owned-hugepage"
    untouched = tmp_path / "unrelated"
    untouched.write_text("keep")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    mapping = Mock()
    monkeypatch.setattr(allocator, "_create_hugetlbfs_file", lambda _: (str(path), fd))
    with patch.object(allocator.mmap, "mmap", return_value=mapping) as mmap_call:
        if not fail_wrap:
            mmap_call.side_effect = RuntimeError("injected mmap failure")
        with (
            patch.object(allocator, "_wrap_mmap_tensor", side_effect=RuntimeError("injected wrap failure")),
            pytest.raises(RuntimeError, match="injected"),
        ):
            allocator.alloc_hugepage_tensor(16, torch.int64, shareable=True)
    assert not path.exists() and untouched.read_text() == "keep"
    with pytest.raises(OSError):
        os.fstat(fd)
    assert mapping.close.call_count == int(fail_wrap)


def _exit_soon():
    time.sleep(0.05)


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires fork")
def test_unbounded_wait_observes_child_exit():
    handle = tm.TransferManagerInterProcessHandle(NS(tp_size=1, dp_size=1), NS(), "/tmp/unused")
    handle.process = mp.get_context("fork").Process(target=_exit_soon)
    handle.process.start()
    start = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            handle.wait(timeout=None)
        assert time.monotonic() - start < 2
        assert handle.process.exitcode == 0
    finally:
        handle.shutdown()
