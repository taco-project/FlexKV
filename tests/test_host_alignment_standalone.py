"""No-Torch registration geometry, rollback and metadata lifetime contracts."""

import ast
import ctypes
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_host():
    tree = ast.parse((ROOT / "flexkv/transfer/host_buffer.py").read_text())
    selected = {"host_register_align_bytes", "_align_cudahost_chunk_bytes", "_get_cudahost_chunk_bytes",
                "cudaHostRegister", "cudaHostUnregister", "_unregister_host_range"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    for node in nodes:
        node.returns = None
        for arg in ast.walk(node):
            if isinstance(arg, ast.arg):
                arg.annotation = None
    driver = SimpleNamespace(cudaHostRegister=Mock(return_value=0), cudaHostUnregister=Mock(return_value=0))
    metadata = (Mock(), Mock())
    ns = dict(ctypes=ctypes, os=os, _FLEXKV_CUDAHOST_CHUNK_SAFETY_MARGIN_BYTES=4096,
              _get_cudart=lambda: driver, _get_cudahost_register_flags=lambda: 0,
              _host_region_api=lambda: metadata, _registered_ranges={},
              _registered_ranges_lock=threading.Lock(), flexkv_logger=Mock())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "host_buffer.py", "exec"), ns)
    return ns, driver, metadata


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", str(64 / 1024**2))
    return load_host()


def buffer(size=128 * 1024):
    return SimpleNamespace(data_ptr=lambda: 0x100000, numel=lambda: size, element_size=lambda: 1)


@pytest.mark.parametrize("groups,itemsize,expected", [(None, 2, 12288), ([object()], 2, 6144),
                                                       (None, 1, 6144)])
def test_stride_unit_conversion(host, groups, itemsize, expected):
    ns, _, _ = host
    layout = SimpleNamespace(layer_groups=groups, get_block_stride=lambda: 6144)
    assert ns["host_register_align_bytes"](layout, SimpleNamespace(itemsize=itemsize)) == expected


@pytest.mark.parametrize("stride,expected", [(0, 60 * 1024), (4096, 60 * 1024),
                                              (24 * 1024, 48 * 1024), (6000, None)])
def test_lcm_chunk_geometry(host, stride, expected):
    ns, _, _ = host
    if expected is None:
        with pytest.raises(ValueError, match="register align quantum"):
            ns["_get_cudahost_chunk_bytes"](stride)
    else:
        assert ns["_get_cudahost_chunk_bytes"](stride) == expected


def test_aligned_metadata_is_reused_after_environment_change(host, monkeypatch):
    ns, driver, metadata = host
    tensor = buffer()
    ns["cudaHostRegister"](tensor, align_bytes=24 * 1024)
    metadata[0].assert_called_once_with(0x100000, 128 * 1024, 48 * 1024)
    assert [call.args[1].value for call in driver.cudaHostRegister.call_args_list] == [49152, 49152, 32768]
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", "0")
    ns["cudaHostUnregister"](tensor)
    assert [call.args[0].value for call in driver.cudaHostUnregister.call_args_list] == [
        0x100000, 0x100000 + 49152, 0x100000 + 98304]
    metadata[1].assert_called_once_with(0x100000)
    assert not ns["_registered_ranges"]


def test_cuda_failure_rolls_back_aligned_chunks_before_metadata(host):
    ns, driver, metadata = host
    driver.cudaHostRegister.side_effect = [0, 7]
    with pytest.raises(RuntimeError, match="offset=49152"):
        ns["cudaHostRegister"](buffer(), align_bytes=24 * 1024)
    assert driver.cudaHostUnregister.call_count == 1
    metadata[0].assert_not_called()
    assert not ns["_registered_ranges"]


def test_metadata_failure_rolls_back_all_aligned_chunks(host):
    ns, driver, metadata = host
    metadata[0].side_effect = ValueError("overlap")
    with pytest.raises(ValueError, match="overlap"):
        ns["cudaHostRegister"](buffer(), align_bytes=24 * 1024)
    assert driver.cudaHostUnregister.call_count == 3
    assert not ns["_registered_ranges"]


def test_unregister_failure_retains_effective_geometry(host):
    ns, driver, metadata = host
    tensor = buffer()
    ns["cudaHostRegister"](tensor, align_bytes=24 * 1024)
    driver.cudaHostUnregister.side_effect = [0, 1, 0]
    ns["cudaHostUnregister"](tensor)
    assert ns["_registered_ranges"][tensor.data_ptr()][1] == 48 * 1024
    metadata[1].assert_not_called()


@pytest.mark.parametrize("chunk_gb", [str(4096 / 1024**3), str(2048 / 1024**3)])
def test_tiny_positive_chunk_never_falls_back_to_whole_pool(host, monkeypatch, chunk_gb):
    ns, driver, _ = host
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", chunk_gb)
    with pytest.raises(ValueError, match="safety margin"):
        ns["cudaHostRegister"](buffer())
    driver.cudaHostRegister.assert_not_called()


def test_explicit_zero_preserves_whole_pool_registration(host, monkeypatch):
    ns, driver, metadata = host
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", "0")
    ns["cudaHostRegister"](buffer(), align_bytes=12345)
    assert driver.cudaHostRegister.call_count == 1
    metadata[0].assert_called_once_with(0x100000, 128 * 1024, 0)


def test_workers_pass_each_pool_geometry_and_remote_does_not_pin_pool():
    source = (ROOT / "flexkv/transfer/workers/gpu_cpu.py").read_text()
    assert "host_register_align_bytes(cpu_kv_layout, dtype)" in source
    assert "host_register_align_bytes(swa_cpu_kv_layout, swa_dt)" in source
    runtime = (ROOT / "flexkv/transfer/workers/runtime.py").read_text()
    assert "cudaHostRegister(tensor, align_bytes=align_bytes)" in runtime
    remote = (ROOT / "flexkv/transfer/workers/remote.py").read_text()
    assert "_register_host_tensor" not in remote
    assert "if backend.needs_pinned_block_ids:" in remote


def load_method(path, owner, name, **dependencies):
    import __future__
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(flexkv_logger=Mock(), **dependencies)
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def real_register_worker(host):
    from types import MethodType
    ns, _, _ = host
    method = load_method("flexkv/transfer/workers/runtime.py", "TransferWorkerBase", "_register_host_tensor",
                         cudaHostRegister=ns["cudaHostRegister"])
    worker = SimpleNamespace(worker_id=1, _host_registered=[], _op_buffer_pinned=False)
    worker._register_host_tensor = MethodType(method, worker)
    return worker


@pytest.mark.parametrize("ndim,stride,expected_chunk", [(1, 1, 60 * 1024),
                                                        (2, 4096, 32 * 1024),
                                                        (3, 3072, 48 * 1024)])
def test_op_buffer_real_registration_uses_row_pitch(host, ndim, stride, expected_chunk):
    ns, driver, metadata = host
    worker = real_register_worker(host)
    tensor = SimpleNamespace(data_ptr=lambda: 0x100000, numel=lambda: 16 * 1024,
                             element_size=lambda: 8, stride=lambda axis: stride, dim=lambda: ndim)
    worker.op_buffer_tensor = tensor
    pin = load_method("flexkv/transfer/workers/runtime.py", "TransferWorkerBase", "_pin_op_buffer")
    pin(worker)
    metadata[0].assert_called_once_with(0x100000, 128 * 1024, expected_chunk)
    assert worker._host_registered == [(tensor, "op_buffer")]
    assert worker._op_buffer_pinned
    count = driver.cudaHostRegister.call_count
    pin(worker)
    assert driver.cudaHostRegister.call_count == count
    ns["cudaHostUnregister"](tensor)
    assert driver.cudaHostUnregister.call_count == count


def test_op_buffer_pin_failure_keeps_retryable_state(host):
    ns, driver, metadata = host
    worker = real_register_worker(host)
    worker.op_buffer_tensor = SimpleNamespace(data_ptr=lambda: 0x100000, numel=lambda: 16 * 1024,
                                              element_size=lambda: 8, stride=lambda axis: 4096,
                                              dim=lambda: 2)
    pin = load_method("flexkv/transfer/workers/runtime.py", "TransferWorkerBase", "_pin_op_buffer")
    driver.cudaHostRegister.return_value = 7
    with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
        pin(worker)
    assert not worker._op_buffer_pinned
    assert not worker._host_registered
    metadata[0].assert_not_called()
    driver.cudaHostRegister.return_value = 0
    pin(worker)
    assert worker._op_buffer_pinned
    assert len(worker._host_registered) == 1
    ns["cudaHostUnregister"](worker.op_buffer_tensor)


@pytest.mark.parametrize("register_fails,prepare_ok", [(False, True), (False, False), (True, True)])
def test_nixl_actual_attach_passes_byte_stride_and_preserves_ownership(host, register_fails, prepare_ok):
    ns, driver, metadata = host
    worker = real_register_worker(host)
    tensor = SimpleNamespace(data_ptr=lambda: 0x100000, numel=lambda: 32 * 1024,
                             element_size=lambda: 4)
    strides = SimpleNamespace(layer_stride=65536, kv_stride=32768, chunk_bytes=4096)
    cpu = SimpleNamespace(blocks=tensor, block_stride=24 * 1024, require_strides=lambda _: strides)
    geometry = SimpleNamespace(require_cpu=lambda _: cpu)
    def prepare(blocks):
        assert blocks is tensor
        assert worker._host_registered == [(tensor, "nixl_cpu_pool")]
        return prepare_ok
    session = SimpleNamespace(prepare_dram_cpu=Mock(side_effect=prepare))
    backend = SimpleNamespace(nixl_backend="POSIX", name="nixl", _session=session)
    attach = load_method("flexkv/transfer/backends.py", "NixlFileBackend", "_attach_cpu")
    if register_fails:
        driver.cudaHostRegister.return_value = 7
        with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
            attach(backend, worker, geometry)
        session.prepare_dram_cpu.assert_not_called()
        assert not worker._host_registered
        metadata[0].assert_not_called()
    else:
        if prepare_ok:
            attach(backend, worker, geometry)
        else:
            with pytest.raises(RuntimeError, match="prepare_dram_cpu failed"):
                attach(backend, worker, geometry)
        metadata[0].assert_called_once_with(0x100000, 128 * 1024, 48 * 1024)
        assert worker._host_registered == [(tensor, "nixl_cpu_pool")]
        ns["cudaHostUnregister"](tensor)
        assert driver.cudaHostUnregister.call_count == 3
