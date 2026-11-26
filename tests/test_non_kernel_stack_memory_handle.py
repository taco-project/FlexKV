from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flexkv.common import memory_handle, platform


@pytest.fixture
def ipc(monkeypatch):
    monkeypatch.setattr(platform, "kernel_transfer_built", lambda: False)
    monkeypatch.setenv("FLEXKV_DISABLE_VMM_PROBE", "1")
    monkeypatch.setenv("FLEXKV_FORCE_DIRECT_IPC", "0")
    monkeypatch.setenv("FLEXKV_STRICT_IPC_MODE", "1")
    driver = SimpleNamespace(cuPointerGetAttribute=Mock(return_value=0))
    monkeypatch.setattr(memory_handle, "libcuda", driver)
    return driver


def cuda_tensor(device="cuda:0"):
    return SimpleNamespace(is_cuda=True, device=torch.device(device),
                           data_ptr=lambda: 0x10000, shape=(4,), dtype=torch.uint8,
                           numel=lambda: 4)


def test_recommended_presets_skip_vmm_probe(ipc):
    assert not memory_handle._is_vmm_pointer(0x10000)
    ipc.cuPointerGetAttribute.assert_not_called()


def test_non_kernel_stack_can_reenable_vmm_probe(ipc, monkeypatch):
    monkeypatch.setenv("FLEXKV_DISABLE_VMM_PROBE", "0")
    assert memory_handle._is_vmm_pointer(0x10000)
    ipc.cuPointerGetAttribute.assert_called_once()


def test_probe_escape_hatch_on_kernel_build(ipc, monkeypatch):
    monkeypatch.setattr(platform, "kernel_transfer_built", lambda: True)
    assert not memory_handle._is_vmm_pointer(0x10000)
    ipc.cuPointerGetAttribute.assert_not_called()


def test_nvidia_default_still_probes(ipc, monkeypatch):
    monkeypatch.setattr(platform, "kernel_transfer_built", lambda: True)
    monkeypatch.delenv("FLEXKV_DISABLE_VMM_PROBE", raising=False)
    assert memory_handle._is_vmm_pointer(0x10000)
    ipc.cuPointerGetAttribute.assert_called_once()


def test_recommended_presets_use_torch_export_for_zero_copy_request(ipc):
    tensor = cuda_tensor()
    rebuild = Mock()
    handle = SimpleNamespace(
        _export_tensor_handle=Mock(return_value=(rebuild, ("owner",), tensor.device)),
        _export_cuda_ipc_handle=Mock())
    memory_handle.TensorSharedHandle._init_from_tensor(handle, tensor, -1, True)
    assert handle.rebuild_func is rebuild
    assert handle.rebuild_args == ("owner",)
    assert handle.handle_type == "torch_reduce"
    handle._export_cuda_ipc_handle.assert_not_called()
    ipc.cuPointerGetAttribute.assert_not_called()


def test_direct_ipc_request_is_honored_when_opted_in(ipc, monkeypatch):
    monkeypatch.setenv("FLEXKV_FORCE_DIRECT_IPC", "1")
    tensor = cuda_tensor()
    handle = SimpleNamespace(_export_tensor_handle=Mock(),
                             _export_cuda_ipc_handle=Mock(return_value=(b"ipc", 0)))
    memory_handle.TensorSharedHandle._init_from_tensor(handle, tensor, -1, True)
    assert handle.handle_type == "cuda_ipc"
    assert handle.device == tensor.device
    handle._export_tensor_handle.assert_not_called()


def test_strict_export_failure_does_not_fall_back(ipc):
    handle = SimpleNamespace(
        _export_tensor_handle=Mock(side_effect=RuntimeError("export failed")),
        _export_cuda_ipc_handle=Mock(), _export_vmm=Mock())
    with pytest.raises(RuntimeError, match="forbids the fallback"):
        memory_handle.TensorSharedHandle._init_from_tensor(handle, cuda_tensor(), -1, False)
    handle._export_cuda_ipc_handle.assert_not_called()
    handle._export_vmm.assert_not_called()


def test_non_strict_export_failure_falls_back_to_direct_ipc(ipc, monkeypatch):
    monkeypatch.setenv("FLEXKV_STRICT_IPC_MODE", "0")
    handle = SimpleNamespace(
        _export_tensor_handle=Mock(side_effect=RuntimeError("export failed")),
        _export_cuda_ipc_handle=Mock(return_value=(b"ipc", 0)))
    memory_handle.TensorSharedHandle._init_from_tensor(handle, cuda_tensor(), -1, False)
    assert handle.handle_type == "cuda_ipc"


def test_owner_device_cannot_be_remapped(ipc):
    handle = SimpleNamespace(_export_tensor_handle=Mock())
    with pytest.raises(ValueError, match="owner's logical device"):
        memory_handle.TensorSharedHandle._init_from_tensor(handle, cuda_tensor(), 1, False)
    handle._export_tensor_handle.assert_not_called()


def test_owner_device_is_not_rewritten_into_rebuild_args(ipc):
    tensor = cuda_tensor()
    handle = SimpleNamespace(
        _export_tensor_handle=Mock(return_value=(Mock(), ("wrapped",), tensor.device)))
    memory_handle.TensorSharedHandle._init_from_tensor(handle, tensor, 0, False)
    assert handle.rebuild_args == ("wrapped",)
    assert handle.device == tensor.device


@pytest.mark.parametrize("kind", ["cuda_ipc", "vmm_fabric", "vmm_posix_fd"])
def test_non_torch_handles_are_imported(ipc, kind):
    imported = object()
    handle = SimpleNamespace(handle_type=kind, use_direct_ipc=kind == "cuda_ipc",
                             _get_tensor_impl=Mock(return_value=imported))
    assert memory_handle.TensorSharedHandle.get_tensor(handle) is imported


def test_rebuild_device_mismatch_does_not_copy(ipc):
    tensor = torch.empty(1, device="cpu")
    with pytest.raises(RuntimeError, match="shared allocation alias"):
        memory_handle.TensorSharedHandle._import_tensor_handle(
            lambda: tensor, (), torch.device("cuda:0"))
