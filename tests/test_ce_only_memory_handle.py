from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flexkv.common import memory_handle


@pytest.fixture
def ipc(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.setattr(memory_handle, "_IS_XMLIR", False)
    driver = SimpleNamespace(cuPointerGetAttribute=Mock(return_value=0))
    monkeypatch.setattr(memory_handle, "libcuda", driver)
    return driver


def test_ce_only_skips_vmm_probe(ipc):
    assert not memory_handle._is_vmm_pointer(0x10000)
    ipc.cuPointerGetAttribute.assert_not_called()


def test_probe_escape_hatch_without_master_switch(ipc, monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "0")
    monkeypatch.setenv("FLEXKV_DISABLE_VMM_PROBE", "1")
    assert not memory_handle._is_vmm_pointer(0x10000)
    ipc.cuPointerGetAttribute.assert_not_called()


def test_nvidia_default_still_probes(ipc, monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "0")
    monkeypatch.delenv("FLEXKV_DISABLE_VMM_PROBE", raising=False)
    assert memory_handle._is_vmm_pointer(0x10000)
    ipc.cuPointerGetAttribute.assert_called_once()


def test_torch_export_is_used_even_for_zero_copy_request(ipc):
    device = torch.device("cuda:0")
    tensor = SimpleNamespace(is_cuda=True, device=device)
    rebuild = Mock()
    handle = SimpleNamespace(
        _export_tensor_handle=Mock(return_value=(rebuild, ("owner",), device)))
    memory_handle.TensorSharedHandle._init_from_tensor(handle, tensor, -1, True)
    assert handle.rebuild_func is rebuild
    assert handle.rebuild_args == ("owner",)
    assert handle.handle_type == "torch_reduce"
    assert not handle.use_direct_ipc
    ipc.cuPointerGetAttribute.assert_not_called()


def test_export_failure_does_not_fall_back(ipc):
    tensor = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    handle = SimpleNamespace(
        _export_tensor_handle=Mock(side_effect=RuntimeError("export failed")),
        _export_cuda_ipc_handle=Mock(), _export_vmm=Mock())
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        memory_handle.TensorSharedHandle._init_from_tensor(handle, tensor, -1, False)
    handle._export_cuda_ipc_handle.assert_not_called()
    handle._export_vmm.assert_not_called()


def test_owner_device_cannot_be_remapped(ipc):
    tensor = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    handle = SimpleNamespace(_export_tensor_handle=Mock())
    with pytest.raises(ValueError, match="owner's logical device"):
        memory_handle.TensorSharedHandle._init_from_tensor(handle, tensor, 1, False)
    handle._export_tensor_handle.assert_not_called()


@pytest.mark.parametrize("kind", ["cuda_ipc", "vmm_fabric", "vmm_posix_fd"])
def test_incompatible_imported_handles_are_rejected(ipc, kind):
    handle = SimpleNamespace(handle_type=kind, use_direct_ipc=True)
    with pytest.raises(RuntimeError, match="requires a torch_reduce handle"):
        memory_handle.TensorSharedHandle.get_tensor(handle)


def test_rebuild_device_mismatch_does_not_copy(ipc):
    tensor = torch.empty(1, device="cpu")
    with pytest.raises(RuntimeError, match="shared allocation alias"):
        memory_handle.TensorSharedHandle._import_tensor_handle(
            lambda: tensor, (), torch.device("cuda:0"))
