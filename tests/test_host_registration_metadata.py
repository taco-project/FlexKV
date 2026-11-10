from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flexkv.transfer import host_buffer


@pytest.fixture
def registered(monkeypatch):
    driver = SimpleNamespace(cudaHostRegister=Mock(return_value=0),
                             cudaHostUnregister=Mock(return_value=0))
    api = (Mock(), Mock())
    monkeypatch.setattr(host_buffer, "_get_cudart", lambda: driver)
    monkeypatch.setattr(host_buffer, "_host_region_api", lambda: api)
    monkeypatch.setattr(host_buffer, "_registered_ranges", {})
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", str(64 / 1024**2))
    tensor = SimpleNamespace(data_ptr=lambda: 0x100000,
                             numel=lambda: 128 * 1024, element_size=lambda: 1)
    return driver, api, tensor


def test_recorded_chunk_survives_configuration_change(registered, monkeypatch):
    driver, api, tensor = registered
    host_buffer.cudaHostRegister(tensor)
    api[0].assert_called_once_with(0x100000, 128 * 1024, 60 * 1024)
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", "0")
    host_buffer.cudaHostUnregister(tensor)
    assert [call.args[0].value for call in driver.cudaHostUnregister.call_args_list] == [
        0x100000, 0x100000 + 60 * 1024, 0x100000 + 120 * 1024]
    api[1].assert_called_once_with(0x100000)
    assert not host_buffer._registered_ranges


def test_failed_cuda_registration_does_not_publish_metadata(registered):
    driver, api, tensor = registered
    driver.cudaHostRegister.side_effect = [0, 1]
    with pytest.raises(RuntimeError, match="offset=61440"):
        host_buffer.cudaHostRegister(tensor)
    api[0].assert_not_called()
    assert driver.cudaHostUnregister.call_count == 1
    assert not host_buffer._registered_ranges


def test_metadata_failure_rolls_back_all_cuda_chunks(registered):
    driver, api, tensor = registered
    api[0].side_effect = ValueError("overlap")
    with pytest.raises(ValueError, match="overlap"):
        host_buffer.cudaHostRegister(tensor)
    assert driver.cudaHostUnregister.call_count == 3
    assert not host_buffer._registered_ranges


def test_unregister_failure_retains_registered_range(registered):
    driver, api, tensor = registered
    host_buffer.cudaHostRegister(tensor)
    driver.cudaHostUnregister.side_effect = [0, 1, 0]
    host_buffer.cudaHostUnregister(tensor)
    api[1].assert_not_called()
    assert 0x100000 in host_buffer._registered_ranges


def test_old_extension_rejects_chunked_ce_before_registration(registered, monkeypatch):
    driver, _, tensor = registered
    monkeypatch.setattr(host_buffer, "_host_region_api", lambda: None)
    monkeypatch.setenv("FLEXKV_USE_CE_TRANSFER_H2D", "1")
    with pytest.raises(RuntimeError, match="Rebuild"):
        host_buffer.cudaHostRegister(tensor)
    driver.cudaHostRegister.assert_not_called()


def test_whole_registration_publishes_one_range(registered, monkeypatch):
    driver, api, tensor = registered
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", "0")
    host_buffer.cudaHostRegister(tensor)
    assert driver.cudaHostRegister.call_count == 1
    api[0].assert_called_once_with(0x100000, 128 * 1024, 0)
    host_buffer.cudaHostUnregister(tensor)
    api[1].assert_called_once_with(0x100000)


@pytest.mark.parametrize("ce_only,flags", [("0", 3), ("1", 0)])
def test_platform_register_flags(registered, monkeypatch, ce_only, flags):
    driver, _, tensor = registered
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", ce_only)
    monkeypatch.delenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", raising=False)
    host_buffer.cudaHostRegister(tensor)
    assert all(call.args[2].value == flags
               for call in driver.cudaHostRegister.call_args_list)
    host_buffer.cudaHostUnregister(tensor)


@pytest.mark.parametrize("flags", ["1", "2", "3", "4", "-1"])
def test_ce_only_rejects_nondefault_flags(registered, monkeypatch, flags):
    driver, _, tensor = registered
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", flags)
    with pytest.raises(ValueError, match="FLAGS=0"):
        host_buffer.cudaHostRegister(tensor)
    driver.cudaHostRegister.assert_not_called()


def test_ce_only_rejects_mapped_allocation(registered, monkeypatch):
    import torch

    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    with pytest.raises(RuntimeError, match="forbidden"):
        host_buffer.alloc_mapped_host_tensor(16, torch.uint8)


def test_registered_allocation_lives_until_last_storage_view(registered, monkeypatch):
    import gc
    import torch

    driver, api, _ = registered
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0")
    tensor = host_buffer.alloc_registered_host_tensor(128 * 1024, torch.uint8)
    ptr = tensor.data_ptr()
    view = tensor.detach()
    del tensor
    gc.collect()
    driver.cudaHostUnregister.assert_not_called()
    assert ptr in host_buffer._registered_ranges
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", "0")
    del view
    gc.collect()
    assert driver.cudaHostUnregister.call_count == 3
    api[1].assert_called_once_with(ptr)
    assert ptr not in host_buffer._registered_ranges


def test_registered_mapping_is_retained_if_unpin_fails(registered, monkeypatch):
    import gc
    import torch

    driver, api, _ = registered
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0")
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", "0")
    monkeypatch.setattr(host_buffer, "_failed_registered_mappings", {})
    tensor = host_buffer.alloc_registered_host_tensor(4096, torch.uint8)
    ptr = tensor.data_ptr()
    driver.cudaHostUnregister.return_value = 1
    del tensor
    gc.collect()
    assert ptr in host_buffer._failed_registered_mappings
    assert ptr in host_buffer._registered_ranges
    api[1].assert_not_called()
