from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flexkv.common import ipc_uuid, memory_handle


def rebuild_stub(tensor_type, shape, stride, offset, storage_type, dtype,
                 storage_device, handle):
    return storage_device, handle


@pytest.fixture
def devices(monkeypatch):
    uuids = ["device-a", "device-b"]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(uuids))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        uuid=uuids[index.index if isinstance(index, torch.device) else index]))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return uuids


def test_uuid_remaps_consumer_ordinal(devices):
    args = (None, (), (), 0, None, None, 0, b"shared-allocation")
    rebuild, packed = ipc_uuid.wrap_rebuild(rebuild_stub, args, 0)
    devices.reverse()
    assert rebuild(*packed) == (1, b"shared-allocation")
    assert args[6] == 0


def test_preserves_existing_uuid_wrapper(devices):
    def external_wrapper(*args):
        return ipc_uuid.device_from_uuid(args[6]).index

    args = (None,) * 6 + ("device-a",)
    rebuild, packed = ipc_uuid.wrap_rebuild(external_wrapper, args, 0)
    devices.reverse()
    assert rebuild(*packed) == 1


def test_missing_uuid_fails_before_rebuild(devices):
    args = (None,) * 6 + (0, b"shared-allocation")
    rebuild, packed = ipc_uuid.wrap_rebuild(rebuild_stub, args, 0)
    devices[0] = "different-device"
    with pytest.raises(RuntimeError, match="not visible"):
        rebuild(*packed)


def test_unknown_argument_protocol_fails_closed(devices):
    with pytest.raises(RuntimeError, match="signature"):
        ipc_uuid.wrap_rebuild(lambda *args: args, (0,), 0)


def test_ce_only_export_installs_wrapper(devices, monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    args = (None,) * 6 + (0, b"shared-allocation")
    monkeypatch.setattr(memory_handle.reductions, "reduce_tensor",
                        lambda tensor: (rebuild_stub, args))
    tensor = SimpleNamespace(device=torch.device("cuda:0"), is_cuda=True,
                             _typed_storage=lambda: SimpleNamespace(
                                 _share_cuda_=lambda: None))
    rebuild, packed, device = memory_handle.TensorSharedHandle._export_tensor_handle(tensor)
    assert rebuild is ipc_uuid.rebuild_by_uuid
    assert device == tensor.device
    devices.reverse()
    assert rebuild(*packed)[0] == 1


def test_nvidia_export_keeps_original_protocol(devices, monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "0")
    monkeypatch.setattr(memory_handle, "_IS_XMLIR", False)
    args = (None,) * 6 + (0, b"shared-allocation")
    monkeypatch.setattr(memory_handle.reductions, "reduce_tensor",
                        lambda tensor: (rebuild_stub, args))
    tensor = SimpleNamespace(device=torch.device("cuda:0"), is_cuda=True,
                             _typed_storage=lambda: SimpleNamespace(
                                 _share_cuda_=lambda: None))
    rebuild, packed, _ = memory_handle.TensorSharedHandle._export_tensor_handle(tensor)
    assert rebuild is rebuild_stub
    assert packed is args
