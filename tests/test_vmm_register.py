"""End-to-end tests for VMM-backed TensorSharedHandle sharing.

These exercise the path FlexKV needs when vLLM's ``enable_sleep_mode`` is on:
KV cache tensors are allocated via ``cuMemCreate`` + ``cuMemMap`` (CUDA VMM),
which the legacy ``cudaIpcGetMemHandle`` API cannot export. FlexKV instead
uses ``cuMemExportToShareableHandle`` with a fabric handle so the resulting
bytes still fit through the existing ZMQ control plane.

Requires:
  * CUDA-capable GPU
  * libcuda + libcudart present at load time
  * Driver + hardware that expose ``CU_MEM_HANDLE_TYPE_FABRIC`` (H100+
    with nvidia-imex running); FD-only setups skip cleanly.

Tests that need a second process spawn one with ``multiprocessing.spawn``
so the importer gets a distinct CUDA context, matching FlexKV's real
KVManager topology.
"""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
from typing import Optional

import pytest
import torch

from flexkv.common import memory_handle
from flexkv.common.memory_handle import (
    CUDA_SUCCESS,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    CU_MEM_ALLOCATION_TYPE_PINNED,
    CU_MEM_HANDLE_TYPE_FABRIC,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    CU_MEM_LOCATION_TYPE_DEVICE,
    CUmemAccessDesc,
    CUmemAllocationProp,
    TensorSharedHandle,
    _is_vmm_pointer,
    libcuda,
)


# ---------------------------------------------------------------------------
# GPU / driver availability
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    return torch.cuda.is_available() and libcuda is not None


requires_cuda_vmm = pytest.mark.skipif(
    not _cuda_available(),
    reason="requires a CUDA GPU + libcuda for VMM primitives",
)


def _fabric_supported(device_id: int = 0) -> bool:
    """Query CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED."""
    if libcuda is None:
        return False
    # 128 = CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
    flag = ctypes.c_int(0)
    res = libcuda.cuDeviceGetAttribute(
        ctypes.byref(flag), ctypes.c_int(128), ctypes.c_int(device_id)
    ) if hasattr(libcuda, "cuDeviceGetAttribute") else 1
    return res == CUDA_SUCCESS and flag.value != 0


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------


@requires_cuda_vmm
def test_is_vmm_pointer_false_for_cudamalloc() -> None:
    """Legacy cudaMalloc allocations must be reported as non-VMM."""
    t = torch.zeros(1024, dtype=torch.uint8, device="cuda:0")
    assert _is_vmm_pointer(t.data_ptr()) is False


@requires_cuda_vmm
def test_is_vmm_pointer_true_for_vmm() -> None:
    """A cuMemCreate/cuMemMap allocation must be reported as VMM."""
    va, alloc_size, mem_handle = _vmm_alloc(bytes_=2 * 1024 * 1024, device_id=0)
    try:
        assert _is_vmm_pointer(va) is True
    finally:
        _vmm_free(va, alloc_size, mem_handle)


# ---------------------------------------------------------------------------
# Full round-trip through TensorSharedHandle
# ---------------------------------------------------------------------------


@requires_cuda_vmm
def test_vmm_export_produces_fabric_or_skips() -> None:
    """Exporting a VMM tensor should yield a fabric handle when supported.

    On boxes without fabric (older drivers / no imex), the exporter is
    expected to raise NotImplementedError with a message pointing users at
    sleep_mode; skip rather than fail there.
    """
    if not _fabric_supported():
        pytest.skip("fabric handle type not supported on this device")

    va, alloc_size, mem_handle = _vmm_alloc_shareable_fabric(
        bytes_=2 * 1024 * 1024, device_id=0
    )
    try:
        # Wrap the VA range as a torch tensor via the existing zero-copy
        # helper so we exercise the same code path FlexKV uses.
        tensor = TensorSharedHandle._create_tensor_from_cuda_ptr(
            va,
            shape=(alloc_size,),
            dtype=torch.uint8,
            device=torch.device("cuda:0"),
        )
        handle = TensorSharedHandle(tensor)
        assert handle.handle_type == "vmm_fabric"
        assert handle.ipc_handle is not None
        assert len(handle.ipc_handle) == 64
        assert handle.vmm_allocation_size == alloc_size
        assert handle.vmm_granularity > 0
        assert handle.offset == 0
    finally:
        _vmm_free(va, alloc_size, mem_handle)


@requires_cuda_vmm
def test_vmm_round_trip_across_processes() -> None:
    """Import in a subprocess should see the same memory (writes propagate)."""
    if not _fabric_supported():
        pytest.skip("fabric handle type not supported on this device")

    # Preferable to use spawn so the child has a fresh CUDA context.
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_vmm_importer_child,
        args=(child_conn,),
        daemon=True,
    )
    proc.start()
    try:
        va, alloc_size, mem_handle = _vmm_alloc_shareable_fabric(
            bytes_=2 * 1024 * 1024, device_id=0
        )
        try:
            tensor = TensorSharedHandle._create_tensor_from_cuda_ptr(
                va,
                shape=(alloc_size,),
                dtype=torch.uint8,
                device=torch.device("cuda:0"),
            )
            # Pre-write a sentinel so the child can verify visibility.
            tensor.fill_(0x5A)
            torch.cuda.synchronize()

            handle = TensorSharedHandle(tensor)
            parent_conn.send({
                "handle_type": handle.handle_type,
                "ipc_handle": handle.ipc_handle,
                "shape": handle.tensor_shape,
                "dtype": str(handle.tensor_dtype),
                "device_id": 0,
                "offset": handle.offset,
                "size": handle.vmm_allocation_size,
                "granularity": handle.vmm_granularity,
            })

            reply = parent_conn.recv()
            assert reply["status"] == "ok", reply
            assert reply["first_byte"] == 0x5A
        finally:
            _vmm_free(va, alloc_size, mem_handle)
    finally:
        proc.join(timeout=30)
        assert proc.exitcode == 0, f"child exit code = {proc.exitcode}"


# ---------------------------------------------------------------------------
# Helpers: raw driver-API allocation without depending on vLLM
# ---------------------------------------------------------------------------


def _vmm_alloc(bytes_: int, device_id: int):
    """Allocate a legacy VMM range (no shareable handle) for detection tests."""
    return _vmm_alloc_shareable(bytes_, device_id, handle_type=0)


def _vmm_alloc_shareable_fabric(bytes_: int, device_id: int):
    return _vmm_alloc_shareable(
        bytes_, device_id, handle_type=CU_MEM_HANDLE_TYPE_FABRIC
    )


def _vmm_alloc_shareable(bytes_: int, device_id: int, handle_type: int):
    assert libcuda is not None
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    prop.requestedHandleTypes = handle_type

    granularity = ctypes.c_size_t()
    _check(libcuda.cuMemGetAllocationGranularity(
        ctypes.byref(granularity),
        ctypes.byref(prop),
        ctypes.c_int(0),  # MINIMUM
    ))
    align = int(granularity.value)
    aligned = ((bytes_ + align - 1) // align) * align

    mem_handle = ctypes.c_void_p()
    _check(libcuda.cuMemCreate(
        ctypes.byref(mem_handle),
        ctypes.c_size_t(aligned),
        ctypes.byref(prop),
        ctypes.c_ulonglong(0),
    ))
    va = ctypes.c_ulonglong(0)
    _check(libcuda.cuMemAddressReserve(
        ctypes.byref(va),
        ctypes.c_size_t(aligned),
        ctypes.c_size_t(align),
        ctypes.c_ulonglong(0),
        ctypes.c_ulonglong(0),
    ))
    _check(libcuda.cuMemMap(
        ctypes.c_ulonglong(va.value),
        ctypes.c_size_t(aligned),
        ctypes.c_ulonglong(0),
        mem_handle,
        ctypes.c_ulonglong(0),
    ))
    access = CUmemAccessDesc()
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device_id
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _check(libcuda.cuMemSetAccess(
        ctypes.c_ulonglong(va.value),
        ctypes.c_size_t(aligned),
        ctypes.byref(access),
        ctypes.c_size_t(1),
    ))
    return va.value, aligned, mem_handle


def _vmm_free(va: int, size: int, mem_handle) -> None:
    assert libcuda is not None
    libcuda.cuMemUnmap(ctypes.c_ulonglong(va), ctypes.c_size_t(size))
    libcuda.cuMemAddressFree(ctypes.c_ulonglong(va), ctypes.c_size_t(size))
    libcuda.cuMemRelease(mem_handle)


def _check(res: int) -> None:
    if res != CUDA_SUCCESS:
        raise RuntimeError(f"driver call returned CUresult={res}")


def _vmm_importer_child(conn) -> None:
    """Subprocess entry point: import handle, verify data, echo back."""
    try:
        msg = conn.recv()
        handle = TensorSharedHandle(
            data=msg["ipc_handle"],
            device_id=msg["device_id"],
            tensor_shape=msg["shape"],
            tensor_dtype=msg["dtype"],
            offset=msg["offset"],
            handle_type=msg["handle_type"],
            vmm_allocation_size=msg["size"],
            vmm_granularity=msg["granularity"],
        )
        tensor = handle.get_tensor()
        torch.cuda.synchronize()
        first_byte = int(tensor[0].item())
        conn.send({"status": "ok", "first_byte": first_byte})
    except Exception as exc:  # noqa: BLE001
        conn.send({"status": "error", "error": repr(exc)})
        raise
