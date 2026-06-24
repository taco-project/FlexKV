from __future__ import annotations

import os

import pytest
import torch

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import DeviceType
from flexkv.storage.storage_engine import StorageEngine
from flexkv.storage.allocator import (
    DEFAULT_HUGE_PAGE_SIZE,
    HugePageTensorHandle,
    HugePageAllocator,
    _live_hugepage_mappings,
    alloc_hugepage_tensor,
    free_hugepage_tensor,
    get_worker_hugepage_handle,
    materialize_worker_tensor,
)
from tests.hugepage.conftest import (
    alloc_hugepage_or_skip,
    cuda_ops_or_skip,
    unregister_suppress,
)

PAGE = DEFAULT_HUGE_PAGE_SIZE


def test_basic_alloc_free() -> None:
    n_bytes = 16 * 1024 * 1024
    n_elem = n_bytes // 2
    tensor = alloc_hugepage_or_skip(n_elem, torch.bfloat16, PAGE)
    addr = tensor.data_ptr()

    try:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.numel() == n_elem
        assert tensor.dtype == torch.bfloat16
        assert tensor.device.type == "cpu"
        assert addr != 0
        assert addr % PAGE == 0
        assert addr in _live_hugepage_mappings

        tensor.view(torch.int16).fill_(0x5A5A)
        assert int(tensor.view(torch.int16)[0].item()) == 0x5A5A
    finally:
        free_hugepage_tensor(tensor)

    assert addr not in _live_hugepage_mappings


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        alloc_hugepage_tensor(0, torch.float32, page_size_bytes=PAGE)

    with pytest.raises(ValueError):
        alloc_hugepage_tensor(1, torch.float32, page_size_bytes=PAGE + 1)


def test_non_hugetlbfs_fallback_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("FLEXKV_HUGETLBFS_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="is not a hugetlbfs mount"):
        alloc_hugepage_tensor(1024, torch.float32, page_size_bytes=1024 * 1024 * 1024)


def test_worker_hugepage_handle_round_trip() -> None:
    tensor = alloc_hugepage_or_skip(1024 * 1024, torch.bfloat16, PAGE)

    try:
        handle = get_worker_hugepage_handle(tensor, tensor.numel(), tensor.dtype)
        if handle is None:
            pytest.skip("non-shareable hugepage allocation path on this host")

        rebuilt = materialize_worker_tensor(handle)
        rebuilt.view(torch.int16)[0] = 0x1234

        assert isinstance(handle, HugePageTensorHandle)
        assert int(tensor.view(torch.int16)[0].item()) == 0x1234
        free_hugepage_tensor(rebuilt)
    finally:
        free_hugepage_tensor(tensor)


def test_hugepage_allocator_fallback() -> None:
    layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=2,
        num_block=64,
        tokens_per_block=16,
        num_head=8,
        head_size=128,
        is_mla=False,
    )
    dtype = torch.bfloat16
    page_size = 1024 * 1024 * 1024
    old_dir = os.environ.get("FLEXKV_HUGETLBFS_DIR")
    os.environ["FLEXKV_HUGETLBFS_DIR"] = "/nonexistent/flexkv_hugetlbfs"
    try:
        handle = HugePageAllocator.allocate(
            layout=layout,
            dtype=dtype,
            page_size_bytes=page_size,
        )
    finally:
        if old_dir is None:
            os.environ.pop("FLEXKV_HUGETLBFS_DIR", None)
        else:
            os.environ["FLEXKV_HUGETLBFS_DIR"] = old_dir

    tensor = handle.get_tensor()
    assert isinstance(tensor, torch.Tensor)
    assert tensor.numel() == layout.get_total_elements()
    assert tensor.dtype == dtype
    assert tensor.data_ptr() not in _live_hugepage_mappings
    HugePageAllocator.free(handle)


def test_cuda_host_register() -> None:
    cudaHostRegister, _ = cuda_ops_or_skip()
    tensor = alloc_hugepage_or_skip(1024 * 1024, torch.bfloat16, PAGE)

    try:
        cudaHostRegister(tensor)

        gpu_tensor = torch.empty_like(tensor, device="cuda")
        tensor.fill_(1.25)
        gpu_tensor.copy_(tensor, non_blocking=True)
        torch.cuda.synchronize()
        out = torch.empty_like(tensor)
        out.copy_(gpu_tensor, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.all(out == 1.25).item()
    finally:
        unregister_suppress(tensor)
        free_hugepage_tensor(tensor)


def test_host_buffer_release_is_idempotent() -> None:
    from flexkv.transfer.host_buffer import allocate_host_buffer

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for pinned host buffer allocation")

    handle = allocate_host_buffer(
        num_elements=1024,
        dtype=torch.bfloat16,
        use_hugepage=False,
        hugepage_size_bytes=PAGE,
    )

    handle.release()
    handle.release()

    assert not handle.is_hugepage
    assert not handle.is_cuda_registered


def test_storage_engine_cpu_cache_uses_hugepage_when_enabled() -> None:
    if not os.path.isdir(os.environ.get("FLEXKV_HUGETLBFS_DIR", "/mnt/hugepages")):
        pytest.skip("hugetlbfs mount not available")

    model_config = ModelConfig(
        num_layers=1,
        num_kv_heads=1,
        head_size=128,
        use_mla=False,
        dtype=torch.bfloat16,
    )
    cache_config = CacheConfig(
        enable_cpu=True,
        enable_ssd=False,
        num_cpu_blocks=8,
        tokens_per_block=16,
        use_hugepage_cpu_buffer=True,
        hugepage_size_bytes=PAGE,
    )

    storage_engine = StorageEngine(model_config, cache_config)
    cpu_handle = storage_engine.get_storage_handle(DeviceType.CPU)
    cpu_tensor = cpu_handle.get_tensor()

    try:
        if cpu_tensor.data_ptr() not in _live_hugepage_mappings:
            pytest.skip("hugepage CPU cache allocation fell back on this host")
        assert cpu_tensor.data_ptr() in _live_hugepage_mappings
        worker_tensor = cpu_handle.get_worker_tensor()
        assert isinstance(worker_tensor, HugePageTensorHandle)
    finally:
        HugePageAllocator.free(cpu_handle)


def test_storage_engine_cpu_cache_falls_back_when_hugepage_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    model_config = ModelConfig(
        num_layers=1,
        num_kv_heads=1,
        head_size=128,
        use_mla=False,
        dtype=torch.bfloat16,
    )
    cache_config = CacheConfig(
        enable_cpu=True,
        enable_ssd=False,
        num_cpu_blocks=8,
        tokens_per_block=16,
        use_hugepage_cpu_buffer=True,
        hugepage_size_bytes=1024 * 1024 * 1024,
    )
    monkeypatch.setenv("FLEXKV_HUGETLBFS_DIR", "/nonexistent/flexkv_hugetlbfs")

    storage_engine = StorageEngine(model_config, cache_config)
    cpu_handle = storage_engine.get_storage_handle(DeviceType.CPU)
    cpu_tensor = cpu_handle.get_tensor()

    assert cpu_tensor.data_ptr() not in _live_hugepage_mappings
    HugePageAllocator.free(cpu_handle)
