from __future__ import annotations

import gc
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from flexkv.storage.allocator import (
    DEFAULT_HUGE_PAGE_SIZE,
    alloc_hugepage_tensor,
    free_hugepage_tensor,
)
from tests.hugepage.conftest import cuda_ops_or_skip, unregister_suppress

PAGE = DEFAULT_HUGE_PAGE_SIZE
_NUM_PAGES = 8
_NUM_BYTES = _NUM_PAGES * PAGE
_NUM_ELEMENTS = _NUM_BYTES // 2


def _read_meminfo_hugepages() -> tuple[int, int, int]:
    total = free = size_kb = 0
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("HugePages_Total:"):
                total = int(line.split()[1])
            elif line.startswith("HugePages_Free:"):
                free = int(line.split()[1])
            elif line.startswith("Hugepagesize:"):
                size_kb = int(line.split()[1])
    return total, free, size_kb * 1024


def _require_hugepages(num_pages: int) -> tuple[int, int]:
    total, free, _ = _read_meminfo_hugepages()
    if total < num_pages:
        pytest.skip(f"need at least {num_pages} huge pages")
    return total, free


def _simulate_tmp_cpu_buffer_init(tmp_num_elements: int) -> tuple[torch.Tensor, bool]:
    tmp_cpu_buffer = torch.empty(
        tmp_num_elements,
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )
    needs_unpin = False
    hugepage_buf = None
    cudaHostRegister, _ = cuda_ops_or_skip()
    try:
        hugepage_buf = alloc_hugepage_tensor(
            num_elements=tmp_num_elements,
            dtype=torch.bfloat16,
            page_size_bytes=PAGE,
        )
        cudaHostRegister(hugepage_buf)
    except Exception:
        if hugepage_buf is not None:
            free_hugepage_tensor(hugepage_buf)
    else:
        tmp_cpu_buffer = hugepage_buf
        needs_unpin = True

    return tmp_cpu_buffer, needs_unpin


class MockMooncakeEngine:
    def __init__(self) -> None:
        self.registered: set[int] = set()

    def regist_buffer(self, ptr: int, size: int) -> int:
        assert ptr != 0
        assert size > 0
        self.registered.add(ptr)
        return 0

    def unregist_buffer(self, ptr: int) -> int:
        assert ptr in self.registered
        self.registered.discard(ptr)
        return 0


def test_full_lifecycle_hugepage() -> None:
    _, free_before = _require_hugepages(_NUM_PAGES)
    mooncake = MockMooncakeEngine()
    tmp_cpu_buffer, needs_unpin = _simulate_tmp_cpu_buffer_init(_NUM_ELEMENTS)

    mooncake.regist_buffer(
        tmp_cpu_buffer.data_ptr(),
        tmp_cpu_buffer.numel() * tmp_cpu_buffer.element_size(),
    )

    _, free_after_alloc, _ = _read_meminfo_hugepages()
    consumed = free_before - free_after_alloc
    if needs_unpin:
        assert consumed == _NUM_PAGES
        assert len(mooncake.registered) == 1
    else:
        assert consumed == 0

    tmp_cpu_buffer.view(torch.int16).fill_(0x7B7B)
    assert int(tmp_cpu_buffer.view(torch.int16)[0].item()) == 0x7B7B

    mooncake.unregist_buffer(tmp_cpu_buffer.data_ptr())
    if needs_unpin:
        unregister_suppress(tmp_cpu_buffer)
        free_hugepage_tensor(tmp_cpu_buffer)

    assert len(mooncake.registered) == 0

    del tmp_cpu_buffer
    gc.collect()

    _, free_after_free, _ = _read_meminfo_hugepages()
    assert free_after_free == free_before


def test_fallback_when_cuda_host_register_fails() -> None:
    _, free_before = _require_hugepages(_NUM_PAGES)

    with patch(
        "flexkv.transfer.host_buffer.cudaHostRegister",
        side_effect=RuntimeError("injected cudaHostRegister failure"),
    ):
        tmp_cpu_buffer, needs_unpin = _simulate_tmp_cpu_buffer_init(_NUM_ELEMENTS)
        assert not needs_unpin

        mooncake = MockMooncakeEngine()
        mooncake.regist_buffer(
            tmp_cpu_buffer.data_ptr(),
            tmp_cpu_buffer.numel() * tmp_cpu_buffer.element_size(),
        )
        mooncake.unregist_buffer(tmp_cpu_buffer.data_ptr())

        del tmp_cpu_buffer
        gc.collect()

    _, free_after, _ = _read_meminfo_hugepages()
    assert free_after == free_before
