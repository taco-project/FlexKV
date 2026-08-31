"""Unit tests for large HugePage-backed Mooncake registrations."""

import pytest

from flexkv.storage.allocator import _shareable_mapping_alignment
from flexkv.transfer.worker import (
    _register_mooncake_regions,
    _split_mooncake_registration_regions,
    _unregister_mooncake_regions,
)


HUGEPAGE_SIZE = 2 << 20
MAPPING_ALIGNMENT = 32 << 20
MAX_MR_SIZE = 512 << 30
BLOCK_SIZE = 6_409_728
LOGICAL_SIZE = 847_564_743_168
MAPPED_SIZE = 847_584_952_320
BASE_PTR = 0x80000000


class _RecordingClient:
    def __init__(self, fail_on_register: int | None = None):
        self.fail_on_register = fail_on_register
        self.register_calls = []
        self.unregister_calls = []

    def register_buffer(self, ptr: int, size: int) -> None:
        self.register_calls.append((ptr, size))
        if self.fail_on_register == len(self.register_calls):
            raise RuntimeError("injected registration failure")

    def unregister_buffer(self, ptr: int) -> None:
        self.unregister_calls.append(ptr)


def test_shareable_mapping_alignment_is_opt_in(monkeypatch):
    monkeypatch.delenv("FLEXKV_HUGEPAGE_MAPPING_ALIGNMENT_BYTES", raising=False)
    assert _shareable_mapping_alignment(HUGEPAGE_SIZE) == HUGEPAGE_SIZE

    monkeypatch.setenv("FLEXKV_HUGEPAGE_MAPPING_ALIGNMENT_BYTES", str(MAPPING_ALIGNMENT))
    assert _shareable_mapping_alignment(HUGEPAGE_SIZE) == MAPPING_ALIGNMENT


@pytest.mark.parametrize("value", ["invalid", str(HUGEPAGE_SIZE // 2), str(24 << 20)])
def test_shareable_mapping_alignment_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("FLEXKV_HUGEPAGE_MAPPING_ALIGNMENT_BYTES", value)
    with pytest.raises(ValueError):
        _shareable_mapping_alignment(HUGEPAGE_SIZE)


def test_large_glm_pool_splits_into_two_aligned_regions():
    regions = _split_mooncake_registration_regions(
        base_ptr=BASE_PTR,
        logical_size=LOGICAL_SIZE,
        mapped_size=MAPPED_SIZE,
        block_size=BLOCK_SIZE,
        max_mr_size=MAX_MR_SIZE,
        size_alignment=MAPPING_ALIGNMENT,
        pointer_alignment=HUGEPAGE_SIZE,
    )

    assert regions == [
        (BASE_PTR, 420_067_934_208),
        (BASE_PTR + 420_067_934_208, 427_517_018_112),
    ]
    assert sum(size for _, size in regions) == MAPPED_SIZE
    assert all(size <= MAX_MR_SIZE for _, size in regions)
    assert all(ptr % HUGEPAGE_SIZE == 0 for ptr, _ in regions)
    assert all(size % MAPPING_ALIGNMENT == 0 for _, size in regions)
    assert regions[0][1] % BLOCK_SIZE == 0
    assert regions[-1][0] + regions[-1][1] >= BASE_PTR + LOGICAL_SIZE


def test_2g_limit_rejects_unconfigured_large_block_alignment():
    block_size = 25_638_912
    mapped_size = 8 << 30
    logical_size = (mapped_size // block_size) * block_size

    with pytest.raises(ValueError, match="cannot hold one aligned KV region"):
        _split_mooncake_registration_regions(
            base_ptr=BASE_PTR,
            logical_size=logical_size,
            mapped_size=mapped_size,
            block_size=block_size,
            max_mr_size=2 << 30,
            size_alignment=MAPPING_ALIGNMENT,
            pointer_alignment=HUGEPAGE_SIZE,
        )


def test_2g_aligned_mrs_may_span_blocks_when_explicitly_enabled():
    block_size = 25_638_912
    mapped_size = 700 << 30
    logical_size = (mapped_size // block_size) * block_size

    regions = _split_mooncake_registration_regions(
        base_ptr=BASE_PTR,
        logical_size=logical_size,
        mapped_size=mapped_size,
        block_size=block_size,
        max_mr_size=2 << 30,
        size_alignment=MAPPING_ALIGNMENT,
        pointer_alignment=HUGEPAGE_SIZE,
        allow_block_spanning_mrs=True,
    )

    assert len(regions) == 350
    assert all(size == 2 << 30 for _, size in regions)
    assert all(ptr % HUGEPAGE_SIZE == 0 for ptr, _ in regions)
    assert regions[0][1] % block_size != 0


@pytest.mark.parametrize("mapped_gib,expected_regions", [(8, 5), (700, 354)])
def test_2g_unaligned_mrs_keep_every_large_block_inside_one_mr(mapped_gib, expected_regions):
    block_size = 25_638_912
    mapped_size = mapped_gib << 30
    logical_size = (mapped_size // block_size) * block_size

    regions = _split_mooncake_registration_regions(
        base_ptr=BASE_PTR,
        logical_size=logical_size,
        mapped_size=mapped_size,
        block_size=block_size,
        max_mr_size=2 << 30,
        size_alignment=MAPPING_ALIGNMENT,
        pointer_alignment=HUGEPAGE_SIZE,
        allow_unaligned_block_mrs=True,
    )

    assert len(regions) == expected_regions
    assert sum(size for _, size in regions) == mapped_size
    assert all(size <= 2 << 30 for _, size in regions)
    assert all(size % block_size == 0 for _, size in regions[:-1])
    assert all((ptr - BASE_PTR) % block_size == 0 for ptr, _ in regions)
    assert regions[-1][0] + regions[-1][1] == BASE_PTR + mapped_size
    assert any(ptr % HUGEPAGE_SIZE != 0 for ptr, _ in regions[1:])


def test_mooncake_split_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _split_mooncake_registration_regions(
            base_ptr=BASE_PTR,
            logical_size=BLOCK_SIZE * 2,
            mapped_size=MAPPING_ALIGNMENT,
            block_size=BLOCK_SIZE,
            max_mr_size=2 << 30,
            size_alignment=MAPPING_ALIGNMENT,
            pointer_alignment=HUGEPAGE_SIZE,
            allow_block_spanning_mrs=True,
            allow_unaligned_block_mrs=True,
        )


def test_partial_registration_failure_rolls_back_completed_regions():
    regions = [(BASE_PTR, MAPPING_ALIGNMENT), (BASE_PTR + MAPPING_ALIGNMENT, MAPPING_ALIGNMENT)]
    client = _RecordingClient(fail_on_register=2)

    with pytest.raises(RuntimeError, match="injected registration failure"):
        _register_mooncake_regions(client, regions)

    assert client.register_calls == regions
    assert client.unregister_calls == [BASE_PTR]


def test_unregister_runs_in_reverse_order():
    regions = [(BASE_PTR, MAPPING_ALIGNMENT), (BASE_PTR + MAPPING_ALIGNMENT, MAPPING_ALIGNMENT)]
    client = _RecordingClient()

    _unregister_mooncake_regions(client, regions)

    assert client.unregister_calls == [BASE_PTR + MAPPING_ALIGNMENT, BASE_PTR]
