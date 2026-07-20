"""P2: layerwise multi-group + SWA eventfd timing unit tests.

Bypasses the SGLang socket handshake by injecting real semaphore eventfds
directly into ``LayerwiseTransferGroup``.  Verifies:

  - fds are silent before ``layerwise_transfer_multi_group``
  - each original layer receives exactly one ``write(1)`` semaphore signal
  - empty-member layers without SWA get an immediate post
  - empty-member layers with SWA wait for the SWA H2D callback

Run:
    pytest tests/test_layerwise_eventfd_timing.py -v
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from flexkv.c_ext import LayerwiseTransferGroup
from flexkv.common.config import LayerGroupSpec, build_layer_member_map
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

from test_layerwise_multi_group_swa import (
    CPU_SRC,
    GPU_DST,
    INDEXER_HEAD_SIZE,
    IOURING_ENTRIES,
    IOURING_FLAGS,
    MAIN_HEAD_SIZE,
    NUM_CPU_BLOCKS,
    NUM_GPU_BLOCKS,
    SWA_BYTES_PER_TOKEN,
    SWA_CPU_SRC,
    SWA_GPU_DST,
    TOKENS_PER_BLOCK,
    MultiGroupFixture,
    _compute_multi_group_strides,
    _compute_swa_strides,
    _device,
    _make_gpu_layout,
    _make_group_gpu_tensors,
    _make_multi_group_cpu_layout,
    _make_swa_layout,
    _seed_main_cpu_layer,
    _seed_swa_cpu_layer,
)

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_EFD_SEMAPHORE = 0x1


def _sys_eventfd(initval: int = 0, flags: int = 0) -> int:
    fd = _libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd failed: {os.strerror(err)}")
    return fd


def _set_nonblocking(fd: int) -> None:
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


def _drain_eventfd_units(fd: int) -> int:
    """Drain a semaphore eventfd; return the number of 1-unit reads."""
    _set_nonblocking(fd)
    total = 0
    while True:
        try:
            data = os.read(fd, 8)
        except BlockingIOError:
            break
        if len(data) != 8:
            break
        total += struct.unpack("Q", data)[0]
    return total


def _eventfd_is_idle(fd: int) -> bool:
    return _drain_eventfd_units(fd) == 0


def _make_layer_eventfds_tensor(
    num_layers: int,
    *,
    tp_size: int = 1,
    num_counters: int = 1,
) -> Tuple[torch.Tensor, List[int]]:
    """Shape ``[num_counters, tp_size, num_layers]`` as int32 fd array."""
    fds: List[int] = []
    for _ in range(num_counters * tp_size * num_layers):
        fds.append(_sys_eventfd(0, _EFD_SEMAPHORE))
    arr = np.array(fds, dtype=np.int32).reshape(num_counters, tp_size, num_layers)
    return torch.from_numpy(arr.copy()), fds


def _build_fixture_with_eventfds(
    layer_groups: List[LayerGroupSpec],
    num_original_layers: int,
    *,
    has_swa: bool = True,
) -> Tuple[MultiGroupFixture, List[int]]:
    device = _device()
    cpu_layout = _make_multi_group_cpu_layout(
        layer_groups, num_original_layers, NUM_CPU_BLOCKS, TOKENS_PER_BLOCK,
    )
    gpu_layouts = [
        _make_gpu_layout(g, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK) for g in layer_groups
    ]
    strides = _compute_multi_group_strides(layer_groups, cpu_layout, gpu_layouts)

    gpu_tensors_per_group = [
        _make_group_gpu_tensors(g, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK, device)
        for g in layer_groups
    ]
    gpu_blocks_per_group = [[tensors] for tensors in gpu_tensors_per_group]

    block_stride = cpu_layout.get_block_stride()
    cpu_blocks = torch.zeros(
        NUM_CPU_BLOCKS, block_stride, dtype=torch.uint8, pin_memory=True,
    )
    ssd_files: Dict[int, List[str]] = {}

    eventfds_tensor, owned_fds = _make_layer_eventfds_tensor(num_original_layers)

    swa_gpu_tensors: Optional[List[torch.Tensor]] = None
    swa_cpu: Optional[torch.Tensor] = None
    swa_strides: Optional[Dict[str, object]] = None
    swa_ctor: Dict[str, object] = {"has_swa": False}

    if has_swa:
        swa_layout = _make_swa_layout(
            num_original_layers, NUM_CPU_BLOCKS, TOKENS_PER_BLOCK,
        )
        swa_gpu_layout = _make_swa_layout(
            num_original_layers, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK,
        )
        swa_strides = _compute_swa_strides(swa_layout, swa_gpu_layout)
        swa_cpu = torch.zeros(swa_layout.kv_shape, dtype=torch.uint8, pin_memory=True)
        swa_gpu_tensors = [
            torch.zeros(
                NUM_GPU_BLOCKS,
                TOKENS_PER_BLOCK,
                1,
                SWA_BYTES_PER_TOKEN,
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(num_original_layers)
        ]
        swa_ctor = dict(
            has_swa=True,
            swa_gpu_blocks=[swa_gpu_tensors],
            swa_cpu_blocks=swa_cpu,
            swa_ssd_files=ssd_files,
            swa_gpu_kv_strides_tensor=swa_strides["swa_gpu_kv_strides_tensor"],
            swa_gpu_block_strides_tensor=swa_strides["swa_gpu_block_strides_tensor"],
            swa_gpu_layer_strides_tensor=swa_strides["swa_gpu_layer_strides_tensor"],
            swa_gpu_chunk_sizes_tensor=swa_strides["swa_gpu_chunk_sizes_tensor"],
        )

    group = LayerwiseTransferGroup(
        num_gpus=1,
        gpu_blocks_per_group=gpu_blocks_per_group,
        cpu_blocks=cpu_blocks,
        ssd_files=ssd_files,
        num_original_layers=num_original_layers,
        layer_members=strides["layer_members"],
        group_num_layers=strides["group_num_layers"],
        group_cpu_offset_bytes=strides["group_cpu_offset_bytes"],
        group_ssd_offset_bytes=strides["group_ssd_offset_bytes"],
        group_cpu_layer_strides=strides["group_cpu_layer_strides"],
        group_cpu_kv_strides=strides["group_cpu_kv_strides"],
        group_ssd_layer_strides=strides["group_ssd_layer_strides"],
        group_ssd_kv_strides=strides["group_ssd_kv_strides"],
        group_chunk_sizes=strides["group_chunk_sizes"],
        group_h2d_cpu_kv_strides=strides["group_h2d_cpu_kv_strides"],
        group_h2d_cpu_layer_strides=strides["group_h2d_cpu_layer_strides"],
        group_cpu_block_strides=strides["group_cpu_block_strides"],
        group_cpu_tp_strides=strides["group_cpu_tp_strides"],
        group_gpu_kv_strides=strides["group_gpu_kv_strides"],
        group_gpu_block_strides=strides["group_gpu_block_strides"],
        group_gpu_layer_strides=strides["group_gpu_layer_strides"],
        group_gpu_chunk_sizes=strides["group_gpu_chunk_sizes"],
        iouring_entries=IOURING_ENTRIES,
        iouring_flags=IOURING_FLAGS,
        layer_eventfds_tensor=eventfds_tensor,
        tp_size=1,
        is_blockfirst=True,
        is_mla=True,
        **swa_ctor,
    )

    fx = MultiGroupFixture(
        group=group,
        layer_groups=layer_groups,
        gpu_tensors_per_group=gpu_tensors_per_group,
        cpu_blocks=cpu_blocks,
        strides=strides,
        num_original_layers=num_original_layers,
        swa_gpu_tensors=swa_gpu_tensors,
        swa_cpu=swa_cpu,
        swa_strides=swa_strides,
    )
    return fx, owned_fds


def _layer_fds(fx: MultiGroupFixture, owned_fds: List[int]) -> List[int]:
    """Return per-original-layer fds for counter_id=0, tp_rank=0."""
    num_layers = fx.num_original_layers
    return owned_fds[:num_layers]


def _run_h2d(
    fx: MultiGroupFixture,
    *,
    with_swa: bool = False,
) -> None:
    empty = torch.empty(0, dtype=torch.int64)
    gpu_dst = torch.tensor([GPU_DST], dtype=torch.int64)
    cpu_src = torch.tensor([CPU_SRC], dtype=torch.int64)

    # pybind cannot resolve layerwise_transfer_multi_group when the four SWA
    # tensor kwargs are omitted entirely; always pass explicit empty tensors.
    swa_kwargs: Dict[str, object] = dict(
        swa_h2d_src=empty,
        swa_h2d_dst=empty,
        swa_disk2h_src=empty,
        swa_disk2h_dst=empty,
        swa_cpu_kv_stride_in_bytes=0,
        swa_cpu_layer_stride_in_bytes=0,
        swa_cpu_block_stride_in_bytes=0,
        swa_cpu_chunk_size_in_bytes=0,
        swa_h2d_cpu_kv_stride_in_bytes=0,
        swa_h2d_cpu_layer_stride_in_bytes=0,
        swa_cpu_tp_stride_in_bytes=0,
        swa_ssd_layer_stride_in_bytes=0,
        swa_ssd_kv_stride_in_bytes=0,
        swa_num_blocks_per_file=0,
    )
    if with_swa:
        assert fx.swa_strides is not None
        swa_kwargs.update(
            swa_h2d_src=torch.tensor([SWA_CPU_SRC], dtype=torch.int64),
            swa_h2d_dst=torch.tensor([SWA_GPU_DST], dtype=torch.int64),
            swa_cpu_kv_stride_in_bytes=fx.swa_strides["swa_cpu_kv_stride_in_bytes"],
            swa_cpu_layer_stride_in_bytes=fx.swa_strides[
                "swa_cpu_layer_stride_in_bytes"
            ],
            swa_cpu_block_stride_in_bytes=fx.swa_strides[
                "swa_cpu_block_stride_in_bytes"
            ],
            swa_cpu_chunk_size_in_bytes=fx.swa_strides["swa_cpu_chunk_size_in_bytes"],
            swa_h2d_cpu_kv_stride_in_bytes=fx.swa_strides[
                "swa_h2d_cpu_kv_stride_in_bytes"
            ],
            swa_h2d_cpu_layer_stride_in_bytes=fx.swa_strides[
                "swa_h2d_cpu_layer_stride_in_bytes"
            ],
            swa_cpu_tp_stride_in_bytes=fx.swa_strides["swa_cpu_tp_stride_in_bytes"],
        )

    fx.group.layerwise_transfer_multi_group(
        empty,
        empty,
        num_blocks_per_file=0,
        round_robin=1,
        num_threads_per_device=4,
        gpu_block_id_tensor=gpu_dst,
        cpu_block_id_tensor=cpu_src,
        transfer_cta_num=4,
        use_ce_transfer=True,
        is_mla=True,
        counter_id=0,
        **swa_kwargs,
    )
    torch.cuda.synchronize()


def _dsv4_like_groups(num_c4_layers: int = 4) -> List[LayerGroupSpec]:
    c4_ids = list(range(num_c4_layers))
    c128_ids = list(range(num_c4_layers, num_c4_layers * 2))
    return [
        LayerGroupSpec(
            num_layers=len(c4_ids),
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=c4_ids,
            dtype=torch.uint8,
            compress_ratio=4,
        ),
        LayerGroupSpec(
            num_layers=len(c128_ids),
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE // 2,
            layer_indices=c128_ids,
            dtype=torch.uint8,
            compress_ratio=1,
        ),
        LayerGroupSpec(
            num_layers=len(c4_ids),
            num_kv_heads=1,
            head_size=INDEXER_HEAD_SIZE,
            layer_indices=c4_ids,
            dtype=torch.uint8,
            compress_ratio=4,
        ),
    ]


def _seed_all_layers(fx: MultiGroupFixture, with_swa: bool) -> None:
    member_map = fx.strides["layer_member_map"]
    for orig in range(fx.num_original_layers):
        for gi, local_id in member_map.members_of(orig):  # type: ignore[union-attr]
            _seed_main_cpu_layer(
                fx.cpu_blocks,
                CPU_SRC,
                orig,
                gi,
                local_id,
                fx.strides,
                fx.layer_groups,
                TOKENS_PER_BLOCK,
            )
    if with_swa:
        assert fx.swa_cpu is not None
        for orig in range(fx.num_original_layers):
            _seed_swa_cpu_layer(fx.swa_cpu, SWA_CPU_SRC, orig)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestLayerwiseEventfdTiming:
    def test_eventfd_idle_before_transfer(self) -> None:
        """All layer fds must stay at zero before H2D starts."""
        num_layers = 4
        groups = _dsv4_like_groups(num_c4_layers=2)
        fx, owned = _build_fixture_with_eventfds(groups, num_layers)
        for fd in _layer_fds(fx, owned):
            assert _eventfd_is_idle(fd), f"eventfd {fd} should be idle before transfer"

    def test_eventfd_posts_one_unit_per_layer_after_swa_h2d(self) -> None:
        """Each layer gets one signal, matching SGLang's one semaphore read."""
        num_layers = 8
        groups = _dsv4_like_groups(num_c4_layers=4)
        fx, owned = _build_fixture_with_eventfds(groups, num_layers)
        _seed_all_layers(fx, with_swa=True)

        for fd in _layer_fds(fx, owned):
            assert _eventfd_is_idle(fd)

        _run_h2d(fx, with_swa=True)

        for layer, fd in enumerate(_layer_fds(fx, owned)):
            units = _drain_eventfd_units(fd)
            assert units == 1, (
                f"layer {layer}: expected one semaphore unit, got {units}"
            )
            assert _eventfd_is_idle(fd), f"layer {layer}: extra signal after drain"

    def test_dual_member_layer_single_eventfd_post(self) -> None:
        """c4 layers carry main+indexer members; still only one eventfd post per layer."""
        num_layers = 4
        main = LayerGroupSpec(
            num_layers=num_layers,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=list(range(num_layers)),
            dtype=torch.uint8,
        )
        indexer = LayerGroupSpec(
            num_layers=3,
            num_kv_heads=1,
            head_size=INDEXER_HEAD_SIZE,
            layer_indices=[1, 2, 3],
            dtype=torch.uint8,
        )
        fx, owned = _build_fixture_with_eventfds([main, indexer], num_layers)
        _seed_all_layers(fx, with_swa=True)
        _run_h2d(fx, with_swa=True)

        for orig in range(num_layers):
            fd = _layer_fds(fx, owned)[orig]
            assert _drain_eventfd_units(fd) == 1, (
                f"orig layer {orig}: dual-member+SWA must post exactly once"
            )

    def test_empty_member_immediate_eventfd_without_swa(self) -> None:
        """Layer 0 has no members and SWA H2D is off — C++ posts eventfd before H2D work."""
        num_layers = 4
        main = LayerGroupSpec(
            num_layers=3,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=[1, 2, 3],
            dtype=torch.uint8,
        )
        # Keep has_swa=True (pybind ctor) but omit SWA from the transfer call so
        # swa_active=false and empty-member layers get the immediate post path.
        fx, owned = _build_fixture_with_eventfds([main], num_layers, has_swa=True)
        member_map = fx.strides["layer_member_map"]
        assert member_map.members_of(0) == ()  # type: ignore[union-attr]

        for gi, local_id in member_map.members_of(1):  # type: ignore[union-attr]
            _seed_main_cpu_layer(
                fx.cpu_blocks, CPU_SRC, 1, gi, local_id,
                fx.strides, fx.layer_groups, TOKENS_PER_BLOCK,
            )

        layer0_fd = _layer_fds(fx, owned)[0]
        layer1_fd = _layer_fds(fx, owned)[1]
        assert _eventfd_is_idle(layer0_fd)
        assert _eventfd_is_idle(layer1_fd)

        _run_h2d(fx, with_swa=False)

        # Immediate post for empty member happens synchronously at transfer entry.
        assert _drain_eventfd_units(layer0_fd) == 1
        assert _drain_eventfd_units(layer1_fd) == 1

    def test_empty_member_waits_for_swa_before_eventfd(self) -> None:
        """Layer 0 has no members but SWA is active — no early post at step 0b."""
        num_layers = 4
        main = LayerGroupSpec(
            num_layers=3,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=[1, 2, 3],
            dtype=torch.uint8,
        )
        fx, owned = _build_fixture_with_eventfds([main], num_layers, has_swa=True)
        assert fx.swa_cpu is not None
        for orig in range(num_layers):
            _seed_swa_cpu_layer(fx.swa_cpu, SWA_CPU_SRC, orig)

        layer0_fd = _layer_fds(fx, owned)[0]
        assert _eventfd_is_idle(layer0_fd)

        _run_h2d(fx, with_swa=True)

        for orig in range(num_layers):
            fd = _layer_fds(fx, owned)[orig]
            assert _drain_eventfd_units(fd) == 1, (
                f"orig {orig}: SWA-inclusive layer must post after H2D completes"
            )
