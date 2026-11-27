"""Blocks that are multiples of the device's direct-I/O alignment but not of 4 KiB (DeepSeek-V4's
1,671,680-byte blocks) are written with O_DIRECT like they are read: the bytes land where they should
and no page of the file stays cached after a write."""
import os
import shutil
import subprocess

import pytest
import torch

from flexkv.c_ext import SSDIOCTX
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.workers.cpu_ssd import CPUSSDDiskTransferWorker


def _aligned_u8(num_bytes: int, align: int = 4096) -> torch.Tensor:
    raw = torch.empty(num_bytes + align, dtype=torch.uint8)
    off = (-raw.data_ptr()) % align
    return raw[off:off + num_bytes]


def _worker(path, layout, cpu, num_blocks):
    worker = CPUSSDDiskTransferWorker.__new__(CPUSSDDiskTransferWorker)
    worker.ioctx = SSDIOCTX({0: [str(path)]}, 1, 512, 0)
    worker.cpu_layer_ptrs = torch.tensor([cpu.data_ptr()])
    worker.has_multi_group = False
    worker.cpu_layout_type = layout.type
    worker.num_layers = layout.num_layer
    worker.num_kv_heads = 1
    worker.tp_group_size = 1
    worker.kv_dim = layout.kv_dim
    worker.num_blocks_per_file = num_blocks
    worker.round_robin = 1
    worker.cpu_layer_stride_in_bytes = layout.get_layer_stride()
    worker.ssd_layer_stride_in_bytes = layout.get_layer_stride()
    worker.cpu_kv_stride_in_bytes = layout.get_kv_stride()
    worker.ssd_kv_stride_in_bytes = layout.get_kv_stride()
    worker.chunk_size_in_bytes = layout.get_chunk_size()
    worker.block_stride_in_bytes = layout.get_block_stride()
    return worker


def _resident_pages(path) -> int:
    out = subprocess.run(["fincore", "--raw", "--noheadings", "--output", "PAGES", str(path)],
                         check=True, capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else -1


def _drop(path) -> None:
    with open(path, "rb+") as f:
        os.fsync(f.fileno())
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def test_v4_style_blocks_are_written_direct(tmp_path):
    if shutil.which("fincore") is None:
        pytest.skip("fincore (util-linux) not installed")
    # 3 layers x 16 tokens x 1 head x 32 bytes = 1536 B blocks, 512 B layer stride: 512-aligned, not 4 KiB
    num_blocks, layers, tokens, head = 64, 3, 16, 32
    layout = KVCacheLayout(type=KVCacheLayoutType.BLOCKFIRST, num_layer=layers, num_block=num_blocks,
                           tokens_per_block=tokens, num_head=1, head_size=head, kv_dim=1)
    block = layout.get_block_stride()
    assert block % 512 == 0 and block % 4096 != 0
    path = tmp_path / "cache.bin"
    with open(path, "wb") as f:
        f.truncate(num_blocks * block)                  # sparse, like the allocator
    source = torch.randint(0, 256, (num_blocks * block,), dtype=torch.uint8)
    cpu = _aligned_u8(num_blocks * block)
    cpu.copy_(source)
    worker = _worker(path, layout, cpu, num_blocks)
    if worker.ioctx.get_direct_align() != 512:
        pytest.skip("this device does not take 512-aligned direct I/O")
    _drop(path)

    ids = torch.arange(num_blocks)
    worker._issue_ssd_transfer(ids, ids, False, 0, layers, num_threads=2)        # CPU -> SSD
    assert _resident_pages(path) == 0, "512-aligned blocks must be written with O_DIRECT"
    disk = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
    assert torch.equal(disk, source)

    back = _aligned_u8(num_blocks * block)
    back.zero_()
    worker2 = _worker(path, layout, back, num_blocks)   # SSDIOCTX fadvises WILLNEED: open before the drop
    _drop(path)
    worker2._issue_ssd_transfer(ids, ids, True, 0, layers, num_threads=2)        # SSD -> CPU
    assert _resident_pages(path) == 0
    assert torch.equal(back, source)
