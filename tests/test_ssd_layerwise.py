import pytest
import torch

from flexkv.c_ext import SSDIOCTX
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.workers.cpu_ssd import CPUSSDDiskTransferWorker


@pytest.mark.parametrize("tp_group_size", [1, 2, 4])
@pytest.mark.parametrize("num_kv_heads", [1, 8])
@pytest.mark.parametrize("kv_dim", [1, 2])
@pytest.mark.parametrize("layer_span", [(0, 1), (1, 3), (0, 4)])
def test_blockfirst_layer_restore(
    tmp_path, tp_group_size, num_kv_heads, kv_dim, layer_span
):
    """Restore every TP slice without modifying other layers or blocks."""
    num_blocks, num_layers = 6, 4
    storage_ranks = tp_group_size if num_kv_heads > 1 else 1
    layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=num_layers,
        num_block=num_blocks,
        tokens_per_block=4,
        num_head=num_kv_heads,
        head_size=16,
        kv_dim=kv_dim,
        num_kv_heads=num_kv_heads,
    )
    shape = (
        num_blocks, storage_ranks, num_layers, kv_dim,
        4, num_kv_heads // storage_ranks, 16,
    )
    source = torch.arange(
        num_blocks * storage_ranks * num_layers * kv_dim, dtype=torch.uint8
    ).reshape(*shape[:4], 1, 1, 1).expand(shape).contiguous()
    destination = torch.full(shape, 255, dtype=torch.uint8)
    path = tmp_path / "cache.bin"
    path.write_bytes(source.numpy().tobytes())

    worker = CPUSSDDiskTransferWorker.__new__(CPUSSDDiskTransferWorker)
    worker.ioctx = SSDIOCTX({0: [str(path)]}, 1, 512, 0)
    worker.cpu_layer_ptrs = torch.tensor([destination.data_ptr()])
    worker.has_multi_group = False
    worker.cpu_layout_type = layout.type
    worker.num_layers = num_layers
    worker.num_kv_heads = num_kv_heads
    worker.tp_group_size = tp_group_size
    worker.kv_dim = kv_dim
    worker.num_blocks_per_file = num_blocks
    worker.round_robin = 1
    worker.cpu_layer_stride_in_bytes = layout.get_layer_stride()
    worker.ssd_layer_stride_in_bytes = layout.get_layer_stride()
    worker.cpu_kv_stride_in_bytes = layout.get_kv_stride()
    worker.ssd_kv_stride_in_bytes = layout.get_kv_stride()
    worker.chunk_size_in_bytes = layout.get_chunk_size()
    worker.block_stride_in_bytes = layout.get_block_stride()

    source_ids = torch.tensor([4, 1])
    destination_ids = torch.tensor([2, 5])
    start, end = layer_span
    expected = destination.clone()
    expected[destination_ids, :, start:end] = source[source_ids, :, start:end]
    worker._issue_ssd_transfer(
        destination_ids, source_ids, True, start, end, num_threads=1
    )

    assert torch.equal(destination, expected)
