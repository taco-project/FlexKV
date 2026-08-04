"""Round-trip the packed 4D KV layout through the D2H/H2D transfer kernels.

vLLM >= PR #44455 hands FlexKV a 4D KV cache — (num_blocks, num_kv_heads,
block_size, 2 * head_size) — with K and V interleaved in the last dim instead
of split across a leading kv dim.  ``KVCacheLayout.packed_kv`` folds that pair
into ``head_size`` and reports a single KV region, so the copy kernels must see
``single_kv_region=True`` while TP head-sharding still behaves like MHA.

These tests assert the bytes survive a GPU->CPU->GPU round trip and that the
CPU mirror lands where the layout's own strides say it should.
"""

import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

c_ext = pytest.importorskip("flexkv.c_ext")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU")

DTYPE = torch.bfloat16
NUM_LAYERS = 4
NUM_BLOCKS = 8
TOKENS_PER_BLOCK = 16
NUM_HEADS = 4
HEAD_SIZE = 64  # logical; the packed tensor carries 2 * HEAD_SIZE


def _make_layouts(cpu_layout_name):
    """GPU LAYERFIRST + CPU layout for a packed cache, as the adapter builds them."""
    packed_head_size = 2 * HEAD_SIZE
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=NUM_LAYERS, num_block=NUM_BLOCKS,
        tokens_per_block=TOKENS_PER_BLOCK, num_head=NUM_HEADS,
        head_size=packed_head_size, is_mla=False, packed_kv=True)
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType[cpu_layout_name],
        num_layer=NUM_LAYERS, num_block=NUM_BLOCKS,
        tokens_per_block=TOKENS_PER_BLOCK, num_head=NUM_HEADS,
        head_size=packed_head_size, is_mla=False, packed_kv=True)
    return gpu_layout, cpu_layout


def _make_gpu_blocks(device=0):
    """One 4D NHD tensor per layer, matching vLLM's allocation then permute.

    vLLM allocates (blocks, tokens, heads, content) and permutes to the logical
    (blocks, heads, tokens, content); ``.contiguous()`` here keeps the same
    token-major memory order FlexKV's shape-implied strides assume.
    """
    blocks = []
    for i in range(NUM_LAYERS):
        t = torch.arange(
            NUM_BLOCKS * NUM_HEADS * TOKENS_PER_BLOCK * 2 * HEAD_SIZE,
            dtype=torch.float32, device=f"cuda:{device}")
        t = (t + i * 1000).to(DTYPE).view(
            NUM_BLOCKS, TOKENS_PER_BLOCK, NUM_HEADS, 2 * HEAD_SIZE)
        blocks.append(t.permute(0, 2, 1, 3))
    return blocks


def _make_tp_gpu_blocks(num_gpus):
    heads_per_rank = NUM_HEADS // num_gpus
    blocks_per_rank = []
    for rank in range(num_gpus):
        rank_blocks = []
        for layer in range(NUM_LAYERS):
            numel = (NUM_BLOCKS * TOKENS_PER_BLOCK * heads_per_rank
                     * 2 * HEAD_SIZE)
            values = torch.arange(
                numel, dtype=torch.int32, device=f"cuda:{rank}")
            values = (values % 97 + rank * 128 + layer * 16).to(DTYPE)
            tensor = values.view(
                NUM_BLOCKS, TOKENS_PER_BLOCK, heads_per_rank, 2 * HEAD_SIZE)
            rank_blocks.append(tensor.permute(0, 2, 1, 3))
        blocks_per_rank.append(rank_blocks)
    return blocks_per_rank


def test_packed_layout_reports_one_kv_region():
    gpu_layout, _ = _make_layouts("LAYERFIRST")
    assert gpu_layout.kv_dim == 1
    assert gpu_layout.single_kv_region is True
    # is_mla must stay False: packed MHA still splits heads across TP ranks.
    assert gpu_layout.is_mla is False


def test_transfer_binding_separates_ce_mla_from_single_kv_region():
    signature = c_ext.transfer_kv_blocks.__doc__ or ""
    assert "single_kv_region: bool" in signature
    assert "ce_is_mla: bool" in signature

    empty_ids = torch.empty(0, dtype=torch.int64)
    with pytest.raises(
        RuntimeError,
        match="ce_is_mla=True requires single_kv_region=True",
    ):
        c_ext.transfer_kv_blocks(
            gpu_block_id_tensor=empty_ids,
            gpu_tensor_ptrs_tensor=empty_ids,
            gpu_kv_stride_in_bytes=0,
            gpu_block_stride_in_bytes=0,
            gpu_layer_stride_in_bytes=0,
            cpu_block_id_tensor=empty_ids,
            cpu_tensor=torch.empty(0),
            cpu_kv_stride_in_bytes=0,
            cpu_layer_stride_in_bytes=0,
            cpu_block_stride_in_bytes=0,
            chunk_size_in_bytes=0,
            start_layer_id=0,
            num_layers=0,
            single_kv_region=False,
            ce_is_mla=True,
        )


def test_packed_layout_strides_match_the_real_tensor():
    """FlexKV never reads tensor.stride(), so shape-implied strides must agree."""
    gpu_layout, _ = _make_layouts("LAYERFIRST")
    tensor = _make_gpu_blocks()[0]
    assert tensor.stride(0) == gpu_layout.get_block_stride()      # block
    assert tensor.stride(2) == NUM_HEADS * 2 * HEAD_SIZE          # token
    assert tensor.stride(1) == 2 * HEAD_SIZE                      # head
    assert tensor.numel() == gpu_layout.get_layer_stride()        # layer


@pytest.mark.parametrize("cpu_layout_name", ["LAYERFIRST", "BLOCKFIRST"])
@pytest.mark.parametrize("use_ce_transfer", [False, True], ids=["kernel", "ce"])
def test_packed_kv_d2h_h2d_roundtrip(cpu_layout_name, use_ce_transfer):
    gpu_layout, cpu_layout = _make_layouts(cpu_layout_name)
    gpu_blocks = _make_gpu_blocks()
    expected = [b.clone() for b in gpu_blocks]

    cpu_tensor = torch.zeros(
        tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)
    gpu_ptrs = torch.tensor(
        [b.data_ptr() for b in gpu_blocks], dtype=torch.int64).pin_memory()

    itemsize = DTYPE.itemsize
    block_ids = torch.arange(NUM_BLOCKS, dtype=torch.int64).pin_memory()

    kwargs = dict(
        gpu_block_id_tensor=block_ids,
        gpu_tensor_ptrs_tensor=gpu_ptrs,
        gpu_kv_stride_in_bytes=gpu_layout.get_kv_stride() * itemsize,
        gpu_block_stride_in_bytes=gpu_layout.get_block_stride() * itemsize,
        gpu_layer_stride_in_bytes=gpu_layout.get_layer_stride() * itemsize,
        cpu_block_id_tensor=block_ids,
        cpu_tensor=cpu_tensor,
        cpu_kv_stride_in_bytes=cpu_layout.get_kv_stride() * itemsize,
        cpu_layer_stride_in_bytes=cpu_layout.get_layer_stride() * itemsize,
        cpu_block_stride_in_bytes=cpu_layout.get_block_stride() * itemsize,
        chunk_size_in_bytes=gpu_layout.get_chunk_size() * itemsize,
        start_layer_id=0,
        num_layers=NUM_LAYERS,
        use_ce_transfer=use_ce_transfer,
        ce_path_opt=use_ce_transfer,
        single_kv_region=gpu_layout.single_kv_region,
        ce_is_mla=gpu_layout.is_mla,
        is_blockfirst=(cpu_layout.type == KVCacheLayoutType.BLOCKFIRST),
    )

    c_ext.transfer_kv_blocks(is_host_to_device=False, **kwargs)
    torch.cuda.synchronize()

    assert cpu_tensor.abs().sum().item() > 0, "D2H copied nothing"

    for b in gpu_blocks:
        b.zero_()
    torch.cuda.synchronize()

    c_ext.transfer_kv_blocks(is_host_to_device=True, **kwargs)
    torch.cuda.synchronize()

    for i, (got, want) in enumerate(zip(gpu_blocks, expected, strict=True)):
        assert torch.equal(got, want), f"layer {i} mismatch after round trip"


def test_packed_cpu_mirror_matches_layout_strides():
    """The CPU mirror of one block must be byte-identical to the GPU block.

    Catches a region-count mismatch: were the kernels to treat the packed cache
    as two KV regions, they would write only half of each block's content.
    """
    gpu_layout, cpu_layout = _make_layouts("LAYERFIRST")
    gpu_blocks = _make_gpu_blocks()
    cpu_tensor = torch.zeros(
        tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)
    gpu_ptrs = torch.tensor(
        [b.data_ptr() for b in gpu_blocks], dtype=torch.int64).pin_memory()
    itemsize = DTYPE.itemsize
    block_ids = torch.arange(NUM_BLOCKS, dtype=torch.int64).pin_memory()

    c_ext.transfer_kv_blocks(
        gpu_block_id_tensor=block_ids,
        gpu_tensor_ptrs_tensor=gpu_ptrs,
        gpu_kv_stride_in_bytes=gpu_layout.get_kv_stride() * itemsize,
        gpu_block_stride_in_bytes=gpu_layout.get_block_stride() * itemsize,
        gpu_layer_stride_in_bytes=gpu_layout.get_layer_stride() * itemsize,
        cpu_block_id_tensor=block_ids,
        cpu_tensor=cpu_tensor,
        cpu_kv_stride_in_bytes=cpu_layout.get_kv_stride() * itemsize,
        cpu_layer_stride_in_bytes=cpu_layout.get_layer_stride() * itemsize,
        cpu_block_stride_in_bytes=cpu_layout.get_block_stride() * itemsize,
        chunk_size_in_bytes=gpu_layout.get_chunk_size() * itemsize,
        start_layer_id=0, num_layers=NUM_LAYERS,
        is_host_to_device=False,
        single_kv_region=gpu_layout.single_kv_region,
        ce_is_mla=gpu_layout.is_mla,
    )
    torch.cuda.synchronize()

    # CPU LAYERFIRST shape is [layer, kv=1, block, token, head, content];
    # the GPU tensor is [block, head, token, content].
    for layer in range(NUM_LAYERS):
        mirror = cpu_tensor[layer, 0]                      # [block, token, head, content]
        want = gpu_blocks[layer].permute(0, 2, 1, 3).cpu()  # -> same order
        assert torch.equal(mirror, want), f"layer {layer} CPU mirror mismatch"


def test_packed_kv_ce_partial_block_roundtrip():
    """Packed MHA must stay non-MLA on the partial BLOCKFIRST CE path."""
    gpu_layout, cpu_layout = _make_layouts("BLOCKFIRST")
    gpu_blocks = _make_gpu_blocks()
    expected_layer = gpu_blocks[1].clone()
    cpu_tensor = torch.zeros(
        tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)
    gpu_ptrs = torch.tensor(
        [block.data_ptr() for block in gpu_blocks],
        dtype=torch.int64,
    ).pin_memory()
    itemsize = DTYPE.itemsize
    block_ids = torch.arange(NUM_BLOCKS, dtype=torch.int64).pin_memory()

    kwargs = dict(
        gpu_block_id_tensor=block_ids,
        gpu_tensor_ptrs_tensor=gpu_ptrs,
        gpu_kv_stride_in_bytes=gpu_layout.get_kv_stride() * itemsize,
        gpu_block_stride_in_bytes=gpu_layout.get_block_stride() * itemsize,
        gpu_layer_stride_in_bytes=gpu_layout.get_layer_stride() * itemsize,
        cpu_block_id_tensor=block_ids,
        cpu_tensor=cpu_tensor,
        cpu_kv_stride_in_bytes=cpu_layout.get_kv_stride() * itemsize,
        cpu_layer_stride_in_bytes=cpu_layout.get_layer_stride() * itemsize,
        cpu_block_stride_in_bytes=cpu_layout.get_block_stride() * itemsize,
        chunk_size_in_bytes=gpu_layout.get_chunk_size() * itemsize,
        start_layer_id=1,
        num_layers=1,
        use_ce_transfer=True,
        ce_path_opt=True,
        single_kv_region=True,
        ce_is_mla=False,
        is_blockfirst=True,
    )

    c_ext.transfer_kv_blocks(is_host_to_device=False, **kwargs)
    torch.cuda.synchronize()

    for block in gpu_blocks:
        block.zero_()
    torch.cuda.synchronize()

    c_ext.transfer_kv_blocks(is_host_to_device=True, **kwargs)
    torch.cuda.synchronize()

    assert torch.equal(gpu_blocks[1], expected_layer)
    for layer in (0, 2, 3):
        assert torch.count_nonzero(gpu_blocks[layer]).item() == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
@pytest.mark.parametrize("use_ce_transfer", [False, True], ids=["kernel", "ce"])
def test_packed_kv_layerwise_h2d_preserves_tp_head_shards(use_ce_transfer):
    """Layerwise H2D must use one KV region without enabling MLA TP behavior."""
    num_gpus = 2
    heads_per_rank = NUM_HEADS // num_gpus
    packed_head_size = 2 * HEAD_SIZE
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=NUM_LAYERS,
        num_block=NUM_BLOCKS,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_head=heads_per_rank,
        head_size=packed_head_size,
        is_mla=False,
        packed_kv=True,
    )
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=NUM_LAYERS,
        num_block=NUM_BLOCKS,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_head=NUM_HEADS,
        head_size=packed_head_size,
        is_mla=False,
        packed_kv=True,
    )
    cpu_layout_tp = cpu_layout.div_head(num_gpus)
    gpu_blocks = _make_tp_gpu_blocks(num_gpus)
    expected = [[tensor.clone() for tensor in rank] for rank in gpu_blocks]
    cpu_tensor = torch.zeros(
        tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)
    itemsize = DTYPE.itemsize
    block_ids = torch.arange(NUM_BLOCKS, dtype=torch.int64).pin_memory()

    tp_group = c_ext.TPTransferThreadGroup(
        num_gpus=num_gpus,
        gpu_block_ptrs_flat=[
            tensor.data_ptr() for rank in gpu_blocks for tensor in rank
        ],
        num_tensors_per_gpu=NUM_LAYERS,
        cpu_blocks_ptr=cpu_tensor.data_ptr(),
        num_layers=NUM_LAYERS,
        gpu_kv_strides_in_bytes=[
            gpu_layout.get_kv_stride() * itemsize
        ] * num_gpus,
        gpu_block_strides_in_bytes=[
            gpu_layout.get_block_stride() * itemsize
        ] * num_gpus,
        gpu_layer_strides_in_bytes=[
            gpu_layout.get_layer_stride() * itemsize
        ] * num_gpus,
        gpu_chunk_sizes_in_bytes=[
            gpu_layout.get_chunk_size() * itemsize
        ] * num_gpus,
        gpu_device_ids=list(range(num_gpus)),
        enable_nvcomp=False,
        is_blockfirst=True,
        is_mla=False,
    )
    tp_group.tp_group_transfer(
        gpu_block_id_tensor=block_ids,
        cpu_block_id_tensor=block_ids,
        cpu_kv_stride_in_bytes=cpu_layout_tp.get_kv_stride() * itemsize,
        cpu_layer_stride_in_bytes=cpu_layout_tp.get_layer_stride() * itemsize,
        cpu_block_stride_in_bytes=cpu_layout.get_block_stride() * itemsize,
        cpu_tp_stride_in_bytes=(
            cpu_layout.get_block_stride() * itemsize // num_gpus
        ),
        transfer_num_cta=4,
        is_host_to_device=False,
        use_ce_transfer=use_ce_transfer,
        layer_id=0,
        layer_granularity=NUM_LAYERS,
        is_mla=False,
        packed_kv=True,
    )
    for device in range(num_gpus):
        torch.cuda.synchronize(device)
    del tp_group

    for rank in gpu_blocks:
        for tensor in rank:
            tensor.zero_()
    for device in range(num_gpus):
        torch.cuda.synchronize(device)

    def strides_tensor(getter):
        return torch.tensor(
            [getter() * itemsize] * num_gpus, dtype=torch.int64)

    empty_tensor = torch.empty(0)
    empty_ids = torch.empty(0, dtype=torch.int64)
    layerwise_group = c_ext.LayerwiseTransferGroup(
        num_gpus=num_gpus,
        gpu_blocks=gpu_blocks,
        cpu_blocks=cpu_tensor,
        ssd_files={},
        num_layers=NUM_LAYERS,
        gpu_kv_strides_tensor=strides_tensor(gpu_layout.get_kv_stride),
        gpu_block_strides_tensor=strides_tensor(gpu_layout.get_block_stride),
        gpu_layer_strides_tensor=strides_tensor(gpu_layout.get_layer_stride),
        gpu_chunk_sizes_tensor=strides_tensor(gpu_layout.get_chunk_size),
        iouring_entries=0,
        iouring_flags=0,
        layer_eventfds_tensor=torch.empty(0, dtype=torch.int32),
        tp_size=num_gpus,
        has_swa=False,
        swa_gpu_blocks=[],
        swa_cpu_blocks=empty_tensor,
        swa_ssd_files={},
        swa_gpu_kv_strides_tensor=empty_tensor,
        swa_gpu_block_strides_tensor=empty_tensor,
        swa_gpu_layer_strides_tensor=empty_tensor,
        swa_gpu_chunk_sizes_tensor=empty_tensor,
        is_blockfirst=True,
        is_mla=False,
    )
    layerwise_group.layerwise_transfer(
        ssd_block_ids=empty_ids,
        cpu_block_ids_d2h=empty_ids,
        ssd_layer_stride_in_bytes=0,
        ssd_kv_stride_in_bytes=0,
        num_blocks_per_file=0,
        round_robin=0,
        num_threads_per_device=0,
        gpu_block_id_tensor=block_ids,
        cpu_block_id_tensor=block_ids,
        cpu_kv_stride_in_bytes=cpu_layout.get_kv_stride() * itemsize,
        cpu_layer_stride_in_bytes=cpu_layout.get_layer_stride() * itemsize,
        cpu_block_stride_in_bytes=cpu_layout.get_block_stride() * itemsize,
        cpu_chunk_size_in_bytes=cpu_layout.get_chunk_size() * itemsize,
        h2d_cpu_kv_stride_in_bytes=(
            cpu_layout_tp.get_kv_stride() * itemsize
        ),
        h2d_cpu_layer_stride_in_bytes=(
            cpu_layout_tp.get_layer_stride() * itemsize
        ),
        cpu_tp_stride_in_bytes=(
            cpu_layout.get_block_stride() * itemsize // num_gpus
        ),
        transfer_cta_num=4,
        use_ce_transfer=use_ce_transfer,
        num_layers=NUM_LAYERS,
        layer_granularity=1,
        is_mla=False,
        packed_kv=True,
    )
    for device in range(num_gpus):
        torch.cuda.synchronize(device)
    del layerwise_group

    for rank, (got_layers, expected_layers) in enumerate(
            zip(gpu_blocks, expected, strict=True)):
        for layer, (got, want) in enumerate(
                zip(got_layers, expected_layers, strict=True)):
            assert torch.equal(got, want), (
                f"rank {rank} layer {layer} mismatch after layerwise H2D"
            )
