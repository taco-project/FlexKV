import os
import shutil
import tempfile

import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.compression.ans import ans_utils
from nvcomp_common_utils import (
    make_gpu_cache,
    resolve_case,
)

try:
    from flexkv.c_ext import (
        ANSTransferContext,
        SSDIOCTX,
        TPTransferThreadGroup,
        transfer_kv_blocks_ans_comp,
        transfer_kv_blocks_ssd_packed,
    )
    NVCOMP_AVAILABLE = True
except ImportError:
    NVCOMP_AVAILABLE = False


DEVICE = "cuda:0"


def setup_ssd_files(num_blocks, file_size_bytes, tmpdir=None):
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="flexkv_ssd_test_")
    dev_dir = os.path.join(tmpdir, "dev0")
    os.makedirs(dev_dir, exist_ok=True)
    fpath = os.path.join(dev_dir, "ssd_cache_0_0.bin")
    with open(fpath, "wb+", buffering=0) as f:
        os.truncate(f.fileno(), file_size_bytes)
        os.fsync(f.fileno())
    return {0: [fpath]}, tmpdir, num_blocks


def ssd_file_size(layout, dtype):
    return layout.get_total_elements() * dtype.itemsize


def cpu_payload_offset(block_id, rank, layer_id, kv_id, *, block_stride,
                       layer_stride, kv_stride, rank_stride=0):
    return (
        int(block_id) * block_stride
        + int(rank) * rank_stride
        + layer_id * layer_stride
        + kv_id * kv_stride
    )


def table_comp_bytes(table_i64, table_ranked, rank, block_id, layer_id, kv_id):
    if table_ranked:
        return int(table_i64[rank, block_id, layer_id, kv_id].item())
    return int(table_i64[block_id, layer_id, kv_id].item())


def snapshot_compressed_slots(
    cpu_cache, size_table, block_ids, *, ranks, table_ranked, num_layers,
    kv_dim, block_stride, layer_stride, kv_stride, rank_stride=0,
):
    cpu_bytes = cpu_cache.view(torch.uint8).reshape(-1)
    table_i64 = size_table.to(torch.int64)
    snapshot = {}
    for block_idx, block_id in enumerate(block_ids.tolist()):
        for rank in ranks:
            for layer_id in range(num_layers):
                for kv_id in range(kv_dim):
                    comp_bytes = table_comp_bytes(
                        table_i64, table_ranked, rank, block_id, layer_id, kv_id)
                    offset = cpu_payload_offset(
                        block_id, rank, layer_id, kv_id,
                        block_stride=block_stride,
                        layer_stride=layer_stride,
                        kv_stride=kv_stride,
                        rank_stride=rank_stride)
                    snapshot[(block_idx, rank, layer_id, kv_id)] = cpu_bytes[
                        offset:offset + comp_bytes].clone()
    return snapshot


def assert_compressed_slots_equal(
    cpu_cache, size_table, block_ids, expected, *, ranks, table_ranked,
    num_layers, kv_dim, block_stride, layer_stride, kv_stride, rank_stride=0,
):
    cpu_bytes = cpu_cache.view(torch.uint8).reshape(-1)
    table_i64 = size_table.to(torch.int64)
    for block_idx, block_id in enumerate(block_ids.tolist()):
        for rank in ranks:
            for layer_id in range(num_layers):
                for kv_id in range(kv_dim):
                    comp_bytes = table_comp_bytes(
                        table_i64, table_ranked, rank, block_id, layer_id, kv_id)
                    expected_payload = expected[(block_idx, rank, layer_id, kv_id)]
                    assert comp_bytes == expected_payload.numel(), (
                        f"compressed size mismatch: block={block_id}, "
                        f"rank={rank}, layer={layer_id}, kv={kv_id}")
                    offset = cpu_payload_offset(
                        block_id, rank, layer_id, kv_id,
                        block_stride=block_stride,
                        layer_stride=layer_stride,
                        kv_stride=kv_stride,
                        rank_stride=rank_stride)
                    actual = cpu_bytes[offset:offset + comp_bytes]
                    assert torch.equal(actual, expected_payload), (
                        f"compressed payload mismatch: block={block_id}, "
                        f"rank={rank}, layer={layer_id}, kv={kv_id}")


@pytest.mark.parametrize("cpu_layout_name", [
    pytest.param("BLOCKFIRST", id="bfirst"),
    pytest.param("LAYERFIRST", id="lfirst"),
])
@pytest.mark.parametrize("dtype", [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float8_e4m3fn, id="fp8"),
])
@pytest.mark.parametrize("tp_size", [pytest.param(1, id="tp_1")])
@pytest.mark.parametrize("is_mla", [
    pytest.param(False, id="mha"),
    pytest.param(True, id="mla"),
])
@pytest.mark.parametrize("chunk_size_per_device", [
    pytest.param(8 * 1024, id="cpd_8KB"),
    pytest.param(16 * 1024, id="cpd_16KB"),
    pytest.param(32 * 1024, id="cpd_32KB"),
    pytest.param(18 * 1024, id="cpd_18KB"),  # MLA
])
def test_ssd_roundtrip_non_tp(
    cpu_layout_name, dtype, tp_size, is_mla, chunk_size_per_device,
    num_layers=4, transfer_blocks=2,
):
    if not NVCOMP_AVAILABLE:
        pytest.skip("nvcomp not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if is_mla != (chunk_size_per_device == 18 * 1024):
        pytest.skip("18KB is MLA-only; 8/16/32KB are MHA-only")

    case = resolve_case(chunk_size_per_device, tp_size, is_mla, dtype)
    num_heads, head_size = case["num_head"], case["head_dim"]
    tokens_per_block, kv_dim = case["tokens_per_block"], case["kv_dim"]
    elem = dtype.itemsize
    layout_type = KVCacheLayoutType[cpu_layout_name.upper()]

    num_cpu_blocks = transfer_blocks * 2
    num_ssd_blocks = transfer_blocks

    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers,
        num_block=transfer_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_heads,
        head_size=head_size,
        is_mla=is_mla,
    )
    cpu_layout = KVCacheLayout(
        type=layout_type,
        num_layer=num_layers,
        num_block=num_cpu_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_heads,
        head_size=head_size,
        is_mla=is_mla,
    )
    ssd_layout = KVCacheLayout(
        type=layout_type,
        num_layer=num_layers,
        num_block=num_ssd_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_heads,
        head_size=head_size,
        is_mla=is_mla,
    )

    shape_per_layer = (
        kv_dim, transfer_blocks, tokens_per_block, num_heads, head_size)
    gpu_cache = make_gpu_cache(
        (num_layers,) + shape_per_layer, dtype, DEVICE)
    gpu_blocks = [gpu_cache[layer] for layer in range(num_layers)]
    gpu_ptrs = torch.tensor(
        [block.data_ptr() for block in gpu_blocks],
        dtype=torch.int64,
    ).pin_memory()

    cpu_cache = torch.zeros(tuple(cpu_layout.kv_shape), dtype=dtype).pin_memory()
    cpu_size_table = torch.zeros(
        (num_cpu_blocks, num_layers, kv_dim), dtype=torch.uint32).pin_memory()
    ssd_size_table = torch.zeros(
        (num_ssd_blocks, num_layers, kv_dim), dtype=torch.uint32).pin_memory()

    ssd_files, tmpdir, num_blocks_per_file = setup_ssd_files(
        num_ssd_blocks, ssd_file_size(ssd_layout, dtype))

    gpu_block_ids = torch.arange(transfer_blocks, dtype=torch.int64).pin_memory()
    write_cpu_ids = torch.arange(transfer_blocks, dtype=torch.int64).pin_memory()
    read_cpu_ids = torch.arange(
        transfer_blocks, transfer_blocks * 2, dtype=torch.int64).pin_memory()
    ssd_block_ids = torch.arange(transfer_blocks, dtype=torch.int64).pin_memory()

    chunk_bytes = gpu_layout.get_chunk_size() * elem
    assert chunk_bytes == case["chunk_bytes"]

    data_type = ans_utils.data_type(dtype)
    batch_size, _ = ans_utils.get_nvcomp_batch_size(
        chunk_bytes, data_type=data_type)
    ctx = ANSTransferContext(batch_size, chunk_bytes, data_type)
    stream = torch.cuda.Stream(device=0)

    try:
        ioctx = SSDIOCTX(ssd_files, len(ssd_files), 0, 0)

        with torch.cuda.stream(stream):
            d2h_bytes = transfer_kv_blocks_ans_comp(
                ctx,
                gpu_block_ids,
                gpu_ptrs,
                gpu_layout.get_kv_stride() * elem,
                gpu_layout.get_block_stride() * elem,
                gpu_layout.get_layer_stride() * elem,
                write_cpu_ids,
                cpu_cache,
                cpu_layout.get_kv_stride() * elem,
                cpu_layout.get_layer_stride() * elem,
                cpu_layout.get_block_stride() * elem,
                chunk_bytes,
                0,
                num_layers,
                is_mla,
                0,
                cpu_size_table.data_ptr(),
                cpu_size_table.stride(0),
                cpu_size_table.stride(1),
            )
        stream.synchronize()
        assert d2h_bytes == int(cpu_size_table.to(torch.int64)[write_cpu_ids].sum().item())
        assert (cpu_size_table.to(torch.int64)[write_cpu_ids] > 0).all()
        assert (cpu_size_table.to(torch.int64)[read_cpu_ids] == 0).all()
        expected_payloads = snapshot_compressed_slots(
            cpu_cache, cpu_size_table, write_cpu_ids,
            ranks=[0],
            table_ranked=False,
            num_layers=num_layers,
            kv_dim=kv_dim,
            block_stride=cpu_layout.get_block_stride() * elem,
            layer_stride=cpu_layout.get_layer_stride() * elem,
            kv_stride=cpu_layout.get_kv_stride() * elem)

        h2disk_bytes = transfer_kv_blocks_ssd_packed(
            ioctx=ioctx,
            cpu_layer_id_list=torch.arange(num_layers, dtype=torch.int32),
            cpu_tensor_ptr=cpu_cache.data_ptr(),
            ssd_block_ids=ssd_block_ids,
            cpu_block_ids=write_cpu_ids,
            cpu_layer_stride_in_bytes=cpu_layout.get_layer_stride() * elem,
            cpu_kv_stride_in_bytes=cpu_layout.get_kv_stride() * elem,
            chunk_size_in_bytes=chunk_bytes,
            block_stride_in_bytes=cpu_layout.get_block_stride() * elem,
            is_read=False,
            num_blocks_per_file=num_blocks_per_file,
            layout_type=cpu_layout.type.value,
            total_layers=num_layers,
            is_mla=is_mla,
            cpu_size_table_ptr=cpu_size_table.data_ptr(),
            cpu_size_table_block_stride=cpu_size_table.stride(0),
            cpu_size_table_layer_stride=cpu_size_table.stride(1),
            ssd_size_table_ptr=ssd_size_table.data_ptr(),
            ssd_size_table_block_stride=ssd_size_table.stride(0),
            ssd_size_table_layer_stride=ssd_size_table.stride(1),
        )
        assert h2disk_bytes == int(ssd_size_table.to(torch.int64)[ssd_block_ids].sum().item())
        assert torch.equal(
            cpu_size_table.to(torch.int64)[write_cpu_ids],
            ssd_size_table.to(torch.int64)[ssd_block_ids],
        )

        cpu_cache.zero_()
        cpu_size_table.zero_()
        disk2h_bytes = transfer_kv_blocks_ssd_packed(
            ioctx=ioctx,
            cpu_layer_id_list=torch.arange(num_layers, dtype=torch.int32),
            cpu_tensor_ptr=cpu_cache.data_ptr(),
            ssd_block_ids=ssd_block_ids,
            cpu_block_ids=read_cpu_ids,
            cpu_layer_stride_in_bytes=cpu_layout.get_layer_stride() * elem,
            cpu_kv_stride_in_bytes=cpu_layout.get_kv_stride() * elem,
            chunk_size_in_bytes=chunk_bytes,
            block_stride_in_bytes=cpu_layout.get_block_stride() * elem,
            is_read=True,
            num_blocks_per_file=num_blocks_per_file,
            layout_type=cpu_layout.type.value,
            total_layers=num_layers,
            is_mla=is_mla,
            cpu_size_table_ptr=cpu_size_table.data_ptr(),
            cpu_size_table_block_stride=cpu_size_table.stride(0),
            cpu_size_table_layer_stride=cpu_size_table.stride(1),
            ssd_size_table_ptr=ssd_size_table.data_ptr(),
            ssd_size_table_block_stride=ssd_size_table.stride(0),
            ssd_size_table_layer_stride=ssd_size_table.stride(1),
        )
        assert disk2h_bytes == h2disk_bytes
        assert torch.equal(
            cpu_size_table.to(torch.int64)[read_cpu_ids],
            ssd_size_table.to(torch.int64)[ssd_block_ids],
        )
        assert (cpu_size_table.to(torch.int64)[write_cpu_ids] == 0).all()
        assert_compressed_slots_equal(
            cpu_cache, cpu_size_table, read_cpu_ids, expected_payloads,
            ranks=[0],
            table_ranked=False,
            num_layers=num_layers,
            kv_dim=kv_dim,
            block_stride=cpu_layout.get_block_stride() * elem,
            layer_stride=cpu_layout.get_layer_stride() * elem,
            kv_stride=cpu_layout.get_kv_stride() * elem)
    finally:
        ctx.destroy()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize("cpu_layout_name", [
    pytest.param("BLOCKFIRST", id="bfirst"),
    pytest.param("LAYERFIRST", id="lfirst"),
])
@pytest.mark.parametrize("dtype", [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float8_e4m3fn, id="fp8"),
])
@pytest.mark.parametrize("tp_size", [
    pytest.param(2, id="tp_2"),
    pytest.param(4, id="tp_4"),
    pytest.param(8, id="tp_8"),
])
@pytest.mark.parametrize("is_mla", [
    pytest.param(False, id="mha"),
    pytest.param(True, id="mla"),
])
@pytest.mark.parametrize("chunk_size_per_device", [
    pytest.param(8 * 1024, id="cpd_8KB"),
    pytest.param(16 * 1024, id="cpd_16KB"),
    pytest.param(32 * 1024, id="cpd_32KB"),
    pytest.param(18 * 1024, id="cpd_18KB"),  # MLA
])
def test_ssd_roundtrip_tp(
    cpu_layout_name, dtype, tp_size, is_mla, chunk_size_per_device,
    num_layers=4, transfer_blocks=2,
):
    if not NVCOMP_AVAILABLE:
        pytest.skip("nvcomp not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"tp_size={tp_size} needs {tp_size} GPUs")
    if is_mla != (chunk_size_per_device == 18 * 1024):
        pytest.skip("18KB is MLA-only; 8/16/32KB are MHA-only")

    case = resolve_case(chunk_size_per_device, tp_size, is_mla, dtype)
    num_heads, head_size = case["num_head"], case["head_dim"]
    tokens_per_block, kv_dim = case["tokens_per_block"], case["kv_dim"]
    elem = dtype.itemsize
    layout_type = KVCacheLayoutType[cpu_layout_name.upper()]

    num_cpu_blocks = transfer_blocks * 2
    num_ssd_blocks = transfer_blocks
    heads_per_rank = num_heads if is_mla else num_heads // tp_size

    gpu_full_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers,
        num_block=transfer_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_heads,
        head_size=head_size,
        is_mla=is_mla,
    )
    gpu_rank_layout = gpu_full_layout if is_mla else gpu_full_layout.div_head(tp_size)
    cpu_layout = KVCacheLayout(
        type=layout_type,
        num_layer=num_layers,
        num_block=num_cpu_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_heads,
        head_size=head_size,
        is_mla=is_mla,
    )
    ssd_layout = KVCacheLayout(
        type=layout_type,
        num_layer=num_layers,
        num_block=num_ssd_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_heads,
        head_size=head_size,
        is_mla=is_mla,
    )

    shape_per_layer = (
        kv_dim, transfer_blocks, tokens_per_block, heads_per_rank, head_size)
    all_gpu = []
    if is_mla:
        base_cache = make_gpu_cache(
            (num_layers,) + shape_per_layer, dtype, DEVICE).cpu()
        for rank in range(tp_size):
            cache = base_cache.to(f"cuda:{rank}")
            all_gpu.append([cache[layer] for layer in range(num_layers)])
    else:
        for rank in range(tp_size):
            cache = make_gpu_cache(
                (num_layers,) + shape_per_layer, dtype, f"cuda:{rank}")
            all_gpu.append([cache[layer] for layer in range(num_layers)])

    cpu_cache = torch.zeros(tuple(cpu_layout.kv_shape), dtype=dtype).pin_memory()
    if is_mla:
        cpu_size_table = torch.zeros(
            (num_cpu_blocks, num_layers, kv_dim), dtype=torch.uint32).pin_memory()
        ssd_size_table = torch.zeros(
            (num_ssd_blocks, num_layers, kv_dim), dtype=torch.uint32).pin_memory()
        cpu_size_table_tp = None
        ssd_size_table_tp = None
    else:
        cpu_size_table = None
        ssd_size_table = None
        cpu_size_table_tp = torch.zeros(
            (tp_size, num_cpu_blocks, num_layers, kv_dim),
            dtype=torch.uint32).pin_memory()
        ssd_size_table_tp = torch.zeros(
            (tp_size, num_ssd_blocks, num_layers, kv_dim),
            dtype=torch.uint32).pin_memory()

    ssd_files, tmpdir, num_blocks_per_file = setup_ssd_files(
        num_ssd_blocks, ssd_file_size(ssd_layout, dtype))

    block_ids = torch.arange(transfer_blocks, dtype=torch.int64).pin_memory()
    write_cpu_ids = torch.arange(transfer_blocks, dtype=torch.int64).pin_memory()
    read_cpu_ids = torch.arange(
        transfer_blocks, transfer_blocks * 2, dtype=torch.int64).pin_memory()
    ssd_block_ids = torch.arange(transfer_blocks, dtype=torch.int64).pin_memory()

    per_rank_chunk = gpu_rank_layout.get_chunk_size() * elem
    assert per_rank_chunk == case["chunk_bytes"]
    data_type = ans_utils.data_type(dtype)
    batch_size, _ = ans_utils.get_nvcomp_batch_size(
        per_rank_chunk, data_type=data_type)
    gpu_block_ptrs_flat = [
        all_gpu[rank][layer].data_ptr()
        for rank in range(tp_size)
        for layer in range(num_layers)
    ]
    gpu_device_ids = [all_gpu[rank][0].device.index for rank in range(tp_size)]
    tg = TPTransferThreadGroup(
        tp_size,
        gpu_block_ptrs_flat,
        num_layers,
        cpu_cache.data_ptr(),
        num_layers,
        [gpu_rank_layout.get_kv_stride() * elem] * tp_size,
        [gpu_rank_layout.get_block_stride() * elem] * tp_size,
        [gpu_rank_layout.get_layer_stride() * elem] * tp_size,
        [per_rank_chunk] * tp_size,
        gpu_device_ids,
        True,
        batch_size,
        data_type,
    )

    if cpu_layout.type == KVCacheLayoutType.BLOCKFIRST and not is_mla:
        cpu_stride_layout = cpu_layout.div_head(tp_size)
    else:
        cpu_stride_layout = cpu_layout
    cpu_kv_stride = cpu_stride_layout.get_kv_stride() * elem
    cpu_layer_stride = cpu_stride_layout.get_layer_stride() * elem
    cpu_block_stride = cpu_layout.get_block_stride() * elem
    cpu_tp_stride = cpu_block_stride // tp_size

    if is_mla:
        st_args = (
            cpu_size_table.data_ptr(),
            0,
            cpu_size_table.stride(0),
            cpu_size_table.stride(1),
        )
    else:
        st_args = (
            cpu_size_table_tp.data_ptr(),
            cpu_size_table_tp.stride(0),
            cpu_size_table_tp.stride(1),
            cpu_size_table_tp.stride(2),
        )

    try:
        ioctx = SSDIOCTX(ssd_files, len(ssd_files), 0, 0)

        tg.tp_group_transfer_ans(
            block_ids,
            write_cpu_ids,
            cpu_kv_stride,
            cpu_layer_stride,
            cpu_block_stride,
            cpu_tp_stride,
            4,
            False,
            False,
            0,
            num_layers,
            is_mla,
            *st_args,
        )
        for rank in range(tp_size):
            torch.cuda.synchronize(rank)
        if is_mla:
            assert (cpu_size_table.to(torch.int64)[write_cpu_ids] > 0).all()
        else:
            assert (cpu_size_table_tp.to(torch.int64)[:, write_cpu_ids] > 0).all()
        expected_payloads = snapshot_compressed_slots(
            cpu_cache,
            cpu_size_table if is_mla else cpu_size_table_tp,
            write_cpu_ids,
            ranks=[0] if is_mla else list(range(tp_size)),
            table_ranked=not is_mla,
            num_layers=num_layers,
            kv_dim=kv_dim,
            block_stride=cpu_block_stride,
            layer_stride=cpu_layer_stride,
            kv_stride=cpu_kv_stride,
            rank_stride=0 if is_mla else cpu_tp_stride)

        h2disk_bytes = transfer_kv_blocks_ssd_packed(
            ioctx=ioctx,
            cpu_layer_id_list=torch.arange(num_layers, dtype=torch.int32),
            cpu_tensor_ptr=cpu_cache.data_ptr(),
            ssd_block_ids=ssd_block_ids,
            cpu_block_ids=write_cpu_ids,
            cpu_layer_stride_in_bytes=cpu_layer_stride,
            cpu_kv_stride_in_bytes=cpu_kv_stride,
            chunk_size_in_bytes=per_rank_chunk,
            block_stride_in_bytes=cpu_block_stride,
            is_read=False,
            num_blocks_per_file=num_blocks_per_file,
            layout_type=cpu_layout.type.value,
            total_layers=num_layers,
            is_mla=is_mla,
            cpu_size_table_ptr=(cpu_size_table if is_mla else cpu_size_table_tp).data_ptr(),
            cpu_size_table_block_stride=(cpu_size_table if is_mla else cpu_size_table_tp).stride(0 if is_mla else 1),
            cpu_size_table_layer_stride=(cpu_size_table if is_mla else cpu_size_table_tp).stride(1 if is_mla else 2),
            ssd_size_table_ptr=(ssd_size_table if is_mla else ssd_size_table_tp).data_ptr(),
            ssd_size_table_block_stride=(ssd_size_table if is_mla else ssd_size_table_tp).stride(0 if is_mla else 1),
            ssd_size_table_layer_stride=(ssd_size_table if is_mla else ssd_size_table_tp).stride(1 if is_mla else 2),
            tp_size=1 if is_mla else tp_size,
            cpu_tp_rank_stride_in_bytes=0 if is_mla else cpu_tp_stride,
            cpu_size_table_rank_stride=0 if is_mla else cpu_size_table_tp.stride(0),
            ssd_size_table_rank_stride=0 if is_mla else ssd_size_table_tp.stride(0),
        )

        if is_mla:
            assert h2disk_bytes == int(ssd_size_table.to(torch.int64)[ssd_block_ids].sum().item())
            assert torch.equal(
                cpu_size_table.to(torch.int64)[write_cpu_ids],
                ssd_size_table.to(torch.int64)[ssd_block_ids],
            )
            cpu_size_table.zero_()
        else:
            assert h2disk_bytes == int(ssd_size_table_tp.to(torch.int64)[:, ssd_block_ids].sum().item())
            assert torch.equal(
                cpu_size_table_tp.to(torch.int64)[:, write_cpu_ids],
                ssd_size_table_tp.to(torch.int64)[:, ssd_block_ids],
            )
            cpu_size_table_tp.zero_()

        cpu_cache.zero_()
        disk2h_bytes = transfer_kv_blocks_ssd_packed(
            ioctx=ioctx,
            cpu_layer_id_list=torch.arange(num_layers, dtype=torch.int32),
            cpu_tensor_ptr=cpu_cache.data_ptr(),
            ssd_block_ids=ssd_block_ids,
            cpu_block_ids=read_cpu_ids,
            cpu_layer_stride_in_bytes=cpu_layer_stride,
            cpu_kv_stride_in_bytes=cpu_kv_stride,
            chunk_size_in_bytes=per_rank_chunk,
            block_stride_in_bytes=cpu_block_stride,
            is_read=True,
            num_blocks_per_file=num_blocks_per_file,
            layout_type=cpu_layout.type.value,
            total_layers=num_layers,
            is_mla=is_mla,
            cpu_size_table_ptr=(cpu_size_table if is_mla else cpu_size_table_tp).data_ptr(),
            cpu_size_table_block_stride=(cpu_size_table if is_mla else cpu_size_table_tp).stride(0 if is_mla else 1),
            cpu_size_table_layer_stride=(cpu_size_table if is_mla else cpu_size_table_tp).stride(1 if is_mla else 2),
            ssd_size_table_ptr=(ssd_size_table if is_mla else ssd_size_table_tp).data_ptr(),
            ssd_size_table_block_stride=(ssd_size_table if is_mla else ssd_size_table_tp).stride(0 if is_mla else 1),
            ssd_size_table_layer_stride=(ssd_size_table if is_mla else ssd_size_table_tp).stride(1 if is_mla else 2),
            tp_size=1 if is_mla else tp_size,
            cpu_tp_rank_stride_in_bytes=0 if is_mla else cpu_tp_stride,
            cpu_size_table_rank_stride=0 if is_mla else cpu_size_table_tp.stride(0),
            ssd_size_table_rank_stride=0 if is_mla else ssd_size_table_tp.stride(0),
        )
        assert disk2h_bytes == h2disk_bytes
        if is_mla:
            assert torch.equal(
                cpu_size_table.to(torch.int64)[read_cpu_ids],
                ssd_size_table.to(torch.int64)[ssd_block_ids],
            )
            assert (cpu_size_table.to(torch.int64)[write_cpu_ids] == 0).all()
        else:
            assert torch.equal(
                cpu_size_table_tp.to(torch.int64)[:, read_cpu_ids],
                ssd_size_table_tp.to(torch.int64)[:, ssd_block_ids],
            )
            assert (cpu_size_table_tp.to(torch.int64)[:, write_cpu_ids] == 0).all()
        assert_compressed_slots_equal(
            cpu_cache,
            cpu_size_table if is_mla else cpu_size_table_tp,
            read_cpu_ids,
            expected_payloads,
            ranks=[0] if is_mla else list(range(tp_size)),
            table_ranked=not is_mla,
            num_layers=num_layers,
            kv_dim=kv_dim,
            block_stride=cpu_block_stride,
            layer_stride=cpu_layer_stride,
            kv_stride=cpu_kv_stride,
            rank_stride=0 if is_mla else cpu_tp_stride)
    finally:
        del tg
        shutil.rmtree(tmpdir, ignore_errors=True)
