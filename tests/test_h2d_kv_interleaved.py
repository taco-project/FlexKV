"""Layerwise H2D into Recsys/HSTU pages: GPU is [block, kv, token, head, dim].

K/V interleave makes gpu_block_stride == 2*chunk, so the 1D phys-contig check
fails. memcpy2d must land the bytes in one CE hop (pitch = 2*chunk). The old
GATHER_SCATTER path H2D'd into packed staging and index_copy_'d back — 2x the
bytes, which is why Recsys layerwise H2D was ~2x naive fused.

Run:
    pytest tests/test_h2d_kv_interleaved.py -v
"""
from __future__ import annotations

import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.region_batch import (
    RegionSpec,
    build_region_batch,
    make_requests,
)
from flexkv.transfer.template import gpu_strides_from_tensor

from eventfd_probe import Fds

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)

DTYPE = torch.float16
ES = DTYPE.itemsize


def test_layerwise_h2d_kv_interleaved_gpu_roundtrip():
    num_layers, num_blocks, tpb, num_heads, head_dim = 8, 32, 32, 4, 64
    kv_dim = 2
    chunk = tpb * num_heads * head_dim * ES

    # Recsys DeviceKVCache: [layers, pages, 2, page, heads, dim] unbound on 0.
    gpu_full = torch.zeros(
        (num_layers, num_blocks, kv_dim, tpb, num_heads, head_dim),
        dtype=DTYPE, device="cuda:0",
    )
    gpu_layers = list(gpu_full.unbind(dim=0))
    measured = gpu_strides_from_tensor(gpu_layers[0], tpb, ES, kv_dim)
    assert measured is not None
    kv_s, blk_s, layer_s = measured
    assert blk_s == 2 * chunk, (
        f"expected interleaved GPU pitch 2*chunk={2 * chunk}, got {blk_s}"
    )

    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers,
        num_block=num_blocks,
        tokens_per_block=tpb,
        num_head=num_heads,
        head_size=head_dim,
        kv_dim=kv_dim,
        num_kv_heads=num_heads,
    )
    cpu = torch.arange(
        cpu_layout.kv_shape.numel(), dtype=torch.int32,
    ).to(DTYPE).reshape(tuple(cpu_layout.kv_shape)).pin_memory()

    spec = RegionSpec(
        name="kv",
        cpu_ptr=cpu.data_ptr(),
        cpu_kv_stride=cpu_layout.get_kv_stride() * ES,
        cpu_layer_stride=cpu_layout.get_layer_stride() * ES,
        cpu_block_stride=cpu_layout.get_block_stride() * ES,
        cpu_tp_stride=cpu_layout.get_block_stride() * ES,
        gpu_block_ptrs_flat=[t.data_ptr() for t in gpu_layers],
        num_tensors_per_gpu=num_layers,
        gpu_kv_strides=[kv_s],
        gpu_block_strides=[blk_s],
        gpu_layer_strides=[layer_s],
        gpu_chunk_sizes=[chunk],
        num_layers=num_layers,
        kv_dim=kv_dim,
        num_kv_heads=num_heads,
    )
    group = build_region_batch(
        [spec], [0],
        ce_path_opt=True,
        ce_enable_memcpy2d=True,
        is_blockfirst=False,
        num_kv_heads=num_heads,
    )
    ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    requests = []
    for layer in range(num_layers):
        req = make_requests(
            1, ids, ids, True,
            transfer_num_cta=4, use_ce_transfer=True,
            layer_id=layer, layer_granularity=1,
        )[0]
        req.milestone_layer = layer
        requests.append(req)

    with Fds(num_counters=1, tp_size=1, num_layers=num_layers) as fds:
        group.set_layer_eventfds(fds.tensor(), 1, num_layers, "hostfunc")
        group.submit_layerwise(requests, [], 0)
        ok, err = group.wait_layer_completion(120.0)
        assert ok, f"layerwise H2D did not complete: {err}"
        torch.cuda.synchronize()
        del group

    # CPU LAYERFIRST is [layer, kv, block, ...]; GPU page is [block, kv, ...].
    gpu_host = gpu_full.cpu()
    for layer in range(num_layers):
        for kv in range(kv_dim):
            torch.testing.assert_close(
                gpu_host[layer, :, kv],
                cpu[layer, kv],
                rtol=0, atol=0,
                msg=f"layer={layer} kv={kv}",
            )
