"""RegionBatchGroup must move exactly the bytes the per-group path moved.

The batched path is a performance change, not a semantic one: a DSv4-shaped
model with a full-attention region and an indexer region should end up with
byte-identical CPU and GPU contents whether the two regions went out as two
``TPTransferThreadGroup`` round trips or as one ``RegionBatchGroup.submit``.
These compare the two directly rather than checking each against a hand-derived
expectation, so a shared misunderstanding of the layout cannot pass.

Needs >= 2 GPUs and an extension built with RegionBatchGroup.
"""
import gc

import pytest
import torch

from flexkv.c_ext import TPTransferThreadGroup
from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.region_batch import (
    RegionSpec,
    build_region_batch,
    make_requests,
    region_batch_available,
)

NUM_GPUS = min(4, torch.cuda.device_count()) if torch.cuda.is_available() else 0

pytestmark = [
    pytest.mark.skipif(NUM_GPUS < 2,
                       reason=f"Need at least 2 GPUs, found {NUM_GPUS}"),
    pytest.mark.skipif(not region_batch_available(),
                       reason="extension built without RegionBatchGroup"),
]

DTYPE = torch.float16
ES = DTYPE.itemsize

# (num_layers, num_heads) per region. Different layer counts is the point:
# it is what makes these separate regions rather than one bigger one.
REGION_SHAPES = [(4, 8), (2, 8)]
NUM_BLOCKS = 8
TPB = 16
HEAD_DIM = 128


@pytest.fixture(autouse=True)
def _cleanup_gpu_mem():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Region:
    """One region's tensors and strides, built the way the worker builds them."""

    def __init__(self, name, num_layers, num_heads, kv_dim, cpu_layout_name,
                 num_gpus, seed):
        self.name = name
        self.num_layers = num_layers
        self.num_kv_heads = num_heads
        self.kv_dim = kv_dim
        self.num_gpus = num_gpus
        heads_per_rank = num_heads // num_gpus

        self.gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=num_layers, num_block=NUM_BLOCKS, tokens_per_block=TPB,
            num_head=heads_per_rank, head_size=HEAD_DIM, kv_dim=kv_dim,
            num_kv_heads=num_heads)
        cpu_layout = KVCacheLayout(
            type=KVCacheLayoutType[cpu_layout_name],
            num_layer=num_layers, num_block=NUM_BLOCKS, tokens_per_block=TPB,
            num_head=num_heads, head_size=HEAD_DIM, kv_dim=kv_dim,
            num_kv_heads=num_heads)
        cpu_layout_tp = (cpu_layout.div_head(num_gpus)
                         if cpu_layout.type == KVCacheLayoutType.BLOCKFIRST
                         else cpu_layout)

        self.cpu_kv_stride = cpu_layout_tp.get_kv_stride() * ES
        self.cpu_layer_stride = cpu_layout_tp.get_layer_stride() * ES
        self.cpu_block_stride = cpu_layout.get_block_stride() * ES
        self.cpu_tp_stride = self.cpu_block_stride // num_gpus

        self.gpu_kv_stride = self.gpu_layout.get_kv_stride() * ES
        self.gpu_block_stride = self.gpu_layout.get_block_stride() * ES
        self.gpu_layer_stride = self.gpu_layout.get_layer_stride() * ES
        self.gpu_chunk_size = self.gpu_layout.get_chunk_size() * ES

        # Per-rank GPU buffer, one tensor per layer (the vLLM shape).
        self.per_rank = []
        for g in range(num_gpus):
            full = torch.zeros(
                (num_layers, kv_dim, NUM_BLOCKS, TPB, heads_per_rank, HEAD_DIM),
                dtype=DTYPE, device=f"cuda:{g}")
            self.per_rank.append([full[i] for i in range(num_layers)])
            self._keepalive = getattr(self, "_keepalive", [])
            self._keepalive.append(full)

        self.cpu = torch.zeros(tuple(cpu_layout.kv_shape), dtype=DTYPE,
                               pin_memory=True)
        self.seed = seed
        self.fill_gpu()

    def fill_gpu(self):
        """Distinct per (region, rank, layer) so a swapped region or rank shows."""
        for g in range(self.num_gpus):
            for l, t in enumerate(self.per_rank[g]):
                base = (self.seed * 7919 + g * 131 + l * 17) % 997
                ramp = torch.arange(t.numel(), device=t.device,
                                    dtype=torch.float32) % 97
                t.view(-1)[:] = ((base + ramp) / 997.0).to(DTYPE)

    def zero_gpu(self):
        for g in range(self.num_gpus):
            for t in self.per_rank[g]:
                t.zero_()

    def gpu_ptrs_flat(self):
        return [self.per_rank[g][l].data_ptr()
                for g in range(self.num_gpus)
                for l in range(self.num_layers)]

    def snapshot_gpu(self):
        return [[t.detach().clone().cpu() for t in self.per_rank[g]]
                for g in range(self.num_gpus)]

    def make_tp_group(self, is_blockfirst):
        return TPTransferThreadGroup(
            num_gpus=self.num_gpus,
            gpu_block_ptrs_flat=self.gpu_ptrs_flat(),
            num_tensors_per_gpu=self.num_layers,
            cpu_blocks_ptr=self.cpu.data_ptr(),
            num_layers=self.num_layers,
            gpu_kv_strides_in_bytes=[self.gpu_kv_stride] * self.num_gpus,
            gpu_block_strides_in_bytes=[self.gpu_block_stride] * self.num_gpus,
            gpu_layer_strides_in_bytes=[self.gpu_layer_stride] * self.num_gpus,
            gpu_chunk_sizes_in_bytes=[self.gpu_chunk_size] * self.num_gpus,
            gpu_device_ids=list(range(self.num_gpus)),
            enable_nvcomp=False, nvcomp_batch_size=0, nvcomp_data_type=0,
            ce_segment_threshold=GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
            ce_path_opt=GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
            ce_enable_memcpy2d=GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
            ce_gather_threads=GLOBAL_CONFIG_FROM_ENV.ce_gather_threads,
            ce_gather_nt=GLOBAL_CONFIG_FROM_ENV.ce_gather_nt,
            is_blockfirst=is_blockfirst,
            num_kv_heads=self.num_kv_heads,
        )

    def to_spec(self):
        return RegionSpec(
            name=self.name,
            cpu_ptr=self.cpu.data_ptr(),
            cpu_kv_stride=self.cpu_kv_stride,
            cpu_layer_stride=self.cpu_layer_stride,
            cpu_block_stride=self.cpu_block_stride,
            cpu_tp_stride=self.cpu_tp_stride,
            gpu_block_ptrs_flat=self.gpu_ptrs_flat(),
            num_tensors_per_gpu=self.num_layers,
            gpu_kv_strides=[self.gpu_kv_stride] * self.num_gpus,
            gpu_block_strides=[self.gpu_block_stride] * self.num_gpus,
            gpu_layer_strides=[self.gpu_layer_stride] * self.num_gpus,
            gpu_chunk_sizes=[self.gpu_chunk_size] * self.num_gpus,
            num_layers=self.num_layers,
            kv_dim=self.kv_dim,
            num_kv_heads=self.num_kv_heads,
        )


def build_regions(kv_dim, cpu_layout_name, num_gpus):
    return [Region(f"region{i}", nl, nh, kv_dim, cpu_layout_name, num_gpus,
                   seed=i + 1)
            for i, (nl, nh) in enumerate(REGION_SHAPES)]


def sync_all(num_gpus):
    for g in range(num_gpus):
        torch.cuda.synchronize(g)


def block_ids(n):
    # Pinned, like worker.py: the copy kernel dereferences these host pointers
    # from the device, so pageable memory is an illegal access, not a slow path.
    return torch.arange(n, dtype=torch.int64).pin_memory()


def per_group_d2h(regions, is_blockfirst, use_ce):
    ids = block_ids(NUM_BLOCKS)
    for r in regions:
        tp = r.make_tp_group(is_blockfirst)
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=r.cpu_kv_stride,
            cpu_layer_stride_in_bytes=r.cpu_layer_stride,
            cpu_block_stride_in_bytes=r.cpu_block_stride,
            cpu_tp_stride_in_bytes=r.cpu_tp_stride,
            transfer_num_cta=4, is_host_to_device=False,
            use_ce_transfer=use_ce, layer_id=0,
            layer_granularity=r.num_layers, kv_dim=r.kv_dim,
            num_kv_heads=r.num_kv_heads,
            kv_shared_across_ranks_mode="sharded")
        del tp
    sync_all(regions[0].num_gpus)


def batched(regions, is_blockfirst, num_gpus):
    return build_region_batch(
        [r.to_spec() for r in regions], list(range(num_gpus)),
        ce_segment_threshold=GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
        ce_path_opt=GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
        ce_enable_memcpy2d=GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
        is_blockfirst=is_blockfirst,
        num_kv_heads=regions[0].num_kv_heads,
        ce_gather_threads=GLOBAL_CONFIG_FROM_ENV.ce_gather_threads,
        ce_gather_nt=GLOBAL_CONFIG_FROM_ENV.ce_gather_nt,
    )


@pytest.mark.parametrize("cpu_layout_name", ["LAYERFIRST", "BLOCKFIRST"])
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["packed", "plain"])
@pytest.mark.parametrize("use_ce", [False, True], ids=["cuda", "ce"])
def test_d2h_writes_the_same_cpu_bytes_as_the_per_group_path(
        cpu_layout_name, kv_dim, use_ce):
    num_gpus = NUM_GPUS
    is_blockfirst = cpu_layout_name == "BLOCKFIRST"

    ref = build_regions(kv_dim, cpu_layout_name, num_gpus)
    per_group_d2h(ref, is_blockfirst, use_ce)
    expected = [r.cpu.clone() for r in ref]
    del ref

    got = build_regions(kv_dim, cpu_layout_name, num_gpus)
    group = batched(got, is_blockfirst, num_gpus)
    group.submit(
        make_requests(len(got), block_ids(NUM_BLOCKS), block_ids(NUM_BLOCKS),
                      False, transfer_num_cta=4, use_ce_transfer=use_ce),
        True)
    sync_all(num_gpus)

    for i, (r, exp) in enumerate(zip(got, expected)):
        assert torch.equal(r.cpu, exp), (
            f"region {i} ({r.name}): batched D2H differs from the per-group "
            f"path; {int((r.cpu != exp).sum())} of {exp.numel()} elements")
    del group


@pytest.mark.parametrize("cpu_layout_name", ["LAYERFIRST", "BLOCKFIRST"])
@pytest.mark.parametrize("use_ce", [False, True], ids=["cuda", "ce"])
def test_roundtrip_through_the_batch_restores_every_regions_gpu_data(
        cpu_layout_name, use_ce):
    """D2H then H2D, both batched: each rank must get its own bytes back.

    A region index used for the wrong region's strides, or a rank offset
    applied to the wrong region, survives D2H alone -- it shows up here.
    """
    num_gpus = NUM_GPUS
    is_blockfirst = cpu_layout_name == "BLOCKFIRST"
    regions = build_regions(2, cpu_layout_name, num_gpus)
    before = [r.snapshot_gpu() for r in regions]

    group = batched(regions, is_blockfirst, num_gpus)
    ids = block_ids(NUM_BLOCKS)
    group.submit(make_requests(len(regions), ids, ids, False,
                               transfer_num_cta=4, use_ce_transfer=use_ce),
                 True)
    sync_all(num_gpus)

    for r in regions:
        r.zero_gpu()
    sync_all(num_gpus)

    group.submit(make_requests(len(regions), ids, ids, True,
                               transfer_num_cta=4, use_ce_transfer=use_ce),
                 True)
    sync_all(num_gpus)

    for ri, r in enumerate(regions):
        for g in range(num_gpus):
            for l in range(r.num_layers):
                got = r.per_rank[g][l].detach().cpu()
                assert torch.equal(got, before[ri][g][l]), (
                    f"region {r.name} rank {g} layer {l} did not round-trip")
    del group


