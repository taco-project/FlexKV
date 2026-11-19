"""A vLLM KV cache group with mixed page sizes whose heads TP splits (GQA/MHA),
through the GPU<->CPU worker of a node with two TP ranks (TP2, or TP4 over two
nodes): each rank's pages go to its own slot of the node's host block and come
back to that rank.

Built from the production pieces: FlexKVConfig turns vLLM's KVCacheConfig into
layer groups, FlexKVWorkerConnector._register_layer_groups describes each
rank's caches, the CPU layout is StorageEngine's, and GPUCPUTransferWorker
moves the bytes. Needs 2 GPUs.
"""
import pytest
import torch

pytest.importorskip("vllm.v1.kv_cache_layout")

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV  # noqa: E402
from flexkv.common.transfer import TransferType  # noqa: E402
from flexkv.integration.config import FlexKVConfig  # noqa: E402
from flexkv.server.shm_radix_bootstrap import cpu_kv_layout  # noqa: E402
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker  # noqa: E402

from test_vllm_layer_groups import (  # noqa: E402
    NUM_BLOCKS, _gqa_config, _register, _vllm_cache_views)

LOCAL_RANKS = 2
NUM_CPU_BLOCKS = 4
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

pytestmark = pytest.mark.skipif(
    NUM_GPUS < LOCAL_RANKS, reason=f"Need {LOCAL_RANKS} GPUs, found {NUM_GPUS}")


class _Handle:
    """An in-process TensorSharedHandle: the tensor itself, no CUDA IPC."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.device = tensor.device

    def get_tensor(self):
        return self.tensor


def _vllm_caches(config, device, seed):
    """One rank's raw KV allocation, random bytes, and its caches as vLLM's
    worker hands them over."""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (config.kv_cache_tensors[0].size,), dtype=torch.uint8,
                        generator=generator).to(device)
    return raw, _vllm_cache_views(config, config.kv_cache_groups[0].layer_names, raw=raw)


def _page(raw, config, name, block):
    """The bytes of block ``block`` of cache ``name`` in a rank's allocation."""
    for tensor in config.kv_cache_tensors:
        if name in tensor.layers:
            spec = config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs[name]
            start = (tensor.offset + tensor.layers.index(name) * tensor.layer_stride
                     + block * tensor.block_stride)
            return raw[start:start + spec.page_size_bytes]
    raise KeyError(name)


@pytest.mark.parametrize("region_batch", [True, False], ids=["region-batch", "thread-groups"])
@pytest.mark.parametrize("layout", ["BLHNC", "LBHNC"])
@pytest.mark.parametrize("tp_size,nnodes", [(2, 1), (4, 2)], ids=["tp2", "tp4-over-2-nodes"])
def test_split_heads_round_trip_through_per_rank_slots(
        monkeypatch, tp_size, nnodes, layout, region_batch):
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "region_batch", region_batch)
    config = _gqa_config(scheduler=False, layout=layout)
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = tp_size
    flexkv_config.model_config.nnodes = nnodes
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)
    model_config = flexkv_config.model_config
    group = flexkv_config.vllm_kv_group
    num_groups = len(model_config.layer_groups)

    raws, registrations = [], []
    for rank in range(LOCAL_RANKS):
        raw, caches = _vllm_caches(config, torch.device("cuda", rank), seed=rank)
        raws.append(raw)
        registrations.append(_register(flexkv_config, caches))

    host_layout = cpu_kv_layout(model_config, flexkv_config.cache_config, NUM_CPU_BLOCKS)
    host = torch.zeros(tuple(host_layout.kv_shape), dtype=torch.uint8)
    worker = GPUCPUTransferWorker(
        worker_id=0,
        transfer_conn=None,
        finished_ops_queue=None,
        op_buffer_tensor=torch.zeros(64, dtype=torch.int64),
        gpu_blocks=[[_Handle(t) for t in reg["kv_caches"]] for reg in registrations],
        cpu_blocks=host,
        gpu_kv_layouts=[reg["kv_layout"] for reg in registrations],
        cpu_kv_layout=host_layout,
        dtype=registrations[0]["kv_caches"][0].dtype,
        tp_group_size=model_config.effective_tp_size_per_node,
        layer_groups=model_config.layer_groups,
        gpu_blocks_per_group=[
            [[_Handle(t) for t in reg["handles_per_group"][gi]] for reg in registrations]
            for gi in range(num_groups)],
        gpu_layouts_per_group=[
            [reg["gpu_layouts"][gi] for reg in registrations] for gi in range(num_groups)],
    )
    try:
        gpu_src = torch.tensor([1, 6, 11], dtype=torch.int64).pin_memory()
        cpu_ids = torch.tensor([3, 0, 2], dtype=torch.int64).pin_memory()
        gpu_dst = torch.tensor([4, 9, NUM_BLOCKS - 1], dtype=torch.int64).pin_memory()
        names = [name for names in group.group_layer_names for name in names]

        worker._transfer_impl(gpu_src, cpu_ids, TransferType.D2H)

        slot = host.shape[1] // LOCAL_RANKS
        for gpu_block, cpu_block in zip(gpu_src.tolist(), cpu_ids.tolist()):
            for rank in range(LOCAL_RANKS):
                pages = torch.cat([_page(raws[rank], config, name, gpu_block) for name in names])
                assert torch.equal(host[cpu_block, rank * slot:(rank + 1) * slot], pages.cpu())

        originals = [raw.clone() for raw in raws]
        for raw in raws:
            raw.zero_()
        worker._transfer_impl(cpu_ids, gpu_dst, TransferType.H2D)

        for rank in range(LOCAL_RANKS):
            expected = torch.zeros_like(originals[rank])
            for src, dst in zip(gpu_src.tolist(), gpu_dst.tolist()):
                for name in names:
                    _page(expected, config, name, dst).copy_(
                        _page(originals[rank], config, name, src))
            assert torch.equal(raws[rank], expected), f"rank {rank}"
    finally:
        worker.shutdown()
