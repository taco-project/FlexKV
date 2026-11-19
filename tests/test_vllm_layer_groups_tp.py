"""A vLLM KV cache group with mixed page sizes whose heads TP splits (GQA/MHA),
through the GPU<->CPU worker at TP2: each rank's pages go to its own slot of
the host block and come back to that rank.

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
from flexkv.integration.vllm.vllm_v1_adapter import FlexKVWorkerConnector  # noqa: E402
from flexkv.server.shm_radix_bootstrap import cpu_kv_layout  # noqa: E402
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker  # noqa: E402

from test_vllm_layer_groups import NUM_BLOCKS, _gqa_config  # noqa: E402

TP = 2
NUM_CPU_BLOCKS = 4
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

pytestmark = pytest.mark.skipif(NUM_GPUS < TP, reason=f"Need {TP} GPUs, found {NUM_GPUS}")


class _Handle:
    """An in-process TensorSharedHandle: the tensor itself, no CUDA IPC."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.device = tensor.device

    def get_tensor(self):
        return self.tensor


class _Registration:
    """Stands in for KVTPClient: keeps what the connector registers."""

    def register_to_server(self, **kwargs):
        self.kwargs = kwargs


def _vllm_caches(config, device, seed):
    """One rank's raw KV allocation, random bytes, and its caches as vLLM's
    worker hands them over: per layer a logical ``[B, H, N, C]`` view whose
    block ``b`` sits at ``offset + l * layer_stride + b * block_stride``."""
    specs = config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    size = config.kv_cache_tensors[0].size
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=generator).to(device)
    data = raw.view(torch.bfloat16)
    caches = {}
    for tensor in config.kv_cache_tensors:
        for index, name in enumerate(tensor.layers):
            spec = specs[name]
            content = spec.head_size + spec.head_size_v
            caches[name] = torch.as_strided(
                data,
                (config.num_blocks, spec.num_kv_heads, spec.block_size, content),
                (tensor.block_stride // 2, spec.block_size * content, content, 1),
                (tensor.offset + index * tensor.layer_stride) // 2)
    return raw, caches


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
def test_split_heads_round_trip_through_per_rank_slots(monkeypatch, layout, region_batch):
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "region_batch", region_batch)
    config = _gqa_config(scheduler=False, layout=layout)
    flexkv_config = FlexKVConfig()
    flexkv_config.model_config.tp_size = TP
    flexkv_config._configure_kv_layout_from_vllm_cache_config(config)
    model_config = flexkv_config.model_config
    group = flexkv_config.vllm_kv_group
    num_groups = len(model_config.layer_groups)

    raws, registrations = [], []
    for rank in range(TP):
        raw, caches = _vllm_caches(config, torch.device("cuda", rank), seed=rank)
        connector = object.__new__(FlexKVWorkerConnector)
        connector.flexkv_config = flexkv_config
        connector.enable_layerwise_transfer = False
        connector.layerwise_counter_pool = None
        connector.tp_client = _Registration()
        connector._register_layer_groups(caches, group, resume=False)
        raws.append(raw)
        registrations.append(connector.tp_client.kwargs)

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
        tp_group_size=TP,
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

        slot = host.shape[1] // TP
        for gpu_block, cpu_block in zip(gpu_src.tolist(), cpu_ids.tolist()):
            for rank in range(TP):
                pages = torch.cat([_page(raws[rank], config, name, gpu_block) for name in names])
                assert torch.equal(host[cpu_block, rank * slot:(rank + 1) * slot], pages.cpu())

        originals = [raw.clone() for raw in raws]
        for raw in raws:
            raw.zero_()
        worker._transfer_impl(cpu_ids, gpu_dst, TransferType.H2D)

        for rank in range(TP):
            expected = torch.zeros_like(originals[rank])
            for src, dst in zip(gpu_src.tolist(), gpu_dst.tolist()):
                for name in names:
                    _page(expected, config, name, dst).copy_(
                        _page(originals[rank], config, name, src))
            assert torch.equal(raws[rank], expected), f"rank {rank}"
    finally:
        worker.shutdown()
