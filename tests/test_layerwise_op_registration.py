"""CPU control-path tests using the real unified worker and mocked CUDA resources."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.pool import PoolId
from flexkv.transfer.completion import CompletionContract
from flexkv.transfer.worker_op import WorkerLayerwiseTransferOp
from flexkv.transfer.workers import gpu_cpu
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker, _Pool

pytestmark = pytest.mark.unit


@pytest.fixture
def create_worker(monkeypatch):
    registrations = []
    monkeypatch.setattr(GPUCPUTransferWorker, "_ensure_cuda_device", lambda *a: None)
    monkeypatch.setattr(GPUCPUTransferWorker, "_import_tensor_handles", lambda self, handles: handles)
    monkeypatch.setattr(GPUCPUTransferWorker, "_register_host_tensor",
                        lambda self, tensor, label: registrations.append((tensor, label)))
    monkeypatch.setattr(GPUCPUTransferWorker, "_init_uniform", lambda self, *a, pool_id, **kw:
                        _Pool(pool_id=pool_id, name=pool_id.name, layer_members=[[], []]))
    monkeypatch.setattr(GPUCPUTransferWorker, "_build_region_batch",
                        lambda self, layout: setattr(self, "region_batch", Mock()))
    monkeypatch.setattr(GPUCPUTransferWorker, "_materialize_thread_groups", lambda self: None)
    monkeypatch.setattr(gpu_cpu, "receive_layer_eventfds", lambda *a, **kw: torch.tensor([3, 4]))
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "layerwise_completion_contract", CompletionContract.PER_LAYER)
    cpu, swa, shared = torch.empty(16), torch.empty(8), torch.zeros((2, 4), dtype=torch.int64)
    gpu = SimpleNamespace(device=torch.device("cuda:0"))
    layout = SimpleNamespace(kv_dim=1, num_kv_heads=1, num_layer=2, type=None)

    def create(completion=None, socket=None, with_swa=False):
        kwargs = dict(worker_id=1, transfer_conn=Mock(), finished_ops_queue=Mock(),
                      op_buffer_tensor=shared, gpu_blocks=[[gpu]], cpu_blocks=cpu,
                      gpu_kv_layouts=[layout], cpu_kv_layout=layout, dtype=torch.float32,
                      tp_group_size=1, compressor=Mock(), completion=completion,
                      layerwise_eventfd_socket=socket)
        if with_swa:
            kwargs.update(swa_cpu_blocks=swa, swa_gpu_blocks=[[gpu]],
                          swa_cpu_kv_layout=layout, swa_gpu_kv_layouts=[layout])
        return GPUCPUTransferWorker(**kwargs)
    return create, registrations, shared, cpu, swa


@pytest.mark.parametrize("completion,socket,pinned", [
    (None, None, True), ("whole", None, True), (CompletionContract.WHOLE, None, True),
    ("per_layer", "test.sock", False), (CompletionContract.PER_LAYER, "test.sock", False),
    (None, "test.sock", False),
])
@pytest.mark.parametrize("with_swa", [False, True])
def test_registration_follows_id_consumption(create_worker, completion, socket, pinned, with_swa):
    create, registrations, shared, cpu, swa = create_worker
    worker = create(completion, socket, with_swa)
    assert worker._op_buffer_pinned is pinned
    assert any(tensor is shared for tensor, _ in registrations) is pinned
    assert any(tensor is cpu for tensor, _ in registrations)
    assert any(tensor is swa for tensor, _ in registrations) is with_swa


def test_layerwise_dispatch_never_registers_shared_ids(create_worker, monkeypatch):
    create, registrations, shared, _, _ = create_worker
    worker = create("per_layer", "test.sock")
    monkeypatch.setattr(worker, "_launch_layerwise", Mock(return_value=True))
    op = object.__new__(WorkerLayerwiseTransferOp)
    assert worker.launch_transfer(op)
    assert all(tensor is not shared for tensor, _ in registrations)


@pytest.mark.parametrize("src_slot,dst_slot", [(0, 1), (-1, 1), (0, -1), (-1, -1)])
def test_classic_dispatch_on_layerwise_worker_still_pins_shared_views(create_worker, monkeypatch, src_slot, dst_slot):
    create, registrations, shared, _, _ = create_worker
    worker = create("per_layer", "test.sock")
    needs_shared = src_slot >= 0 or dst_slot >= 0

    def consume(op):
        assert worker._op_buffer_pinned is needs_shared
        return torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64)

    monkeypatch.setattr(worker, "get_transfer_block_ids", consume)
    monkeypatch.setattr(worker, "_pool_for", lambda op: worker._pools[PoolId.FULL_KV])
    monkeypatch.setattr(worker, "_is_per_group", lambda pool: False)
    op = SimpleNamespace(src_slot_id=src_slot, dst_slot_id=dst_slot,
                         src_block_ids=np.array([0]), dst_block_ids=np.array([0]))
    assert worker.launch_transfer(op)
    assert worker.launch_transfer(op)
    assert sum(tensor is shared for tensor, _ in registrations) == int(needs_shared)
