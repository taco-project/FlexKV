"""Real host registration, H2D and eventfd completion through the constructor.

Only IPC transport is replaced by local tensor handles and local eventfds;
registration, worker geometry, ID staging and native copies run on CUDA.
"""
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import LayerwiseTransferOp, TransferOp, TransferType
from flexkv.transfer.worker_op import WorkerLayerwiseTransferOp, WorkerTransferOp
from flexkv.transfer.workers import gpu_cpu
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("use_ce", [False, True])
@pytest.mark.parametrize("completion", ["whole", "per_layer"])
@pytest.mark.parametrize("with_swa", [False, True])
def test_constructor_registration_and_copy(monkeypatch, use_ce, completion, with_swa):
    layout = KVCacheLayout(KVCacheLayoutType.LAYERFIRST, num_layer=2, num_block=8,
                           tokens_per_block=1, num_head=1, head_size=32, kv_dim=1)
    cpu = torch.arange(np.prod(layout.kv_shape), dtype=torch.int64).reshape(layout.kv_shape)
    cpu = (cpu % 251).to(torch.uint8)
    swa_cpu = (cpu + 3).clone()
    gpu = [torch.zeros(layout.kv_shape[1:], dtype=torch.uint8, device="cuda") for _ in range(2)]
    swa_gpu = [torch.zeros_like(tensor) for tensor in gpu]
    shared = torch.zeros((2, 16), dtype=torch.int64).share_memory_()
    shared[0, 0], shared[1, 0] = 2, 5
    fds = [os.eventfd(0, os.EFD_NONBLOCK) for _ in range(2)]
    monkeypatch.setattr(gpu_cpu, "receive_layer_eventfds", lambda *a, **kw: torch.tensor(fds, dtype=torch.int32))

    def handles(tensors):
        return [[SimpleNamespace(device=t.device, get_tensor=lambda t=t: t) for t in tensors]]

    kwargs = {}
    if with_swa:
        kwargs.update(swa_gpu_blocks=handles(swa_gpu), swa_cpu_blocks=swa_cpu,
                      swa_gpu_kv_layouts=[layout], swa_cpu_kv_layout=layout, swa_dtype=torch.uint8)
    worker = GPUCPUTransferWorker(
        0, Mock(), Mock(), shared, handles(gpu), cpu, [layout], layout, torch.uint8, 1,
        use_ce_transfer_h2d=use_ce, use_ce_transfer_d2h=use_ce,
        completion=completion, layerwise_eventfd_socket="local-test" if completion == "per_layer" else None,
        **kwargs,
    )
    try:
        assert worker._op_buffer_pinned == (completion == "whole")
        if completion == "per_layer":
            op = LayerwiseTransferOp(0, np.array([2], dtype=np.int64), np.array([5], dtype=np.int64),
                                     np.array([1], dtype=np.int64) if with_swa else None,
                                     np.array([6], dtype=np.int64) if with_swa else None)
            worker.launch_transfer(WorkerLayerwiseTransferOp(op))
            assert not worker._op_buffer_pinned
            assert [os.eventfd_read(fd) for fd in fds] == [1, 1]
            if with_swa:
                for layer in range(2):
                    assert torch.equal(swa_gpu[layer][:, 6].cpu(), swa_cpu[layer, :, 1])
        else:
            op = TransferOp(0, TransferType.H2D, np.array([2], dtype=np.int64), np.array([5], dtype=np.int64))
            op.src_slot_id, op.dst_slot_id, op.valid_block_num = 0, 1, 1
            worker.launch_transfer(WorkerTransferOp(op))
        torch.cuda.synchronize()
        for layer in range(2):
            assert torch.equal(gpu[layer][:, 5].cpu(), cpu[layer, :, 2])
            assert torch.count_nonzero(gpu[layer][:, :5]) == 0
            assert torch.count_nonzero(gpu[layer][:, 6:]) == 0

        # Mixed dispatch retains the shared-slot fallback on a per-layer worker.
        shared[0, 0], shared[1, 0] = 5, 3
        back = TransferOp(0, TransferType.D2H, np.array([5], dtype=np.int64), np.array([3], dtype=np.int64))
        back.src_slot_id, back.dst_slot_id, back.valid_block_num = 0, 1, 1
        worker.launch_transfer(WorkerTransferOp(back))
        assert worker._op_buffer_pinned
        torch.cuda.synchronize()
        assert torch.equal(cpu[:, :, 3], cpu[:, :, 2])
    finally:
        worker.shutdown()
        for fd in fds:
            os.close(fd)
    assert worker._host_registered == []
