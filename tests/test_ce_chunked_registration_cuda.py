"""CE and SM copies span several separately registered host chunks."""
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
def test_pitched_copy_across_registered_chunks(monkeypatch, use_ce, completion):
    # Registration uses 60 KiB chunks after the alignment cushion. Each CPU
    # layer's 8-row pitched copy spans 120 KiB, and some rows straddle a chunk.
    monkeypatch.setenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", str(64 / 1024**2))
    cpu_layout = KVCacheLayout(KVCacheLayoutType.BLOCKFIRST, num_layer=2,
                              num_block=8, tokens_per_block=1,
                              num_head=1, head_size=8192, kv_dim=1)
    gpu_layout = KVCacheLayout(KVCacheLayoutType.LAYERFIRST, num_layer=2,
                              num_block=8, tokens_per_block=1,
                              num_head=1, head_size=8192, kv_dim=1)
    cpu = (torch.arange(8 * 2 * 8192, dtype=torch.int64) % 251).to(torch.uint8).share_memory_()
    expected = cpu.reshape(8, 2, 8192).clone()
    gpu = [torch.zeros((1, 8, 1, 8192), dtype=torch.uint8, device="cuda") for _ in range(2)]
    shared = torch.zeros((2, 16), dtype=torch.int64).share_memory_()
    shared[:, :8] = torch.arange(8)
    handles = [[SimpleNamespace(device=t.device, get_tensor=lambda t=t: t) for t in gpu]]
    fds = [os.eventfd(0, os.EFD_NONBLOCK) for _ in range(2)]
    monkeypatch.setattr(gpu_cpu, "receive_layer_eventfds", lambda *a, **k: torch.tensor(fds, dtype=torch.int32))
    worker = GPUCPUTransferWorker(0, Mock(), Mock(), shared, handles, cpu,
        [gpu_layout], cpu_layout, torch.uint8, 1,
        use_ce_transfer_h2d=use_ce, use_ce_transfer_d2h=use_ce,
        completion=completion, layerwise_eventfd_socket="test" if completion=="per_layer" else None)
    try:
        ids = np.arange(8, dtype=np.int64)
        if completion == "per_layer":
            worker.launch_transfer(WorkerLayerwiseTransferOp(LayerwiseTransferOp(0, ids, ids)))
            assert [os.eventfd_read(fd) for fd in fds] == [1, 1]
        else:
            op = TransferOp(0, TransferType.H2D, ids, ids)
            op.src_slot_id, op.dst_slot_id, op.valid_block_num = 0, 1, 8
            worker.launch_transfer(WorkerTransferOp(op))
        torch.cuda.synchronize()
        for layer in range(2):
            assert torch.equal(gpu[layer].cpu().reshape(8, 8192), expected[:, layer])
            gpu[layer].copy_(((gpu[layer].to(torch.int16) + 7) % 251).to(torch.uint8))
        expected = ((expected.to(torch.int16) + 7) % 251).to(torch.uint8)
        cpu.zero_()
        op = TransferOp(1, TransferType.D2H, ids, ids)
        op.src_slot_id, op.dst_slot_id, op.valid_block_num = 0, 1, 8
        worker.launch_transfer(WorkerTransferOp(op))
        torch.cuda.synchronize()
        assert torch.equal(cpu.reshape_as(expected), expected)
    finally:
        worker.shutdown()
        for fd in fds:
            os.close(fd)
