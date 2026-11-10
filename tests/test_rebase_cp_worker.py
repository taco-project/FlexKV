"""Owner routing on main's WHOLE/PER_LAYER region planner, without GPU DMA."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from flexkv.common.pool import PoolId
from flexkv.common.transfer import TransferType
from flexkv.transfer.completion import CompletionContract
from flexkv.transfer.workers import gpu_cpu as worker_module
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker
from flexkv.transfer.transfer_engine import TransferEngine


@pytest.fixture
def worker(monkeypatch):
    def requests(num_regions, gpu_ids, cpu_ids, is_h2d, **kwargs):
        return [SimpleNamespace(gpu_block_id_tensor=gpu_ids, cpu_block_id_tensor=cpu_ids,
                                is_host_to_device=is_h2d, **kwargs)]

    monkeypatch.setattr(worker_module, "make_requests", requests)
    w = GPUCPUTransferWorker.__new__(GPUCPUTransferWorker)
    w.h2d_layer_holders = [-1, -1, 0, 1]
    w.layer_shard_enabled = True
    w._rank_share_mode = "sharded"
    w._owner_d2h_share_mode = "rank0_only"
    w.transfer_num_cta_h2d = w.transfer_num_cta_d2h = 4
    w.use_ce_transfer_h2d = w.use_ce_transfer_d2h = True
    w._use_async_launch = True
    w._completion = CompletionContract.PER_LAYER
    w._num_original_layers = 4
    w.num_layers = 2
    w._request_cache = {}
    w._layerwise_completion_timeout_s = 1
    w.worker_id = 0
    pool = SimpleNamespace(
        pool_id=PoolId.FULL_KV, region_indices=[0, 1, 2, 3],
        layer_members=[[], [], [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 1)]],
    )
    w._pools = {PoolId.FULL_KV: pool}
    w._layer_milestones = [
        [], [], [(PoolId.FULL_KV, region, 0) for region in range(4)],
        [(PoolId.FULL_KV, 0, 1)],
    ]
    w.region_batch = SimpleNamespace(
        num_regions=4, submit=MagicMock(), wait_all_streams=MagicMock(),
        submit_layerwise=MagicMock(), wait_layer_completion=MagicMock(return_value=(True, "")),
    )
    return w


def test_whole_h2d_routes_every_sidecar_to_the_same_owner(worker):
    ids = torch.tensor([1, 2], dtype=torch.int64)
    worker._transfer_impl(ids, ids, TransferType.H2D)
    requests, sync = worker.region_batch.submit.call_args.args
    assert sync is False
    assert [req.h2d_rank for req in requests] == [0, 0, 0, 0, 1]
    assert all(req.layer_granularity == 1 and req.is_host_to_device for req in requests)
    worker.region_batch.wait_all_streams.assert_called_once()


def test_physical_d2h_uses_designated_owner_not_sharded_scratch(worker):
    ids = torch.tensor([1], dtype=torch.int64)
    worker._transfer_impl(ids, ids, TransferType.D2H)
    requests = worker.region_batch.submit.call_args.args[0]
    assert [req.designated_rank for req in requests] == [0, 0, 0, 0, 1]
    assert all(req.share_mode == "rank0_only" and req.h2d_rank == -1 for req in requests)
    assert all(not req.is_host_to_device for req in requests)


def test_nonphysical_holders_do_not_change_d2h_mode(worker):
    worker.layer_shard_enabled = False
    worker.h2d_layer_holders = [0, 0, 0, 1]
    ids = torch.tensor([1], dtype=torch.int64)
    worker._transfer_impl(ids, ids, TransferType.D2H)
    requests = worker.region_batch.submit.call_args.args[0]
    assert len(requests) == 1
    assert requests[0].share_mode == "sharded"
    assert requests[0].region_indices == [0, 1, 2, 3]


def test_default_cp1_keeps_whole_region_requests(worker):
    worker.h2d_layer_holders = None
    worker.layer_shard_enabled = False
    ids = torch.tensor([1], dtype=torch.int64)
    worker._transfer_impl(ids, ids, TransferType.H2D)
    requests = worker.region_batch.submit.call_args.args[0]
    assert len(requests) == 1
    assert not hasattr(requests[0], "h2d_rank")


def test_layerwise_posts_nonlocal_empty_layers_and_keeps_original_upper_bound(worker):
    ids = torch.tensor([1], dtype=torch.int64)
    worker._layerwise_transfer_impl(ids, ids, None, None, counter_id=2)
    requests, empty_layers, counter_id = worker.region_batch.submit_layerwise.call_args.args
    assert empty_layers == [0, 1]
    assert counter_id == 2
    assert [req.milestone_layer for req in requests] == [2, 2, 2, 2, 3]
    assert [req.h2d_rank for req in requests] == [0, 0, 0, 0, 1]
    worker.region_batch.wait_layer_completion.assert_called_once_with(1)


def test_layerwise_cache_is_rebound_without_rebuilding_static_owner_plan(worker):
    first = worker._compiled_requests(False)
    assert first is worker._compiled_requests(False)
    ids = torch.tensor([9], dtype=torch.int64)
    worker._layerwise_transfer_impl(ids, ids, None, None, counter_id=0)
    assert all(req.gpu_block_id_tensor is ids for req in first.requests)


def test_nonlocal_layer_with_local_members_fails_before_submit(worker):
    worker._pools[PoolId.FULL_KV].layer_members[0] = [(0, 0)]
    ids = torch.tensor([1], dtype=torch.int64)
    with pytest.raises(ValueError, match="Nonlocal CP"):
        worker._transfer_impl(ids, ids, TransferType.H2D)
    worker.region_batch.submit.assert_not_called()


def test_transfer_engine_passes_local_holder_map_to_worker(monkeypatch):
    engine = TransferEngine.__new__(TransferEngine)
    engine.model_config = SimpleNamespace(
        effective_tp_size_per_node=2, h2d_layer_holders=[-1, -1, 0, 1],
        layer_shard_enabled=True,
    )
    engine._cpu_handle = SimpleNamespace(get_worker_tensor=lambda: object(), kv_layout=object())
    engine.mp_ctx = object()
    engine.finished_ops_queue = object()
    engine.pin_buffer = SimpleNamespace(get_buffer=lambda: object())
    engine._compressors = {"gpu_cpu": object()}
    engine._get_multi_group_kwargs = lambda key: {}
    engine._get_swa_kwargs = lambda key: {}
    create = MagicMock(return_value=object())
    monkeypatch.setattr(GPUCPUTransferWorker, "create_worker", create)
    handle = SimpleNamespace(get_tensor_handle_list=lambda: [], kv_layout=object(), dtype=torch.uint8)
    engine._create_gpu_cpu_worker("key", [handle, handle])
    assert create.call_args.kwargs["h2d_layer_holders"] == [-1, -1, 0, 1]
    assert create.call_args.kwargs["layer_shard_enabled"] is True


def test_rejected_server_launch_fails_only_unsubmitted_tasks():
    from types import SimpleNamespace
    from flexkv.kvtask import KVTaskEngine, TaskStatus
    engine = KVTaskEngine.__new__(KVTaskEngine)
    failed = []
    engine._fail_task = failed.append
    engine.tasks = {
        1: SimpleNamespace(status=TaskStatus.READY, graph=object(), transfer_failed=False),
        2: SimpleNamespace(status=TaskStatus.RUNNING, graph=object(), transfer_failed=False),
    }
    engine.reject_launch([1, 2, 3], -1, "bad slot mapping")
    assert failed == [1]                          # RUNNING keeps its real terminal
    assert engine.tasks[1].transfer_failed and not engine.tasks[2].transfer_failed
    failed.clear()
    engine.reject_launch([1], 99, "merge failed")  # fresh batch: children failed, batch NOTFOUND
    assert failed == [1] and 99 not in engine.tasks


def test_server_launch_handler_reports_rejection():
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "flexkv/server/server.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.FunctionDef) and n.name == "_handle_launch_task_request")
    text = ast.unparse(fn)
    assert "reject_launch(list(req.task_ids), req.batch_id, str(exc))" in text
    assert text.rstrip().endswith("raise")
