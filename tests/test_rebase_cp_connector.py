"""CP owner preflight, NEXTN sidecars and load ownership protocol tests."""

from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from flexkv.common.config import ModelConfig
from flexkv.integration.sglang.connector import FlexKVConnector, _PreparedLayerwiseLoad


def connector():
    c = FlexKVConnector.__new__(FlexKVConnector)
    c.page_size = 4
    c._is_dsv4 = False
    c._swa_kv_pool = None
    c._deduplicate_indexer_group = True
    c._indexer_layer_ids = [0, 2]
    c._draft_namespace = None
    c._sync_ctx = SimpleNamespace(
        needs_sync=False, is_sync_leader=True, world_rank=0,
        should_send_slot_mapping_to_remote=False, all_reduce_min=lambda value: value,
    )
    c.rank_info = SimpleNamespace(num_layers_per_pp_stage=4)
    c._log_cache_op = MagicMock()
    c.kv_manager = MagicMock()
    c._pending_lookups = {}
    c._pending_lookup_contexts = {}
    c._launched_load_tids = []
    c._launched_load_contexts = {}
    c._prepared_layerwise_loads = {}
    c._layerwise_batch_launch_error = None
    c._inflight_stores = {}
    c._inflight_store_contexts = {}
    c._prefetch_sessions = {}
    c._prefetch_result_sessions = {}
    c._chunked_prefetch = False
    c._prefetch_enabled = True
    c._prefetch_contexts = {}
    c._ongoing_prefetches = {}
    c._prefetch_planned_tokens = {}
    c._prefetch_loaded_tokens = {}
    return c


def physical_pool(rank=0, page_major=False):
    shape = (3, 1, 4, 8) if page_major else (12, 1, 8)
    empty = (0, *shape[1:])
    owners = [0, 0, 1, 1]
    return SimpleNamespace(
        layer_shard_enabled=True, layer_shard_size=2, layer_shard_rank=rank,
        start_layer=0, layer_num=4, _get_layer_owner_rank=lambda layer: owners[layer],
        kv_buffer=[torch.empty(shape if owner == rank else empty, dtype=torch.uint8)
                   for owner in owners],
        remote_kv_buffer=torch.empty(shape, dtype=torch.uint8),
        skip_topk_layers=[False, True, False, True],
        index_k_with_scale_buffer=[
            torch.empty((3 if rank == 0 else 0, 16), dtype=torch.uint8), None,
            torch.empty((3 if rank == 1 else 0, 16), dtype=torch.uint8), None,
        ],
        remote_index_k_with_scale_buffer=torch.empty((3, 16), dtype=torch.uint8),
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("page_major", [False, True])
def test_cp_preflight_resolves_only_nonowners_to_scratch(rank, page_major):
    c = connector()
    pool = physical_pool(rank, page_major)
    main, index = c._resolve_cp_kv_buffers(pool)
    assert index.layer_indices == (0, 2)
    assert index.logical_layer_count == 4
    for layer, tensor in enumerate(main):
        expected = pool.kv_buffer[layer] if layer // 2 == rank else pool.remote_kv_buffer
        assert tensor is expected
    assert index.buffers[rank] is pool.index_k_with_scale_buffer[rank * 2]
    assert index.buffers[1 - rank] is pool.remote_index_k_with_scale_buffer


@pytest.mark.parametrize("fault", ["missing_owned", "live_nonowner", "bad_pages", "scratch_alias", "bad_ids"])
def test_cp_indexer_rejects_invalid_original_buffers(fault):
    c, pool = connector(), physical_pool()
    if fault == "missing_owned":
        pool.index_k_with_scale_buffer[0] = torch.empty((0, 16), dtype=torch.uint8)
    elif fault == "live_nonowner":
        pool.index_k_with_scale_buffer[2] = torch.empty((3, 16), dtype=torch.uint8)
    elif fault == "bad_pages":
        pool.remote_index_k_with_scale_buffer = torch.empty((2, 16), dtype=torch.uint8)
    elif fault == "scratch_alias":
        pool.index_k_with_scale_buffer[0] = pool.remote_index_k_with_scale_buffer
    else:
        c._indexer_layer_ids = [0, 1]
    with pytest.raises(ValueError):
        c._resolve_cp_kv_buffers(pool)


def test_cp_nondedup_keeps_logical_indexer_numbering():
    c, pool = connector(), physical_pool()
    c._deduplicate_indexer_group = False
    _, index = c._resolve_cp_kv_buffers(pool)
    assert index.layer_indices == (0, 1, 2, 3)
    assert len(index.buffers) == 4


def test_geometry_disagreement_is_rejected():
    c = connector()
    c._sync_ctx.all_reduce_min = MagicMock(side_effect=[10, -11])
    with pytest.raises(ValueError, match="differs across"):
        c._agree_cache_geometry([[3, 16], "torch.uint8"], "Indexer")


def test_peer_preflight_failure_prevents_registration():
    c = connector()
    c._sync_ctx.all_reduce_min = lambda value: 0
    with pytest.raises(ValueError, match="another cache rank"):
        c._require_cache_preflight(None, "GPU buffers")
    c.kv_manager.start.assert_not_called()


def test_nextn_kv_and_indexer_have_one_host_row_per_owner():
    c = connector()
    model = ModelConfig(num_layers=4, num_kv_heads=1, kv_dim=1, cp_size=2,
                        h2d_layer_holders=[0, 0, 1, 1], layer_shard_enabled=True)
    model.freeze()
    c.model_config = model
    draft = SimpleNamespace(layer_num=1, layer_shard_enabled=False,
                            kv_buffer=[torch.empty((12, 1, 8), dtype=torch.bfloat16)],
                            index_k_with_scale_buffer=[torch.empty((3, 16), dtype=torch.uint8)])
    groups = c._build_draft_groups(draft, physical_pool().kv_buffer)
    assert len(groups) == 2
    assert [entry[0].layer_indices for entry in groups] == [[0, 2], [0, 2]]
    assert [entry[0].compress_ratio for entry in groups] == [1, 4]
    assert [entry[1].tokens_per_block for entry in groups] == [4, 1]
    for _, _, handles in groups:
        assert len(handles) == 2 and handles[0] is handles[1]


def test_extra_key_and_layout_namespace_do_not_collide_on_nul():
    c = connector()
    ids = array("q", [1, 2, 3, 4])
    key = SimpleNamespace(token_ids=ids, extra_key="tenant\x00a")
    tokens, namespace = c._radix_key_input(key, ["base"])
    assert tokens is ids
    assert "\x00" not in namespace[0]
    assert namespace != c._radix_key_input(ids, ["base"], "tenant")[1]
    assert c._cache_namespace(namespace) is namespace
    c._draft_namespace = "layout-one"
    first = c._cache_namespace(namespace)
    c._draft_namespace = "layout-two"
    assert first != c._cache_namespace(namespace)


def test_lookup_no_writer_retires_terminal_mask():
    c = connector()
    c.kv_manager.get_match.return_value = (41, np.zeros(4, dtype=bool))
    tid, hit = c.lookup_kv([1, 2, 3, 4], torch.ones(4, dtype=torch.bool), rid="r")
    assert (tid, hit) == (41, 0)
    c.kv_manager.cancel.assert_called_once_with([41])
    assert not c._pending_lookups


def test_store_no_writer_retires_terminal_and_preserves_encoded_array():
    c = connector()
    c.kv_manager.put_match.return_value = (42, np.zeros(4, dtype=bool))
    assert c.store_kv("r", array("q", [1, 2, 3, 4]), range(4)) == -1
    c.kv_manager.cancel.assert_called_once_with([42])
    c.kv_manager.launch.assert_not_called()
    assert not c._inflight_stores


def test_partial_store_launch_failure_keeps_task_drainable():
    c = connector()
    c.kv_manager.put_match.return_value = (43, np.ones(4, dtype=bool))
    c.kv_manager.launch.side_effect = RuntimeError("partial D2H")
    with pytest.raises(RuntimeError, match="partial D2H"):
        c.store_kv("r", [1, 2, 3, 4], torch.arange(4))
    assert c._inflight_stores["r"] == 43


def test_sync_restore_waits_for_actual_batch_whole_completion():
    c = connector()
    c._pending_lookups["r"] = 1
    c.kv_manager.launch.return_value = [99]
    c.kv_manager.wait.return_value = {99: SimpleNamespace(status="success")}
    assert c.retrieve_kv("r", torch.arange(4)) == 4
    c.kv_manager.wait.assert_called_once_with([99], timeout=30.0, completely=True)
    assert c.kv_manager.launch.call_args.kwargs["layerwise_transfer"] is False
    assert c.kv_manager.launch.call_args.kwargs["as_batch"] is False
    assert not c._launched_load_tids


def test_sync_restore_timeout_keeps_actual_batch_id():
    c = connector()
    c._pending_lookups["r"] = 1
    c.kv_manager.launch.return_value = [99]
    c.kv_manager.wait.return_value = {}
    with pytest.raises(RuntimeError, match="not safely restored"):
        c.retrieve_kv("r", torch.arange(4))
    assert c._launched_load_tids == [99]


def test_sync_follower_retains_unknown_submission_reset_guard():
    c = connector()
    c._pending_lookups["r"] = 1
    c._sync_ctx.is_sync_leader = False
    c._sync_ctx.needs_sync = True
    c._sync_ctx.scatter = lambda payload, **kwargs: {
        "error": "send outcome unknown", "loaded": 0, "unresolved": True,
    }
    with pytest.raises(RuntimeError, match="send outcome unknown"):
        c.retrieve_kv("r", torch.arange(4))
    with pytest.raises(RuntimeError, match="unacknowledged"):
        c.reset()


@pytest.mark.parametrize("status", ["cancelled", "not_found", "pending"])
def test_load_drain_does_not_release_unproved_quiescence(status):
    c = connector()
    c._launched_load_tids = [99]
    c.kv_manager.wait.return_value = {99: SimpleNamespace(status=status)}
    c.drain_launched_loads()
    assert c._launched_load_tids == [99]
    c.kv_manager.wait.assert_called_once_with([99], timeout=0.0, completely=True)


def test_failed_drained_load_raises_and_retires_terminal():
    c = connector()
    c._launched_load_tids = [99]
    c.kv_manager.wait.return_value = {99: SimpleNamespace(status="failed")}
    with pytest.raises(RuntimeError, match="restore failed"):
        c.drain_launched_loads()
    assert not c._launched_load_tids


def test_deferred_batch_partial_launch_retains_actual_graph_id_and_destinations():
    c = connector()
    context = c._new_op_context("load", "r", task_id=1)
    mapping = torch.arange(4)
    c._prepared_layerwise_loads["r"] = _PreparedLayerwiseLoad(
        1, 0, mapping, None, context, 0,
    )
    c.layer_done_counter = MagicMock()
    error = RuntimeError("partial batch")
    error.launched_task_ids = [99]
    c.kv_manager.launch.side_effect = error
    with pytest.raises(RuntimeError, match="partial batch"):
        c.flush_layerwise_loads(["r"])
    assert c._launched_load_tids == [99]
    assert c._prepared_layerwise_loads["r"].slot_mapping is mapping
    with pytest.raises(RuntimeError, match="unacknowledged"):
        c.reset()


def test_optional_window_receives_namespace_exactly_once():
    c = connector()
    c._draft_namespace = "layout"
    c._l3_prefetch_window = MagicMock()
    c._l3_prefetch_window.start.return_value = 77
    tokens = [1, 2, 3, 4]
    assert c.prefetch_async("r", tokens, namespace=["tenant"], extra_key="adapter") == 77
    _, namespace = c._radix_key_input(tokens, ["tenant"], "adapter")
    args = c._l3_prefetch_window.start.call_args.args
    assert args[0] == "r" and args[1] == tokens
    assert args[3] == c._cache_namespace(namespace)
    c.kv_manager.prefetch_async.assert_not_called()


def test_optional_window_progress_and_cancel_do_not_enter_legacy_path():
    c = connector()
    c._l3_prefetch_window = MagicMock()
    c._l3_prefetch_window.check.return_value = True
    c._l3_prefetch_window.pop_loaded.return_value = 12
    c.release_pending = MagicMock()
    assert c.check_prefetch_progress("r") is True
    assert c.pop_prefetch_loaded_tokens("r") == 12
    c.cancel_prefetch("r")
    c._l3_prefetch_window.cancel.assert_called_once_with("r")
    c.release_pending.assert_called_once_with("r")
    c.kv_manager.try_wait.assert_not_called()


def test_optional_window_idle_uses_noncollective_entrypoint():
    c = connector()
    c._l3_prefetch_window = MagicMock()
    c.drain_launched_loads()
    c._l3_prefetch_window.poll_idle.assert_called_once_with()
    c._l3_prefetch_window.poll.assert_not_called()
    c._sync_ctx.is_sync_leader = False
    c.drain_launched_loads()
    assert c._l3_prefetch_window.poll_idle.call_count == 1


def test_window_shutdown_failure_preserves_manager_and_retryability():
    c = connector()
    c._l3_prefetch_window = MagicMock()
    c._l3_prefetch_window.drain.side_effect = RuntimeError("pending DMA")
    with pytest.raises(RuntimeError, match="pending DMA"):
        c.shutdown()
    c._l3_prefetch_window.drain.assert_called_once_with(collective=False)
    c.kv_manager.shutdown.assert_not_called()
    assert not getattr(c, "_shutdown_done", False)


def test_cache_validation_sglang_store_api_is_available():
    from flexkv.integration.sglang.connector import _IndexerBufferGroup, _is_sparse_indexer_group
    obj = FlexKVConnector.__new__(FlexKVConnector)
    obj._inflight_stores = {}
    assert not obj.has_inflight_stores() and obj.inflight_store_count() == 0
    obj._inflight_stores = {"rid": 7}
    assert obj.has_inflight_stores() and obj.inflight_store_count() == 1
    # main reports a store only after the whole graph drained.
    assert obj.has_pending_store_graphs() is False
    assert obj.progress_store_graphs() is False
    assert FlexKVConnector.sparse_indexer is False
    full = _IndexerBufferGroup(buffers=(1, 2), layer_indices=(0, 1), logical_layer_count=2)
    subset = _IndexerBufferGroup(buffers=(1,), layer_indices=(1,), logical_layer_count=2)
    assert not _is_sparse_indexer_group(None)
    assert not _is_sparse_indexer_group(full)
    assert _is_sparse_indexer_group(subset)
