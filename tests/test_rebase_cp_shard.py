"""Physical CP node layouts and framework-group control routing (CPU only)."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV, LayerGroupSpec, ModelConfig
from flexkv.common.cp_shard import cp_node_cache_model, slice_cp_gpu_groups
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.integration.sglang import comm as comm_module
from flexkv.integration.sglang.comm import FlexKVComm


@pytest.fixture
def cp_model(monkeypatch):
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "kv_shared_across_ranks_mode", "single_write")
    groups = [
        LayerGroupSpec(32, 1, 8, list(range(32)), dtype=torch.uint8),
        LayerGroupSpec(16, 1, 16, list(range(0, 32, 2)), dtype=torch.uint8, compress_ratio=4),
        LayerGroupSpec(16, 1, 8, list(range(16)), dtype=torch.bfloat16),
        LayerGroupSpec(16, 1, 16, list(range(16)), dtype=torch.uint8, compress_ratio=4),
    ]
    model = ModelConfig(
        num_layers=32, num_kv_heads=1, head_size=8, kv_dim=1,
        cp_size=16, attn_cp_size=16, nnodes=2,
        layer_groups=groups, layer_shard_enabled=True,
        h2d_layer_holders=list(range(16)) * 2,
    )
    model.freeze()
    return model


def test_local_registration_retains_storage_handle_protocol():
    from flexkv.common.cp_shard import slice_cp_storage_handle
    from flexkv.common.storage import AccessHandleType, StorageHandle
    layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST, num_layer=4, num_block=2,
        tokens_per_block=4, num_head=1, head_size=8, kv_dim=1, num_kv_heads=1)
    tensors = [torch.empty(2, 4, 8) for _ in range(4)]
    original = StorageHandle(AccessHandleType.TENSOR, tensors, layout, torch.float32,
                             gpu_device_id=7)
    local, groups, layouts = slice_cp_storage_handle(original, [tensors], [layout], [[1, 3]])
    assert isinstance(local, StorageHandle)
    assert local.dtype == original.dtype and local.gpu_device_id == 7
    assert local.get_tensor_list()[0] is tensors[1]
    assert local.get_tensor_list()[1] is tensors[3]
    assert local.kv_layout.num_layer == 2
    assert original.kv_layout.num_layer == 4 and len(original.data) == 4


def test_cp1_preserves_model_and_registration_identity():
    model = ModelConfig()
    model.freeze()
    local, selections = cp_node_cache_model(model, 0)
    assert local is model and selections is None
    blocks, layouts = [[object()]], [object()]
    result = slice_cp_gpu_groups(blocks, layouts, selections)
    assert result[0] is blocks and result[1] is layouts


@pytest.mark.parametrize("node", [0, 1])
def test_cp16_packs_only_physical_node_layers(cp_model, node):
    local, selections = cp_node_cache_model(cp_model, node)
    first = node * 8
    assert local.cp_cache_node_rank == node
    assert [g.num_layers for g in local.layer_groups] == [16, 8, 8, 8]
    for layer, owner in enumerate(cp_model.h2d_layer_holders):
        expected = owner - first if first <= owner < first + 8 else -1
        assert local.h2d_layer_holders[layer] == expected
        assert bool(local.layer_member_map.members[layer]) == (expected >= 0)
    for global_group, local_group, indices in zip(cp_model.layer_groups, local.layer_groups, selections):
        assert local_group.layer_indices == [global_group.layer_indices[i] for i in indices]
    assert cp_model.cp_cache_node_rank is None
    assert cp_model.layer_groups[0].num_layers == 32
    with pytest.raises(ValueError, match="already"):
        cp_node_cache_model(local, node)


@pytest.mark.parametrize("node", [-1, 2, True])
def test_invalid_node_rejected(cp_model, node):
    with pytest.raises(ValueError):
        cp_node_cache_model(cp_model, node)


def test_invalid_owner_and_all_write_rejected(cp_model, monkeypatch):
    with pytest.raises(ValueError, match="holder"):
        cp_node_cache_model(replace(cp_model, h2d_layer_holders=[16] * 32), 0)
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "kv_shared_across_ranks_mode", "all_write")
    with pytest.raises(ValueError):
        cp_node_cache_model(cp_model, 0)


def test_node_local_handle_order_matches_host_groups(cp_model):
    local, selections = cp_node_cache_model(cp_model, 1)
    layouts = [KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST, num_layer=g.num_layers,
        num_block=3, tokens_per_block=4 // g.compress_ratio,
        num_head=1, head_size=g.head_size, kv_dim=1, num_kv_heads=1,
    ) for g in cp_model.layer_groups]
    blocks = [[object() for _ in range(g.num_layers)] for g in cp_model.layer_groups]
    sliced_blocks, sliced_layouts = slice_cp_gpu_groups(blocks, layouts, selections)
    for group, old_layout, new_layout, indices, spec in zip(
        sliced_blocks, layouts, sliced_layouts, selections, local.layer_groups
    ):
        assert len(group) == new_layout.num_layer == spec.num_layers
        assert old_layout is not new_layout
        assert old_layout.num_layer >= new_layout.num_layer
    assert all(sliced_blocks[gi][j] is blocks[gi][index]
               for gi, indices in enumerate(selections) for j, index in enumerate(indices))
    for invalid in ([], [-1], [32], [0, 0], [1, 0]):
        with pytest.raises(ValueError, match="selection"):
            slice_cp_gpu_groups(blocks, layouts, [invalid, *selections[1:]])


class Group:
    def __init__(self, ranks):
        self.ranks = ranks

    def size(self):
        return len(self.ranks)


def rank_info(pp=1, tp=2, cp=2, pp_rank=0, tp_rank=0, cp_rank=0):
    return SimpleNamespace(
        model_config=SimpleNamespace(pp_size=pp, attn_tp_size=tp, attn_cp_size=cp),
        pp_rank=pp_rank, attn_tp_rank=tp_rank, attn_cp_rank=cp_rank,
        pp_size_per_node=1,
    )


def test_dp_replica_routes_only_its_actual_ranks(monkeypatch):
    monkeypatch.setattr(comm_module.dist, "get_global_rank", lambda group, i: group.ranks[i])
    c = FlexKVComm(rank_info(), 10, attn_tp_group=Group([10, 12]),
                   attn_cp_group=Group([10, 14]), world_group=object())
    assert c._tp_group_ranks == [10, 12]
    assert c._cp_leader_ranks == [10, 14]
    follower = FlexKVComm(rank_info(tp_rank=1), 12, attn_tp_group=Group([10, 12]),
                          attn_cp_group=Group([12, 16]), world_group=object())
    assert follower._cp_leader_ranks == []
    assert follower._tp_group_ranks == [10, 12]


def test_noncontiguous_pp_stage_members_and_leader(monkeypatch):
    monkeypatch.setattr(comm_module.dist, "get_global_rank", lambda group, i: group.ranks[i])

    def gather(cliques, members, group):
        cliques[:] = [[20, 22], [24, 26]]

    monkeypatch.setattr(comm_module.dist, "all_gather_object", gather)
    c = FlexKVComm(rank_info(pp=2, pp_rank=1, tp_rank=1, cp_rank=1), 26,
                   pp_group=Group([16, 26]), attn_tp_group=Group([24, 26]),
                   attn_cp_group=Group([22, 26]), world_group=object())
    assert c._pp_stage_member_ranks == [20, 22, 24, 26]
    assert c._pp_stage_leader_rank == 20
    c._recv_tensor = MagicMock()
    c._bcast_to_stage_members(torch.zeros(1, dtype=torch.int64), 123)
    assert c._recv_tensor.call_args.kwargs["src"] == 20


def test_disabled_collectives_do_not_resolve_groups(monkeypatch):
    resolve = MagicMock(side_effect=AssertionError("must not resolve world group"))
    monkeypatch.setattr(FlexKVComm, "_resolve_world_cpu_group", resolve)
    c = FlexKVComm(rank_info(pp=2), 10, enable_collective_sync=False)
    assert c.all_reduce_min(7) == 7
    c.barrier()
    resolve.assert_not_called()


def test_framework_rank_mismatch_fails_before_control_traffic(monkeypatch):
    monkeypatch.setattr(comm_module.dist, "get_global_rank", lambda group, i: group.ranks[i])
    with pytest.raises(RuntimeError, match="group/rank mismatch"):
        FlexKVComm(rank_info(tp_rank=1), 10, attn_tp_group=Group([10, 12]),
                   attn_cp_group=Group([10, 14]), world_group=object())
