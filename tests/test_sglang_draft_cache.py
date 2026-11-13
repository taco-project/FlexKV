"""Draft members must survive slot reuse with their own physical dtype/layout."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "flexkv_draft_plan_under_test", Path(__file__).parents[1] / "flexkv/integration/sglang/draft_cache.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def mla(layers, dtype=torch.uint8, pages=4):
    return NS(
        layer_num=layers,
        kv_buffer=[torch.arange(pages * 4 * 8).reshape(pages * 4, 1, 8).to(dtype) + i for i in range(layers)],
    )


def plan(target, draft, identity=None):
    return module.build_draft_cache_plan(
        target_pool=target,
        draft_pools=(draft,),
        target_layers=target.layer_num,
        page_size=4,
        target_groups=module.standard_target_groups(target, 4, target.layer_num),
        is_dsv4=False,
        identity=identity or {"model": "fixture", "algorithm": "EAGLE"},
    )


@pytest.mark.parametrize("draft_dtype,mode", [(torch.uint8, "packed"), (torch.bfloat16, "sidecar")])
def test_target_and_draft_roundtrip_after_reusing_different_slots(draft_dtype, mode):
    target, draft = mla(2), mla(1, draft_dtype)
    p = plan(target, draft)
    assert p.modes == [mode]
    source, destination = torch.tensor([2, 0]), torch.tensor([1, 3])
    buffers = [b for g in p.full_groups for b in g["buffers"]]
    saved = [b[source].clone() for b in buffers]
    for b in buffers:
        b.fill_(37)  # a different request took every old device slot
    for b, host in zip(buffers, saved):
        b[destination] = host
    for b, host in zip(buffers, saved):
        assert torch.equal(b[destination], host)
        assert bool((b[source] == 37).all())
    assert buffers[-1].dtype == draft_dtype
    assert p.transfer_layers == 3


def test_layout_identity_distinguishes_dtype_and_draft_model():
    target = mla(2)
    first = plan(target, mla(1))
    assert first.fingerprint != plan(target, mla(1, torch.bfloat16)).fingerprint
    # Buffer capacity is not part of the serialized page's identity.
    assert first.fingerprint == plan(mla(2, pages=8), mla(1, pages=8)).fingerprint
    assert plan(target, mla(1), {"model": "one"}).fingerprint != plan(target, mla(1), {"model": "two"}).fingerprint
    assert plan(target, mla(1), {"tp_size": 4}).fingerprint != plan(target, mla(1), {"tp_size": 8}).fingerprint
    draft = mla(1)
    draft.start_layer = 78
    first_stage = plan(target, draft).fingerprint
    draft.start_layer = 79
    assert first_stage != plan(target, draft).fingerprint


def test_fully_skipped_indexer_is_not_a_transfer_group():
    target, draft = mla(2), mla(1)
    target.index_k_with_scale_buffer = [torch.empty(0, 8) for _ in range(2)]
    draft.index_k_with_scale_buffer = [torch.empty(0, 8)]
    p = plan(target, draft)
    assert len(p.full_groups) == 1
    assert p.full_groups[0]["layer_ids"] == [0, 1, 2]
    assert p.modes == ["packed", "empty"]


def test_sidecar_cannot_use_a_smaller_physical_slot_space():
    with pytest.raises(ValueError, match="physical page space"):
        plan(mla(2), mla(1, pages=2))
    # A larger arena can safely address every target slot, but does not join
    # the target's same-capacity group.
    assert plan(mla(2), mla(1, pages=8)).modes == ["sidecar"]


@pytest.mark.parametrize("draft_dtype,mode", [(torch.uint8, "packed"), (torch.bfloat16, "sidecar")])
def test_deepseek_swa_keeps_three_draft_layers_with_target_mapping(draft_dtype, mode):
    mapping = torch.arange(16)
    target = NS(
        layer_num=2,
        swa_size=16,
        full_to_swa_index_mapping=mapping,
        swa_kv_pool=NS(page_size=4, kv_buffer=[torch.zeros(4, 32, dtype=torch.uint8) for _ in range(2)]),
    )
    draft = NS(
        layer_num=3,
        compression_ratios=[0, 0, 0],
        swa_size=16,
        full_to_swa_index_mapping=mapping,
        swa_kv_pool=NS(page_size=4, kv_buffer=[torch.ones(4, 32, dtype=draft_dtype) for _ in range(3)]),
    )
    groups = module.standard_target_groups(mla(2), 4, 2)
    p = module.build_draft_cache_plan(
        target_pool=target,
        draft_pools=(draft,),
        target_layers=2,
        page_size=4,
        target_groups=groups,
        is_dsv4=True,
        identity={},
    )
    assert [layer for group in p.swa_groups for layer in group["layer_ids"]] == [0, 1, 2, 3, 4]
    assert sum(len(group["buffers"]) for group in p.swa_groups) == 5
    assert p.modes == [mode]
    assert p.transfer_layers == 5
    draft.full_to_swa_index_mapping = mapping.flip(0)
    with pytest.raises(ValueError, match="mapping"):
        module.build_draft_cache_plan(
            target_pool=target,
            draft_pools=(draft,),
            target_layers=2,
            page_size=4,
            target_groups=groups,
            is_dsv4=True,
            identity={},
        )


@pytest.mark.parametrize("mapping_kind", ["target_missing", "draft_missing", "dtype_alias", "shape_alias"])
def test_swa_mapping_contract_rejects_missing_or_reinterpreted_aliases(mapping_kind):
    source = torch.arange(16)
    target_map, draft_map = source, source
    if mapping_kind == "target_missing":
        target_map = None
    elif mapping_kind == "draft_missing":
        draft_map = None
    elif mapping_kind == "dtype_alias":
        draft_map = source.view(torch.int32)
    else:
        draft_map = source.view(4, 4)
    target = NS(
        layer_num=2,
        swa_size=16,
        full_to_swa_index_mapping=target_map,
        swa_kv_pool=NS(page_size=4, kv_buffer=[torch.zeros(4, 32, dtype=torch.uint8) for _ in range(2)]),
    )
    draft = NS(
        layer_num=1,
        compression_ratios=[0],
        swa_size=16,
        full_to_swa_index_mapping=draft_map,
        swa_kv_pool=NS(page_size=4, kv_buffer=[torch.ones(4, 32, dtype=torch.uint8)]),
    )
    with pytest.raises(ValueError, match="mapping"):
        module.build_draft_cache_plan(
            target_pool=target,
            draft_pools=(draft,),
            target_layers=2,
            page_size=4,
            target_groups=module.standard_target_groups(mla(2), 4, 2),
            is_dsv4=True,
            identity={},
        )


def test_request_ring_does_not_become_a_prefix_sidecar():
    target, draft = mla(2), mla(1)
    draft.swa_req_ring_size = 256
    with pytest.raises(NotImplementedError, match="request rings"):
        plan(target, draft)


def test_last_target_layer_fences_every_draft_depth():
    waits = []
    counter = NS(wait_until=waits.append)
    target = module.TargetDraftTransferCounter(counter, 3, 6)
    target.wait_until(0)
    assert waits == [0]
    target.wait_until(2)
    assert waits == [0, 2, 0, 1, 2, 3, 4, 5]
    module.DraftTransferCounter(counter, 6).wait_until(1)
    assert waits[-6:] == [0, 1, 2, 3, 4, 5]
