"""Execute the draft registration's actual CUDA page views through native IO."""

from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.integration.sglang.draft_cache import build_draft_cache_plan, standard_target_groups
from flexkv.transfer.region_batch import build_region_batch, make_requests
from test_layerwise_multi_group_swa import _compute_multi_group_strides, _main_region_specs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["mla_mixed_dtype", "deepseek_packed_swa", "deepseek_sidecar_swa"])
@pytest.mark.parametrize("use_ce", [False, True])
def test_registered_draft_members_survive_native_d2h_zero_layerwise_h2d(kind, use_ce):
    torch.cuda.set_device(0)
    page, blocks, width = 16, 8, 32

    def mla(layers, dtype):
        return NS(
            layer_num=layers,
            kv_buffer=[torch.zeros(blocks * page, 1, width, dtype=dtype, device="cuda") for _ in range(layers)],
        )

    target = mla(2, torch.uint8)
    draft = mla(1, torch.bfloat16)
    groups = standard_target_groups(target, page, 2)
    if kind == "mla_mixed_dtype":
        target.index_k_with_scale_buffer = [torch.zeros(blocks, 64, dtype=torch.uint8, device="cuda") for _ in range(2)]
        draft.index_k_with_scale_buffer = [torch.zeros(blocks, 64, dtype=torch.uint8, device="cuda")]
        groups = standard_target_groups(target, page, 2)
    else:
        mapping = torch.arange(blocks * page, device="cuda")
        target.swa_size = draft.swa_size = blocks * page
        target.full_to_swa_index_mapping = draft.full_to_swa_index_mapping = mapping
        target.swa_kv_pool = NS(
            page_size=page,
            kv_buffer=[torch.zeros(blocks, page * width, dtype=torch.uint8, device="cuda") for _ in range(2)],
        )
        draft = NS(
            layer_num=3,
            compression_ratios=[0, 0, 0],
            swa_size=blocks * page,
            full_to_swa_index_mapping=mapping,
            swa_kv_pool=NS(
                page_size=page,
                kv_buffer=[
                    torch.zeros(
                        blocks,
                        page * width,
                        dtype=torch.bfloat16 if kind == "deepseek_sidecar_swa" else torch.uint8,
                        device="cuda",
                    )
                    for _ in range(3)
                ],
            ),
        )
    plan = build_draft_cache_plan(
        target_pool=target,
        draft_pools=(draft,),
        target_layers=2,
        page_size=page,
        target_groups=groups,
        is_dsv4=kind.startswith("deepseek"),
        identity={},
    )
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._draft_cache_plan = plan
    connector._dsv4_layer_groups = plan.full_groups
    connector._dsv4_state_groups = []
    connector._swa_kv_pool = getattr(target, "swa_kv_pool", None)
    connector.rank_info = NS(num_layers_per_pp_stage=plan.transfer_layers)
    connector.model_config = NS(kv_dim=1, num_kv_heads=1)
    connector.page_size = page
    connector.tp_client = Mock()
    connector._label = "draft native fixture"
    full_handles = [b for g in plan.full_groups for b in g["buffers"]]
    connector._register_dsv4_to_server(full_handles)
    registration = connector.tp_client.register_to_server.call_args.kwargs
    descriptions = [
        ("full", registration["layer_groups"], registration["gpu_layouts"], registration["handles_per_group"], 2, 5)
    ]
    if registration["swa_caches"]:
        descriptions.append(
            (
                "swa",
                registration["swa_layer_groups"],
                registration["swa_gpu_layouts"],
                registration["swa_handles_per_group"],
                3,
                6,
            )
        )
    regions, keepalive, expected, rows = [], [], [], []
    for label, specs, layouts, handles, host_slot, destination in descriptions:
        cpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.BLOCKFIRST,
            num_layer=plan.transfer_layers,
            num_block=blocks,
            tokens_per_block=page,
            num_head=1,
            head_size=width,
            kv_dim=1,
            num_kv_heads=1,
            layer_groups=specs,
            tp_size=1,
        )
        cpu = torch.zeros(blocks, cpu_layout.get_block_stride(), dtype=torch.uint8, pin_memory=True)
        keepalive.append(cpu)
        strides = _compute_multi_group_strides(specs, cpu_layout, layouts)
        new_regions = _main_region_specs(specs, strides, cpu, handles)
        start = len(regions)
        regions.extend(replace(region, name=f"{label}.group{i}") for i, region in enumerate(new_regions))
        for gi, spec in enumerate(specs):
            for local, original in enumerate(spec.layer_indices):
                tensor = handles[gi][local]
                tensor.fill_(7 + len(expected))
                if tensor.is_floating_point():
                    tensor.add_(0.125)
                expected.append((tensor, tensor[1].cpu().clone(), destination))
                rows.append((start + gi, local, original, host_slot, destination))
    native = build_region_batch(regions, [0], is_blockfirst=True, num_kv_heads=1)
    torch.cuda.synchronize()
    down = []
    for region_index, (label, specs, layouts, handles, host_slot, destination) in zip(
        [0] + ([len(descriptions[0][1])] if len(descriptions) > 1 else []), descriptions
    ):
        down.extend(
            make_requests(
                native.num_regions,
                torch.tensor([1]).pin_memory(),
                torch.tensor([host_slot]).pin_memory(),
                False,
                transfer_num_cta=4,
                use_ce_transfer=use_ce,
                region_indices=list(range(region_index, region_index + len(specs))),
            )
        )
    native.submit(down, True)
    for tensor, _, _ in expected:
        tensor.zero_()
    torch.cuda.synchronize()
    up = []
    for region, local, original, host_slot, destination in rows:
        request = make_requests(
            native.num_regions,
            torch.tensor([destination]).pin_memory(),
            torch.tensor([host_slot]).pin_memory(),
            True,
            transfer_num_cta=4,
            use_ce_transfer=use_ce,
            region_indices=[region],
            layer_id=local,
            layer_granularity=1,
        )[0]
        request.milestone_layer = original
        up.append(request)
    covered = {row[2] for row in rows}
    empty = [layer for layer in range(plan.transfer_layers) if layer not in covered]
    native.submit_layerwise(up, empty, 0)
    ok, error = native.wait_layer_completion(120.0)
    assert ok, error
    torch.cuda.synchronize()
    for tensor, golden, destination in expected:
        assert torch.equal(tensor[destination].cpu(), golden)
        assert bool((tensor[1] == 0).all()), "old GPU source must remain zero"
