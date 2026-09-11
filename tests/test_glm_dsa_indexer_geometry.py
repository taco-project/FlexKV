"""Regression tests for GLM DSA page-packed indexer transfer geometry."""

from types import SimpleNamespace

import pytest
import torch

from flexkv.common.config import LayerGroupSpec
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.server.client import KVTPClient
from flexkv.transfer.worker import _validate_multi_group_chunk_layout


PAGE_SIZE = 64
INDEX_HEAD_SIZE = 8448
MAIN_HEAD_SIZE = 576
NUM_LAYERS = 8
# Synthetic sparse membership; independent of any model architecture.
ACTIVE_INDEXER_LAYERS = [0, 3, 7]


def _groups(compress_ratio: int):
    return [
        LayerGroupSpec(
            num_layers=NUM_LAYERS,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=list(range(NUM_LAYERS)),
            compress_ratio=1,
            dtype=torch.bfloat16,
        ),
        LayerGroupSpec(
            num_layers=NUM_LAYERS,
            num_kv_heads=1,
            head_size=INDEX_HEAD_SIZE,
            layer_indices=list(range(NUM_LAYERS)),
            compress_ratio=compress_ratio,
            dtype=torch.uint8,
        ),
    ]


def test_connector_describes_one_indexer_row_per_page():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = PAGE_SIZE
    connector.rank_info = SimpleNamespace(num_layers_per_pp_stage=NUM_LAYERS)

    indexer_buffers = [
        torch.empty((1, INDEX_HEAD_SIZE), dtype=torch.uint8)
        for _ in range(NUM_LAYERS)
    ]
    group = connector._compact_indexer_buffers(
        indexer_buffers,
        [False] * NUM_LAYERS,
        layer_shard_enabled=False,
    )
    specs = connector._build_indexer_layer_group_specs(
        [torch.empty((1, 1, MAIN_HEAD_SIZE), dtype=torch.bfloat16)], group
    )

    assert specs[1].head_size == INDEX_HEAD_SIZE
    assert specs[1].compress_ratio == PAGE_SIZE


def test_connector_compacts_skip_topk_indexer_group_for_ordinary_cp():
    skip_mask = [i not in ACTIVE_INDEXER_LAYERS for i in range(NUM_LAYERS)]
    buffers = [
        torch.empty(
            (0 if skip_mask[i] else 11, INDEX_HEAD_SIZE), dtype=torch.uint8
        )
        for i in range(NUM_LAYERS)
    ]

    group = FlexKVConnector._compact_indexer_buffers(
        buffers,
        skip_mask,
        # Ordinary CP keeps a complete cache pool on every CP rank.
        layer_shard_enabled=False,
    )

    assert group.logical_layer_count == NUM_LAYERS
    assert list(group.layer_indices) == ACTIVE_INDEXER_LAYERS
    assert len(group.buffers) == len(ACTIVE_INDEXER_LAYERS)
    assert all(buffer.shape == (11, INDEX_HEAD_SIZE) for buffer in group.buffers)


@pytest.mark.parametrize("dedup", [False, True])
def test_connector_resolves_only_the_selected_indexer_layout(dedup):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._is_dsv4 = False
    connector._deduplicate_indexer_group = dedup
    buffers = [
        torch.empty((11, INDEX_HEAD_SIZE), dtype=torch.uint8),
        torch.empty((0, INDEX_HEAD_SIZE), dtype=torch.uint8),
        torch.empty((11, INDEX_HEAD_SIZE), dtype=torch.uint8),
    ]
    main = torch.empty((11, 1, MAIN_HEAD_SIZE), dtype=torch.bfloat16)
    cache = SimpleNamespace(
        kv_buffer=[main] * 3,
        index_k_with_scale_buffer=buffers,
        skip_topk_layers=[False, True, False],
        layer_shard_enabled=False,
    )
    resolved, group = connector._resolve_kv_buffers(cache)
    assert resolved[0] is main
    assert group.logical_layer_count == 3
    assert group.layer_indices == ((0, 2) if dedup else (0, 1, 2))
    assert len(group.buffers) == (2 if dedup else 3)
    if not dedup:
        assert group.buffers[0] is group.buffers[1]
    # Resolution must not replace the pool's skip-layer placeholders.
    assert cache.index_k_with_scale_buffer[1].shape[0] == 0


def test_connector_rejects_indexer_dedup_with_cp_layer_split():
    with pytest.raises(RuntimeError, match="CP DSA cache layer split"):
        FlexKVConnector._compact_indexer_buffers(
            [torch.empty((1, INDEX_HEAD_SIZE), dtype=torch.uint8)],
            [False],
            layer_shard_enabled=True,
        )


def test_connector_compact_group_keeps_original_layer_member_ids():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = PAGE_SIZE
    connector.rank_info = SimpleNamespace(num_layers_per_pp_stage=3)
    active0 = torch.empty((11, INDEX_HEAD_SIZE), dtype=torch.uint8)
    skipped = torch.empty((0, INDEX_HEAD_SIZE), dtype=torch.uint8)
    active2 = torch.empty((11, INDEX_HEAD_SIZE), dtype=torch.uint8)
    group = connector._compact_indexer_buffers(
        [active0, skipped, active2],
        [False, True, False],
        layer_shard_enabled=False,
    )

    specs = connector._build_indexer_layer_group_specs(
        [torch.empty((1, 1, MAIN_HEAD_SIZE), dtype=torch.bfloat16)], group
    )

    assert specs[0].layer_indices == [0, 1, 2]
    assert specs[1].num_layers == 2
    assert specs[1].layer_indices == [0, 2]


def test_tp_client_rejects_inconsistent_compact_group_registration():
    client = KVTPClient.__new__(KVTPClient)
    spec = LayerGroupSpec(
        num_layers=2,
        num_kv_heads=1,
        head_size=INDEX_HEAD_SIZE,
        layer_indices=[0, 2],
        compress_ratio=PAGE_SIZE,
        dtype=torch.uint8,
    )
    layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=1,
        num_block=11,
        tokens_per_block=1,
        num_head=1,
        head_size=INDEX_HEAD_SIZE,
        kv_dim=1,
        num_kv_heads=1,
    )

    with pytest.raises(ValueError, match="multi-group layer count mismatch"):
        client.register_to_server(
            kv_caches=[torch.empty((1, 1, MAIN_HEAD_SIZE))],
            kv_layout=layout,
            layer_groups=[spec],
            gpu_layouts=[layout],
            handles_per_group=[
                [torch.empty((11, INDEX_HEAD_SIZE), dtype=torch.uint8)]
            ],
        )


def test_connector_aliases_skip_topk_zero_row_indexer_buffers():
    active0 = torch.empty((11, INDEX_HEAD_SIZE), dtype=torch.uint8)
    skipped = torch.empty((0, INDEX_HEAD_SIZE), dtype=torch.uint8)
    active2 = torch.empty((11, INDEX_HEAD_SIZE), dtype=torch.uint8)

    resolved = FlexKVConnector._alias_empty_indexer_buffers([active0, skipped, active2])

    assert resolved == [active0, active0, active2]
    assert resolved[1].data_ptr() != 0


def test_connector_rejects_leading_skip_topk_placeholder():
    skipped = torch.empty((0, INDEX_HEAD_SIZE), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="leading skip-topk"):
        FlexKVConnector._alias_empty_indexer_buffers([skipped])

def test_corrected_glm_dsa_block_arithmetic():
    groups = _groups(PAGE_SIZE)
    layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=NUM_LAYERS,
        num_block=1,
        tokens_per_block=PAGE_SIZE,
        num_head=1,
        head_size=MAIN_HEAD_SIZE,
        # GLM combined-KV stores one KV stream per group in this pool.
        kv_dim=1,
        layer_groups=groups,
    )
    # GLM combined-KV has kv_dim=1; the final factor is BF16 bytes.
    main = NUM_LAYERS * PAGE_SIZE * MAIN_HEAD_SIZE * 2
    indexer = NUM_LAYERS * INDEX_HEAD_SIZE
    assert main == 589_824
    assert indexer == 67_584
    assert layout.get_block_stride() == 657_408
    assert layout.kv_shape == torch.Size([1, 657_408])


def test_compact_indexer_block_arithmetic():
    groups = [
        LayerGroupSpec(
            num_layers=NUM_LAYERS,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=list(range(NUM_LAYERS)),
            compress_ratio=1,
            dtype=torch.bfloat16,
        ),
        LayerGroupSpec(
            num_layers=len(ACTIVE_INDEXER_LAYERS),
            num_kv_heads=1,
            head_size=INDEX_HEAD_SIZE,
            layer_indices=ACTIVE_INDEXER_LAYERS,
            compress_ratio=PAGE_SIZE,
            dtype=torch.uint8,
        ),
    ]
    layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=NUM_LAYERS,
        num_block=1,
        tokens_per_block=PAGE_SIZE,
        num_head=1,
        head_size=MAIN_HEAD_SIZE,
        kv_dim=1,
        layer_groups=groups,
    )

    main = NUM_LAYERS * PAGE_SIZE * MAIN_HEAD_SIZE * 2
    compact_indexer = len(ACTIVE_INDEXER_LAYERS) * INDEX_HEAD_SIZE
    assert compact_indexer == 25_344
    assert layout.get_block_stride() == 615_168
    assert layout.kv_shape == torch.Size([1, 615_168])


def test_old_ratio_1_geometry_is_rejected_before_transfer():
    with pytest.raises(ValueError, match="group_chunk=540672.*layout_chunk=8448"):
        _validate_multi_group_chunk_layout(
            PAGE_SIZE * INDEX_HEAD_SIZE,
            INDEX_HEAD_SIZE,
            group_index=1,
            group_tpb=PAGE_SIZE,
            layout_tpb=1,
            head_size=INDEX_HEAD_SIZE,
            compress_ratio=1,
        )


def test_corrected_geometry_passes_fail_fast_check():
    _validate_multi_group_chunk_layout(
        INDEX_HEAD_SIZE,
        INDEX_HEAD_SIZE,
        group_index=1,
        group_tpb=1,
        layout_tpb=1,
        head_size=INDEX_HEAD_SIZE,
        compress_ratio=PAGE_SIZE,
    )
