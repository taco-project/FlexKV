"""Regression tests for GLM DSA page-packed indexer transfer geometry."""

from types import SimpleNamespace

import pytest
import torch

from flexkv.common.config import LayerGroupSpec
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.transfer.worker import _validate_multi_group_chunk_layout


PAGE_SIZE = 64
INDEX_HEAD_SIZE = 8448
MAIN_HEAD_SIZE = 576
NUM_LAYERS = 78


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

    specs = connector._build_indexer_layer_group_specs(
        [torch.empty((1, 1, MAIN_HEAD_SIZE), dtype=torch.bfloat16)],
        [torch.empty((1, INDEX_HEAD_SIZE), dtype=torch.uint8)] * NUM_LAYERS,
    )

    assert specs[1].head_size == INDEX_HEAD_SIZE
    assert specs[1].compress_ratio == PAGE_SIZE


def test_connector_aliases_skip_topk_zero_row_indexer_buffers():
    active0 = torch.empty((11, INDEX_HEAD_SIZE), dtype=torch.uint8)
    skipped = torch.empty((0, INDEX_HEAD_SIZE), dtype=torch.uint8)
    active2 = torch.empty((11, INDEX_HEAD_SIZE), dtype=torch.uint8)

    resolved = FlexKVConnector._alias_empty_indexer_buffers(
        [active0, skipped, active2]
    )

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
    assert main == 5_750_784
    assert indexer == 658_944
    assert layout.get_block_stride() == 6_409_728
    assert layout.kv_shape == torch.Size([1, 6_409_728])


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
