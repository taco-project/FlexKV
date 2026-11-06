"""V4.1 ratio-1/2 KV + indexer registration helpers."""

from types import SimpleNamespace

import pytest
import torch

from flexkv.integration.sglang.connector import (
    FlexKVConnector,
    _pack_dsv4_indexer_full_pages,
)


def test_pack_indexer_noop_when_page_matches_full_page():
    buf = torch.arange(8 * 64, dtype=torch.uint8).reshape(8, 64)
    packed, logical = _pack_dsv4_indexer_full_pages(
        [buf],
        full_page_size=256,
        compress_ratio=4,
        physical_page_size=64,
    )
    assert logical == 64
    assert packed[0] is buf or packed[0].shape == buf.shape


def test_pack_indexer_groups_v41_ratio1_pages():
    # page_size=256, ratio=1, indexer page=64 -> 4 physical pages per FlexKV page
    rows = 12
    cols = 80
    buf = torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols)
    packed, logical = _pack_dsv4_indexer_full_pages(
        [buf],
        full_page_size=256,
        compress_ratio=1,
        physical_page_size=64,
    )
    assert logical == 256
    assert packed[0].shape == (3, 4 * cols)
    assert torch.equal(packed[0][0], buf[:4].reshape(-1))
    assert torch.equal(packed[0][1], buf[4:8].reshape(-1))


def test_pack_indexer_groups_v41_ratio2_pages():
    rows = 8
    cols = 40
    buf = torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols)
    packed, logical = _pack_dsv4_indexer_full_pages(
        [buf],
        full_page_size=256,
        compress_ratio=2,
        physical_page_size=64,
    )
    assert logical == 128
    assert packed[0].shape == (4, 2 * cols)


def test_pack_indexer_rejects_untiled_page():
    buf = torch.zeros(4, 32, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="does not tile"):
        _pack_dsv4_indexer_full_pages(
            [buf],
            full_page_size=256,
            compress_ratio=1,
            physical_page_size=96,
        )


class _FakePool:
    def __init__(self, n_layers, n_pages, page_size, bytes_per_token):
        self.page_size = page_size
        self._bpt = bytes_per_token
        self.kv_buffer = [
            torch.zeros(n_pages, page_size * bytes_per_token, dtype=torch.uint8)
            for _ in range(n_layers)
        ]

    def get_bytes_per_token(self):
        return self._bpt


class _FakeIndexer:
    def __init__(self, n_layers, n_pages, page_size, bytes_per_token):
        self.page_size = page_size
        self.index_k_with_scale_buffer = [
            torch.zeros(n_pages, page_size * bytes_per_token, dtype=torch.uint8)
            for _ in range(n_layers)
        ]


class _FakeState:
    def __init__(self, *, ring_size, rows, cols, request_scoped=False):
        self.ring_size = ring_size
        self.request_scoped = request_scoped
        self.kv_score_buffer = SimpleNamespace(
            kv_score=torch.zeros(rows, cols, dtype=torch.float32)
        )


def _connector_for_resolve(*, swa_multi_group=False, swa_kv_pool=None):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 256
    connector._dsv4_layer_groups = []
    connector._dsv4_state_groups = []
    connector._is_dsv4 = True
    connector._swa_kv_pool = swa_kv_pool
    connector.cache_config = SimpleNamespace(swa=SimpleNamespace(multi_group=False))
    connector.flexkv_config = SimpleNamespace(
        user_config=SimpleNamespace(swa_multi_group=swa_multi_group)
    )
    return connector


def test_resolve_registers_ratio_1_and_2_kv_and_indexer():
    connector = _connector_for_resolve(swa_multi_group=False)

    ratios = [1, 1, 2, 2, 4, 128]
    kvcache = SimpleNamespace(
        compression_ratios=ratios,
        _stage_start=0,
        _stage_end=len(ratios),
        sources_by_ratio={1: [0], 2: [2], 4: [4], 128: [5]},
        kv_pools={
            1: _FakePool(1, 4, 256, 528),
            2: _FakePool(1, 4, 128, 288),
            4: _FakePool(1, 4, 64, 528),
            128: _FakePool(1, 4, 2, 528),
        },
        index_pools={
            1: _FakeIndexer(1, 8, 64, 80),
            2: _FakeIndexer(1, 8, 64, 80),
            4: _FakeIndexer(1, 4, 64, 72),
        },
        c4_kv_pool=None,
        c128_kv_pool=None,
        c4_indexer_kv_pool=None,
        _unified_kv=False,
        compress_state_pools=[None] * 6,
        indexer_compress_state_pools=[None] * 6,
        index_k_with_scale_buffer=None,
    )

    buffers, indexer_group = FlexKVConnector._resolve_kv_buffers(connector, kvcache)
    names = [g["name"] for g in connector._dsv4_layer_groups]
    assert names == [
        "c4",
        "c128",
        "c1",
        "c2",
        "c4_indexer",
        "c1_indexer",
        "c2_indexer",
    ]
    by_name = {g["name"]: g for g in connector._dsv4_layer_groups}
    assert by_name["c1"]["ratio"] == 1
    assert by_name["c1"]["sub_page_size"] == 256
    assert by_name["c2"]["sub_page_size"] == 128
    assert by_name["c1_indexer"]["sub_page_size"] == 256
    assert by_name["c1_indexer"]["buffers"][0].shape[0] == 2  # 8 physical / 4
    assert by_name["c2_indexer"]["sub_page_size"] == 128
    assert by_name["c2_indexer"]["buffers"][0].shape[0] == 4  # 8 physical / 2
    assert by_name["c4_indexer"]["sub_page_size"] == 64
    assert indexer_group is None
    assert len(buffers) == sum(len(g["buffers"]) for g in connector._dsv4_layer_groups)


def test_resolve_keeps_legacy_c4_c128_attributes():
    connector = _connector_for_resolve(swa_multi_group=False)

    c4 = _FakePool(2, 4, 64, 585)
    c128 = _FakePool(1, 4, 2, 585)
    indexer = _FakeIndexer(2, 4, 64, 72)
    kvcache = SimpleNamespace(
        compression_ratios=[4, 4, 128],
        _stage_start=0,
        _stage_end=3,
        sources_by_ratio=None,
        kv_pools=None,
        index_pools=None,
        c4_kv_pool=c4,
        c128_kv_pool=c128,
        c4_indexer_kv_pool=indexer,
        _unified_kv=False,
        compress_state_pools=[None, None, None],
        indexer_compress_state_pools=[None, None, None],
        index_k_with_scale_buffer=None,
    )
    FlexKVConnector._resolve_kv_buffers(connector, kvcache)
    names = [g["name"] for g in connector._dsv4_layer_groups]
    assert names == ["c4", "c128", "c4_indexer"]
    assert connector._dsv4_layer_groups[0]["layer_ids"] == [0, 1]
    assert connector._dsv4_layer_groups[1]["layer_ids"] == [2]


def test_resolve_registers_paged_c4_states_for_classic_dsv4():
    connector = _connector_for_resolve(
        swa_multi_group=None,
        swa_kv_pool=object(),
    )
    attn = _FakeState(ring_size=64, rows=4, cols=32)
    indexer = _FakeState(ring_size=64, rows=4, cols=16)
    kvcache = SimpleNamespace(
        compression_ratios=[4, 4, 128],
        _stage_start=0,
        _stage_end=3,
        sources_by_ratio={4: [0, 1], 128: [2]},
        kv_pools={
            4: _FakePool(2, 4, 64, 585),
            128: _FakePool(1, 4, 2, 585),
        },
        index_pools={4: _FakeIndexer(2, 4, 64, 72)},
        _unified_kv=False,
        compress_state_pools=[attn, attn, _FakeState(ring_size=2, rows=4, cols=8, request_scoped=True)],
        indexer_compress_state_pools=[indexer, indexer, None],
        swa_page_size=256,
        index_k_with_scale_buffer=None,
    )
    connector._kvcache = kvcache
    FlexKVConnector._resolve_kv_buffers(connector, kvcache)
    names = [g["name"] for g in connector._dsv4_state_groups]
    assert names == ["c4_attention_state", "c4_indexer_state"]
    assert connector._dsv4_state_groups[0]["layer_ids"] == [0, 1]


def test_resolve_skips_request_scoped_flash_pair_state():
    connector = _connector_for_resolve(
        swa_multi_group=None,
        swa_kv_pool=object(),
    )
    pair = _FakeState(ring_size=2, rows=4, cols=8, request_scoped=True)
    kvcache = SimpleNamespace(
        compression_ratios=[1, 1, 2, 2],
        _stage_start=0,
        _stage_end=4,
        sources_by_ratio={1: [0], 2: [2]},
        kv_pools={
            1: _FakePool(1, 4, 256, 528),
            2: _FakePool(1, 4, 128, 288),
        },
        index_pools={
            1: _FakeIndexer(1, 8, 64, 80),
            2: _FakeIndexer(1, 8, 64, 80),
        },
        _unified_kv=False,
        compress_state_pools=[None, None, pair, None],
        indexer_compress_state_pools=[None, None, None, None],
        swa_page_size=256,
        index_k_with_scale_buffer=None,
    )
    connector._kvcache = kvcache
    FlexKVConnector._resolve_kv_buffers(connector, kvcache)
    assert [g["name"] for g in connector._dsv4_layer_groups] == [
        "c1",
        "c2",
        "c1_indexer",
        "c2_indexer",
    ]
    assert connector._dsv4_state_groups == []


def test_resolve_skips_state_sidecars_without_paged_swa():
    connector = _connector_for_resolve(swa_multi_group=None, swa_kv_pool=None)
    attn = _FakeState(ring_size=64, rows=4, cols=32)
    kvcache = SimpleNamespace(
        compression_ratios=[4],
        _stage_start=0,
        _stage_end=1,
        sources_by_ratio={4: [0]},
        kv_pools={4: _FakePool(1, 4, 64, 585)},
        index_pools={},
        _unified_kv=False,
        compress_state_pools=[attn],
        indexer_compress_state_pools=[None],
        swa_page_size=256,
        index_k_with_scale_buffer=None,
    )
    connector._kvcache = kvcache
    FlexKVConnector._resolve_kv_buffers(connector, kvcache)
    assert [g["name"] for g in connector._dsv4_layer_groups] == ["c4"]
    assert connector._dsv4_state_groups == []

