import pytest
import torch

pytest.importorskip("vllm")

from flexkv.integration.vllm.vllm_v1_adapter import (
    _FlexKVConnectorStats,
    _assert_token_major,
)


def test_packed_nhd_layout_is_accepted():
    physical = torch.empty((2, 16, 8, 36))
    packed_nhd = physical.permute(0, 2, 1, 3)

    _assert_token_major(packed_nhd, token_dim=2, head_dim=1)


def test_packed_hnd_layout_is_rejected():
    packed_hnd = torch.empty((2, 8, 16, 36))

    with pytest.raises(ValueError, match=r"head-major \(HND\)"):
        _assert_token_major(packed_hnd, token_dim=2, head_dim=1)


def test_flexkv_connector_stats_implements_current_vllm_contract():
    empty = _FlexKVConnectorStats(data={"num_get_requests": 0})
    stats = _FlexKVConnectorStats(data={
        "num_get_requests": 1,
        "num_get_query_tokens": 32,
        "num_gpu_matched_tokens": 0,
        "num_flexkv_matched_tokens": 16,
        "get_gpu_match_ratio": 0.0,
        "get_flexkv_match_ratio": 0.5,
    })

    assert empty.is_empty() is True
    assert stats.is_empty() is False
    assert stats.reduce()["num_flexkv_matched_tokens"] == 16
    empty.aggregate(stats)
    assert empty.data["get_flexkv_match_ratio"] == 0.5
    empty.reset()
    assert empty.is_empty() is True
