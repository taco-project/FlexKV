import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flexkv.integration.config import FlexKVConfig


pytestmark = pytest.mark.unit


@pytest.fixture
def config():
    config = FlexKVConfig()
    config.cache_config.tokens_per_block = 16
    return config


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize(
    "shape,requested_heads,physical_heads,head_size,kv_dim,shared,dtype",
    [
        pytest.param((1, 16, 576), 1, 1, 576, 1, True, torch.bfloat16, id="mla"),
        pytest.param((1, 1, 16, 128), 1, 1, 128, 1, False, torch.bfloat16, id="packed-single-head"),
        pytest.param((1, 4, 16, 128), 4, 4, 128, 1, False, torch.bfloat16, id="packed-multi-head"),
        pytest.param((2, 1, 16, 1, 64), 1, 1, 64, 2, False, torch.bfloat16, id="split-single-head"),
        pytest.param((1, 2, 16, 4, 64), 4, 4, 64, 2, False, torch.bfloat16, id="split-multi-head"),
        pytest.param((1, 2, 16, 36), 1, 2, 36, 1, False, torch.uint8, id="packed-nvfp4"),
        pytest.param((2, 1, 16, 1, 36), 1, 1, 36, 2, False, torch.uint8, id="split-nvfp4"),
    ],
)
def test_backend_storage_layout(
    monkeypatch, config, tp_size, shape, requested_heads, physical_heads, head_size, kv_dim, shared, dtype
):
    config.model_config.tp_size = tp_size
    config.model_config.dtype = dtype
    backend = SimpleNamespace(get_kv_cache_shape=Mock(return_value=shape))
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.kv_transfer.kv_connector.utils",
        SimpleNamespace(get_current_attn_backend=lambda vllm_config: backend),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.config", SimpleNamespace(set_current_vllm_config=nullcontext)
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        model_config=SimpleNamespace(
            get_num_kv_heads=lambda parallel_config: requested_heads,
            get_head_size=lambda: 64,
        ),
    )
    cache_dtype = "nvfp4" if dtype == torch.uint8 else "auto"

    config._configure_kv_layout_from_vllm_backend(vllm_config, cache_dtype)

    assert config.model_config.num_kv_heads == physical_heads * (1 if shared else tp_size)
    assert config.model_config.head_size == head_size
    assert config.model_config.kv_dim == kv_dim
    backend.get_kv_cache_shape.assert_called_once_with(1, 16, requested_heads, 64, cache_dtype)


@pytest.fixture
def cache_specs(monkeypatch):
    class AttentionSpec(SimpleNamespace):
        pass

    class MLAAttentionSpec(AttentionSpec):
        pass

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.kv_cache_interface",
        SimpleNamespace(
            AttentionSpec=AttentionSpec,
            MLAAttentionSpec=MLAAttentionSpec,
            iter_layer_specs=lambda specs: specs,
        ),
    )

    def make_spec(shared=False, physical_heads=1, head_size=576, dtype=torch.bfloat16):
        spec_type = MLAAttentionSpec if shared else AttentionSpec
        return spec_type(
            num_heads=physical_heads,
            state_content_size_bytes=head_size * torch.empty((), dtype=dtype).element_size(),
            dtype=dtype,
            block_size=16,
            tokens_per_state=1,
            page_size_padded=None,
        )

    return make_spec


def _kv_cache_config(*group_specs, prefix_cacheable_group_ids=None):
    """A vLLM KVCacheConfig with one KV cache group per list of layer specs,
    every group prefix-cacheable unless ``prefix_cacheable_group_ids`` says."""
    groups = [
        SimpleNamespace(
            layer_names=[f"g{group_id}.layers.{i}.attn" for i in range(len(specs))],
            kv_cache_spec=specs,
        )
        for group_id, specs in enumerate(group_specs)
    ]
    if prefix_cacheable_group_ids is None:
        prefix_cacheable_group_ids = tuple(range(len(groups)))
    return SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=groups,
        prefix_cacheable_group_ids=prefix_cacheable_group_ids,
    )


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize(
    "shared,physical_heads,head_size,dtype",
    [
        pytest.param(True, 1, 576, torch.bfloat16, id="mla"),
        pytest.param(False, 1, 128, torch.bfloat16, id="packed-single-head"),
        pytest.param(False, 4, 128, torch.bfloat16, id="packed-multi-head"),
        pytest.param(False, 2, 36, torch.uint8, id="packed-nvfp4"),
    ],
)
def test_resolved_spec_storage_layout(config, cache_specs, tp_size, shared, physical_heads, head_size, dtype):
    config.model_config.tp_size = tp_size
    config.model_config.dtype = dtype
    spec = cache_specs(shared, physical_heads, head_size, dtype)
    kv_cache_config = _kv_cache_config([spec, spec])

    config._configure_kv_layout_from_vllm_backend(SimpleNamespace(), "auto", kv_cache_config)

    assert config.model_config.num_kv_heads == physical_heads * (1 if shared else tp_size)
    assert config.model_config.head_size == head_size
    assert config.model_config.kv_dim == 1


def test_resolved_specs_reject_mixed_rank_sharing(config, cache_specs):
    config.model_config.tp_size = 2
    kv_cache_config = _kv_cache_config([cache_specs(shared=True), cache_specs(shared=False)])

    with pytest.raises(ValueError, match="requires one uniform vLLM attention storage layout"):
        config._configure_kv_layout_from_vllm_cache_config(kv_cache_config)


def test_hybrid_kv_cache_stores_its_prefix_cacheable_group(config, cache_specs):
    # Only the prefix-cacheable group is stored, so only its specs must share
    # one storage layout.
    config.model_config.tp_size = 2
    kv_cache_config = _kv_cache_config(
        [cache_specs(shared=True), cache_specs(shared=False)],
        [cache_specs(shared=True)],
        prefix_cacheable_group_ids=(1,),
    )

    config._configure_kv_layout_from_vllm_cache_config(kv_cache_config)

    assert config.vllm_kv_group.group_id == 1
    assert config.vllm_kv_group.cache_layer_names == ("g1.layers.0.attn",)
    assert config.model_config.num_kv_heads == 1


def test_hybrid_kv_cache_rejects_several_prefix_cacheable_groups(config, cache_specs):
    kv_cache_config = _kv_cache_config([cache_specs(shared=True)], [cache_specs(shared=True)])

    with pytest.raises(ValueError, match="--disable-hybrid-kv-cache-manager"):
        config._configure_kv_layout_from_vllm_cache_config(kv_cache_config)
