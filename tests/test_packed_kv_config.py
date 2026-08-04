import json
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
import torch

from flexkv.common.config import ModelConfig, UserConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.tracer import FlexKVTracer
from flexkv.integration.config import FlexKVConfig


class _FakeModelConfig:
    dtype = torch.bfloat16
    is_deepseek_mla = False

    def get_num_layers(self, parallel_config):
        return 2

    def get_head_size(self):
        return 64

    def get_num_kv_heads(self, parallel_config):
        return max(1, 4 // parallel_config.tensor_parallel_size)

    def get_total_num_kv_heads(self):
        return 4


class _FakeMLAModelConfig(_FakeModelConfig):
    is_deepseek_mla = True

    def get_head_size(self):
        return 576

    def get_num_kv_heads(self, parallel_config):
        return 1


class _FakeBackend:
    def __init__(self, shape):
        self.shape = shape

    def get_kv_cache_shape(self, *args):
        return self.shape


def _fake_vllm_config(cache_dtype, tp_size=1, model_config=None):
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        context_parallel_size=1,
        nnodes=1,
    )
    return SimpleNamespace(
        model_config=model_config or _FakeModelConfig(),
        cache_config=SimpleNamespace(block_size=16, cache_dtype=cache_dtype),
        parallel_config=parallel_config,
    )


def _install_fake_backend(monkeypatch, shape):
    module_name = "vllm.distributed.kv_transfer.kv_connector.utils"
    fake_module = types.ModuleType(module_name)
    fake_module.get_current_attn_backend = lambda vllm_config: _FakeBackend(shape)
    monkeypatch.setitem(sys.modules, module_name, fake_module)


@pytest.mark.parametrize(
    (
        "cache_dtype",
        "tp_size",
        "cache_shape",
        "num_kv_heads",
        "packed_kv",
        "head_size",
        "kv_dim",
    ),
    [
        ("auto", 1, (1, 4, 16, 128), 4, True, 128, 1),
        ("nvfp4", 1, (1, 8, 16, 36), 8, True, 36, 1),
        ("nvfp4", 1, (1, 2, 16, 4, 36), 4, False, 36, 2),
        ("nvfp4", 1, (2, 1, 16, 4, 36), 4, False, 36, 2),
        ("auto", 2, (1, 2, 16, 128), 4, True, 128, 1),
        ("nvfp4", 2, (1, 4, 16, 36), 8, True, 36, 1),
        ("nvfp4", 2, (1, 2, 16, 2, 36), 4, False, 36, 2),
    ],
)
def test_vllm_cache_shape_drives_physical_storage_layout(
    monkeypatch,
    cache_dtype,
    tp_size,
    cache_shape,
    num_kv_heads,
    packed_kv,
    head_size,
    kv_dim,
):
    _install_fake_backend(monkeypatch, cache_shape)
    monkeypatch.setenv("LOCAL_RANK", "0")
    config = FlexKVConfig(user_config=UserConfig(cpu_cache_gb=1))

    config.post_init_from_vllm_config(_fake_vllm_config(cache_dtype, tp_size=tp_size))

    assert config.model_config.num_kv_heads == num_kv_heads
    assert config.model_config.packed_kv is packed_kv
    assert config.model_config.head_size == head_size
    assert config.model_config.kv_dim == kv_dim
    assert config.model_config.bytes_per_token_per_layer == (
        num_kv_heads * head_size * kv_dim * config.model_config.dtype.itemsize
    )


def test_backend_physical_head_count_is_authoritative(monkeypatch):
    _install_fake_backend(monkeypatch, (1, 6, 16, 36))
    monkeypatch.setenv("LOCAL_RANK", "0")
    config = FlexKVConfig(user_config=UserConfig(cpu_cache_gb=1))

    config.post_init_from_vllm_config(_fake_vllm_config("nvfp4"))

    assert config.model_config.num_kv_heads == 6
    assert config.model_config.head_size == 36
    assert config.model_config.packed_kv is True


@pytest.mark.parametrize("physical_head_size", [576, 656])
def test_vllm_mla_cache_shape_drives_physical_storage_layout(
    monkeypatch,
    physical_head_size,
):
    _install_fake_backend(monkeypatch, (1, 16, physical_head_size))
    monkeypatch.setenv("LOCAL_RANK", "0")
    config = FlexKVConfig(user_config=UserConfig(cpu_cache_gb=1))

    config.post_init_from_vllm_config(
        _fake_vllm_config(
            "auto",
            tp_size=4,
            model_config=_FakeMLAModelConfig(),
        )
    )

    assert config.model_config.use_mla is True
    assert config.model_config.num_kv_heads == 1
    assert config.model_config.head_size == physical_head_size
    assert config.model_config.packed_kv is False
    assert config.model_config.kv_dim == 1


def test_unsupported_backend_cache_rank_is_rejected(monkeypatch):
    _install_fake_backend(monkeypatch, (1, 4, 16))
    config = FlexKVConfig(user_config=UserConfig(cpu_cache_gb=1))

    with pytest.raises(ValueError, match="expected 4 or 5 dimensions"):
        config.post_init_from_vllm_config(_fake_vllm_config("auto"))


def test_vllm_backend_layout_api_is_required(monkeypatch):
    module_name = "vllm.distributed.kv_transfer.kv_connector.utils"
    monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    config = FlexKVConfig(user_config=UserConfig(cpu_cache_gb=1))

    with pytest.raises(ImportError):
        config.post_init_from_vllm_config(_fake_vllm_config("auto"))


def test_model_config_string_includes_packed_kv():
    model_config = ModelConfig(packed_kv=True)

    assert "packed_kv=True" in str(model_config)


def test_trace_config_includes_packed_kv():
    tracer = object.__new__(FlexKVTracer)
    tracer.enabled = True
    tracer._lock = threading.Lock()
    tracer._buffer = []
    tracer._last_flush_time = time.time()
    tracer.flush_interval_ms = 10**9

    model_config = ModelConfig(packed_kv=True)
    cache_config = SimpleNamespace(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=False,
        enable_gds=False,
        remote_cache_size_mode="file",
        num_cpu_blocks=1,
        num_ssd_blocks=0,
        num_gds_blocks=0,
        num_remote_blocks=0,
        ssd_cache_dir=[],
        gds_cache_dir=[],
        remote_file_size=0,
        remote_file_num=0,
        remote_file_prefix="",
        remote_cache_path="",
        remote_config_custom={},
    )
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=1,
        num_block=1,
        tokens_per_block=1,
        num_head=1,
        head_size=2,
        is_mla=False,
        packed_kv=True,
    )

    tracer.trace_config(model_config, cache_config, gpu_layout)

    record = json.loads(tracer._buffer[0])
    assert record["data"]["model_config"]["packed_kv"] is True
    assert record["data"]["gpu_layout"]["packed_kv"] is True
