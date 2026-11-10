import os

import pytest

from flexkv.common import platform


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    keys = set(platform.CE_ONLY_ENV_PRESETS) | {
        "FLEXKV_ENABLE_CE_ONLY", "FLEXKV_ENABLE_GDS", "FLEXKV_ENABLE_NVCOMP",
        "FLEXKV_GDR_CONFIG", "FLEXKV_ENABLE_GDR", "FLEXKV_REGION_BATCH",
        "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("value", [None, "0", "false", "no", "off"])
def test_disabled_is_a_noop(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", value)
    monkeypatch.setenv("FLEXKV_ENABLE_MPS", "1")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    before = dict(os.environ)
    assert platform.apply_ce_only_env_presets() == {}
    assert dict(os.environ) == before


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_presets_are_idempotent(monkeypatch, value):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", value)
    assert platform.apply_ce_only_env_presets() == platform.CE_ONLY_ENV_PRESETS
    assert platform.apply_ce_only_env_presets() == {}


def test_backend_detection_never_enables_presets(monkeypatch):
    monkeypatch.setattr(platform, "is_emulated_cuda_backend", lambda: True)
    assert not platform.ce_only_mode()
    assert platform.apply_ce_only_env_presets() == {}


@pytest.mark.parametrize("config", [
    '{"enabled":false}',
    '{"enabled":true,"h2d":{"path":"gdr"},"d2h":{"path":"gdr"}}',
    '{"enabled":true,"h2d":{"path":"original"},"d2h":{"path":"gdr"}}',
])
def test_gdr_and_region_batch_are_not_overridden(monkeypatch, config):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.setenv("FLEXKV_ENABLE_GDR", "1")
    monkeypatch.setenv("FLEXKV_GDR_CONFIG", config)
    monkeypatch.setenv("FLEXKV_REGION_BATCH", "1")
    monkeypatch.setenv("FLEXKV_CE_PATH_OPT", "1")
    platform.apply_ce_only_env_presets()
    assert os.environ["FLEXKV_GDR_CONFIG"] == config
    assert os.environ["FLEXKV_ENABLE_GDR"] == "1"
    assert os.environ["FLEXKV_REGION_BATCH"] == "1"
    assert os.environ["FLEXKV_CE_PATH_OPT"] == "1"


@pytest.mark.parametrize("key,value", [
    ("FLEXKV_ENABLE_MPS", "1"),
    ("FLEXKV_USE_CE_TRANSFER_H2D", "0"),
    ("FLEXKV_USE_CE_TRANSFER_D2H", "0"),
    ("FLEXKV_ENABLE_CE_MEMCPY2D", "1"),
    ("FLEXKV_DISABLE_VMM_PROBE", "0"),
    ("FLEXKV_DISABLE_VMM_PROBE", "garbage"),
    ("FLEXKV_FORCE_DIRECT_IPC", "1"),
    ("FLEXKV_CE_GATHER_BACKEND", "aten"),
    ("FLEXKV_CE_GATHER_BACKEND", ""),
    ("FLEXKV_LAYERWISE_NOTIFY_MODE", "hostfunc"),
    ("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0x3"),
    ("FLEXKV_CUDAHOST_REGISTER_FLAGS", "4"),
    ("FLEXKV_CUDAHOST_REGISTER_FLAGS", "-1"),
    ("FLEXKV_ENABLE_GDS", "1"),
    ("FLEXKV_ENABLE_NVCOMP", "1"),
])
def test_conflicts_fail_before_injecting_presets(monkeypatch, key, value):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.setenv(key, value)
    before = dict(os.environ)
    with pytest.raises(platform.CeOnlyConfigError, match=key):
        platform.apply_ce_only_env_presets()
    assert dict(os.environ) == before


def test_explicit_compatible_values_are_preserved(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0x0")
    platform.apply_ce_only_env_presets()
    assert os.environ["FLEXKV_CUDAHOST_REGISTER_FLAGS"] == "0x0"


@pytest.mark.parametrize("key", ["PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"])
@pytest.mark.parametrize("conf,expected", [
    ("expandable_segments:True", ""),
    ("expandable_segments:False,max_split_size_mb:64", "max_split_size_mb:64"),
    ("max_split_size_mb:64,expandable_segments:True", "max_split_size_mb:64"),
    ("max_split_size_mb:64,expandable_segments:True,backend:native",
     "max_split_size_mb:64,backend:native"),
    ("roundup_power2_divisions:[256:1,512:2,>:8],expandable_segments:True",
     "roundup_power2_divisions:[256:1,512:2,>:8]"),
])
def test_allocator_cleanup_preserves_other_options(monkeypatch, key, conf, expected):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.setenv(key, conf)
    platform.apply_ce_only_env_presets()
    assert os.environ.get(key, "") == expected


def test_layerwise_and_explicit_region_batch_stay_env_authoritative():
    import ast
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "flexkv/common/config.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.FunctionDef) and n.name == "update_default_config_from_user_config")
    text = ast.unparse(fn)
    assert "env_authoritative_keys = {'enable_layerwise_transfer'}" in text
    assert "if os.environ.get('FLEXKV_REGION_BATCH') is not None" in text
    assert text.index("if global_attr_name in env_authoritative_keys") < text.index(
        "if global_attr_name in global_config_attrs")
