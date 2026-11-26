import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_spec = importlib.util.spec_from_file_location(
    "cuda_compat_platform", Path(__file__).resolve().parents[1] / "flexkv/common/platform.py")
platform = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(platform)
REAL_KERNEL_TRANSFER_BUILT = platform.kernel_transfer_built


def build(monkeypatch, kernel_transfer):
    monkeypatch.setattr(platform, "kernel_transfer_built", lambda: kernel_transfer)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    keys = set(platform.NON_KERNEL_STACK_ENV_PRESETS) | {
        "FLEXKV_ENABLE_GDS", "FLEXKV_ENABLE_NVCOMP",
        "FLEXKV_GDR_CONFIG", "FLEXKV_ENABLE_GDR", "FLEXKV_REGION_BATCH",
        "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    build(monkeypatch, True)


def test_kernel_build_is_a_noop(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_GDS", "1")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    before = dict(os.environ)
    assert not platform.non_kernel_stack_mode()
    assert platform.apply_non_kernel_stack_env_presets() == {}
    assert dict(os.environ) == before


def test_presets_are_idempotent(monkeypatch):
    build(monkeypatch, False)
    assert platform.non_kernel_stack_mode()
    assert platform.apply_non_kernel_stack_env_presets() == platform.NON_KERNEL_STACK_ENV_PRESETS
    assert platform.apply_non_kernel_stack_env_presets() == {}


@pytest.mark.parametrize("recorded,expected", [(None, True), ("True", True), ("False", False)])
def test_build_info_selects_mode(monkeypatch, tmp_path, recorded, expected):
    monkeypatch.setattr(platform, "kernel_transfer_built", REAL_KERNEL_TRANSFER_BUILT)
    info = tmp_path / "_build_info.py"
    if recorded is not None:
        info.write_text(f"KERNEL_TRANSFER = {recorded}\n")
    monkeypatch.setattr(platform, "_BUILD_INFO", str(info))
    monkeypatch.setattr(platform, "_kernel_transfer_built", None)
    assert platform.kernel_transfer_built() is expected
    assert platform.non_kernel_stack_mode() is not expected


def test_mode_never_detects_installed_runtimes():
    tree = ast.parse(Path(platform.__file__).read_text())
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names}
    imports.update(node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert imports <= {"importlib.util", "logging", "os", "typing"}
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    calls = {ast.unparse(node.func) for node in ast.walk(functions["non_kernel_stack_mode"])
             if isinstance(node, ast.Call)}
    assert calls == {"kernel_transfer_built"}


def test_setup_records_build_capability():
    setup_source = (Path(__file__).resolve().parents[1] / "setup.py").read_text()
    assert 'Path("flexkv/_build_info.py").write_text(' in setup_source
    assert "KERNEL_TRANSFER = {enable_kernel_transfer}" in setup_source


@pytest.mark.parametrize("config", [
    '{"enabled":false}',
    '{"enabled":true,"h2d":{"path":"gdr"},"d2h":{"path":"gdr"}}',
    '{"enabled":true,"h2d":{"path":"original"},"d2h":{"path":"gdr"}}',
])
def test_gdr_and_region_batch_are_not_overridden(monkeypatch, config):
    build(monkeypatch, False)
    monkeypatch.setenv("FLEXKV_ENABLE_GDR", "1")
    monkeypatch.setenv("FLEXKV_GDR_CONFIG", config)
    monkeypatch.setenv("FLEXKV_REGION_BATCH", "1")
    monkeypatch.setenv("FLEXKV_CE_PATH_OPT", "1")
    platform.apply_non_kernel_stack_env_presets()
    assert os.environ["FLEXKV_GDR_CONFIG"] == config
    assert os.environ["FLEXKV_ENABLE_GDR"] == "1"
    assert os.environ["FLEXKV_REGION_BATCH"] == "1"
    assert os.environ["FLEXKV_CE_PATH_OPT"] == "1"


@pytest.fixture
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(platform, "_warned_overrides", set())


@pytest.mark.parametrize("key,value", [
    ("FLEXKV_USE_CE_TRANSFER_H2D", "0"),
    ("FLEXKV_USE_CE_TRANSFER_D2H", "0"),
    ("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0x3"),
    ("FLEXKV_CUDAHOST_REGISTER_FLAGS", "4"),
    ("FLEXKV_CUDAHOST_REGISTER_FLAGS", "-1"),
])
def test_kernel_path_settings_fail_before_injecting_presets(monkeypatch, key, value):
    build(monkeypatch, False)
    monkeypatch.setenv(key, value)
    before = dict(os.environ)
    with pytest.raises(platform.NonKernelStackConfigError, match=key):
        platform.apply_non_kernel_stack_env_presets()
    assert dict(os.environ) == before


@pytest.mark.parametrize("key,value", [
    ("FLEXKV_ENABLE_CE_MEMCPY2D", "1"),
    ("FLEXKV_DISABLE_VMM_PROBE", "0"),
    ("FLEXKV_FORCE_DIRECT_IPC", "1"),
    ("FLEXKV_STRICT_IPC_MODE", "0"),
    ("FLEXKV_CE_GATHER_BACKEND", "aten"),
    ("FLEXKV_LAYERWISE_NOTIFY_MODE", "hostfunc"),
    ("FLEXKV_ENABLE_GDS", "1"),
    ("FLEXKV_ENABLE_NVCOMP", "1"),
])
def test_recommended_overrides_are_kept_with_warning(monkeypatch, caplog, fresh_warnings,
                                                     key, value):
    build(monkeypatch, False)
    monkeypatch.setenv(key, value)
    with caplog.at_level("WARNING", logger="flexkv.platform"):
        applied = platform.apply_non_kernel_stack_env_presets()
    assert os.environ[key] == value
    assert key not in applied
    assert [r for r in caplog.records if key in r.getMessage()]
    caplog.clear()
    with caplog.at_level("WARNING", logger="flexkv.platform"):
        platform.apply_non_kernel_stack_env_presets()
    assert not caplog.records


def test_recommended_values_do_not_warn(monkeypatch, caplog, fresh_warnings):
    build(monkeypatch, False)
    for key, value in platform.NON_KERNEL_STACK_RECOMMENDED_ENV.items():
        monkeypatch.setenv(key, value)
    with caplog.at_level("WARNING", logger="flexkv.platform"):
        platform.apply_non_kernel_stack_env_presets()
    assert not caplog.records


def test_explicit_compatible_values_are_preserved(monkeypatch):
    build(monkeypatch, False)
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0x0")
    platform.apply_non_kernel_stack_env_presets()
    assert os.environ["FLEXKV_CUDAHOST_REGISTER_FLAGS"] == "0x0"


@pytest.mark.parametrize("key", ["PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"])
@pytest.mark.parametrize("conf,warns", [
    ("expandable_segments:True", True),
    ("max_split_size_mb:64, expandable_segments : TRUE", True),
    ("expandable_segments:False,max_split_size_mb:64", False),
    ("max_split_size_mb:64", False),
])
def test_allocator_config_is_kept_and_expandable_segments_warns(
        monkeypatch, caplog, fresh_warnings, key, conf, warns):
    build(monkeypatch, False)
    monkeypatch.setenv(key, conf)
    with caplog.at_level("WARNING", logger="flexkv.platform"):
        platform.apply_non_kernel_stack_env_presets()
    assert os.environ[key] == conf
    assert bool([r for r in caplog.records if "expandable_segments" in r.getMessage()]) == warns


@pytest.mark.parametrize("kernel_transfer,nvtx_flag,expected", [(True, None, True), (False, None, False),
                                                      (True, "0", False), (False, "1", True)])
def test_nvtx_preserves_default_api_and_explicit_overrides(monkeypatch, kernel_transfer, nvtx_flag, expected):
    from unittest.mock import Mock
    build(monkeypatch, kernel_transfer)
    if nvtx_flag is None:
        monkeypatch.delenv("FLEXKV_ENABLE_NVTX", raising=False)
    else:
        monkeypatch.setenv("FLEXKV_ENABLE_NVTX", nvtx_flag)
    root = Path(__file__).resolve().parents[1]
    common = ModuleType("flexkv.common")
    common.platform = platform
    package = ModuleType("flexkv")
    package.common = common
    for name, module in (("flexkv", package), ("flexkv.common", common),
                         ("flexkv.common.platform", platform)):
        monkeypatch.setitem(sys.modules, name, module)
    real = ModuleType("nvtx")
    for name in ("annotate", "start_range", "end_range", "push_range", "pop_range", "mark"):
        setattr(real, name, Mock())
    monkeypatch.setitem(sys.modules, "nvtx", real)
    spec = importlib.util.spec_from_file_location("nvtx_contract", root / "flexkv/common/nvtx_compat.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    assert wrapper.HAS_NVTX is expected
    if expected:
        assert wrapper.start_range is real.start_range
    else:
        with wrapper.annotate("disabled"):
            wrapper.mark("disabled")
        real.mark.assert_not_called()
