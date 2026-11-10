"""Source-level build regressions; these do not substitute native runtime tests."""

import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _setup_function(name, **scope):
    module = ast.parse((ROOT / "setup.py").read_text())
    node = next(node for node in module.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"os": os, "Path": Path, **scope}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "setup.py", "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True), ("off", False), ("true", True)])
def test_build_boolean(monkeypatch, value, expected):
    monkeypatch.setenv("FLEXKV_TEST_CAPABILITY", value)
    assert _setup_function("_build_bool")("FLEXKV_TEST_CAPABILITY", not expected) is expected


def test_host_mirror_tracks_source_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "csrc").mkdir()
    source = tmp_path / "csrc" / "test.cu"
    source.write_text("int version = 1;\n")
    dependencies = []
    mirror = _setup_function("_host_compile_cuda_sources", build_dir=str(tmp_path / "build"),
                             hpp_sources=dependencies)
    sources = mirror(["csrc/test.cu", "csrc/unchanged.cpp"])
    assert Path(sources[0]).suffix == ".cpp"
    assert Path(sources[0]).read_text() == source.read_text()
    assert sources[1] == "csrc/unchanged.cpp"
    assert "csrc/test.cu" in dependencies
    source.write_text("int version = 2;\n")
    mirror(["csrc/test.cu"])
    assert Path(sources[0]).read_text() == source.read_text()


def test_build_is_independent_of_runtime_master_switch():
    setup_source = (ROOT / "setup.py").read_text()
    assert 'os.environ.get("FLEXKV_ENABLE_CE_ONLY"' not in setup_source
    assert "FLEXKV_ENABLE_KERNEL_TRANSFER" in setup_source
    for source in ("ce_trace.cpp", "device_thread_pool.cpp", "transfer_backend.cpp",
                   "region_batch.cpp", "layer_notify.cpp"):
        assert f'"csrc/{source}"' in setup_source
    assert '"csrc/layerwise.cpp"' not in setup_source


def test_staged_path_preserves_checked_host_dma_and_error_drain():
    source = (ROOT / "csrc/ce_transfer.cu").read_text()
    staged = source.split("void ce_transfer_memcpy_staged(", 1)[1].split(
        "// ---- GATHER_SCATTER:", 1)[0]
    assert "checked_host_memcpy_async" in staged
    assert 'drain_stream_or_abort(stream, "memcpy-staged transfer")' in staged
    assert "index_select" not in staged
    assert "index_copy_" not in staged
    assert "cudaMemcpy2DAsync" not in staged
    assert "backend=memcpy" in staged


def test_no_kernel_branch_is_not_a_noop():
    source = (ROOT / "csrc/transfer.cu").read_text()
    assert "Kernel transfer was not compiled; enable CE transfer" in source
    assert "record(path, &analysis)" in source
    assert "path != CEPath::CONTIG_DIRECT" in source
    assert "path != CEPath::SEGMENT_DIRECT" in source
