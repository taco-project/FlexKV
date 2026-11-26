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


def test_build_selects_kernel_transfer_and_host_sources():
    setup_source = (ROOT / "setup.py").read_text()
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


def _build_capabilities():
    module = ast.parse((ROOT / "setup.py").read_text())
    start = next(i for i, node in enumerate(module.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "enable_kernel_transfer"
                         for t in node.targets))
    end = next(i for i in range(start, len(module.body))
               if isinstance(module.body[i], ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "cpp_sources"
                       for t in module.body[i].targets))
    namespace = dict(os=os, _build_bool=_setup_function("_build_bool"),
                     enable_gds=False, enable_nvcomp=False)
    exec(compile(ast.Module(body=module.body[start:end], type_ignores=[]),
                 "setup.py", "exec"), namespace)
    return namespace


def test_default_build_keeps_native_capabilities(monkeypatch):
    for name in ("FLEXKV_ENABLE_KERNEL_TRANSFER", "FLEXKV_ENABLE_NVTX",
                 "FLEXKV_ENABLE_GDR", "FLEXKV_GDR_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "")
    flags = _build_capabilities()
    assert flags["enable_kernel_transfer"] is True
    assert flags["enable_nvtx"] is True
    assert flags["enable_gdr"] is False
    assert flags["gdr_backend"] == "native"


@pytest.mark.parametrize("backend", [None, "auto", "mapped"])
def test_host_only_gdr_auto_resolves_to_mapped(monkeypatch, backend):
    monkeypatch.setenv("FLEXKV_ENABLE_KERNEL_TRANSFER", "0")
    monkeypatch.setenv("FLEXKV_ENABLE_GDR", "1")
    monkeypatch.delenv("FLEXKV_ENABLE_NVTX", raising=False)
    if backend is None:
        monkeypatch.delenv("FLEXKV_GDR_BACKEND", raising=False)
    else:
        monkeypatch.setenv("FLEXKV_GDR_BACKEND", backend)
    flags = _build_capabilities()
    assert flags["gdr_backend"] == "mapped"
    assert flags["enable_kernel_transfer"] is False
    assert flags["enable_nvtx"] is False
    assert flags["enable_gdr"] is True


def test_host_only_gdr_rejects_explicit_native(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_KERNEL_TRANSFER", "0")
    monkeypatch.setenv("FLEXKV_ENABLE_GDR", "1")
    monkeypatch.setenv("FLEXKV_GDR_BACKEND", "native")
    with pytest.raises(RuntimeError, match="FLEXKV_GDR_BACKEND=mapped"):
        _build_capabilities()


def test_native_gdr_keeps_kernels(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_KERNEL_TRANSFER", "1")
    monkeypatch.setenv("FLEXKV_ENABLE_GDR", "1")
    monkeypatch.delenv("FLEXKV_GDR_BACKEND", raising=False)
    assert _build_capabilities()["gdr_backend"] == "native"
    monkeypatch.setenv("FLEXKV_GDR_BACKEND", "native")
    assert _build_capabilities()["enable_kernel_transfer"] is True
    monkeypatch.setenv("FLEXKV_GDR_BACKEND", "mapped")
    with pytest.raises(RuntimeError, match="KERNEL_TRANSFER=0"):
        _build_capabilities()


@pytest.mark.parametrize("backend", ["", "cuda", "legacy", "portable"])
def test_gdr_backend_rejects_unsupported_names(monkeypatch, backend):
    monkeypatch.setenv("FLEXKV_GDR_BACKEND", backend)
    with pytest.raises(ValueError, match="must be auto, native or mapped"):
        _build_capabilities()


def test_gdr_backend_allowlist_has_no_aliases():
    module = ast.parse((ROOT / "setup.py").read_text())
    checks = [node for node in ast.walk(module) if isinstance(node, ast.Compare)
              and isinstance(node.left, ast.Name) and node.left.id == "gdr_backend"
              and any(isinstance(op, ast.NotIn) for op in node.ops)]
    assert len(checks) == 1
    assert ast.literal_eval(checks[0].comparators[0]) == ("native", "mapped")


@pytest.mark.parametrize("backend", ["native", "mapped"])
def test_gdr_backend_compiler_and_link_flags(monkeypatch, backend):
    monkeypatch.setenv("FLEXKV_GDR_WR_API", "1")
    module = ast.parse((ROOT / "setup.py").read_text())
    start = next(i for i, node in enumerate(module.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "capability_flags"
                         for t in node.targets))
    scope = dict(_build_bool=_setup_function("_build_bool"),
                 enable_kernel_transfer=backend == "native", enable_nvtx=False,
                 enable_gdr=True, gdr_backend=backend, extra_link_args=[])
    exec(compile(ast.Module(body=module.body[start:start + 2], type_ignores=[]),
                 "setup.py", "exec"), scope)
    assert "-DFLEXKV_ENABLE_GDR" in scope["capability_flags"]
    assert "-libverbs" in scope["extra_link_args"]
    assert ("-DFLEXKV_GDR_MAPPED" in scope["capability_flags"]) == (backend == "mapped")
    assert ("-lcuda" in scope["extra_link_args"]) == (backend == "native")


def test_worker_factory_only_passes_supported_keywords():
    engine_tree = ast.parse((ROOT / "flexkv/transfer/transfer_engine.py").read_text())
    worker_tree = ast.parse((ROOT / "flexkv/transfer/workers/gpu_cpu.py").read_text())
    cls = next(node for node in worker_tree.body
               if isinstance(node, ast.ClassDef) and node.name == "GPUCPUTransferWorker")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    allowed = {arg.arg for arg in init.args.args + init.args.kwonlyargs} | {"mp_ctx", "backend"}
    factory = next(node for node in ast.walk(engine_tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "_create_gpu_cpu_worker")
    calls = [node for node in ast.walk(factory) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "create_worker"]
    assert len(calls) == 1
    assert {kw.arg for kw in calls[0].keywords if kw.arg} <= allowed
    worker_source = (ROOT / "flexkv/transfer/workers/gpu_cpu.py").read_text()
    for name in ("h2d_layer_holders", "layer_shard_enabled", "owner =", "cp_cache_node_rank"):
        assert name not in worker_source


def _cythonized_sources():
    module = ast.parse((ROOT / "setup.py").read_text())
    lists = {}
    for node in ast.walk(module):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("python_files", "excluded_files")):
            lists[node.targets[0].id] = ast.literal_eval(node.value)
    included = {path for pattern in lists["python_files"] for path in ROOT.glob(pattern)}
    excluded = {path for pattern in lists["excluded_files"] for path in ROOT.glob(pattern)}
    return sorted(included - excluded)


def test_cythonized_modules_have_no_negative_integer_subscripts():
    directives = (ROOT / "setup.py").read_text()
    assert '"wraparound": False' in directives
    sources = _cythonized_sources()
    assert any(path.name == "connector.py" for path in sources)
    assert any(path.name == "kvtask.py" for path in sources)
    offenders = []
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.UnaryOp)
                    and isinstance(node.slice.op, ast.USub)):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {ast.unparse(node)}")
    assert offenders == []
