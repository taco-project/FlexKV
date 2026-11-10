import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "flexkv/transfer" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_put_trace_is_disabled_by_default(monkeypatch, tmp_path):
    for key in list(__import__("os").environ):
        if key.startswith("FLEXKV_PUT_TRACE"):
            monkeypatch.delenv(key)
    module = _load("put_trace")
    assert not module.ENABLED
    module.worker_begin(object(), 1)
    module.worker_end(object(), 2, "ok")
    assert module._fd is None


def test_cache_probe_disabled_has_no_cuda_import_side_effect(monkeypatch):
    monkeypatch.delenv("FLEXKV_CACHE_PROBE", raising=False)
    module = _load("cache_probe")
    # Gate is at the call site (FLEXKV_CACHE_PROBE=1); the module itself has no
    # ENABLED flag and must stay importable without touching CUDA.
    assert not hasattr(module, "ENABLED")
    assert "torch" not in getattr(module, "__dict__", {})
    assert 'os.environ.get("FLEXKV_CACHE_PROBE") == "1"' in (
        ROOT / "flexkv/transfer/workers/runtime.py").read_text()



def test_runtime_probe_is_gated_and_after_launch():
    source = (ROOT / "flexkv/transfer/workers/runtime.py").read_text()
    assert 'os.environ.get("FLEXKV_CACHE_PROBE") == "1"' in source
    assert source.index("transfer_status = self.launch_transfer(op)") < source.index("verify_transfer(self, op")


def test_default_collection_excludes_hardware_gdr_suites():
    source = (ROOT / "tests/gdr/conftest.py").read_text()
    assert "FLEXKV_RUN_GDR_TESTS" in source
    assert "collect_ignore" in source
