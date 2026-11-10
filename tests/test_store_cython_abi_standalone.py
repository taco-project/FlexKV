"""Compile the production store_kv method; CPU endpoints replace GPU/RDMA only.

Run with Python, numpy, setuptools, Cython and a C compiler. Missing build
dependencies fail rather than silently replacing this with a Python-only test.
The legacy variant changes only token_ids back to List[int] to reproduce the
2026-11-05 first-request crash on both leader and follower.
"""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

from serving_cpu_test_support import Serving, Tensor, finish
from submission_test_support import ROOT


class StoreCythonABITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import Cython  # noqa: F401 -- this suite requires a real compiler
        cls.directory = tempfile.TemporaryDirectory(prefix="flexkv-store-cython-")
        cls.addClassCleanup(cls.directory.cleanup)
        directory = Path(cls.directory.name)
        tree = ast.parse((ROOT / "flexkv/integration/sglang/connector.py").read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "FlexKVConnector")
        method = next(n for n in owner.body if getattr(n, "name", "") == "store_kv")
        old = copy.deepcopy(method)
        next(a for a in old.args.args if a.arg == "token_ids").annotation = ast.parse(
            "List[int]", mode="eval").body
        header = ast.parse("""from __future__ import annotations
from typing import Any, Callable, List, Optional, Sequence, Union
import numpy as np
import time
torch = None
put_trace = None
logger = None
FlexKVScatterChannel = None
_SGLANG_REQ_ID_UNSET = object()
""")
        for name, node in (("Current", method), ("Legacy", old)):
            header.body.append(ast.ClassDef(name=name, bases=[], keywords=[],
                                           body=[node], decorator_list=[]))
        (directory / "store_abi.py").write_text(ast.unparse(ast.fix_missing_locations(header)))
        # Match the repository's release compiler directives, including typing
        # defaults. Do not disable annotation_typing to make the test pass.
        setup_tree = ast.parse((ROOT / "setup.py").read_text())
        call = next(n for n in ast.walk(setup_tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "cythonize")
        directives = ast.literal_eval(next(k.value for k in call.keywords
                                          if k.arg == "compiler_directives"))
        (directory / "setup.py").write_text(
            "from setuptools import setup\nfrom Cython.Build import cythonize\n"
            f"setup(ext_modules=cythonize('store_abi.py', compiler_directives={directives!r}))\n")
        result = subprocess.run([sys.executable, "-B", "setup.py", "build_ext", "--inplace"],
                                cwd=directory, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        extension = next(directory.glob("store_abi*.so"))
        spec = importlib.util.spec_from_file_location("store_abi", extension)
        cls.compiled = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.compiled)

    def serving(self):
        service = Serving()
        self.addCleanup(service.close)
        for name in ("torch", "put_trace", "logger", "FlexKVScatterChannel"):
            setattr(self.compiled, name, getattr(service.harness.module, name))
        service.conn.store_kv = MethodType(self.compiled.Current.store_kv, service.conn)
        return service

    def test_legacy_compiled_binding_reproduces_both_reported_errors(self):
        for ids in (range(4096), np.arange(4096, dtype=np.int64)):
            with self.subTest(kind=type(ids).__name__):
                with self.assertRaisesRegex(TypeError, "token_ids.*incorrect type.*expected list"):
                    self.compiled.Legacy.store_kv(None, "r", ids, None)

    def test_compiled_leader_stores_full_pages_for_all_supported_inputs(self):
        for kind in (list, tuple, np.asarray):
            for lazy in (False, True):
                with self.subTest(kind=kind.__name__, lazy=lazy):
                    service = self.serving()
                    p = service.p
                    p.gpu[8:12] = p.payload
                    ids = kind(p.ids)
                    slots = Tensor(np.arange(8 * 64, 12 * 64), np.int64)
                    provider = Mock(return_value=slots)
                    tid = service.conn.store_kv("r", ids, provider if lazy else slots,
                        namespace=p.codec.cache_namespace("image:content", True))
                    self.assertGreaterEqual(tid, 0)
                    finish(p, tid)
                    match = p.manager.cpu_cache_engine.match(p.sequence)
                    np.testing.assert_array_equal(p.host[match.physical_blocks], p.payload)
                    if lazy:
                        provider.assert_called_once_with()

    def test_compiled_follower_preserves_range_and_lazy_source_ownership(self):
        service = self.serving()
        obj = service.conn
        obj._sync_ctx = NS(is_sync_leader=False, needs_sync=True,
            should_send_slot_mapping_to_remote=False,
            scatter=lambda *a, **kw: dict(rid="r", task_id=17, active=True,
                slot_count=4096, unmatched_mask=[], error=""))
        obj.kv_manager = Mock()
        source = Mock(return_value=Tensor(np.arange(4096), np.int64))
        self.assertEqual(obj.store_kv("r", range(4096), source), 17)
        self.assertEqual(obj._inflight_stores["r"], 17)
        source.assert_called_once_with()
        obj.kv_manager.put_match.assert_not_called()
        obj.kv_manager.launch.assert_not_called()

    def test_compiled_argument_validation_still_rejects_length_mismatch(self):
        service = self.serving()
        with self.assertRaisesRegex(ValueError, "token_ids has 64 entries"):
            service.conn.store_kv("r", np.arange(64), Tensor(np.arange(63)))


if __name__ == "__main__":
    unittest.main()
