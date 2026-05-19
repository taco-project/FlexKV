"""Unit tests for Phase 0 task 0-A / 0-J CacheConfig additions.

These tests cover:

* ``RemoteEndpoint`` dataclass construction & field names
  (task 0-J scaffolding for Remote-endpoint discovery).
* ``CacheConfig.remote_endpoints_by_sd`` default value and basic usage.
* ``CacheConfig.enable_sharing_domain`` auto-derivation from P2P flags
  (task 0-A ``__post_init__`` behaviour).
* ``CacheConfig`` sharing-domain field defaults match plan.md §1.2 0-A.

We bypass the ``tests/conftest.py`` torch-heavy import pipeline via
``--noconftest`` — all we need here is the config module itself and the
stdlib dataclass machinery.  We **do** import ``torch`` indirectly (it's a
hard dependency of ``flexkv.common.config``), so the test machine needs a
CPU-only torch install but nothing CUDA.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

# Important: loading ``flexkv.common.config`` from source so we pick up
# the freshly-edited file instead of the stale compiled ``config.so``
# that sits next to it on some deployment machines.  ``flexkv.common``
# itself has an empty ``__init__``, so importing sub-modules does not
# pull in the ``flexkv.cache`` package (which requires CUDA-linked c_ext).
pkg_root = Path(__file__).resolve().parent.parent


def _load_config():
    """Load ``flexkv.common.config`` preferring the ``.py`` source.

    If a stale ``config.so`` shadows our ``.py`` on ``sys.path`` we
    side-step it by loading the source file directly.  That keeps the
    tests relevant regardless of whether the target environment has
    rebuilt its Cython/pybind ``flexkv/common/config.so`` artefact.
    """
    src = pkg_root / "flexkv" / "common" / "config.py"
    assert src.exists(), f"missing source file {src}"
    spec = importlib.util.spec_from_file_location(
        "_cfg_under_test", str(src),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # so @dataclass can resolve the module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg_mod():
    return _load_config()


# ---------------------------------------------------------------------------
# RemoteEndpoint
# ---------------------------------------------------------------------------
class TestRemoteEndpoint:
    def test_construct(self, cfg_mod):
        ep = cfg_mod.RemoteEndpoint(
            ip="10.0.0.1",
            gpu_register_port="6001",
            command_port="6002",
            result_port="6003",
        )
        assert ep.ip == "10.0.0.1"
        assert ep.gpu_register_port == "6001"
        assert ep.command_port == "6002"
        assert ep.result_port == "6003"

    def test_is_dataclass(self, cfg_mod):
        assert dataclasses.is_dataclass(cfg_mod.RemoteEndpoint)
        field_names = {f.name for f in dataclasses.fields(cfg_mod.RemoteEndpoint)}
        assert field_names == {"ip", "gpu_register_port", "command_port", "result_port"}

    def test_fields_are_strings(self, cfg_mod):
        # The ``master_ports`` tuple that this mirrors is a tuple of str, so
        # we enforce the same here — users should str(int) explicitly.
        for f in dataclasses.fields(cfg_mod.RemoteEndpoint):
            assert f.type is str or f.type == "str"


# ---------------------------------------------------------------------------
# CacheConfig sharing-domain defaults
# ---------------------------------------------------------------------------
class TestCacheConfigDefaults:
    def test_sharing_domain_fields_exist(self, cfg_mod):
        cc = cfg_mod.CacheConfig()
        # All five sharing-domain knobs are present with documented defaults.
        assert cc.enable_sharing_domain is False
        assert cc.instance_id is None
        assert cc.session_epoch is None
        assert cc.instance_session_ttl_seconds == 8
        assert cc.instance_session_renew_interval_seconds == 3
        assert cc.refcount_leak_timeout_seconds == 30
        assert cc.refcount_leak_scan_interval_seconds == 10
        assert cc.remote_endpoints_by_sd == {}

    def test_remote_endpoints_default_is_fresh_per_instance(self, cfg_mod):
        """Guard against the classic ``default=[]`` dataclass foot-gun —
        each CacheConfig must get its own empty dict."""
        a = cfg_mod.CacheConfig()
        b = cfg_mod.CacheConfig()
        a.remote_endpoints_by_sd["sd0"] = cfg_mod.RemoteEndpoint(
            ip="x", gpu_register_port="1", command_port="2", result_port="3",
        )
        assert b.remote_endpoints_by_sd == {}


class TestPostInit:
    def test_enable_sharing_domain_on_p2p_cpu(self, cfg_mod):
        cc = cfg_mod.CacheConfig(enable_p2p_cpu=True)
        assert cc.enable_sharing_domain is True

    def test_enable_sharing_domain_on_p2p_ssd(self, cfg_mod):
        cc = cfg_mod.CacheConfig(enable_p2p_ssd=True)
        assert cc.enable_sharing_domain is True

    def test_enable_sharing_domain_off_by_default(self, cfg_mod):
        cc = cfg_mod.CacheConfig()
        assert cc.enable_sharing_domain is False

    def test_explicit_enable_sharing_domain_preserved(self, cfg_mod):
        cc = cfg_mod.CacheConfig(enable_sharing_domain=True)
        # Explicit enable must NOT be clobbered by the auto-derive logic.
        assert cc.enable_sharing_domain is True

    def test_enable_kv_sharing_still_auto_derived(self, cfg_mod):
        # Regression test: the pre-Batch-A behaviour of auto-deriving
        # ``enable_kv_sharing`` from p2p flags must still hold.
        cc = cfg_mod.CacheConfig(enable_p2p_cpu=True)
        assert cc.enable_kv_sharing is True
        assert cc.enable_remote is False  # p2p_cpu alone does NOT enable 3rd remote


# ---------------------------------------------------------------------------
# ModelConfig new properties
# ---------------------------------------------------------------------------
class TestModelConfigProperties:
    def test_tp_node_count_no_cross_node(self, cfg_mod):
        import torch
        mc = cfg_mod.ModelConfig(
            num_layers=16, num_kv_heads=8, head_size=128,
            dtype=torch.bfloat16, use_mla=False,
            tp_size=4, pp_size=1, nnodes=1,
        )
        assert mc.tp_node_count == 1
        # tp_node_idx is now per-rank (lives on RankInfo, not ModelConfig)
        ri = cfg_mod.RankInfo(model_config=mc, tp_rank=0, pp_rank=0)
        assert ri.tp_node_idx == 0

    def test_tp_node_count_cross_node(self, cfg_mod):
        import torch
        mc = cfg_mod.ModelConfig(
            num_layers=16, num_kv_heads=8, head_size=128,
            dtype=torch.bfloat16, use_mla=False,
            tp_size=8, pp_size=1, nnodes=2,
        )
        # 8 TP / 2 nodes = 4 per node; tp_rank=5 → node index 1
        assert mc.tp_node_count == 2
        ri = cfg_mod.RankInfo(model_config=mc, tp_rank=5, pp_rank=0)
        assert ri.tp_node_idx == 1

    def test_model_id_stable(self, cfg_mod):
        import torch
        a = cfg_mod.ModelConfig(
            num_layers=32, num_kv_heads=8, head_size=128,
            dtype=torch.bfloat16, use_mla=False,
        ).model_id
        b = cfg_mod.ModelConfig(
            num_layers=32, num_kv_heads=8, head_size=128,
            dtype=torch.bfloat16, use_mla=False,
            tp_size=4, pp_size=2,  # topology differs
        ).model_id
        # model_id must ignore topology and depend only on architecture.
        assert a == b
        assert len(a) == 16

    def test_model_id_changes_with_architecture(self, cfg_mod):
        import torch
        a = cfg_mod.ModelConfig(
            num_layers=32, num_kv_heads=8, head_size=128,
            dtype=torch.bfloat16, use_mla=False,
        ).model_id
        b = cfg_mod.ModelConfig(
            num_layers=32, num_kv_heads=8, head_size=128,
            dtype=torch.float16, use_mla=False,  # dtype differs
        ).model_id
        assert a != b
