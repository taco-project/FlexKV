"""Automatic pool budget and production planner/engine wiring, without native DMA."""

import ast
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import pytest

from flexkv.prefetch.budget import DEFAULT_PREFETCH_MAX_PINNED_BYTES, resolve_prefetch_budget
from flexkv.prefetch.types import PrefetchCapacityExhausted

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


def execute(nodes, path, scope):
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


def function(filename, name, **scope):
    path = ROOT / filename
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return execute([node], path, scope)[name]


@pytest.fixture
def geometry(monkeypatch):
    # Execute the real block sizing function; only the optional native-facing
    # config types are substituted. Fractional compressed tokens must stay exact.
    config = ModuleType("flexkv.common.config")
    config.block_size_in_bytes_for_cache = function("flexkv/common/config.py", "block_size_in_bytes_for_cache")
    config.RankInfo = lambda model: NS(token_size_in_bytes_per_pp_stage=model.token_bytes)
    config.GLOBAL_CONFIG_FROM_ENV = NS(kv_shared_across_ranks_mode="sharded")
    monkeypatch.setitem(sys.modules, "flexkv.common.config", config)
    model = NS(num_kv_heads=2, token_bytes=GIB // 4, layer_groups=None,
               effective_tp_size_per_node=4, tp_size=1, kv_dim=1, dtype=NS(itemsize=1),
               nnodes=1, use_trtllm_subprocess=False)
    cache = NS(tokens_per_block=4, num_cpu_blocks=512, swa=None,
               prefetch_max_pinned_bytes=DEFAULT_PREFETCH_MAX_PINNED_BYTES)
    return config, model, cache


@pytest.mark.parametrize("blocks,expected", [
    (512, 128), (258, 128), (257, 127), (256, 127),
    (10, 4), (5, 1), (3, 0), (1, 0),
])
def test_default_limit_reserves_half_the_actual_l2_and_one_block(geometry, blocks, expected):
    _, model, cache = geometry
    cache.num_cpu_blocks = blocks
    budget = resolve_prefetch_budget(model, cache)
    assert budget.byte_limit == expected * GIB
    assert DEFAULT_PREFETCH_MAX_PINNED_BYTES == 128 * GIB


def test_runtime_physical_capacity_takes_precedence_over_config(geometry):
    _, model, cache = geometry
    assert resolve_prefetch_budget(model, cache, cpu_blocks=4).byte_limit == GIB


@pytest.mark.parametrize("replicas", [1, 4])
def test_exact_compressed_blocks_and_registered_sidecars(geometry, replicas):
    config, model, cache = geometry
    model.num_kv_heads = 1
    model.layer_groups = [NS(compress_ratio=4, num_layers=1, num_kv_heads=1, head_size=3, dtype=None)]
    config.GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode = "all_write" if replicas == 4 else "sharded"
    cache.num_cpu_blocks = 256
    cache.swa = NS(enabled=True, multi_group=True, snapshot_bytes=11)
    budget = resolve_prefetch_budget(model, cache)
    assert budget.block_bytes == 3 * replicas  # rounding bytes/token would give zero
    assert budget.swa_slot_bytes == 11 * replicas
    assert budget.byte_limit == 127 * 3 * replicas
    assert resolve_prefetch_budget(model, cache, swa_aware=False).swa_slot_bytes == 0


def test_multi_group_snapshot_must_have_registered_geometry(geometry):
    _, model, cache = geometry
    cache.swa = NS(enabled=True, multi_group=True, snapshot_bytes=None)
    with pytest.raises(ValueError, match="snapshot geometry"):
        resolve_prefetch_budget(model, cache)


def planner_class():
    path = ROOT / "flexkv/prefetch/planner.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
               and n.name == "MooncakeChunkPlanner")
    return execute([cls], path, dict(ThreadPoolExecutor=ThreadPoolExecutor, threading=threading,
                   copy=copy, PrefetchCapacityExhausted=PrefetchCapacityExhausted))[cls.name]


def planner_fixture(geometry, blocks=512):
    _, model, cache = geometry
    cpu = NS(num_total_blocks=blocks, lock_node=lambda node: None, unlock=lambda node: None)
    engine = NS(model_config=model, cache_config=cache, cache_engine=NS(cpu_cache_engine=cpu),
                _prefetch=NS(reserved_bytes=0))
    return planner_class()(engine), cpu


def test_backend_pins_can_cross_2_gib_but_include_inflight_credit(geometry):
    planner, cpu = planner_fixture(geometry)
    try:
        node = NS(size=lambda: 4, parent=None)
        cpu.match = lambda prefix: NS(num_matched_blocks=4, last_node=node)
        context = NS(sequence=NS(token_ids=list(range(16)), block_hashes=list(range(4))),
                     swa_aware=False, node=None, swa_node=None, pinned_bytes=0)
        assert planner.max_pinned_bytes == 128 * GIB
        planner.engine._prefetch.reserved_bytes = 125 * GIB
        assert not planner._pin_prefix(context, 16)  # 4 + 125 exceeds 128
        assert planner.pinned_bytes == 0
        planner.engine._prefetch.reserved_bytes = 124 * GIB
        assert planner._pin_prefix(context, 16)
        assert planner.pinned_bytes == context.pinned_bytes == 4 * GIB
        planner.engine._prefetch.reserved_bytes = 125 * GIB
        assert not planner._pin_prefix(context, 16)
        assert context.pinned_bytes == 4 * GIB  # failure cannot release its lease
    finally:
        planner.executor.shutdown()


def test_task_engine_initializes_the_real_planner_with_automatic_budget(geometry, monkeypatch):
    _, model, cache = geometry
    cache.enable_chunked_prefetch = True
    cache.use_mooncake_store_backend = True
    cache.enable_ssd = cache.enable_kv_sharing = False
    cache.prefetch_options = {}
    cache.prefetch_max_sessions = 128
    cache.prefetch_max_reserved_bytes = 512 * 1024**2
    cache.prefetch_result_ttl_s = 60
    module = ModuleType("flexkv.prefetch.planner")
    module.MooncakeChunkPlanner = planner_class()
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Base:
        def __init__(self, model, cache, *args):
            self.model_config, self.cache_config = model, cache
            self.cache_engine = NS(cpu_cache_engine=NS(num_total_blocks=9))

    path = ROOT / "flexkv/kvtask.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
               and n.name == "KVTaskEngine")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    scope = dict(KVTaskManager=Base, FlexKVTracer=lambda: NS(trace_config=lambda *a, **kw: None),
                 get_global_collector=lambda: None, os=os)
    engine_type = execute([cls], path, scope)["KVTaskEngine"]
    engine = engine_type(model, cache)
    try:
        assert engine._prefetch_backend.max_pinned_bytes == 3 * GIB
        assert engine._prefetch.backend is engine._prefetch_backend
    finally:
        engine._prefetch_backend.executor.shutdown()


def config_type(name):
    # Load the actual field declarations without the optional native imports.
    path = ROOT / "flexkv/common/config.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == name)
    cls.body = [n for n in cls.body if isinstance(n, ast.AnnAssign)
                and n.target.id in ("enable_chunked_prefetch", "prefetch_max_pinned_bytes")]
    return execute([cls], path, dict(dataclass=dataclass,
        DEFAULT_PREFETCH_MAX_PINNED_BYTES=DEFAULT_PREFETCH_MAX_PINNED_BYTES))[name]


@pytest.mark.parametrize("name", ["UserConfig", "CacheConfig"])
def test_config_defaults_share_the_128_gib_ceiling(name):
    assert config_type(name)().prefetch_max_pinned_bytes == 128 * GIB


@pytest.mark.parametrize("setting", [None, 2 * GIB, 64 * GIB])
def test_file_limit_defaults_to_128_gib_and_honors_explicit_values(tmp_path, setting):
    config = {"enable_chunked_prefetch": True}
    if setting is not None:
        config["prefetch_max_pinned_bytes"] = setting
    load = function("flexkv/common/config.py", "load_user_config_from_file",
                    UserConfig=config_type("UserConfig"), fields=fields, json=json)
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(config))
    cfg = load(str(path))
    assert cfg.enable_chunked_prefetch
    assert cfg.prefetch_max_pinned_bytes == (128 * GIB if setting is None else setting)
    assert not hasattr(cfg, "override_prefetch_max_pinned_bytes")


@pytest.mark.parametrize("setting", [None, 2 * GIB, 256 * GIB])
def test_environment_limit_defaults_to_128_gib_and_honors_explicit_values(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("FLEXKV_PREFETCH_MAX_PINNED_BYTES", raising=False)
    else:
        monkeypatch.setenv("FLEXKV_PREFETCH_MAX_PINNED_BYTES", str(setting))
    load = function("flexkv/common/config.py", "load_user_config_from_env", UserConfig=NS,
                    os=os, json=json, parse_path_list=lambda value: [value],
                    _is_rocm_runtime=lambda: False, _parse_swa_multi_group_env=lambda value: None,
                    DEFAULT_PREFETCH_MAX_PINNED_BYTES=DEFAULT_PREFETCH_MAX_PINNED_BYTES)
    cfg = load()
    assert cfg.prefetch_max_pinned_bytes == (128 * GIB if setting is None else setting)


@pytest.mark.parametrize("configured,blocks,expected", [(2, 512, 2), (64, 10, 4), (256, 512, 255)])
def test_configured_limit_and_half_pool_capacity_both_apply(geometry, configured, blocks, expected):
    _, model, cache = geometry
    cache.prefetch_max_pinned_bytes = configured * GIB
    assert resolve_prefetch_budget(model, cache, cpu_blocks=blocks).byte_limit == expected * GIB


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, float("inf")])
def test_invalid_configured_limit_is_rejected(geometry, limit):
    _, model, cache = geometry
    cache.prefetch_max_pinned_bytes = limit
    with pytest.raises(ValueError, match="prefetch_max_pinned_bytes"):
        resolve_prefetch_budget(model, cache)


def test_user_limit_is_copied_to_runtime_cache_config():
    class GeometryBoundary(Exception):
        pass
    def sizing(*args):
        raise GeometryBoundary
    copy_config = function("flexkv/common/config.py", "update_default_config_from_user_config",
                           block_size_in_bytes_for_cache=sizing)
    user = NS(enable_chunked_prefetch=True, prefetch_options={}, prefetch_max_sessions=128,
              prefetch_max_reserved_bytes=512 * 1024**2, prefetch_max_pinned_bytes=64 * GIB,
              prefetch_result_ttl_s=60)
    cache = NS(prefetch_max_pinned_bytes=128 * GIB)
    with pytest.raises(GeometryBoundary):
        copy_config(NS(model_config=object()), cache, user)
    assert cache.prefetch_max_pinned_bytes == 64 * GIB


def test_pool_without_speculative_capacity_rejects_instead_of_waiting(geometry):
    planner, _ = planner_fixture(geometry, blocks=1)
    try:
        with pytest.raises(PrefetchCapacityExhausted, match="speculative block"):
            planner.begin([], None, 0, NS(swa_aware=False))
    finally:
        planner.executor.shutdown()


def test_clients_share_the_backend_pin_budget(geometry):
    planner, cpu = planner_fixture(geometry)
    def match(prefix):
        blocks = len(prefix.block_hashes)
        node = NS(size=lambda: blocks, parent=None)
        return NS(num_matched_blocks=blocks, last_node=node,
                  num_ready_matched_blocks=blocks, last_ready_node=node)
    cpu.match = match
    def context(blocks):
        return NS(sequence=NS(token_ids=list(range(4 * blocks)), block_hashes=list(range(blocks))),
                  swa_aware=False, node=None, swa_node=None, pinned_bytes=0)
    try:
        first, second, tail = context(80), context(48), context(1)
        assert planner._pin_prefix(first, 320)
        assert planner._pin_prefix(second, 192)
        assert planner.pinned_bytes == 128 * GIB
        assert not planner._pin_prefix(tail, 4)
        assert first.pinned_bytes == 80 * GIB and second.pinned_bytes == 48 * GIB
        assert tail.pinned_bytes == 0
    finally:
        planner.executor.shutdown()
