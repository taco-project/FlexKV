"""CPU-only execution of cache-planning methods without the native extension."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    method.returns = None
    for node in ast.walk(method):
        if isinstance(node, ast.arg):
            node.annotation = None
    namespace = dict(np=np, os=os, nvtx=Mock(), get_nvtx_default_color=lambda: 0)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


block_range = load_method("flexkv/cache/cache_engine.py", "GlobalCacheEngine", "_get_block_range")
get_match = load_method("flexkv/kvtask.py", "KVTaskEngine", "get_match")


@pytest.mark.parametrize("enabled", ["0", "1"])
@pytest.mark.parametrize("size", [0, 1, 63, 1023, 1024, 4096])
def test_block_bounds_match_legacy_with_holes(monkeypatch, enabled, size):
    monkeypatch.setenv("FLEXKV_EXEC_EFFICIENCY_OPT", enabled)
    rng = np.random.default_rng(17)
    engine = SimpleNamespace(tokens_per_block=64)
    for mode in ("empty", "full", "holes", "trailing_false"):
        mask = rng.integers(0, 2, size=size).astype(bool)
        if mode == "empty":
            mask[:] = False
        elif mode == "full":
            mask[:] = True
        elif size:
            mask[size - 1] = mode == "holes"
        ids = np.where(mask)[0]
        expected = (0, 0) if not ids.size else (int(ids[0]) // 64, int(ids[ids.size - 1]) // 64 + 1)
        before = mask.copy()
        assert block_range(engine, mask) == expected
        np.testing.assert_array_equal(mask, before)


def test_large_mask_fast_path_avoids_index_array(monkeypatch):
    monkeypatch.setenv("FLEXKV_EXEC_EFFICIENCY_OPT", "1")
    mask = np.ones(4096, dtype=bool)
    mask[:65] = False
    monkeypatch.setattr(np, "where", Mock(side_effect=AssertionError("unexpected index allocation")))
    assert block_range(SimpleNamespace(tokens_per_block=64), mask) == (1, 64)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_noncontiguous_and_integer_masks_keep_legacy_bounds(monkeypatch, enabled):
    monkeypatch.setenv("FLEXKV_EXEC_EFFICIENCY_OPT", enabled)
    backing = np.arange(8192) % 3 == 0
    for mask in (backing[::2], backing.astype(np.int64)):
        ids = np.where(mask)[0]
        expected = (int(ids[0]) // 64, int(ids[ids.size - 1]) // 64 + 1)
        assert block_range(SimpleNamespace(tokens_per_block=64), mask) == expected


def engine_for_match():
    return SimpleNamespace(
        _update_tasks=Mock(),
        _get_match_impl=Mock(return_value=(31, np.array([True]))),
        tracer=SimpleNamespace(trace_request=Mock()),
    )


@pytest.mark.parametrize("enabled", ["0", "1"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("kind", ["none", "empty", "holes", "indices"])
def test_placeholder_matches_legacy_dtype_shape_and_arguments(monkeypatch, enabled, dtype, kind):
    monkeypatch.setenv("FLEXKV_EXEC_EFFICIENCY_OPT", enabled)
    tokens = np.arange(2048, dtype=dtype)
    mask = {"none": None, "empty": np.zeros(2048, dtype=bool),
            "holes": np.arange(2048) % 3 == 0,
            "indices": np.array([0, 7, 1000])}[kind]
    expected = np.zeros_like(tokens if mask is None else tokens[mask])
    engine = engine_for_match()
    result = get_match(engine, tokens, token_mask=mask, task_id=19, namespace=["test"], swa_aware=True)
    actual = engine._get_match_impl.call_args.args[1]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert engine._get_match_impl.call_args.kwargs["swa_aware"] is True
    assert engine._get_match_impl.call_args.kwargs["namespace"] == ["test"]
    assert result[0] == 31
    engine._update_tasks.assert_called_once_with(timeout=0)


def test_boolean_placeholder_does_not_gather_tokens(monkeypatch):
    monkeypatch.setenv("FLEXKV_EXEC_EFFICIENCY_OPT", "1")

    class NoGather(np.ndarray):
        def __getitem__(self, key):
            raise AssertionError("token gather is unnecessary for a placeholder")

    tokens = np.arange(2048).view(NoGather)
    engine = engine_for_match()
    get_match(engine, tokens, token_mask=np.ones(2048, dtype=bool))
    assert engine._get_match_impl.call_args.args[1].shape == (2048,)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_wrong_mask_shape_still_raises_before_planning(monkeypatch, enabled):
    monkeypatch.setenv("FLEXKV_EXEC_EFFICIENCY_OPT", enabled)
    engine = engine_for_match()
    with pytest.raises(IndexError):
        get_match(engine, np.arange(10), token_mask=np.ones(9, dtype=bool))
    engine._get_match_impl.assert_not_called()
