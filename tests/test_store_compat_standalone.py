"""CPU-only store ownership and prefix-boundary contracts."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def store_method():
    tree = ast.parse((ROOT / "flexkv/integration/sglang/connector.py").read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "FlexKVConnector")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "store_kv")
    method.returns = None
    for node in ast.walk(method):
        if isinstance(node, ast.arg):
            node.annotation = None
    namespace = dict(np=np, _SGLANG_REQ_ID_UNSET=object(), non_kernel_stack_mode=lambda: False,
                     FlexKVScatterChannel=SimpleNamespace(STORE_START=1))
    exec(compile(ast.Module(body=[method], type_ignores=[]), "connector.py", "exec"), namespace)
    return namespace["store_kv"]


store_kv = store_method()


class Slots(np.ndarray):
    def numel(self):
        return self.size


def slots(n):
    return np.arange(n, dtype=np.int64).view(Slots)


def connector(*, leader=True, payload=None):
    sync = SimpleNamespace(is_sync_leader=leader, needs_sync=not leader,
                           is_cross_node_pp=False, should_send_slot_mapping_to_remote=False,
                           scatter=Mock(return_value=payload))
    manager = SimpleNamespace(
        put_match=Mock(side_effect=lambda **kwargs: (17, np.ones(len(kwargs["token_ids"]), dtype=bool))),
        launch=Mock(return_value=[17]),
        cancel=Mock(),
    )
    return SimpleNamespace(
        _sync_ctx=sync, kv_manager=manager, page_size=1, _swa_kv_pool=None,
        _inflight_stores={}, _inflight_store_contexts={},
        _new_op_context=lambda *args: SimpleNamespace(task_id=-1),
        _store_profile_scope=lambda *args: nullcontext(),
        _log_cache_op=Mock(), _to_cpu_int64=lambda value: value,
        _build_swa_slot_mapping=lambda value: None,
    )


def test_launch_failure_keeps_real_task_id_and_context():
    engine = connector()

    def partial_launch(**kwargs):
        assert engine._inflight_stores["r"] == 17
        raise RuntimeError("partial submission")

    engine.kv_manager.launch.side_effect = partial_launch
    with pytest.raises(RuntimeError, match="partial submission"):
        store_kv(engine, "r", list(range(8)), slots(8))
    assert engine._inflight_stores == {"r": 17}
    assert engine._inflight_store_contexts["r"].task_id == 17


@pytest.mark.parametrize("error,active,rid", [("partial submission", False, "r"),
                                              ("", True, "wrong-rid")])
def test_follower_retains_writer_before_payload_validation(error, active, rid):
    payload = dict(rid=rid, task_id=17, active=active, error=error, slot_count=8)
    engine = connector(leader=False, payload=payload)
    with pytest.raises(RuntimeError):
        store_kv(engine, "r", list(range(8)), slots(8))
    assert engine._inflight_stores == {"r": 17}
    assert engine._inflight_store_contexts["r"].task_id == 17
    engine.kv_manager.launch.assert_not_called()


def test_success_retains_source_until_completion():
    engine = connector()
    assert store_kv(engine, "r", list(range(8)), slots(8), namespace=["cache"]) == 17
    assert engine._inflight_stores == {"r": 17}
    assert engine.kv_manager.put_match.call_args.kwargs["namespace"] == ["cache"]
    assert engine.kv_manager.launch.call_args.kwargs["as_batch"] is False


@pytest.mark.parametrize("prefix", [0, 1, 3, 4, 7, 8])
def test_prefix_is_page_aligned_before_planning(prefix):
    engine = connector()
    engine.page_size = 4
    original = slots(8)
    result = store_kv(engine, "r", list(range(8)), original, prefix_length=prefix)
    expected = prefix // 4 * 4
    if expected == 0:
        assert result == -1
        engine.kv_manager.put_match.assert_not_called()
    else:
        assert result == 17
        np.testing.assert_array_equal(engine.kv_manager.put_match.call_args.kwargs["token_ids"],
                                      np.arange(expected))
        np.testing.assert_array_equal(engine.kv_manager.launch.call_args.kwargs["slot_mappings"][0],
                                      np.arange(expected))
    np.testing.assert_array_equal(original, np.arange(8))


@pytest.mark.parametrize("prefix", [-1, 9, True, 3.5])
def test_invalid_prefix_is_rejected_without_submission(prefix):
    engine = connector()
    with pytest.raises(ValueError, match="prefix_length"):
        store_kv(engine, "r", list(range(8)), slots(8), prefix_length=prefix)
    engine.kv_manager.put_match.assert_not_called()


def test_prefix_cannot_reuse_later_swa_snapshot():
    engine = connector()
    engine._swa_kv_pool = object()
    with pytest.raises(ValueError, match="SWA state"):
        store_kv(engine, "r", list(range(8)), slots(8), prefix_length=4)
    engine.kv_manager.put_match.assert_not_called()


def test_empty_page_size_one_store_does_not_allocate_task():
    engine = connector()
    assert store_kv(engine, "r", [], slots(0)) == -1
    engine.kv_manager.put_match.assert_not_called()


@pytest.mark.parametrize("cancel_error", [False, True])
def test_cached_store_retires_terminal_without_creating_writer(cancel_error):
    engine = connector()
    engine.kv_manager.put_match.side_effect = None
    engine.kv_manager.put_match.return_value = (17, np.zeros(8, dtype=bool))
    if cancel_error:
        engine.kv_manager.cancel.side_effect = RuntimeError("cancel transport failed")
        with pytest.raises(RuntimeError, match="cancel transport failed"):
            store_kv(engine, "r", list(range(8)), slots(8))
    else:
        assert store_kv(engine, "r", list(range(8)), slots(8)) == -1
    engine.kv_manager.cancel.assert_called_once_with([17])
    engine.kv_manager.launch.assert_not_called()
    assert not engine._inflight_stores


def test_follower_does_not_retain_negative_id_on_match_failure():
    engine = connector(leader=False, payload=dict(rid="r", task_id=-1, active=False, error="match failed"))
    with pytest.raises(RuntimeError, match="match failed"):
        store_kv(engine, "r", list(range(8)), slots(8))
    assert not engine._inflight_stores


@pytest.mark.parametrize("prefix", [None, 4, 8])
def test_non_kernel_stack_copies_complete_indices_before_device_slice(monkeypatch, prefix):
    class DeviceSlots:
        is_cuda = True

        def __len__(self):
            return 9

        def __getitem__(self, key):
            raise AssertionError("device slicing must not run")

    monkeypatch.setitem(store_kv.__globals__, "non_kernel_stack_mode", lambda: True)
    monkeypatch.setitem(store_kv.__globals__, "torch", SimpleNamespace(Tensor=DeviceSlots))
    device = DeviceSlots()
    host = slots(9)
    copies = []
    engine = connector()
    engine.page_size = 4

    def to_cpu(value):
        if isinstance(value, DeviceSlots):
            copies.append(value)
            return host
        return value

    engine._to_cpu_int64 = to_cpu
    assert store_kv(engine, "r", list(range(9)), device, prefix_length=prefix) == 17
    assert copies == [device]
    assert len(engine.kv_manager.launch.call_args.kwargs["slot_mappings"][0]) == (prefix or 8)


def test_default_store_does_not_copy_entire_device_indices(monkeypatch):
    monkeypatch.setitem(store_kv.__globals__, "non_kernel_stack_mode", lambda: False)
    engine = connector()
    engine.page_size = 4
    indices = slots(9)
    copies = []
    engine._to_cpu_int64 = lambda value: copies.append(value.size) or value
    assert store_kv(engine, "r", list(range(9)), indices) == 17
    assert copies == [8]
