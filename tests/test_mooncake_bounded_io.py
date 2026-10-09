"""SDK contract tests; no Mooncake server or device transfer is started."""

from types import SimpleNamespace

import pytest

from flexkv.external.mooncake_store_utils import MooncakeStoreClient, MooncakeStoreConfig


def client(get_results=None, put_results=None, limit=8, exists=None):
    calls = []

    def batch_get_into(keys, pointers, sizes):
        calls.append(("get", list(keys), list(pointers), list(sizes)))
        return [get_results[k] for k in keys] if get_results is not None else list(sizes)

    def batch_put_from(keys, pointers, sizes):
        calls.append(("put", list(keys), list(pointers), list(sizes)))
        return [put_results[k] for k in keys] if put_results is not None else [0] * len(keys)

    c = MooncakeStoreClient.__new__(MooncakeStoreClient)
    c._is_setup = True
    c._config = MooncakeStoreConfig(max_get_batch_bytes=limit, max_put_batch_bytes=limit)
    c._store = SimpleNamespace(
        batch_get_into=batch_get_into,
        batch_put_from=batch_put_from,
        batch_is_exist=lambda keys: [int(k in (exists or [])) for k in keys],
    )
    return c, calls


@pytest.mark.parametrize("actual", [-1, 0, 3, 5])
def test_single_get_rejects_non_exact_read(actual):
    c, calls = client(get_results={"a": actual})
    assert c.get("a", 100, 4) is False
    assert len(calls) == 1  # no retry


@pytest.mark.parametrize("limit,groups", [(8, [["a", "b"], ["c"], ["d"]]), (0, [["a", "b", "c", "d"]])])
def test_get_batches_preserve_order_and_partial_results(limit, groups):
    c, calls = client(get_results={"a": 4, "b": -1, "c": 4, "d": 5}, limit=limit)
    assert c.batch_get(list("abcd"), [10, 20, 30, 40], [4, 4, 4, 5]) == [True, False, True, True]
    assert [call[1] for call in calls] == groups
    assert [p for call in calls for p in call[2]] == [10, 20, 30, 40]
    assert all(sum(call[3]) <= limit for call in calls) if limit else True


def test_put_batches_keep_existing_keys_and_per_key_status():
    c, calls = client(put_results={"a": 0, "c": -1, "d": 0}, exists=["b"])
    assert c.batch_put(list("abcd"), [10, 20, 30, 40], [4, 4, 5, 4]) == [True, True, False, True]
    assert [call[1] for call in calls] == [["a"], ["c"], ["d"]]
    assert c.put("a", 10, 4) is True


@pytest.mark.parametrize("operation", ["batch_get", "batch_put"])
@pytest.mark.parametrize("sizes", [[4], [4, 0], [4, -1], [4, 9]])
def test_invalid_buffers_fail_before_io(operation, sizes):
    c, calls = client()
    with pytest.raises(ValueError):
        getattr(c, operation)(["a", "b"], [10, 20], sizes)
    assert calls == []


@pytest.mark.parametrize(
    "operation,sdk", [("batch_get", "batch_get_into"), ("batch_put", "batch_put_from"), ("batch_put", "batch_is_exist")]
)
@pytest.mark.parametrize("results", [None, [], [0, 0]])
def test_malformed_result_count_raises(operation, sdk, results):
    c, _ = client()

    def malformed(*args):
        return results

    setattr(c._store, sdk, malformed)
    with pytest.raises(RuntimeError, match="results for 1 keys"):
        getattr(c, operation)(["a"], [10], [4])


@pytest.mark.parametrize("operation", ["batch_get", "batch_put"])
def test_empty_batch_does_not_call_sdk(operation):
    c, calls = client()
    assert getattr(c, operation)([], [], []) == []
    assert calls == []


@pytest.mark.parametrize("limit", [-1, 1.5, True, "8"])
def test_invalid_budget_rejected(limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        MooncakeStoreConfig(max_get_batch_bytes=limit)
