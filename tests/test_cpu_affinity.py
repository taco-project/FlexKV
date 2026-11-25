import os

import pytest

from flexkv.common import numa
from flexkv.common.cpu_affinity import ENV_CPUS, apply_configured_cpu_affinity


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_spread_nodes_cache():
    numa.get_spread_nodes.cache_clear()
    yield
    numa.get_spread_nodes.cache_clear()


@pytest.mark.parametrize("spread", [False, True])
def test_explicit_cpu_affinity_is_independent_of_memory_spreading(monkeypatch, spread):
    monkeypatch.setattr(numa, "_read_status_list", lambda key: [0, 1])
    monkeypatch.setattr(numa, "node_cpus", lambda node: {2 * node, 2 * node + 1})
    monkeypatch.setattr(numa.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv(numa.ENV_NUMA_NODES, raising=False)
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, str(int(spread)))
    monkeypatch.setenv(ENV_CPUS, "1,3")
    affinity_calls = []
    memory_calls = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: affinity_calls.append((pid, cpus)), raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 3}, raising=False)
    listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: ["123"] if path == "/proc/self/task" else listdir(path))
    monkeypatch.setattr("flexkv.common.cpu_affinity.threading.get_native_id", lambda: 42)
    monkeypatch.setattr(numa, "set_task_mempolicy", lambda mode, nodes: memory_calls.append((mode, nodes)))

    numa.reset_inherited_numa_binding("TransferManager")

    assert affinity_calls == [(0, {1, 3}), (123, {1, 3})]
    expected = [(numa._MPOL_INTERLEAVE, (0, 1))] if spread else []
    assert memory_calls == expected


def test_in_process_worker_only_binds_calling_thread(monkeypatch):
    monkeypatch.setenv(ENV_CPUS, "4-5")
    calls = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, cpus)), raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5}, raising=False)

    assert apply_configured_cpu_affinity("TaskRuntime", all_threads=False)
    assert calls == [(0, {4, 5})]


@pytest.mark.parametrize("spec", ["9-4", "1,-2", "1,", "x"])
def test_invalid_explicit_cpu_affinity(monkeypatch, spec):
    monkeypatch.setenv(ENV_CPUS, spec)

    with pytest.raises(ValueError, match=ENV_CPUS):
        numa.reset_inherited_numa_binding("KVServer")
