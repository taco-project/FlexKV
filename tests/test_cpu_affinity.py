import os
from types import SimpleNamespace

import pytest

from flexkv.common import cpu_affinity, numa
from flexkv.common.cpu_affinity import ENV_CPUS, ENV_CPUS_PER_NODE, apply_configured_cpu_affinity
from flexkv.prefetch.runtime import TaskRuntime


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_spread_nodes_cache():
    numa.get_spread_nodes.cache_clear()
    yield
    numa.get_spread_nodes.cache_clear()


@pytest.fixture(autouse=True)
def _clear_affinity_environment(monkeypatch):
    monkeypatch.delenv(ENV_CPUS, raising=False)
    monkeypatch.delenv(ENV_CPUS_PER_NODE, raising=False)


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


@pytest.mark.parametrize("explicit", [False, True])
def test_subprocess_affinity_moves_existing_threads(monkeypatch, explicit):
    monkeypatch.setenv(ENV_CPUS_PER_NODE, "1")
    if explicit:
        monkeypatch.setenv(ENV_CPUS, "4-5")
    paths = {0: "0-3,8-11", 1: "4-7,12-15"}
    monkeypatch.setattr(cpu_affinity, "_usable_nodes", lambda: [0, 1])
    monkeypatch.setattr(cpu_affinity, "_core_key", lambda cpu: ((cpu % 8) // 4, cpu % 4))
    monkeypatch.setattr(cpu_affinity.Path, "read_text", lambda path: paths[int(path.parent.name[4:])])
    requested = {4, 5} if explicit else {3, 7, 11, 15}
    effective = requested - {5, 7, 15}
    affinity_calls = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: affinity_calls.append((pid, cpus)), raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: effective, raising=False)
    listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: ["42", "123"] if path == "/proc/self/task" else listdir(path))
    monkeypatch.setattr(cpu_affinity.threading, "get_native_id", lambda: 42)

    assert apply_configured_cpu_affinity("TransferManager")
    assert affinity_calls == [(0, requested), (123, effective)]


@pytest.mark.parametrize("setting", [ENV_CPUS, ENV_CPUS_PER_NODE])
def test_task_runtime_keeps_host_affinity(monkeypatch, setting):
    monkeypatch.setenv(setting, "1")
    monkeypatch.setattr(cpu_affinity, "_tail_cpus", lambda count: {1})
    host_mask = {4, 5}
    calls = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, cpus)), raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: calls[-1][1] if calls else host_mask, raising=False)
    engine = SimpleNamespace(
        transfer_handles=[], _update_tasks=lambda **kwargs: None,
        _reap_completed_tasks=lambda: None,
        _next_runtime_wakeup=lambda poll_s: poll_s,
        _prefetch=SimpleNamespace(tick=lambda: False, sessions={}),
    )
    runtime = TaskRuntime(engine)
    runtime.start()
    try:
        assert runtime.call(os.sched_getaffinity, 0) == host_mask
        assert calls == []
    finally:
        runtime.stop()


@pytest.mark.parametrize("spec", ["9-4", "1,-2", "1,", "x"])
def test_invalid_explicit_cpu_affinity(monkeypatch, spec):
    monkeypatch.setenv(ENV_CPUS, spec)

    with pytest.raises(ValueError, match=ENV_CPUS):
        numa.reset_inherited_numa_binding("KVServer")
