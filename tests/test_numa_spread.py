"""NUMA spreading of the shared CPU KV pool (flexkv.common.numa).

Pure planning/parsing logic is tested everywhere. Tests that need real NUMA
syscalls are skipped unless the host exposes at least two memory nodes and
the process may call mbind / set_mempolicy.
"""

import ctypes
import math
import mmap
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from flexkv.common import numa
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.storage import allocator as alloc

pytestmark = pytest.mark.unit

MiB = 1 << 20


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(numa.ENV_ENABLE_NUMA_SPREAD, raising=False)
    monkeypatch.delenv(numa.ENV_NUMA_NODES, raising=False)
    numa.get_spread_nodes.cache_clear()
    yield
    numa.get_spread_nodes.cache_clear()


@pytest.fixture
def no_syscalls(monkeypatch):
    """Fail the test if any NUMA syscall is attempted."""
    def _forbidden(name, *args):
        raise AssertionError(f"unexpected NUMA syscall {name}")
    monkeypatch.setattr(numa, "_syscall", _forbidden)


def fake_host(monkeypatch, mems, cpus_by_node, arch="x86_64"):
    monkeypatch.setattr(numa, "_read_status_list", lambda key: list(mems))
    monkeypatch.setattr(numa, "node_cpus", lambda n: set(cpus_by_node.get(n, ())))
    monkeypatch.setattr(numa.platform, "machine", lambda: arch)


def blockfirst_layout(num_block=64):
    return KVCacheLayout(type=KVCacheLayoutType.BLOCKFIRST, num_layer=2,
                         num_block=num_block, tokens_per_block=16, num_head=2,
                         head_size=64)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_enable_flag_defaults_off(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, value)
    assert numa.numa_spread_enabled() is expected


@pytest.mark.parametrize("value", ["true", "2", "yes", ""])
def test_enable_flag_rejects_other_values(monkeypatch, value):
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, value)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        numa.numa_spread_enabled()


@pytest.mark.parametrize("spec,expected", [
    ("0", [0]),
    ("0,1", [0, 1]),
    ("0-3", [0, 1, 2, 3]),
    (" 3, 0-1 ,1,", [0, 1, 3]),
    ("", []),
])
def test_parse_list(spec, expected):
    assert numa._parse_list(spec) == expected


def test_parse_list_rejects_garbage():
    with pytest.raises(ValueError):
        numa._parse_list("0,a")


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------
def test_spread_nodes_empty_when_disabled(monkeypatch):
    fake_host(monkeypatch, [0, 1], {0: {0}, 1: {1}})
    assert numa.get_spread_nodes() == ()


def test_spread_nodes_skip_memory_only_nodes(monkeypatch):
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, "1")
    fake_host(monkeypatch, [0, 1, 2], {0: {0, 1}, 1: {2, 3}})  # node 2: CXL, no CPUs
    assert numa.get_spread_nodes() == (0, 1)


def test_spread_nodes_need_two_nodes(monkeypatch):
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, "1")
    fake_host(monkeypatch, [0], {0: {0}})
    assert numa.get_spread_nodes() == ()


def test_spread_nodes_env_is_intersected_with_mems_allowed(monkeypatch):
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, "1")
    monkeypatch.setenv(numa.ENV_NUMA_NODES, "1-3")
    fake_host(monkeypatch, [0, 1, 2], {0: {0}, 1: {1}, 2: {2}})
    assert numa.get_spread_nodes() == (1, 2)


def test_spread_nodes_env_may_select_memory_only_nodes(monkeypatch):
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, "1")
    monkeypatch.setenv(numa.ENV_NUMA_NODES, "0,2")
    fake_host(monkeypatch, [0, 1, 2], {0: {0}, 1: {1}})
    assert numa.get_spread_nodes() == (0, 2)


def test_spread_nodes_invalid_env(monkeypatch):
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, "1")
    monkeypatch.setenv(numa.ENV_NUMA_NODES, "x")
    fake_host(monkeypatch, [0, 1], {0: {0}, 1: {1}})
    with pytest.raises(ValueError, match=numa.ENV_NUMA_NODES):
        numa.get_spread_nodes()


def test_spread_nodes_unsupported_arch(monkeypatch):
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, "1")
    fake_host(monkeypatch, [0, 1], {0: {0}, 1: {1}}, arch="riscv64")
    assert numa.get_spread_nodes() == ()


# ---------------------------------------------------------------------------
# Syscall argument encoding
# ---------------------------------------------------------------------------
def test_nodemask_encoding():
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask, maxnode = numa._nodemask([0, 1])
    assert len(mask) == 1 and mask[0] == 0b11 and maxnode.value == bits + 1
    mask, maxnode = numa._nodemask([bits + 1])
    assert len(mask) == 2 and mask[0] == 0 and mask[1] == 0b10
    assert maxnode.value == 2 * bits + 1


# ---------------------------------------------------------------------------
# Region planning
# ---------------------------------------------------------------------------
def check_plan(regions, logical, unit, nodes):
    assert [r[2] for r in regions] == list(nodes)
    assert regions[0][0] == 0
    for (off, length, _), nxt in zip(regions, regions[1:] + [(logical, 0, None)]):
        assert off + length == nxt[0]  # contiguous, no gap / overlap
        assert off % unit == 0         # block- and page-aligned start
        assert length > 0
    assert sum(r[1] for r in regions) == logical


@pytest.mark.parametrize("nodes", [(0, 1), (0, 1, 2, 3), (3, 5)])
@pytest.mark.parametrize("block,page", [(4096, 4096), (3 * 4096, 4096), (6144, 4096), (MiB, 2 * MiB)])
def test_plan_regions_aligned_contiguous_balanced(nodes, block, page):
    unit = math.lcm(block, page)
    logical = unit * 1000 + 12345  # tail that is not a whole unit
    regions = numa.plan_cpu_pool_numa_regions(logical, block, page, nodes)
    check_plan(regions, logical, unit, nodes)
    head_lengths = [r[1] for r in regions[:-1]]
    assert max(head_lengths) - min(head_lengths) <= unit


def test_plan_regions_disabled_or_single_node():
    assert numa.plan_cpu_pool_numa_regions(1 << 30, 4096, 4096) == []  # flag off
    assert numa.plan_cpu_pool_numa_regions(1 << 30, 4096, 4096, (0,)) == []
    assert numa.plan_cpu_pool_numa_regions(0, 4096, 4096, (0, 1)) == []


def test_plan_regions_too_coarse():
    unit = 4096
    too_small = unit * numa._MIN_UNITS_PER_REGION * 2 - 1
    assert numa.plan_cpu_pool_numa_regions(too_small, unit, unit, (0, 1)) == []
    regions = numa.plan_cpu_pool_numa_regions(too_small + 1, unit, unit, (0, 1))
    check_plan(regions, too_small + 1, unit, (0, 1))


def test_plan_regions_deterministic():
    args = (123 * MiB + 7, 3 * 4096, 2 * MiB, (0, 1, 2))
    assert numa.plan_cpu_pool_numa_regions(*args) == numa.plan_cpu_pool_numa_regions(*args)


def test_bind_cpu_pool_covers_mapping_padding(monkeypatch):
    calls = []
    monkeypatch.setattr(numa, "bind_range", lambda a, l, n: calls.append((a, l, n)))
    regions = [(0, 100, 0), (100, 100, 1)]
    assert numa.bind_cpu_pool(1000, regions, 256, "t") is True
    assert calls == [(1000, 100, 0), (1100, 156, 1)]


def test_bind_cpu_pool_reports_failure(monkeypatch):
    def fail(*_):
        raise OSError(1, "EPERM")
    monkeypatch.setattr(numa, "bind_range", fail)
    assert numa.bind_cpu_pool(0, [(0, 1, 0), (1, 1, 1)], 2, "t") is False


# ---------------------------------------------------------------------------
# touch_pages must never write
# ---------------------------------------------------------------------------
def test_touch_pages_is_read_only():
    # A write to a PROT_READ mapping would SIGSEGV, so run it in a subprocess.
    code = (
        "import ctypes, mmap\n"
        "from flexkv.common import numa\n"
        "libc = ctypes.CDLL(None)\n"
        "libc.mmap.restype = ctypes.c_void_p\n"
        "libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,"
        " ctypes.c_int, ctypes.c_int, ctypes.c_long]\n"
        "p = libc.mmap(None, 2 * numa.PAGE_SIZE, mmap.PROT_READ,"
        " mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)\n"
        "numa.touch_pages([p, p + numa.PAGE_SIZE + 5])\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]


def test_touch_pages_keeps_content():
    buf = (ctypes.c_uint8 * 8)(*range(8))
    numa.touch_pages([ctypes.addressof(buf) + i for i in range(8)])
    assert list(buf) == list(range(8))


# ---------------------------------------------------------------------------
# Default (flag off) is the legacy path
# ---------------------------------------------------------------------------
def test_default_reset_binding_is_noop(no_syscalls):
    before = os.sched_getaffinity(0)
    numa.reset_inherited_numa_binding("test")
    assert os.sched_getaffinity(0) == before


def test_default_cpu_allocator_is_legacy(no_syscalls, monkeypatch):
    for name in ("populate_shmem_fd", "prefault_cpu_pool", "bind_cpu_pool"):
        monkeypatch.setattr(numa, name, lambda *a, **k: pytest.fail("NUMA path taken"))
    layout = blockfirst_layout()
    handle = alloc.CPUAllocator.allocate(layout, torch.bfloat16)
    assert handle.data.numel() == layout.get_total_elements()
    assert handle.data.dtype == torch.bfloat16
    assert not handle.data.is_shared()  # legacy torch.empty, not the memfd pool


def test_default_mooncake_registration_is_legacy(no_syscalls, monkeypatch):
    from flexkv.transfer import backends

    monkeypatch.setattr(backends, "touch_pages", lambda *_: pytest.fail("touched pages"))
    cfg = SimpleNamespace(hugepage_size_bytes=2 * MiB, mooncake_max_mr_size_bytes=1 << 40,
                          mooncake_mr_split_policy="strict")

    def backend(nbytes, block):
        b = backends.MooncakeStoreBackend.__new__(backends.MooncakeStoreBackend)
        b._cpu_buffer = torch.empty(nbytes, dtype=torch.uint8)
        b.block_size_bytes = block
        b.cache_config = cfg
        return b

    plain = backend(MiB, 4096)
    assert plain._registration_regions(SimpleNamespace(mapped_size=None)) == [
        (plain._cpu_buffer.data_ptr(), MiB)]

    seen = {}
    monkeypatch.setattr(backends, "_split_mooncake_registration_regions",
                        lambda **kw: seen.update(kw) or [(kw["base_ptr"], kw["mapped_size"])])
    huge = backend(8 * MiB, MiB)
    huge._registration_regions(SimpleNamespace(mapped_size=10 * MiB))
    assert seen == dict(base_ptr=huge._cpu_buffer.data_ptr(), logical_size=8 * MiB,
                        mapped_size=10 * MiB, block_size=MiB, max_mr_size=1 << 40,
                        size_alignment=2 * MiB, pointer_alignment=2 * MiB,
                        mr_split_policy="strict")


# ---------------------------------------------------------------------------
# Allocator and Mooncake backend plan with the same block size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float32])
def test_numa_block_bytes_matches_worker_block_stride(dtype):
    from flexkv.transfer.workers.remote import CPURemoteTransferWorker

    layout = blockfirst_layout()
    worker = CPURemoteTransferWorker.__new__(CPURemoteTransferWorker)
    worker.has_multi_group = False
    worker.num_layers = layout.num_layer
    worker.kv_dim = layout.kv_dim
    worker.num_kv_heads = layout.num_kv_heads
    worker.cpu_blocks = torch.empty(0)
    worker.cpu_layer_ptrs = torch.empty(0)
    worker.cpu_blocks_mapped_size = None
    geometry = worker._build_geometry(layout, dtype)
    assert alloc._numa_block_bytes(layout, dtype) == geometry.cpu.block_stride


def test_numa_block_bytes_multi_group_is_already_bytes():
    layout = SimpleNamespace(type=KVCacheLayoutType.BLOCKFIRST, num_block=4,
                             layer_groups=[object()], get_block_stride=lambda: 12345)
    assert alloc._numa_block_bytes(layout, torch.bfloat16) == 12345


def test_numa_block_bytes_non_blockfirst():
    layout = KVCacheLayout(type=KVCacheLayoutType.LAYERFIRST, num_layer=2, num_block=8,
                           tokens_per_block=16, num_head=2, head_size=64)
    assert alloc._numa_block_bytes(layout, torch.bfloat16) == 1


# ---------------------------------------------------------------------------
# Real placement (needs >= 2 NUMA nodes and mbind permission)
# ---------------------------------------------------------------------------
def _real_spread_supported() -> bool:
    nodes = [n for n in numa._read_status_list("Mems_allowed_list") if numa.node_cpus(n)]
    if len(nodes) < 2 or numa.platform.machine().lower() not in numa._SYSCALLS:
        return False
    buf = mmap_anon(numa.PAGE_SIZE)
    try:
        numa.bind_range(buf, numa.PAGE_SIZE, nodes[0])
        return True
    except OSError:
        return False


def mmap_anon(size):
    libc = ctypes.CDLL(None)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    return libc.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE,
                     mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)


@pytest.mark.skipif(not _real_spread_supported(), reason="needs >=2 NUMA nodes and mbind")
def test_spread_cpu_pool_lands_on_planned_nodes(monkeypatch):
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, "1")
    nodes = numa.get_spread_nodes()
    layout = blockfirst_layout(num_block=1024)  # 64 MiB in bf16
    tensor = alloc.CPUAllocator.allocate(layout, torch.bfloat16).data
    assert tensor.is_shared() and tensor.numel() == layout.get_total_elements()
    nbytes = tensor.numel() * tensor.element_size()
    regions = numa.plan_cpu_pool_numa_regions(
        nbytes, alloc._numa_block_bytes(layout, torch.bfloat16), numa.PAGE_SIZE, nodes)
    assert len(regions) == len(nodes)
    base = tensor.data_ptr()
    for off, length, node in regions:
        addrs = [base + off + k * (length // 16) for k in range(16)]
        numa.touch_pages(addrs)
        assert set(numa.query_page_nodes(addrs)) == {node}
