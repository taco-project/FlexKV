"""NUMA spreading of FlexKV's shared CPU KV pool (Mooncake-style).

Problem
-------
FlexKV's KVServer is launched by the ``dp_client_id == 0`` caller (e.g. the
SGLang scheduler of GPU0) and spawns the TransferManager, which allocates the
CPU KV pool shared by *every* GPU and spawns the transfer workers. Linux keeps
the task mempolicy and CPU affinity across fork/exec, so when the launcher is
NUMA-bound (SGLang ``SGLANG_AUTO_NUMA_BIND`` wraps each scheduler with
``numactl --cpunodebind=N --membind=N``) the whole pool lands on GPU0's node,
and every RDMA transfer from it goes through that node's NICs.

What ``FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD=1`` (opt-in) does
---------------------------------------------------------
1. ``reset_inherited_numa_binding``: KVServer / TransferManager drop the
   launcher's binding (CPU affinity -> all spread nodes, task mempolicy ->
   interleave), so incidental host allocations are no longer pinned to GPU0.
2. ``plan_cpu_pool_numa_regions`` + ``bind_range``: like Mooncake's
   ``allocate_buffer_numa_segments``, the pool is split into one contiguous,
   KV-block- and page-aligned region per NUMA node, each ``mbind``-ed to its
   node.
3. The Mooncake store backend registers one MR per region (instead of one MR
   for the whole pool). Mooncake labels every MR with the NUMA node of its
   first page (``cpu:N``), so transfers of each region pick that node's NICs.

``FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD=0`` (default) keeps the inherited memory policy.
``FLEXKV_NUMA_NODES`` (e.g. ``0,1`` or ``0-3``) restricts the nodes used;
default is every node in the cpuset that has CPUs.
``FLEXKV_CLIENT_NUMA_MAP`` optionally maps DP client IDs to nodes, e.g.
``0:0,1:1``. Unmapped clients use GPU topology when all their TP GPUs
share one node. Other clients allocate across the CPU pool's regions.

``FLEXKV_CPUS`` and ``FLEXKV_CPUS_PER_NODE`` set CPU placement independently
of the memory policy, only for KVServer and TransferManager child processes
and the workers that inherit their masks. In-process workers are not rebound.
"""

import ctypes
import functools
import math
import os
import platform
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from flexkv.common.cpu_affinity import apply_configured_cpu_affinity
from flexkv.common.debug import flexkv_logger

ENV_ENABLE_NUMA_SPREAD = "FLEXKV_ENABLE_CPU_CACHE_NUMA_SPREAD"
ENV_NUMA_NODES = "FLEXKV_NUMA_NODES"
ENV_CLIENT_NUMA_MAP = "FLEXKV_CLIENT_NUMA_MAP"
# The Internal KVServer may clear CUDA visibility for its own GPU runtime.
_ENV_CLIENT_CUDA_VISIBLE_DEVICES = "FLEXKV_CLIENT_CUDA_VISIBLE_DEVICES"

_DRM_CLASS = Path("/sys/class/drm")
_NVIDIA_GPUS = Path("/proc/driver/nvidia/gpus")
_PCI_DEVICES = Path("/sys/bus/pci/devices")

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")

_MPOL_DEFAULT = 0
_MPOL_PREFERRED = 1
_MPOL_INTERLEAVE = 3

# A region boundary may move by up to half a unit; refuse to spread when that
# could unbalance the regions by more than ~1/16.
_MIN_UNITS_PER_REGION = 8

_SYSCALLS: Dict[str, Dict[str, int]] = {
    "x86_64": {"mbind": 237, "set_mempolicy": 238, "move_pages": 279},
    "aarch64": {"mbind": 235, "set_mempolicy": 237, "move_pages": 239},
}
_SYSCALLS["amd64"] = _SYSCALLS["x86_64"]
_SYSCALLS["arm64"] = _SYSCALLS["aarch64"]

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def numa_spread_enabled() -> bool:
    value = os.getenv(ENV_ENABLE_NUMA_SPREAD, "0").strip()
    if value not in ("0", "1"):
        raise ValueError(
            f"{ENV_ENABLE_NUMA_SPREAD} must be 0 or 1, got {value!r}")
    return value == "1"


def _parse_list(spec: str) -> List[int]:
    out: Set[int] = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _read_status_list(key: str) -> List[int]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return _parse_list(line.split(":", 1)[1])
    except (OSError, ValueError):
        pass
    return []


def node_cpus(node: int) -> Set[int]:
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            return set(_parse_list(f.read()))
    except (OSError, ValueError):
        return set()


@functools.cache
def get_spread_nodes() -> Tuple[int, ...]:
    """NUMA nodes the CPU pool is spread over; empty when spreading is off
    or there is nothing to spread over (fewer than two usable nodes)."""
    if not numa_spread_enabled():
        return ()
    allowed = _read_status_list("Mems_allowed_list")
    spec = os.getenv(ENV_NUMA_NODES, "").strip()
    if spec:
        try:
            requested = _parse_list(spec)
        except ValueError as e:
            raise ValueError(f"invalid {ENV_NUMA_NODES}={spec!r}") from e
        nodes = [n for n in requested if n in allowed]
        dropped = sorted(set(requested) - set(nodes))
        if dropped:
            flexkv_logger.warning(
                f"[NUMA] {ENV_NUMA_NODES}: nodes {dropped} are not in "
                f"Mems_allowed {allowed}; ignored")
    else:
        # Skip memory-only nodes (CXL/HBM expanders) unless asked for.
        nodes = [n for n in allowed if node_cpus(n)]
    if platform.machine().lower() not in _SYSCALLS:
        flexkv_logger.warning(
            f"[NUMA] unsupported arch {platform.machine()}; spreading disabled")
        return ()
    return tuple(nodes) if len(nodes) >= 2 else ()


def _read_numa_node(device: Path) -> Optional[int]:
    try:
        node = int((device / "numa_node").read_text())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


@functools.lru_cache(maxsize=1)
def _gpu_numa_nodes() -> Tuple[Optional[int], ...]:
    # DRM render minors give the host render-device order without counting
    # display-only devices such as BMC adapters. Deduplicate PCI functions.
    devices = {}
    vendors = set()
    render_nodes = (path for path in _DRM_CLASS.glob("renderD*")
                    if path.name.removeprefix("renderD").isdigit())
    for render in sorted(render_nodes, key=lambda path: int(path.name.removeprefix("renderD"))):
        try:
            device = (render / "device").resolve(strict=True)
            vendor = (device / "vendor").read_text().strip()
        except (OSError, RuntimeError):
            continue
        if vendor in ("0x1002", "0x10de"):
            vendors.add(vendor)
            devices[device] = _read_numa_node(device)

    nvidia = []
    for info in _NVIDIA_GPUS.glob("*/information"):
        try:
            fields = dict(line.split(":", 1) for line in info.read_text().splitlines() if ":" in line)
            minor = int(fields["Device Minor"])
        except (OSError, ValueError, KeyError):
            continue
        nvidia.append((minor, info.parent.name, fields.get("Model", "").strip()))
    if nvidia:
        vendors.add("0x10de")
    if len(vendors) > 1:
        # A single ordinal space cannot describe multiple GPU runtimes.
        return ()

    pci_order = os.getenv("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    if nvidia:
        if pci_order:
            nvidia.sort(key=lambda gpu: gpu[1])
        else:
            # CUDA's default speed heuristic can reorder different models.
            if len({model for _, _, model in nvidia}) > 1:
                return ()
            nvidia.sort()
        return tuple(_read_numa_node(_PCI_DEVICES / bus) for _, bus, _ in nvidia)
    if pci_order and vendors == {"0x10de"}:
        devices = dict(sorted(devices.items()))
    return tuple(devices.values())


def query_gpu_numa_node(physical_gpu_id: int) -> Optional[int]:
    """Return the NUMA node of a host GPU ordinal using Linux driver topology."""
    nodes = _gpu_numa_nodes()
    return nodes[physical_gpu_id] if 0 <= physical_gpu_id < len(nodes) else None


def resolve_client_numa_node(dp_client_id: int, tp_size: int) -> Optional[int]:
    """Prefer a node only if every GPU used by this client is local to it."""
    if dp_client_id < 0 or tp_size <= 0:
        return None
    spec = os.getenv(ENV_CLIENT_NUMA_MAP, "").strip()
    if spec:
        overrides = {}
        try:
            for entry in spec.split(","):
                client, node = map(int, entry.split(":"))
                if client < 0 or node < 0 or client in overrides:
                    raise ValueError(entry)
                overrides[client] = node
        except ValueError:
            flexkv_logger.warning(f"[NUMA] invalid {ENV_CLIENT_NUMA_MAP}={spec!r}")
            return None
        if dp_client_id in overrides:
            return overrides[dp_client_id]

    first = dp_client_id * tp_size
    gpu_ids = list(range(first, first + tp_size))
    # HIP's mask uses the ordinals already exposed by ROCr. HIP and CUDA
    # visibility are aliases on ROCm, so apply only one at that level.
    cuda = os.getenv("CUDA_VISIBLE_DEVICES", os.getenv(_ENV_CLIENT_CUDA_VISIBLE_DEVICES))
    hip = os.getenv("HIP_VISIBLE_DEVICES", cuda)
    for mask in (hip, os.getenv("ROCR_VISIBLE_DEVICES")):
        if mask is None:
            continue
        try:
            visible = [int(value) for value in mask.split(",")]
            if any(gpu < 0 for gpu in visible):
                return None
            gpu_ids = [visible[logical] for logical in gpu_ids]
        except (IndexError, ValueError):
            return None

    nodes = {query_gpu_numa_node(gpu) for gpu in gpu_ids}
    if len(nodes) == 1:
        node = nodes.pop()
        return node if node is not None and node >= 0 else None
    return None


# ---------------------------------------------------------------------------
# Syscall wrappers
# ---------------------------------------------------------------------------
def _syscall(name: str, *args) -> None:
    nr = _SYSCALLS[platform.machine().lower()][name]
    ret = _libc.syscall(ctypes.c_long(nr), *args)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{name}: {os.strerror(err)}")


def _nodemask(nodes: Sequence[int]):
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    words = max(nodes) // bits + 1
    mask = (ctypes.c_ulong * words)()
    for n in nodes:
        mask[n // bits] |= 1 << (n % bits)
    # The kernel reads maxnode - 1 bits.
    return mask, ctypes.c_ulong(words * bits + 1)


def set_task_mempolicy(mode: int, nodes: Sequence[int] = ()) -> None:
    if mode == _MPOL_DEFAULT:
        _syscall("set_mempolicy", ctypes.c_int(mode), None, ctypes.c_ulong(0))
        return
    mask, maxnode = _nodemask(nodes)
    _syscall("set_mempolicy", ctypes.c_int(mode), mask, maxnode)


def bind_range(addr: int, length: int, node: int) -> None:
    """``mbind`` [addr, addr+length) to ``node`` (MPOL_PREFERRED).

    PREFERRED rather than Mooncake's BIND: a node that runs short spills to
    the others instead of OOM-killing / SIGBUS-ing the server. For shmem
    (the default pool) the policy is stored on the shared object, so pages
    faulted later by any process follow it; for hugetlbfs it is per-VMA, so
    the caller must prefault the range in this process.
    """
    mask, maxnode = _nodemask([node])
    _syscall("mbind", ctypes.c_void_p(addr), ctypes.c_ulong(length),
             ctypes.c_int(_MPOL_PREFERRED), mask, maxnode, ctypes.c_uint(0))


def query_page_nodes(addrs: Sequence[int]) -> List[int]:
    """NUMA node of the page holding each address (negative errno when the
    page is not populated), via move_pages(2) without moving anything."""
    n = len(addrs)
    if n == 0:
        return []
    pages = (ctypes.c_void_p * n)(*[a & ~(PAGE_SIZE - 1) for a in addrs])
    status = (ctypes.c_int * n)()
    _syscall("move_pages", ctypes.c_int(0), ctypes.c_ulong(n), pages, None,
             status, ctypes.c_int(0))
    return list(status)


def touch_pages(addrs: Sequence[int]) -> None:
    """Map the page holding each address into this process by reading it.

    A read fault on shmem / hugetlbfs maps the existing page (or allocates it
    under the mapping's policy); nothing is written, so this is safe even
    while other processes are writing the pool.
    """
    for a in addrs:
        ctypes.c_uint8.from_address(a).value  # noqa: B018


# ---------------------------------------------------------------------------
# Process-level binding
# ---------------------------------------------------------------------------
def reset_inherited_numa_binding(role: str) -> None:
    """Drop the NUMA binding inherited from the launcher (see module doc).

    Must run before the CPU pool is allocated and before worker processes /
    threads are created: mempolicy and affinity are inherited at creation.
    """
    configured_cpus = apply_configured_cpu_affinity(role)
    nodes = get_spread_nodes()
    if not nodes:
        return
    before_cpus = len(os.sched_getaffinity(0))
    if not configured_cpus:
        target_cpus = set().union(*(node_cpus(node) for node in nodes))
        if target_cpus:
            try:
                # The kernel intersects the mask with the cpuset.
                os.sched_setaffinity(0, target_cpus)
            except OSError as e:
                flexkv_logger.warning(f"[NUMA][{role}] sched_setaffinity failed: {e}")
    try:
        set_task_mempolicy(_MPOL_INTERLEAVE, nodes)
    except OSError as e:
        flexkv_logger.warning(
            f"[NUMA][{role}] set_mempolicy(interleave) failed: {e}; keeping the "
            "inherited policy (docker may need --cap-add SYS_NICE)")
        return
    flexkv_logger.info(
        f"[NUMA][{role}] pid={os.getpid()} reset inherited binding: "
        f"mempolicy=interleave{list(nodes)}, cpus {before_cpus} -> "
        f"{len(os.sched_getaffinity(0))}")


# ---------------------------------------------------------------------------
# Pool layout
# ---------------------------------------------------------------------------
def plan_cpu_pool_numa_regions(
    logical_bytes: int,
    block_bytes: int,
    page_align_bytes: int,
    nodes: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int, int]]:
    """Split a pool into one contiguous region per NUMA node.

    Returns ``[(offset, length, node), ...]`` covering ``[0, logical_bytes)``;
    callers extend the last region over any mapping padding. Every boundary
    is a multiple of ``lcm(block_bytes, page_align_bytes)`` so that no KV
    block straddles two regions (Mooncake cannot transfer a block across MRs)
    and every region starts on a page (mbind / HugePage MR alignment).

    Deterministic in its inputs: the allocator (TransferManager) and the
    Mooncake backend (worker process) both call it and must agree.
    Returns ``[]`` when spreading is off or the pool is too small to split.
    """
    if nodes is None:
        nodes = get_spread_nodes()
    n = len(nodes)
    if n < 2 or logical_bytes <= 0:
        return []
    unit = math.lcm(max(1, int(block_bytes)), max(1, int(page_align_bytes)))
    if logical_bytes < unit * _MIN_UNITS_PER_REGION * n:
        flexkv_logger.warning(
            f"[NUMA] cannot split the CPU pool ({logical_bytes} B) over {n} "
            f"nodes: region boundaries must be multiples of lcm(block={block_bytes} B, "
            f"page/mapping alignment={page_align_bytes} B) = {unit} B, which is "
            "too coarse; not spreading")
        return []
    total_units = logical_bytes // unit
    bounds = [0]
    for i in range(1, n):
        bounds.append(((total_units * i + n // 2) // n) * unit)
    bounds.append(logical_bytes)
    return [(bounds[i], bounds[i + 1] - bounds[i], int(nodes[i]))
            for i in range(n)]


def plan_cpu_pool_numa_block_ranges(
    num_blocks: int,
    block_bytes: int,
    page_align_bytes: int,
    is_blockfirst: bool = True,
    nodes: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int, int]]:
    """Translate a BLOCKFIRST pool's byte ranges into half-open block ranges."""
    if not is_blockfirst or block_bytes <= 0 or num_blocks <= 0:
        return []
    return [
        (offset // block_bytes, (offset + length) // block_bytes, node)
        for offset, length, node in plan_cpu_pool_numa_regions(
            num_blocks * block_bytes, block_bytes, page_align_bytes, nodes=nodes)
    ]


def bind_cpu_pool(base_ptr: int,
                  regions: Sequence[Tuple[int, int, int]],
                  mapped_bytes: int,
                  role: str) -> bool:
    """mbind every planned region; the last one also covers the padding up to
    ``mapped_bytes``. Returns False (pool keeps the inherited policy) on
    failure."""
    try:
        for i, (off, length, node) in enumerate(regions):
            if i == len(regions) - 1:
                length = mapped_bytes - off
            bind_range(base_ptr + off, length, node)
    except OSError as e:
        flexkv_logger.warning(
            f"[NUMA][{role}] mbind failed: {e}; CPU pool is NOT spread over "
            "NUMA nodes (docker may need --cap-add SYS_NICE)")
        return False
    return True


def populate_shmem_fd(fd: int,
                      regions: Sequence[Tuple[int, int, int]],
                      total_bytes: int,
                      chunk_bytes: int = 1 << 30) -> float:
    """Allocate every page of a mbind-ed shmem fd up front; returns seconds.

    Without this the pages are faulted lazily inside ``cudaHostRegister`` /
    ``ibv_reg_mr`` of 16+ processes at once, which measured ~3x slower than
    registering already-populated memory (the legacy pool was populated by
    torch's copy into /dev/shm). shmem fallocate honours the shared policy
    set by mbind, so each region still lands on its node; running each
    region's fallocate on that node's CPUs keeps page zeroing node-local.
    """
    t0 = time.monotonic()
    errors: List[BaseException] = []

    def _run(off: int, length: int, node: int) -> None:
        try:
            cpus = node_cpus(node) & os.sched_getaffinity(0)
            if cpus:
                os.sched_setaffinity(0, cpus)  # this thread only
            end = off + length
            # Chunked so both regions make progress under shmem's inode lock.
            while off < end:
                n = min(chunk_bytes, end - off)
                os.posix_fallocate(fd, off, n)
                off += n
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = []
    for i, (off, length, node) in enumerate(regions):
        if i == len(regions) - 1:
            length = total_bytes - off
        th = threading.Thread(target=_run, args=(off, length, node), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return time.monotonic() - t0


def prefault_cpu_pool(base_ptr: int,
                      regions: Sequence[Tuple[int, int, int]],
                      mapped_bytes: int,
                      page_bytes: int,
                      max_threads: int = 16) -> None:
    """Populate every page, each region by threads running on its node (as
    Mooncake does for HugeTLB segments). Required for hugetlbfs, whose mbind
    policy only applies to faults through this process's mapping."""
    n = len(regions)
    threads_per_node = max(1, max_threads // n)
    errors: List[BaseException] = []

    def _worker(start: int, end: int, node: int) -> None:
        try:
            cpus = node_cpus(node) & os.sched_getaffinity(0)
            if cpus:
                os.sched_setaffinity(0, cpus)  # this thread only
            for addr in range(start, end, page_bytes):
                ctypes.memset(addr, 0, 1)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = []
    for i, (off, length, node) in enumerate(regions):
        if i == n - 1:
            length = mapped_bytes - off
        pages = (length + page_bytes - 1) // page_bytes
        per_thread = (pages + threads_per_node - 1) // threads_per_node
        for t in range(threads_per_node):
            begin = off + t * per_thread * page_bytes
            end = min(off + length, begin + per_thread * page_bytes)
            if begin < end:
                th = threading.Thread(
                    target=_worker, args=(base_ptr + begin, base_ptr + end, node),
                    daemon=True)
                th.start()
                threads.append(th)
    for th in threads:
        th.join()
    if errors:
        raise errors[0]


def describe_cpu_pool_placement(base_ptr: int,
                                regions: Sequence[Tuple[int, int, int]],
                                samples_per_region: int = 64) -> str:
    """Human-readable 'planned node -> sampled actual nodes' per region."""
    parts = []
    for off, length, node in regions:
        step = max(PAGE_SIZE, length // samples_per_region)
        addrs = [base_ptr + off + k * step for k in range(samples_per_region)
                 if k * step < length]
        # move_pages only sees pages mapped in this process.
        touch_pages(addrs)
        try:
            status = query_page_nodes(addrs)
        except OSError as e:
            return f"<move_pages failed: {e}>"
        hist: Dict[str, int] = {}
        for s in status:
            key = f"N{s}" if s >= 0 else "unfaulted"
            hist[key] = hist.get(key, 0) + 1
        parts.append(
            f"[off={off} len={length} plan=N{node} sampled={hist}]")
    return " ".join(parts)
