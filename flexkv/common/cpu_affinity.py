"""Explicit Linux CPU placement for the FlexKV child processes."""

import os
from pathlib import Path
import threading

from flexkv.common.debug import flexkv_logger


ENV_CPUS = "FLEXKV_CPUS"
ENV_CPUS_PER_NODE = "FLEXKV_CPUS_PER_NODE"


def _parse_cpu_list(spec: str) -> set[int]:
    cpus = set()
    for part in spec.split(","):
        first, separator, last = part.strip().partition("-")
        start = int(first)
        end = int(last) if separator else start
        if start < 0 or end < start:
            raise ValueError(f"invalid CPU range {part!r}")
        cpus.update(range(start, end + 1))
    return cpus


def _usable_nodes() -> list[int]:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("Mems_allowed_list:"):
                allowed = _parse_cpu_list(line.split(":", 1)[1].strip())
                requested = os.getenv("FLEXKV_NUMA_NODES", "").strip()
                return sorted(allowed & _parse_cpu_list(requested) if requested else allowed)
    return []


def _core_key(cpu: int) -> tuple[int, int]:
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        return (int((topology / "physical_package_id").read_text()), int((topology / "core_id").read_text()))
    except (OSError, ValueError):
        return (1 << 30, cpu)


def _tail_cpus(count: int) -> set[int]:
    reserved = set()
    for node in _usable_nodes():
        try:
            spec = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
        except FileNotFoundError:
            continue
        if not spec:
            continue
        groups: dict[tuple[int, int], set[int]] = {}
        for cpu in _parse_cpu_list(spec):
            groups.setdefault(_core_key(cpu), set()).add(cpu)
        if count >= len(groups):
            flexkv_logger.warning(
                f"[NUMA] {ENV_CPUS_PER_NODE}={count} covers all {len(groups)} cores on node {node}")
        for key in sorted(groups)[-count:]:
            reserved.update(groups[key])
    return reserved


def _bind_existing_threads(role: str, effective: set[int]) -> None:
    current_tid = threading.get_native_id()
    for entry in os.listdir("/proc/self/task"):
        tid = int(entry)
        if tid == current_tid:
            continue
        try:
            os.sched_setaffinity(tid, effective)
        except ProcessLookupError:
            continue  # The library thread exited while enumerating tasks.
        except OSError as exc:
            flexkv_logger.warning(f"[NUMA][{role}] setting affinity for thread {tid} failed: {exc}")


def apply_configured_cpu_affinity(role: str) -> bool:
    """Bind a standalone FlexKV child before it creates its own workers."""
    spec = os.getenv(ENV_CPUS, "").strip()
    if spec:
        setting = f"{ENV_CPUS}={spec}"
        try:
            cpus = _parse_cpu_list(spec)
        except ValueError as exc:
            raise ValueError(f"invalid {setting!r}") from exc
    else:
        spec = os.getenv(ENV_CPUS_PER_NODE, "").strip()
        if not spec:
            return False
        setting = f"{ENV_CPUS_PER_NODE}={spec}"
        try:
            count = int(spec)
            if count < 0:
                raise ValueError("negative core count")
        except ValueError as exc:
            raise ValueError(f"invalid {setting!r}") from exc
        if not count:
            return False
        try:
            cpus = _tail_cpus(count)
        except OSError as exc:
            flexkv_logger.warning(f"[NUMA][{role}] {setting}: cannot read topology: {exc}")
            return True
        if not cpus:
            flexkv_logger.warning(f"[NUMA][{role}] {setting}: no usable cores; keeping inherited affinity")
            return True
    try:
        os.sched_setaffinity(0, cpus)
    except OSError as exc:
        flexkv_logger.warning(
            f"[NUMA][{role}] {setting}: sched_setaffinity failed: "
            f"{exc}; keeping the inherited affinity")
        return True

    effective = os.sched_getaffinity(0)
    _bind_existing_threads(role, effective)

    identity = f"pid={os.getpid()} tid={threading.get_native_id()}"
    if effective != cpus:
        flexkv_logger.warning(
            f"[NUMA][{role}] {identity} {setting} was narrowed by the system; "
            f"effective CPUs: {sorted(effective)}")
    else:
        flexkv_logger.info(f"[NUMA][{role}] {identity} applied {setting}")
    return True
