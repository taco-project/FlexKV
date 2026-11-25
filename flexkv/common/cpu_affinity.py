"""Explicit Linux CPU placement for the FlexKV child processes."""

import os
import threading

from flexkv.common.debug import flexkv_logger


ENV_CPUS = "FLEXKV_CPUS"


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


def apply_configured_cpu_affinity(role: str, *, all_threads: bool = True) -> bool:
    """Bind a FlexKV process or an in-process worker and report if configured."""
    spec = os.getenv(ENV_CPUS, "").strip()
    if not spec:
        return False
    try:
        cpus = _parse_cpu_list(spec)
    except ValueError as exc:
        raise ValueError(f"invalid {ENV_CPUS}={spec!r}") from exc

    try:
        os.sched_setaffinity(0, cpus)
    except OSError as exc:
        flexkv_logger.warning(
            f"[NUMA][{role}] {ENV_CPUS}={spec}: sched_setaffinity failed: "
            f"{exc}; keeping the inherited affinity")
        return True

    effective = os.sched_getaffinity(0)
    # Imports during child startup may already have started library threads.
    # Apply the same mask before FlexKV creates any of its own workers.
    if all_threads:
        _bind_existing_threads(role, effective)

    identity = f"pid={os.getpid()} tid={threading.get_native_id()}"
    if effective != cpus:
        flexkv_logger.warning(
            f"[NUMA][{role}] {identity} {ENV_CPUS}={spec} was narrowed by the system; "
            f"effective CPUs: {sorted(effective)}")
    else:
        flexkv_logger.info(f"[NUMA][{role}] {identity} applied {ENV_CPUS}={spec}")
    return True
