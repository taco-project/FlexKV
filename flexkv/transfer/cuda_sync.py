"""Bounded shutdown synchronization for explicitly owned CUDA devices."""

import threading
import time
from typing import Iterable, List, Tuple

import torch

from flexkv.common.debug import flexkv_logger


def synchronize_cuda_devices(
    device_ids: Iterable[int], timeout_s: float, *, name: str
) -> bool:
    """Drain this process's work on every device within one timeout budget.

    CUDA's current device is thread-local. Each daemon selects its device
    before synchronizing; separate threads let healthy devices drain even if
    another device hangs. False means at least one device did not drain.
    """
    devices = tuple(sorted(set(device_ids)))
    if not devices or not torch.cuda.is_initialized():
        return True

    deadline = time.monotonic() + timeout_s
    pending: List[Tuple[int, threading.Event, List[BaseException]]] = []

    def drain(device_id: int, done: threading.Event, errors: List[BaseException]) -> None:
        try:
            if device_id < 0:
                raise ValueError(f"Invalid CUDA device ID: {device_id}")
            torch.cuda.set_device(device_id)
            torch.cuda.synchronize(device=device_id)
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    for device_id in devices:
        done = threading.Event()
        errors: List[BaseException] = []
        thread = threading.Thread(
            target=drain,
            args=(device_id, done, errors),
            name=f"{name}-cuda-{device_id}-drain",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError as exc:
            errors.append(exc)
            done.set()
        pending.append((device_id, done, errors))

    success = True
    for device_id, done, errors in pending:
        if not done.wait(timeout=max(0.0, deadline - time.monotonic())):
            flexkv_logger.warning(
                "%s: CUDA device %d did not drain within %.1fs",
                name, device_id, timeout_s,
            )
            success = False
        elif errors:
            flexkv_logger.warning(
                "%s: CUDA device %d drain failed: %r",
                name, device_id, errors[0],
            )
            success = False
    return success
