"""A test-side stand-in for sglang's end of the per-layer eventfd protocol.

Production hands FlexKV a table of file descriptors that some other process
created and waits on; a test has to be both ends at once.  This is the
consumer end: it creates the same table FlexKV would receive over the UDS,
hands it over as the int32 tensor ``set_layer_eventfds`` wants, and lets a
test read back how many semaphore units each (counter, rank, layer) cell got.

The table shape is the contract, not an implementation detail:
``[num_counters][tp_size][num_layers]``, flat, row-major.  ``Fds.of`` is the
only place that index arithmetic is written down on the test side, so a
transposed table shows up as a wrong-cell assertion rather than as a hang.

The fds are ``EFD_SEMAPHORE``: each read subtracts one unit rather than
draining the counter, which is what makes "exactly one unit per layer"
checkable at all.
"""
import ctypes
import fcntl
import os
import struct
from typing import List

import numpy as np
import torch

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
EFD_SEMAPHORE = 0x1


def sys_eventfd(initval: int = 0, flags: int = 0) -> int:
    """``eventfd(2)``; Python's ``os`` has no wrapper for it."""
    fd = _libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd failed: {os.strerror(err)}")
    return fd


def drain(fd: int) -> int:
    """Read a semaphore eventfd dry, returning the number of units posted.

    Non-blocking, so a layer that was never signalled reads as 0 instead of
    hanging the test -- which is the failure we most want to see reported.
    """
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    total = 0
    while True:
        try:
            data = os.read(fd, 8)
        except BlockingIOError:
            break
        if len(data) != 8:
            break
        total += struct.unpack("Q", data)[0]
    return total


class Fds:
    """One eventfd per (counter, rank, layer), in the table's own order."""

    def __init__(self, num_counters: int, tp_size: int, num_layers: int):
        self.num_counters = num_counters
        self.tp_size = tp_size
        self.num_layers = num_layers
        self.fds: List[int] = [
            sys_eventfd(0, EFD_SEMAPHORE)
            for _ in range(num_counters * tp_size * num_layers)
        ]

    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(np.array(self.fds, dtype=np.int32).copy())

    def of(self, counter: int, rank: int, layer: int) -> int:
        return self.fds[(counter * self.tp_size + rank) * self.num_layers
                        + layer]

    def units(self, layer: int, counter: int = 0) -> List[int]:
        """Units posted for ``layer``, one entry per rank."""
        return [drain(self.of(counter, r, layer)) for r in range(self.tp_size)]

    def close(self) -> None:
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "Fds":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
