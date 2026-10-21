"""Split a merged SSD block list into contiguous on-disk runs.

as_batch=1 concatenates per-user pages. The C++ LAYERFIRST kernel issues one
sequential pread only when a thread slice is CPU+SSD contiguous. Feeding it
one SSD-contiguous run at a time restores the as_batch=0 per-user shape.

Sort key matches ``transfer_ssd.cpp``: ``(fd, infile)`` with
``fd = raw % num_files_per_device``, ``infile = raw // num_files_per_device``.
This module is numpy-only so unit tests do not need ``c_ext``.
"""
from typing import List, Tuple

import numpy as np


def partition_contiguous_ssd_runs_np(
    ssd_block_ids: np.ndarray,
    cpu_block_ids: np.ndarray,
    num_files_per_device: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    ssd = np.asarray(ssd_block_ids, dtype=np.int64).reshape(-1)
    cpu = np.asarray(cpu_block_ids, dtype=np.int64).reshape(-1)
    if ssd.size != cpu.size:
        raise ValueError(
            f"ssd/cpu block counts differ: {ssd.size} vs {cpu.size}")
    n = int(ssd.size)
    if n == 0:
        return []
    nfiles = max(int(num_files_per_device), 1)
    if n == 1:
        return [(cpu.copy(), ssd.copy())]
    fd = np.remainder(ssd, nfiles)
    infile = np.floor_divide(ssd, nfiles)
    order = np.lexsort((infile, fd))
    ssd = ssd[order]
    cpu = cpu[order]
    fd = fd[order]
    infile = infile[order]
    same_fd = fd[1:] == fd[:-1]
    consecutive = infile[1:] == (infile[:-1] + 1)
    break_idx = np.nonzero(~(same_fd & consecutive))[0]
    starts = np.concatenate(([0], break_idx + 1, [n]))
    runs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(starts) - 1):
        start = int(starts[i])
        end = int(starts[i + 1])
        runs.append((cpu[start:end].copy(), ssd[start:end].copy()))
    return runs
