"""SSD-contiguous run split for LAYERFIRST layer-major pread.

Loads ssd_runs.py by path so this file does not import flexkv.__init__ (torch).
"""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "flexkv_ssd_runs",
    Path(__file__).resolve().parents[1] / "flexkv" / "transfer" / "ssd_runs.py",
)
_ssd_runs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ssd_runs)
partition_contiguous_ssd_runs_np = _ssd_runs.partition_contiguous_ssd_runs_np


def _ids(*values):
    return np.array(values, dtype=np.int64)


def _run_sizes(runs):
    return [int(cpu.size) for cpu, _ in runs]


def _ssd_values(runs):
    return [ssd.tolist() for _, ssd in runs]


def test_empty_and_single():
    empty = np.array([], dtype=np.int64)
    assert partition_contiguous_ssd_runs_np(empty, empty) == []
    runs = partition_contiguous_ssd_runs_np(_ids(7), _ids(3))
    assert _run_sizes(runs) == [1]
    assert runs[0][0].tolist() == [3]
    assert runs[0][1].tolist() == [7]


def test_one_user_stays_one_run():
    ssd = np.arange(64, dtype=np.int64)
    cpu = np.arange(10, 74, dtype=np.int64)
    runs = partition_contiguous_ssd_runs_np(ssd, cpu)
    assert _run_sizes(runs) == [64]
    assert runs[0][1].tolist() == list(range(64))
    assert runs[0][0].tolist() == list(range(10, 74))


def test_merged_users_split_like_as_batch_0():
    ssd = np.concatenate(
        [np.arange(0, 64, dtype=np.int64), np.arange(100, 164, dtype=np.int64)])
    cpu = np.concatenate(
        [np.arange(0, 64, dtype=np.int64), np.arange(200, 264, dtype=np.int64)])
    runs = partition_contiguous_ssd_runs_np(ssd, cpu)
    assert _run_sizes(runs) == [64, 64]
    assert _ssd_values(runs) == [list(range(0, 64)), list(range(100, 164))]
    assert runs[1][0].tolist() == list(range(200, 264))


def test_interleaved_concat_recovers_per_user_after_sort():
    ssd = _ids(0, 100, 1, 101, 2, 102)
    cpu = _ids(10, 20, 11, 21, 12, 22)
    runs = partition_contiguous_ssd_runs_np(ssd, cpu)
    assert _ssd_values(runs) == [[0, 1, 2], [100, 101, 102]]
    assert runs[0][0].tolist() == [10, 11, 12]
    assert runs[1][0].tolist() == [20, 21, 22]


def test_gap_starts_new_run():
    ssd = _ids(0, 1, 2, 10, 11)
    cpu = _ids(0, 1, 2, 3, 4)
    runs = partition_contiguous_ssd_runs_np(ssd, cpu)
    assert _ssd_values(runs) == [[0, 1, 2], [10, 11]]


def test_two_files_group_by_fd_then_infile():
    ssd = _ids(0, 1, 2, 4)
    cpu = _ids(10, 11, 12, 13)
    runs = partition_contiguous_ssd_runs_np(ssd, cpu, num_files_per_device=2)
    assert _ssd_values(runs) == [[0, 2, 4], [1]]
    assert runs[0][0].tolist() == [10, 12, 13]


def test_mismatched_lengths_raise():
    try:
        partition_contiguous_ssd_runs_np(_ids(0, 1), _ids(0))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
