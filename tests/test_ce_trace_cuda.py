"""Capture native CUDA launches in separate processes, then replay their bytes."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flexkv.transfer.ce_replay import load_trace, replay

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    # Tiny synthetic tensors should not fan out across every visible host core
    # when a test runner has only a small cgroup CPU quota.
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


CAPTURE = r'''
import sys
import os
from flexkv import c_ext
from flexkv.transfer.ce_replay import CETraceEntry, replay
from test_ce_replay import record
mode = sys.argv[1]
assert c_ext.ce_trace_enabled() == (mode != "disabled")
if mode == "write_failure":
    os.symlink("/dev/full", f"{os.environ['FLEXKV_CE_TRACE_FILE']}.{os.getpid()}.jsonl")
def capture():
    for kind in range(3):
        for direction in ("H2D", "D2H"):
            for backend in ("copy_engine", "sm_kernel"):
                for adaptive in (False, True) if backend == "copy_engine" else (False,):
                    data = record(kind, direction)
                    data["transfer_backend"] = backend
                    data["ce_path_id"] = -1 if backend == "copy_engine" else None
                    data["ce_config"]["path_opt_enabled"] = adaptive
                    assert replay(CETraceEntry(data))["byte_check"] == "passed"
if mode == "worker_exit":
    from types import SimpleNamespace
    from flexkv.transfer.workers.runtime import TransferWorkerBase
    class Worker(TransferWorkerBase):
        def __init__(self, *args):
            pass
        def run(self):
            capture()
        def launch_transfer(self, op):
            return True
        def shutdown(self):
            pass
    Worker._worker_process(0, None, None, None, SimpleNamespace(set=lambda: None))
    # multiprocessing can bypass C++ static destructors. The worker finally
    # must have drained every record before this abrupt interpreter exit.
    os._exit(0)
capture()
assert c_ext.ce_trace_shutdown() == 0
assert c_ext.ce_trace_shutdown() == 0
'''


@pytest.mark.parametrize("mode", ["enabled", "disabled", "unwritable", "write_failure", "truncated", "worker_exit"])
def test_capture_replay_and_logging_failure(tmp_path, mode):
    prefix = tmp_path / "capture"
    if mode == "unwritable":
        # A regular file used as a directory fails even for a root test runner.
        parent = tmp_path / "not-a-directory"
        parent.write_text("x")
        prefix = parent / "capture"
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", FLEXKV_CE_TRACE="0" if mode == "disabled" else "1",
               FLEXKV_CE_TRACE_FILE=str(prefix), FLEXKV_CE_TRACE_MAX_BLOCKS="1" if mode == "truncated" else "0")
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).parents[1])])
    result = subprocess.run([sys.executable, "-c", CAPTURE, mode], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = list(tmp_path.glob("capture.*.jsonl"))
    if mode in ("disabled", "unwritable", "write_failure"):
        assert len(paths) == (1 if mode == "write_failure" else 0)
        if mode != "disabled":
            assert "CE trace disabled" in result.stderr
        return
    assert len(paths) == 1
    entries = load_trace(paths[0])
    assert len(entries) == 18
    assert [entry.record["trace_id"] for entry in entries] == list(range(18))
    assert {entry.record["tensor_kind"] for entry in entries} == {0, 1, 2}
    for entry in entries:
        assert entry.record["offsets"] == {"gpu": 8, "cpu": 16}
        assert entry.record["total_num_layers"] == 4
        assert entry.record["dropped_before"] == 0
        if mode == "truncated":
            with pytest.raises(ValueError, match="truncated"):
                entry.layout()
        else:
            assert replay(entry)["byte_check"] == "passed"
            if entry.record["transfer_backend"] == "copy_engine":
                assert replay(entry, per_block=True)["byte_check"] == "passed"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_resolved_region_and_tp_captures(tmp_path):
    prefix = tmp_path / "region"
    script = r'''
from flexkv import c_ext
from test_region_batch_equivalence import (
    test_d2h_writes_the_same_cpu_bytes_as_the_per_group_path,
    test_roundtrip_through_the_batch_restores_every_regions_gpu_data,
)
for layout in ("LAYERFIRST", "BLOCKFIRST"):
    for ce in (False, True):
        test_d2h_writes_the_same_cpu_bytes_as_the_per_group_path(layout, 2, ce)
        test_roundtrip_through_the_batch_restores_every_regions_gpu_data(layout, ce)
assert c_ext.ce_trace_shutdown() == 0
'''
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", FLEXKV_CE_TRACE="1",
               FLEXKV_CE_TRACE_FILE=str(prefix), FLEXKV_CE_TRACE_MAX_BLOCKS="0")
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).parents[1])])
    for _ in range(2):
        result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    paths = list(tmp_path.glob("region.*.jsonl"))
    assert len(paths) == 2  # per-worker files never interleave JSON lines
    for path in paths:
        entries = load_trace(path)
        assert entries
        assert {entry.record["device"] for entry in entries} == {0, 1}
        assert any(entry.record["offsets"]["cpu"] > 0 for entry in entries)
        assert len({entry.record["tid"] for entry in entries}) >= 2
        # Keep the captured source/destination offsets and every region's layout.
        for entry in entries:
            assert replay(entry, device=entry.record["device"])["byte_check"] == "passed"
