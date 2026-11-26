"""CPU-only coverage of the opt-in PUT/GET timeline and process propagation."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "flexkv/transfer/put_trace.py"


def load_module():
    spec = importlib.util.spec_from_file_location("put_trace_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trace(monkeypatch):
    for name in ("FLEXKV_PUT_TRACE_CONFIG", "FLEXKV_PUT_TRACE_RUN_ID", "FLEXKV_PUT_TRACE", "FLEXKV_PUT_TRACE_FILE"):
        monkeypatch.delenv(name, raising=False)
    module = load_module()
    yield module
    module.configure()
    module._close_file()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def enable(trace, tmp_path):
    output = tmp_path / "put.jsonl"
    trace.configure({"enabled": True, "output_file": str(output)})
    return output


def test_disabled_is_no_io_no_clock(trace, tmp_path, monkeypatch):
    path = tmp_path / "disabled.jsonl"
    assert trace.configure({"enabled": False, "output_file": str(path)})["enabled"] is False
    monkeypatch.setattr(trace.time, "perf_counter_ns", Mock(side_effect=AssertionError("clock")))
    trace.emit("unused")
    assert trace.now() == 0
    trace.request_created(1)
    trace.bind_graph(None, 1, 0)
    trace.submitted(None, None, 0)
    trace.worker_begin(None, 0)
    trace.worker_end(None, 0, "ok")
    assert not path.exists()


@pytest.mark.parametrize("value", [True, [], {"enabled": "false"}, {"enabled": 1},
                                    {"output_file": ""}, {"output_file": 1},
                                    {"unknown_key": True}])
def test_rejects_invalid_config(trace, value):
    with pytest.raises(ValueError):
        trace.configure(value)


def test_environment_expansion_and_file_security(trace, tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_TEST_DIR", str(tmp_path))
    config = trace.configure({"enabled": True, "output_file": "$TRACE_TEST_DIR/a.jsonl"})
    assert config["output_file"] == str(tmp_path / "a.jsonl")
    assert stat.S_IMODE((tmp_path / "a.jsonl").stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="requires environment variable"):
        trace.configure({"enabled": True, "output_file": "${MISSING_TEST_TRACE_ROOT}/a"})
    link = tmp_path / "symlink.jsonl"
    link.symlink_to(tmp_path / "a.jsonl")
    with pytest.raises(OSError):
        trace.configure({"enabled": True, "output_file": str(link)})
    assert not trace.ENABLED


def test_fifo_is_rejected_without_blocking(trace, tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises((OSError, ValueError)):
        trace.configure({"enabled": True, "output_file": str(path)})
    assert not trace.ENABLED


def test_graph_worker_terminal_and_retirement_protocol(trace, tmp_path):
    output = enable(trace, tmp_path)
    op = SimpleNamespace(op_id=8, graph_id=7, transfer_type=SimpleNamespace(name="D2H"), predecessors=set())
    graph = SimpleNamespace(graph_id=7, _op_map={8: op})
    created = trace.request_created(5)
    trace.bind_graph(graph, 5, created)
    task = SimpleNamespace(graph=graph, _put_trace=graph._put_trace)
    trace.graph_event("submit_graph", graph)
    trace.graph_event("scheduler_admit", graph)
    child = SimpleNamespace(op_id=9, graph_id=7, transfer_type=op.transfer_type)
    trace.replica(op, child)
    worker_op = SimpleNamespace(valid_block_num=3, _received_ns=trace.now())
    trace.submitted(child, worker_op, 1)
    trace.batch_prepared([worker_op], trace.now())
    trace.worker_begin(worker_op, trace.now())
    trace.mark("producer_fence_begin")
    trace.mark("producer_fence_end")
    trace.volume(bytes=768)
    trace.worker_end(worker_op, trace.now(), "ok")
    trace.task_event("gpu_release_eligible", task)
    task.graph = None
    trace.task_event("task_terminal", task, status="completed")
    trace.task_event("retired", task)
    events = records(output)
    assert [e["event"] for e in events] == ["request_created", "planned", "submit_graph", "scheduler_admit",
                                           "worker_submit", "worker_done", "gpu_release_eligible",
                                           "task_terminal", "retired"]
    assert len({e["request"] for e in events}) == 1
    done = events[5]
    assert done["parent_op"] == 8 and done["op"] == 9
    assert done["physical"] == events[4]["ticket"]
    assert done["volume"]["bytes"] == 768 and done["logical_blocks"] == 3
    assert "producer_fence_end" in [name for name, _ in done["points"]]
    assert all(e["wall_ns"] > 0 for e in events)


def test_coalesced_physical_transfer_is_not_double_counted(trace, tmp_path):
    output = enable(trace, tmp_path)
    batch = SimpleNamespace(done=True, finished_ns=trace.now())
    for ticket, size in (("first", 1024), ("second", 0)):
        op = SimpleNamespace(_put_diag={"ticket": ticket}, _gdr_batch=batch, valid_block_num=2)
        trace.worker_begin(op, trace.now())
        trace.volume(bytes=size)
        trace.worker_end(op, trace.now(), "ok")
    done = records(output)
    assert len(done) == 2
    assert {row["physical"] for row in done} == {"first"}
    assert {row["volume"]["bytes"] for row in done} == {1024}
    assert {row["ns"] for row in done} == {batch.finished_ns}
    assert trace._local.points is None


def test_runtime_write_failure_does_not_escape_or_repeat(trace, tmp_path, monkeypatch):
    enable(trace, tmp_path)
    monkeypatch.setattr(trace.os, "write", Mock(side_effect=OSError("disk full")))
    with pytest.warns(RuntimeWarning, match="disabled"):
        trace.emit("failure")
    trace.emit("ignored")
    assert trace.os.write.call_count == 1


def test_short_writes_are_completed(trace, tmp_path, monkeypatch):
    output = enable(trace, tmp_path)
    write = os.write
    monkeypatch.setattr(trace.os, "write", lambda fd, data: write(fd, data[:7]))
    trace.emit("partial_write", number=17)
    assert records(output)[0]["number"] == 17


def test_threads_and_fresh_process_share_run_and_append(trace, tmp_path):
    output = enable(trace, tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: trace.emit("thread", number=i), range(80)))
    script = ("import importlib.util; s=importlib.util.spec_from_file_location('trace_child', "
              + repr(str(SOURCE)) + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
              "[m.emit('child', number=i) for i in range(80)]")
    processes = [subprocess.Popen([sys.executable, "-c", script], env=os.environ.copy(),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, (stdout, stderr)
    result = records(output)
    assert len(result) == 240
    assert len({r["run"] for r in result}) == 1
    assert len({r["pid"] for r in result}) == 3
    assert all(r["host"] and r["ns"] > 0 for r in result)


@pytest.mark.parametrize("outcome", ["ok", "failed", "partial"])
@pytest.mark.parametrize("write_failure", [False, True])
def test_real_worker_loop_reports_result_even_when_tracing_fails(trace, tmp_path, monkeypatch, outcome, write_failure):
    import __future__
    import time
    import numpy as np
    from enum import Enum
    output = enable(trace, tmp_path)
    tree = ast.parse((ROOT / "flexkv/transfer/workers/runtime.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TransferWorkerBase")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    transfer_type = Enum("TransferType", "H2D D2H LAYERWISE")
    result_type = type("WorkerTransferResult", (), {})
    namespace = dict(put_trace=trace, trace=Mock(build_worker_metrics=Mock(return_value=None)),
                     time=time, nvtx=Mock(), flexkv_logger=Mock(), _WORKER_IDLE_POLL_S=60,
                     TransferType=transfer_type, WorkerTransferResult=result_type,
                     get_nvtx_range_color=lambda _: "blue")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "runtime.py", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    op = SimpleNamespace(transfer_type=transfer_type.D2H, transfer_op_id=1,
                         transfer_graph_id=2, valid_block_num=3,
                         _put_diag={"ticket": "test/worker/1"})
    connection = SimpleNamespace(poll=Mock(return_value=True), recv=Mock(side_effect=[op, None]))
    result = result_type()
    result.block_results = np.array([0, -1, 2], dtype=np.int64)
    def launch(_):
        if outcome == "failed":
            raise RuntimeError("DMA failed")
        trace.volume(bytes=384)
        return result if outcome == "partial" else True
    worker = SimpleNamespace(transfer_conn=connection, worker_id=1, launch_transfer=launch,
                             finished_ops_queue=Mock(), _trace_bytes_per_block=lambda: 128)
    if write_failure:
        monkeypatch.setattr(trace.os, "write", Mock(side_effect=OSError("disk full")))
        with pytest.warns(RuntimeWarning, match="disabled"):
            namespace["run"](worker)
    else:
        namespace["run"](worker)
        done = records(output)[0]
        assert done["event"] == "worker_done"
        assert done["status"] == outcome
        if outcome != "failed":
            assert done["volume"]["bytes"] == 384
        if outcome == "partial":
            assert done["volume"]["completed_blocks"] == 2
    expected = (result, True, None) if outcome == "partial" else (1, outcome == "ok", None)
    worker.finished_ops_queue.put.assert_called_once_with(expected)


@pytest.mark.parametrize("stage", ["batch_prepared", "worker_begin", "worker_end"])
def test_diagnostic_exceptions_do_not_drop_or_fail_accepted_batch(trace, tmp_path, monkeypatch, stage):
    import __future__
    import time
    from enum import Enum
    enable(trace, tmp_path)
    diagnostic = Mock(side_effect=ValueError("bad diagnostic metadata"))
    monkeypatch.setattr(trace, stage, diagnostic)
    tree = ast.parse((ROOT / "flexkv/transfer/workers/runtime.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TransferWorkerBase")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    transfer_type = Enum("TransferType", "D2H LAYERWISE")
    namespace = dict(put_trace=trace, trace=Mock(build_worker_metrics=Mock(return_value=None)),
                     time=time, nvtx=Mock(), flexkv_logger=Mock(), _WORKER_IDLE_POLL_S=60,
                     TransferType=transfer_type, WorkerTransferResult=type("WorkerTransferResult", (), {}),
                     get_nvtx_range_color=lambda _: "blue")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "runtime.py", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    ops = [SimpleNamespace(transfer_type=transfer_type.D2H, transfer_op_id=i, transfer_graph_id=1,
                           valid_block_num=3, _put_diag={"ticket": f"worker/{i}"}) for i in (1, 2)]
    connection = SimpleNamespace(
        poll=Mock(side_effect=[True, True, True, KeyboardInterrupt("accepted batch was dropped")]),
        recv=Mock(side_effect=ops + [None]))
    worker = SimpleNamespace(transfer_conn=connection, worker_id=1, launch_transfer=Mock(return_value=True),
                             finished_ops_queue=Mock(), _trace_bytes_per_block=lambda: 128)
    namespace["run"](worker)
    assert diagnostic.call_count == (1 if stage == "batch_prepared" else 2)
    assert [call.args[0] for call in worker.launch_transfer.call_args_list] == ops
    assert [call.args[0] for call in worker.finished_ops_queue.put.call_args_list] == [
        (1, True, None), (2, True, None)]


def test_current_architecture_has_event_and_config_wiring():
    expected = {
        "flexkv/kvtask.py": ("request_created", "bind_graph", "task_terminal", "gpu_release_eligible", "retired"),
        "flexkv/transfer/workers/handle.py": ("put_trace.submitted", "worker_send_failed"),
        "flexkv/transfer/workers/runtime.py": ("put_trace.worker_begin", "put_trace.worker_end", "bytes=int(transfer_size)"),
        "flexkv/transfer/transfer_engine.py": ("put_trace.configure", "put_trace.replica", '"op_done"'),
        "flexkv/transfer/scheduler.py": ("scheduler_admit", "op_ready"),
        "flexkv/common/config.py": ("parse_put_trace_config", "put_trace_run_id = put_trace.RUN"),
    }
    for path, markers in expected.items():
        source = (ROOT / path).read_text()
        ast.parse(source)
        assert all(marker in source for marker in markers), path
    assert '"flexkv/transfer/put_trace.py"' in (ROOT / "setup.py").read_text()
