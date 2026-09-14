"""Parent-side handle for one worker process.

Kept apart from ``runtime`` because it runs in the *parent*: it never touches
CUDA, never imports a worker class, and is what ``TransferEngine`` holds.
"""

import threading
import time
from multiprocessing.connection import Connection
from typing import Any, Union

import torch.multiprocessing as mp

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import LayerwiseTransferOp, TransferOp
from flexkv.transfer import trace
from flexkv.transfer.worker_op import (
    WorkerLayerwiseTransferOp,
    WorkerTransferOp,
)


class WorkerHandle:
    """handle for worker process"""
    def __init__(self, worker_id: int, transfer_conn: Connection, process: mp.Process, ready_event: Any):
        self.worker_id = worker_id
        self.transfer_conn = transfer_conn
        self.process = process
        self.ready_event = ready_event
        # One handle is reachable from several TransferType keys (a mooncake
        # or GDS worker serves both directions), and TransferEngine shuts
        # handles down one thread each -- so shutdown() can be entered twice,
        # concurrently, on the same object. See the note on shutdown().
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False

    def submit_transfer(self, op: Union[TransferOp, LayerwiseTransferOp]) -> None:
        if isinstance(op, LayerwiseTransferOp):
            worker_op = WorkerLayerwiseTransferOp(op)
        else:
            worker_op = WorkerTransferOp(op)
        if trace._TRACE_ON:
            submitted_ns = time.perf_counter_ns()
            worker_op.prof_submitted_ns = submitted_ns
            trace.set_submit_ns(op.op_id, submitted_ns)
            trace.inc_inflight()
        self.transfer_conn.send(worker_op)

    def control(
        self, command: str, payload: Any = None, timeout: float = 120.0
    ) -> Any:
        request_id = f"{self.worker_id}:{time.monotonic_ns()}"
        self.transfer_conn.send({
            "type": "control",
            "command": command,
            "payload": payload,
            "request_id": request_id,
        })
        if not self.transfer_conn.poll(timeout):
            raise TimeoutError(
                f"Worker {self.worker_id} timed out handling {command}"
            )
        reply = self.transfer_conn.recv()
        if reply.get("request_id") != request_id:
            raise RuntimeError(f"Unexpected worker control reply: {reply}")
        if "error" in reply:
            raise RuntimeError(
                f"Worker {self.worker_id} {command} failed: {reply['error']}"
            )
        return reply.get("result")

    def shutdown(self) -> None:
        """Stop the worker process. Idempotent and safe to call concurrently.

        The second caller must not re-enter the body: ``process.terminate()``
        and ``transfer_conn.close()`` race with the first caller's
        ``join()``/``close()``, and ``__del__`` can fire the whole sequence a
        third time from the GC thread. Holding the lock for the *duration* --
        not just around the flag -- also makes a second caller block until the
        process is really gone, so ``shutdown(); assert not is_alive()`` holds
        whichever call site wins.
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True

            try:
                self.transfer_conn.send(None)
            except (BrokenPipeError, OSError, EOFError):
                pass  # Pipe already closed / peer gone

            timeout = float(GLOBAL_CONFIG_FROM_ENV.worker_shutdown_timeout_s)
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                flexkv_logger.warning(
                    f"[WorkerHandle] worker {self.worker_id} still alive after "
                    f"{timeout:.0f}s graceful shutdown; force terminate"
                )
                self.process.terminate()
                self.process.join(timeout=30)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join()

            try:
                self.transfer_conn.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            if getattr(self, "process", None) is not None and self.process.is_alive():
                self.shutdown()
        except Exception:
            pass
