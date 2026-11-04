"""Persistent, single-owner control IPC, isolated from ordinary DP responses."""

from concurrent.futures import Future
from contextlib import suppress
import os
import queue
import threading
import uuid

import zmq
from flexkv.common.request import StoreReservationError

from flexkv.server.request import ControlRequest
from flexkv.server.utils import get_zmq_socket


class ControlTransportError(RuntimeError):
    """The server may have acted without delivering a trustworthy response."""

    retain_slots = True


class ControlClient:
    """Queue control work without waiting on scheduler threads.

    Only this owner creates, uses and closes its sockets. A slow response is
    never abandoned: later commands (including close) follow it on the same
    ordered stream, so a late start is still released. Synchronous callers may
    time out waiting, but the owner retains the actual transaction.
    """

    def __init__(self, server_port, client_id):
        self.server_port = server_port
        self.client_id = client_id
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.closed = False
        self.error = None
        self.close_future = None
        self.thread = threading.Thread(
            target=self._run, name="flexkv-control-ipc", daemon=True,
        )
        self.thread.start()

    def submit(self, action, **payload):
        future = Future()
        future.set_running_or_notify_cancel()
        with self.lock:
            if self.closed or self.error is not None:
                raise RuntimeError("control owner is closed or failed") from self.error
            self.commands.put_nowait((future, action, payload))
        return future

    def close(self):
        with self.lock:
            if self.close_future is None:
                self.closed = True
                self.close_future = Future()
                if self.error is not None:
                    self.close_future.set_exception(self.error)
                else:
                    self.commands.put_nowait((self.close_future, "close", {}))
            future = self.close_future
        # Never destroy an in-use socket from the caller. On timeout the owner
        # keeps its transaction and resources, and shutdown reports the failure.
        future.result(timeout=30)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError("control owner did not stop")

    def _run(self):
        context = zmq.Context()
        path = "/tmp/flexkv-control-" + uuid.uuid4().hex
        reply_port = "ipc://" + path
        send = reply = None
        current = []
        try:
            reply = get_zmq_socket(context, zmq.PULL, reply_port, True)
            send = get_zmq_socket(context, zmq.PUSH, self.server_port, False)
            send.setsockopt(zmq.SNDTIMEO, 1000)
            sequence = 0
            while True:
                current = [self.commands.get()]
                while len(current) < 64 and current[len(current) - 1][1] != "close":
                    try:
                        current.append(self.commands.get_nowait())
                    except queue.Empty:
                        break
                sequence += 1
                send.send_pyobj(ControlRequest(
                    self.client_id, reply_port, "batch",
                    {"calls": [(action, payload) for _, action, payload in current]},
                    request_id=sequence,
                ))
                response = reply.recv_pyobj()
                if response.request_id != sequence:
                    raise RuntimeError("control response sequence mismatch")
                if response.error:
                    raise RuntimeError(response.error)
                if len(response.result) != len(current):
                    raise RuntimeError("control response length mismatch")
                stopping = current[len(current) - 1][1] == "close"
                for (future, _, _), result in zip(current, response.result):
                    if result.error:
                        error = (
                            StoreReservationError(result.error, result.cleanup_task_ids)
                            if result.cleanup_task_ids is not None
                            else RuntimeError(result.error)
                        )
                        future.set_exception(error)
                    else:
                        future.set_result(result.result)
                current = []
                if stopping:
                    return
        except BaseException as exc:
            failure = ControlTransportError(str(exc))
            failure.__cause__ = exc
            with self.lock:
                self.error = failure
                self.closed = True
                pending = list(current)
                while True:
                    try:
                        pending.append(self.commands.get_nowait())
                    except queue.Empty:
                        break
            for future, _, _ in pending:
                if not future.done():
                    future.set_exception(failure)
        finally:
            if send is not None:
                send.close(linger=0)
            if reply is not None:
                reply.close(linger=0)
            context.term()
            with suppress(FileNotFoundError):
                os.unlink(path)
