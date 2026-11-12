"""FIFO sender for a slow worker pipe, independent of the shared scheduler.

Accepted transfer metadata retains its original graph/buffer lease until the
normal worker completion. A failed or rejected send reports the existing
negative-completion protocol; it is never retried or reported as successful.
"""

import queue
import threading


class SenderClosed(RuntimeError):
    pass


class SenderFull(RuntimeError):
    pass


class OrderedPipeSender:
    def __init__(self, connection, on_failed, *, capacity=2048, name="worker-pipe"):
        if capacity < 1:
            raise ValueError("Sender capacity must be positive")
        self.connection = connection
        self.on_failed = on_failed
        self.capacity = capacity
        # Reserve one entry for the shutdown sentinel, even when data is full.
        self._queue = queue.Queue(capacity + 1)
        self._lock = threading.Lock()
        self._accepting = True
        self._error = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False

    def _start_locked(self):
        if not self._started:
            try:
                self._thread.start()
            except Exception as exc:
                self._error = repr(exc)
                self._accepting = False
                raise SenderClosed(self._error) from exc
            self._started = True

    def submit(self, payload):
        if payload is None:
            raise ValueError("Use close() for the shutdown sentinel")
        with self._lock:
            if not self._accepting:
                raise SenderClosed(self._error or "Worker sender is closing")
            if self._queue.qsize() >= self.capacity:
                raise SenderFull("Worker sender queue is full")
            self._start_locked()
            self._queue.put_nowait(payload)

    def close(self):
        with self._lock:
            if not self._accepting:
                return
            self._start_locked()
            self._accepting = False
            self._queue.put_nowait(None)

    def join(self, timeout=None):
        if not self._started:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self):
        while True:
            payload = self._queue.get()
            try:
                self.connection.send(payload)
            except Exception as exc:
                rejected = [payload]
                with self._lock:
                    self._accepting = False
                    self._error = repr(exc)
                    while True:
                        try:
                            rejected.append(self._queue.get_nowait())
                            self._queue.task_done()
                        except queue.Empty:
                            break
                # No scheduler lock or queue lock is held across callbacks.
                for item in rejected:
                    if item is not None:
                        self.on_failed(item, exc)
                return
            finally:
                self._queue.task_done()
            if payload is None:
                return
