"""Single task-state/completion owner and bounded control-IPC outbox."""

from concurrent.futures import Future
from functools import wraps
import queue
import threading


def on_runtime(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        runtime = getattr(self, "_runtime", None)
        if runtime is not None and not runtime.is_owner:
            return runtime.call(method, self, *args, **kwargs)
        return method(self, *args, **kwargs)

    return call


class TaskRuntime:
    """Own engine state; callers wait on Futures without consuming result pipes.

    Poll active work every 2ms because existing handles lack selectable
    completion descriptors. When idle, sleep until a command or a retained
    result expires, so a hot GPU cache does not pay for background polling.
    Each pass reserves at most one chunk/session to bound control work.
    """

    def __init__(self, engine, poll_s=0.002):
        self.engine = engine
        self.poll_s = poll_s
        self.commands = queue.Queue(maxsize=1024)
        self.wake = threading.Event()
        self.changed = threading.Condition()
        self.stopping = False
        self.error = None
        self.state_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run, name="flexkv-task-control", daemon=True
        )

    @property
    def is_owner(self):
        return threading.current_thread() is self.thread

    def start(self):
        self.thread.start()

    def call(self, fn, *args, **kwargs):
        if self.is_owner:
            return fn(*args, **kwargs)
        future = Future()
        # Atomically check liveness and enqueue against the final drain.
        with self.state_lock:
            if self.error:
                raise RuntimeError(
                    "FlexKV task runtime failed; inflight buffers retained"
                ) from self.error
            if self.stopping:
                raise RuntimeError("FlexKV task runtime is stopped")
            self.commands.put_nowait((future, fn, args, kwargs))
            self.wake.set()
        return future.result()

    def _run(self):
        try:
            while not self.stopping:
                self.wake.clear()
                for _ in range(64):
                    try:
                        future, fn, args, kwargs = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        future.set_result(fn(*args, **kwargs))
                    except Exception as exc:
                        future.set_exception(exc)
                for handle in self.engine.transfer_handles:
                    if handle.error:
                        raise RuntimeError(
                            "transfer submission failed; completion is uncertain"
                        ) from handle.error
                self.engine._update_tasks(timeout=0)
                self.engine._reap_completed_tasks()
                submitted = self.engine._prefetch.tick()
                with self.changed:
                    self.changed.notify_all()
                # Do not sleep while a second window slot can be filled. Once
                # the window is full, wait for completion/command, not busy-spin.
                active = self.engine._prefetch.sessions.values()
                runnable = any(
                    s.reason is None
                    and s.target is not None
                    and s.cursor < s.target
                    and len(s.chunks) < s.options.max_inflight_chunks
                    for s in active
                )
                if self.stopping:
                    break
                # A command batch is bounded above; pending commands must not
                # sleep after their wake event was consumed at loop entry.
                if (runnable and submitted) or not self.commands.empty():
                    timeout = 0
                else:
                    timeout = self.engine._next_runtime_wakeup(self.poll_s)
                self.wake.wait(timeout)
        except BaseException as exc:
            self.error = exc
        finally:
            with self.state_lock:
                self.stopping = True
            while True:
                try:
                    future, _, _, _ = self.commands.get_nowait()
                except queue.Empty:
                    break
                future.set_exception(RuntimeError("FlexKV task runtime stopped"))
            with self.changed:
                self.changed.notify_all()

    def stop(self):
        with self.state_lock:
            self.stopping = True
        self.wake.set()
        if not self.is_owner:
            self.thread.join()


class QueuedTransferHandle:
    """One sender per existing handle. Queued graphs already count as inflight.

    The public handle's data graph and worker protocol are unchanged. A send
    exception is ambiguous: retain target buffers, surface unhealthy runtime,
    and never retry an operation that may already be running.
    """

    def __init__(self, handle, capacity=1024):
        self.closed = False
        self.original = handle
        self._handle = handle._handle
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.thread = threading.Thread(
            target=self._run, name="flexkv-graph-submit", daemon=True
        )
        self.thread.start()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def submit(self, graph, task_end_op_id=-1):
        self.queue.put_nowait(("submit", (graph,), {"task_end_op_id": task_end_op_id}))

    def submit_batch(self, graphs):
        self.queue.put_nowait(("submit_batch", (graphs,), {}))

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                name, args, kwargs = item
                getattr(self.original, name)(*args, **kwargs)
            except BaseException as exc:
                self.error = exc
                return
            finally:
                self.queue.task_done()

    def shutdown(self):
        # Caller has already drained all graphs. Never close a pipe under its
        # writer or free registered buffers on an uncertain-send failure.
        if self.closed:
            return
        if self.error:
            raise RuntimeError(
                "cannot shutdown an uncertain transfer handle"
            ) from self.error
        self.queue.put(None)
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("transfer outbox did not drain")
        self.original.shutdown()
        self.closed = True
