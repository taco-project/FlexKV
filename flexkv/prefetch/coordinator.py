"""Single-owner session scheduler. The backend owns cache plans, not policy.

Backend methods: begin, resolve, reserve, submit_batch, complete, commit,
discard, release. resolve is nonblocking (None means metadata still pending).
An admitted chunk retains credit through commit/discard, including out-of-order
completion. Transport-accepted work is never cancelled or forgotten.
"""

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import logging
import time
import uuid
from typing import Any, Dict, Optional

from .policy import freeze_policies, get_policy
from .types import (
    PrefetchCapacityExhausted,
    PrefetchHandle,
    PrefetchOptions,
    PrefetchSnapshot,
)


@dataclass
class Session:
    handle: PrefetchHandle
    options: PrefetchOptions
    created: float
    deadline: Optional[float]
    context: Any
    target: Optional[int] = None
    start: int = 0
    cursor: int = 0
    committed: int = 0
    chunks: OrderedDict = field(default_factory=OrderedDict)
    spans: list = field(default_factory=list)
    reason: Optional[str] = None
    sealed_seq: Optional[int] = None
    submitted: int = 0
    version: int = 0
    terminal_at: Optional[float] = None
    released: bool = False
    error: Optional[str] = None
    discard_result: bool = False
    sealed_at: Optional[float] = None


class PrefetchCoordinator:
    def __init__(
        self,
        backend,
        *,
        clock=time.monotonic,
        max_sessions=128,
        max_reserved_bytes=512 * 1024 * 1024,
        result_ttl_s=60.0,
        max_batch_graphs=32,
        epoch=None,
    ):
        if (
            max_sessions <= 0
            or max_reserved_bytes <= 0
            or max_batch_graphs <= 0
            or result_ttl_s <= 0
        ):
            raise ValueError("prefetch runtime limits must be positive")
        self.backend = backend
        self.clock = clock
        self.max_sessions = max_sessions
        self.max_reserved_bytes = max_reserved_bytes
        self.result_ttl_s = result_ttl_s
        self.max_batch_graphs = max_batch_graphs
        self.epoch = epoch or uuid.uuid4().hex
        self.sessions: Dict[PrefetchHandle, Session] = OrderedDict()
        self.expired = OrderedDict()  # bounded lightweight terminal tombstones
        self.graphs = {}
        self.reserved_bytes = 0
        self._next_id = 0
        self.accepting = True
        freeze_policies()

    def start(self, tokens, options=None, namespace=None, client_id=0):
        options = options or PrefetchOptions()
        options.validate()
        if not self.accepting:
            raise RuntimeError("prefetch admission is closed")
        self.reap()
        # Include retained result leases in the admission bound.
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError("prefetch session capacity exhausted")
        handle = PrefetchHandle(self.epoch, self._next_id)
        self._next_id += 1
        now = self.clock()
        context = self.backend.begin(tokens, namespace, client_id, options)
        n = max(0, len(tokens) - options.candidate_start_token)
        deadline = now + options.budget(n) if options.policy == "timeout" else None
        self.sessions[handle] = Session(handle, options, now, deadline, context)
        return handle

    def _get(self, handle):
        if not isinstance(handle, PrefetchHandle) or handle.epoch != self.epoch:
            raise KeyError("unknown or stale prefetch handle")
        return self.sessions[handle]

    def seal(self, session, reason):
        if session.reason is None:
            session.reason = reason
            session.sealed_at = self.clock()
            session.sealed_seq = session.submitted
            session.version += 1
            self.backend.cancel_query(session.context)

    def stop(self, handle, reason="request_abort"):
        if handle in self.expired:
            return self.expired[handle]
        session = self._get(handle)
        self.seal(session, reason)
        if reason in ("reset", "shutdown"):
            session.discard_result = True
        self._finalize(session)
        return self.snapshot(handle)

    def demand(self, handles):
        for handle in handles:
            if handle in self.expired:
                continue
            session = self._get(handle)
            reason = get_policy(session.options.policy)(
                "demand", self.clock(), session.deadline
            )
            if reason:
                self.seal(session, reason)

    def tick(self):
        batch = []
        # Rotate active requests for fair bounded admission.
        for handle in list(self.sessions):
            session = self.sessions[handle]
            if session.terminal_at is not None:
                continue
            reason = get_policy(session.options.policy)(
                "tick", self.clock(), session.deadline
            )
            if reason:
                self.seal(session, reason)
            self._commit_ready(session)
            self._finalize(session)
            if session.reason is not None:
                continue
            if len(batch) >= self.max_batch_graphs:
                break
            try:
                chunk = self._plan_and_claim(session)
                if chunk is not None:
                    batch.append(chunk)
            except PrefetchCapacityExhausted:
                self.seal(session, "capacity")
                self._finalize(session)
            except Exception as exc:
                session.error = f"{type(exc).__name__}: {exc}"
                self.seal(session, "backend_error")
                self._finalize(session)
            self.sessions.move_to_end(handle)
        if batch:
            # The backend must either accept all descriptors into its bounded
            # outbox or report a known-not-submitted failure. Ambiguous transport
            # failures keep resources retained and make the runtime unhealthy.
            self.backend.submit_batch(batch)
        self.reap()
        return len(batch)

    def _plan_and_claim(self, session):
        """Reserve one chunk and record ownership before handing it to IPC."""
        if session.target is None:
            bounds = self.backend.resolve(session.context)
            if bounds is None:
                return None
            session.start, session.target = bounds
            session.cursor = session.committed = session.start
            session.version += 1
        if session.cursor >= session.target:
            self.seal(session, "complete")
            self._finalize(session)
            return None
        if len(session.chunks) >= session.options.max_inflight_chunks:
            return None
        chunk = self.backend.reserve(
            session.context,
            session.cursor,
            session.target,
            session.options,
            self.max_reserved_bytes - self.reserved_bytes,
        )
        if chunk is None:
            # Do not wait for capacity held by our own pinned prefix.
            if not session.chunks:
                self.seal(session, "capacity")
                self._finalize(session)
            return None
        # Planning may have consumed time: check again at CLAIMED.
        reason = get_policy(session.options.policy)(
            "tick", self.clock(), session.deadline
        )
        if reason:
            self.backend.discard(session.context, chunk)
            self.seal(session, reason)
            self._finalize(session)
            return None
        if (
            chunk.begin != session.cursor
            or chunk.end <= chunk.begin
            or chunk.end > session.target
        ):
            self.backend.discard(session.context, chunk)
            raise RuntimeError("invalid prefetch chunk span")
        if chunk.nbytes > self.max_reserved_bytes - self.reserved_bytes:
            self.backend.discard(session.context, chunk)
            raise RuntimeError("prefetch backend exceeded reservation budget")
        session.chunks[chunk.graph.graph_id] = chunk
        self.graphs[chunk.graph.graph_id] = session
        self.reserved_bytes += chunk.nbytes
        session.cursor = chunk.end
        session.submitted += 1
        session.version += 1
        if session.cursor == session.target:
            self.seal(session, "complete")
        return chunk

    def on_completion(self, completion):
        session = self.graphs.get(completion.graph_id)
        if session is None:
            return False
        chunk = session.chunks[completion.graph_id]
        self.backend.complete(chunk, completion)
        session.version += 1
        return True

    def _commit_ready(self, session):
        while session.chunks:
            graph_id, chunk = next(iter(session.chunks.items()))
            if not chunk.done:
                break
            if session.discard_result or chunk.begin != session.committed:
                self.backend.discard(session.context, chunk)
            else:
                try:
                    end = self.backend.commit(session.context, chunk)
                except Exception as exc:
                    # A cache mutation failure has ambiguous ownership; stop the
                    # runtime instead of double-freeing partially inserted data.
                    session.error = f"publication failure: {exc}"
                    raise
                if not chunk.begin <= end <= chunk.end:
                    raise RuntimeError("invalid published prefetch prefix")
                if end > chunk.begin:
                    self._add_span(session, chunk.begin, end)
                session.committed = end
                if end != chunk.end and session.reason in (None, "complete"):
                    session.reason = "partial_read"
                    session.sealed_seq = session.submitted
                    if session.sealed_at is None:
                        session.sealed_at = self.clock()
            self.reserved_bytes -= chunk.nbytes
            del session.chunks[graph_id]
            self.graphs.pop(graph_id, None)
            session.version += 1

    @staticmethod
    def _add_span(session, begin, end):
        if session.spans and session.spans[-1][1] == begin:
            session.spans[-1] = (session.spans[-1][0], end)
        else:
            session.spans.append((begin, end))

    def _finalize(self, session):
        if session.reason is None or session.chunks or session.terminal_at is not None:
            return
        session.terminal_at = self.clock()
        session.version += 1
        logging.getLogger(__name__).info(
            "[FlexKV-Prefetch] epoch=%s session_id=%d policy=%s reason=%s "
            "submitted_chunks=%d sealed_submit_seq=%d loaded_tokens=%d "
            "elapsed_ms=%.3f drain_ms=%.3f error=%r",
            session.handle.epoch,
            session.handle.session_id,
            session.options.policy,
            session.reason,
            session.submitted,
            session.sealed_seq,
            self.snapshot(session.handle).loaded_tokens,
            (session.terminal_at - session.created) * 1000,
            (
                session.terminal_at
                - (
                    session.sealed_at
                    if session.sealed_at is not None
                    else session.terminal_at
                )
            )
            * 1000,
            session.error,
        )
        if session.released or session.discard_result:
            self.backend.release(session.context)

    def release(self, handle):
        if handle not in self.sessions:
            return
        session = self._get(handle)
        if session.released:
            return
        session.released = True
        self.seal(session, "request_abort")
        if session.terminal_at is not None:
            self.backend.release(session.context)
        session.version += 1

    def snapshot(self, handle):
        if handle in self.expired:
            return self.expired[handle]
        session = self._get(handle)
        terminal = session.terminal_at is not None
        state = "terminal" if terminal else ("draining" if session.reason else "active")
        if session.target is None and not session.reason:
            state = "planning"
        reusable_end = self.backend.result_end(session.context, session.committed)
        spans = tuple(
            (begin, min(end, reusable_end))
            for begin, end in session.spans
            if begin < reusable_end
        )
        outcome = None
        if terminal:
            if session.reason in ("request_abort", "reset", "shutdown"):
                outcome = "aborted"
            elif session.error:
                outcome = "partial" if spans else "failed"
            elif session.target is not None and reusable_end == session.target:
                outcome = "complete" if spans else "empty"
            else:
                outcome = "partial" if spans else "empty"
        return PrefetchSnapshot(
            handle,
            session.version,
            state,
            terminal,
            outcome,
            session.reason,
            session.target,
            reusable_end,
            spans,
            len(session.chunks),
            sum(c.nbytes for c in session.chunks.values()),
            session.submitted,
            session.sealed_seq,
            not session.released and not session.discard_result,
            session.error,
        )

    def next_wakeup(self, poll_s):
        """Keep active work moving; retained results only need their TTL timer."""
        now = self.clock()
        delay = None
        for session in self.sessions.values():
            if session.terminal_at is None:
                return poll_s
            expiry = max(0, session.terminal_at + self.result_ttl_s - now)
            delay = expiry if delay is None else min(delay, expiry)
        return delay

    def reap(self):
        now = self.clock()
        for handle, session in list(self.sessions.items()):
            if session.terminal_at is None:
                continue
            if now - session.terminal_at >= self.result_ttl_s:
                self.release(handle)
                self.expired[handle] = replace(
                    self.snapshot(handle),
                    version=session.version + 1,
                    state="expired",
                    outcome="expired",
                    lease_valid=False,
                    reusable_prefix_end_token=0,
                    l3_loaded_spans=(),
                )
                del self.sessions[handle]
                while len(self.expired) > self.max_sessions * 4:
                    self.expired.popitem(last=False)

    def stop_all(self, reason):
        self.accepting = False
        for handle in list(self.sessions):
            self.stop(handle, reason)

    @property
    def drained(self):
        return not self.graphs
