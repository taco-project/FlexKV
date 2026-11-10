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
        max_data_sessions=0,
        epoch=None,
        metrics=None,
    ):
        if (
            max_sessions <= 0
            or max_reserved_bytes <= 0
            or max_batch_graphs <= 0
            or result_ttl_s <= 0
        ):
            raise ValueError("prefetch runtime limits must be positive")
        if (
            isinstance(max_data_sessions, bool)
            or not isinstance(max_data_sessions, int)
            or max_data_sessions < 0
        ):
            raise ValueError("max_data_sessions must be a nonnegative integer")
        self.max_data_sessions = max_data_sessions
        self._data_sessions = set()
        self.backend = backend
        self.clock = clock
        self.max_sessions = max_sessions
        self.max_reserved_bytes = max_reserved_bytes
        self.result_ttl_s = result_ttl_s
        self.max_batch_graphs = max_batch_graphs
        self.epoch = epoch or uuid.uuid4().hex
        self.metrics = metrics
        self.sessions: Dict[PrefetchHandle, Session] = OrderedDict()
        # Released/expired terminal snapshots only: never retain a context or
        # charge an admission slot after the caller has relinquished its lease.
        self.expired = OrderedDict()
        self.graphs = {}
        self.reserved_bytes = 0
        self._next_id = 0
        self.accepting = True
        freeze_policies()

    def start(self, tokens, options=None, namespace=None, client_id=0, block_hashes=None):
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
        # Only forward hashes when given, so backends without the argument keep working.
        extra = {} if block_hashes is None else {"block_hashes": block_hashes}
        context = self.backend.begin(tokens, namespace, client_id, options, **extra)
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
            if session.terminal_at is not None:
                self._retire(session)
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
        # Finish a bounded wave of real reads before opening more sessions.
        # Every request keeps its metadata future, deadline and result lease;
        # waiting for this slot is not a capacity miss.
        if (
            self.max_data_sessions
            and session.handle not in self._data_sessions
            and len(self._data_sessions) >= self.max_data_sessions
        ):
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
            # Outstanding graphs hold temporary byte credits. A late session
            # without its first chunk must wait for those graphs to drain,
            # rather than turn transient contention into an empty cache miss.
            # With no reserved I/O, retain the permanent-capacity fallback.
            if not session.chunks and not self.reserved_bytes:
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
        self._data_sessions.add(session.handle)
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
        last = len(session.spans) - 1
        if last >= 0 and session.spans[last][1] == begin:
            session.spans[last] = (session.spans[last][0], end)
        else:
            session.spans.append((begin, end))

    def _finalize(self, session):
        if session.reason is None or session.chunks or session.terminal_at is not None:
            return
        session.terminal_at = self.clock()
        # No submitted graph remains here; releasing a data slot does not
        # release the separately retained CPU result/pin lease.
        self._data_sessions.discard(session.handle)
        session.version += 1
        snapshot = self.snapshot(session.handle)
        elapsed_s = session.terminal_at - session.created
        drain_s = session.terminal_at - (
            session.sealed_at if session.sealed_at is not None else session.terminal_at
        )
        logger = logging.getLogger(__name__)
        logger.info(
            "[FlexKV-Prefetch] epoch=%s session_id=%d policy=%s reason=%s "
            "submitted_chunks=%d sealed_submit_seq=%d loaded_tokens=%d "
            "elapsed_ms=%.3f drain_ms=%.3f error=%r",
            session.handle.epoch,
            session.handle.session_id,
            session.options.policy,
            session.reason,
            session.submitted,
            session.sealed_seq,
            snapshot.loaded_tokens,
            elapsed_s * 1000,
            drain_s * 1000,
            session.error,
        )
        if self.metrics is not None:
            try:
                self.metrics.record_prefetch_terminal(
                    session.options.policy, session.reason, snapshot.outcome,
                    snapshot.loaded_tokens if snapshot.lease_valid else 0, elapsed_s, drain_s,
                )
            except Exception:
                logger.exception("Failed to record chunked prefetch metrics")
        if session.released or session.discard_result:
            self._retire(session)

    def _retire(self, session):
        """Return admission credit, retaining only an unprotected snapshot.

        Claimed graphs must have drained before their context can be released.
        An unconsumed terminal lease stays in sessions until release or TTL;
        tombstones instead have a separate, bounded history budget.
        """
        if session.terminal_at is None or session.chunks:
            raise RuntimeError("cannot retire an undrained prefetch session")
        session.version += 1
        snapshot = replace(self.snapshot(session.handle), lease_valid=False)
        self.backend.release(session.context)
        del self.sessions[session.handle]
        self.expired[session.handle] = snapshot
        while len(self.expired) > self.max_sessions * 4:
            self.expired.popitem(last=False)

    def release(self, handle):
        if handle not in self.sessions:
            return
        session = self._get(handle)
        if session.released:
            return
        session.released = True
        self.seal(session, "request_abort")
        session.version += 1
        if session.terminal_at is not None:
            self._retire(session)
        else:
            # A query-only session has nothing to drain. Inflight sessions
            # remain in the ledger until tick finalizes their last graph.
            self._finalize(session)

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
                # release() already retired the context and bounded the history.

    def stop_all(self, reason):
        self.accepting = False
        for handle in list(self.sessions):
            self.stop(handle, reason)

    @property
    def drained(self):
        return not self.graphs
