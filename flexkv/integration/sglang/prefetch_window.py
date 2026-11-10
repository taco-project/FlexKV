"""Opt-in host-only L3 admission window, independent of main's chunk runtime."""

import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)
_TERMINAL = frozenset(("success", "failed", "cancelled"))


def parse_prefetch_max_blocks(raw, num_cpu_blocks=0):
    value = "0" if raw is None else str(raw).strip().lower()
    if value in ("", "0"):
        return 0
    if value == "auto":
        if num_cpu_blocks <= 0:
            raise ValueError("FLEXKV_L3_PREFETCH_MAX_BLOCKS=auto needs a positive CPU pool")
        return max(1, num_cpu_blocks // 2)
    budget = int(value)
    if budget < 0:
        raise ValueError("FLEXKV_L3_PREFETCH_MAX_BLOCKS must be nonnegative")
    return budget


def _positive(name, default):
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _seconds(name, default):
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass
class _Entry:
    rid: str
    tokens: object
    count: int
    blocks: int
    namespace: object
    handle: int
    state: str = "staged"
    task_ids: set = field(default_factory=set)
    loaded: int = 0
    planned: int = 0
    failed: bool = False
    detached: bool = False
    deadline: float = 0.0
    ready_at: float = 0.0


class PrefetchWindow:
    """Scheduler-thread helper; completion credits require a whole-graph drain.

    A timeout/cancel only detaches the request; its block credit remains held
    until ``wait(completely=True)`` certifies completion. NOTFOUND is unknown,
    not a drain certificate. All rank decisions originate on the sync leader.
    Namespaces passed to ``start`` are already encoded by the connector.
    """

    @classmethod
    def from_connector(cls, connector, *, scatter_channels):
        cpu_blocks = int(getattr(connector.cache_config, "num_cpu_blocks", 0) or 0)
        budget = parse_prefetch_max_blocks(
            os.environ.get("FLEXKV_L3_PREFETCH_MAX_BLOCKS"), cpu_blocks)
        if not budget:
            return None
        if getattr(connector, "_chunked_prefetch", False):
            raise ValueError("L3 block window and main chunked prefetch cannot both be enabled")
        if cpu_blocks and budget > cpu_blocks:
            raise ValueError("L3 block window cannot exceed the CPU pool")
        model_config = getattr(connector, "model_config", None)
        if int(getattr(model_config, "pp_size", 1) or 1) != 1:
            raise NotImplementedError("L3 block window requires PP=1")
        # Every admission/expiry decision is made by the sync leader and fanned
        # out; without collective sync each TP/CP rank would decide on its own.
        if (int(getattr(model_config, "effective_tp_size", 1) or 1) > 1
                and not connector._sync_ctx.needs_sync):
            raise ValueError("L3 block window requires collective sync across TP/CP")
        return cls(connector, budget, scatter_channels=scatter_channels)

    def __init__(self, connector, max_blocks, *, scatter_channels, clock=time.monotonic):
        self.connector = connector
        self.channels = scatter_channels
        self.clock = clock
        self.max_blocks = int(max_blocks)
        self.page_size = int(connector.page_size)
        if self.max_blocks <= 0 or self.page_size <= 0:
            raise ValueError("L3 window and page size must be positive")
        self.max_inflight = _positive("FLEXKV_L3_PREFETCH_MAX_INFLIGHT", 64)
        self.max_entries = _positive("FLEXKV_L3_PREFETCH_MAX_STAGED", 512)
        self.max_queued_blocks = _positive("FLEXKV_L3_PREFETCH_MAX_QUEUED_BLOCKS", max_blocks)
        self.poll_interval = _seconds("FLEXKV_L3_PREFETCH_POLL_INTERVAL_S", 0.01)
        self.wait_s = _seconds("FLEXKV_L3_PREFETCH_WAIT_S", 1.0)
        self.unconsumed_ttl = _seconds("FLEXKV_L3_PREFETCH_UNCONSUMED_TTL_S", 60.0)
        self.entries = OrderedDict()
        self.held_blocks = 0
        self._next_handle = 1 << 62
        self._next_poll = 0.0
        self._poison = ""

    @property
    def _leader(self):
        return self.connector._sync_ctx.is_sync_leader

    def _scatter(self, payload, channel, *, blocking=False):
        if self.connector._sync_ctx.needs_sync:
            try:
                return self.connector._sync_ctx.scatter(payload, channel=channel, blocking=blocking)
            except BaseException:
                self._poison = "L3 control fanout failed; admission ownership is retained"
                raise
        return payload

    def _can_admit(self, blocks):
        return (self.held_blocks + blocks <= self.max_blocks
                and sum(e.state == "active" for e in self.entries.values()) < self.max_inflight)

    def _launch(self, entry):
        result = {"ids": [], "planned": 0, "error": "", "poison": ""}
        try:
            tokens = entry.tokens() if callable(entry.tokens) else entry.tokens
            tokens = np.asarray(tokens, dtype=np.int64)
            if tokens.ndim != 1 or len(tokens) != entry.count:
                raise ValueError("prefetch token count differs from the admission budget")
        except Exception as exc:
            result["error"] = str(exc)
            return result
        try:
            submitted = self.connector.kv_manager.prefetch_async(
                token_ids=tokens, namespace=entry.namespace,
                swa_aware=getattr(self.connector, "_swa_kv_pool", None) is not None)
            task_id = int(submitted[0] if isinstance(submitted, tuple) else submitted)
            if task_id >= 0:
                result["ids"] = [task_id]
            if isinstance(submitted, tuple) and len(submitted) > 1:
                result["planned"] = int(submitted[1])
        except BaseException as exc:
            result["error"] = str(exc) or type(exc).__name__
            launched = getattr(exc, "launched_task_ids", None)
            if launched is None:
                result["poison"] = "Unacknowledged L3 submission: " + result["error"]
            else:
                try:
                    result["ids"] = list(dict.fromkeys(int(tid) for tid in launched if int(tid) >= 0))
                except (TypeError, ValueError):
                    result["poison"] = "Unacknowledged L3 submission: invalid task ownership metadata"
        return result

    def _apply_launch(self, entry, launch):
        self._poison = launch["poison"] or self._poison
        entry.tokens = None
        if not launch["ids"] and not self._poison:
            self.entries.pop(entry.rid, None)
            return
        entry.state = "active"
        entry.task_ids = set(launch["ids"])
        entry.planned = launch["planned"]
        entry.deadline = self.clock() + self.wait_s
        entry.detached = bool(launch["error"])
        self.held_blocks += entry.blocks
        if launch["ids"]:
            entry.handle = launch["ids"][0]
        self._next_poll = 0.0
        if launch["error"]:
            logger.warning("L3 prefetch admission failed for %s: %s", entry.rid, launch["error"])
            self._cancel_ids(entry.task_ids)

    def start(self, rid, token_ids, sglang_req_id=None, namespace=None, *, token_count=None):
        if self._poison:
            raise RuntimeError(self._poison)
        if not rid or not self.connector._prefetch_enabled:
            return -1
        existing = self.entries.get(rid)
        if existing is not None:
            return -1 if existing.detached else existing.handle
        count = int(token_count) if callable(token_ids) else len(token_ids)
        blocks = (count + self.page_size - 1) // self.page_size
        payload = {"action": "reject", "launch": None}
        entry = _Entry(rid, token_ids, count, blocks,
                       list(namespace) if namespace is not None else None, self._next_handle)
        if self._leader and 0 < blocks <= self.max_blocks and len(self.entries) < self.max_entries:
            staged = [e for e in self.entries.values() if e.state == "staged"]
            if not staged and self._can_admit(blocks):
                payload = {"action": "launch", "launch": self._launch(entry)}
            elif sum(e.blocks for e in staged) + blocks <= self.max_queued_blocks:
                payload["action"] = "stage"
        if self._leader and payload["action"] != "reject":
            self._next_handle += 1
            self.entries[rid] = entry
            if payload["action"] == "launch":
                self._apply_launch(entry, payload["launch"])
        payload = self._scatter(payload, self.channels.PREFETCH_START)
        if payload["action"] == "reject":
            return -1
        if not self._leader:
            self._next_handle += 1
            self.entries[rid] = entry
            if payload["action"] == "launch":
                self._apply_launch(entry, payload["launch"])
        if self._poison:
            raise RuntimeError(self._poison)
        return entry.handle if rid in self.entries and not entry.detached else -1

    def _cancel_ids(self, task_ids):
        if task_ids and self._leader:
            try:
                self.connector.kv_manager.cancel(list(task_ids))
            except Exception as exc:
                logger.warning("L3 cancellation remains pending drain: %s", exc)

    def _records(self, responses):
        records = []
        for rid, entry in self.entries.items():
            for tid in entry.task_ids:
                response = responses.get(tid)
                status = getattr(response, "status", None)
                status = str(getattr(status, "value", status)).lower()
                if status not in _TERMINAL:
                    continue
                mask = getattr(response, "return_mask", None)
                loaded = 0
                if status == "success":
                    if mask is None:
                        loaded = entry.planned
                    elif isinstance(mask, list):
                        loaded = sum(int(np.count_nonzero(part)) for part in mask)
                    else:
                        loaded = int(np.count_nonzero(mask))
                records.append((rid, tid, status, min(loaded, entry.count)))
        return records

    def _apply_results(self, records):
        for rid, tid, status, loaded in records:
            entry = self.entries.get(rid)
            if entry is None or tid not in entry.task_ids:
                continue
            entry.task_ids.remove(tid)
            entry.failed |= status != "success"
            entry.loaded = max(entry.loaded, loaded)
            if not entry.task_ids:
                entry.state = "ready"
                entry.ready_at = self.clock()
                if entry.failed or entry.detached:
                    entry.loaded = 0
                if entry.detached:
                    self._release(rid)
                elif not entry.loaded:
                    self.held_blocks -= entry.blocks
                    entry.blocks = 0

    def _release(self, rid):
        entry = self.entries.pop(rid, None)
        if entry is not None and entry.state != "staged":
            self.held_blocks -= entry.blocks

    def _collect_responses(self, timeout=0):
        responses = dict(getattr(self, "_idle_responses", {}))
        ids = list(dict.fromkeys(tid for entry in self.entries.values()
                                 for tid in entry.task_ids if tid not in responses))
        if ids:
            responses.update(self.connector.kv_manager.wait(
                ids, timeout=timeout, completely=True) or {})
        return responses

    def poll_idle(self):
        """Drive leader-only idle ticks without creating collective clock edges."""
        if not self._leader or not self.entries:
            return
        if self._poison:
            raise RuntimeError(self._poison)
        now = self.clock()
        if now < getattr(self, "_next_idle_poll", 0.0):
            return
        self._next_idle_poll = now + self.poll_interval
        try:
            responses = self._collect_responses()
            completed = {record[1] for record in self._records(responses)}
            self._idle_responses = {tid: responses[tid] for tid in completed}
        except Exception as exc:
            logger.warning("L3 idle poll failed; retaining block ownership: %s", exc)

    def poll(self):
        if self._poison:
            raise RuntimeError(self._poison)
        if not self.entries:
            return
        payload = {"results": [], "expired": [], "released": [], "promoted": []}
        if self._leader and self.clock() >= self._next_poll:
            now = self.clock()
            self._next_poll = now + self.poll_interval
            try:
                responses = self._collect_responses()
                payload["results"] = self._records(responses)
                self._idle_responses = {}
            except Exception as exc:
                logger.warning("L3 poll failed; retaining block ownership: %s", exc)
            self._apply_results(payload["results"])
            for rid, entry in list(self.entries.items()):
                if entry.state == "active" and not entry.detached and now >= entry.deadline:
                    payload["expired"].append(rid)
                    entry.detached = True
                elif entry.state == "ready" and now - entry.ready_at >= self.unconsumed_ttl:
                    payload["released"].append(rid)
                    self._release(rid)
            self._cancel_ids([tid for rid in payload["expired"] for tid in self.entries[rid].task_ids])
            for rid, entry in list(self.entries.items()):
                if entry.state != "staged":
                    continue
                if not self._can_admit(entry.blocks):
                    break
                launch = self._launch(entry)
                payload["promoted"].append((rid, launch))
                self._apply_launch(entry, launch)
                if self._poison:
                    break
        payload = self._scatter(payload, self.channels.PREFETCH_PROGRESS)
        if not self._leader:
            self._apply_results(payload["results"])
            for rid in payload["expired"]:
                self.entries[rid].detached = True
            for rid in payload["released"]:
                self._release(rid)
            for rid, launch in payload["promoted"]:
                self._apply_launch(self.entries[rid], launch)
        if self._poison:
            raise RuntimeError(self._poison)

    def check(self, rid):
        self.poll()
        entry = self.entries.get(rid)
        return entry is None or entry.detached or entry.state == "ready"

    def pop_loaded(self, rid):
        entry = self.entries.get(rid)
        if entry is None or entry.state != "ready":
            return 0
        loaded = entry.loaded
        self._release(rid)
        return loaded

    def cancel(self, rid):
        entry = self.entries.get(rid)
        if entry is None:
            return
        if entry.state != "active":
            self._release(rid)
            return
        entry.detached = True
        self._cancel_ids(entry.task_ids)

    def drain(self, timeout=30.0, *, collective=True):
        if not collective and not self._leader:
            return
        if self._poison:
            raise RuntimeError(self._poison + "; refusing to release the host window")
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("L3 drain timeout must be finite and nonnegative")
        for rid, entry in list(self.entries.items()):
            if entry.state == "active":
                entry.detached = True
            else:
                self._release(rid)
        payload = {"results": [], "error": ""}
        if self._leader:
            ids = list(dict.fromkeys(tid for entry in self.entries.values() for tid in entry.task_ids))
            self._cancel_ids(ids)
            if ids:
                try:
                    responses = self._collect_responses(timeout=timeout)
                    payload["results"] = self._records(responses)
                    self._idle_responses = {}
                except Exception as exc:
                    payload["error"] = str(exc)
        if collective:
            payload = self._scatter(payload, self.channels.RESET, blocking=True)
        self._apply_results(payload["results"])
        if self.entries or payload["error"]:
            raise RuntimeError("L3 drain incomplete; retaining host ownership: " + payload["error"])
