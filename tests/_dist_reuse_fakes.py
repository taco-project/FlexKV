"""Shared test fakes for dist_reuse unit tests.

Underscore-prefixed so pytest doesn't try to collect this file.  Uses no
external dependencies beyond the Python stdlib — in particular, it does
**not** import ``redis`` or ``torch``, so it is safe to import on a
CPU-only test machine with the ``--noconftest`` flag.
"""
from __future__ import annotations

import fnmatch
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


class FakeRedis:
    """Just enough of the redis-py surface for dist_reuse unit tests.

    Thread-safe at the command level (not at multi-command transactions) —
    individual method calls are each protected by an internal RLock.  The
    fake supports:

    * ``set/get/expire/delete``
    * ``hset(name, key=, value=)``, ``hset(name, mapping=)``,
      ``hgetall(name)``, ``hget(name, field)``
    * ``scan(cursor, match, count)``, ``scan_iter(match, count)``
    * ``incr(name)``
    * ``publish(channel, message)``  (no-op; return subscriber count = 0)
    * ``pipeline()``  (emits a ``FakePipeline`` that queues ops and
      applies them atomically on ``execute()``)
    * ``rpush(name, *values)``, ``lrange(name, start, end)``
    * ``exists(*names)``

    TTLs are honored lazily — a key is treated as expired only when a
    command touches it and finds ``now >= expiry_at``.  This matches
    redis-py's observable behaviour closely enough for unit testing.
    """

    def __init__(self, time_fn=time.monotonic) -> None:
        self._strs: Dict[str, str] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._lists: Dict[str, List[str]] = {}
        self._expiry: Dict[str, float] = {}
        self._time = time_fn
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- GC
    def _gc(self, name: str) -> None:
        exp = self._expiry.get(name)
        if exp is not None and self._time() >= exp:
            self._strs.pop(name, None)
            self._hashes.pop(name, None)
            self._lists.pop(name, None)
            self._expiry.pop(name, None)

    def _all_keys(self) -> List[str]:
        """Snapshot of every live key (after lazy GC)."""
        keys = set(self._strs) | set(self._hashes) | set(self._lists)
        for k in list(keys):
            self._gc(k)
        return list(set(self._strs) | set(self._hashes) | set(self._lists))

    # ------------------------------------------------------------- strings
    def set(self, name: str, value: str, ex: Optional[int] = None) -> bool:
        with self._lock:
            self._strs[name] = value
            if ex is not None:
                self._expiry[name] = self._time() + float(ex)
            else:
                self._expiry.pop(name, None)
            return True

    def get(self, name: str) -> Optional[str]:
        with self._lock:
            self._gc(name)
            return self._strs.get(name)

    def incr(self, name: str) -> int:
        with self._lock:
            self._gc(name)
            cur = int(self._strs.get(name, "0"))
            cur += 1
            self._strs[name] = str(cur)
            return cur

    def exists(self, *names: str) -> int:
        with self._lock:
            count = 0
            for n in names:
                self._gc(n)
                if (
                    n in self._strs
                    or n in self._hashes
                    or n in self._lists
                ):
                    count += 1
            return count

    # ------------------------------------------------------------ TTL
    def expire(self, name: str, ex: int) -> bool:
        with self._lock:
            self._gc(name)
            if (
                name not in self._strs
                and name not in self._hashes
                and name not in self._lists
            ):
                return False
            self._expiry[name] = self._time() + float(ex)
            return True

    def delete(self, *names: str) -> int:
        with self._lock:
            n = 0
            for k in names:
                removed = False
                if k in self._strs:
                    self._strs.pop(k, None)
                    removed = True
                if k in self._hashes:
                    self._hashes.pop(k, None)
                    removed = True
                if k in self._lists:
                    self._lists.pop(k, None)
                    removed = True
                self._expiry.pop(k, None)
                if removed:
                    n += 1
            return n

    # ------------------------------------------------------------ hashes
    def hset(
        self,
        name: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        mapping: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self._lock:
            h = self._hashes.setdefault(name, {})
            new_fields = 0
            if mapping:
                for k, v in mapping.items():
                    if k not in h:
                        new_fields += 1
                    h[str(k)] = str(v)
            if key is not None:
                if key not in h:
                    new_fields += 1
                h[str(key)] = str(value) if value is not None else ""
            return new_fields

    def hget(self, name: str, field: str) -> Optional[str]:
        with self._lock:
            self._gc(name)
            return self._hashes.get(name, {}).get(field)

    def hgetall(self, name: str) -> Dict[str, str]:
        with self._lock:
            self._gc(name)
            return dict(self._hashes.get(name, {}))

    # -------------------------------------------------------------- lists
    def rpush(self, name: str, *values: Any) -> int:
        with self._lock:
            lst = self._lists.setdefault(name, [])
            for v in values:
                lst.append(str(v))
            return len(lst)

    def lrange(self, name: str, start: int, end: int) -> List[str]:
        with self._lock:
            self._gc(name)
            lst = self._lists.get(name, [])
            if end == -1:
                end = len(lst) - 1
            return list(lst[start : end + 1])

    # --------------------------------------------------------------- scan
    def scan(
        self,
        cursor: int = 0,
        match: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Tuple[int, List[str]]:
        with self._lock:
            all_keys = sorted(self._all_keys())
        if match is None:
            filtered = all_keys
        else:
            filtered = [k for k in all_keys if fnmatch.fnmatchcase(k, match)]
        # For simplicity (and since FakeRedis is tiny) we return everything
        # in one go; real redis returns batches of size ~count.
        return 0, filtered

    def scan_iter(
        self, match: Optional[str] = None, count: Optional[int] = None
    ) -> Iterable[str]:
        _, keys = self.scan(cursor=0, match=match, count=count)
        return iter(keys)

    # ---------------------------------------------------- misc / no-ops
    def ping(self) -> bool:
        return True

    def publish(self, channel: str, message: str) -> int:
        return 0

    def pubsub(self):
        raise NotImplementedError("FakeRedis.pubsub() is not supported in unit tests")

    def close(self) -> None:
        pass

    # ---------------------------------------------------------- pipeline
    def pipeline(self, transaction: bool = True):  # noqa: ARG002 — kept for compat
        return _FakePipeline(self)

    # ------------------------------------------------------------ helpers
    def force_expire(self, name: str) -> None:
        """Test helper: drop a key as if its TTL had elapsed."""
        with self._lock:
            self._strs.pop(name, None)
            self._hashes.pop(name, None)
            self._lists.pop(name, None)
            self._expiry.pop(name, None)

    def snapshot(self) -> Dict[str, Any]:
        """Return a read-only view of the current store — convenient for
        assert-style introspection in tests."""
        with self._lock:
            return {
                "strs": dict(self._strs),
                "hashes": {k: dict(v) for k, v in self._hashes.items()},
                "lists": {k: list(v) for k, v in self._lists.items()},
            }


class _FakePipeline:
    """Mimic ``redis.Redis.pipeline()`` — queue commands, execute on flush."""

    def __init__(self, owner: FakeRedis) -> None:
        self._owner = owner
        self._ops: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []

    def hset(self, *args, **kwargs):
        self._ops.append(("hset", args, kwargs))
        return self

    def delete(self, *args, **kwargs):
        self._ops.append(("delete", args, kwargs))
        return self

    def rpush(self, *args, **kwargs):
        self._ops.append(("rpush", args, kwargs))
        return self

    def set(self, *args, **kwargs):
        self._ops.append(("set", args, kwargs))
        return self

    def expire(self, *args, **kwargs):
        self._ops.append(("expire", args, kwargs))
        return self

    def execute(self) -> List[Any]:
        results = []
        for name, args, kwargs in self._ops:
            method = getattr(self._owner, name)
            results.append(method(*args, **kwargs))
        self._ops.clear()
        return results


class ManualClock:
    """Deterministic monotonic clock, for time-sensitive tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt
