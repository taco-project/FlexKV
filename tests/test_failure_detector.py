"""Unit tests for ``flexkv.cache.failure_detector`` (Phase 0 task 0-L).

Uses an in-memory Redis fake — no real Redis or network required.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

from flexkv.common.dist_reuse.failure_detector import (
    FailureDetector,
    InstanceSession,
    RedisSessionClient,
    make_session_epoch,
)
from flexkv.common.dist_reuse.sharing_domain_namespace import SharingDomainNamespace


# ---------------------------------------------------------------------------
# In-memory Redis fake
# ---------------------------------------------------------------------------
class FakeRedis:
    """Just enough of the redis-py surface for FailureDetector / RedisSessionClient."""

    def __init__(self, time_fn=time.monotonic) -> None:
        self._store: Dict[str, str] = {}
        self._expiry: Dict[str, float] = {}
        self._time = time_fn
        self._lock = threading.RLock()

    # ---- core ops ----
    def set(self, name: str, value: str, ex: Optional[int] = None) -> bool:
        with self._lock:
            self._store[name] = value
            if ex is not None:
                self._expiry[name] = self._time() + float(ex)
            else:
                self._expiry.pop(name, None)
            return True

    def get(self, name: str) -> Optional[str]:
        with self._lock:
            self._gc(name)
            return self._store.get(name)

    def expire(self, name: str, ex: int) -> bool:
        with self._lock:
            self._gc(name)
            if name not in self._store:
                return False
            self._expiry[name] = self._time() + float(ex)
            return True

    def delete(self, *names: str) -> int:
        with self._lock:
            n = 0
            for k in names:
                if k in self._store:
                    self._store.pop(k, None)
                    self._expiry.pop(k, None)
                    n += 1
            return n

    def scan_iter(self, match: Optional[str] = None, count: Optional[int] = None) -> Iterable[str]:
        with self._lock:
            # GC any expired keys before scanning.
            for k in list(self._store):
                self._gc(k)
            keys = list(self._store)
        if match is None:
            return iter(keys)
        # Translate Redis glob '*' → fnmatch.  We only ever use "prefix*suffix" patterns.
        import fnmatch
        return iter(k for k in keys if fnmatch.fnmatchcase(k, match))

    # ---- helpers ----
    def _gc(self, name: str) -> None:
        exp = self._expiry.get(name)
        if exp is not None and self._time() >= exp:
            self._store.pop(name, None)
            self._expiry.pop(name, None)

    def force_expire(self, name: str) -> None:
        """Test helper — drop a key as if its TTL had elapsed."""
        with self._lock:
            self._store.pop(name, None)
            self._expiry.pop(name, None)


# ---------------------------------------------------------------------------
# Manual clock
# ---------------------------------------------------------------------------
class ManualClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


# ---------------------------------------------------------------------------
# make_session_epoch
# ---------------------------------------------------------------------------
class TestSessionEpoch:
    def test_format(self):
        e = make_session_epoch()
        assert isinstance(e, str)
        assert "-" in e
        ms_part, rand_part = e.split("-", 1)
        assert len(ms_part) == 12
        assert len(rand_part) == 8
        # All hex
        int(ms_part, 16)
        int(rand_part, 16)

    def test_unique_per_call(self):
        seen = {make_session_epoch() for _ in range(100)}
        assert len(seen) == 100


# ---------------------------------------------------------------------------
# RedisSessionClient
# ---------------------------------------------------------------------------
class TestSessionClient:
    def _client(self, **overrides):
        clock = ManualClock()
        fake = FakeRedis(time_fn=clock)
        kwargs = dict(
            instance_id="inst-A",
            epoch="epoch-1",
            ttl_seconds=5,
            master_zmq_addr="tcp://10.0.0.1:6666",
            node_ids=[1, 2, 3],
            mooncake_addrs_by_sd={"sd0": "10.0.0.1:5555"},
        )
        kwargs.update(overrides)
        return RedisSessionClient(fake, **kwargs), fake, clock

    def test_register_writes_payload(self):
        sc, fake, _ = self._client()
        sc.register()
        raw = fake.get(sc.key)
        assert raw is not None
        payload = json.loads(raw)
        assert payload["instance_id"] == "inst-A"
        assert payload["epoch"] == "epoch-1"
        assert payload["node_ids"] == [1, 2, 3]
        assert payload["mooncake_addrs_by_sd"] == {"sd0": "10.0.0.1:5555"}

    def test_renew_extends_ttl(self):
        sc, fake, clock = self._client(ttl_seconds=5)
        sc.register()
        clock.advance(3.0)
        sc.renew()
        clock.advance(3.0)  # 6s total since register, but renewed at 3s → still alive
        assert fake.get(sc.key) is not None

    def test_renew_revives_expired_key(self):
        sc, fake, clock = self._client(ttl_seconds=2)
        sc.register()
        clock.advance(5.0)
        # Key has expired (lazy GC inside fake.get).
        assert fake.get(sc.key) is None
        # renew() should fall back to register().
        sc.renew()
        assert fake.get(sc.key) is not None

    def test_unregister(self):
        sc, fake, _ = self._client()
        sc.register()
        sc.unregister()
        assert fake.get(sc.key) is None

    def test_bad_ttl(self):
        with pytest.raises(ValueError):
            RedisSessionClient(FakeRedis(), instance_id="x", epoch="e", ttl_seconds=0)


# ---------------------------------------------------------------------------
# FailureDetector
# ---------------------------------------------------------------------------
class TestFailureDetector:
    def _seed(self, fake: FakeRedis, instance_id: str, epoch: str, ttl: int = 60):
        key = SharingDomainNamespace.instance_session_key(instance_id)
        payload = {
            "instance_id": instance_id,
            "epoch": epoch,
            "master_zmq_addr": "tcp://x:1",
            "node_ids": [],
            "mooncake_addrs_by_sd": {},
        }
        fake.set(key, json.dumps(payload), ex=ttl)

    def _detector(self, *, lost_log: List[str], seen_log: List[Tuple[str, str]]):
        clock = ManualClock()
        fake = FakeRedis(time_fn=clock)

        def on_lost(pid: str) -> None:
            lost_log.append(pid)

        def on_seen(pid: str, session: InstanceSession) -> None:
            seen_log.append((pid, session.epoch))

        fd = FailureDetector(
            fake,
            self_instance_id="self",
            poll_interval_seconds=0.5,
            on_peer_lost=on_lost,
            on_peer_seen=on_seen,
            time_fn=clock,
        )
        return fd, fake, clock

    def test_detects_new_peer(self):
        lost: List[str] = []
        seen: List[Tuple[str, str]] = []
        fd, fake, _ = self._detector(lost_log=lost, seen_log=seen)
        self._seed(fake, "peer-A", "e1")
        fd.poll_once()
        assert seen == [("peer-A", "e1")]
        assert lost == []
        # Second poll: no new event for the same peer/epoch.
        seen.clear()
        fd.poll_once()
        assert seen == []

    def test_detects_disappearance(self):
        lost: List[str] = []
        seen: List[Tuple[str, str]] = []
        fd, fake, _ = self._detector(lost_log=lost, seen_log=seen)
        self._seed(fake, "peer-A", "e1")
        fd.poll_once()
        # Simulate TTL expiry.
        fake.force_expire(SharingDomainNamespace.instance_session_key("peer-A"))
        fd.poll_once()
        assert lost == ["peer-A"]

    def test_detects_epoch_change(self):
        lost: List[str] = []
        seen: List[Tuple[str, str]] = []
        fd, fake, _ = self._detector(lost_log=lost, seen_log=seen)
        self._seed(fake, "peer-A", "e1")
        fd.poll_once()
        # Restart: same instance_id, new epoch.
        self._seed(fake, "peer-A", "e2")
        fd.poll_once()
        assert lost == ["peer-A"]
        # Two seen events: initial appear + post-restart re-appear.
        assert seen == [("peer-A", "e1"), ("peer-A", "e2")]

    def test_ignores_self(self):
        lost: List[str] = []
        seen: List[Tuple[str, str]] = []
        fd, fake, _ = self._detector(lost_log=lost, seen_log=seen)
        self._seed(fake, "self", "self-epoch")
        fd.poll_once()
        assert seen == []
        assert lost == []

    def test_skips_malformed_payload(self):
        lost: List[str] = []
        seen: List[Tuple[str, str]] = []
        fd, fake, _ = self._detector(lost_log=lost, seen_log=seen)
        # Garbage payload at a valid-looking key.
        fake.set(
            SharingDomainNamespace.instance_session_key("peer-A"),
            "not-json", ex=60,
        )
        fd.poll_once()  # must not raise
        assert seen == []
        assert lost == []

    def test_invalid_constructor_args(self):
        fake = FakeRedis()
        with pytest.raises(ValueError):
            FailureDetector(fake, "self", poll_interval_seconds=0)
        with pytest.raises(ValueError):
            FailureDetector(fake, "")

    def test_known_peers_view(self):
        lost: List[str] = []
        seen: List[Tuple[str, str]] = []
        fd, fake, _ = self._detector(lost_log=lost, seen_log=seen)
        self._seed(fake, "peer-A", "e1")
        self._seed(fake, "peer-B", "e1")
        fd.poll_once()
        assert fd.known_peers() == {"peer-A", "peer-B"}

    def test_lifecycle(self):
        """Light smoke test of start()/stop() — keeps the polling thread
        cycle short and verifies the thread terminates cleanly."""
        lost: List[str] = []
        seen: List[Tuple[str, str]] = []
        clock = ManualClock()
        fake = FakeRedis(time_fn=clock)
        fd = FailureDetector(
            fake, "self",
            poll_interval_seconds=0.05,
            on_peer_lost=lambda pid: lost.append(pid),
            on_peer_seen=lambda pid, s: seen.append((pid, s.epoch)),
            time_fn=clock,
        )
        fd.start()
        # Seed AFTER start() so the polling thread observes the event.
        self._seed(fake, "peer-X", "e1")
        # Give the loop a few iterations.
        time.sleep(0.3)
        fd.stop(timeout=2.0)
        assert "peer-X" in {pid for pid, _ in seen}

    def test_double_start_raises(self):
        lost: List[str] = []
        seen: List[Tuple[str, str]] = []
        fd, _, _ = self._detector(lost_log=lost, seen_log=seen)
        fd.start()
        try:
            with pytest.raises(RuntimeError):
                fd.start()
        finally:
            fd.stop(timeout=2.0)
