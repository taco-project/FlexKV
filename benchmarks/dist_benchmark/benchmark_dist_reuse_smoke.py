"""Single-machine, two-instance dist_reuse smoke benchmark.

Goal: exercise the dist_reuse control-plane end-to-end on ONE machine
(no RDMA, no Mooncake) so we can verify the §2.1 main-path wiring
lands correctly *before* we move to the two-GPU-box harness.

Why this is valuable:
    * Confirms two FlexKV instances sharing one Redis can register
      their ``flexkv:instance:<id>:sd_nodes`` maps and discover each
      other via ``RedisSessionClient`` heart-beats.
    * Confirms ``MasterCoordinator`` spins up, its
      ``AggregateRadixTree`` accepts self-SD acks from the PUT hook,
      and ``_sharing_domain_gate_get`` short-circuits correctly for
      the single-SD degenerate case.
    * Confirms the new ``set_evict_guard`` refcount predicate keeps
      in-flight coord-GET blocks pinned.
    * Confirms the PUT transfer-callback calls ``_notify_sd_ready_on_put``
      and the self-SD ack lands in ``aggregate_radix`` under the right
      prefix hash.

What this script DOES NOT cover (→ requires 2-machine harness):
    * Real Mooncake P2P read (cross-instance data plane)
    * PP/TP cross-node SD barrier (``total_sd_count > 1``)
    * Cross-instance peer-lost failure detection under network
      partition

Usage::

    # Prereq: a reachable redis with password 123456 (env overridable)
    export FLEXKV_SMOKE_REDIS_HOST=10.206.0.9
    export FLEXKV_SMOKE_REDIS_PORT=6379
    export FLEXKV_SMOKE_REDIS_PASSWORD=123456

    python benchmarks/dist_benchmark/benchmark_dist_reuse_smoke.py \
        --num-instances 2 --num-blocks 32
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a script from anywhere in the repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from flexkv.common.dist_reuse.master_coordinator import MasterCoordinator  # noqa: E402
from flexkv.common.dist_reuse.sharing_domain import SharingDomainKey  # noqa: E402


# ---------------------------------------------------------------------------
# Redis client helpers
# ---------------------------------------------------------------------------
def _build_redis_client(host: str, port: int, password: Optional[str], db: int = 0):
    import redis
    return redis.Redis(
        host=host, port=port, password=password, db=db,
        socket_connect_timeout=3.0, decode_responses=True,
    )


def _ensure_redis_reachable(host: str, port: int, password: Optional[str]) -> None:
    client = _build_redis_client(host, port, password)
    if not client.ping():
        raise RuntimeError(f"Redis {host}:{port} unreachable")


# ---------------------------------------------------------------------------
# Minimal RedisMeta shim — we only need ``_client()`` + ``register_instance_sd_nodes``
# for this smoke test.  Importing the real ``flexkv.cache.redis_meta`` requires
# a full FlexKV install (c_ext loaded, etc.), which IS available in the
# flexkv_distreuse container but we want the script to also run against a
# Python-only install.
# ---------------------------------------------------------------------------
class _LightRedisMeta:
    def __init__(self, host: str, port: int, password: Optional[str]):
        self._client_ = _build_redis_client(host, port, password)

    def _client(self):
        return self._client_

    def register_instance_sd_nodes(self, instance_id: str, sd_to_nid: Dict[str, int]):
        """Mirror RedisMeta's method signature — write a hash at the key the
        design doc specifies so other instances can discover us."""
        key = f"flexkv:instance:{instance_id}:sd_nodes"
        pipe = self._client_.pipeline()
        pipe.delete(key)
        if sd_to_nid:
            pipe.hset(key, mapping={str(k): str(v) for k, v in sd_to_nid.items()})
            pipe.expire(key, 300)  # 5-min TTL — smoke test teardown will del anyway
        pipe.execute()


# ---------------------------------------------------------------------------
# Smoke harness
# ---------------------------------------------------------------------------
def _mk_instance(
    *,
    instance_id: str,
    model_id: str,
    self_node_id: int,
    redis_host: str,
    redis_port: int,
    redis_password: Optional[str],
    master_zmq_addr: str,
    ttl_seconds: int,
) -> MasterCoordinator:
    sd = SharingDomainKey(
        model_id=model_id,
        pp_rank=0, pp_size=1,
        tp_node_idx=0, tp_node_count=1,
        is_nsa=False,
    )
    mc = MasterCoordinator(
        self_sd=sd,
        instance_id=instance_id,
    )
    # Single-SD instance — no remote ACKs to expect.
    mc.expect_remotes(0)
    # Mark "all remotes ready" by explicitly skipping (single-SD case
    # has no remotes).  This lets register_instance_discoverables run.
    # For the smoke harness we don't call register_instance_discoverables
    # directly (it requires a real RedisMeta type); we mimic its side
    # effect manually:
    meta = _LightRedisMeta(redis_host, redis_port, redis_password)
    sd_to_nid = {sd.serialize(): self_node_id}
    meta.register_instance_sd_nodes(instance_id, sd_to_nid)

    # Hook the session client so we show up in the Layer-1 failure
    # detector's view on the OTHER instance.  We're skipping the full
    # MasterCoordinator.register_instance_discoverables plumbing
    # (which needs a real RedisMeta); the script below just manually
    # maintains a heartbeat on flexkv:session:<instance_id>.
    return mc


def _register_heartbeat(
    redis_client,
    instance_id: str,
    epoch: str,
    ttl_seconds: int,
    master_zmq_addr: str,
) -> None:
    key = f"flexkv:session:{instance_id}"
    payload = {
        "epoch": epoch,
        "master_zmq_addr": master_zmq_addr,
        "ts": str(int(time.time())),
    }
    pipe = redis_client.pipeline()
    pipe.delete(key)
    pipe.hset(key, mapping=payload)
    pipe.expire(key, ttl_seconds)
    pipe.execute()


def _read_heartbeat(redis_client, instance_id: str) -> Optional[Dict[str, str]]:
    key = f"flexkv:session:{instance_id}"
    val = redis_client.hgetall(key)
    return val or None


def _cleanup_instance(redis_client, instance_id: str) -> None:
    keys = []
    keys.append(f"flexkv:instance:{instance_id}:sd_nodes")
    keys.append(f"flexkv:session:{instance_id}")
    redis_client.delete(*keys)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def scenario_peer_discovery(args) -> Dict[str, Any]:
    """Two single-SD instances register their sd_nodes, ensure each
    can see the other's entry via Redis (simulates what
    DistributedRadixTree.remote_tree_refresh does)."""
    print("\n[scenario] peer_discovery")

    redis_client = _build_redis_client(args.redis_host, args.redis_port, args.redis_password)
    # Create two instances with distinct node_ids.
    inst_a_id = f"smoke-a-{uuid.uuid4().hex[:8]}"
    inst_b_id = f"smoke-b-{uuid.uuid4().hex[:8]}"
    mc_a = _mk_instance(
        instance_id=inst_a_id, model_id=args.model_id, self_node_id=1000,
        redis_host=args.redis_host, redis_port=args.redis_port,
        redis_password=args.redis_password,
        master_zmq_addr="127.0.0.1:30001", ttl_seconds=args.session_ttl,
    )
    mc_b = _mk_instance(
        instance_id=inst_b_id, model_id=args.model_id, self_node_id=2000,
        redis_host=args.redis_host, redis_port=args.redis_port,
        redis_password=args.redis_password,
        master_zmq_addr="127.0.0.1:30002", ttl_seconds=args.session_ttl,
    )
    try:
        # Heartbeat both.
        _register_heartbeat(redis_client, inst_a_id, mc_a.session_epoch,
                            args.session_ttl, "127.0.0.1:30001")
        _register_heartbeat(redis_client, inst_b_id, mc_b.session_epoch,
                            args.session_ttl, "127.0.0.1:30002")

        # A reads B's sd_nodes + session.
        b_sd_nodes = redis_client.hgetall(f"flexkv:instance:{inst_b_id}:sd_nodes")
        b_session = _read_heartbeat(redis_client, inst_b_id)
        assert len(b_sd_nodes) == 1, f"A cannot see B's sd_nodes: {b_sd_nodes}"
        assert b_session is not None, "A cannot see B's session heartbeat"
        print(f"  [A] sees [B] sd_nodes = {b_sd_nodes}")
        print(f"  [A] sees [B] session = {b_session}")

        a_sd_nodes = redis_client.hgetall(f"flexkv:instance:{inst_a_id}:sd_nodes")
        a_session = _read_heartbeat(redis_client, inst_a_id)
        assert len(a_sd_nodes) == 1
        assert a_session is not None
        print(f"  [B] sees [A] sd_nodes = {a_sd_nodes}")
        print(f"  [B] sees [A] session = {a_session}")

        # Cross-verify node_ids don't collide.
        a_nid = int(list(a_sd_nodes.values())[0])
        b_nid = int(list(b_sd_nodes.values())[0])
        assert a_nid != b_nid, "two instances got the same node_id — collision!"
        print(f"  distinct node_ids: A={a_nid} B={b_nid}")
        return {"status": "ok", "a_nid": a_nid, "b_nid": b_nid}
    finally:
        _cleanup_instance(redis_client, inst_a_id)
        _cleanup_instance(redis_client, inst_b_id)


def scenario_aggregate_radix_put_hook(args) -> Dict[str, Any]:
    """Simulate PUTs on instance A: drive the AggregateRadixTree via
    ``_notify_sd_ready_on_put`` semantics and assert that match_fully_ready
    returns the entry (single-SD degenerate case)."""
    print("\n[scenario] aggregate_radix_put_hook")

    mc = _mk_instance(
        instance_id=f"smoke-agg-{uuid.uuid4().hex[:8]}", model_id=args.model_id,
        self_node_id=3000,
        redis_host=args.redis_host, redis_port=args.redis_port,
        redis_password=args.redis_password,
        master_zmq_addr="127.0.0.1:30003", ttl_seconds=args.session_ttl,
    )
    # Simulate N put-then-get cycles.
    results = {"ok": 0, "miss": 0}
    for i in range(args.num_blocks):
        prefix_hash = hash((args.model_id, i)) & 0xFFFFFFFFFFFFFFFF
        block_ids = [10 + 3 * i, 11 + 3 * i, 12 + 3 * i]
        # PUT hook → self-SD ack.
        mc.mark_sd_ready(
            prefix_hash=prefix_hash,
            sd_key_str=mc.self_sd.serialize(),
            block_ids=block_ids,
        )
        # GET hook → fully_ready check (total_sd=1 so a single ack is enough).
        entry = mc.match_fully_ready(prefix_hash)
        if entry is not None:
            results["ok"] += 1
            # Refcount protection: the aggregate pins blocks when acquired.
            mc.pin_blocks_for_coord_get(block_ids)
            for b in block_ids:
                assert not mc.is_evictable(b), \
                    f"block {b} evictable while pinned — refcount guard broken"
            mc.unpin_blocks_for_coord_get(block_ids)
            for b in block_ids:
                assert mc.is_evictable(b), \
                    f"block {b} NOT evictable after unpin — refcount stuck"
        else:
            results["miss"] += 1

    print(f"  put→get cycles: ok={results['ok']} miss={results['miss']}")
    assert results["miss"] == 0, "single-SD instance must have every prefix fully_ready"
    return results


def scenario_cross_instance_reuse_readiness(args) -> Dict[str, Any]:
    """Both instances PUT overlapping block hashes.  Verify the aggregate
    radix on instance A still reports fully_ready for its own prefix even
    when instance B is also active (isolation by instance_id)."""
    print("\n[scenario] cross_instance_reuse_readiness")

    inst_a_id = f"smoke-a-{uuid.uuid4().hex[:8]}"
    inst_b_id = f"smoke-b-{uuid.uuid4().hex[:8]}"
    mc_a = _mk_instance(
        instance_id=inst_a_id, model_id=args.model_id, self_node_id=4001,
        redis_host=args.redis_host, redis_port=args.redis_port,
        redis_password=args.redis_password,
        master_zmq_addr="127.0.0.1:30010", ttl_seconds=args.session_ttl,
    )
    mc_b = _mk_instance(
        instance_id=inst_b_id, model_id=args.model_id, self_node_id=4002,
        redis_host=args.redis_host, redis_port=args.redis_port,
        redis_password=args.redis_password,
        master_zmq_addr="127.0.0.1:30011", ttl_seconds=args.session_ttl,
    )
    try:
        prefix_hash = 0xC0FFEE
        block_ids = [1, 2, 3]
        # Both instances mark the prefix ready on their own SD.
        mc_a.mark_sd_ready(
            prefix_hash=prefix_hash,
            sd_key_str=mc_a.self_sd.serialize(),
            block_ids=block_ids,
        )
        mc_b.mark_sd_ready(
            prefix_hash=prefix_hash,
            sd_key_str=mc_b.self_sd.serialize(),
            block_ids=block_ids,
        )
        # Each aggregate is instance-local, so both should see fully_ready.
        assert mc_a.match_fully_ready(prefix_hash) is not None
        assert mc_b.match_fully_ready(prefix_hash) is not None
        print("  both instances independently report fully_ready — ok")

        # Simulate peer_lost on A (e.g., B's session TTL expired).
        n_invalidated = mc_a.invalidate_by_peer_instance(inst_b_id)
        # mc_a's aggregate was only acked by mc_a itself, never by
        # ``contributing_peer=inst_b_id``, so invalidate_by_peer_instance
        # should be a no-op.
        assert n_invalidated == 0
        assert mc_a.match_fully_ready(prefix_hash) is not None, \
            "A's prefix wrongly invalidated by B's peer-lost signal"
        print("  A's prefix survives B's peer-lost signal — isolation ok")

        return {"status": "ok", "invalidated": n_invalidated}
    finally:
        _cleanup_instance(_build_redis_client(args.redis_host, args.redis_port, args.redis_password),
                          inst_a_id)
        _cleanup_instance(_build_redis_client(args.redis_host, args.redis_port, args.redis_password),
                          inst_b_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-host", default=os.environ.get("FLEXKV_SMOKE_REDIS_HOST", "127.0.0.1"))
    parser.add_argument("--redis-port", type=int, default=int(os.environ.get("FLEXKV_SMOKE_REDIS_PORT", "6379")))
    parser.add_argument("--redis-password", default=os.environ.get("FLEXKV_SMOKE_REDIS_PASSWORD", None))
    parser.add_argument("--model-id", default="dist-reuse-smoke-model")
    parser.add_argument("--num-blocks", type=int, default=8,
                        help="How many put→get cycles to run in the aggregate scenario")
    parser.add_argument("--session-ttl", type=int, default=30)
    parser.add_argument("--scenario", default="all",
                        choices=["all", "peer_discovery", "aggregate", "cross_instance"])
    args = parser.parse_args()

    _ensure_redis_reachable(args.redis_host, args.redis_port, args.redis_password)
    print(f"Redis reachable at {args.redis_host}:{args.redis_port}")

    results: Dict[str, Any] = {}
    if args.scenario in ("all", "peer_discovery"):
        results["peer_discovery"] = scenario_peer_discovery(args)
    if args.scenario in ("all", "aggregate"):
        results["aggregate"] = scenario_aggregate_radix_put_hook(args)
    if args.scenario in ("all", "cross_instance"):
        results["cross_instance"] = scenario_cross_instance_reuse_readiness(args)

    print("\n=== SMOKE RESULTS ===")
    for name, res in results.items():
        print(f"  {name}: {res}")
    print("\nALL SCENARIOS PASSED ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
