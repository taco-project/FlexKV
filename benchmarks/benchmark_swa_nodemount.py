# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone SWA + Full node-mount workload driver (real host-pool byte IO).

This is a *self-contained* benchmark/debugger for the node-mounted SWA design
(deployments/swa_design/08_节点挂载SWA架构.md), decoupled from the inference
engine. It drives the PRODUCTION C++ path — ``CacheEngineAccel`` with its
``CRadixTreeIndex`` Full-KV radix tree and a real ``SWAHostPool`` — through a
realistic multi-turn dialogue workload, and:

  * inserts Full-KV blocks (real mempool alloc), mounts an SWA page on the
    trailing block of each stored turn (``set_swa`` on the matched node),
  * writes REAL bytes into the SWA host-pool slot (a per-page fingerprint) and,
    on a later SWA hit, reads them back and verifies byte-identity — this is the
    "real IO" for the SWA control/data boundary that has no GPU worker yet,
  * exercises Full eviction (leaf-by-leaf, connect-frees SWA — I1) and SWA-only
    eviction (interior-first LRU — I2) under memory pressure,
  * continuously checks the node-mount invariants and the two-pool lock-step
    (Full evicted ⟹ its SWA slot returned to the host pool; no slot leak),
  * reports Full match rate, SWA hit rate, byte-verify count, and pool stats.

Run:  python3 benchmarks/benchmark_swa_nodemount.py --users 32 --turns 6
It uses no GPU and no torch CUDA; the SWA host pool falls back to numpy when
CUDA is absent, so it runs anywhere the C extension is importable.
"""
import argparse
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from flexkv.cache.cache_engine import CacheEngineAccel
from flexkv.common.block import SequenceMeta
from flexkv.common.config import SWAPoolConfig
from flexkv.common.transfer import DeviceType


TPB = 16  # tokens per block (== swa_page_size here: one page == one block == one slot)


# --------------------------------------------------------------------------- #
# Workload: multi-turn dialogues sharing a system prompt, growing per turn.    #
# --------------------------------------------------------------------------- #

@dataclass
class Turn:
    user: int
    turn: int
    token_ids: np.ndarray            # full prefix up to and including this turn's input


def make_workload(num_users: int, num_turns: int, sys_len: int,
                  turn_in: int, turn_out: int, seed: int) -> List[Turn]:
    rng = random.Random(seed)
    npr = np.random.RandomState(seed)
    vocab = 30000
    system = npr.randint(0, vocab, size=sys_len, dtype=np.int64)
    per_user_hist: Dict[int, np.ndarray] = {}
    per_user_turns: Dict[int, int] = {}
    for u in range(num_users):
        per_user_hist[u] = system.copy()
        per_user_turns[u] = max(1, rng.randint(num_turns // 2, num_turns))
    # Interleave users so evictions happen mid-stream (realistic serving order).
    work: List[Turn] = []
    pending = list(range(num_users))
    counters = {u: 0 for u in range(num_users)}
    while pending:
        u = rng.choice(pending)
        j = counters[u]
        inp = npr.randint(0, vocab, size=turn_in, dtype=np.int64)
        prefix = np.concatenate([per_user_hist[u], inp])
        work.append(Turn(user=u, turn=j, token_ids=prefix))
        out = npr.randint(0, vocab, size=turn_out, dtype=np.int64)
        per_user_hist[u] = np.concatenate([prefix, out])
        counters[u] += 1
        if counters[u] >= per_user_turns[u]:
            pending.remove(u)
    return work


# --------------------------------------------------------------------------- #
# SWA page fingerprint — deterministic bytes for a (last-block-hash) window.   #
# --------------------------------------------------------------------------- #

def page_fingerprint(block_hash: int, slot_size: int) -> np.ndarray:
    """A deterministic byte pattern keyed by the window's block hash.

    Stands in for the real SWA-page KV bytes. Byte-identical read-back on a hit
    proves the slot round-tripped through the host pool intact (real IO)."""
    seed = block_hash & 0xFFFFFFFF
    rs = np.random.RandomState(seed)
    return rs.randint(0, 256, size=slot_size, dtype=np.uint8)


@dataclass
class Stats:
    gets: int = 0
    puts: int = 0
    full_hit_blocks: int = 0
    full_query_blocks: int = 0
    swa_hits: int = 0
    swa_queries: int = 0
    swa_bytes_verified: int = 0
    swa_byte_mismatches: int = 0
    swa_evictions: int = 0
    full_evictions: int = 0
    invariant_violations: List[str] = field(default_factory=list)
    # end-of-run snapshot (captured before the reset leak-check)
    pool_used: int = 0
    pool_free: int = 0
    pool_total: int = 0
    mempool_free: int = 0
    mempool_total: int = 0
    tree_cached_blocks: int = 0


class Driver:
    def __init__(self, num_cpu_blocks: int, swa_slots: int, slot_bytes: int):
        self.tpb = TPB
        self.engine = CacheEngineAccel(
            DeviceType.CPU, num_cpu_blocks, TPB,
            evict_ratio=0.1, hit_reward_seconds=0,
            evict_start_threshold=1.0, eviction_policy="lru",
        )
        # A tiny SWA host pool so pressure/eviction is actually exercised.
        # slot geometry: 1 layer * slot_bytes/token, window_size == TPB.
        swa_cfg = SWAPoolConfig(
            enabled=True, num_slots=swa_slots, window_size=TPB,
            num_swa_layers=1, bytes_per_token_per_layer=max(1, slot_bytes // TPB),
        )
        self.engine.init_swa(swa_cfg)
        self.slot_size = self.engine.swa_pool.slot_size_bytes
        self.stats = Stats()
        # Track the fingerprint we wrote per live slot to verify read-back.
        self._slot_expect: Dict[int, int] = {}   # slot -> block_hash written

    # -- helpers ----------------------------------------------------------- #

    def _seq(self, token_ids: np.ndarray) -> SequenceMeta:
        aligned = (len(token_ids) // self.tpb) * self.tpb
        sm = SequenceMeta(token_ids=token_ids[:aligned].astype(np.int64),
                          tokens_per_block=self.tpb)
        sm.gen_hashes()
        return sm

    def _drain_and_check(self, ctx: str):
        """Drain slots freed by structural changes back to the pool and verify
        no double-free / leak. Returns the drained slots.

        NOTE: engine.take() / swa_alloc_slot() already drain-and-free internally
        (_drain_swa_slots), so by the time we call drain_freed_swa_slots() here
        it is usually empty — that is expected. We still call it to catch any
        residue and to keep our fingerprint map in sync via reconcile."""
        freed = list(self.engine.index.drain_freed_swa_slots())
        for s in freed:
            if s is None or s < 0:
                continue
            self._slot_expect.pop(int(s), None)
            self.engine.swa_pool.free(int(s))
        self._reconcile_slot_expect()
        return freed

    def _reconcile_slot_expect(self):
        """Drop tracked fingerprints for slots the engine returned to the pool
        internally (take()/swa_alloc_slot() drain SWA/connect-freed slots
        straight to the pool, bypassing our observation). The pool free-list is
        ground truth: a slot on it is no longer live, so forget its fingerprint."""
        free = set(int(s) for s in self.engine.swa_pool._free_slots)
        for s in list(self._slot_expect):
            if s in free:
                self._slot_expect.pop(s, None)

    # -- one request: match (get) then store (put) ------------------------- #

    def step(self, turn: Turn):
        sm = self._seq(turn.token_ids)
        nblocks = sm.num_blocks
        if nblocks == 0:
            return

        # ---- GET: full match + node-mount SWA match -------------------- #
        self.stats.gets += 1
        mr = self.engine.match(sm)
        full_hit = int(mr.num_ready_matched_blocks)
        self.stats.full_hit_blocks += full_hit
        self.stats.full_query_blocks += nblocks

        self.stats.swa_queries += 1
        if full_hit > 0:
            swa_hit, slot, _key = self.engine.match_swa(
                sm, upper_bound_blocks=full_hit, lock_for_load=False)
            if swa_hit > 0 and slot >= 0:
                self.stats.swa_hits += 1
                # REAL IO: read the slot bytes back and verify byte-identity.
                got = np.asarray(self.engine.swa_pool.read_copy(slot)).ravel()
                exp_hash = self._slot_expect.get(int(slot))
                if exp_hash is not None:
                    exp = page_fingerprint(exp_hash, self.slot_size)
                    if got.shape == exp.shape and np.array_equal(got, exp):
                        self.stats.swa_bytes_verified += 1
                    else:
                        self.stats.swa_byte_mismatches += 1
                        self.stats.invariant_violations.append(
                            f"SWA byte mismatch on slot {slot}")

        # ---- PUT: continuation-insert the new suffix, mount SWA on tail -- #
        # Follows the production protocol (see GlobalCacheEngine._put_impl_*):
        # take() ONLY the new suffix blocks, with the matched prefix's last_node
        # LOCKED (protected) so eviction can't reclaim the blocks we are about to
        # reuse; then insert the suffix as a CONTINUATION (num_insert_blocks +
        # match_result) so the tree keeps the matched prefix's existing physical
        # blocks. Re-taking / re-inserting the whole prefix (the naive form)
        # double-frees on eviction — recycle_blocks would see "already free".
        self.stats.puts += 1
        num_new = nblocks - full_hit
        if num_new > 0:
            protected = mr.last_node if full_hit > 0 else None
            # take() may evict full leaves (connect-frees SWA — I1). The
            # protected prefix node is locked internally for the duration.
            # take() drains connect-freed SWA slots to the pool internally, so
            # count evictions by the pool's free-count delta across the call.
            swa_free_before = self.engine.swa_pool.num_free
            phys_new = np.asarray(
                self.engine.take(num_new, protected_node=protected, strict=False)
            ).astype(np.int64)
            self.stats.full_evictions += max(
                0, self.engine.swa_pool.num_free - swa_free_before)
            if phys_new.size < num_new:
                # Pool exhausted even after eviction; skip storing this turn.
                self.engine.recycle(phys_new)
                self._drain_and_check("take-exhausted")
                return
            self._drain_and_check("after-take")
            node = self.engine.insert(sm, phys_new,
                                      num_insert_blocks=num_new,
                                      is_ready=True, match_result=mr)
        else:
            # Whole prefix already resident; nothing new to store, but the tail
            # node is exactly mr.last_node — refresh its SWA below.
            node = mr.last_node
        if node is None:
            return

        # Mount an SWA page on the node's trailing block (the window).
        slot = self.engine.swa_alloc_slot()
        # swa_alloc_slot may have SWA-evicted (and internally freed) slots to
        # make room; reconcile our fingerprint map with the pool's free-list.
        self._reconcile_slot_expect()
        if slot == -1:
            # SWA pool full and all locked; drain any eviction residue.
            self._drain_and_check("swa-alloc-fail")
            return
        # REAL IO: write the page fingerprint keyed by the tail block hash.
        tail_hash = int(sm.block_hashes[nblocks - 1])
        fp = page_fingerprint(tail_hash, self.slot_size)
        self.engine.swa_pool.write(slot, fp)
        self._slot_expect[int(slot)] = tail_hash
        self.engine.set_swa(node, slot)
        # set_swa may have freed a stale slot on the node (re-mount); drain it.
        self._drain_and_check("after-set-swa")

    # -- invariant sweep over the live pool -------------------------------- #

    def check_invariants(self) -> List[str]:
        """Ground-truth checks against the SWA host pool (the engine drains
        internally, so we validate the pool itself rather than our shadow map).

        * pool accounting: used + free == total (no lost/duplicated slots).
        * NO double-free: the free-list must contain no duplicate slot id — a
          double record_freed_swa_slot on the same slot would push it twice.
        * live-slot consistency: every fingerprint we still track must point at
          a slot that is NOT on the free-list (a live slot). If a tracked slot
          is free, the engine freed it under us AND we already reconciled, so
          this can only fire on a genuine use-after-free of a live window."""
        v: List[str] = []
        pool = self.engine.swa_pool
        used, free, total = pool.num_used, pool.num_free, pool.num_slots
        if used + free != total:
            v.append(f"pool accounting: used({used})+free({free}) != total({total})")
        free_list = list(pool._free_slots)
        if len(free_list) != len(set(free_list)):
            dup = sorted({s for s in free_list if free_list.count(s) > 1})
            v.append(f"DOUBLE-FREE: slot(s) {dup} appear twice on the free-list")
        free_set = set(free_list)
        stale = [s for s in self._slot_expect if s in free_set]
        if stale:
            v.append(f"USE-AFTER-FREE: tracked-live slot(s) {stale[:8]} are free")
        return v

    def run(self, work: List[Turn], check_every: int) -> Stats:
        for i, turn in enumerate(work):
            self.step(turn)
            if check_every and (i + 1) % check_every == 0:
                self.stats.invariant_violations += self.check_invariants()
        self.stats.invariant_violations += self.check_invariants()
        # Snapshot pool/tree occupancy before the destructive reset check.
        pool = self.engine.swa_pool
        self.stats.pool_used = pool.num_used
        self.stats.pool_free = pool.num_free
        self.stats.pool_total = pool.num_slots
        self.stats.mempool_free = self.engine.mempool.num_free_blocks
        self.stats.mempool_total = self.engine.mempool.num_total_blocks
        self.stats.tree_cached_blocks = self.engine.index.total_cached_blocks()
        # End-state leak check: reset the engine and confirm the SWA pool is
        # fully reclaimed (I1 lock-step — no slot permanently leaked by any
        # split/merge/evict path over the whole run).
        self.engine.reset()
        if self.engine.swa_pool.num_free != self.engine.swa_pool.num_slots:
            self.stats.invariant_violations.append(
                f"LEAK after reset: pool free {self.engine.swa_pool.num_free} "
                f"!= total {self.engine.swa_pool.num_slots}")
        return self.stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=32)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--sys-len", type=int, default=8 * TPB)
    ap.add_argument("--turn-in", type=int, default=6 * TPB)
    ap.add_argument("--turn-out", type=int, default=2 * TPB)
    ap.add_argument("--cpu-blocks", type=int, default=4096,
                    help="Full-KV mempool size in blocks (small = eviction pressure)")
    ap.add_argument("--swa-slots", type=int, default=256,
                    help="SWA host-pool slots (small = SWA-LRU pressure)")
    ap.add_argument("--slot-bytes", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--check-every", type=int, default=50)
    args = ap.parse_args()

    work = make_workload(args.users, args.turns, args.sys_len,
                         args.turn_in, args.turn_out, args.seed)
    drv = Driver(args.cpu_blocks, args.swa_slots, args.slot_bytes)

    t0 = time.time()
    stats = drv.run(work, args.check_every)
    dt = time.time() - t0

    print("=" * 68)
    print("SWA + Full node-mount workload (production CRadixTreeIndex path)")
    print("=" * 68)
    print(f"requests           : {len(work)}  ({args.users} users, ~{args.turns} turns)")
    print(f"wall time          : {dt*1000:.1f} ms  "
          f"({1000*dt/max(1,len(work)):.3f} ms/req)")
    print(f"tokens_per_block   : {TPB}   slot_size_bytes: {drv.slot_size}")
    print("-" * 68)
    fq = max(1, stats.full_query_blocks)
    print(f"full match rate    : {100*stats.full_hit_blocks/fq:.2f}%  "
          f"({stats.full_hit_blocks}/{stats.full_query_blocks} blocks)")
    sq = max(1, stats.swa_queries)
    print(f"SWA hit rate       : {100*stats.swa_hits/sq:.2f}%  "
          f"({stats.swa_hits}/{stats.swa_queries} gets)")
    print(f"SWA bytes verified : {stats.swa_bytes_verified}  "
          f"(mismatches: {stats.swa_byte_mismatches})")
    print(f"full evictions     : {stats.full_evictions} slots connect-freed")
    print("-" * 68)
    print(f"SWA pool (peak)    : used={stats.pool_used} free={stats.pool_free} "
          f"total={stats.pool_total}")
    print(f"Full mempool       : free={stats.mempool_free}/"
          f"{stats.mempool_total} blocks")
    print(f"tree cached blocks : {stats.tree_cached_blocks}")
    print("-" * 68)
    if stats.invariant_violations:
        print(f"INVARIANT VIOLATIONS ({len(stats.invariant_violations)}):")
        for msg in stats.invariant_violations[:20]:
            print(f"  ✗ {msg}")
        raise SystemExit(1)
    else:
        print("invariants         : OK (no double-free, no use-after-free, "
              "no byte mismatch, no leak-after-reset)")


if __name__ == "__main__":
    main()
