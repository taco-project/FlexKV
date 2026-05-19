"""§2.6 — Cross-node end-to-end test scaffolding.

Real dist_reuse e2e (≥ 2 GPU machines, real Mooncake RDMA path,
cross-instance reuse) cannot run on a single-box CI.  This module
provides a ``pytest`` scaffold so that once a multi-machine harness is
available, wiring the tests is a matter of filling in the fixtures.

All tests here are skipped by default via ``pytest.mark.multinode``
unless the ``FLEXKV_MULTINODE_TEST=1`` environment variable is set.
The intent is:

* CI default — tests are collected (so ``pytest --collect-only`` shows
  them) but skipped with a clear reason.
* Developer local multi-machine run — set the env var, point
  ``FLEXKV_MULTINODE_MASTER_ADDR`` / ``FLEXKV_MULTINODE_REMOTE_ADDR``
  at pre-booted instances, run pytest.

When we do light up the harness (docs/dist_reuse/
implementation_gap_2026-05-11.md §2.6), fill in:

1. ``launch_two_node_instance`` fixture: boots one Master + one Remote
   using ``scripts/multi-nodes/start_dist_reuse_serving.sh`` and
   collects their shutdown hooks.
2. Client-side orchestrator: sends a prompt to both instances, captures
   hit-rate counters, latency, and bytes transferred over Mooncake.
3. Assertion on cross-instance reuse: after instance A processes a
   common prefix, instance B's cold-start of the same prefix must
   show ``remote_hits > 0`` in ``kv_manager.stats()`` (or equivalent).

**Status (2026-05-18)**: the single-SD degenerate path was verified
end-to-end on a 2-machine RDMA harness with
``benchmark_dist_direct.py``.  That result is codified below as
``test_single_sd_degenerate_reuse_cache_ratio_100pct`` so CI can
regression-gate it when the harness env is available.  The
multi-SD (PP>1 / tp_node_count>1) implementation has landed in code
(Phase D-1/D-2/D-3 — see
``docs/dist_reuse/proposal_unify_with_graph_dispatch_2026-05-15.md``)
but the real-hardware e2e tests below remain ``xfail`` pending a
working multi-SD harness fixture (``s4_multi_sd_pp2.sh``).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


def _multinode_enabled() -> bool:
    return os.environ.get("FLEXKV_MULTINODE_TEST", "") == "1"


# Project-level marker; any test tagged with this is opt-in.
multinode = pytest.mark.multinode


pytestmark = pytest.mark.skipif(
    not _multinode_enabled(),
    reason="Set FLEXKV_MULTINODE_TEST=1 and provide the harness addresses "
           "(FLEXKV_MULTINODE_MASTER_ADDR / FLEXKV_MULTINODE_REMOTE_ADDR) "
           "to enable cross-node e2e tests. See docs/dist_reuse/"
           "implementation_gap_2026-05-11.md §2.6.",
)


# ---------------------------------------------------------------------------
# Fixtures (stubs; fill in when harness is ready)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def master_addr() -> str:
    addr = os.environ.get("FLEXKV_MULTINODE_MASTER_ADDR")
    if not addr:
        pytest.skip("FLEXKV_MULTINODE_MASTER_ADDR not set")
    return addr


@pytest.fixture(scope="module")
def remote_addr() -> str:
    addr = os.environ.get("FLEXKV_MULTINODE_REMOTE_ADDR")
    if not addr:
        pytest.skip("FLEXKV_MULTINODE_REMOTE_ADDR not set")
    return addr


@pytest.fixture(scope="module")
def two_instance_cluster(master_addr, remote_addr) -> Dict[str, Any]:
    """Two pre-booted FlexKV instances ready for cross-instance reuse.

    TODO(§2.6): replace this placeholder with real boot+teardown logic
    driven by ``scripts/multi-nodes/start_dist_reuse_serving.sh``.
    For now, tests assume the user booted them by hand and we just
    expose their addresses.
    """
    return {
        "master": master_addr,
        "remote": remote_addr,
    }


# ---------------------------------------------------------------------------
# Helpers — parse benchmark_dist_direct.py stdout
# ---------------------------------------------------------------------------
_CACHE_RATIO_RE = re.compile(
    r"cache_ratio:\s*(?P<pct>\d+(?:\.\d+)?)%"
)
_STATS_REUSE_RE = re.compile(
    r"\[STATS\]\[REUSE\].*?flexkv_global=(?P<global_hit>\d+).*?"
)


def _parse_benchmark_result(stdout: str) -> Dict[str, float]:
    """Extract cache_ratio and REUSE stats from benchmark_dist_direct.py
    stdout.

    Returns
    -------
    dict
        ``{"cache_ratio_pct": float, "flexkv_global_hits": int}``.
        Missing fields default to 0.
    """
    out: Dict[str, float] = {"cache_ratio_pct": 0.0, "flexkv_global_hits": 0}
    m = _CACHE_RATIO_RE.search(stdout)
    if m:
        out["cache_ratio_pct"] = float(m.group("pct"))
    m = _STATS_REUSE_RE.search(stdout)
    if m:
        out["flexkv_global_hits"] = int(m.group("global_hit"))
    return out


# ---------------------------------------------------------------------------
# Test bodies
# ---------------------------------------------------------------------------
@multinode
def test_single_sd_degenerate_reuse_cache_ratio_100pct(two_instance_cluster):
    """Single-SD degenerate dist_reuse e2e regression test.

    Captures the state reached on 2026-05-13:
    - Two-machine harness (146 ⇄ 129) with mooncake RDMA transfer.
    - PP=1, tp_node_count=1 → ``total_sd_count == 1``.
    - PUT on machine A, GET on machine B with same seed → 100% hit.

    The ``benchmark_dist_direct.py`` invocation shape is verbatim to
    what the manual harness uses — if this assertion flips (e.g. 0%
    after some code change), the single-SD path of §2.1 broke and a
    real bisect is needed.

    Environment preconditions:

    * ``FLEXKV_MULTINODE_BENCHMARK_CMD_PUT`` — full shell command that
      starts the PUT-side benchmark on machine A (Master).  Must block
      until "Press Enter to shutdown" appears in stdout.
    * ``FLEXKV_MULTINODE_BENCHMARK_CMD_GET`` — full shell command for
      the GET side on machine B.
    * Both commands must use the same ``--seed``,
      ``--sequence-length``, and ``--batch-size`` so the hash chain
      matches.

    If either env var is missing, the test is ``skip``ed (we don't
    xfail — a harness mismatch isn't a code regression).
    """
    cmd_put = os.environ.get("FLEXKV_MULTINODE_BENCHMARK_CMD_PUT")
    cmd_get = os.environ.get("FLEXKV_MULTINODE_BENCHMARK_CMD_GET")
    if not cmd_put or not cmd_get:
        pytest.skip(
            "Set FLEXKV_MULTINODE_BENCHMARK_CMD_PUT and "
            "FLEXKV_MULTINODE_BENCHMARK_CMD_GET to exercise this test."
        )

    # Fire PUT in the background.  The benchmark prints
    # "Press Enter to shutdown" when the prefix is idle in Redis and
    # ready to be consumed.
    put_proc = subprocess.Popen(
        shlex.split(cmd_put),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait until PUT is idle (max 60 s).
        deadline = _deadline(60)
        put_stdout_buf = []
        while True:
            if put_proc.poll() is not None:
                pytest.fail(
                    f"PUT side exited prematurely with rc={put_proc.returncode}: "
                    f"{''.join(put_stdout_buf)[:2000]}"
                )
            line = put_proc.stdout.readline() if put_proc.stdout else ""
            if line:
                put_stdout_buf.append(line)
                if "Press Enter" in line or "Data published" in line:
                    break
            if _past_deadline(deadline):
                pytest.fail(
                    "PUT side never reached idle state within 60s.\n"
                    f"stdout:\n{''.join(put_stdout_buf)[-2000:]}"
                )

        # Now run GET and capture full stdout.
        get = subprocess.run(
            shlex.split(cmd_get),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert get.returncode == 0, (
            f"GET side crashed (rc={get.returncode}): {get.stderr[-2000:]}"
        )

        stats = _parse_benchmark_result(get.stdout + "\n" + get.stderr)
        assert stats["cache_ratio_pct"] >= 99.9, (
            f"Cross-instance reuse regressed — "
            f"expected ~100% cache hit, got {stats['cache_ratio_pct']:.2f}%.\n"
            f"Last 2 KB of GET stdout:\n{get.stdout[-2000:]}"
        )
        assert stats["flexkv_global_hits"] > 0, (
            f"No flexkv_global hits registered — aggregate radix not "
            f"rebuilt on GET side.\nLast 2 KB of GET stdout:\n"
            f"{get.stdout[-2000:]}"
        )
    finally:
        # Tear down PUT side cleanly.
        try:
            put_proc.terminate()
            put_proc.wait(timeout=10)
        except Exception:
            put_proc.kill()


@multinode
def test_cross_instance_reuse_hit_rate(two_instance_cluster):
    """Multi-SD hit-rate validation (PP>1 or tp_node_count>1).

    Still ``xfail`` — the code-level implementation (Phase D-1/D-2/D-3
    of ``proposal_unify_with_graph_dispatch_2026-05-15.md``) has
    landed, but a working multi-SD harness fixture (PP=2 yaml +
    ``s4_multi_sd_pp2.sh`` 2-instance launcher) is still required
    to drive this test end-to-end.  Once that fixture is online the
    test body should mirror
    ``test_single_sd_degenerate_reuse_cache_ratio_100pct`` but with
    a multi-SD yaml (``pp_size: 2``) and additionally assert that
    peer-SD D2H clones produced ``CompletedOp(sd_key,
    contributing_node_id, success=True)`` events on the master's
    completion sink (once the metrics hook exposing them lands).
    """
    pytest.xfail(
        "Multi-SD e2e still blocked on the multi-SD harness fixture "
        "(s4_multi_sd_pp2.sh) — the D-1/D-2/D-3 code path is in "
        "place; see proposal_unify_with_graph_dispatch_2026-05-15.md "
        "§11."
    )


@multinode
def test_master_node_failure_triggers_peer_invalidation(two_instance_cluster):
    """Kill the Master; the Remote must stop accepting coord GETs and
    its peer-lost hook should fire on other instances."""
    pytest.xfail("harness not implemented yet — §2.6 TODO")


@multinode
def test_mooncake_transfer_sync_read_path(two_instance_cluster):
    """End-to-end validation of the Phase 1-C data path: a coord GET
    triggers a real Mooncake RDMA read that lands bytes into the
    requesting instance's CPU block pool.  Relies on real HWs.

    Note: single-SD Mooncake read path IS already covered by
    ``test_single_sd_degenerate_reuse_cache_ratio_100pct`` above —
    ``op_remote2h`` in ``_get_impl_global`` dispatches the Mooncake
    read.  This test remains for the multi-SD GET fan-out path:
    Master constructs a unified ``TransferOpGraph`` with one
    ``PEERH2H`` op per peer SD (target_node_ids stamping handled by
    ``GlobalCacheEngine._maybe_attach_multi_sd_peerh2h_ops``, Phase
    D-3), each peer SD's Remote executes its own clone, and the
    master receives one ``CompletedOp`` per SD before triggering the
    H2D.
    """
    pytest.xfail(
        "Multi-SD PEERH2H fan-out needs the same multi-SD harness "
        "fixture (s4_multi_sd_pp2.sh) as "
        "test_cross_instance_reuse_hit_rate."
    )


# ---------------------------------------------------------------------------
# Conftest-level registration — lets ``pytest -m multinode`` select these
# ---------------------------------------------------------------------------
def pytest_configure(config):  # pragma: no cover  (hook, no unit test)
    config.addinivalue_line(
        "markers",
        "multinode: requires ≥ 2 GPU machines and FLEXKV_MULTINODE_TEST=1",
    )


# ---------------------------------------------------------------------------
# Internal helpers for the stdout-polling loop
# ---------------------------------------------------------------------------
import time


def _deadline(seconds: float) -> float:
    return time.monotonic() + float(seconds)


def _past_deadline(deadline: float) -> bool:
    return time.monotonic() >= deadline
