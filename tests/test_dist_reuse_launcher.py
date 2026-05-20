"""§2.5 — Smoke test for start_dist_reuse_serving.sh.

Exercises the shell script's ``--dry-run`` mode: no vLLM/sglang boot,
but we verify the script accepts the documented flag set, emits the
expected ``mooncake_config.json`` shape based on topology, and prints
a summary.

This is deliberately smoke-level — a full multi-node boot requires
real GPU machines (§2.6).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "multi-nodes" / "start_dist_reuse_serving.sh"


# Skip everything when bash isn't available (Windows CI etc.).
pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(),
    reason="start_dist_reuse_serving.sh not present",
)


def _run(args, cwd=None, env=None, check_returncode=True):
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=cwd or str(REPO_ROOT),
        env=env or os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if check_returncode and proc.returncode != 0:
        raise AssertionError(
            f"script exited {proc.returncode}\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


class TestDryRun:
    def test_master_rank_0_emits_mooncake_config(self, tmp_path):
        # Copy script to tmp so we can inspect CFG_DIR artefacts without
        # polluting the repo.
        proc = _run([
            "--nnodes", "2",
            "--node-rank", "0",
            "--master-ip", "10.0.0.1",
            "--tp-size", "8",
            "--pp-size", "1",
            "--cp-size", "4",
            "--model", "/tmp/fake-model",
            "--redis-host", "10.0.0.1",
            "--redis-password", "x",
            "--rdma-device", "mlx5_0",
            "--instance-id", "unit-test-master",
            "--dry-run",
        ])
        assert "node_rank=0/2" in proc.stdout
        assert "instance_id        : unit-test-master" in proc.stdout

        # Check the generated mooncake config looks like valid JSON.
        cfg = (REPO_ROOT / "scripts" / "multi-nodes" / "gen"
               / "dist_reuse_node0" / "mooncake_config.json")
        assert cfg.exists(), f"expected mooncake config at {cfg}"
        data = json.loads(cfg.read_text())
        assert data["metadata_backend"] == "redis"
        assert data["device_name"] == "mlx5_0"
        assert data["engine_port"] == 12345    # base (+ node_rank 0)

    def test_cp_only_offmaster_still_emits_full_config(self):
        """CP-only cross-node instance, node_rank=1.

        Earlier revisions of this script tried to mirror the
        connector's role-decision in bash by emitting an *empty*
        sentinel mooncake_config.json for the CP-peer-only case so the
        Python connector could side-channel on file existence.  That
        coupling was reverted (see commit ``review-fixes(F1+F2+F3)``)
        because it duplicated launch-time logic that belongs in the
        connector.  The new contract is: shell *always* writes a valid
        mooncake config, the Python layer decides whether to
        instantiate a TransferEngine.  This test pins that contract.
        """
        proc = _run([
            "--nnodes", "2",
            "--node-rank", "1",
            "--master-ip", "10.0.0.1",
            "--tp-size", "4",          # TP fits on one node (≤ 8 gpus)
            "--pp-size", "1",
            "--cp-size", "2",          # CP = 2, cross-node
            "--model", "/tmp/fake-model",
            "--redis-host", "10.0.0.1",
            "--redis-password", "x",
            "--dry-run",
        ])
        assert "node_rank=1/2" in proc.stdout
        assert "is_multinode_cp    : true" in proc.stdout

        cfg = (REPO_ROOT / "scripts" / "multi-nodes" / "gen"
               / "dist_reuse_node1" / "mooncake_config.json")
        assert cfg.exists()
        # No more empty-sentinel hack — every node gets a valid config
        # and the connector decides what to do with it at runtime.
        assert cfg.stat().st_size > 0
        data = json.loads(cfg.read_text())
        assert data["metadata_backend"] == "redis"

    def test_tp_cross_node_offmaster_emits_full_config(self):
        proc = _run([
            "--nnodes", "2",
            "--node-rank", "1",
            "--master-ip", "10.0.0.1",
            "--tp-size", "16",         # > gpus_per_node default 8 → tp_node_count=2
            "--pp-size", "1",
            "--cp-size", "1",
            "--model", "/tmp/fake-model",
            "--redis-host", "10.0.0.1",
            "--redis-password", "x",
            "--dry-run",
        ])
        assert "is_multinode_tp    : true" in proc.stdout

        cfg = (REPO_ROOT / "scripts" / "multi-nodes" / "gen"
               / "dist_reuse_node1" / "mooncake_config.json")
        assert cfg.exists() and cfg.stat().st_size > 0
        data = json.loads(cfg.read_text())
        assert data["engine_port"] == 12346     # base + 1


class TestValidation:
    def test_missing_required_arg_exits_nonzero(self):
        proc = _run(
            ["--nnodes", "2", "--dry-run"],      # missing --node-rank etc.
            check_returncode=False,
        )
        assert proc.returncode != 0

    def test_nnodes_greater_than_two_rejected(self):
        """Current dist_reuse deployment constraint (§3.3) — fail loudly
        rather than silently proceed."""
        proc = _run([
            "--nnodes", "3",
            "--node-rank", "0",
            "--master-ip", "10.0.0.1",
            "--tp-size", "8", "--pp-size", "1", "--cp-size", "1",
            "--model", "/tmp/x", "--redis-host", "10.0.0.1",
            "--dry-run",
        ], check_returncode=False)
        assert proc.returncode != 0
        assert "supports <= 2 physical nodes" in proc.stdout or \
               "supports <= 2 physical nodes" in proc.stderr

    def test_bad_node_rank_rejected(self):
        proc = _run([
            "--nnodes", "2",
            "--node-rank", "2",              # out of range
            "--master-ip", "10.0.0.1",
            "--tp-size", "8", "--pp-size", "1", "--cp-size", "1",
            "--model", "/tmp/x", "--redis-host", "10.0.0.1",
            "--dry-run",
        ], check_returncode=False)
        assert proc.returncode != 0
