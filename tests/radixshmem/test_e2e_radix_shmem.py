"""End-to-end test of FLEXKV_ENABLE_RADIXSHMEM=1 on one node: one or two DP scheduler
processes and one radix-server, on FlexKV's own process model (dp_size 1: the
KVTaskEngine and its TE subprocess in the engine process; dp_size 2: dp0 embeds
the KVServer whose KVTaskEngine and TE serve both DPs).

For every ``dp_size`` in the parametrization:

  * the test starts the operator's radix-server (index + SlotStore, the
    node's CPU KV pool) with nothing but a name and a byte budget; every
    KVManager hands it FlexKV's geometry and adopts the slot counts it plans,
    and so does the process that builds the KVTaskEngine. Which server to
    attach to is the ``server.name`` of a small YAML written per run
    (FLEXKV_RADIXSHMEM_CONFIG_PATH), which is how a deployment names it too.
  * Phase 1: every DP PUTs its own requests concurrently.
  * Phase 2 (dp_size > 1): dp0 PUTs a prefix that dp1 then finds with
    ``get_match`` -- the shared index is what the radixshmem path exists for.
  * Phase 3: every DP writes a rank-specific byte pattern into its GPU blocks,
    PUTs them (D2H into the SlotStore), zeroes the GPU blocks, GETs them back
    (H2D out of the SlotStore) and compares byte for byte. A transfer that
    reached another DP's GPU, or read the wrong SlotStore slot, shows up here
    as a mismatch rather than as an error.

Requires ``dp_size`` CUDA devices; skips otherwise. Run inside the container:

    PYTHONPATH=/path/to/FlexKV:/path/to/radixshmem/python \\
    LD_LIBRARY_PATH=$RADIXSHMEM_LIBS:$TORCH_LIB:$LD_LIBRARY_PATH \\
    python -m pytest tests/test_e2e_radix_shmem.py -s
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import shutil
import tempfile
import time

import numpy as np
import pytest
import torch

from radix_e2e_common import (
    TOKENS_PER_BLOCK,
    build_request,
    clear_blocks,
    get_blocks,
    mismatched_blocks,
    start_tp_client,
    stop_tp_client,
    start_radix_server,
    stop_radix_server,
    sweep_radix_files,
    wait_kv_manager_ready,
    write_pattern,
    write_radix_config,
)

NUM_GPU_BLOCKS = 256
NUM_CPU_BLOCKS = 4096
BLOCK_PER_REQUEST = 32
# A fixed prefix that dp0 writes and dp1 later looks up across the shared index.
SHARED_SEED = 0x5EED
SHARED_START_BLOCK = 128
# GPU blocks phase 3 owns, past the ranges phases 1 and 2 use.
ROUNDTRIP_START_BLOCK = 192


def _dp_proc(dp_client_id: int, dp_size: int, server_id: str, config_path: str,
             barrier, result_q) -> None:
    """Full lifecycle of one DP scheduler process."""
    # Before the first flexkv import: GLOBAL_CONFIG_FROM_ENV is read at import.
    # With dp_size > 1 the DPs are KVServer clients and must agree on
    # server_recv_port (and therefore on the gpu_register_port the TE listens on).
    recv_port = f"ipc:///tmp/flexkv_{server_id}"
    os.environ.update({
        "FLEXKV_ENABLE_RADIXSHMEM": "1",
        "FLEXKV_RADIXSHMEM_CONFIG_PATH": config_path,
        "FLEXKV_ENABLE_MPS": "0",
        "FLEXKV_SERVER_RECV_PORT": recv_port,
    })

    from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV, ModelConfig
    from flexkv.common.request import KVResponseStatus
    from flexkv.kvmanager import KVManager

    GLOBAL_CONFIG_FROM_ENV.enable_radixshmem = True
    GLOBAL_CONFIG_FROM_ENV.radixshmem_config_path = config_path
    GLOBAL_CONFIG_FROM_ENV.enable_mps = False
    GLOBAL_CONFIG_FROM_ENV.server_recv_port = recv_port

    model_config = ModelConfig(
        num_layers=2, num_kv_heads=4, head_size=128,
        dtype=torch.float16, tp_size=1, dp_size=dp_size,
    )
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK, enable_cpu=True, enable_ssd=False,
        num_cpu_blocks=NUM_CPU_BLOCKS,
    )

    tag = f"[dp{dp_client_id}]"
    report = {"dp": dp_client_id}
    kvm = None
    tp_proc = None
    try:
        kvm = KVManager(model_config, cache_config, dp_client_id=dp_client_id)
        kvm.start()
        # Each DP drives its own GPU (device id = dp id); no process pins
        # CUDA_VISIBLE_DEVICES, so the TE sees every device the DPs registered.
        tp_proc, gpu_tensors = start_tp_client(
            kvm, dp_client_id, dp_client_id, model_config, cache_config, NUM_GPU_BLOCKS)
        wait_kv_manager_ready(kvm, timeout=120)
        print(f"{tag} READY", flush=True)

        # --- Phase 1: private PUTs, concurrently on every DP. ---
        own = []
        for i in range(3):
            tok, slot, _blk = build_request(
                BLOCK_PER_REQUEST, i * BLOCK_PER_REQUEST, seed=1000 * dp_client_id + i)
            own.append(kvm.put_async(token_ids=tok, slot_mapping=slot))
        statuses = kvm.wait(own, timeout=60, completely=True)
        report["private_puts"] = (
            sum(1 for s in statuses.values() if s.status == KVResponseStatus.SUCCESS),
            len(own))
        print(f"{tag} private puts: {report['private_puts']}", flush=True)

        # --- Phase 2: dp0 PUTs a shared prefix; dp1 finds it in the shared index. ---
        report["shared_hit_blocks"] = -1
        if dp_size > 1:
            tok, slot, _blk = build_request(BLOCK_PER_REQUEST, SHARED_START_BLOCK,
                                            seed=SHARED_SEED)
            if dp_client_id == 0:
                task_id = kvm.put_async(token_ids=tok, slot_mapping=slot)
                statuses = kvm.wait([task_id], timeout=60, completely=True)
                report["shared_put_ok"] = all(
                    s.status == KVResponseStatus.SUCCESS for s in statuses.values())
                barrier.wait(120)           # release dp1 to look it up
            else:
                barrier.wait(120)           # until dp0 finished the shared put
                # Index visibility is asynchronous after the store: poll.
                hit = 0
                for _ in range(50):
                    _tid, mask = kvm.get_match(token_ids=tok)
                    hit = (int(np.count_nonzero(mask)) // TOKENS_PER_BLOCK
                           if mask is not None else 0)
                    if hit > 0:
                        break
                    time.sleep(0.2)
                report["shared_hit_blocks"] = hit
                print(f"{tag} cross-DP match hit_blocks={hit}", flush=True)

        # --- Phase 3: byte round trip through this DP's own GPU. ---
        tok, slot, blk = build_request(BLOCK_PER_REQUEST, ROUNDTRIP_START_BLOCK,
                                       seed=7000 + dp_client_id)
        # The DPs use the same block ids on different devices, so the phases are
        # kept in lockstep: a stray transfer then lands on a block under check.
        write_pattern(gpu_tensors, blk, writer=dp_client_id)
        barrier.wait(120)
        task_id = kvm.put_async(token_ids=tok, slot_mapping=slot)
        report["roundtrip_put_ok"] = all(
            s.status == KVResponseStatus.SUCCESS
            for s in kvm.wait([task_id], timeout=60, completely=True).values())
        barrier.wait(120)
        clear_blocks(gpu_tensors, blk)
        barrier.wait(120)
        report["roundtrip_hit_blocks"] = get_blocks(kvm, tok, slot)
        barrier.wait(120)
        report["roundtrip_mismatched"] = mismatched_blocks(gpu_tensors, blk, writer=dp_client_id)
        print(f"{tag} roundtrip: put_ok={report['roundtrip_put_ok']} "
              f"hit_blocks={report['roundtrip_hit_blocks']} "
              f"mismatched={len(report['roundtrip_mismatched'])}", flush=True)
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()
        print(f"{tag} FAILED\n{report['error']}", flush=True)
        with contextlib.suppress(Exception):
            barrier.abort()   # unblock the other DPs' waits
    finally:
        stop_tp_client(tp_proc)
        if kvm is not None:
            try:
                kvm.shutdown()
            except Exception as exc:  # noqa: BLE001
                report.setdefault("error", f"shutdown: {exc}")
        result_q.put(report)


def _run(dp_size: int) -> dict:
    server_id = f"e2e{dp_size}dp_{os.getpid()}"
    workdir = tempfile.mkdtemp(prefix="flexkv_radix_e2e_")
    name = f"/{server_id}"
    config_path = write_radix_config(workdir, {"server": {"name": name, "ready_timeout_s": 300}})
    # The operator's server: a byte budget that holds NUM_CPU_BLOCKS blocks of
    # this test's model (the DP processes bring the geometry and adopt the count).
    from flexkv.common.config import CacheConfig, ModelConfig
    from flexkv.server.shm_radix_bootstrap import cpu_block_bytes
    block_bytes = cpu_block_bytes(
        ModelConfig(num_layers=2, num_kv_heads=4, head_size=128, dtype=torch.float16,
                    tp_size=1, dp_size=dp_size),
        CacheConfig(tokens_per_block=TOKENS_PER_BLOCK, enable_cpu=True, enable_ssd=False,
                    num_cpu_blocks=NUM_CPU_BLOCKS))
    sweep_radix_files(server_id)
    server = start_radix_server(name, NUM_CPU_BLOCKS * block_bytes,
                                log_path=os.path.join(workdir, "radix-server.log"))
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(dp_size)
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=_dp_proc,
                    args=(dp, dp_size, server_id, config_path, barrier, result_q),
                    daemon=False)
        for dp in range(dp_size)
    ]
    reports = {}
    try:
        for proc in procs:
            proc.start()
        deadline = time.monotonic() + 420
        while len(reports) < dp_size and time.monotonic() < deadline:
            try:
                report = result_q.get(timeout=5)
                reports[report["dp"]] = report
            except Exception:
                if not any(proc.is_alive() for proc in procs):
                    break
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
        stop_radix_server(server)
        sweep_radix_files(server_id)
        shutil.rmtree(workdir, ignore_errors=True)
    return reports


@pytest.mark.e2e
@pytest.mark.parametrize("dp_size", [1, 2])
def test_radix_shmem_put_get_roundtrip(dp_size):
    pytest.importorskip("shmradix")
    if not torch.cuda.is_available() or torch.cuda.device_count() < dp_size:
        pytest.skip(f"needs {dp_size} CUDA device(s)")

    reports = _run(dp_size)

    assert len(reports) == dp_size, f"only {len(reports)}/{dp_size} DP processes reported"
    errors = {dp: r["error"] for dp, r in reports.items() if "error" in r}
    assert not errors, errors
    for dp, report in sorted(reports.items()):
        ok, total = report["private_puts"]
        assert ok == total, f"dp{dp}: private puts {ok}/{total}"
        assert report["roundtrip_put_ok"], f"dp{dp}: round-trip put did not complete"
        assert report["roundtrip_hit_blocks"] == BLOCK_PER_REQUEST, \
            f"dp{dp}: round-trip GET matched {report['roundtrip_hit_blocks']}/{BLOCK_PER_REQUEST}"
        assert report["roundtrip_mismatched"] == [], \
            f"dp{dp}: {len(report['roundtrip_mismatched'])} block(s) hold KV that is not its " \
            f"own -- its transfers reached another DP's GPU or the wrong SlotStore slot"
    if dp_size > 1:
        assert reports[0].get("shared_put_ok"), "dp0's shared-prefix put did not complete"
        assert reports[1]["shared_hit_blocks"] == BLOCK_PER_REQUEST, \
            f"cross-DP prefix sharing: dp1 matched {reports[1]['shared_hit_blocks']}/" \
            f"{BLOCK_PER_REQUEST} blocks of dp0's prefix"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
