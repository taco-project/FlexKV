"""End-to-end DATA check for the radixshmem peer path: prefetch pulls a peer's
blocks, GET serves them from the local pool, the GPU holds the peer's bytes.

Two full FlexKV nodes on one host (two processes, two GPUs), one radixshmem
cluster:

  * each node's KVManager launches its own radix-server (index + SlotStore =
    the node's CPU pool + RDMA transfer engine) from one shared YAML
    (FLEXKV_RADIXSHMEM_CONFIG_PATH: cluster_id, expected_min_nodes=2, registry,
    RDMA devices), told apart by the per-node FLEXKV_RADIX_NODE_NAME override
    as co-located nodes are; the two servers rendezvous in one etcd namespace,
    get dense cluster ranks and an RHT to route by;
  * node 0 PUTs a window of GPU blocks holding a per-block pattern;
  * node 1 calls ``KVManager.prefetch_async`` for the same tokens: the index walk
    finds the prefix on node 0 over RDMA, node 1's radix-server RDMA-reads the
    bytes into node 1's SlotStore and the blocks are published in node 1's tree;
    ``try_wait`` reports the pulled range;
  * node 1 then does the ordinary local ``get_match`` + ``launch`` + ``wait`` and the
    GPU blocks are compared byte for byte with what node 0 wrote.

A second window checks the extension case: node 1 already holds the first
LOCAL_HEAD_BLOCKS of it (its own bytes), the prefetch pulls only the tail, and
the GET serves head and tail from the right writers.

Requires >=2 CUDA devices, an ACTIVE RDMA port, a shmradix built with RDMA +
etcd + mooncake, and an etcd (FLEXKV_TEST_RADIX_REGISTRY, or ``etcd`` on PATH
for a private one); skips otherwise. Run inside the container:

    PYTHONPATH=/path/to/FlexKV:/path/to/radixshmem/python \\
    LD_LIBRARY_PATH=$RADIXSHMEM_LIBS:$TORCH_LIB:$LD_LIBRARY_PATH \\
    python -m pytest tests/test_e2e_radix_prefetch_p2p.py -s
"""
from __future__ import annotations

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
    active_rdma_devices,
    build_request,
    clear_blocks,
    match_and_load,
    mismatched_blocks,
    put_prefix,
    start_private_etcd,
    start_tp_client,
    stop_private_etcd,
    stop_tp_client,
    sweep_radix_files,
    wait_kv_manager_ready,
    write_pattern,
    write_radix_config,
)

WORLD_SIZE = 2
NUM_GPU_BLOCKS = 128
NUM_CPU_BLOCKS = 256
NUM_REQUEST_BLOCKS = 32
LOCAL_HEAD_BLOCKS = 10
FIRST_BLOCK = 8
SECOND_FIRST_BLOCK = 64
SEED_A = 0x5EED
SEED_B = 0xBEEF


def _node_name(rank: int) -> str:
    return f"r{rank}"


def _prefetch_until(kvm, token_ids, want_pulled_blocks: int, timeout: float = 60.0):
    """prefetch_async + try_wait until the pulled range reaches the expectation
    (the peer's RHT publication is asynchronous, early rounds may miss).
    Returns (pulled_blocks, rounds)."""
    from flexkv.common.request import KVResponseStatus
    deadline = time.monotonic() + timeout
    rounds = 0
    pulled = 0
    while time.monotonic() < deadline:
        rounds += 1
        task_id = kvm.prefetch_async(token_ids=token_ids)
        response = None
        while time.monotonic() < deadline:
            done = kvm.try_wait([task_id])
            if task_id in done and done[task_id].status != KVResponseStatus.TIMEOUT:
                response = done[task_id]
                break
            time.sleep(0.02)
        if response is None:
            break
        mask = response.return_mask
        pulled = int(np.count_nonzero(mask)) // TOKENS_PER_BLOCK if mask is not None else 0
        if pulled >= want_pulled_blocks:
            break
        time.sleep(0.5)
    return pulled, rounds


def _node_proc(rank, gpu_id, cluster_id, config_path,
               reader_ready, written, read_done, result_q):
    """One FlexKV node: rank 0 writes the windows, rank 1 prefetches and reads."""
    # Before any CUDA context exists: each node drives a different device while
    # addressing it as device 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    node_name = _node_name(rank)
    # FlexKV's own IPC names are per node too (its radix regions get the
    # node name appended through the same override).
    recv_port = f"ipc:///tmp/flexkv_{cluster_id}_{node_name}"
    os.environ.update({
        "FLEXKV_ENABLE_RADIXSHMEM": "1",
        "FLEXKV_RADIXSHMEM_CONFIG_PATH": config_path,
        # The two per-node overrides of the global file: etcd keys membership
        # by node identity, which defaults to the bind IP the co-located nodes
        # share -- so name each node and give the loopback address explicitly.
        "FLEXKV_RADIX_NODE_NAME": node_name,
        "FLEXKV_RADIX_RPC_ADDRESS": "127.0.0.1",
        "FLEXKV_ENABLE_MPS": "0",
        "FLEXKV_SERVER_RECV_PORT": recv_port,
    })

    from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV, ModelConfig
    from flexkv.kvmanager import KVManager

    # Built from env at import time; set the fields that matter explicitly in
    # case a parent import happened earlier in this process.
    GLOBAL_CONFIG_FROM_ENV.enable_radixshmem = True
    GLOBAL_CONFIG_FROM_ENV.radixshmem_config_path = config_path
    GLOBAL_CONFIG_FROM_ENV.radix_node_name = node_name
    GLOBAL_CONFIG_FROM_ENV.radix_rpc_address = "127.0.0.1"
    GLOBAL_CONFIG_FROM_ENV.enable_mps = False
    GLOBAL_CONFIG_FROM_ENV.server_recv_port = recv_port

    tag = f"[node r{rank}]"
    model_config = ModelConfig(
        num_layers=2, num_kv_heads=4, head_size=128,
        dtype=torch.float16, tp_size=1, dp_size=1,
    )
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        enable_cpu=True, enable_ssd=False, enable_remote=False,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        # Peer reuse follows the radixshmem YAML (expected_min_nodes=2 below),
        # not enable_p2p_cpu.
    )

    report = {"rank": rank}
    kvm = None
    tp_proc = None
    try:
        kvm = KVManager(model_config, cache_config, dp_client_id=0)
        kvm.start()
        tp_proc, gpu_tensors = start_tp_client(kvm, 0, 0, model_config, cache_config,
                                               NUM_GPU_BLOCKS)
        wait_kv_manager_ready(kvm, timeout=180)
        report["node_id"] = int(cache_config.distributed_node_id)
        print(f"{tag} READY as cluster rank {report['node_id']}", flush=True)

        tokens_a, slots_a, blocks_a = build_request(NUM_REQUEST_BLOCKS, FIRST_BLOCK, SEED_A)
        tokens_b, slots_b, blocks_b = build_request(NUM_REQUEST_BLOCKS, SECOND_FIRST_BLOCK,
                                                    SEED_B)

        if rank == 1:
            # Window B: this node owns the head before node 0 publishes anything.
            write_pattern(gpu_tensors, blocks_b[:LOCAL_HEAD_BLOCKS], writer=1)
            report["head_put_ok"] = put_prefix(kvm, tokens_b, slots_b, LOCAL_HEAD_BLOCKS)
            print(f"{tag} head put ok={report['head_put_ok']}", flush=True)
            reader_ready.set()
            if not written.wait(240):
                raise TimeoutError("writer did not publish in 240s")

            # 1) Window A, nothing local: the prefetch pulls all of it off node 0,
            #    then the GET serves it from this node's pool.
            pulled, rounds = _prefetch_until(kvm, tokens_a, NUM_REQUEST_BLOCKS)
            report["a_pulled_blocks"], report["a_prefetch_rounds"] = pulled, rounds
            clear_blocks(gpu_tensors, blocks_a)
            report["a_hit_blocks"] = match_and_load(kvm, tokens_a, slots_a)
            report["a_mismatched"] = mismatched_blocks(gpu_tensors, blocks_a, writer=0)
            print(f"{tag} window A: pulled={pulled} rounds={rounds} "
                  f"hit={report['a_hit_blocks']} mismatched={len(report['a_mismatched'])}",
                  flush=True)

            # 2) Window B, local head: the prefetch pulls only the tail.
            pulled, rounds = _prefetch_until(kvm, tokens_b,
                                             NUM_REQUEST_BLOCKS - LOCAL_HEAD_BLOCKS)
            report["b_pulled_blocks"], report["b_prefetch_rounds"] = pulled, rounds
            clear_blocks(gpu_tensors, blocks_b)
            report["b_hit_blocks"] = match_and_load(kvm, tokens_b, slots_b)
            report["b_head_mismatched"] = mismatched_blocks(
                gpu_tensors, blocks_b[:LOCAL_HEAD_BLOCKS], writer=1)
            report["b_tail_mismatched"] = mismatched_blocks(
                gpu_tensors, blocks_b[LOCAL_HEAD_BLOCKS:], writer=0)
            print(f"{tag} window B: pulled={pulled} rounds={rounds} "
                  f"hit={report['b_hit_blocks']} "
                  f"head_mismatched={len(report['b_head_mismatched'])} "
                  f"tail_mismatched={len(report['b_tail_mismatched'])}", flush=True)
            read_done.set()
        else:
            if not reader_ready.wait(240):
                raise TimeoutError("reader did not lay down its head in 240s")
            write_pattern(gpu_tensors, blocks_a, writer=0)
            ok = put_prefix(kvm, tokens_a, slots_a, NUM_REQUEST_BLOCKS)
            write_pattern(gpu_tensors, blocks_b, writer=0)
            report["put_ok"] = put_prefix(kvm, tokens_b, slots_b, NUM_REQUEST_BLOCKS) and ok
            print(f"{tag} put ok={report['put_ok']}", flush=True)
            written.set()
            # Stay up: the reader's radix-server reads THIS node's SlotStore.
            if not read_done.wait(300):
                raise TimeoutError("reader did not finish in 300s")
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()
        print(f"{tag} FAILED\n{report['error']}", flush=True)
        reader_ready.set()
        written.set()
        read_done.set()
    finally:
        stop_tp_client(tp_proc)
        if kvm is not None:
            try:
                kvm.shutdown()
            except Exception as exc:  # noqa: BLE001
                report.setdefault("error", f"shutdown: {exc}")
        result_q.put(report)


def _run(registry: str, rdma_dev: str) -> dict:
    # One etcd namespace (and shm prefix) per run keeps concurrent runs apart.
    cluster_id = f"p2p{os.getpid()}"
    workdir = tempfile.mkdtemp(prefix="flexkv_radix_p2p_")
    config_path = write_radix_config(workdir, {
        "cluster": {
            "cluster_id": cluster_id,
            "expected_min_nodes": WORLD_SIZE,
            "registry": registry,
            "index_dev": rdma_dev,
            "rht_slots_per_bucket": 4,
        },
        "data": {"transfer_devices": [rdma_dev], "prefault": False},
    })
    ctx = mp.get_context("spawn")
    reader_ready, written, read_done = ctx.Event(), ctx.Event(), ctx.Event()
    result_q = ctx.Queue()
    procs = []
    reports = {}
    try:
        for rank in range(WORLD_SIZE):
            proc = ctx.Process(
                target=_node_proc,
                args=(rank, rank, cluster_id, config_path,
                      reader_ready, written, read_done, result_q),
                daemon=False,
            )
            proc.start()
            procs.append(proc)
        deadline = time.monotonic() + 600
        while len(reports) < WORLD_SIZE and time.monotonic() < deadline:
            try:
                report = result_q.get(timeout=5)
                reports[report["rank"]] = report
            except Exception:
                if not any(proc.is_alive() for proc in procs):
                    break
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
        sweep_radix_files(cluster_id)
        shutil.rmtree(workdir, ignore_errors=True)
    return reports


@pytest.fixture
def cluster():
    """(etcd registry, rdma device), skipping when the prerequisites are absent;
    starts a private etcd when none is configured."""
    pytest.importorskip("shmradix")
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"needs {WORLD_SIZE} CUDA devices")
    devices = active_rdma_devices()
    if not devices:
        pytest.skip("no ACTIVE RDMA port (checked /sys/class/infiniband/*)")
    registry = os.getenv("FLEXKV_TEST_RADIX_REGISTRY", "")
    proc = workdir = None
    if not registry:
        proc, workdir, registry = start_private_etcd("flexkv_p2p_etcd_")
        if not registry:
            pytest.skip("set FLEXKV_TEST_RADIX_REGISTRY or put etcd on PATH")
    try:
        yield registry, devices[0]
    finally:
        stop_private_etcd(proc, workdir)


@pytest.mark.e2e
def test_prefetch_pulls_peer_blocks_over_rdma(cluster):
    registry, rdma_dev = cluster
    reports = _run(registry, rdma_dev)

    assert len(reports) == WORLD_SIZE, f"only {len(reports)}/{WORLD_SIZE} nodes reported"
    errors = {rank: r["error"] for rank, r in reports.items() if "error" in r}
    assert not errors, errors
    writer, reader = reports[0], reports[1]
    assert writer["node_id"] != reader["node_id"], "both nodes got the same cluster rank"
    assert writer["put_ok"], "writer's puts did not complete"
    assert reader["head_put_ok"], "reader's head put did not complete"

    tail = NUM_REQUEST_BLOCKS - LOCAL_HEAD_BLOCKS
    assert reader["a_pulled_blocks"] >= NUM_REQUEST_BLOCKS, \
        f"window A: prefetch pulled {reader['a_pulled_blocks']}/{NUM_REQUEST_BLOCKS}"
    assert reader["a_hit_blocks"] == NUM_REQUEST_BLOCKS, \
        f"window A: local GET matched {reader['a_hit_blocks']}/{NUM_REQUEST_BLOCKS}"
    assert reader["a_mismatched"] == [], f"window A: wrong bytes in {reader['a_mismatched']}"
    assert reader["b_pulled_blocks"] >= tail, \
        f"window B: prefetch pulled {reader['b_pulled_blocks']}/{tail} tail blocks"
    assert reader["b_hit_blocks"] == NUM_REQUEST_BLOCKS, \
        f"window B: local GET matched {reader['b_hit_blocks']}/{NUM_REQUEST_BLOCKS}"
    assert reader["b_head_mismatched"] == [], \
        f"window B: head does not hold node 1's bytes: {reader['b_head_mismatched']}"
    assert reader["b_tail_mismatched"] == [], \
        f"window B: tail does not hold node 0's bytes: {reader['b_tail_mismatched']}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
