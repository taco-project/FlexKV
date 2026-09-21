"""Shared pieces of the radixshmem end-to-end tests.

Used by ``test_e2e_radix_shmem.py`` (one node, one or two DP processes on one
radix-server) and ``test_e2e_radix_prefetch_p2p.py`` (two nodes, one cluster):
GPU block patterns and their byte comparison, request construction, the TP
client subprocess that owns a DP's GPU tensors, KVManager put/get wrappers, and
the RDMA / etcd discovery the cross-node test needs.

FlexKV is imported lazily inside the functions: the DP / node subprocesses set
their ``FLEXKV_*`` environment before the first ``flexkv`` import, and this
module is re-imported by every spawned child.
"""
from __future__ import annotations

import contextlib
import glob
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

TOKENS_PER_BLOCK = 16
PATTERN_SEED = 0x5EED


# ------------------------------------------------------------------ host

def free_port() -> int:
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def active_rdma_devices() -> List[str]:
    """RDMA devices with at least one ACTIVE port, honoring FLEXKV_TEST_RDMA_DEVICES."""
    override = os.getenv("FLEXKV_TEST_RDMA_DEVICES", "").strip()
    names = ([d for d in override.split(",") if d] if override
             else sorted(os.path.basename(p)
                         for p in glob.glob("/sys/class/infiniband/*")))
    active = []
    for name in names:
        for state in glob.glob(f"/sys/class/infiniband/{name}/ports/*/state"):
            with contextlib.suppress(OSError):
                with open(state) as handle:
                    if "ACTIVE" in handle.read():
                        active.append(name)
                        break
    return active


def start_private_etcd(prefix: str = "flexkv_etcd_"):
    """Start a single-member etcd on free ports. Returns (proc, workdir,
    registry) or (None, None, None) when no ``etcd`` binary is on PATH."""
    etcd = shutil.which("etcd")
    if not etcd:
        return None, None, None
    client_port, peer_port = free_port(), free_port()
    workdir = tempfile.mkdtemp(prefix=prefix)
    proc = subprocess.Popen(
        [etcd, "--name", "t", "--data-dir", os.path.join(workdir, "data"),
         "--listen-client-urls", f"http://127.0.0.1:{client_port}",
         "--advertise-client-urls", f"http://127.0.0.1:{client_port}",
         "--listen-peer-urls", f"http://127.0.0.1:{peer_port}",
         "--initial-advertise-peer-urls", f"http://127.0.0.1:{peer_port}",
         "--initial-cluster", f"t=http://127.0.0.1:{peer_port}"],
        stdout=open(os.path.join(workdir, "etcd.log"), "w"), stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError):
            with socket.create_connection(("127.0.0.1", client_port), timeout=0.5):
                return proc, workdir, f"etcd://127.0.0.1:{client_port}"
        time.sleep(0.2)
    proc.kill()
    shutil.rmtree(workdir, ignore_errors=True)
    return None, None, None


def stop_private_etcd(proc, workdir) -> None:
    if proc is not None:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(10)
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)


def write_radix_config(workdir: str, config: dict, name: str = "radixshmem.yaml") -> str:
    """Write the run's radixshmem YAML (``FLEXKV_RADIXSHMEM_CONFIG_PATH``) and
    return its path."""
    import yaml
    path = os.path.join(workdir, name)
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return path


def sweep_radix_files(cluster_id: str) -> None:
    """Drop what a run under ``cluster_id`` (the radixshmem namespace; every
    shm / socket / IPC name of the run contains it) may have left in shm / tmp."""
    for pattern in (f"/dev/shm/*{cluster_id}*", f"/dev/hugepages/*{cluster_id}*",
                    f"/tmp/flexkv_{cluster_id}*"):
        for stale in glob.glob(pattern):
            with contextlib.suppress(OSError):
                os.unlink(stale)


# ----------------------------------------------------------- GPU bytes

def block_pattern(layer: int, block_id: int, shape, dtype,
                  writer: int = 0) -> torch.Tensor:
    """Deterministic content for one (layer, block) as written by ``writer``.

    Random rather than structured so a transfer landing on the wrong block or
    layer cannot compare equal; generated on the CPU so every process derives
    it identically. ``writer`` makes the same block differ between writers,
    which is how a byte comparison alone tells whose copy a GET served.
    """
    generator = torch.Generator().manual_seed(
        PATTERN_SEED + writer * 1_000_003 + block_id * 128 + layer)
    return torch.randn(tuple(shape), generator=generator).to(dtype)


def write_pattern(gpu_tensors, block_ids, writer: int = 0) -> None:
    for layer, tensor in enumerate(gpu_tensors):
        for block_id in block_ids:
            block = tensor[:, block_id]
            block.copy_(block_pattern(layer, int(block_id), block.shape, tensor.dtype, writer))
    torch.cuda.synchronize()


def clear_blocks(gpu_tensors, block_ids) -> None:
    """Zero the blocks a GET is supposed to fill, so a no-op fails the check."""
    for tensor in gpu_tensors:
        for block_id in block_ids:
            tensor[:, block_id].zero_()
    torch.cuda.synchronize()


def mismatched_blocks(gpu_tensors, block_ids, writer: int = 0) -> list:
    """(layer, block) pairs whose content is not what ``writer`` wrote."""
    bad = []
    for layer, tensor in enumerate(gpu_tensors):
        for block_id in block_ids:
            got = tensor[:, block_id].cpu()
            want = block_pattern(layer, int(block_id), got.shape, got.dtype, writer)
            if not torch.equal(got, want):
                bad.append((layer, int(block_id)))
    return bad


# ------------------------------------------------------------ requests

def build_request(num_blocks: int, first_block: int, seed: int,
                  tokens_per_block: int = TOKENS_PER_BLOCK):
    """(token_ids, slot_mapping, block_ids) for ``num_blocks`` GPU blocks starting
    at ``first_block``; token ids are seeded so two processes agree on them."""
    rng = np.random.default_rng(seed)
    block_ids = np.arange(first_block, first_block + num_blocks, dtype=np.int64)
    slot_mapping = (np.repeat(block_ids, tokens_per_block) * tokens_per_block
                    + np.tile(np.arange(tokens_per_block), num_blocks))
    token_ids = rng.integers(0, 32000, size=slot_mapping.shape, dtype=np.int64)
    return token_ids, slot_mapping, block_ids


def put_prefix(kvm, token_ids, slot_mapping, num_blocks: int,
               tokens_per_block: int = TOKENS_PER_BLOCK) -> bool:
    """PUT the first ``num_blocks`` blocks; True if the task completed."""
    from flexkv.common.request import KVResponseStatus
    num_tokens = num_blocks * tokens_per_block
    task_id = kvm.put_async(token_ids=token_ids[:num_tokens],
                            slot_mapping=slot_mapping[:num_tokens])
    status = kvm.wait([task_id], timeout=120, completely=True)
    return all(r.status == KVResponseStatus.SUCCESS for r in status.values())


def get_blocks(kvm, token_ids, slot_mapping,
               tokens_per_block: int = TOKENS_PER_BLOCK) -> int:
    """One-shot ``get_async`` + wait; matched blocks, 0 if it did not succeed."""
    from flexkv.common.request import KVResponseStatus
    task_id = kvm.get_async(token_ids=token_ids, slot_mapping=slot_mapping)
    response = kvm.wait([task_id], timeout=120, completely=True)[task_id]
    if response.status != KVResponseStatus.SUCCESS or response.return_mask is None:
        return 0
    return int(np.count_nonzero(response.return_mask)) // tokens_per_block


def match_and_load(kvm, token_ids, slot_mapping,
                   tokens_per_block: int = TOKENS_PER_BLOCK) -> int:
    """The sglang connector's two steps: ``get_match`` then ``launch`` + wait for
    the matched prefix. Returns the matched blocks, 0 if the load failed."""
    from flexkv.common.request import KVResponseStatus
    task_id, mask = kvm.get_match(token_ids=token_ids)
    hit_tokens = int(np.count_nonzero(mask))
    if hit_tokens == 0:
        return 0
    kvm.launch([task_id], [slot_mapping[:hit_tokens]])
    response = kvm.wait([task_id], timeout=120, completely=True)[task_id]
    if response.status != KVResponseStatus.SUCCESS:
        return 0
    return hit_tokens // tokens_per_block


def wait_kv_manager_ready(kvm, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if kvm.is_ready():
            return
        time.sleep(0.1)
    raise RuntimeError(f"KVManager not ready in {timeout:.0f}s")


# ---------------------------------------------------------- TP client

def tp_client_proc(server_recv_port: str, dp_client_id: int, device_id: int,
                   model_config, cache_config, num_gpu_blocks: int, child_conn) -> None:
    """Spawn target: owns one DP's GPU tensors, registers them with the TE and
    hands their IPC handles back over ``child_conn``; then stays alive so the
    tensors do."""
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.common.memory_handle import TensorSharedHandle
    from flexkv.server.client import KVTPClient

    tp_client = KVTPClient(server_recv_port, dp_client_id, device_id)
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads // model_config.tp_size,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    gpu_blocks = [
        torch.zeros(size=tuple(gpu_layout.kv_shape[1:]),
                    dtype=model_config.dtype).cuda(device_id)
        for _ in range(model_config.num_layers)
    ]
    tp_client.register_to_server(gpu_blocks, gpu_layout)
    child_conn.send([TensorSharedHandle(t) for t in gpu_blocks])
    child_conn.close()
    while True:
        time.sleep(1)


def start_tp_client(kvm, dp_client_id: int, device_id: int, model_config, cache_config,
                    num_gpu_blocks: int) -> Tuple[mp.Process, list]:
    """Start the TP client for ``kvm`` and return (process, this process's view
    of its GPU tensors)."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=tp_client_proc,
        args=(kvm.gpu_register_port, dp_client_id, device_id, model_config,
              cache_config, num_gpu_blocks, child_conn),
        daemon=True,
    )
    proc.start()
    gpu_tensors = [handle.get_tensor() for handle in parent_conn.recv()]
    return proc, gpu_tensors


def stop_tp_client(proc: Optional[mp.Process]) -> None:
    if proc is not None:
        proc.terminate()
        proc.join(timeout=10)
