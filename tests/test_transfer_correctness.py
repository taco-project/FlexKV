"""
FlexKV transfer correctness test suite.

Systematically verifies KV Cache data correctness across all combinations of:
  - Parallel configurations: PP=2, TP=2, PP+TP, simulated cross-node PP/TP variants
  - Storage backends: CPU-only, CPU+SSD, GDS, Layerwise CPU-only, Layerwise CPU+SSD

Core verification principle:
  Fill GPU blocks with deterministic hash values keyed on (layer_id, kv_id, head_id, token_ids),
  perform a full round-trip transfer, then verify the restored data matches the original hashes.
  Any routing error (layer offset, head shard offset, etc.) will cause verification failure.

GPU availability strategy:
  - Prefer real multi-GPU testing when enough physical GPUs are available.
  - Automatically downgrade to override_device_id simulation on fewer GPUs.
  - Skip tests that cannot be simulated at all.
  - Mark simulated tests with [simulated] in the test label.

Extensibility:
  - Add new storage backends: add an entry to BACKEND_REGISTRY.
  - Add new parallel configs (e.g. CP): add an entry to PARALLEL_CONFIG_REGISTRY.
  - Core test logic (_run_transfer_test) requires no modification.
"""

import os
import time
import threading
import ctypes
import socket
import struct
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import multiprocessing as mp

from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.request import KVResponseStatus
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.kvmanager import KVManager
from flexkv.server.client import KVTPClient

from common_utils import (
    generate_request_pair,
    block_ids_2_slot_mapping,
    GPUKVCacheVerifier,
)

# ---------------------------------------------------------------------------
# Global test constants
# ---------------------------------------------------------------------------
NUM_LAYERS_TOTAL  = 4      # total model layers
NUM_KV_HEADS      = 8      # total KV heads (before TP sharding)
HEAD_SIZE         = 64
TOKENS_PER_BLOCK  = 16
DTYPE             = torch.float16
NUM_GPU_BLOCKS    = 64
BLOCK_PER_REQ     = 4
NUM_REQUESTS      = NUM_GPU_BLOCKS // BLOCK_PER_REQ   # 16 requests

# ---------------------------------------------------------------------------
# Task 1: Parallel configuration registry & GPU adaptive strategy
# ---------------------------------------------------------------------------

@dataclass
class WorkerSpec:
    """Describes a single logical worker (one PP stage × one TP rank)."""
    pp_rank: int
    tp_rank: int
    pp_start_layer: int   # absolute layer index where this PP stage starts
    num_layers: int       # number of layers owned by this PP stage
    num_heads: int        # number of KV heads owned by this TP rank (after sharding)
    head_offset: int      # global head index offset for this TP rank
    device_id: int        # physical GPU device id to use for tensor allocation
    register_device_id: int  # logical device_id used for KVManager registration


@dataclass
class ParallelConfig:
    """Describes a parallel configuration to be tested."""
    name: str
    pp_size: int
    tp_size: int
    # Minimum number of physical GPUs needed for a "real" (non-simulated) run.
    # Set to 1 if even a single GPU suffices for a real run.
    min_gpus_for_real: int
    # Whether this config can be simulated via override_device_id when GPUs are insufficient.
    can_simulate: bool = True

    def total_workers(self) -> int:
        return self.pp_size * self.tp_size

    def resolve_device_assignment(self, available_gpus: int) -> Tuple[List[WorkerSpec], bool]:
        """Return (worker_specs, is_simulated).

        Strategy:
          1. If available_gpus >= min_gpus_for_real → assign workers to distinct GPUs (real).
          2. Elif can_simulate → assign all workers to GPU 0 via override_device_id (simulated).
          3. Else → return ([], False) to signal skip.
        """
        if available_gpus == 0:
            return [], False

        layers_per_pp = NUM_LAYERS_TOTAL // self.pp_size
        heads_per_tp  = NUM_KV_HEADS // self.tp_size

        if available_gpus >= self.min_gpus_for_real:
            # Real multi-GPU: assign each worker to a distinct GPU (round-robin if needed)
            is_simulated = False
            workers = []
            worker_idx = 0
            for pp_rank in range(self.pp_size):
                for tp_rank in range(self.tp_size):
                    device_id = worker_idx % available_gpus
                    workers.append(WorkerSpec(
                        pp_rank=pp_rank,
                        tp_rank=tp_rank,
                        pp_start_layer=pp_rank * layers_per_pp,
                        num_layers=layers_per_pp,
                        num_heads=heads_per_tp,
                        head_offset=tp_rank * heads_per_tp,
                        device_id=device_id,
                        register_device_id=worker_idx,  # unique logical id
                    ))
                    worker_idx += 1
            return workers, is_simulated
        elif self.can_simulate:
            # Simulated: all workers share GPU 0, but use distinct register_device_id
            is_simulated = True
            workers = []
            worker_idx = 0
            for pp_rank in range(self.pp_size):
                for tp_rank in range(self.tp_size):
                    workers.append(WorkerSpec(
                        pp_rank=pp_rank,
                        tp_rank=tp_rank,
                        pp_start_layer=pp_rank * layers_per_pp,
                        num_layers=layers_per_pp,
                        num_heads=heads_per_tp,
                        head_offset=tp_rank * heads_per_tp,
                        device_id=0,           # all on GPU 0
                        register_device_id=worker_idx,
                    ))
                    worker_idx += 1
            return workers, is_simulated
        else:
            return [], False


# Registry of all parallel configurations to test.
# To add a new config (e.g. CP), simply append a new ParallelConfig here.
PARALLEL_CONFIG_REGISTRY: List[ParallelConfig] = [
    ParallelConfig(
        name="pp2",
        pp_size=2, tp_size=1,
        min_gpus_for_real=2,
        can_simulate=True,
    ),
    ParallelConfig(
        name="tp2",
        pp_size=1, tp_size=2,
        min_gpus_for_real=2,
        can_simulate=True,
    ),
    ParallelConfig(
        name="pp2_tp2",
        pp_size=2, tp_size=2,
        min_gpus_for_real=4,
        can_simulate=True,
    ),
    ParallelConfig(
        name="sim_cross_node_pp2",
        pp_size=2, tp_size=1,
        min_gpus_for_real=2,
        can_simulate=True,
    ),
    ParallelConfig(
        name="sim_cross_node_tp2",
        pp_size=1, tp_size=2,
        min_gpus_for_real=2,
        can_simulate=True,
    ),
    ParallelConfig(
        name="sim_cross_node_pp2_tp2",
        pp_size=2, tp_size=2,
        min_gpus_for_real=4,
        can_simulate=True,
    ),
]

# ---------------------------------------------------------------------------
# Task 3: Storage backend registry (CacheConfig factory)
# ---------------------------------------------------------------------------

class Backend(str, Enum):
    CPU_ONLY           = "cpu_only"
    CPU_SSD            = "cpu_ssd"
    GDS                = "gds"
    LAYERWISE_CPU_ONLY = "layerwise_cpu_only"
    LAYERWISE_CPU_SSD  = "layerwise_cpu_ssd"


def make_cache_config(backend: Backend) -> Optional[CacheConfig]:
    """Factory: return a CacheConfig for the given backend, or None if unavailable.

    Returns None when:
      - backend == GDS and FLEXKV_ENABLE_GDS != "1"
    The caller should call pytest.skip() when None is returned.

    To add a new backend, add a new branch here — no other code needs changing.
    """
    if backend == Backend.CPU_ONLY:
        return CacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            enable_cpu=True,
            enable_ssd=False,
            enable_gds=False,
            enable_remote=False,
            num_cpu_blocks=512,
            num_ssd_blocks=0,
        )
    elif backend == Backend.CPU_SSD:
        return CacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            enable_cpu=True,
            enable_ssd=True,
            enable_gds=False,
            enable_remote=False,
            num_cpu_blocks=512,   # large enough to hold all blocks without SSD eviction
            num_ssd_blocks=512,
            ssd_cache_dir=["/tmp/flexkv_test_ssd_cache"],
        )
    elif backend == Backend.GDS:
        if os.environ.get("FLEXKV_ENABLE_GDS", "0") != "1":
            return None
        return CacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            enable_cpu=True,
            enable_ssd=True,
            enable_gds=True,
            enable_remote=False,
            num_cpu_blocks=256,
            num_ssd_blocks=512,
        )
    elif backend == Backend.LAYERWISE_CPU_ONLY:
        return CacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            enable_cpu=True,
            enable_ssd=False,
            enable_gds=False,
            enable_remote=False,
            num_cpu_blocks=512,
            num_ssd_blocks=0,
        )
    elif backend == Backend.LAYERWISE_CPU_SSD:
        return CacheConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            enable_cpu=True,
            enable_ssd=True,
            enable_gds=False,
            enable_remote=False,
            num_cpu_blocks=512,   # large enough to hold all blocks without SSD eviction
            num_ssd_blocks=512,
            ssd_cache_dir=["/tmp/flexkv_test_ssd_cache"],
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def is_layerwise_backend(backend: Backend) -> bool:
    return backend in (Backend.LAYERWISE_CPU_ONLY, Backend.LAYERWISE_CPU_SSD)


# ---------------------------------------------------------------------------
# Task 2: Generic multi-worker registration framework
# ---------------------------------------------------------------------------

def _worker_process_fn(
    worker_spec: WorkerSpec,
    gpu_register_port: str,
    model_config: ModelConfig,
    cache_config: CacheConfig,
    num_gpu_blocks: int,
    child_conn,
) -> None:
    """Subprocess: allocate GPU blocks for one (pp_rank, tp_rank) worker and register to KVManager.

    Sends a list of TensorSharedHandle back through child_conn, then stays alive
    until the process is terminated by the parent.
    """
    try:
        device_id = worker_spec.device_id
        kv_dim = 2  # non-MLA

        # Allocate GPU blocks for this worker's layers
        gpu_blocks = []
        for _ in range(worker_spec.num_layers):
            t = torch.zeros(
                kv_dim,
                num_gpu_blocks,
                TOKENS_PER_BLOCK,
                worker_spec.num_heads,
                HEAD_SIZE,
                dtype=DTYPE,
            ).cuda(device_id)
            gpu_blocks.append(t)

        gpu_kv_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=worker_spec.num_layers,
            num_block=num_gpu_blocks,
            tokens_per_block=TOKENS_PER_BLOCK,
            num_head=worker_spec.num_heads,
            head_size=HEAD_SIZE,
            is_mla=False,
        )

        tp_client = KVTPClient(
            gpu_register_port=gpu_register_port,
            dp_client_id=0,
            pp_rank=worker_spec.pp_rank,
            pp_start_layer=worker_spec.pp_start_layer,
            device_id=worker_spec.register_device_id,
        )
        tp_client.register_to_server(
            gpu_blocks,
            gpu_kv_layout,
            override_device_id=worker_spec.register_device_id,
        )

        # Send handles back to main process
        shared = [TensorSharedHandle(t) for t in gpu_blocks]
        child_conn.send(shared)
        child_conn.close()

        # Stay alive until terminated
        while True:
            time.sleep(1)

    except Exception:
        traceback.print_exc()
        try:
            child_conn.send(None)
            child_conn.close()
        except Exception:
            pass


def _shutdown_processes(procs: list) -> None:
    """Terminate all worker processes gracefully."""
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)


def _collect_gpu_blocks(
    pipes: list,
    workers: List[WorkerSpec],
    test_label: str,
) -> List[List[TensorSharedHandle]]:
    """Collect TensorSharedHandle lists from all worker pipes.

    Returns a flat list indexed by worker index (same order as `workers`).
    Each element is a list of TensorSharedHandle, one per layer in that worker's PP stage.
    """
    all_handles = []
    for idx, (parent_conn, worker) in enumerate(zip(pipes, workers)):
        handles = parent_conn.recv()
        assert handles is not None, (
            f"[{test_label}] Worker pp_rank={worker.pp_rank} tp_rank={worker.tp_rank} "
            f"failed to register"
        )
        all_handles.append(handles)
        parent_conn.close()
    return all_handles


def _build_combined_verifier(
    all_handles: List[List[TensorSharedHandle]],
    workers: List[WorkerSpec],
    pp_size: int,
    tp_size: int,
) -> GPUKVCacheVerifier:
    """Build a GPUKVCacheVerifier that covers all PP stages and TP ranks.

    GPUKVCacheVerifier expects:
      shared_gpu_blocks[tp_rank][layer_local_idx] = tensor

    For PP+TP, we concatenate layers across PP stages for each TP rank:
      shared_gpu_blocks[tp_rank] = [pp0_layer0, pp0_layer1, ..., pp1_layer0, pp1_layer1, ...]

    The verifier's num_layers = NUM_LAYERS_TOTAL (all layers across all PP stages).
    The verifier's num_head = heads_per_tp (per TP rank, after sharding).
    """
    layers_per_pp = NUM_LAYERS_TOTAL // pp_size
    heads_per_tp  = NUM_KV_HEADS // tp_size

    # Materialise handles into tensors
    # worker_tensors[worker_idx][layer_local] = tensor
    worker_tensors = []
    for handles in all_handles:
        worker_tensors.append([h.get_tensor() for h in handles])

    # Build per-TP-rank combined layer list
    # For tp_rank r: collect layers from all PP stages in order
    # Shape: blocks_by_tp[tp_rank][layer_global] = torch.Tensor
    blocks_by_tp: List[List[torch.Tensor]] = []
    for tp_rank in range(tp_size):
        layers_for_this_tp = []
        for pp_rank in range(pp_size):
            worker_idx = pp_rank * tp_size + tp_rank
            layers_for_this_tp.extend(worker_tensors[worker_idx])
        blocks_by_tp.append(layers_for_this_tp)

    full_gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=NUM_LAYERS_TOTAL,
        num_block=NUM_GPU_BLOCKS,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_head=heads_per_tp,
        head_size=HEAD_SIZE,
        is_mla=False,
    )

    verifier = GPUKVCacheVerifier(
        shared_gpu_blocks=blocks_by_tp,
        gpu_kv_layout=full_gpu_layout,
        tp_size=tp_size,
        tokens_per_block=TOKENS_PER_BLOCK,
        dtype=DTYPE,
        gpu_layout_type=0,  # LAYERFIRST, one tensor per layer
    )
    return verifier


# ---------------------------------------------------------------------------
# Mock SGLang eventfd client (needed for layerwise tests)
# ---------------------------------------------------------------------------

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_EFD_SEMAPHORE = 0x1


def _sys_eventfd(initval: int = 0, flags: int = 0) -> int:
    fd = _libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd failed: {os.strerror(err)}")
    return fd


def _send_fds_via_scm(sock: socket.socket, fds: list, extra_data: bytes = b"x"):
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], ancdata)


def _mock_sglang_eventfd_client(
    socket_path: str,
    tp_rank: int,
    tp_size: int,
    num_layers: int,
    num_counters: int = 3,
    max_retries: int = 120,
    retry_interval: float = 0.5,
):
    """Background thread: simulate SGLang sending eventfds to LayerwiseTransferWorker."""
    created_fds = []
    try:
        for _ in range(num_counters * num_layers):
            created_fds.append(_sys_eventfd(0, _EFD_SEMAPHORE))

        sock = None
        for _ in range(max_retries):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(socket_path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                sock.close()
                sock = None
                time.sleep(retry_interval)

        if sock is None:
            print(f"[MockEventfdClient] FAILED to connect to {socket_path}")
            return

        metadata = struct.pack("iiii", tp_rank, tp_size, num_layers, num_counters)
        sock.sendall(metadata)

        fd_idx = 0
        for counter_id in range(num_counters):
            fds = created_fds[fd_idx:fd_idx + num_layers]
            fd_idx += num_layers
            _send_fds_via_scm(sock, fds, struct.pack("i", counter_id))

        sock.settimeout(30.0)
        ack = sock.recv(1)
        if ack and ack[0] == 1:
            print(f"[MockEventfdClient] Eventfd handshake OK (tp_rank={tp_rank})")
        sock.close()
    except Exception as e:
        print(f"[MockEventfdClient] Error: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Task 4: Core transfer test function (supports any parallel config × backend)
# ---------------------------------------------------------------------------

def _run_transfer_test(
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
    workers: List[WorkerSpec],
    is_simulated: bool,
    layerwise: bool,
    test_label: str,
) -> None:
    """Run a full round-trip transfer test for the given parallel config and backend.

    Steps:
      1. Start KVManager with the given pp_size / tp_size.
      2. Spawn one subprocess per worker; each registers its GPU blocks.
      3. Fill GPU blocks with deterministic hash values.
      4. PUT all requests.
      5. Clear GPU blocks to zero.
      6. GET data back.
      7. Verify all PP stages × TP ranks match original hash values.
    """
    sim_tag = " [simulated]" if is_simulated else ""
    print(f"\n[{test_label}]{sim_tag} Starting transfer test "
          f"(pp={parallel_config.pp_size}, tp={parallel_config.tp_size}, "
          f"backend={'layerwise+' if layerwise else ''}{cache_config})")

    model_config = ModelConfig(
        num_layers=NUM_LAYERS_TOTAL,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
        use_mla=False,
        tp_size=parallel_config.tp_size,
        dp_size=1,
        pp_size=parallel_config.pp_size,
    )

    # ---- Layerwise env setup ----
    orig_layerwise_env  = os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER")
    orig_layerwise_flag = GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer
    if layerwise:
        os.environ["FLEXKV_ENABLE_LAYERWISE_TRANSFER"] = "1"
        GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = True

    procs: List[mp.Process] = []
    eventfd_threads: List[threading.Thread] = []

    try:
        kvmanager = KVManager(
            model_config=model_config,
            cache_config=cache_config,
            dp_client_id=0,
        )

        # Start mock eventfd clients BEFORE kvmanager.start() for layerwise.
        # FlexKV creates one LayerwiseWorker per pp_rank, each listening on its
        # own socket (e.g. _pp0.sock, _pp1.sock when pp_size > 1).  Each socket
        # expects tp_size connections (one per TP rank).  We must mock all of them.
        if layerwise:
            base_socket = os.environ.get(
                "FLEXKV_LAYERWISE_EVENTFD_SOCKET",
                "/tmp/flexkv_layerwise_eventfd.sock",
            )
            layers_per_pp = NUM_LAYERS_TOTAL // parallel_config.pp_size
            for pp_rank in range(parallel_config.pp_size):
                # Replicate build_layerwise_eventfd_socket_path logic
                if parallel_config.pp_size > 1:
                    root, ext = os.path.splitext(base_socket)
                    sock_path = f"{root}_pp{pp_rank}{ext}"
                else:
                    sock_path = base_socket
                # Send eventfds for each TP rank on this PP rank's socket
                for tp_rank in range(parallel_config.tp_size):
                    t = threading.Thread(
                        target=_mock_sglang_eventfd_client,
                        args=(sock_path, tp_rank, parallel_config.tp_size,
                              layers_per_pp),
                        daemon=True,
                    )
                    t.start()
                    eventfd_threads.append(t)

        kvmanager.start()

        mp_ctx = mp.get_context("spawn")
        pipes = []

        for worker in workers:
            parent_conn, child_conn = mp_ctx.Pipe()
            pipes.append(parent_conn)
            p = mp_ctx.Process(
                target=_worker_process_fn,
                args=(worker, kvmanager.gpu_register_port,
                      model_config, cache_config,
                      NUM_GPU_BLOCKS, child_conn),
                daemon=True,
            )
            procs.append(p)
            p.start()

        # Collect GPU block handles from all workers
        all_handles = _collect_gpu_blocks(pipes, workers, test_label)

        # Build combined verifier covering all PP stages × TP ranks
        verifier = _build_combined_verifier(
            all_handles, workers,
            parallel_config.pp_size, parallel_config.tp_size,
        )

        # Wait for KVManager to be ready
        for _ in range(60):
            if kvmanager.is_ready():
                break
            time.sleep(1)
        assert kvmanager.is_ready(), f"[{test_label}] KVManager not ready after 60s"

        # ---- Generate requests ----
        request_pairs = [
            generate_request_pair(i, BLOCK_PER_REQ, NUM_GPU_BLOCKS, TOKENS_PER_BLOCK, 1)
            for i in range(NUM_REQUESTS)
        ]

        # ---- PUT phase ----
        print(f"[{test_label}] PUT phase: writing {NUM_REQUESTS} requests...")
        for token_ids, block_ids, _ in request_pairs:
            verifier.fill_gpu_blocks(token_ids, block_ids)
            write_req = kvmanager.put_async(
                token_ids=token_ids,
                slot_mapping=block_ids_2_slot_mapping(block_ids, TOKENS_PER_BLOCK),
                token_mask=None,
            )
            results = kvmanager.wait([write_req], completely=True)
            assert results[write_req].status == KVResponseStatus.SUCCESS, \
                f"[{test_label}] PUT failed for request"
            verifier.clear_gpu_blocks(block_ids)
        print(f"[{test_label}] PUT phase done.")

        # ---- GET phase ----
        print(f"[{test_label}] GET phase: reading back {NUM_REQUESTS} requests...")
        total_hit  = 0
        total_miss = 0

        if layerwise:
            # Layerwise: batch all GETs together
            task_ids      = []
            slot_mappings = []
            req_info      = []
            for token_ids, block_ids, _ in request_pairs:
                task_id, _ = kvmanager.get_match(token_ids=token_ids, token_mask=None)
                task_ids.append(task_id)
                slot_mappings.append(block_ids_2_slot_mapping(block_ids, TOKENS_PER_BLOCK))
                req_info.append((token_ids, block_ids))

            returned_ids = kvmanager.launch(
                task_ids=task_ids,
                slot_mappings=slot_mappings,
                as_batch=True,
                layerwise_transfer=True,
            )
            batch_id = returned_ids[0]
            batch_results = kvmanager.wait(batch_id, completely=True)
            kvresponse = batch_results[batch_id]
            assert kvresponse.status == KVResponseStatus.SUCCESS, \
                f"[{test_label}] Layerwise batch GET failed: {kvresponse.status}"

            for idx, (token_ids, block_ids) in enumerate(req_info):
                mask = kvresponse.get_mask(idx)
                total_hit  += mask.sum().item()
                total_miss += len(mask) - mask.sum().item()
                valid_tokens = mask.sum().item() // TOKENS_PER_BLOCK * TOKENS_PER_BLOCK
                if valid_tokens > 0:
                    assert verifier.verify_kv_blocks(
                        token_ids[:valid_tokens],
                        block_ids[:valid_tokens // TOKENS_PER_BLOCK],
                    ), f"[{test_label}] Data mismatch after layerwise GET (request {idx})"
        else:
            # Non-layerwise: launch each GET individually
            for token_ids, block_ids, _ in request_pairs:
                task_id, _ = kvmanager.get_match(token_ids=token_ids, token_mask=None)
                kvmanager.launch(
                    task_ids=[task_id],
                    slot_mappings=[block_ids_2_slot_mapping(block_ids, TOKENS_PER_BLOCK)],
                )
                results = kvmanager.wait([task_id], completely=True)
                kvresponse = results[task_id]
                assert kvresponse.status == KVResponseStatus.SUCCESS, \
                    f"[{test_label}] GET failed"
                total_hit  += kvresponse.return_mask.sum().item()
                total_miss += len(kvresponse.return_mask) - kvresponse.return_mask.sum().item()
                valid_tokens = kvresponse.return_mask.sum().item() // TOKENS_PER_BLOCK * TOKENS_PER_BLOCK
                if valid_tokens > 0:
                    assert verifier.verify_kv_blocks(
                        token_ids[:valid_tokens],
                        block_ids[:valid_tokens // TOKENS_PER_BLOCK],
                    ), f"[{test_label}] Data mismatch after GET"

        print(f"[{test_label}] GET phase done: hit={total_hit}, miss={total_miss}")

        # When backing store is large enough, expect 0 misses
        num_cpu = cache_config.num_cpu_blocks
        num_ssd = cache_config.num_ssd_blocks
        if (cache_config.enable_cpu and num_cpu >= NUM_GPU_BLOCKS) or \
           (cache_config.enable_ssd and num_ssd >= NUM_GPU_BLOCKS) or \
           (cache_config.enable_gds and num_ssd >= NUM_GPU_BLOCKS):
            assert total_miss == 0, \
                f"[{test_label}] Expected 0 cache miss, got {total_miss}"

        print(f"[{test_label}] PASSED ✓")

    finally:
        _shutdown_processes(procs)
        kvmanager.shutdown()
        # Restore layerwise env
        if layerwise:
            if orig_layerwise_env is None:
                os.environ.pop("FLEXKV_ENABLE_LAYERWISE_TRANSFER", None)
            else:
                os.environ["FLEXKV_ENABLE_LAYERWISE_TRANSFER"] = orig_layerwise_env
            GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = orig_layerwise_flag
            for t in eventfd_threads:
                t.join(timeout=10)


# ---------------------------------------------------------------------------
# Task 5: Parameterized test cases with skip / downgrade logic
# ---------------------------------------------------------------------------

# Build the full test matrix: (parallel_config, backend)
# Each combination is a separate pytest test case.
_TEST_PARAMS = []
for _pc in PARALLEL_CONFIG_REGISTRY:
    for _be in Backend:
        _TEST_PARAMS.append(pytest.param(
            _pc, _be,
            id=f"{_pc.name}__{_be.value}",
        ))


@pytest.mark.parametrize("parallel_config,backend", _TEST_PARAMS)
def test_transfer_correctness(parallel_config: ParallelConfig, backend: Backend):
    """Parametrized transfer correctness test: parallel_config × storage backend.

    Automatically:
      - Skips if no CUDA devices are available.
      - Skips GDS tests when FLEXKV_ENABLE_GDS != "1".
      - Downgrades to simulated (override_device_id) when physical GPUs are insufficient.
      - Skips if the config cannot be simulated and GPUs are insufficient.
    """
    # --- GPU availability check ---
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        pytest.skip("No CUDA devices available")

    # --- Backend availability check ---
    cache_config = make_cache_config(backend)
    if cache_config is None:
        pytest.skip(f"Backend '{backend.value}' is not available "
                    f"(set FLEXKV_ENABLE_GDS=1 to enable GDS tests)")

    # --- Resolve device assignment (real multi-GPU vs simulated) ---
    workers, is_simulated = parallel_config.resolve_device_assignment(available_gpus)
    if not workers:
        pytest.skip(
            f"Parallel config '{parallel_config.name}' requires "
            f">= {parallel_config.min_gpus_for_real} GPUs for real testing "
            f"and cannot be simulated. Available: {available_gpus}"
        )

    sim_tag = " [simulated]" if is_simulated else " [real]"
    test_label = f"{parallel_config.name}__{backend.value}{sim_tag}"

    layerwise = is_layerwise_backend(backend)

    _run_transfer_test(
        parallel_config=parallel_config,
        cache_config=cache_config,
        workers=workers,
        is_simulated=is_simulated,
        layerwise=layerwise,
        test_label=test_label,
    )
