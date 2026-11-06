"""A peer's SSD <-> local CPU (PEERSSD2H / H2PEERSSD), radixshmem mode.

The SSD tier is indexed by an index-only radix-server per node; a cluster-wide
match ends in a peer's run whose slot ids are that peer's SSD block ids. This
worker moves such a run into the local CPU pool with a destination-initiated,
synchronous ``read()`` of a pluggable :class:`RemoteSSDReader`, and -- for the
SSD global dedup of PUTs -- local CPU blocks into slots allocated in a peer's
SSD with its ``write()``:

  * ``FLEXKV_REMOTE_SSD_READER=module:factory`` -- ``factory(worker, pool_id)``
    returns the reader (the real cross-node transport plugs in here);
  * unset -- :class:`SharedFileRemoteSSDReader`, which reads the peer's SSD
    files directly. It needs the peer's files to be visible at the paths the
    peer published (one host, or a shared file system) and serves validation,
    not production.

Every node's TE publishes where its SSD files are and how they are laid out
(:func:`publish_ssd_manifest`); a reader resolves a peer's rank in the SSD
index cluster to its node name through the SSD radix-server.
"""

import importlib
import json
import os
import time
from typing import Any, Dict, Optional, Protocol

import torch

from flexkv import c_ext
from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.pool import PoolId
from flexkv.common.storage import StorageHandle
from flexkv.common.transfer import TransferType
from flexkv.transfer.worker_op import WorkerTransferOp, WorkerTransferResult
from flexkv.transfer.workers.cpu_ssd import CPUSSDDiskTransferWorker

_MANIFEST_VERSION = 1


class RemoteSSDReader(Protocol):
    """Moves blocks of a peer's SSD into this node's CPU pool, synchronously.

    ``src_ssd_blocks`` are the peer's SSD block ids of pool ``pool_id``,
    ``dst_cpu_blocks`` the local CPU slots of the same pool (same length,
    int64). ``layer_id`` / ``layer_end`` bound the layers moved (the whole
    block when the layout is opaque). Raises on failure: the op then fails
    and the plan gives its staging slots back."""

    def read(self, peer_rank: int, pool_id: PoolId, src_ssd_blocks: torch.Tensor,
             dst_cpu_blocks: torch.Tensor, layer_id: int, layer_end: int) -> None: ...

    # Optional (H2PEERSSD): local CPU blocks -> the peer's SSD blocks, which
    # the caller allocated in the peer's SSD index. Raises on failure; the
    # source CPU blocks must stay unchanged until it returns.
    def write(self, peer_rank: int, pool_id: PoolId, src_cpu_blocks: torch.Tensor,
              dst_ssd_blocks: torch.Tensor) -> None: ...


def _manifest_dir() -> str:
    path = GLOBAL_CONFIG_FROM_ENV.peer_ssd_manifest_dir
    if not path:
        raise RuntimeError(
            "FLEXKV_PEER_SSD_MANIFEST_DIR is not set: the shared-file SSD reader finds a "
            "peer's SSD files through the manifests every node publishes there")
    return path


def _handle_entry(handle: Optional[StorageHandle]) -> Optional[Dict[str, Any]]:
    if handle is None:
        return None
    files = handle.get_file_list()
    if not isinstance(files, dict):
        files = {0: list(files)}
    return {"files": {str(k): list(v) for k, v in files.items()},
            "num_blocks_per_file": int(handle.num_blocks_per_file),
            "num_blocks": int(handle.kv_layout.num_block)}


def publish_ssd_manifest(cache_config: CacheConfig,
                         ssd_handle: Optional[StorageHandle],
                         swa_ssd_handle: Optional[StorageHandle]) -> Optional[str]:
    """Write this node's SSD file manifest (``<dir>/<node_name>.json``), where
    the shared-file reader of every peer looks for it. The node name is the
    SSD radix-server's. Also checks that the SSD index does not name more
    blocks than the files hold. Returns the path, None when no directory is
    configured (a custom reader brings its own discovery)."""
    from flexkv.server.shm_radix_bootstrap import attach_ssd_index_client

    client = attach_ssd_index_client(cache_config, attach_index=False, label="TransferManager-SSD")
    try:
        node_name = client.info.node_name or f"rank{client.info.rank}"
        pools = (client.info.geometry or {}).get("pools", {})
    finally:
        client.close()
    entries = {"full": _handle_entry(ssd_handle), "swa": _handle_entry(swa_ssd_handle)}
    for kind, entry in entries.items():
        indexed = int((pools.get(kind) or {}).get("num_slots", 0))
        held = entry["num_blocks"] if entry is not None else 0
        if indexed > held:
            raise ValueError(
                f"SSD radix index names {indexed} {kind} blocks but this FlexKV's SSD files "
                f"hold {held}")
    if not GLOBAL_CONFIG_FROM_ENV.peer_ssd_manifest_dir:
        return None
    directory = _manifest_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{node_name}.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({"version": _MANIFEST_VERSION, "node_name": node_name,
                   "written": time.time(), **entries}, f)
    os.replace(tmp, path)
    flexkv_logger.info(f"[peer-ssd] published SSD manifest {path}")
    return path


class SharedFileRemoteSSDReader:
    """Reads a peer's SSD files directly, with this worker's own io_uring
    engine and addressing (the peer's layout must equal this node's)."""

    def __init__(self, worker: "PeerSSDReadWorker", pool_id: PoolId):
        self._worker = worker
        self._kind = "swa" if pool_id is PoolId.SWA else "full"
        self._ioctx: Dict[int, Any] = {}
        self._node_names: Dict[int, str] = {}
        self._client = None

    def _node_name(self, rank: int) -> str:
        if rank not in self._node_names:
            import radixshmem
            if self._client is None:
                self._client = radixshmem.RadixClient(self._worker.ssd_server_name)
            for rec in self._client.nodes():
                if int(rec.get("rank", -1)) >= 0:
                    self._node_names[int(rec["rank"])] = str(rec.get("node_name", ""))
        name = self._node_names.get(rank)
        if not name:
            raise RuntimeError(f"[peer-ssd] no SSD index node has rank {rank}")
        return name

    def _peer_ctx(self, rank: int) -> Any:
        ctx = self._ioctx.get(rank)
        if ctx is not None:
            return ctx
        node = self._node_name(rank)
        path = os.path.join(_manifest_dir(), f"{node}.json")
        with open(path) as f:
            manifest = json.load(f)
        entry = manifest.get(self._kind)
        if entry is None:
            raise RuntimeError(f"[peer-ssd] {path} has no {self._kind} SSD pool")
        files = {int(k): v for k, v in entry["files"].items()}
        w = self._worker
        mine = {d: len(v) for d, v in w.ssd_files.items()}
        theirs = {d: len(v) for d, v in files.items()}
        if mine != theirs or int(entry["num_blocks_per_file"]) != int(w.num_blocks_per_file):
            raise RuntimeError(
                f"[peer-ssd] {node}'s {self._kind} SSD layout ({theirs} files, "
                f"{entry['num_blocks_per_file']} blocks/file) differs from this node's "
                f"({mine}, {w.num_blocks_per_file}); the shared-file reader needs them equal")
        ctx = c_ext.SSDIOCTX(files, len(files), GLOBAL_CONFIG_FROM_ENV.iouring_entries,
                             GLOBAL_CONFIG_FROM_ENV.iouring_flags)
        self._ioctx[rank] = ctx
        flexkv_logger.info(f"[peer-ssd] opened {node}'s {self._kind} SSD files (rank {rank})")
        return ctx

    def read(self, peer_rank: int, pool_id: PoolId, src_ssd_blocks: torch.Tensor,
             dst_cpu_blocks: torch.Tensor, layer_id: int, layer_end: int) -> None:
        w = self._worker
        own = w.ioctx
        w.ioctx = self._peer_ctx(peer_rank)
        try:
            w.read_local_layout(src_ssd_blocks, dst_cpu_blocks, layer_id, layer_end)
        finally:
            w.ioctx = own

    def write(self, peer_rank: int, pool_id: PoolId, src_cpu_blocks: torch.Tensor,
              dst_ssd_blocks: torch.Tensor) -> None:
        w = self._worker
        own = w.ioctx
        w.ioctx = self._peer_ctx(peer_rank)
        try:
            w.write_local_layout(src_cpu_blocks, dst_ssd_blocks)
        finally:
            w.ioctx = own


def _load_reader(worker: "PeerSSDReadWorker", pool_id: PoolId) -> RemoteSSDReader:
    spec = GLOBAL_CONFIG_FROM_ENV.remote_ssd_reader
    if not spec:
        return SharedFileRemoteSSDReader(worker, pool_id)
    module, _, factory = spec.partition(":")
    if not module or not factory:
        raise ValueError(f"FLEXKV_REMOTE_SSD_READER={spec!r} must be 'module:factory'")
    return getattr(importlib.import_module(module), factory)(worker, pool_id)


class PeerSSDReadWorker(CPUSSDDiskTransferWorker):
    """PEERSSD2H / H2PEERSSD: this node's CPU pool and SSD layout (the peer's
    is the same by the cluster's geometry), a :class:`RemoteSSDReader` for
    the bytes."""

    def __init__(self, *args: Any, pool_id: PoolId = PoolId.FULL_KV,
                 ssd_server_name: str = "", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.pool_id = pool_id
        self.ssd_server_name = ssd_server_name
        self._reader = _load_reader(self, pool_id)

    def read_local_layout(self, src_ssd_blocks: torch.Tensor, dst_cpu_blocks: torch.Tensor,
                          layer_id: int, layer_end: int) -> None:
        """A DISK2H with this worker's addressing against ``self.ioctx``."""
        gran = -1 if layer_end >= self.num_layers else layer_end - layer_id
        self._transfer_impl(src_ssd_blocks, dst_cpu_blocks, TransferType.DISK2H,
                            layer_id=layer_id, layer_granularity=gran)

    def write_local_layout(self, src_cpu_blocks: torch.Tensor,
                           dst_ssd_blocks: torch.Tensor) -> None:
        """An H2DISK of whole blocks with this worker's addressing against ``self.ioctx``."""
        self._transfer_impl(src_cpu_blocks, dst_ssd_blocks, TransferType.H2DISK,
                            layer_id=0, layer_granularity=-1)

    def _launch_write(self, transfer_op: WorkerTransferOp) -> WorkerTransferResult:
        """H2PEERSSD: ``src_block_node_ids`` names the peer. A failure is
        reported per block, not raised: the PUT has its CPU copy, and the plan
        gives the peer's slots back."""
        src, dst = self.get_transfer_block_ids(transfer_op)
        ranks = transfer_op.src_block_node_ids
        n = len(src)
        start = time.time()
        try:
            if ranks is None or len(ranks) != n:
                raise RuntimeError("H2PEERSSD op carries no per-block target rank")
            peers = torch.unique(torch.as_tensor(ranks, dtype=torch.int64)).tolist()
            if len(peers) != 1:
                raise RuntimeError(f"H2PEERSSD op spans ranks {peers}")
            write = getattr(self._reader, "write", None)
            if write is None:
                raise RuntimeError(f"{type(self._reader).__name__} cannot write a peer's SSD")
            write(int(peers[0]), self.pool_id, src, dst)
        except Exception as e:  # noqa: BLE001
            flexkv_logger.warning(
                f"[peer-ssd] H2PEERSSD of {n} blocks failed: {e!r}",
                exc_info=not getattr(e, "quiet", False))
            return WorkerTransferResult(transfer_op.transfer_op_id, (False,) * n)
        self._log_transfer_performance(
            transfer_op, int(self.block_stride_in_bytes) * transfer_op.valid_block_num,
            start, time.time())
        return WorkerTransferResult(transfer_op.transfer_op_id, (True,) * n)

    def launch_transfer(self, transfer_op: WorkerTransferOp):
        if transfer_op.transfer_type == TransferType.H2PEERSSD:
            return self._launch_write(transfer_op)
        src, dst = self.get_transfer_block_ids(transfer_op)
        ranks = transfer_op.src_block_node_ids
        if ranks is None or len(ranks) != len(src):
            raise RuntimeError("PEERSSD2H op carries no per-block source rank")
        ranks = torch.as_tensor(ranks, dtype=torch.int64)
        layer_id = int(transfer_op.layer_id or 0)
        gran = int(transfer_op.layer_granularity)
        layer_end = self.num_layers if gran < 0 else min(layer_id + gran, self.num_layers)
        start = time.time()
        for rank in torch.unique(ranks).tolist():
            sel = ranks == rank
            self._reader.read(int(rank), self.pool_id, src[sel], dst[sel], layer_id, layer_end)
        end = time.time()
        self._log_transfer_performance(
            transfer_op, int(self.block_stride_in_bytes) * transfer_op.valid_block_num,
            start, end)
        return True
