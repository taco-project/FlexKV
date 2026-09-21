# SPDX-License-Identifier: Apache-2.0
# cython: boundscheck=True, wraparound=True
"""
Bootstrap for the radixshmem-backed CPU tier.

One ``radix-server`` process per node owns the radix index shm, the SlotStore
(the CPU KV pool: one slot per block, a FULL pool and optionally an SWA pool),
the RDMA transfer engine and the etcd data-plane entry. In this mode FlexKV
neither allocates CPU KV memory nor moves bytes between nodes itself:

  * the bootstrap DP process (instance 0, dp 0) launches the server from the
    FlexKV configuration -- geometry from ``CacheConfig``, everything else from
    the YAML at ``FLEXKV_RADIXSHMEM_CONFIG_PATH`` (``flexkv.common.radixshmem_config``)
    -- (``FLEXKV_RADIX_SERVER_LAUNCH_MODE=embedded``) or expects one started by
    the operator (``external``);
  * every DP scheduler process, the TE process and its transfer workers attach
    with ``shmradix.RadixClient(name)``: index operations, ``store`` (the
    SlotStore mapping) and ``pull_async`` (the server-side peer pull).

Naming: index ``/shmradix_<local_id>_cpu`` where ``local_id`` is the YAML's
``cluster.cluster_id`` (plus ``_<node_name>`` when FLEXKV_RADIX_NODE_NAME names
one of several co-located nodes). A cluster node's index gets ``_<node_name>``
appended by radixshmem itself and is resolved through the gRPC socket
``/dev/shm/shmradix_<local_id>_cpu.sock``, so attachers only need the base
name. The SlotStore is ``<index>_data``.

Geometry: FlexKV stays the source of slot counts and slot bytes. One FULL slot
holds exactly one CPU block as ``StorageEngine`` lays it out (BLOCKFIRST: all
layers of a block contiguous), one SWA slot one SWA page. ``slot_align`` is
chosen so that the SlotStore stride equals the block size exactly, which lets
the H2D / D2H workers address the pool with the strides of a plain tensor. The
TE re-checks the attached regions against the layouts it builds (``check_geometry``).
"""
from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
import signal
import time
from typing import Any, Dict, List, Optional

import torch

from flexkv.common.config import (GLOBAL_CONFIG_FROM_ENV, CacheConfig, LayerGroupSpec,
                                  ModelConfig, SWAPoolConfig)
from flexkv.common.debug import flexkv_logger
from flexkv.common.radixshmem_config import (REGISTER_CHUNK_TOKENS, RadixShmemConfig,
                                              get_radixshmem_config)
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

try:
    import shmradix
except ImportError:  # pragma: no cover
    shmradix = None


_SHM_PREFIX = "/shmradix"
# Pool bases are page aligned regardless; a larger per-slot alignment only pads.
_MAX_SLOT_ALIGN = 4096
DEFAULT_HUGETLBFS_DIR = "/mnt/hugepages"


def _ensure_shmradix() -> None:
    if shmradix is None:
        raise ImportError(
            "shmradix is not installed; install it from the radixshmem repo "
            "(pip install -e radixshmem/python)")
    for name in ("RadixServer", "RadixServerConfig", "IndexConfig", "DataPlaneConfig",
                 "ClusterConfig", "RadixClient"):
        if not hasattr(shmradix, name):
            raise ImportError(
                f"shmradix lacks {name}: FlexKV needs the RadixServer / RadixClient "
                f"surface of radixshmem (transfer-server branch or later)")


def radix_index_name(local_id: str) -> str:
    """Base shm name of the CPU tier's index for ``local_id``
    (``RadixShmemConfig.local_id``); it also names the gRPC socket."""
    return f"{_SHM_PREFIX}_{local_id}_cpu"


def radix_data_name(local_id: str) -> str:
    return radix_index_name(local_id) + "_data"


def radix_socket_path(local_id: str) -> str:
    return "/dev/shm/" + radix_index_name(local_id).lstrip("/").replace("/", "_") + ".sock"


# ------------------------------------------------------------------ geometry

def _resolve_groups(groups: Optional[List[LayerGroupSpec]],
                    default_dtype: torch.dtype) -> Optional[List[LayerGroupSpec]]:
    if groups is None:
        return None
    return [g if g.dtype is not None else dataclasses.replace(g, dtype=default_dtype)
            for g in groups]


def num_layers_per_pp_stage(model_config: ModelConfig, cache_config: CacheConfig) -> int:
    """Layers one CPU block covers: what the adapter recorded, else an even split."""
    recorded = int(getattr(cache_config, "_num_layers_per_pp_stage", 0) or 0)
    if recorded > 0:
        return recorded
    return max(1, model_config.num_layers // max(1, model_config.pp_size))


def layout_block_bytes(layout: KVCacheLayout, dtype: torch.dtype) -> int:
    """Bytes of one block of ``layout`` (a multi-group layout is byte-flat)."""
    if layout.layer_groups is not None:
        return int(layout.get_block_stride())
    return int(layout.get_block_stride()) * dtype.itemsize


def cpu_kv_layout(model_config: ModelConfig, cache_config: CacheConfig,
                  num_blocks: int) -> KVCacheLayout:
    """The CPU FULL layout exactly as ``StorageEngine`` builds it."""
    return KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
        num_layer=num_layers_per_pp_stage(model_config, cache_config),
        num_block=num_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads_per_node,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
        num_kv_heads=model_config.num_kv_heads,
        layer_groups=_resolve_groups(model_config.layer_groups, model_config.dtype),
        tp_size=model_config.tp_size,
    )


def cpu_block_bytes(model_config: ModelConfig, cache_config: CacheConfig) -> int:
    return layout_block_bytes(cpu_kv_layout(model_config, cache_config, 1), model_config.dtype)


def swa_pool_config(cache_config: CacheConfig) -> Optional[SWAPoolConfig]:
    swa = cache_config.swa
    if swa is None or not swa.enabled or swa.num_slots <= 0:
        return None
    return swa


def swa_cpu_kv_layout(model_config: ModelConfig, cache_config: CacheConfig,
                      num_blocks: int) -> KVCacheLayout:
    """The CPU SWA layout exactly as ``StorageEngine`` builds it (uint8, one page
    per slot; DSv4 sidecar groups come from ``SWAPoolConfig.layer_groups``)."""
    swa = cache_config.swa
    return KVCacheLayout(
        type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
        num_layer=swa.num_swa_layers,
        num_block=num_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=1,
        head_size=swa.bytes_per_token_per_layer,
        kv_dim=1,
        num_kv_heads=1,
        layer_groups=_resolve_groups(swa.layer_groups, torch.uint8),
        tp_size=model_config.tp_size,
    )


def swa_block_bytes(model_config: ModelConfig, cache_config: CacheConfig) -> int:
    return layout_block_bytes(swa_cpu_kv_layout(model_config, cache_config, 1), torch.uint8)


def slot_align_for(*sizes: int) -> int:
    """Largest power of two <= 4096 dividing every size. radixshmem rounds each
    slot's stride up to ``slot_align``, so this keeps stride == slot bytes."""
    align = _MAX_SLOT_ALIGN
    for size in sizes:
        if size <= 0:
            continue
        while size % align:
            align //= 2
    return max(align, 1)


@dataclasses.dataclass(frozen=True)
class RadixGeometry:
    """What FlexKV expects the server's regions to look like."""
    tokens_per_block: int
    full_slots: int
    full_slot_bytes: int
    swa_slots: int = 0
    swa_slot_bytes: int = 0
    swa_window_blocks: int = 0

    @property
    def slot_align(self) -> int:
        return slot_align_for(self.full_slot_bytes, self.swa_slot_bytes)

    @property
    def data_bytes(self) -> int:
        return self.full_slots * self.full_slot_bytes + self.swa_slots * self.swa_slot_bytes

    def describe(self) -> str:
        s = (f"tokens_per_block={self.tokens_per_block}, FULL {self.full_slots} x "
             f"{self.full_slot_bytes} B")
        if self.swa_slots:
            s += (f", SWA {self.swa_slots} x {self.swa_slot_bytes} B "
                  f"(window {self.swa_window_blocks})")
        return s + f", slot_align={self.slot_align}, data={self.data_bytes / 2**30:.2f} GiB"


def expected_geometry(model_config: ModelConfig, cache_config: CacheConfig) -> RadixGeometry:
    if GLOBAL_CONFIG_FROM_ENV.cpu_layout_type != KVCacheLayoutType.BLOCKFIRST:
        raise ValueError(
            "radixshmem needs FLEXKV_CPU_LAYOUT=BLOCKFIRST: one SlotStore slot is one "
            "contiguous block, which LAYERFIRST does not give")
    if cache_config.num_cpu_blocks <= 0:
        raise ValueError(f"cache_config.num_cpu_blocks={cache_config.num_cpu_blocks} must be > 0")
    geo = RadixGeometry(
        tokens_per_block=cache_config.tokens_per_block,
        full_slots=int(cache_config.num_cpu_blocks),
        full_slot_bytes=cpu_block_bytes(model_config, cache_config),
    )
    swa = swa_pool_config(cache_config)
    if swa is not None:
        if swa.window_blocks < 1:
            raise ValueError(
                f"cache_config.swa.window_blocks={swa.window_blocks} must be >= 1")
        if swa.num_slots < swa.window_blocks:
            # All-or-none window allocation: a pool smaller than one window can
            # never store anything, so fail at startup.
            raise ValueError(
                f"cache_config.swa.num_slots={swa.num_slots} cannot hold one "
                f"{swa.window_blocks}-block SWA window; raise num_slots or disable SWA")
        geo = dataclasses.replace(
            geo, swa_slots=int(swa.num_slots),
            swa_slot_bytes=swa_block_bytes(model_config, cache_config),
            swa_window_blocks=int(swa.window_blocks))
    return geo


def check_geometry(client: "shmradix.RadixClient", expected: RadixGeometry,
                   label: str = "radixshmem") -> None:
    """Fail closed when the attached regions differ from FlexKV's own layout: a
    slot count or stride mismatch would otherwise become a silent misaddressed
    transfer."""
    g = client.geometry
    pools = g["pools"]
    diffs: List[str] = []
    if int(g["block_size"]) != expected.tokens_per_block:
        diffs.append(f"tokens_per_block server={g['block_size']} flexkv={expected.tokens_per_block}")
    full = pools["full"]
    if int(full["num_slots"]) != expected.full_slots:
        diffs.append(f"FULL slots server={full['num_slots']} flexkv={expected.full_slots}")
    if int(full["slot_bytes"]) != expected.full_slot_bytes:
        diffs.append(f"FULL slot_bytes server={full['slot_bytes']} flexkv={expected.full_slot_bytes}")
    if not client.info.data_plane:
        diffs.append("server is index-only (no SlotStore); FlexKV needs the data plane")
    else:
        store = client.store
        stride = int(store.pool(shmradix.ComponentType.FULL).slot_bytes)
        if stride != expected.full_slot_bytes:
            diffs.append(f"FULL stride server={stride} flexkv={expected.full_slot_bytes} "
                         f"(slot_align must divide the block size)")
    swa = pools.get("swa")
    if expected.swa_slots > 0:
        if swa is None:
            diffs.append("server has no SWA pool but FlexKV's SWA tier is on")
        else:
            if int(swa["num_slots"]) != expected.swa_slots:
                diffs.append(f"SWA slots server={swa['num_slots']} flexkv={expected.swa_slots}")
            if int(swa["slot_bytes"]) != expected.swa_slot_bytes:
                diffs.append(f"SWA slot_bytes server={swa['slot_bytes']} flexkv={expected.swa_slot_bytes}")
            if int(swa.get("window_blocks", 0)) != expected.swa_window_blocks:
                diffs.append(f"SWA window server={swa.get('window_blocks')} "
                             f"flexkv={expected.swa_window_blocks}")
            if client.info.data_plane:
                stride = int(client.store.pool(shmradix.ComponentType.SWA).slot_bytes)
                if stride != expected.swa_slot_bytes:
                    diffs.append(f"SWA stride server={stride} flexkv={expected.swa_slot_bytes}")
    elif swa is not None:
        diffs.append("server has an SWA pool that FlexKV's configuration does not")
    if diffs:
        raise ValueError(
            f"{label}: the attached radixshmem regions do not match FlexKV's configuration "
            f"({expected.describe()}): " + "; ".join(diffs))


# ------------------------------------------------------------- server config

def build_radix_server_config(model_config: ModelConfig,
                              cache_config: CacheConfig,
                              rcfg: Optional[RadixShmemConfig] = None,
                              ) -> "shmradix.RadixServerConfig":
    """The one radix-server this FlexKV node needs: index sized from
    ``cache_config`` (FULL slots = ``num_cpu_blocks``, SWA slots = ``swa.num_slots``),
    SlotStore sized so that every slot is exactly one block, and the cluster /
    data-plane / index / server settings of ``rcfg`` (default: the process's
    ``FLEXKV_RADIXSHMEM_CONFIG_PATH``) passed through. Raises on an
    inconsistent configuration."""
    _ensure_shmradix()
    if rcfg is None:
        rcfg = get_radixshmem_config()
    geo = expected_geometry(model_config, cache_config)
    index_kwargs = dict(rcfg.index)
    # an RHT registration chunk covers REGISTER_CHUNK_TOKENS tokens unless the file says otherwise
    index_kwargs.setdefault("register_chunk_size",
                            max(1, REGISTER_CHUNK_TOKENS // geo.tokens_per_block))
    index = shmradix.IndexConfig(
        name=radix_index_name(rcfg.local_id),
        tokens_per_block=geo.tokens_per_block,
        full_slots=geo.full_slots,
        swa_slots=geo.swa_slots,
        swa_window_blocks=geo.swa_window_blocks,
        **index_kwargs,
    )
    data = shmradix.DataPlaneConfig(
        data_bytes=geo.data_bytes,
        full_slot_bytes=geo.full_slot_bytes,
        swa_slot_bytes=geo.swa_slot_bytes,
        slot_align=geo.slot_align,
        data_name=radix_data_name(rcfg.local_id),
        **rcfg.data,
    )
    cluster = shmradix.ClusterConfig(**rcfg.cluster)
    server_kwargs = dict(rcfg.server)
    if not server_kwargs.get("hugepage_path") and cache_config.use_hugepage_cpu_buffer:
        server_kwargs["hugepage_path"] = os.environ.get("FLEXKV_HUGETLBFS_DIR",
                                                        DEFAULT_HUGETLBFS_DIR)
    cfg = shmradix.RadixServerConfig(index=index, data=data, cluster=cluster, **server_kwargs)
    flexkv_logger.info(
        f"radixshmem server config for {index.name}: {geo.describe()}, "
        f"{rcfg.describe()}, register_chunk_size={index.register_chunk_size}, "
        f"hugepage_path={cfg.hugepage_path or '(shm)'}, "
        f"prefault={data.prefault}, transfer_devices={data.transfer_devices or '(all)'}")
    return cfg


# ------------------------------------------------------------ server process

def _radix_server_main(cfg, ready, stop, conn) -> None:
    """Body of the radix-server subprocess: bring the server up, report, wait."""
    import shmradix as _shmradix

    def _on_term(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    try:
        server = _shmradix.RadixServer(cfg)
        server.start()
    except BaseException as e:  # noqa: BLE001 - reported to the parent, which raises
        try:
            conn.send(("error", f"{type(e).__name__}: {e}"))
        finally:
            conn.close()
        return
    index = server.index
    try:
        conn.send(("ready", {
            "index_name": index.shm_name(),
            "rank": int(index.rank()),
            "world_size": int(index.world_size()),
            "distributed": bool(index.is_distributed()),
        }))
    finally:
        conn.close()
    ready.set()
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


class RadixServerProcess:
    """The embedded radix-server: a spawned subprocess running ``RadixServer``.

    Not the scheduler process (its gRPC threads and transfer polling thread
    would contend for the GIL, and a clustered server holds RDMA contexts that
    do not survive a fork) and not the TE process (which cannot come up before
    the GPU registrations, while the CEs attach the index at construction).
    """

    def __init__(self, cfg: "shmradix.RadixServerConfig"):
        self.cfg = cfg
        self._ctx = mp.get_context("spawn")
        self._ready = self._ctx.Event()
        self._stop = self._ctx.Event()
        self.process = None
        self.info: Dict[str, Any] = {}

    def start(self, timeout_s: Optional[float] = None) -> "RadixServerProcess":
        if timeout_s is None:
            timeout_s = float(self.cfg.cluster.bootstrap_timeout_sec) + 60.0
        parent, child = self._ctx.Pipe(duplex=False)
        self.process = self._ctx.Process(
            target=_radix_server_main,
            args=(self.cfg, self._ready, self._stop, child),
            name="flexkv-radix-server",
            daemon=True,
        )
        self.process.start()
        child.close()
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                if parent.poll(0.2):
                    kind, payload = parent.recv()
                    break
                if not self.process.is_alive():
                    raise RuntimeError("radix-server exited during startup (see its log)")
                if time.monotonic() > deadline:
                    self.shutdown()
                    raise TimeoutError(
                        f"radix-server did not become ready within {timeout_s:.0f}s "
                        f"(cluster rendezvous or SlotStore prefault still pending?)")
        finally:
            parent.close()
        if kind == "error":
            self.shutdown()
            raise RuntimeError(f"radix-server failed to start: {payload}")
        self.info = payload
        flexkv_logger.info(
            f"radix-server pid={self.process.pid} ready: index={payload['index_name']} "
            f"rank={payload['rank']}/{payload['world_size']} distributed={payload['distributed']}")
        return self

    @property
    def cluster_rank(self) -> int:
        return int(self.info.get("rank", 0))

    def shutdown(self, timeout: float = 15.0) -> None:
        if self.process is None:
            return
        self._stop.set()
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5.0)
        self.process = None


# ------------------------------------------------------------------- attach

def attach_radix_client(name: str,
                        timeout_s: Optional[float] = None,
                        *,
                        rcfg: Optional[RadixShmemConfig] = None,
                        max_outstanding: Optional[int] = None) -> "shmradix.RadixClient":
    """``shmradix.RadixClient(name)``, retried until the server's socket exists
    and the server is ready (an embedded server starts concurrently with the
    CEs, an external one may still be rendezvousing). Endpoint, timeout and
    ``max_outstanding`` default to ``rcfg`` (the process's configuration).

    The returned client owns the index attach, the SlotStore mapping and the
    gRPC channel; keep it alive for as long as its slots are addressed.
    """
    _ensure_shmradix()
    if rcfg is None:
        rcfg = get_radixshmem_config()
    if timeout_s is None:
        timeout_s = rcfg.attach_timeout_s
    if max_outstanding is None:
        max_outstanding = rcfg.client.max_outstanding
    deadline = time.monotonic() + timeout_s
    last: Optional[BaseException] = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"radix-server {name} not attachable within {timeout_s:.0f}s: {last}")
        try:
            return shmradix.RadixClient(name, endpoint=rcfg.endpoint or None,
                                        timeout_s=max(1.0, remaining),
                                        max_outstanding=max_outstanding)
        except Exception as e:  # noqa: BLE001 - socket not there yet, server starting
            last = e
            flexkv_logger.debug(f"attach to radix-server {name} failed (will retry): {e}")
            time.sleep(0.2)


def radix_cluster_rank(client: "shmradix.RadixClient") -> int:
    """The cluster rank etcd assigned this node (0 when standalone)."""
    rank = int(getattr(client.info, "rank", -1))
    return rank if rank >= 0 else int(client.rank())
