# SPDX-License-Identifier: Apache-2.0
# cython: boundscheck=True, wraparound=True
"""
Attaching the radixshmem-backed CPU tier to this node's radix-server.

The radix-server is a process the operator starts on every node, with nothing
model-specific on its command line::

    radix-server --name /flexkv --data-bytes 64G [--swa-ratio 0.5] [cluster flags]

It owns the radix index shm, the SlotStore (the CPU KV pool), the RDMA transfer
engine and the etcd membership. FlexKV neither creates it nor sizes it:

  * the first FlexKV client hands the server FlexKV's *geometry* -- tokens per
    block, bytes of one CPU block, bytes of one SWA page and the SWA window,
    the slot alignment that keeps the SlotStore stride equal to the block
    (``RadixGeometry``, ``expected_geometry``); the server plans the slot
    COUNTS from its byte budget and publishes them (idempotent: every further
    client with the same geometry is accepted, one with another geometry is
    refused);
  * every FlexKV process (the DP schedulers, the TE and its workers) attaches by
    the server's name (``attach_radix_client``), checks the published regions
    against its own layout (``check_geometry``) and takes the slot counts over
    into ``CacheConfig`` (``adopt_geometry``): ``num_cpu_blocks`` and
    ``swa.num_slots`` are the server's, not ``cpu_cache_gb``'s.

Geometry: one FULL slot holds exactly one CPU block as ``StorageEngine`` lays
it out (BLOCKFIRST: all layers of a block contiguous), one SWA slot one SWA
page. ``slot_align`` is chosen so that the SlotStore stride equals the block
size exactly, which lets the H2D / D2H workers address the pool with the
strides of a plain tensor.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional

import torch

from flexkv.common.config import (GLOBAL_CONFIG_FROM_ENV, CacheConfig, LayerGroupSpec,
                                  ModelConfig, SWAPoolConfig)
from flexkv.common.debug import flexkv_logger
from flexkv.common.radixshmem_config import (RadixShmemConfig, default_endpoint,
                                             get_radixshmem_config)
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

try:
    import shmradix
except ImportError:  # pragma: no cover
    shmradix = None


# Pool bases are page aligned regardless; a larger per-slot alignment only pads.
_MAX_SLOT_ALIGN = 4096


def _ensure_shmradix() -> None:
    if shmradix is None:
        raise ImportError(
            "shmradix is not installed; install it from the radixshmem repo "
            "(pip install -e radixshmem/python)")
    for name in ("RadixClient", "Geometry", "GeometryMismatch", "ServerNotReady"):
        if not hasattr(shmradix, name):
            raise ImportError(
                f"shmradix lacks {name}: FlexKV needs a radixshmem whose radix-server takes "
                f"its geometry from the client (RadixClient(name, Geometry))")


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
    """The SWA tier FlexKV wants on the server, or None. The slot count is the
    server's business (it may still be the placeholder ``cpu_cache_gb`` gave)."""
    swa = cache_config.swa
    if swa is None or not swa.enabled:
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
    """FlexKV's side of the geometry: what one slot of each pool must hold. The
    slot counts are not here; the server plans them from its byte budget. Nor
    is the RHT registration chunk unless pinned: ``register_chunk_tokens`` 0
    leaves it to the server's ``--register-chunk-tokens`` (radixshmem's default,
    4096 tokens) and FlexKV adopts the published value (:func:`adopt_geometry`,
    :func:`register_chunk_blocks`)."""
    tokens_per_block: int
    full_slot_bytes: int
    swa_slot_bytes: int = 0
    swa_window_blocks: int = 0
    # RHT registration granularity in tokens; 0 = the server's --register-chunk-tokens.
    register_chunk_tokens: int = 0

    @property
    def has_swa(self) -> bool:
        return self.swa_slot_bytes > 0

    @property
    def slot_align(self) -> int:
        return slot_align_for(self.full_slot_bytes, self.swa_slot_bytes)

    def to_shmradix(self) -> "shmradix.Geometry":
        """The ``shmradix.Geometry`` handed to the server (data mode: bytes and
        window only, no counts)."""
        _ensure_shmradix()
        return shmradix.Geometry(
            block_size=int(self.tokens_per_block),
            full_slot_bytes=int(self.full_slot_bytes),
            swa_slot_bytes=int(self.swa_slot_bytes),
            swa_window_blocks=int(self.swa_window_blocks) if self.has_swa else 0,
            slot_align=int(self.slot_align),
            register_chunk_tokens=int(self.register_chunk_tokens),
        )

    def describe(self) -> str:
        s = f"tokens_per_block={self.tokens_per_block}, FULL slot {self.full_slot_bytes} B"
        if self.has_swa:
            s += f", SWA slot {self.swa_slot_bytes} B (window {self.swa_window_blocks})"
        s += f", slot_align={self.slot_align}"
        if self.register_chunk_tokens:
            s += f", register_chunk_tokens={self.register_chunk_tokens}"
        return s


def register_chunk_blocks(register_chunk_tokens: int, tokens_per_block: int) -> int:
    """The RHT registration chunk in blocks, by radixshmem's rule
    (``ShmConfig::effective_register_chunk_size``): ``register_chunk_tokens //
    block_size``, at least 1. 0 tokens = no chunk alignment."""
    tokens = int(register_chunk_tokens)
    if tokens <= 0:
        return 0
    return max(1, tokens // max(1, int(tokens_per_block)))


def expected_geometry(model_config: ModelConfig, cache_config: CacheConfig) -> RadixGeometry:
    if GLOBAL_CONFIG_FROM_ENV.cpu_layout_type != KVCacheLayoutType.BLOCKFIRST:
        raise ValueError(
            "radixshmem needs FLEXKV_CPU_LAYOUT=BLOCKFIRST: one SlotStore slot is one "
            "contiguous block, which LAYERFIRST does not give")
    geo = RadixGeometry(
        tokens_per_block=int(cache_config.tokens_per_block),
        full_slot_bytes=cpu_block_bytes(model_config, cache_config),
    )
    swa = swa_pool_config(cache_config)
    if swa is not None:
        if swa.window_blocks < 1:
            raise ValueError(
                f"cache_config.swa.window_blocks={swa.window_blocks} must be >= 1")
        geo = dataclasses.replace(
            geo, swa_slot_bytes=swa_block_bytes(model_config, cache_config),
            swa_window_blocks=int(swa.window_blocks))
    return geo


# -------------------------------------------------------------------- attach

def attach_radix_client(name: Optional[str] = None,
                        *,
                        geometry: Any = None,
                        rcfg: Optional[RadixShmemConfig] = None,
                        endpoint: Optional[str] = None,
                        timeout_s: Optional[float] = None,
                        max_outstanding: Optional[int] = None,
                        label: str = "radixshmem") -> "shmradix.RadixClient":
    """A ready ``shmradix.RadixClient`` on the radix-server ``name`` (default:
    the configuration's ``server.name``).

    With ``geometry`` (a :class:`RadixGeometry` or a ``shmradix.Geometry``) the
    client hands the server FlexKV's slot shape on the way; the server plans
    the counts from its budget. That is idempotent, so every FlexKV process
    may bring it; a server already serving another geometry (another model or
    page size on this node) is refused, and so is one whose budget cannot hold
    FlexKV's slots. Without a geometry the call only waits for a server that
    somebody else configured.

    Retries while the server is not reachable yet (the operator may start it
    late), then blocks in ``wait_ready`` -- the rendezvous of a cluster and
    the SlotStore prefault happen there -- for ``timeout_s`` in total
    (default: the configuration's ``server.ready_timeout_s``).
    """
    _ensure_shmradix()
    if rcfg is None:
        rcfg = get_radixshmem_config()
    name = name or rcfg.server_name
    if endpoint is None:
        endpoint = rcfg.endpoint or None
    if timeout_s is None:
        timeout_s = rcfg.ready_timeout_s
    if max_outstanding is None:
        max_outstanding = rcfg.client.max_outstanding
    spec = geometry.to_shmradix() if isinstance(geometry, RadixGeometry) else geometry
    where = endpoint or default_endpoint(name)

    deadline = time.monotonic() + float(timeout_s)
    last: Optional[BaseException] = None
    while True:
        try:
            client = shmradix.RadixClient(name, spec, endpoint=endpoint,
                                          max_outstanding=max_outstanding)
            break
        except shmradix.GeometryMismatch as e:
            raise ValueError(
                f"{label}: radix-server {name} already serves another geometry ({e}); every "
                f"engine attached to one server must run the same model, page size and SWA "
                f"configuration") from e
        except ValueError as e:
            raise ValueError(
                f"{label}: radix-server {name} cannot serve FlexKV's geometry ({e}); check its "
                f"--data-bytes / --swa-ratio") from e
        except Exception as e:  # noqa: BLE001 - not reachable yet: no socket, no listener
            last = e
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{label}: no radix-server named {name} reachable at {where} within "
                    f"{timeout_s:.0f}s (start it with `radix-server --name {name} "
                    f"--data-bytes ...`): {last}") from e
            flexkv_logger.debug(f"{label}: radix-server {name} not reachable yet ({e}); retrying")
            time.sleep(0.5)

    remaining = max(1.0, deadline - time.monotonic())
    try:
        info = client.wait_ready(remaining)
    except TimeoutError as e:
        mode, err = client.info.mode, client.info.last_error
        client.close()
        raise TimeoutError(
            f"{label}: radix-server {name} not ready within {timeout_s:.0f}s (mode={mode}"
            + (f", last_error={err!r}" if err else "")
            + ("; nobody handed it a geometry" if spec is None and mode == "waiting" else "")
            + ")") from e
    except RuntimeError as e:
        client.close()
        raise RuntimeError(f"{label}: radix-server {name} failed to configure: {e}") from e
    flexkv_logger.info(
        f"{label}: attached radix-server {name} ({where}): index={info.index_name}, "
        f"rank={info.rank}/{info.world_size}, data_plane={info.data_plane}, "
        f"geometry={_describe_published(info.geometry)}")
    return client


def _describe_published(g: Optional[Dict[str, Any]]) -> str:
    if not g:
        return "(none)"
    parts = [f"block_size={g.get('block_size')}",
             f"register_chunk_tokens={g.get('register_chunk_tokens', 0)}"]
    for kind, pool in (g.get("pools") or {}).items():
        s = f"{kind.upper()} {pool.get('num_slots')} x {pool.get('slot_bytes')} B"
        if kind == "swa":
            s += f" (window {pool.get('window_blocks')})"
        parts.append(s)
    return ", ".join(parts)


def _published_geometry(client: "shmradix.RadixClient", label: str) -> Dict[str, Any]:
    g = client.geometry
    if not g:
        info = client.status()
        g = info.geometry
        if not g:
            raise ValueError(
                f"{label}: radix-server {client.name} has no geometry yet (mode={info.mode}); "
                f"attach with FlexKV's geometry first")
    return g


def check_geometry(client: "shmradix.RadixClient", expected: RadixGeometry,
                   label: str = "radixshmem") -> None:
    """Fail closed when the server's regions differ from FlexKV's own layout: a
    stride or page mismatch would otherwise become a silent misaddressed
    transfer. Slot counts are not checked here -- they are the server's, taken
    over by :func:`adopt_geometry`."""
    _ensure_shmradix()
    g = _published_geometry(client, label)
    pools = g["pools"]
    diffs: List[str] = []
    if int(g["block_size"]) != expected.tokens_per_block:
        diffs.append(f"tokens_per_block server={g['block_size']} flexkv={expected.tokens_per_block}")
    chunk_tokens = int(g.get("register_chunk_tokens", 0))
    if expected.register_chunk_tokens and chunk_tokens != expected.register_chunk_tokens:
        diffs.append(f"register_chunk_tokens server={chunk_tokens} "
                     f"flexkv={expected.register_chunk_tokens}")
    elif chunk_tokens % max(1, expected.tokens_per_block):
        flexkv_logger.warning(
            f"{label}: radix-server {client.name}'s register_chunk_tokens={chunk_tokens} is not a "
            f"multiple of tokens_per_block={expected.tokens_per_block}; the RHT registration "
            f"chunk is {register_chunk_blocks(chunk_tokens, expected.tokens_per_block)} blocks")
    full = pools["full"]
    if int(full["slot_bytes"]) != expected.full_slot_bytes:
        diffs.append(f"FULL slot_bytes server={full['slot_bytes']} flexkv={expected.full_slot_bytes}")
    if not client.info.data_plane:
        diffs.append("server is index-only (no --data-bytes); FlexKV needs the data plane")
    else:
        store = client.store
        stride = int(store.pool(shmradix.ComponentType.FULL).slot_bytes)
        if stride != expected.full_slot_bytes:
            diffs.append(f"FULL stride server={stride} flexkv={expected.full_slot_bytes} "
                         f"(the server rounds slots up to slot_align={g.get('slot_align')}; "
                         f"FlexKV asks for {expected.slot_align})")
    swa = pools.get("swa")
    if expected.has_swa:
        if swa is None:
            diffs.append("server has no SWA pool but FlexKV's SWA tier is on "
                         "(start it with --swa-ratio)")
        else:
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
            f"{label}: the radix-server's regions do not match FlexKV's layout "
            f"({expected.describe()}): " + "; ".join(diffs))


def adopt_geometry(cache_config: CacheConfig, client: "shmradix.RadixClient",
                   label: str = "radixshmem") -> Dict[str, int]:
    """Take over what the server planned and published: the slot counts into
    ``cache_config`` (``num_cpu_blocks`` = the FULL pool, ``swa.num_slots`` =
    the SWA pool; whatever ``cpu_cache_gb`` had produced was a placeholder in
    this mode) and the RHT registration chunk, which FlexKV does not bring
    itself: ``register_chunk_tokens`` is the server's ``--register-chunk-tokens``
    and ``register_chunk_blocks`` that in FlexKV blocks. Run it after any
    ``recompute_cache_block_counts`` in the same process (that recompute sizes
    from ``cpu_cache_gb`` and would undo this). Returns ``{"full": n, "swa": m,
    "register_chunk_tokens": t, "register_chunk_blocks": b}`` ("swa" only with
    an SWA tier)."""
    g = _published_geometry(client, label)
    pools = g["pools"]
    counts: Dict[str, int] = {"full": int(pools["full"]["num_slots"])}
    before = int(cache_config.num_cpu_blocks)
    cache_config.num_cpu_blocks = counts["full"]
    note = f"FULL {counts['full']} slots (cpu_cache_gb had given {before})"
    swa = swa_pool_config(cache_config)
    if swa is not None:
        published = pools.get("swa")
        if published is None:
            raise ValueError(
                f"{label}: FlexKV's SWA tier is on but radix-server {client.name} planned no SWA "
                f"pool; start it with --swa-ratio")
        counts["swa"] = int(published["num_slots"])
        swa_before = int(swa.num_slots)
        swa.num_slots = counts["swa"]
        note += f", SWA {counts['swa']} slots (had {swa_before})"
    counts["register_chunk_tokens"] = int(g.get("register_chunk_tokens", 0))
    counts["register_chunk_blocks"] = register_chunk_blocks(counts["register_chunk_tokens"],
                                                            int(g["block_size"]))
    note += (f"; RHT registration chunk {counts['register_chunk_tokens']} tokens = "
             f"{counts['register_chunk_blocks']} blocks")
    flexkv_logger.info(f"{label}: adopted radix-server {client.name}'s geometry: {note}")
    return counts


def adopt_radix_server(model_config: ModelConfig, cache_config: CacheConfig,
                       *,
                       rcfg: Optional[RadixShmemConfig] = None,
                       label: str = "radixshmem") -> Dict[str, int]:
    """Attach to this node's radix-server with FlexKV's geometry, take over the
    slot counts it planned (:func:`adopt_geometry`) and its cluster rank
    (``cache_config.distributed_node_id``), then detach. Every process that
    sizes something from ``cache_config`` runs this before it does: the
    KVManager before it starts a KVServer or a KVTaskEngine, the KVTaskEngine
    host itself (a KVServer may be started on its own) and, with a live client,
    the TE. Idempotent: the server accepts the same geometry any number of
    times."""
    geometry = expected_geometry(model_config, cache_config)
    client = attach_radix_client(rcfg=rcfg, geometry=geometry, label=label)
    try:
        counts = adopt_geometry(cache_config, client, label=label)
        cache_config.distributed_node_id = radix_cluster_rank(client)
        flexkv_logger.info(
            f"{label}: radix-server {client.name}: cluster rank "
            f"{cache_config.distributed_node_id}/{client.info.world_size}, "
            f"FlexKV geometry {geometry.describe()}")
        return counts
    finally:
        client.close()


def radix_server_is_distributed(rcfg: Optional[RadixShmemConfig] = None,
                                *,
                                timeout_s: Optional[float] = None,
                                label: str = "radixshmem") -> bool:
    """Whether this node's radix-server is part of a cluster (world_size > 1),
    i.e. whether peer pulls are possible. Asked of the server itself once it
    is ready (a geometry-less attach that only waits), so every process that
    asks gets the same answer -- the framework adapters gate the prefetch path
    on it in every TP rank, and the ranks must agree."""
    client = attach_radix_client(rcfg=rcfg, timeout_s=timeout_s, label=label)
    try:
        return int(client.info.world_size) > 1
    finally:
        client.close()


def radix_cluster_rank(client: "shmradix.RadixClient") -> int:
    """This node's rank in the radix cluster (0 on a standalone server)."""
    rank = int(getattr(client.info, "rank", -1))
    return rank if rank >= 0 else int(client.rank())
