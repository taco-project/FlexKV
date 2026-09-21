# SPDX-License-Identifier: Apache-2.0
# cython: boundscheck=True, wraparound=True
"""The radixshmem-mode configuration file (``FLEXKV_RADIXSHMEM_CONFIG_PATH``).

One YAML, identical on every node of a cluster, with five sections:

  cluster / data / index / server
      Passed through by key to ``shmradix.ClusterConfig`` /
      ``DataPlaneConfig`` / ``IndexConfig`` / ``RadixServerConfig``. Keys are
      validated against the dataclass fields of the installed shmradix, so a
      new radixshmem field is configurable without a FlexKV change and a typo
      fails at startup. Geometry fields (slot counts, slot bytes, alignment,
      shm names) are derived from ``CacheConfig`` and rejected here.
  client
      FlexKV's own RadixClient / prefetch settings.

Per-node values do not belong in a global file: ``cluster.node_name`` and
``cluster.rpc_address`` are rejected. A node derives its identity from the IP
``cluster.rpc_interface`` resolves to; the two environment variables
``FLEXKV_RADIX_NODE_NAME`` / ``FLEXKV_RADIX_RPC_ADDRESS`` override that for
several nodes on one host (tests).

``cluster.cluster_id`` is the only namespace: the etcd key prefix and, through
:meth:`RadixShmemConfig.local_id`, every shm / socket / IPC name on this host.

Reference: ``docs/radixshmem/config_zh.md``.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Any, Dict, Optional, Set, Tuple

import yaml

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

SECTIONS = ("cluster", "data", "index", "server", "client")

# Values FlexKV sets differently from radixshmem's own defaults; anything not
# listed takes the shmradix dataclass default. One more default depends on the
# geometry and is resolved in shm_radix_bootstrap.build_radix_server_config:
# index.register_chunk_size = REGISTER_CHUNK_TOKENS // tokens_per_block, so an
# RHT registration chunk covers REGISTER_CHUNK_TOKENS tokens whatever the block
# size (radixshmem's own default is 128 blocks).
REGISTER_CHUNK_TOKENS = 4096

FLEXKV_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "cluster": {
        "cluster_id": "flexkv",
        "bootstrap_timeout_sec": 120,
        # 1 is a blind overwrite that loses routing entries.
        "rht_slots_per_bucket": 4,
    },
    "data": {},
    "index": {"data_pool_ratio": 8.0},
    "server": {},
}

# Derived from CacheConfig / ModelConfig (shm_radix_bootstrap.expected_geometry)
# or per node; rejected in the file.
FORBIDDEN_KEYS: Dict[str, Set[str]] = {
    "cluster": {"node_name", "rpc_address"},
    "data": {"data_bytes", "full_slot_bytes", "swa_slot_bytes", "mamba_slot_bytes",
             "slot_align", "data_name"},
    "index": {"name", "tokens_per_block", "full_slots", "swa_slots", "swa_window_blocks",
              "mamba_slots", "evict_policy"},
    "server": set(),
}

_RDMA_TRANSPORTS = {"xrc", "dc"}
_REMOTE_OP_TRANSPORTS = {"zmq", "dc"}
_RHT_SLOTS = {1, 2, 4, 8}


class RadixShmemConfigError(ValueError):
    """The file is not a valid radixshmem-mode configuration."""


@dataclasses.dataclass(frozen=True)
class RadixClientSettings:
    """FlexKV-side settings of the RadixClient and the prefetch path."""
    # Server-side deadline of one prefetch pull; the job completes with the
    # local hit when it expires.
    prefetch_timeout_ms: int = 5000
    # Peer pulls in flight per CE process before new prefetches skip the peer
    # walk; kept under max_outstanding so pull_async never blocks.
    prefetch_max_inflight: int = 128
    # Uncollected jobs one RadixClient may hold.
    max_outstanding: int = 256


@dataclasses.dataclass(frozen=True)
class RadixShmemConfig:
    path: Optional[str]
    cluster: Dict[str, Any]
    data: Dict[str, Any]
    index: Dict[str, Any]
    server: Dict[str, Any]
    client: RadixClientSettings = RadixClientSettings()

    # ----------------------------------------------------------- cluster
    @property
    def cluster_id(self) -> str:
        return str(self.cluster["cluster_id"])

    @property
    def node_name(self) -> str:
        return str(self.cluster.get("node_name", ""))

    @property
    def rpc_address(self) -> str:
        return str(self.cluster.get("rpc_address", ""))

    @property
    def expected_min_nodes(self) -> int:
        return int(self.cluster.get("expected_min_nodes", 0))

    @property
    def num_rht_shards(self) -> int:
        return int(self.cluster.get("num_rht_shards", 0))

    @property
    def distributed(self) -> bool:
        """radixshmem's own criterion (ClusterConfig.distributed)."""
        return self.expected_min_nodes > 1 or self.num_rht_shards > 1

    @property
    def bootstrap_timeout_sec(self) -> float:
        return float(self.cluster.get("bootstrap_timeout_sec", 60))

    @property
    def attach_timeout_s(self) -> float:
        """How long a FlexKV process waits for the radix-server: the cluster
        rendezvous plus a margin for SlotStore creation / prefault."""
        return self.bootstrap_timeout_sec + 60.0

    @property
    def local_id(self) -> str:
        """Prefix of every name on this host: the index / SlotStore shm, the
        gRPC socket, FlexKV's TE channels and GPU registration port. The
        cluster id, suffixed with the node name when one was given so that
        co-located nodes do not share regions."""
        return f"{self.cluster_id}_{self.node_name}" if self.node_name else self.cluster_id

    # ------------------------------------------------------------ server
    @property
    def endpoint(self) -> str:
        """gRPC endpoint; "" = radixshmem's unix:///dev/shm/<index>.sock."""
        return str(self.server.get("endpoint", ""))

    @property
    def hugepage_path(self) -> str:
        return str(self.server.get("hugepage_path", ""))

    # ------------------------------------------------------------- tests
    def replace_cluster(self, **changes: Any) -> "RadixShmemConfig":
        """A copy with ``cluster`` keys changed (test helper; bypasses the
        forbidden-key check so node_name / rpc_address can be set)."""
        return dataclasses.replace(self, cluster={**self.cluster, **changes})

    def replace_server(self, **changes: Any) -> "RadixShmemConfig":
        return dataclasses.replace(self, server={**self.server, **changes})

    def describe(self) -> str:
        where = self.path or "(defaults)"
        s = f"{where}: cluster_id={self.cluster_id}"
        if self.distributed:
            s += (f", expected_min_nodes={self.expected_min_nodes}, "
                  f"registry={self.cluster.get('registry')}, "
                  f"rpc_interface={self.cluster.get('rpc_interface') or '-'}, "
                  f"rpc_address={self.rpc_address or '-'}, node_name={self.node_name or '(auto)'}")
        return s


# ------------------------------------------------------------------ loading

def _shmradix_dataclasses():
    try:
        import shmradix
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "shmradix is not installed; install it from the radixshmem repo "
            "(pip install -e radixshmem/python)") from exc
    try:
        return {
            "cluster": shmradix.ClusterConfig,
            "data": shmradix.DataPlaneConfig,
            "index": shmradix.IndexConfig,
            "server": shmradix.RadixServerConfig,
        }
    except AttributeError as exc:
        raise ImportError(
            "shmradix lacks the RadixServer configuration dataclasses: FlexKV needs "
            "the RadixServer / RadixClient surface of radixshmem") from exc


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise RadixShmemConfigError(f"{path}: top level must be a mapping of sections")
    return loaded


def _section(raw: Dict[str, Any], name: str, path: str) -> Dict[str, Any]:
    sec = raw.get(name)
    if sec is None:
        return {}
    if not isinstance(sec, dict):
        raise RadixShmemConfigError(f"{path}: section '{name}' must be a mapping")
    return dict(sec)


def _as_list(value: Any) -> Any:
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def _passthrough_section(name: str, given: Dict[str, Any], dc, path: str) -> Dict[str, Any]:
    """FlexKV defaults overlaid with the file's keys, validated against the
    shmradix dataclass ``dc``."""
    fields = {f.name for f in dataclasses.fields(dc)}
    if name == "server":
        # RadixServerConfig's nested sections are configured by their own
        # sections here, not inline.
        fields -= {"index", "data", "cluster"}
    forbidden = FORBIDDEN_KEYS[name] & set(given)
    if forbidden:
        raise RadixShmemConfigError(
            f"{path}: '{name}.{sorted(forbidden)[0]}' is not configurable: "
            + ("geometry is derived from the FlexKV cache configuration"
               if name in ("data", "index") else
               "it is a per-node value; set FLEXKV_RADIX_NODE_NAME / "
               "FLEXKV_RADIX_RPC_ADDRESS on that node instead"))
    unknown = set(given) - fields
    if unknown:
        raise RadixShmemConfigError(
            f"{path}: unknown key(s) in '{name}': {sorted(unknown)}; "
            f"shmradix.{dc.__name__} has {sorted(fields)}")
    merged = {**FLEXKV_DEFAULTS[name], **given}
    if "transfer_devices" in merged:
        merged["transfer_devices"] = [str(d) for d in _as_list(merged["transfer_devices"])]
    if "rht_shard_holders" in merged:
        merged["rht_shard_holders"] = [int(r) for r in _as_list(merged["rht_shard_holders"])]
    return merged


def _client_section(given: Dict[str, Any], path: str) -> RadixClientSettings:
    fields = {f.name for f in dataclasses.fields(RadixClientSettings)}
    unknown = set(given) - fields
    if unknown:
        raise RadixShmemConfigError(
            f"{path}: unknown key(s) in 'client': {sorted(unknown)}; expected {sorted(fields)}")
    return RadixClientSettings(**{k: int(v) for k, v in given.items()})


def _validate(cfg: RadixShmemConfig, path: str) -> None:
    c = cfg.cluster
    if not cfg.cluster_id:
        raise RadixShmemConfigError(f"{path}: cluster.cluster_id must not be empty")
    if cfg.distributed:
        if not c.get("registry"):
            raise RadixShmemConfigError(
                f"{path}: cluster mode (expected_min_nodes > 1) needs cluster.registry, "
                f"e.g. 'etcd://10.0.0.1:2379'")
        if not c.get("rpc_interface") and not cfg.rpc_address:
            raise RadixShmemConfigError(
                f"{path}: cluster mode needs cluster.rpc_interface (the NIC whose IP peers "
                f"dial and this node's identity derives from) or FLEXKV_RADIX_RPC_ADDRESS")
        if cfg.rpc_address == "0.0.0.0":
            raise RadixShmemConfigError(
                "FLEXKV_RADIX_RPC_ADDRESS=0.0.0.0 gives every node the same identity; "
                "use this node's address")
    if cfg.expected_min_nodes > 0 and cfg.num_rht_shards > cfg.expected_min_nodes:
        raise RadixShmemConfigError(
            f"{path}: cluster.num_rht_shards={cfg.num_rht_shards} exceeds "
            f"expected_min_nodes={cfg.expected_min_nodes}; there cannot be more RHT shard "
            f"holders than nodes")
    slots = int(c.get("rht_slots_per_bucket", 1))
    if slots not in _RHT_SLOTS:
        raise RadixShmemConfigError(
            f"{path}: cluster.rht_slots_per_bucket={slots} must be one of {sorted(_RHT_SLOTS)}")
    for key in ("rht_transport", "peer_index_transport"):
        val = c.get(key)
        if val is not None and val not in _RDMA_TRANSPORTS:
            raise RadixShmemConfigError(
                f"{path}: cluster.{key}={val!r} must be one of {sorted(_RDMA_TRANSPORTS)}")
    rot = c.get("remote_op_transport")
    if rot is not None and rot not in _REMOTE_OP_TRANSPORTS:
        raise RadixShmemConfigError(
            f"{path}: cluster.remote_op_transport={rot!r} must be one of "
            f"{sorted(_REMOTE_OP_TRANSPORTS)}")
    if cfg.client.prefetch_max_inflight >= cfg.client.max_outstanding:
        raise RadixShmemConfigError(
            f"{path}: client.prefetch_max_inflight={cfg.client.prefetch_max_inflight} must be "
            f"below client.max_outstanding={cfg.client.max_outstanding}, or pull_async blocks")
    if cfg.client.prefetch_timeout_ms <= 0:
        raise RadixShmemConfigError(f"{path}: client.prefetch_timeout_ms must be > 0")


def load_radixshmem_config(path: Optional[str] = None,
                           *,
                           node_name: str = "",
                           rpc_address: str = "") -> RadixShmemConfig:
    """Parse ``path`` (None or "" = all defaults, i.e. standalone) and apply
    the per-node overrides. Raises :class:`RadixShmemConfigError` on an
    invalid file, ``ImportError`` without shmradix."""
    dcs = _shmradix_dataclasses()
    label = path or "(defaults)"
    raw = _read_yaml(path) if path else {}
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise RadixShmemConfigError(
            f"{label}: unknown section(s) {sorted(unknown)}; expected {list(SECTIONS)}")
    sections = {name: _passthrough_section(name, _section(raw, name, label), dcs[name], label)
                for name in ("cluster", "data", "index", "server")}
    if node_name:
        sections["cluster"]["node_name"] = str(node_name)
    if rpc_address:
        sections["cluster"]["rpc_address"] = str(rpc_address)
        # radixshmem lets the interface win over the address; an explicit
        # per-node address means the global interface must not apply here.
        sections["cluster"]["rpc_interface"] = ""
    cfg = RadixShmemConfig(path=path or None, client=_client_section(
        _section(raw, "client", label), label), **sections)
    _validate(cfg, label)
    return cfg


# -------------------------------------------------------------- singleton

_lock = threading.Lock()
_cached: Optional[Tuple[Tuple[str, str, str], RadixShmemConfig]] = None


def _env_key() -> Tuple[str, str, str]:
    env = GLOBAL_CONFIG_FROM_ENV
    return (str(env.radixshmem_config_path or ""), str(env.radix_node_name or ""),
            str(env.radix_rpc_address or ""))


def get_radixshmem_config() -> RadixShmemConfig:
    """The process's configuration: loaded from ``GLOBAL_CONFIG_FROM_ENV``
    (``FLEXKV_RADIXSHMEM_CONFIG_PATH`` + the two per-node overrides) on first
    use and whenever those three values change."""
    global _cached
    key = _env_key()
    with _lock:
        if _cached is None or _cached[0] != key:
            path, node_name, rpc_address = key
            _cached = (key, load_radixshmem_config(path or None, node_name=node_name,
                                                   rpc_address=rpc_address))
        return _cached[1]


def set_radixshmem_config(cfg: Optional[RadixShmemConfig]) -> None:
    """Install ``cfg`` as the process's configuration (tests); None reverts to
    loading from the environment."""
    global _cached
    with _lock:
        _cached = None if cfg is None else (_env_key(), cfg)
