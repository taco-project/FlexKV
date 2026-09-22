# SPDX-License-Identifier: Apache-2.0
# cython: boundscheck=True, wraparound=True
"""The radixshmem-mode configuration file (``FLEXKV_RADIXSHMEM_CONFIG_PATH``).

In radixshmem mode the CPU tier is a ``radix-server`` process the operator
starts on every node (``radix-server --name /flexkv --data-bytes 64G ...``):
the index shm, the SlotStore, the transfer engine and the cluster membership
all belong to that process and are set on its command line. FlexKV never
creates a server; it attaches a ``shmradix.RadixClient``. So this file holds
the two things FlexKV has to know, and nothing else:

  server
      Which radix-server to attach to: its ``--name`` (which also derives the
      default gRPC socket ``unix:///dev/shm/<name>.sock``), an ``endpoint``
      override, and how long a FlexKV process waits for the server to exist
      and become ready.
  client
      FlexKV's RadixClient / prefetch settings.

The slot geometry (tokens per block, bytes of one CPU block and one SWA page,
the SWA window) is derived from ``ModelConfig`` / ``CacheConfig`` and handed to
the server by FlexKV's clients (``flexkv.server.shm_radix_bootstrap``); the
slot COUNTS come back from the server, which plans them from its byte budget.
None of that is in this file. A file that still carries the former ``cluster``
/ ``data`` / ``index`` sections is rejected with a pointer to the
``radix-server`` flags they moved to.

No YAML at all is a valid configuration: it attaches to ``radix-server --name
/flexkv`` on the local socket.

Reference: ``docs/radixshmem/config_zh.md``.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Any, Dict, Optional, Tuple

import yaml

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

SECTIONS = ("server", "client")
# Sections of the previous file format. Their keys are radix-server flags now.
RETIRED_SECTIONS = ("cluster", "data", "index")

DEFAULT_SERVER_NAME = "/flexkv"


class RadixShmemConfigError(ValueError):
    """The file is not a valid radixshmem-mode configuration."""


def default_endpoint(server_name: str) -> str:
    """radixshmem's default gRPC socket for ``radix-server --name <server_name>``:
    ``unix:///dev/shm/<name without the leading slash, '/' -> '_'>.sock``."""
    return f"unix:///dev/shm/{server_name.lstrip('/').replace('/', '_')}.sock"


@dataclasses.dataclass(frozen=True)
class RadixServerSettings:
    """Which radix-server this node's FlexKV attaches to."""
    # ``radix-server --name``: the index shm name. Also the default socket
    # (``unix:///dev/shm/<name>.sock``, ``default_endpoint``).
    name: str = DEFAULT_SERVER_NAME
    # gRPC endpoint; "" = the default socket derived from ``name``.
    endpoint: str = ""
    # How long a FlexKV process waits for the server to be reachable AND ready.
    # Covers the operator starting it late, the SlotStore prefault and, on a
    # cluster, the rendezvous (the server's --bootstrap-timeout).
    ready_timeout_s: float = 600.0


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
    server: RadixServerSettings = RadixServerSettings()
    client: RadixClientSettings = RadixClientSettings()

    # ------------------------------------------------------------ server
    @property
    def server_name(self) -> str:
        return self.server.name

    @property
    def endpoint(self) -> str:
        """gRPC endpoint; "" = radixshmem's ``unix:///dev/shm/<name>.sock``."""
        return self.server.endpoint

    @property
    def ready_timeout_s(self) -> float:
        return float(self.server.ready_timeout_s)

    @property
    def default_endpoint(self) -> str:
        """The gRPC endpoint radixshmem derives from the server name when no
        ``endpoint`` is given: ``unix:///dev/shm/<name>.sock``."""
        return default_endpoint(self.server.name)

    # ------------------------------------------------------------- tests
    def replace_server(self, **changes: Any) -> "RadixShmemConfig":
        return dataclasses.replace(self, server=dataclasses.replace(self.server, **changes))

    def replace_client(self, **changes: Any) -> "RadixShmemConfig":
        return dataclasses.replace(self, client=dataclasses.replace(self.client, **changes))

    def describe(self) -> str:
        where = self.path or "(defaults)"
        return (f"{where}: radix-server {self.server_name} "
                f"(endpoint={self.endpoint or self.default_endpoint}, "
                f"ready_timeout_s={self.ready_timeout_s:.0f})")


# ------------------------------------------------------------------ loading

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


def _typed_section(name: str, given: Dict[str, Any], dc, path: str):
    """``dc(**given)`` after checking the keys and coercing the value types."""
    fields = {f.name: f for f in dataclasses.fields(dc)}
    unknown = set(given) - set(fields)
    if unknown:
        raise RadixShmemConfigError(
            f"{path}: unknown key(s) in '{name}': {sorted(unknown)}; expected {sorted(fields)}")
    values: Dict[str, Any] = {}
    for key, value in given.items():
        typ = fields[key].type
        try:
            if typ in ("int", int):
                values[key] = int(value)
            elif typ in ("float", float):
                values[key] = float(value)
            else:
                values[key] = "" if value is None else str(value)
        except (TypeError, ValueError) as exc:
            raise RadixShmemConfigError(f"{path}: '{name}.{key}' has an invalid value {value!r}") from exc
    return dc(**values)


def _validate(cfg: RadixShmemConfig, path: str) -> None:
    name = cfg.server_name
    if not name or not name.startswith("/") or len(name) < 2 or any(c.isspace() for c in name):
        raise RadixShmemConfigError(
            f"{path}: server.name={name!r} must be a shm name that starts with '/' "
            f"(the radix-server's --name, e.g. '/flexkv')")
    if cfg.ready_timeout_s <= 0:
        raise RadixShmemConfigError(f"{path}: server.ready_timeout_s must be > 0")
    if cfg.client.prefetch_max_inflight >= cfg.client.max_outstanding:
        raise RadixShmemConfigError(
            f"{path}: client.prefetch_max_inflight={cfg.client.prefetch_max_inflight} must be "
            f"below client.max_outstanding={cfg.client.max_outstanding}, or pull_async blocks")
    if cfg.client.prefetch_timeout_ms <= 0:
        raise RadixShmemConfigError(f"{path}: client.prefetch_timeout_ms must be > 0")


def load_radixshmem_config(path: Optional[str] = None) -> RadixShmemConfig:
    """Parse ``path`` (None or "" = all defaults: ``radix-server --name /flexkv``
    on the local socket). Raises :class:`RadixShmemConfigError` on an invalid
    file."""
    label = path or "(defaults)"
    raw = _read_yaml(path) if path else {}
    retired = [s for s in RETIRED_SECTIONS if s in raw]
    if retired:
        raise RadixShmemConfigError(
            f"{label}: section(s) {retired} are not FlexKV's any more: the radix-server owns "
            f"its cluster, data plane and index settings and takes them on its command line "
            f"(radix-server --data-bytes / --swa-ratio / --expected-min-nodes / --registry / "
            f"--transfer-dev ...). FlexKV only attaches to it; keep 'server' and 'client' here. "
            f"See docs/radixshmem/config_zh.md")
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise RadixShmemConfigError(
            f"{label}: unknown section(s) {sorted(unknown)}; expected {list(SECTIONS)}")
    cfg = RadixShmemConfig(
        path=path or None,
        server=_typed_section("server", _section(raw, "server", label), RadixServerSettings, label),
        client=_typed_section("client", _section(raw, "client", label), RadixClientSettings, label),
    )
    _validate(cfg, label)
    return cfg


# -------------------------------------------------------------- singleton

_lock = threading.Lock()
_cached: Optional[Tuple[Tuple[str], RadixShmemConfig]] = None


def _env_key() -> Tuple[str]:
    return (str(GLOBAL_CONFIG_FROM_ENV.radixshmem_config_path or ""),)


def get_radixshmem_config() -> RadixShmemConfig:
    """The process's configuration: loaded from ``GLOBAL_CONFIG_FROM_ENV``
    (``FLEXKV_RADIXSHMEM_CONFIG_PATH``) on first use and whenever that value
    changes."""
    global _cached
    key = _env_key()
    with _lock:
        if _cached is None or _cached[0] != key:
            _cached = (key, load_radixshmem_config(key[0] or None))
        return _cached[1]


def set_radixshmem_config(cfg: Optional[RadixShmemConfig]) -> None:
    """Install ``cfg`` as the process's configuration (tests); None reverts to
    loading from the environment."""
    global _cached
    with _lock:
        _cached = None if cfg is None else (_env_key(), cfg)
