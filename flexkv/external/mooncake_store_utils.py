"""
mooncake_store_utils.py
-----------------------
Adapter layer that connects FlexKV to the mooncake-store distributed KV
cache pool (``mooncake.store.MooncakeDistributedStore``).

Three public objects are exposed:

* ``MooncakeStoreConfig``  – plain config dataclass (mirrors ``CacheConfig``
  mooncake-store fields so the worker needs only one argument).
* ``MooncakeStoreClient`` – thin wrapper around
  ``MooncakeDistributedStore`` with FlexKV-friendly helpers.
* ``MooncakeStoreCacheEngine`` – presents a ``match()`` / ``insert()``
  interface compatible with ``CacheEngineAccel`` so it can be used as
  ``remote_cache_engine`` inside the ``KVCacheManager`` factory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Set, Union, Any, List, Tuple, TYPE_CHECKING

import base64
import time
import json
import torch
import requests
import uuid
import numpy as np

from flexkv.common.block import SequenceMeta
from flexkv.common.debug import flexkv_logger
from flexkv.common.type import MatchResultAccel

# from flexkv.common.config import CacheConfig
from flexkv.external.mooncake_store_keys import PoolKind, build_key

if TYPE_CHECKING:
    from flexkv.common.config import CacheConfig

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
SETUP_TIMEOUT = 600

# --- etcd-based master discovery (HA deployments) --------------------------
ETCD_DEFAULT_PORT = 2379
ETCD_REQUEST_TIMEOUT = 3
DEFAULT_CLUSTER_ID = "mooncake_cluster"
# Key the elected master writes itself into; ``cluster_id`` is substituted.
MASTER_VIEW_KEY_TEMPLATE = "mooncake-store/{cluster_id}/master_view"
# Fields to probe when master_view holds a JSON document instead of "host:port".
MASTER_VIEW_ADDRESS_FIELDS = (
    "master_address",
    "address",
    "rpc_address",
    "leader",
    "master",
)


def _split_host_port(addr: str) -> Tuple[str, str]:
    """Split ``host:port`` into its parts, tolerating IPv6 and bare hosts."""
    addr = addr.strip()
    if addr.startswith("["):  # [::1]:2379
        closing = addr.find("]")
        if closing != -1:
            host = addr[: closing + 1]
            rest = addr[closing + 1 :]
            return host, rest[1:] if rest.startswith(":") else ""
    if ":" in addr:
        host, _, port = addr.rpartition(":")
        return host, port
    return addr, ""


def _parse_master_view_value(raw: str) -> Optional[str]:
    """Extract the master host from a ``master_view`` value.

    The value is either a plain ``host:port`` string or a JSON document; both
    forms are accepted so the probe keeps working across mooncake versions.
    """
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        for field_name in MASTER_VIEW_ADDRESS_FIELDS:
            candidate = payload.get(field_name)
            if isinstance(candidate, str) and candidate.strip():
                raw = candidate.strip()
                break
        else:
            return None
    host, _ = _split_host_port(raw)
    return host or None


def _query_etcd_master_view(endpoint: str, key: str) -> Optional[str]:
    """Read ``key`` through the etcd v3 gRPC-gateway and return the master host.

    Uses the HTTP gateway instead of an etcd client library so the adapter
    keeps its current dependency set (``requests`` only).

    TODO(etcd-auth/tls): this probe is unauthenticated and plaintext-only. It
    works for the current deployment (etcd started with a plain ``http://``
    listener and no ``--auth``), but will BREAK in the following cases:

      * etcd with authentication enabled (``etcd --auth``): ``/v3/kv/range``
        returns HTTP 401, ``raise_for_status()`` throws, ``_resolve_master_host``
        swallows it and keeps returning None -> ``check_server`` waits until
        SETUP_TIMEOUT and then fails, looking exactly like "no master elected".
      * etcd behind TLS (``https://`` endpoint): the request either fails the
        certificate check or, with a bare ``verify=False``, silently trusts an
        unverified peer. mTLS (client-cert required) fails outright.

    Expected fix (implement only when a secured etcd is actually deployed):
      1. Extend ``MooncakeStoreConfig`` with non-secret fields: ``etcd_username``,
         ``etcd_ca_cert``, ``etcd_client_cert``, ``etcd_client_key``,
         ``etcd_tls_verify`` (read from the JSON config). The password/private
         key stay OUT of the JSON and are read from env (e.g.
         ``MC_STORE_ETCD_PASSWORD``) per the secrets-in-env baseline.
      2. TLS: pass ``verify=<ca_path or True>`` and ``cert=(client_cert,
         client_key)`` to every ``requests`` call. Default ``verify`` MUST stay
         truthy; never hardcode ``verify=False``.
      3. Token auth: POST ``{"name","password"}`` to ``/v3/auth/authenticate``,
         cache the returned token, send it as the ``Authorization`` header
         (etcd's gateway expects the bare token, no ``Bearer`` prefix), and on a
         401 re-authenticate once and retry (tokens are TTL-bound).
      All new fields default to the current unauthenticated/plaintext behaviour,
      so enabling them is opt-in and introduces no regression.
    """
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={"key": base64.b64encode(key.encode()).decode()},
        timeout=ETCD_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    kvs = response.json().get("kvs") or []
    if not kvs:
        return None
    # The gRPC-gateway omits empty fields, so ``value`` may be absent.
    value = base64.b64decode(kvs[0].get("value", "")).decode(errors="replace")
    return _parse_master_view_value(value)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MooncakeStoreConfig:
    """All settings required to initialise a ``MooncakeDistributedStore``."""

    master_addr: str = (
        ""  # Address of the Mooncake master / etcd endpoint, e.g. "192.168.1.1:2379".
    )

    metadata_server: str = (
        "P2PHANDSHAKE"  # Metadata server type string passed to the store SDK.
    )

    protocol: str = "rdma"  # Transport protocol: "rdma" or "tcp".

    device_name: str = (
        ""  # RDMA device name, e.g. "mlx5_0". Empty string means auto-select.
    )

    local_hostname: str = ""  # Local IP or hostname for RDMA buffer registration.

    global_segment_size: int = (
        256 * 1024 * 1024 * 1024
    )  # Size (bytes) of the memory segment registered with the store. Default 256 GiB.

    enable_ssd_offload: bool = False  # Enable SSD offload.

    ssd_offload_path: Optional[str] = None  # SSD offload path.

    master_metrics_port: int = 9003  # Master metrics port.

    # Payload byte limits per SDK call; zero preserves unbounded batching.
    # Provider-specific allocation overhead is deliberately not estimated here.
    max_get_batch_bytes: int = 0
    max_put_batch_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("max_get_batch_bytes", "max_put_batch_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @classmethod
    def from_file(
        cls, cache_config, override_global_segment_size: Optional[int] = None
    ) -> MooncakeStoreConfig:
        """Load MooncakeStoreConfig from JSON file.

        Parameters
        ----------
        cache_config:
            FlexKV CacheConfig instance (used to locate the config file path).
        override_global_segment_size:
            If provided, overrides the ``global_segment_size`` from the JSON file.
            Use ``0`` to create a *pure-client* instance that does NOT contribute
            any segment to the cluster pool.  This is useful for sidecar workers
            (e.g. indexer) which only need to read/write data into pre-registered
            buffers and do not need their own storage segment.
        """
        file_path = getattr(cache_config, "mooncake_store_config_path", None)
        if file_path is None:
            file_path = os.getenv("FLEXKV_MOONCAKE_STORE_CONFIG_PATH", None)
            if file_path is None:
                raise ValueError(
                    "Mooncake store config file path not found in cache config or "
                    "environment variable FLEXKV_MOONCAKE_STORE_CONFIG_PATH"
                )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Mooncake store config file not found: {file_path}"
            )
        with open(file_path) as f:
            config = json.load(f)

        global_segment_size = (
            override_global_segment_size
            if override_global_segment_size is not None
            else config["global_segment_size"] * 1024 * 1024 * 1024
        )
        flexkv_logger.info(
            "[MooncakeStoreConfig] global_segment_size for mooncake store is: %d GB",
            global_segment_size / 1024 / 1024 / 1024,
        )

        return cls(
            master_addr=config["master_addr"],
            metadata_server=config["metadata_server"],
            protocol=config["protocol"],
            device_name=config["device_name"],
            local_hostname=config["local_hostname"],
            global_segment_size=global_segment_size,
            enable_ssd_offload=config["enable_ssd_offload"],
            ssd_offload_path=config["ssd_offload_path"],
            master_metrics_port=config["master_metrics_port"],
            max_get_batch_bytes=config.get("max_get_batch_bytes", 0),
            max_put_batch_bytes=config.get("max_put_batch_bytes", 0),
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MooncakeStoreClient:
    """
    Thin wrapper around ``mooncake.store.MooncakeDistributedStore`` that adds
    FlexKV-specific helpers (register_buffer, batch_put, batch_get, batch_exists).

    The underlying store object is lazily created on first use to allow
    pickling of the config across process boundaries.
    """

    def __init__(self, config: MooncakeStoreConfig, query_only: bool = False) -> None:
        self._config = config
        self._store = None  # created lazily in setup()
        self._is_setup = False
        self._query_only = query_only
        self.setup()
        print("client setup done")
        if not self._query_only:
            self.check_server()
            print("client check server done")
        if not self._query_only:
            # only warm up when not in query only mode
            self.warm_up()

    def setup(self) -> None:
        """Initialise the underlying MooncakeDistributedStore.

        Must be called once in the worker process *before* any put/get.
        Importing ``mooncake.store`` is deferred here so that processes that
        do not use the backend are not affected by the import cost.
        """
        if self._is_setup:
            return
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "mooncake-store Python SDK not found. "
                "Install it with: pip install mooncake-store"
            ) from exc

        cfg = self._config
        protocol = "rpc_only" if self._query_only else cfg.protocol
        global_segment_size = 0 if self._query_only else cfg.global_segment_size
        local_segment_size = 0 if self._query_only else DEFAULT_LOCAL_BUFFER_SIZE
        self._store = MooncakeDistributedStore()
        ret_code = self._store.setup(
            cfg.local_hostname,  # 1: client_hostname
            cfg.metadata_server,  # 2: metadata_server
            global_segment_size,  # 3: global_segment_size
            local_segment_size,  # 4: local_buffer_size (16 MB SDK buffer)
            protocol,  # 5: protocol
            cfg.device_name,  # 6: device_name
            cfg.master_addr,  # 7: master_server_address
            # cfg.enable_ssd_offload,       # 8: enable ssd offload
            # cfg.ssd_offload_path,         # 9: ssd offload path
        )
        if ret_code:
            raise RuntimeError(
                f"[MooncakeStoreClient] MooncakeDistributedStore.setup() failed "
                f"with error code: {ret_code}"
            )
        self._is_setup = True
        flexkv_logger.info(
            "[MooncakeStoreClient] Store initialised: "
            f"master_addr={cfg.master_addr}, protocol={protocol}"
        )

    @staticmethod
    def _cluster_id() -> str:
        """Mirror the Mooncake SDK: the client only honours $MC_STORE_CLUSTER_ID
        (falling back to the default) and drops one trailing ``/`` when building
        the master_view key, so reading any other source would probe a key the
        SDK never uses."""
        cluster_id = os.getenv("MC_STORE_CLUSTER_ID", "") or DEFAULT_CLUSTER_ID
        if cluster_id.endswith("/"):
            cluster_id = cluster_id[:-1]
        return cluster_id

    def _etcd_endpoints(self) -> List[str]:
        """Return etcd base URLs when ``master_addr`` points at etcd, else [].

        Endpoints may be separated by ``;`` or ``,`` (same as the Mooncake etcd
        wrapper).

        TODO(ha-backend-detection): the Mooncake SDK only enters HA mode for a
        ``<backend>://`` prefix (``ParseHABackendSpec``). Here a bare
        ``<host>:2379`` is also treated as etcd, and other backends such as
        ``redis://`` / ``k8s://`` fall through to direct addressing and spin
        until SETUP_TIMEOUT. Align with the SDK: no ``://`` -> direct master,
        ``etcd://`` -> etcd lookup, any other scheme -> fail fast.
        """
        raw = self._config.master_addr.strip()
        if not raw:
            return []
        is_etcd = raw.startswith("etcd://")
        if is_etcd:
            raw = raw[len("etcd://") :]

        endpoints: List[str] = []
        for entry in raw.replace(";", ",").split(","):
            entry = entry.strip().rstrip("/")
            if not entry:
                continue
            if entry.startswith(("http://", "https://")):
                scheme, _, hostport = entry.partition("://")
            else:
                scheme, hostport = "http", entry
            _, port = _split_host_port(hostport)
            if not port:
                hostport = f"{hostport}:{ETCD_DEFAULT_PORT}"
                port = str(ETCD_DEFAULT_PORT)
            # A bare "<host>:2379" is etcd too, even without the explicit scheme.
            if port == str(ETCD_DEFAULT_PORT):
                is_etcd = True
            endpoints.append(f"{scheme}://{hostport}")

        return endpoints if is_etcd else []

    def _resolve_master_host(self) -> Optional[str]:
        """Resolve the host exposing the master metrics endpoint.

        With direct addressing ``master_addr`` already is the master. In etcd
        based HA setups it is the etcd endpoint instead, so the elected leader
        must be read from ``mooncake-store/<cluster_id>/master_view``. Returns
        ``None`` while no leader is discoverable so the caller keeps waiting.
        """
        endpoints = self._etcd_endpoints()
        if not endpoints:
            host, _ = _split_host_port(self._config.master_addr)
            return host or None

        key = MASTER_VIEW_KEY_TEMPLATE.format(cluster_id=self._cluster_id())
        for endpoint in endpoints:
            try:
                master_host = _query_etcd_master_view(endpoint, key)
            except Exception as exc:  # unreachable endpoint, try the next one
                # WARNING (not debug): a failing etcd query (endpoint down, HTTP
                # 401 once auth is enabled, TLS error) otherwise gets swallowed
                # into a None return and check_server reports it as "waiting for
                # master election", masking the real cause until SETUP_TIMEOUT.
                # Surfacing it here separates "etcd unreachable/rejected" from
                # "no leader elected yet".
                flexkv_logger.warning(
                    "[MooncakeStoreClient] etcd query failed on %s (key=%s): %s",
                    endpoint,
                    key,
                    exc,
                )
                continue
            if master_host:
                return master_host
        return None

    def check_server(self) -> None:
        # Segment providers mount their own pool; only pure clients must wait for
        # an existing segment to show up in the master metrics endpoint.
        require_existing_segments = self._config.global_segment_size == 0

        start_time = time.perf_counter()
        check_result = False
        while time.perf_counter() - start_time < SETUP_TIMEOUT:
            # Re-resolved every round: in HA setups the master may not have won
            # the election yet, and the leader can change while we wait.
            master_server_ip = self._resolve_master_host()
            if master_server_ip is None:
                flexkv_logger.info(
                    "[MooncakeStoreClient] waiting for mooncake master election "
                    "(no %s in etcd at %s), cost_time: %.2f seconds.",
                    MASTER_VIEW_KEY_TEMPLATE.format(cluster_id=self._cluster_id()),
                    self._config.master_addr,
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            segments_url = (
                f"http://{master_server_ip}:"
                f"{self._config.master_metrics_port}/get_all_segments"
            )
            try:
                check_segments_resp = requests.get(segments_url, timeout=3)
            except Exception:
                flexkv_logger.info(
                    "[MooncakeStoreClient] waiting mooncake store server started "
                    "(%s unreachable), cost_time: %.2f seconds.",
                    segments_url,
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            if require_existing_segments and check_segments_resp.text == "":
                flexkv_logger.info(
                    "[MooncakeStoreClient] waiting mooncake store server started "
                    "(no segment mounted yet, %s), cost_time: %.2f seconds.",
                    segments_url,
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            flexkv_logger.info(
                "[MooncakeStoreClient] Mooncake store server started successfully."
            )
            check_result = True
            break

        if not check_result:
            flexkv_logger.error(
                "[MooncakeStoreClient] Launch mooncake store server timeout"
            )
            raise ValueError(
                "[MooncakeStoreClient] Launch mooncake store server timeout"
            )

    def warm_up(self):
        flexkv_logger.info(
            "[MooncakeStoreClient] Warmup the mooncake store server by writing a 4KB warmup value to the store."
        )
        warmup_key = "flexkv_mooncake_store_warmup_key" + uuid.uuid4().hex
        warmup_value = bytes(4 * 1024)  # 4 KB
        # Retry logic to handle Transfer Engine startup race condition
        max_retries = 10
        retry_delay = 1.0  # seconds
        for attempt in range(max_retries):
            ret = self._store.put(warmup_key, warmup_value)
            if ret == 0:
                break
            flexkv_logger.warning(
                f"[MooncakeStoreClient] Warmup put failed (attempt {attempt + 1}/{max_retries}), "
                f"ret={ret}, retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
        else:
            raise RuntimeError(
                f"[MooncakeStoreClient] Warmup put failed after {max_retries} attempts, "
                "Transfer Engine might not be ready"
            )

        assert (
            self._store.is_exist(warmup_key) == 1
        ), "[MooncakeStoreClient] Warmup put failed"
        assert (
            self._store.get(warmup_key) == warmup_value
        ), "[MooncakeStoreClient] Warmup get failed"

    def register_buffer(
        self,
        tensor_or_ptr: Union[torch.Tensor, int],  # ensure the params type
        size: int = 0,
    ) -> None:
        """Register a pinned CPU tensor (or raw ptr + size) with the store for zero-copy RDMA.

        Parameters
        ----------
        tensor_or_ptr:
            Either a ``torch.Tensor`` (ptr and size are derived automatically)
            or a raw integer pointer (``size`` must be provided).
        size:
            Byte size of the region.  Ignored when ``tensor_or_ptr`` is a tensor.
        """
        self._ensure_setup()
        # TODO: we should ensure the tensor is BLOCKFIRST aligned
        # assert
        if isinstance(tensor_or_ptr, torch.Tensor):
            ptr: int = tensor_or_ptr.data_ptr()
            size = tensor_or_ptr.numel() * tensor_or_ptr.element_size()
        else:
            ptr = int(tensor_or_ptr)
        ret_code = self._store.register_buffer(ptr, size)
        if ret_code != 0:
            flexkv_logger.error(
                f"[MooncakeStoreClient] register_buffer failed with error code: {ret_code}"
            )
            raise RuntimeError(
                f"[MooncakeStoreClient] register_buffer failed "
                f"with error code: {ret_code}"
            )
        flexkv_logger.info(
            f"[MooncakeStoreClient] Registered buffer ptr=0x{ptr:x} size={size}"
        )

    def unregister_buffer(self, buffer_or_ptr: Union[torch.Tensor, int]) -> None:
        """Unregister a buffer from the store."""
        self._ensure_setup()
        if isinstance(buffer_or_ptr, torch.Tensor):
            ptr = buffer_or_ptr.data_ptr()
        else:
            ptr = int(buffer_or_ptr)
        ret_code = self._store.unregister_buffer(ptr)
        if ret_code != 0:
            flexkv_logger.error(
                f"[MooncakeStoreClient] unregister_buffer failed with error code: {ret_code}"
            )
            raise RuntimeError(
                f"[MooncakeStoreClient] unregister_buffer failed "
                f"with error code: {ret_code}"
            )
        flexkv_logger.info(f"[MooncakeStoreClient] Unregistered buffer ptr=0x{ptr:x}")

    def put(self, key: str, buffer_ptr: int, buffer_size: int) -> bool:
        """Write a KV block to the store."""
        return self.batch_put([key], [buffer_ptr], [buffer_size])[0]

    def batch_put(
        self,
        key_strs: list[str],
        buffer_ptrs: list[int],
        buffer_sizes: list[int],
    ) -> List[bool]:
        self._ensure_setup()
        self._validate_buffers(key_strs, buffer_ptrs, buffer_sizes)
        if not key_strs:
            return []

        exist_results = self.batch_exists_impl(key_strs)
        flexkv_logger.info(f"[MooncakeStoreClient] batch_put exist_results: {exist_results}")
        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(key_strs)
        total_size = 0
        for i in range(len(key_strs)):
            if exist_results[i] != 1:
                set_keys.append(key_strs[i])
                set_buffer_ptrs.append(buffer_ptrs[i])
                set_buffer_sizes.append(buffer_sizes[i])
                set_indices.append(i)
                total_size += buffer_sizes[i]
            else:
                set_results[i] = 0

        if len(set_keys) > 0:
            start_time = time.perf_counter()
            put_results = self.zero_copy_put_impl(
                set_keys, set_buffer_ptrs, set_buffer_sizes
            )
            end_time = time.perf_counter()
            flexkv_logger.info(
                f"[MooncakeStoreClient] batch_put: "
                f"{len(set_keys)} keys put in {end_time - start_time:.2f} seconds"
            )
            for i in range(len(set_indices)):
                set_results[set_indices[i]] = put_results[i]
        return self._check_success(set_results, is_set_operate=True)

    def get(self, key: str, buffer_ptr: int, buffer_size: int) -> bool:
        """Read a KV block from the store."""
        return self.batch_get([key], [buffer_ptr], [buffer_size])[0]

    def batch_get(
        self,
        key_strs: list[str],
        buffer_ptrs: list[int],
        buffer_sizes: list[int],
    ) -> List[bool]:
        """Read blocks, accepting only an exact byte count for each key."""
        self._ensure_setup()
        get_results = self.zero_copy_get_impl(key_strs, buffer_ptrs, buffer_sizes)
        return [actual == expected for actual, expected in zip(get_results, buffer_sizes, strict=True)]

    def batch_exists(self, keys_strs: list[str]) -> int:
        """
        Check existence of multiple keys in the store.
        Returns:
            int: longest prefix of keys that exist in the store.
        """
        self._ensure_setup()
        exit_results = self.batch_exists_impl(keys_strs)
        flexkv_logger.info(f"[MooncakeStoreClient] batch_exists, exit_results: {exit_results}")
        for i in range(len(keys_strs)):
            if exit_results[i] != 1:
                return i

        return len(keys_strs)

    def exists(self, key: str) -> bool:
        """Check existence of a key in the store."""
        self._ensure_setup()
        result = self._store.batch_is_exist([key])
        return result[0] == 1

    @staticmethod
    def _validate_buffers(keys, pointers, sizes) -> None:
        if pointers is None or sizes is None or not (len(keys) == len(pointers) == len(sizes)):
            raise ValueError("Mooncake key, pointer and size counts must match")
        if any(size <= 0 for size in sizes):
            raise ValueError("Mooncake buffer sizes must be positive")

    @staticmethod
    def _check_result_count(results, expected: int, operation: str) -> None:
        if results is None or len(results) != expected:
            actual = None if results is None else len(results)
            raise RuntimeError(
                f"Mooncake {operation} returned {actual} results for {expected} keys"
            )

    def _bounded_io(self, operation, keys, pointers, sizes, limit: int) -> List[int]:
        self._ensure_setup()
        self._validate_buffers(keys, pointers, sizes)
        # Validate before submitting any slice so an oversized block cannot
        # cause an unnoticed partial write. Keys are never split or retried.
        if limit and any(size > limit for size in sizes):
            raise ValueError(f"Mooncake block exceeds the {limit}-byte batch limit")
        results = []
        start = 0
        batch_bytes = 0
        for end, size in enumerate(sizes):
            if limit and batch_bytes + size > limit:
                batch = operation(keys[start:end], pointers[start:end], sizes[start:end])
                self._check_result_count(batch, end - start, operation.__name__)
                results.extend(batch)
                start = end
                batch_bytes = 0
            batch_bytes += size
        if start < len(keys):
            batch = operation(keys[start:], pointers[start:], sizes[start:])
            self._check_result_count(batch, len(keys) - start, operation.__name__)
            results.extend(batch)
        return results

    def zero_copy_put_impl(
        self, keys_strs: list[str], buffer_ptrs: list[int], buffer_sizes: list[int]
    ) -> List[int]:
        """Write byte-bounded batches, preserving the SDK's per-key statuses."""
        return self._bounded_io(
            self._store.batch_put_from, keys_strs, buffer_ptrs, buffer_sizes,
            self._config.max_put_batch_bytes,
        )

    def zero_copy_get_impl(
        self, keys_strs: list[str], buffer_ptrs: list[int], buffer_sizes: list[int]
    ) -> List[int]:
        """Read byte-bounded batches, preserving the SDK's per-key byte counts."""
        return self._bounded_io(
            self._store.batch_get_into, keys_strs, buffer_ptrs, buffer_sizes,
            self._config.max_get_batch_bytes,
        )

    def batch_exists_impl(self, keys_strs: list[str]) -> List[int]:
        """Return per-key SDK status: 1 exists, 0 missing, -1 error."""
        self._ensure_setup()
        if not keys_strs:
            return []
        results = self._store.batch_is_exist(keys_strs)
        self._check_result_count(results, len(keys_strs), "batch_is_exist")
        return results

    def _check_success(self, results: List[int], is_set_operate: bool) -> List[bool]:
        # put: success when return == 0; get: success when return > 0 (bytes read)
        return [k_res == 0 if is_set_operate else k_res > 0 for k_res in results]

    def _ensure_setup(self) -> None:
        if not self._is_setup:
            raise RuntimeError(
                "MooncakeStoreClient.setup() must be called before any I/O. "
                "Call setup() inside the worker process after forking."
            )

    def clear(self) -> None:
        """Clear the store."""
        self._ensure_setup()
        self._store.remove_all()


# ---------------------------------------------------------------------------
# Cache engine
# ---------------------------------------------------------------------------


class _MooncakeStoreDummyNode:
    """Placeholder node for MooncakeStoreCacheEngine.insert(); unlock is a no-op."""

    def size(self) -> int:
        return 0


class MooncakeStoreCacheEngine:
    """
    Presents the same ``match()`` / ``insert()`` interface as
    ``CacheEngineAccel`` so that it can be dropped in as
    ``KVCacheManager.remote_cache_engine``.

    """

    # Special sentinel returned in matched_pos so the GET path can branch.
    MATCHED_POS = "global"

    def __init__(
        self,
        cache_config: CacheConfig,
    ) -> None:
        self.tokens_per_block = cache_config.tokens_per_block
        # PP isolation + layer-range key suffix; see ``build_key`` docstring.
        # Single source of truth = cache_config (filled by
        # update_default_config_from_user_config + transfer_manager).
        self.pp_rank = int(getattr(cache_config, "mooncake_store_pp_rank", 0) or 0)
        self.pp_size = int(getattr(cache_config, "mooncake_store_pp_size", 1) or 1)
        self.node_layer_start = int(
            getattr(cache_config, "mooncake_store_node_layer_start", 0) or 0
        )
        self.node_layer_end = int(
            getattr(cache_config, "mooncake_store_node_layer_end", 0) or 0
        )
        self.total_layers = int(
            getattr(cache_config, "mooncake_store_total_layers", 0) or 0
        )
        swa_cfg = getattr(cache_config, "swa", None)
        self._swa_enabled = swa_cfg is not None and getattr(swa_cfg, "enabled", False)

        mooncake_store_config = MooncakeStoreConfig.from_file(cache_config)
        if mooncake_store_config is None:
            raise ValueError(
                "[MooncakeStoreCacheEngine] MooncakeStoreConfig is not found in cache config"
            )
        self.mooncake_store_client = MooncakeStoreClient(
            mooncake_store_config, query_only=True
        )

        flexkv_logger.info(
            f"[MooncakeStoreCacheEngine] swa_enabled={self._swa_enabled}"
        )

    @property
    def swa_enabled(self) -> bool:
        return self._swa_enabled

    def _build_pool_key(self, block_hash: str, kind: PoolKind) -> str:
        return build_key(
            block_hash,
            kind,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            node_layer_start=self.node_layer_start,
            node_layer_end=self.node_layer_end,
            total_layers=self.total_layers,
        )

    def _longest_kv_prefix(self, exists: List[int], n_blocks: int) -> int:
        matched = 0
        for i in range(n_blocks):
            if exists[i] == 1:
                matched += 1
            else:
                break
        return matched

    def _joint_match_length(self, block_hashes: List[str]) -> Tuple[int, int, int]:
        """Return (joint_matched, kv_matched, swa_hit_blocks).

        Semantics
        ---------
        * ``kv_matched``   = longest KV prefix from head (0..N).
        * ``joint_matched`` = **largest** ``L`` in ``[1, kv_matched]`` such that
          ``SWA(hash[L-1])`` exists in the store. 0 when no such L. SWA
          snapshots accumulate historically (a store may hold SWA at multiple
          prefix boundaries laid down by different past writers), so joint
          hit is NOT tied to the tail — we scan the SWA existence window from
          right to left within ``kv_matched`` and stop on the first hit.
        * ``swa_hit_blocks`` mirrors ``joint_matched``: SWA is only usable when
          the full-KV prefix up to it is present.

        Query cost
        ----------
        One RPC ``batch_exists_impl(kv_keys + swa_keys)`` — ``2N`` keys total.

        Example
        -------
        kv_exists  = [1,1,1,1,1]  → kv_matched = 5
        swa_exists = [0,0,1,0,0]  → largest L in [1..5] with swa[L-1]=1 is L=3
        → joint_matched = 3
        """
        n_blocks = len(block_hashes)
        if n_blocks == 0:
            return 0, 0, 0
        kv_keys = [self._build_pool_key(h, PoolKind.KV) for h in block_hashes]
        if not self._swa_enabled:
            kv_matched = self.mooncake_store_client.batch_exists(kv_keys)
            return kv_matched, kv_matched, 0

        swa_keys = [self._build_pool_key(h, PoolKind.SWA) for h in block_hashes]
        flexkv_logger.debug(f"[MooncakeStoreCacheEngine] _joint_match_length kv_keys: {kv_keys}")
        flexkv_logger.debug(f"[MooncakeStoreCacheEngine] _joint_match_length swa_keys: {swa_keys}")
        raw = self.mooncake_store_client.batch_exists_impl(kv_keys + swa_keys)
        flexkv_logger.debug(f"[MooncakeStoreCacheEngine] _joint_match_length raw: {raw}")
        kv_exists = raw[:n_blocks]
        swa_exists = raw[n_blocks:]
        kv_matched = self._longest_kv_prefix(kv_exists, n_blocks)
        if kv_matched == 0:
            return 0, 0, 0

        # Right-to-left scan for the largest L; short-circuits on the common
        # case (SWA appended near the tail of the KV prefix).
        joint_matched = 0
        for L in range(kv_matched, 0, -1):
            if swa_exists[L - 1] == 1:
                joint_matched = L
                break
        return joint_matched, kv_matched, joint_matched

    def swa_tail_key_exists(self, tail_hash: str) -> bool:
        if not self._swa_enabled:
            return False
        key = self._build_pool_key(str(tail_hash), PoolKind.SWA)
        return self.mooncake_store_client.batch_exists([key]) > 0

    def match_swa_from_result(self, match_result, sequence_meta, upper_bound_blocks: int,
                              lock_for_load: bool = False):
        """Mooncake REMOTE tier: SWA is keyed by tail hash, not a mounted slot."""
        # lock_for_load is unused: Mooncake SWA is hash-keyed, so no slot lock is needed.
        swa_hit = int(getattr(match_result, "swa_hit_blocks", 0) or 0) if match_result else 0
        if swa_hit <= 0 or upper_bound_blocks <= 0:
            return 0, -1, None
        swa_hit = min(swa_hit, int(upper_bound_blocks))
        return swa_hit, 0, None

    def match_swa_locked(self, sequence_meta, upper_bound_blocks: int):
        return self.match_swa_from_result(None, sequence_meta, upper_bound_blocks, True)

    def match_swa(self, sequence_meta, upper_bound_blocks: int, lock_for_load: bool = False):
        hit, slot, _ = self.match_swa_locked(sequence_meta, upper_bound_blocks)
        return hit, slot

    def swa_alloc_slot(self) -> int:
        return -1

    def set_swa(self, node, slot: int) -> None:
        pass

    def _drain_swa_slots(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public interface (matches CacheEngineAccel)
    # ------------------------------------------------------------------

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        """KV longest prefix + optional SWA tail joint hit (single RPC)."""
        n_blocks = sequence_meta.num_blocks
        if n_blocks == 0:
            return MatchResultAccel(
                num_matched_blocks=0,
                kv_matched_blocks=0,
                swa_hit_blocks=0,
                last_node=None,
                last_node_matched_length=0,
                physical_blocks=np.arange(0, dtype=np.int64),
                matched_pos=self.MATCHED_POS,
            )

        block_hashes = [str(sequence_meta.block_hashes[i]) for i in range(n_blocks)]
        joint_matched, kv_matched, swa_hit = self._joint_match_length(block_hashes)

        return MatchResultAccel(
            num_matched_blocks=joint_matched,
            kv_matched_blocks=kv_matched,
            swa_hit_blocks=swa_hit,
            last_node=None,
            last_node_matched_length=joint_matched,
            physical_blocks=np.arange(joint_matched, dtype=np.int64),
            matched_pos=self.MATCHED_POS,
        )

    def insert(
        self,
        sequence_meta: SequenceMeta,
        physical_block_ids,
        num_insert_blocks: int = -1,
        match_result: Optional[MatchResultAccel] = None,
    ) -> _MooncakeStoreDummyNode:
        """No-op. Writes to mooncake-store are performed by the transfer worker."""
        return _MooncakeStoreDummyNode()

    def reset(self) -> None:
        """No-op. The store manages its own state."""
        pass

    def start(self) -> None:
        """Start the cache engine."""
        pass

    def stop(self) -> None:
        """Stop the cache engine."""
        pass

    def lock_node(self, node: Any) -> None:
        pass

    def unlock(self, node: Any) -> None:
        pass

    def insert_and_publish(self, node: Any) -> bool:
        return True

    # TAKE is skipped as the real index is managed by Mooncake-store itself
    def take(
        self,
        num_required_blocks: int,
        protected_node: Optional[Any] = None,
        strict: bool = True,
    ) -> np.ndarray:
        return np.zeros(num_required_blocks, dtype=np.int64)

    def recycle(self, block_ids: np.ndarray) -> None:
        pass
