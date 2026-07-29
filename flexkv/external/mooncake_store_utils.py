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

    @classmethod
    def from_file(
        cls, cache_config, override_global_segment_size: Optional[int] = None
    ) -> "MooncakeStoreConfig":
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
                    f"Mooncake store config file path not found in cache config or environment variable MOONCAKE_STORE_CONFIG_PATH"
                )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Mooncake store config file not found: {file_path}"
            )
        with open(file_path, "r") as f:
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

    def check_server(self) -> None:
        master_server_ip = self._config.master_addr.split(":")[0]
        segments_url = f"http://{master_server_ip}:{self._config.master_metrics_port}/get_all_segments"
        # Segment providers mount their own pool; only pure clients must wait for
        # an existing segment to show up in the master metrics endpoint.
        require_existing_segments = self._config.global_segment_size == 0

        start_time = time.perf_counter()
        check_result = False
        while time.perf_counter() - start_time < SETUP_TIMEOUT:
            try:
                check_segments_resp = requests.get(segments_url, timeout=3)
            except Exception:
                flexkv_logger.info(
                    "[MooncakeStoreClient] waiting mooncake store server started, cost_time: %.2f seconds.",
                    time.perf_counter() - start_time,
                )
                time.sleep(3)
                continue

            if require_existing_segments and check_segments_resp.text == "":
                flexkv_logger.info(
                    "[MooncakeStoreClient] waiting mooncake store server started, cost_time: %.2f seconds.",
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
        tensor_or_ptr: "Union[torch.Tensor, int]",  # ensure the params type
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
        self._ensure_setup()
        assert (
            buffer_ptr is not None and buffer_size is not None
        ), "[MooncakeStoreClient] buffer_ptr and buffer_size must be provided"
        exist_result = self.batch_exists([key])
        if exist_result == 1:
            flexkv_logger.info(
                f"[MooncakeStoreClient] key {key} already exists, skip put"
            )
            return True
        ret_code = self._store.batch_put_from([key], [buffer_ptr], [buffer_size])

        return ret_code == 0

    def batch_put(
        self,
        key_strs: list[str],
        buffer_ptrs: list[int],
        buffer_sizes: list[int],
    ) -> List[bool]:
        assert buffer_ptrs is not None and buffer_sizes is not None
        assert len(key_strs) == len(buffer_ptrs) == len(buffer_sizes)

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
        self._ensure_setup()
        assert buffer_ptr is not None and buffer_size is not None
        ret_code = self._store.batch_get_into([key], [buffer_ptr], [buffer_size])
        return ret_code[0] >= 0

    def batch_get(
        self,
        key_strs: list[str],
        buffer_ptrs: list[int],
        buffer_sizes: list[int],
    ) -> List[bool]:
        """Read multiple KV blocks from the store in one call."""
        self._ensure_setup()
        get_results = self.zero_copy_get_impl(key_strs, buffer_ptrs, buffer_sizes)
        return self._check_success(get_results, is_set_operate=False)

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
        result = self._store._batch_exist([key])
        return result[0] == 1

    def zero_copy_put_impl(
        self, keys_strs: list[str], buffer_ptrs: list[int], buffer_sizes: list[int]
    ) -> List[int]:
        """Write multiple KV blocks to the store in one call."""
        return self._store.batch_put_from(keys_strs, buffer_ptrs, buffer_sizes)

    def zero_copy_get_impl(
        self, keys_strs: list[str], buffer_ptrs: list[int], buffer_sizes: list[int]
    ) -> List[int]:
        """Read multiple KV blocks from the store in one call."""
        return self._store.batch_get_into(keys_strs, buffer_ptrs, buffer_sizes)

    def batch_exists_impl(self, keys_strs: list[str]) -> List[int]:
        """Check existence of multiple keys in the store.
        Returns:
            List[int]: per-key raw status codes from the SDK.
                1  = key exists,
                0  = key not found,
                -1 = store error.
        """
        return self._store.batch_is_exist(keys_strs)

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
    """Placeholder node for MooncakeStoreCacheEngine.insert(); unlock/set_ready are no-ops."""

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
                f"[MooncakeStoreCacheEngine] MooncakeStoreConfig is not found in cache config"
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
        raw = self.mooncake_store_client.batch_exists_impl(kv_keys + swa_keys)
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
        del lock_for_load
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
                num_ready_matched_blocks=0,
                num_matched_blocks=0,
                kv_matched_blocks=0,
                swa_hit_blocks=0,
                last_ready_node=None,
                last_node=None,
                last_node_matched_length=0,
                physical_blocks=np.arange(0, dtype=np.int64),
                matched_pos=self.MATCHED_POS,
            )

        block_hashes = [str(sequence_meta.block_hashes[i]) for i in range(n_blocks)]
        joint_matched, kv_matched, swa_hit = self._joint_match_length(block_hashes)

        return MatchResultAccel(
            num_ready_matched_blocks=joint_matched,
            num_matched_blocks=joint_matched,
            kv_matched_blocks=kv_matched,
            swa_hit_blocks=swa_hit,
            last_ready_node=None,
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
        is_ready: bool = True,
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

    def set_ready(self, node: Any, ready: bool, ready_length: int) -> None:
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
