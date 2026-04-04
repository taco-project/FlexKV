"""
FlexKV storage backend adapter for SGLang's HiRadixCache (HiCacheStorage) interface.

Architecture note
-----------------
SGLang's HiRadixCache is a 3-level cache:
    GPU KV cache  ->  Host (CPU pinned) memory  ->  Storage backend

This adapter plugs FlexKV in as the "Storage backend" tier, using KVManager in
thread mode (FLEXKV_CPU_ONLY=1) with the CPU-only PUT path. This avoids GPU
block registration while retaining full FlexKV capabilities (radix tree index,
CPU cache, SSD persistence via io_uring).

Data flow
---------
  PUT (backup):
      SGLang Host mem  ->  mem_pool_host.get_data_page()  ->  layout transform
        ->  direct memcpy to FlexKV CPU cache tensor  ->  KVManager radix tree insert
        ->  TransferEngine H2DISK async (if SSD enabled)

  GET (prefetch):
      KVManager.prefetch_async()  ->  SSD->CPU (if needed)  ->  read CPU cache
        ->  layout transform  ->  mem_pool_host.set_from_flat_data_page()

  EXISTS (query):
      KVManager.get_match()  ->  radix tree prefix match  ->  consecutive page count
"""

import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Configure logger with stderr handler for visibility in SGLang
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[FlexKV] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    _log_level = os.environ.get("FLEXKV_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, _log_level, logging.INFO))

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class _Stats:
    get_calls: int = 0
    get_tokens_requested: int = 0
    get_tokens_hit: int = 0
    set_calls: int = 0
    set_tokens_written: int = 0
    set_tokens_deduped: int = 0
    exists_calls: int = 0
    errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            get_rate = (
                self.get_tokens_hit / self.get_tokens_requested
                if self.get_tokens_requested > 0 else 0.0
            )
            return {
                "get_calls": self.get_calls,
                "get_tokens_requested": self.get_tokens_requested,
                "get_tokens_hit": self.get_tokens_hit,
                "get_hit_rate": round(get_rate, 4),
                "set_calls": self.set_calls,
                "set_tokens_written": self.set_tokens_written,
                "set_tokens_deduped": self.set_tokens_deduped,
                "exists_calls": self.exists_calls,
                "errors": self.errors,
            }


# ---------------------------------------------------------------------------
# Helper: extract token_ids from extra_info
# ---------------------------------------------------------------------------

def _get_token_ids(extra_info) -> Optional[List[int]]:
    """Pull token_ids injected by our cache_controller.py patch.

    SGLang may pass token_ids as a plain list or as a RadixKey object
    (which has a .token_ids attribute).
    """
    if extra_info is None:
        return None
    extra = getattr(extra_info, "extra_info", None)
    if not isinstance(extra, dict):
        return None
    val = extra.get("token_ids")
    if val is None:
        return None
    if hasattr(val, "token_ids"):
        val = val.token_ids
    if isinstance(val, (list, tuple)):
        return list(val)
    if hasattr(val, "tolist"):
        return val.tolist()
    return None


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class FlexKVHiCacheStorage(HiCacheStorage):
    """FlexKV as a SGLang HiCacheStorage backend.

    Uses KVManager in thread mode (CPU-only) with direct CPU cache tensor
    access for data filling. Supports CPU cache + optional SSD persistence.

    Configuration
    ~~~~~~~~~~~~~
    Pass JSON via ``--hicache-storage-backend-extra-config``:

    .. code-block:: json

        {
          "num_layers": 32,
          "num_kv_heads": 8,
          "head_size": 128,
          "enable_cpu": true,
          "enable_ssd": false,
          "num_cpu_blocks": 100000
        }

    Note: ``tokens_per_block`` is overridden at runtime by SGLang's
    ``page_size`` from ``mem_pool_host``.
    """

    _MODEL_KEYS = {"num_layers", "num_kv_heads", "head_size", "use_mla",
                   "dtype", "tp_size", "dp_size"}
    _CACHE_KEYS = {"tokens_per_block", "eviction_policy", "enable_cpu",
                   "enable_ssd", "enable_gds", "enable_remote",
                   "num_cpu_blocks", "num_ssd_blocks", "ssd_cache_dir"}

    def __init__(
        self,
        storage_config: "HiCacheStorageConfig",
        mem_pool_host: Any = None,
    ) -> None:
        from flexkv.common.config import ModelConfig, CacheConfig

        extra = storage_config.extra_config or {}

        model_kwargs: Dict[str, Any] = {
            k: extra[k] for k in self._MODEL_KEYS if k in extra
        }
        model_kwargs.setdefault("tp_size", storage_config.tp_size)
        model_kwargs.setdefault("use_mla", storage_config.is_mla_model)
        self._model_config = ModelConfig(**model_kwargs)

        cache_kwargs: Dict[str, Any] = {
            k: extra[k] for k in self._CACHE_KEYS if k in extra
        }
        self._cache_config = CacheConfig(**cache_kwargs)

        logger.info("Initializing  model=%s  cache=%s",
                     self._model_config, self._cache_config)

        self._should_backup: bool = (
            not storage_config.is_mla_model
            or storage_config.tp_rank == 0
        )

        # KVManager created in register_mem_pool_host() after we know page_size
        self._kv_manager = None
        self._cpu_cache_tensor = None  # direct access to CPU cache (thread mode)
        self._elements_per_block: int = 0
        self._page_size: int = self._cache_config.tokens_per_block
        self._mem_pool_host = mem_pool_host
        self._started = False
        self._stats = _Stats()

        logger.info("Ready (KVManager deferred until register_mem_pool_host)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _token_ids_to_numpy(self, token_ids: List[int]) -> np.ndarray:
        return np.array(token_ids, dtype=np.int64)

    def _init_kv_manager(self):
        """Create KVManager in thread mode with CPU-only configuration."""
        from flexkv.kvmanager import KVManager
        from flexkv.common.config import CacheConfig

        # Override tokens_per_block to match SGLang's page_size
        cache_config = CacheConfig(
            tokens_per_block=self._page_size,
            eviction_policy=self._cache_config.eviction_policy,
            enable_cpu=True,
            enable_ssd=self._cache_config.enable_ssd,
            enable_gds=False,
            enable_remote=False,
            num_cpu_blocks=self._cache_config.num_cpu_blocks,
            num_ssd_blocks=self._cache_config.num_ssd_blocks,
            ssd_cache_dir=self._cache_config.ssd_cache_dir,
        )

        # Set env vars for thread mode before creating KVManager
        os.environ["FLEXKV_CPU_ONLY"] = "1"
        os.environ["FLEXKV_INSTANCE_NUM"] = "0"

        # Also patch the already-loaded GLOBAL_CONFIG_FROM_ENV directly,
        # since it was read at import time before env vars were set.
        from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
        GLOBAL_CONFIG_FROM_ENV.instance_num = 0

        self._kv_manager = KVManager(self._model_config, cache_config)
        self._kv_manager.start()

        # Get direct access to CPU cache tensor (thread mode only)
        self._cpu_cache_tensor = self._kv_manager.get_cpu_cache_tensor()

        # Compute elements per block for indexing
        kv_dim = 1 if self._model_config.use_mla else 2
        self._elements_per_block = (
            self._model_config.num_layers * kv_dim *
            self._page_size * self._model_config.num_kv_heads *
            self._model_config.head_size
        )

        logger.info("KVManager started (thread mode, page_size=%d, "
                     "cpu_blocks=%d, ssd=%s, elements_per_block=%d)",
                     self._page_size, cache_config.num_cpu_blocks,
                     cache_config.enable_ssd, self._elements_per_block)

    def _ensure_started(self):
        if not self._started and self._kv_manager is not None:
            self._kv_manager.start()
            self._cpu_cache_tensor = self._kv_manager.get_cpu_cache_tensor()
            self._started = True

    def _sglang_to_flexkv(self, data_page: torch.Tensor) -> torch.Tensor:
        """Convert SGLang data page to FlexKV BLOCKFIRST layout."""
        layout = getattr(self._mem_pool_host, 'layout', 'layer_first')
        if layout == "layer_first":
            return data_page.permute(1, 0, 2, 3, 4).contiguous()
        elif layout == "page_first":
            return data_page.permute(2, 0, 1, 3, 4).contiguous()
        elif layout == "page_first_direct":
            return data_page.squeeze(1).permute(1, 0, 2, 3, 4).contiguous()
        elif layout == "page_head":
            return data_page.squeeze(1).permute(3, 0, 2, 1, 4).contiguous()
        else:
            raise ValueError(f"Unsupported SGLang host pool layout: {layout}")

    def _flexkv_to_sglang(self, block_data: torch.Tensor) -> torch.Tensor:
        """Convert FlexKV BLOCKFIRST to SGLang layout."""
        layout = getattr(self._mem_pool_host, 'layout', 'layer_first')
        if layout == "layer_first":
            return block_data.permute(1, 0, 2, 3, 4).contiguous()
        elif layout == "page_first":
            return block_data.permute(1, 2, 0, 3, 4).contiguous()
        elif layout == "page_first_direct":
            return block_data.permute(1, 0, 2, 3, 4).unsqueeze(1).contiguous()
        elif layout == "page_head":
            return block_data.permute(1, 3, 2, 0, 4).unsqueeze(1).contiguous()
        else:
            raise ValueError(f"Unsupported SGLang host pool layout: {layout}")

    def _get_block_view(self, block_id: int) -> torch.Tensor:
        """Get a flat view of a CPU cache block for direct memcpy."""
        start = int(block_id) * self._elements_per_block
        end = start + self._elements_per_block
        return self._cpu_cache_tensor[start:end]

    def _get_block_shaped(self, block_id: int) -> torch.Tensor:
        """Get a CPU cache block reshaped to BLOCKFIRST: [L, kv_dim, T, H, D]."""
        kv_dim = 1 if self._model_config.use_mla else 2
        return self._get_block_view(block_id).view(
            self._model_config.num_layers, kv_dim,
            self._page_size, self._model_config.num_kv_heads,
            self._model_config.head_size
        )

    # ------------------------------------------------------------------
    # HiCacheStorage interface
    # ------------------------------------------------------------------

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        """Called by HiCacheController after construction."""
        self._mem_pool_host = mem_pool_host

        # Auto-detect model parameters from mem_pool_host when not
        # explicitly provided via extra_config.
        self._auto_detect_model_params(mem_pool_host)

        sglang_page_size = getattr(mem_pool_host, 'page_size', None)
        if sglang_page_size is not None and sglang_page_size != self._page_size:
            logger.info("SGLang page_size=%d overrides configured tokens_per_block=%d",
                        sglang_page_size, self._page_size)
            self._page_size = sglang_page_size

        self._init_kv_manager()
        self._started = True

    def _auto_detect_model_params(self, mem_pool_host: Any) -> None:
        """Fill missing model parameters from SGLang's mem_pool_host.

        SGLang's MHATokenToKVPoolHost exposes layer_num, head_num,
        head_dim which map directly to FlexKV's ModelConfig fields.
        """
        _MAP = [
            ("num_layers", "layer_num"),
            ("num_kv_heads", "head_num"),
            ("head_size", "head_dim"),
        ]
        detected = []
        for flexkv_attr, sglang_attr in _MAP:
            cur = getattr(self._model_config, flexkv_attr, 0)
            if cur <= 1:  # default / not set
                pool_val = getattr(mem_pool_host, sglang_attr, None)
                if pool_val is not None and pool_val > 1:
                    setattr(self._model_config, flexkv_attr, pool_val)
                    detected.append(f"{flexkv_attr}={pool_val}")
        if detected:
            logger.info("Auto-detected from mem_pool_host: %s",
                        ", ".join(detected))

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Optional[Any] = None,
    ) -> int:
        """Return consecutive matched page count from the start."""
        with self._stats._lock:
            self._stats.exists_calls += 1

        token_ids = _get_token_ids(extra_info)
        if not token_ids or self._kv_manager is None:
            return 0

        try:
            from flexkv.common.block import SequenceMeta

            arr = self._token_ids_to_numpy(token_ids)
            page_size = self._page_size
            aligned_len = (len(arr) // page_size) * page_size
            if aligned_len == 0:
                return 0

            # Query the CPU cache engine's radix tree directly
            cache_engine = self._kv_manager.kv_task_engine.cache_engine
            seq_meta = SequenceMeta(token_ids=arr[:aligned_len],
                                    tokens_per_block=page_size)
            match_result = cache_engine.cpu_cache_engine.match(seq_meta)

            return min(match_result.num_ready_matched_blocks, len(keys))

        except Exception:
            logger.exception("batch_exists failed")
            with self._stats._lock:
                self._stats.errors += 1
            return 0

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[Any] = None,
    ) -> List[bool]:
        """Load KV blocks from FlexKV into SGLang's Host memory pool."""
        with self._stats._lock:
            self._stats.get_calls += 1

        token_ids = _get_token_ids(extra_info)
        if not token_ids or self._kv_manager is None:
            return [False] * len(keys)

        with self._stats._lock:
            self._stats.get_tokens_requested += len(token_ids)

        try:
            from flexkv.common.block import SequenceMeta

            arr = self._token_ids_to_numpy(token_ids)
            page_size = self._page_size
            num_pages = len(keys)

            aligned_len = (len(arr) // page_size) * page_size
            if aligned_len == 0:
                return [False] * num_pages

            # Query the CPU cache engine's radix tree directly for physical blocks
            cache_engine = self._kv_manager.kv_task_engine.cache_engine
            seq_meta = SequenceMeta(token_ids=arr[:aligned_len],
                                    tokens_per_block=page_size)
            match_result = cache_engine.cpu_cache_engine.match(seq_meta)
            matched_pages = match_result.num_ready_matched_blocks

            results: List[bool] = []
            for page_idx in range(num_pages):
                if page_idx < matched_pages:
                    block_id = match_result.physical_blocks[page_idx]
                    try:
                        block_data = self._get_block_shaped(block_id).clone()
                        transformed = self._flexkv_to_sglang(block_data)
                        host_token_start = host_indices[page_idx * page_size].item()
                        self._mem_pool_host.set_from_flat_data_page(
                            host_token_start, transformed.flatten()
                        )
                        results.append(True)
                    except Exception:
                        logger.exception("batch_get_v1: failed to read block %d", block_id)
                        results.append(False)
                else:
                    results.append(False)

            hit_count = sum(results)
            with self._stats._lock:
                self._stats.get_tokens_hit += hit_count * page_size

            if hit_count > 0:
                logger.debug("batch_get_v1: %d/%d pages loaded", hit_count, num_pages)
            return results

        except Exception:
            logger.exception("batch_get_v1 failed")
            with self._stats._lock:
                self._stats.errors += 1
            return [False] * len(keys)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[Any] = None,
    ) -> List[bool]:
        """Store KV blocks from SGLang's Host memory pool into FlexKV."""
        if not self._should_backup:
            return [True] * len(keys)

        with self._stats._lock:
            self._stats.set_calls += 1

        token_ids = _get_token_ids(extra_info)
        if not token_ids or self._kv_manager is None:
            return [False] * len(keys)

        try:
            page_size = self._page_size
            num_pages = len(keys)
            arr = self._token_ids_to_numpy(token_ids)

            # CPU-only PUT: allocate blocks, get their IDs
            task_id, cpu_block_ids, return_mask = self._kv_manager.put_cpu(
                token_ids=arr)

            if len(cpu_block_ids) == 0:
                # No new blocks allocated — either fully deduped or alloc failed.
                # Distinguish by querying the radix tree: if pages exist, it's dedup.
                from flexkv.common.block import SequenceMeta as _SM
                aligned_len = (len(arr) // page_size) * page_size
                _seq = _SM(token_ids=arr[:aligned_len], tokens_per_block=page_size)
                cache_engine = self._kv_manager.kv_task_engine.cache_engine
                _mr = cache_engine.cpu_cache_engine.match(_seq)
                if _mr.num_ready_matched_blocks > 0:
                    with self._stats._lock:
                        self._stats.set_tokens_deduped += len(token_ids)
                    return [True] * num_pages
                return [False] * num_pages

            # Fill data into CPU cache tensor for each new block
            for i, block_id in enumerate(cpu_block_ids):
                # Determine which page this block corresponds to
                # cpu_block_ids are for the non-deduped (new) pages
                num_deduped = num_pages - len(cpu_block_ids)
                page_idx = num_deduped + i

                host_token_start = host_indices[page_idx * page_size].item()
                data_page = self._mem_pool_host.get_data_page(
                    host_token_start, flat=False
                )

                # Layout transform: SGLang -> FlexKV BLOCKFIRST
                transformed = self._sglang_to_flexkv(data_page)

                # Direct memcpy to FlexKV CPU cache tensor
                dst = self._get_block_view(block_id)
                dst.copy_(transformed.flatten())

            # Launch H2DISK transfer (async, if SSD enabled)
            self._kv_manager.launch_cpu([task_id])

            with self._stats._lock:
                self._stats.set_tokens_written += len(cpu_block_ids) * page_size
                num_deduped = num_pages - len(cpu_block_ids)
                self._stats.set_tokens_deduped += num_deduped * page_size

            logger.debug("batch_set_v1: wrote %d new pages, %d deduped (total %d)",
                         len(cpu_block_ids), num_deduped, num_pages)
            return [True] * num_pages

        except Exception:
            logger.exception("batch_set_v1 failed")
            with self._stats._lock:
                self._stats.errors += 1
            return [False] * len(keys)

    # Legacy abstract methods
    def get(self, key, target_location=None, target_sizes=None):
        return None

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        return [None] * len(keys)

    def set(self, key, value=None, target_location=None, target_sizes=None):
        return False

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        return False

    def exists(self, key: str) -> bool:
        return False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.to_dict()

    def clear(self) -> None:
        try:
            # TODO: clear KVManager's cache engine
            self._stats = _Stats()
            logger.info("Cache cleared")
        except Exception:
            logger.exception("clear() failed")

    def __del__(self):
        try:
            if self._kv_manager is not None:
                self._kv_manager.shutdown()
        except Exception:
            pass
