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

import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from flexkv.common.debug import flexkv_logger as logger

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
try:
    from sglang.srt.metrics.collector import StorageMetrics
except ImportError:
    from sglang.srt.observability.metrics_collector import StorageMetrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_LOCAL = "local"
MODE_DISTRIBUTED = "distributed"
_VALID_MODES = (MODE_LOCAL, MODE_DISTRIBUTED)

# Error operation labels (used in metrics recording)
_OP_GET = "get"
_OP_SET = "set"
_OP_EXISTS = "exists"

_LAYOUT_LAYER_FIRST = "layer_first"


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

    Two Modes
    ~~~~~~~~~
    **Local mode** (default): Single-node isolated cache, no cross-node sharing.
    **Distributed mode**: Multi-node KV Cache sharing via Distributed RadixTree + Redis GMS.

    Configuration
    ~~~~~~~~~~~~~
    Pass JSON via ``--hicache-storage-backend-extra-config``:

    **Local mode (default)**:

    .. code-block:: json

        {
          "num_layers": 32,
          "num_kv_heads": 8,
          "head_size": 128,
          "enable_cpu": true,
          "enable_ssd": false,
          "num_cpu_blocks": 100000,
          "mode": "local"
        }

    **Distributed mode** (multi-Prefill sharing):

    .. code-block:: json

        {
          "num_layers": 32,
          "num_kv_heads": 8,
          "head_size": 128,
          "enable_cpu": true,
          "enable_ssd": false,
          "num_cpu_blocks": 100000,
          "mode": "distributed",
          "redis_host": "<redis-server-ip>",
          "redis_port": 6379,
          "redis_password": "optional_password"
        }

    Note: ``tokens_per_block`` is overridden at runtime by SGLang's
    ``page_size`` from ``mem_pool_host``.
    """

    _MODEL_KEYS = {"num_layers", "num_kv_heads", "head_size", "use_mla",
                   "dtype", "tp_size", "dp_size"}
    _CACHE_KEYS = {"tokens_per_block", "eviction_policy", "enable_cpu",
                   "enable_ssd", "enable_gds", "enable_remote",
                   "num_cpu_blocks", "num_ssd_blocks", "ssd_cache_dir",
                   "mode", "redis_host", "redis_port", "redis_password",
                   "prefetch_timeout"}
    # Keys accepted in extra_config but NOT passed to CacheConfig
    _ADAPTER_ONLY_KEYS = {"mode", "redis_host", "redis_port", "redis_password",
                          "prefetch_timeout"}

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
            k: extra[k] for k in self._CACHE_KEYS - self._ADAPTER_ONLY_KEYS
            if k in extra
        }
        self._cache_config = CacheConfig(**cache_kwargs)

        # Extract distributed mode configuration
        self._mode: str = extra.get("mode", MODE_LOCAL)
        if self._mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode: {self._mode}. Must be one of {_VALID_MODES}")
        
        self._redis_host: str = extra.get("redis_host", "127.0.0.1")
        self._redis_port: int = extra.get("redis_port", 6379)
        self._redis_password: Optional[str] = extra.get("redis_password", None)
        
        self._prefetch_timeout: float = float(extra.get("prefetch_timeout", 5.0))

        if self._mode == MODE_DISTRIBUTED and not self._redis_host:
            raise ValueError("redis_host is required when mode='distributed'")

        self._should_backup: bool = (
            not storage_config.is_mla_model
            or storage_config.tp_rank == 0
        )

        # KVManager created in register_mem_pool_host() after we know page_size
        self._kv_manager = None
        self._cpu_cache_tensor = None  # direct access to CPU cache (thread mode)
        self._elements_per_block: int = 0
        self._block_shape: tuple = ()  # set after KVManager init
        self._page_size: int = self._cache_config.tokens_per_block
        self._mem_pool_host = mem_pool_host
        self._bytes_per_page: int = 0
        self._gb_per_page: float = 0.0

        # SGLang-compatible metrics buffers (sliding window, cleared on get_stats)
        self._metrics_lock = threading.Lock()
        self._prefetch_pgs: List[int] = []
        self._backup_pgs: List[int] = []
        self._prefetch_bandwidth: List[float] = []
        self._backup_bandwidth: List[float] = []

        self._metrics = None  # resolved after KVManager starts in register_mem_pool_host()

        logger.info("Ready (KVManager deferred until register_mem_pool_host)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _token_ids_to_numpy(self, token_ids: List[int]) -> np.ndarray:
        return np.array(token_ids, dtype=np.int64)

    def _init_kv_manager(self):
        """Create KVManager in thread mode with CPU-only configuration.
        
        Supports two modes:
        - local: single-node, isolated cache (default)
        - distributed: multi-node with Redis GMS + Mooncake P2P transfers
        """
        from flexkv.kvmanager import KVManager
        from flexkv.common.config import CacheConfig

        # Build base cache config (common to both modes)
        cache_config_kwargs = {
            "tokens_per_block": self._page_size,
            "eviction_policy": self._cache_config.eviction_policy,
            "enable_cpu": True,
            "enable_ssd": self._cache_config.enable_ssd,
            "enable_gds": False,
            "num_cpu_blocks": self._cache_config.num_cpu_blocks,
            "num_ssd_blocks": self._cache_config.num_ssd_blocks,
            "ssd_cache_dir": self._cache_config.ssd_cache_dir,
        }

        # Branch on mode: local vs distributed
        if self._mode == MODE_DISTRIBUTED:
            cache_config_kwargs.update({
                "enable_remote": True,
                "enable_kv_sharing": True,
                "enable_p2p_cpu": True,
                "enable_p2p_ssd": self._cache_config.enable_ssd,
                "redis_host": self._redis_host,
                "redis_port": self._redis_port,
                "redis_password": self._redis_password,
            })
            logger.info("Distributed mode enabled: Redis=%s:%d, P2P CPU, P2P SSD=%s",
                       self._redis_host, self._redis_port, self._cache_config.enable_ssd)
        else:
            # Local mode (default): isolated single-node cache
            cache_config_kwargs.update({
                "enable_remote": False,
                "enable_kv_sharing": False,
                "enable_p2p_cpu": False,
                "enable_p2p_ssd": False,
            })
            logger.info("Local mode enabled: isolated node, no cross-node sharing")

        cache_config = CacheConfig(**cache_config_kwargs)

        # Set env vars for thread mode before creating KVManager.
        # WARNING: These modify process-global state. Only one
        # FlexKVHiCacheStorage instance is supported per process.
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

        # Compute elements per block and cache the block shape for indexing
        kv_dim = 1 if self._model_config.use_mla else 2
        self._elements_per_block = (
            self._model_config.num_layers * kv_dim *
            self._page_size * self._model_config.num_kv_heads *
            self._model_config.head_size
        )
        self._block_shape = (
            self._model_config.num_layers, kv_dim,
            self._page_size, self._model_config.num_kv_heads,
            self._model_config.head_size
        )

        # Compute bytes_per_page for bandwidth reporting
        dtype = getattr(self._model_config, 'dtype', None) or torch.float16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float16)
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        self._bytes_per_page = self._elements_per_block * bytes_per_element
        self._gb_per_page = self._bytes_per_page / (1 << 30)

        logger.info("KVManager started (thread mode, page_size=%d, "
                     "cpu_blocks=%d, ssd=%s, elements_per_block=%d, "
                     "bytes_per_page=%d, mode=%s)",
                     self._page_size, cache_config.num_cpu_blocks,
                     cache_config.enable_ssd, self._elements_per_block,
                     self._bytes_per_page, self._mode)

    def _sglang_to_flexkv(self, data_page: torch.Tensor) -> torch.Tensor:
        """Convert SGLang data page to FlexKV BLOCKFIRST layout.

        MHA (5D): SGLang (2, L, T, H, D) -> FlexKV (L, 2, T, H, D)
        MLA (4D): SGLang (L, T, 1, D)    -> FlexKV (L, 1, T, 1, D)
        """
        layout = getattr(self._mem_pool_host, 'layout', _LAYOUT_LAYER_FIRST)
        if self._model_config.use_mla:
            # MLA: data_page is 4D (L, T, 1, D) -> (L, 1, T, 1, D)
            return data_page.unsqueeze(1).contiguous()
        if layout == _LAYOUT_LAYER_FIRST:
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
        """Convert FlexKV BLOCKFIRST to SGLang layout.

        MHA (5D): FlexKV (L, 2, T, H, D) -> SGLang (2, L, T, H, D)
        MLA (5D): FlexKV (L, 1, T, 1, D) -> SGLang (L, T, 1, D)
        """
        layout = getattr(self._mem_pool_host, 'layout', _LAYOUT_LAYER_FIRST)
        if self._model_config.use_mla:
            # MLA: block_data is (L, 1, T, 1, D) -> (L, T, 1, D)
            return block_data.squeeze(1).contiguous()
        if layout == _LAYOUT_LAYER_FIRST:
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
        if end > len(self._cpu_cache_tensor):
            raise ValueError(
                f"block_id {block_id} out of bounds "
                f"(offset {end} > tensor size {len(self._cpu_cache_tensor)})")
        return self._cpu_cache_tensor[start:end]

    def _get_block_shaped(self, block_id: int) -> torch.Tensor:
        """Get a CPU cache block reshaped to BLOCKFIRST: [L, kv_dim, T, H, D].

        Returns a **view** into the CPU cache tensor.  Callers that need
        isolation from concurrent writes (e.g. batch_get_v1) must ``.clone()``
        the result themselves; callers that write *into* the block
        (e.g. batch_set_v1) must use the view directly.
        """
        return self._get_block_view(block_id).view(self._block_shape)

    def _write_layer_to_host(self, block_data: torch.Tensor, layer_id: int,
                             host_start: int, use_mla: bool) -> None:
        """Write one layer of a FlexKV block into SGLang Host kv_buffer.

        Avoids the permute+contiguous overhead of the whole-page path by
        copying directly at layer granularity (layer_first layout).
        """
        end = host_start + self._page_size
        kv_buf = self._mem_pool_host.kv_buffer
        # MHA kv_buffer: (2, L, T, H, D) -> token axis is 2
        # MLA kv_buffer: (L, T, 1, D)    -> token axis is 1
        token_dim = 1 if use_mla else 2
        if host_start < 0 or end > kv_buf.shape[token_dim]:
            logger.error("_write_layer_to_host: OOB host_start=%d end=%d "
                         "kv_buffer.shape=%s", host_start, end, kv_buf.shape)
            return
        layer_slice = block_data[layer_id]
        if use_mla:
            kv_buf[layer_id, host_start:end, :, :] = layer_slice[0]
        else:
            kv_buf[:, layer_id, host_start:end, :, :] = layer_slice

    def _read_layer_from_host(self, block_data: torch.Tensor, layer_id: int,
                              host_start: int, use_mla: bool) -> None:
        """Read one layer from SGLang Host kv_buffer into a FlexKV block."""
        end = host_start + self._page_size
        kv_buf = self._mem_pool_host.kv_buffer
        token_dim = 1 if use_mla else 2
        if host_start < 0 or end > kv_buf.shape[token_dim]:
            logger.error("_read_layer_from_host: OOB host_start=%d end=%d "
                         "kv_buffer.shape=%s", host_start, end, kv_buf.shape)
            return
        if use_mla:
            block_data[layer_id, 0] = kv_buf[layer_id, host_start:end, :, :]
        else:
            block_data[layer_id] = kv_buf[:, layer_id, host_start:end, :, :]

    def _fetch_remote_blocks(self, token_ids: np.ndarray) -> bool:
        """Fetch remote blocks into local CPU cache via prefetch_async.

        Uses KVManager.prefetch_async() which internally discovers remote
        blocks via the distributed radix tree and pulls them over P2P
        (Mooncake Transfer Engine) into the local CPU cache tensor.

        Args:
            token_ids: Full token ID array for the sequence.

        Returns:
            True if fetch succeeded, False otherwise.
        """
        from flexkv.common.request import KVResponseStatus

        try:
            task_id = self._kv_manager.prefetch_async(token_ids=token_ids)
            responses = self._kv_manager.wait([task_id], timeout=self._prefetch_timeout)

            if not responses or task_id not in responses:
                logger.warning("_fetch_remote_blocks: prefetch timeout")
                return False

            resp = responses[task_id]
            if resp.status != KVResponseStatus.SUCCESS:
                logger.warning("_fetch_remote_blocks: prefetch status=%s",
                               resp.status.value)
                return False

            logger.debug("_fetch_remote_blocks: success (task_id=%d)", task_id)
            return True

        except Exception:
            logger.exception("_fetch_remote_blocks failed")
            return False

    # ------------------------------------------------------------------
    # HiCacheStorage interface
    # ------------------------------------------------------------------

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        """Called by HiCacheController after construction."""
        self._mem_pool_host = mem_pool_host

        # Auto-detect model parameters from mem_pool_host when not
        # explicitly provided via extra_config.
        self._auto_detect_model_params(mem_pool_host)

        logger.info("Initializing  model=%s  cache=%s  mode=%s",
                     self._model_config, self._cache_config, self._mode)

        sglang_page_size = getattr(mem_pool_host, 'page_size', None)
        if sglang_page_size is not None and sglang_page_size != self._page_size:
            logger.info("SGLang page_size=%d overrides configured tokens_per_block=%d",
                        sglang_page_size, self._page_size)
            self._page_size = sglang_page_size

        self._init_kv_manager()

        # KVManager has started — global collector is now initialized
        try:
            from flexkv.metrics import get_global_collector
            self._metrics = get_global_collector()
        except Exception:
            self._metrics = None

    def _auto_detect_model_params(self, mem_pool_host: Any) -> None:
        """Fill missing model parameters from SGLang's mem_pool_host.

        SGLang uses two host pool types:
        - MHATokenToKVPoolHost: exposes layer_num, head_num, head_dim
        - MLATokenToKVPoolHost: exposes layer_num, kv_lora_rank, qk_rope_head_dim
        """
        detected = []

        # num_layers: common to both MHA and MLA
        cur_layers = getattr(self._model_config, 'num_layers', 0)
        if cur_layers <= 1:
            pool_val = getattr(mem_pool_host, 'layer_num', None)
            if pool_val is not None and pool_val > 1:
                self._model_config.num_layers = pool_val
                detected.append(f"num_layers={pool_val}")

        if self._model_config.use_mla:
            # MLA: num_kv_heads=1 (unified KV), head_size = kv_lora_rank + qk_rope_head_dim
            kv_lora_rank = getattr(mem_pool_host, 'kv_lora_rank', None)
            qk_rope_head_dim = getattr(mem_pool_host, 'qk_rope_head_dim', None)
            if kv_lora_rank is not None and qk_rope_head_dim is not None:
                self._model_config.num_kv_heads = 1
                mla_head_size = kv_lora_rank + qk_rope_head_dim
                if getattr(self._model_config, 'head_size', 0) <= 1:
                    self._model_config.head_size = mla_head_size
                    detected.append(f"head_size={mla_head_size} "
                                    f"(kv_lora_rank={kv_lora_rank}+"
                                    f"qk_rope_head_dim={qk_rope_head_dim})")
                detected.append("num_kv_heads=1 (MLA)")
        else:
            # MHA: head_num, head_dim
            _MAP = [
                ("num_kv_heads", "head_num"),
                ("head_size", "head_dim"),
            ]
            for flexkv_attr, sglang_attr in _MAP:
                cur = getattr(self._model_config, flexkv_attr, 0)
                if cur <= 1:
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
        """Return consecutive matched page count from the start.

        In distributed mode, also reports remote hits so that SGLang
        proceeds to batch_get_v1 where remote blocks will be fetched.
        """
        if self._metrics:
            self._metrics.record_sglang_batch_exists()

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

            cache_engine = self._kv_manager.kv_task_engine.cache_engine
            seq_meta = SequenceMeta(token_ids=arr[:aligned_len],
                                    tokens_per_block=page_size)

            # In distributed mode, query both local and remote trees
            if (self._mode == MODE_DISTRIBUTED
                    and hasattr(cache_engine.cpu_cache_engine, 'match_all')):
                match_result = cache_engine.cpu_cache_engine.match_all(seq_meta)
            else:
                match_result = cache_engine.cpu_cache_engine.match(seq_meta)

            return min(match_result.num_ready_matched_blocks, len(keys))

        except Exception:
            logger.exception("batch_exists failed")
            if self._metrics:
                self._metrics.record_sglang_error(_OP_EXISTS)
            return 0

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[Any] = None,
    ) -> List[bool]:
        """Load KV blocks from FlexKV into SGLang's Host memory pool.

        For layer_first layout, copies data layer-by-layer to enable future
        pipelining with Host-to-GPU transfers.  An optional callback
        ``extra_info.extra_info["layer_ready_callback"]`` is invoked after
        each layer completes, allowing the caller to overlap work.

        Other layouts fall back to whole-page copy via permute+flatten.
        """
        token_ids = _get_token_ids(extra_info)
        if not token_ids or self._kv_manager is None:
            if self._metrics:
                self._metrics.record_sglang_batch_get(
                    blocks_hit=0, blocks_missed=len(keys))
            return [False] * len(keys)

        try:
            from flexkv.common.block import SequenceMeta

            start_time = time.perf_counter()
            arr = self._token_ids_to_numpy(token_ids)
            page_size = self._page_size
            num_pages = len(keys)

            aligned_len = (len(arr) // page_size) * page_size
            if aligned_len == 0:
                return [False] * num_pages

            cache_engine = self._kv_manager.kv_task_engine.cache_engine
            seq_meta = SequenceMeta(token_ids=arr[:aligned_len],
                                    tokens_per_block=page_size)

            # In distributed mode, query both local and remote trees
            remote_fetched = False
            if (self._mode == MODE_DISTRIBUTED
                    and hasattr(cache_engine.cpu_cache_engine, 'match_all')):
                match_result = cache_engine.cpu_cache_engine.match_all(seq_meta)

                if match_result.matched_pos == "remote":
                    if not self._fetch_remote_blocks(arr):
                        logger.debug("batch_get_v1: remote fetch failed, "
                                     "treating as miss")
                        if self._metrics:
                            self._metrics.record_sglang_remote_fetch(
                                success=False)
                        return [False] * num_pages

                    remote_fetched = True
                    match_result = cache_engine.cpu_cache_engine.match(
                        seq_meta)
            else:
                match_result = cache_engine.cpu_cache_engine.match(seq_meta)

            matched_pages = match_result.num_ready_matched_blocks

            # Extract optional layer-ready callback
            layer_ready_cb = None
            if (extra_info and hasattr(extra_info, 'extra_info')
                    and isinstance(extra_info.extra_info, dict)):
                layer_ready_cb = extra_info.extra_info.get(
                    "layer_ready_callback")

            layout = getattr(self._mem_pool_host, 'layout', _LAYOUT_LAYER_FIRST)

            if layout == _LAYOUT_LAYER_FIRST and matched_pages > 0:
                # Layerwise path: outer loop = layer, inner loop = page.
                # Precompute invariants to avoid repeated work in O(layers*pages) loop.
                results = [False] * num_pages
                num_layers = self._model_config.num_layers
                use_mla = self._model_config.use_mla
                block_views = []
                host_starts = []
                for page_idx in range(matched_pages):
                    bid = match_result.physical_blocks[page_idx]
                    # Clone to isolate from concurrent batch_set_v1 writes
                    block_views.append(self._get_block_shaped(bid).clone())
                    host_starts.append(
                        host_indices[page_idx * page_size].item())

                try:
                    for layer_id in range(num_layers):
                        for page_idx in range(matched_pages):
                            self._write_layer_to_host(
                                block_views[page_idx], layer_id,
                                host_starts[page_idx], use_mla)

                        if layer_ready_cb:
                            layer_ready_cb(layer_id)
                except Exception:
                    logger.exception(
                        "batch_get_v1: layerwise read failed at layer %d",
                        layer_id)
                    return [False] * num_pages

                # Mark matched pages as successful
                for page_idx in range(matched_pages):
                    results[page_idx] = True
            else:
                # Whole-page fallback for non-layer_first layouts
                results = []
                for page_idx in range(num_pages):
                    if page_idx < matched_pages:
                        block_id = match_result.physical_blocks[page_idx]
                        try:
                            block_data = self._get_block_shaped(block_id).clone()
                            transformed = self._flexkv_to_sglang(block_data)
                            host_token_start = host_indices[
                                page_idx * page_size].item()
                            self._mem_pool_host.set_from_flat_data_page(
                                host_token_start, transformed.flatten()
                            )
                            results.append(True)
                        except Exception:
                            logger.exception(
                                "batch_get_v1: failed to read block %d",
                                block_id)
                            results.append(False)
                    else:
                        results.append(False)

            hit_count = sum(results)
            end_time = time.perf_counter()
            miss_count = num_pages - hit_count

            if self._metrics:
                self._metrics.record_sglang_batch_get(hit_count, miss_count)
                if remote_fetched:
                    self._metrics.record_sglang_remote_fetch(
                        success=True, num_blocks=hit_count)

            if hit_count > 0:
                elapsed = end_time - start_time
                with self._metrics_lock:
                    self._prefetch_pgs.append(hit_count)
                    if elapsed > 0:
                        self._prefetch_bandwidth.append(
                            hit_count / elapsed * self._gb_per_page
                        )
                logger.debug("batch_get_v1: %d/%d pages loaded (layerwise=%s)",
                             hit_count, num_pages,
                             layout == _LAYOUT_LAYER_FIRST)
            return results

        except Exception:
            logger.exception("batch_get_v1 failed")
            if self._metrics:
                self._metrics.record_sglang_error(_OP_GET)
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

        token_ids = _get_token_ids(extra_info)
        if not token_ids or self._kv_manager is None:
            return [False] * len(keys)

        try:
            start_time = time.perf_counter()
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
                    if self._metrics:
                        self._metrics.record_sglang_batch_set(
                            blocks_written=0, blocks_deduped=num_pages)
                    return [True] * num_pages
                return [False] * num_pages

            # Fill data into CPU cache tensor for each new block.
            # Deduped blocks are always the prefix (radix tree matches from
            # start), so new blocks start at index num_deduped.
            num_deduped = num_pages - len(cpu_block_ids)
            layout = getattr(self._mem_pool_host, 'layout', _LAYOUT_LAYER_FIRST)

            if layout == _LAYOUT_LAYER_FIRST:
                # Layerwise path: copy layer-by-layer directly from kv_buffer.
                # Precompute invariants to avoid repeated work in O(layers*blocks) loop.
                num_layers = self._model_config.num_layers
                use_mla = self._model_config.use_mla
                block_views = []
                host_starts = []
                for i, block_id in enumerate(cpu_block_ids):
                    page_idx = num_deduped + i
                    block_views.append(self._get_block_shaped(block_id))
                    host_starts.append(
                        host_indices[page_idx * page_size].item())

                for layer_id in range(num_layers):
                    for i in range(len(cpu_block_ids)):
                        self._read_layer_from_host(
                            block_views[i], layer_id,
                            host_starts[i], use_mla)
            else:
                # Whole-page fallback for non-layer_first layouts
                for i, block_id in enumerate(cpu_block_ids):
                    page_idx = num_deduped + i
                    host_token_start = host_indices[
                        page_idx * page_size].item()
                    data_page = self._mem_pool_host.get_data_page(
                        host_token_start, flat=False
                    )
                    transformed = self._sglang_to_flexkv(data_page)
                    dst = self._get_block_view(block_id)
                    dst.copy_(transformed.flatten())

            # Launch H2DISK transfer (async, if SSD enabled)
            self._kv_manager.launch_cpu([task_id])

            end_time = time.perf_counter()
            num_written = len(cpu_block_ids)
            if num_written > 0:
                elapsed = end_time - start_time
                with self._metrics_lock:
                    self._backup_pgs.append(num_written)
                    if elapsed > 0:
                        self._backup_bandwidth.append(
                            num_written / elapsed * self._gb_per_page
                        )

            if self._metrics:
                self._metrics.record_sglang_batch_set(
                    blocks_written=len(cpu_block_ids),
                    blocks_deduped=num_deduped)

            logger.debug("batch_set_v1: wrote %d new pages, %d deduped (total %d)",
                         num_written, num_deduped, num_pages)
            return [True] * num_pages

        except Exception:
            logger.exception("batch_set_v1 failed")
            if self._metrics:
                self._metrics.record_sglang_error(_OP_SET)
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

    def get_stats(self) -> StorageMetrics:
        metrics = StorageMetrics()
        with self._metrics_lock:
            metrics.prefetch_pgs.extend(self._prefetch_pgs)
            metrics.backup_pgs.extend(self._backup_pgs)
            metrics.prefetch_bandwidth.extend(self._prefetch_bandwidth)
            metrics.backup_bandwidth.extend(self._backup_bandwidth)
            self._prefetch_pgs.clear()
            self._backup_pgs.clear()
            self._prefetch_bandwidth.clear()
            self._backup_bandwidth.clear()
        return metrics

    def clear(self) -> None:
        try:
            # TODO: implement KVManager cache engine clear when supported.
            # Currently only metrics are reset; cached blocks remain in the
            # radix tree and CPU cache tensor.
            logger.warning("clear() called but KVManager cache clear is not yet "
                           "implemented — only metrics are reset, cached data persists")
            with self._metrics_lock:
                self._prefetch_pgs.clear()
                self._backup_pgs.clear()
                self._prefetch_bandwidth.clear()
                self._backup_bandwidth.clear()
        except Exception:
            logger.exception("clear() failed")

    def __del__(self):
        try:
            if self._kv_manager is not None:
                self._kv_manager.shutdown()
        except Exception:
            pass
