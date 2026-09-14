from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Optional, List, Tuple, Union

import torch
import hashlib

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV, CacheConfig, LayerGroupSpec, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.pool import PoolEndpoint, PoolId
from flexkv.common.storage import StorageHandle, KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import DeviceType
from flexkv.storage.allocator import (
    CPUAllocator,
    GPUAllocator,
    HugePageAllocator,
    RemoteAllocator,
    SSDAllocator,
)


def _resolve_layer_groups(
    layer_groups: Optional[List[LayerGroupSpec]],
    default_dtype: torch.dtype,
) -> Optional[List[LayerGroupSpec]]:
    """Fill in LayerGroupSpec.dtype=None with the model's default dtype.

    KVCacheLayout's multi-group byte math requires every group to carry an
    explicit dtype; this resolves the inheritance once at storage setup time
    so downstream code can assume g.dtype is set.
    """
    if layer_groups is None:
        return None
    resolved: List[LayerGroupSpec] = []
    for g in layer_groups:
        resolved.append(g if g.dtype is not None else replace(g, dtype=default_dtype))
    return resolved


def _resolve_pool(pool_id: Optional[PoolId], is_swa: bool) -> PoolId:
    """Which pool a caller meant, from either selector.

    ``is_swa`` is kept as an alias because ~30 call sites spell it that way.
    Passing both is an error rather than a precedence rule: a caller that says
    ``pool_id=FULL_KV, is_swa=True`` has a bug, and silently honouring either
    would hide it.
    """
    if pool_id is None:
        return PoolId.from_is_swa(is_swa)
    if is_swa and not pool_id.is_swa:
        raise ValueError(
            f"conflicting pool selectors: pool_id={pool_id.name} and "
            f"is_swa=True; pass one")
    return pool_id


def _pool_from_kwargs(kwargs: Dict[str, Any]) -> PoolId:
    """``allocate``'s pool selector, popped out of its **kwargs bag.

    Popped, not read: ``is_swa``/``pool_id`` are routing information for this
    registry, and the per-device allocator branches below read the same bag.
    Leaving them in has bitten before -- an allocator that forwards **kwargs
    would see a key it has no parameter for.
    """
    pool_id = kwargs.pop('pool_id', None)
    is_swa = kwargs.pop('is_swa', False)
    return _resolve_pool(pool_id, is_swa)


class StorageEngine:
    def _cpu_allocator(self) -> type[CPUAllocator] | type[HugePageAllocator]:
        if self._cache_config.use_hugepage_cpu_buffer:
            return HugePageAllocator
        return CPUAllocator

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 num_layers_per_pp_stage: int,
                 swa_layer_groups: Optional[List[LayerGroupSpec]] = None):
        """Initialize storage engine"""
        # One registry, keyed by the endpoint (pool + tier) plus device id.
        #
        # It used to be two dicts -- ``_storage_handles`` and
        # ``_swa_storage_handles`` -- because the SAME physical device_id holds
        # both a main-KV GPU handle and an independent SWA one, and a
        # (DeviceType, device_id) key cannot tell them apart. The pool is what
        # distinguishes them, so the pool is in the key; two dicts selected by
        # a bool were the same statement made in a shape that could only ever
        # hold two pools, and every accessor had to re-derive the container
        # from that bool.
        self._handles: Dict[Tuple[PoolEndpoint, int], StorageHandle] = {}
        self._model_config = model_config
        self._cache_config = cache_config

        # Resolve per-group dtype inheritance in place so every downstream
        # consumer (KVCacheLayout byte math, worker stride computation,
        # TransferEngine) sees explicit dtypes.
        if self._model_config.layer_groups is not None:
            self._model_config.layer_groups = _resolve_layer_groups(
                self._model_config.layer_groups, self._model_config.dtype
            )
        self._swa_layer_groups = _resolve_layer_groups(
            swa_layer_groups, torch.uint8
        )
        if (
            self._swa_layer_groups is not None
            and GLOBAL_CONFIG_FROM_ENV.cpu_layout_type
            != KVCacheLayoutType.BLOCKFIRST
        ):
            raise ValueError(
                "SWA multi-group sidecars require FLEXKV_CPU_LAYOUT=BLOCKFIRST"
            )

        # For multi-group, the CPU/SSD/Remote buffer is sized in BYTES
        # (kv_shape[1] = bytes_per_block, summed with per-group dtype.itemsize),
        # so the underlying allocator must use uint8.  Single-group keeps its
        # native dtype.
        is_multi_group = self._model_config.layer_groups is not None
        buffer_dtype = torch.uint8 if is_multi_group else self._model_config.dtype

        if self._cache_config.enable_cpu:
            self._cpu_layout: Optional[KVCacheLayout] = KVCacheLayout(
                type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
                num_layer=num_layers_per_pp_stage,
                num_block=self._cache_config.num_cpu_blocks,
                tokens_per_block=self._cache_config.tokens_per_block,
                num_head=self._model_config.num_kv_heads_per_node,
                head_size=self._model_config.head_size,
                kv_dim=self._model_config.kv_dim,
                num_kv_heads=self._model_config.num_kv_heads,
                layer_groups=self._model_config.layer_groups,
                tp_size=self._model_config.tp_size,
            )
            self.allocate(
                device_type=DeviceType.CPU,
                layout=self._cpu_layout,
                dtype=buffer_dtype,
            )

        if self._cache_config.enable_ssd:
            if not GLOBAL_CONFIG_FROM_ENV.ssd_layout_type == self._cpu_layout.type:
                raise ValueError(f"SSD layout type must be the same as CPU layout type: {self._cpu_layout.type}")
            self._ssd_layout: Optional[KVCacheLayout] = KVCacheLayout(
                type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
                num_layer=num_layers_per_pp_stage,
                num_block=self._cache_config.num_ssd_blocks,
                tokens_per_block=self._cache_config.tokens_per_block,
                num_head=self._model_config.num_kv_heads_per_node,
                head_size=self._model_config.head_size,
                kv_dim=self._model_config.kv_dim,
                num_kv_heads=self._model_config.num_kv_heads,
                layer_groups=self._model_config.layer_groups,
                tp_size=self._model_config.tp_size,
            )
            self.allocate(
                device_type=DeviceType.SSD,
                layout=self._ssd_layout,
                dtype=buffer_dtype,
                cache_dir=self._cache_config.ssd_cache_dir,
                max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb
            )

        if self._cache_config.enable_remote:
            if self._cache_config.use_mooncake_store_backend:
                flexkv_logger.info(
                    "[StorageEngine] mooncake-store backend: hot CPU buffer will be "
                    "registered directly (no separate contributed region)."
                )
            else:
                if not GLOBAL_CONFIG_FROM_ENV.remote_layout_type == self._cpu_layout.type:
                    raise ValueError(f"Remote layout type must be the same as CPU layout type: {self._cpu_layout.type}")
                self._remote_layout: Optional[KVCacheLayout] = KVCacheLayout(
                    type=GLOBAL_CONFIG_FROM_ENV.remote_layout_type,
                    num_layer=num_layers_per_pp_stage,
                    num_block=self._cache_config.num_remote_blocks,
                    tokens_per_block=self._cache_config.tokens_per_block,
                    num_head=self._model_config.num_kv_heads_per_node,
                    head_size=self._model_config.head_size,
                    kv_dim=self._model_config.kv_dim,
                    num_kv_heads=self._model_config.num_kv_heads,
                    layer_groups=self._model_config.layer_groups,
                    tp_size=self._model_config.tp_size,
                )
                self.allocate(
                    device_type=DeviceType.REMOTE,
                    layout=self._remote_layout,
                    dtype=buffer_dtype,
                    file_path=self._cache_config.remote_cache_path,
                    remote_config_custom = self._cache_config.remote_config_custom
                )

        # SWA pool allocate
        self._swa_cpu_layout: Optional[KVCacheLayout] = None
        self._swa_ssd_layout: Optional[KVCacheLayout] = None
        self._swa_remote_layout: Optional[KVCacheLayout] = None
        swa_cfg = getattr(self._cache_config, "swa", None)
        if swa_cfg is not None and swa_cfg.enabled:
            swa_tokens_per_block = self._cache_config.tokens_per_block
            if self._cache_config.enable_cpu:
                # uint8, num_head=1, kv_dim=1, num_kv_heads=1; per-token-per-layer bytes -> head_size.
                self._swa_cpu_layout = KVCacheLayout(
                    type=GLOBAL_CONFIG_FROM_ENV.cpu_layout_type,
                    num_layer=swa_cfg.num_swa_layers,
                    num_block=swa_cfg.num_slots,
                    tokens_per_block=swa_tokens_per_block,
                    num_head=1,
                    head_size=swa_cfg.bytes_per_token_per_layer,
                    kv_dim=1,
                    num_kv_heads=1,
                    layer_groups=self._swa_layer_groups,
                    tp_size=self._model_config.tp_size,
                )
                self.allocate(
                    device_type=DeviceType.CPU,
                    layout=self._swa_cpu_layout,
                    dtype=torch.uint8,
                    device_id=0,
                    raw_data=None,
                    pool_id=PoolId.SWA,
                    pin_memory=swa_cfg.pin_memory,
                )


            if self._cache_config.enable_ssd and swa_cfg.num_ssd_slots > 0:
                if self._swa_cpu_layout is None:
                    raise ValueError("SWA SSD tier requires the SWA CPU tier")
                if not GLOBAL_CONFIG_FROM_ENV.ssd_layout_type == self._swa_cpu_layout.type:
                    raise ValueError(
                        f"SWA SSD layout type must match SWA CPU layout type: "
                        f"{self._swa_cpu_layout.type}"
                    )
                self._swa_ssd_layout = KVCacheLayout(
                    type=GLOBAL_CONFIG_FROM_ENV.ssd_layout_type,
                    num_layer=swa_cfg.num_swa_layers,
                    num_block=swa_cfg.num_ssd_slots,
                    tokens_per_block=swa_tokens_per_block,
                    num_head=1,
                    head_size=swa_cfg.bytes_per_token_per_layer,
                    kv_dim=1,
                    num_kv_heads=1,
                    layer_groups=self._swa_layer_groups,
                    tp_size=self._model_config.tp_size,
                )
                self.allocate(
                    device_type=DeviceType.SSD,
                    layout=self._swa_ssd_layout,
                    dtype=torch.uint8,
                    device_id=0,
                    raw_data=None,
                    pool_id=PoolId.SWA,
                    cache_dir=self._cache_config.ssd_cache_dir,
                    max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb,
                )

            if self._cache_config.enable_remote and swa_cfg.num_remote_slots > 0:
                if self._cache_config.use_mooncake_store_backend:
                    flexkv_logger.info(
                        "[StorageEngine] mooncake-store backend: hot SWA CPU buffer will be "
                        "registered directly (no separate contributed region)."
                    )
                else:
                    if self._swa_cpu_layout is None:
                        raise ValueError("SWA REMOTE tier requires the SWA CPU tier")
                    if not GLOBAL_CONFIG_FROM_ENV.remote_layout_type == self._swa_cpu_layout.type:
                        raise ValueError(
                            f"SWA Remote layout type must match SWA CPU layout type: "
                            f"{self._swa_cpu_layout.type}"
                        )
                    self._swa_remote_layout = KVCacheLayout(
                        type=GLOBAL_CONFIG_FROM_ENV.remote_layout_type,
                        num_layer=swa_cfg.num_swa_layers,
                        num_block=swa_cfg.num_remote_slots,
                        tokens_per_block=swa_tokens_per_block,
                    num_head=1,
                    head_size=swa_cfg.bytes_per_token_per_layer,
                    kv_dim=1,
                    num_kv_heads=1,
                    layer_groups=self._swa_layer_groups,
                    tp_size=self._model_config.tp_size,
                )
                    swa_remote_path = self._cache_config.remote_cache_path
                    if isinstance(swa_remote_path, str):
                        swa_remote_path = swa_remote_path + "_swa"
                    elif isinstance(swa_remote_path, list):
                        swa_remote_path = [path + "_swa" for path in swa_remote_path]

                    self.allocate(
                        device_type=DeviceType.REMOTE,
                        layout=self._swa_remote_layout,
                        dtype=torch.uint8,
                        device_id=0,
                        raw_data=None,
                        pool_id=PoolId.SWA,
                        file_path=swa_remote_path,
                        remote_config_custom=self._cache_config.remote_config_custom,
                    )


    def register_gpu_blocks(self,
                            gpu_blocks: List[TensorSharedHandle],
                            gpu_layout: KVCacheLayout,
                            device_id: int = 0,
                            dtype: torch.dtype = torch.float16) -> None:
        self.allocate(
            device_type=DeviceType.GPU,
            layout=gpu_layout,
            dtype=dtype,
            device_id=device_id,
            raw_data=gpu_blocks
        )

    def register_swa_gpu_blocks(self,
                                swa_blocks: List[TensorSharedHandle],
                                swa_layout: KVCacheLayout,
                                device_id: int = 0,
                                dtype: torch.dtype = torch.uint8) -> None:
        """Register the SWA dedicated GPU pool (channel B).

        Registered under ``SWA@GPU`` at the ORIGINAL physical ``device_id``,
        which is what keeps it distinct from the main-KV GPU handle for the
        same device: the pool is in the key, so there is no collision and no
        magic offset. Reuses the standard GPU allocate path
        (GPUAllocator.from_raw_data), which maps each TensorSharedHandle via
        CUDA IPC into the worker process.
        """
        self.allocate(
            device_type=DeviceType.GPU,
            layout=swa_layout,
            dtype=dtype,
            device_id=device_id,
            raw_data=swa_blocks,
            pool_id=PoolId.SWA,
        )

    def get_swa_storage_handle(self, device_id: int = 0) -> StorageHandle:
        """Return the SWA dedicated GPU StorageHandle for a physical device."""
        return self.get_storage_handle(
            DeviceType.GPU, device_id, pool_id=PoolId.SWA
        )

    def has_swa_storage_handle(self, device_id: int = 0) -> bool:
        return self.has_storage_handle(
            DeviceType.GPU, device_id, pool_id=PoolId.SWA
        )

    def allocate(self,
                 device_type: DeviceType,
                 layout: KVCacheLayout,
                 dtype: torch.dtype,
                 device_id: int = 0,
                 raw_data: Optional[Union[List[TensorSharedHandle], List[str], str]] = None,
                 **kwargs: Any) -> bool:
        """
        Create and add an allocator for specified device.

        Args:
            device_type: Type of the device (CPU, GPU, SSD, REMOTE).
            layout: Layout of kv cache.
            dtype: Data type of tensors.
            device_id: Device ID (default 0).
            raw_data: Optional raw data to be used for initialization.
                      The expected type depends on ``device_type``:

                      * ``DeviceType.CPU``    – ``torch.Tensor``
                      * ``DeviceType.GPU``    – ``List[TensorSharedHandle]`` or
                                               ``List[torch.Tensor]``
                      * ``DeviceType.SSD``    – ``str`` or ``List[str]``
                        (file path(s) to existing SSD cache files)
                      * ``DeviceType.REMOTE`` – ``str`` or ``List[str]``
                        (remote file path(s))
            **kwargs: Additional arguments for specific allocator types
                     (e.g., pin_memory for CPU, file_path for Disk).

        Returns:
            bool: True if allocator created successfully, False if already exists.
        """
        # The pool is part of the key, so the same physical device_id can hold
        # both a main-KV and an SWA handle without collision.
        pool_id = _pool_from_kwargs(kwargs)
        key = (PoolEndpoint(pool_id, device_type), device_id)
        if key in self._handles:
            return False

        storage_handle: StorageHandle
        if device_type == DeviceType.CPU:
            cpu_allocator = self._cpu_allocator()
            pin_memory = kwargs.get('pin_memory', False)
            page_size_bytes = kwargs.get(
                'page_size_bytes',
                self._cache_config.hugepage_size_bytes,
            )
            if raw_data is not None:
                assert isinstance(raw_data, torch.Tensor), \
                    "raw_data for CPUAllocator must be Tensor"
                storage_handle = cpu_allocator.from_raw_data(
                    data=raw_data,  # type: ignore
                    layout=layout,
                    dtype=dtype,
                    pin_memory=pin_memory,
                    page_size_bytes=page_size_bytes,
                )
            else:
                storage_handle = cpu_allocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    pin_memory=pin_memory,
                    page_size_bytes=page_size_bytes,
                )
        elif device_type == DeviceType.GPU:
            num_chunks = kwargs.get('num_chunks', 1)
            if raw_data is not None:
                assert isinstance(raw_data, list) and \
                    (all(isinstance(x, TensorSharedHandle) for x in raw_data) or \
                     all(isinstance(x, torch.Tensor) for x in raw_data)), \
                    "raw_data for GPUAllocator must be List[TensorSharedHandle] or List[Tensor]"
                storage_handle = GPUAllocator.from_raw_data(
                    data=raw_data,  # type: ignore
                    layout=layout,
                    dtype=dtype,
                    device_id=device_id
                )
            else:
                storage_handle = GPUAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    num_chunks=num_chunks,
                    device_id=device_id
                )
        elif device_type == DeviceType.SSD:
            cache_dir = kwargs.get('cache_dir')
            max_file_size_gb = kwargs.get('max_file_size_gb', -1)
            if raw_data is not None:
                assert isinstance(raw_data, str) or \
                    (isinstance(raw_data, list) and all(isinstance(x, str) for x in raw_data)), \
                    "raw_data for SSDAllocator must be str or List[str]"
                storage_handle = SSDAllocator.from_raw_data(
                    data=raw_data,  # type: ignore
                    layout=layout,
                    dtype=dtype,
                )
            else:
                if not cache_dir:
                    raise ValueError("cache_dir is required for SSD allocator")
                server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
                hash_value = hashlib.md5(server_recv_port.encode()).hexdigest()
                rand_suffix = f"{hash_value[:6]}"
                file_prefix = f"flexkv_ssdcache_{rand_suffix}"
                # Physically separate each non-default pool's SSD cache files
                # from the main-KV ones: without this, both share the same
                # prefix (server_recv_port is identical in-process) and
                # overwrite each other on disk. FULL_KV keeps the bare prefix
                # so existing cache files stay addressable; SWA still resolves
                # to the "_swa" suffix it always had.
                if pool_id is not PoolId.FULL_KV:
                    file_prefix = f"{file_prefix}_{pool_id.name.lower()}"
                storage_handle = SSDAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    cache_dir=cache_dir,
                    file_prefix=file_prefix,
                    max_file_size_gb=max_file_size_gb
                )
        elif device_type == DeviceType.REMOTE:
            file_path = kwargs.get('file_path')
            remote_config_custom = kwargs.get('remote_config_custom')
            if raw_data is not None:
                if (isinstance(raw_data, str) or \
                    (isinstance(raw_data, list) and all(isinstance(x, str) for x in raw_data))):
                    if not isinstance(remote_config_custom, dict):
                        raise TypeError("remote_config_custom for RemoteAllocator.from_raw_data must be dict[str, Any]")
                    storage_handle = RemoteAllocator.from_raw_data(
                        data=raw_data,  # type: ignore
                        layout=layout,
                        dtype=dtype,
                        remote_config_custom=remote_config_custom
                    )
                else:
                    raise TypeError("raw_data for RemoteAllocator must be str or List[str]")
            else:
                if not file_path:
                    raise ValueError("file_path is required for remote allocator")
                if not isinstance(remote_config_custom, dict):
                    raise TypeError("remote_config_custom for RemoteAllocator must be dict[str, Any]")
                storage_handle = RemoteAllocator.allocate(
                    layout=layout,
                    dtype=dtype,
                    file_path=file_path,
                    remote_config_custom=remote_config_custom
                )
        else:
            raise ValueError(f"Unsupported device type: {device_type}")
        self._handles[key] = storage_handle
        return True

    def get_storage_handle(self,
                           device_type: DeviceType,
                           device_id: int = 0,
                           is_swa: bool = False,
                           pool_id: Optional[PoolId] = None) -> StorageHandle:
        """
        Get accessible handle for specified blocks.

        Args:
            device_type: Type of the device to get handle from.
            device_id: Device ID.
            is_swa: Legacy alias for ``pool_id=PoolId.SWA``.
            pool_id: Which pool's storage to fetch.
        """
        endpoint = PoolEndpoint(_resolve_pool(pool_id, is_swa), device_type)
        key = (endpoint, device_id)
        if key not in self._handles:
            raise ValueError(
                f"Storage handle not found for endpoint: {endpoint}, "
                f"device id: {device_id}"
            )
        return self._handles[key]

    def has_storage_handle(self,
                           device_type: DeviceType,
                           device_id: int = 0,
                           is_swa: bool = False,
                           pool_id: Optional[PoolId] = None) -> bool:
        """
        Check if storage handle exists for given endpoint and device id.

        Args:
            device_type: Type of the device.
            device_id: Device ID.
            is_swa: Legacy alias for ``pool_id=PoolId.SWA``.
            pool_id: Which pool's storage to check.
        """
        endpoint = PoolEndpoint(_resolve_pool(pool_id, is_swa), device_type)
        return (endpoint, device_id) in self._handles

    def pools_at(self, device_type: DeviceType, device_id: int = 0) -> List[PoolId]:
        """Which pools have storage on this tier, in PoolId order.

        The registry answers this directly now. With two dicts the same
        question meant asking each one separately and stitching the answers
        together, which is why the engine's "does this edge carry SWA?" tests
        were spelled differently in each place that asked.
        """
        return sorted(
            endpoint.pool_id
            for (endpoint, did) in self._handles
            if endpoint.device_type == device_type and did == device_id
        )
