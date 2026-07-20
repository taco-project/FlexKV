from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from flexkv.common.config import ModelConfig, LayerGroupSpec
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout
from flexkv.common.request import KVResponseStatus


@dataclass
class RegisterDPClientRequest:
    dp_client_id: int
    model_config: ModelConfig
    client_recv_port: str


@dataclass
class RegisterTPClientRequest:
    dp_client_id: int
    pp_rank: int
    device_id: int
    handles: List[TensorSharedHandle]
    gpu_layout: KVCacheLayout
    # Multi-group fields for heterogeneous KV cache layouts (including
    # DSA/NSA indexer-as-group). When None, the request describes a uniform
    # single-shape model and the legacy ``handles``/``gpu_layout`` path is used.
    layer_groups: Optional[List[LayerGroupSpec]] = None
    gpu_layouts: Optional[List[KVCacheLayout]] = None
    handles_per_group: Optional[List[List[TensorSharedHandle]]] = None
    # SWA transfer fileds
    swa_handles: Optional[List[TensorSharedHandle]] = None
    swa_layout: Optional[KVCacheLayout] = None
    # Optional heterogeneous groups sharing the SWA page id space.  These are
    # used by DSv4 to move SWA KV together with attention/indexer compress
    # states, whose GPU page widths and dtypes differ from SWA KV.
    swa_layer_groups: Optional[List[LayerGroupSpec]] = None
    swa_gpu_layouts: Optional[List[KVCacheLayout]] = None
    swa_handles_per_group: Optional[List[List[TensorSharedHandle]]] = None


@dataclass
class IsReadyRequest:
    dp_client_id: int


@dataclass
class PutRequest:
    dp_client_id: int
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]
    task_id: int = -1
    namespace: Optional[List[str]] = None


@dataclass
class GetRequest:
    dp_client_id: int
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]
    task_id: int = -1
    namespace: Optional[List[str]] = None


@dataclass
class PrefetchRequest:
    dp_client_id: int
    token_ids: np.ndarray
    task_id: int = -1
    namespace: Optional[List[str]] = None


@dataclass
class PutMatchRequest:
    dp_client_id: int
    token_ids: np.ndarray
    token_mask: Optional[np.ndarray]
    task_id: int = -1
    namespace: Optional[List[str]] = None


@dataclass
class GetMatchRequest:
    dp_client_id: int
    token_ids: np.ndarray
    token_mask: Optional[np.ndarray]
    cpu_only: bool = False
    task_id: int = -1
    namespace: Optional[List[str]] = None
    # SWA-aware match: clamp the Full-KV transfer to the reusable SWA window; the
    # SWA window is the trailing block of the returned mask.
    swa_aware: bool = False



@dataclass
class LaunchTaskRequest:
    dp_client_id: int
    task_ids: List[int]
    slot_mappings: List[np.ndarray]
    as_batch: bool = False
    batch_id: int = -1
    layerwise_transfer: bool = False
    counter_id: int = 0  # Counter set index for triple buffering eventfd notification
    swa_slot_mappings: Optional[List[Optional[np.ndarray]]] = None


@dataclass
class CancelTaskRequest:
    dp_client_id: int
    task_ids: List[int]


@dataclass
class WaitRequest:
    dp_client_id: int
    wait_task_ids: List[int]
    wait_timeout: float = 20.0
    completely: bool = False


# Used for async put/get
@dataclass
class TryWaitRequest:
    dp_client_id: int
    try_wait_task_ids: List[int]


@dataclass
class Response:
    dp_client_id: int
    task_id: Optional[int] = None
    mask: Optional[Dict[int, np.ndarray]] = None
    status: Optional[Dict[int, KVResponseStatus]] = None
    is_ready: bool = False
    error_msg: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status is not None and \
               all(self.status[task_id] == KVResponseStatus.SUCCESS for task_id in self.status)


@dataclass
class StartRequest:
    dp_client_id: int


@dataclass
class ShutdownRequest:
    dp_client_id: int


@dataclass
class CheckRunningRequest:
    dp_client_id: int
