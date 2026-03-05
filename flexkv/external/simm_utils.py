import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from flexkv.common.config import CacheConfig

import torch

# Third Party
try:
    from simm.kv import BlockView, Store, register_mr, set_flag
except ImportError as e:
    # TODO: add install guide after SiMM opensource
    raise ImportError(
        "Please install simm by following the instructions at "
        "to run SGLang with SimmConnector."
    ) from e

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
SETUP_TIMEOUT = 600  # 10min

from flexkv.common.debug import flexkv_logger as logger
from flexkv.common.block import SequenceMeta
from flexkv.common.type import MatchResultAccel
import numpy as np

# Avoid circular import: CacheEngine and RadixNode imported where needed
FLEXKV_SIMM_JSON_ENV_VAR = "FLEXKV_SIMM_CONFIG_PATH"


@dataclass
class SiMMConfig:
    manager_address: str
    clnt_threadpool_size: int
    enable_profile: bool

    @staticmethod
    def from_file() -> "SiMMConfig":
        """Load the config from a JSON file."""
        if os.environ.get(FLEXKV_SIMM_JSON_ENV_VAR) is None:
            raise RuntimeError(
                f"Config file path not set. Please set {FLEXKV_SIMM_JSON_ENV_VAR}"
            )
        file_path = os.environ.get(FLEXKV_SIMM_JSON_ENV_VAR)
        try:
            with open(file_path) as fin:
                config = json.load(fin)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {file_path}: {str(e)}")

        if "manager_address" not in config:
            raise ValueError("Manager_address is required in config file")

        return SiMMConfig(
            manager_address=config.get("manager_address"),
            clnt_threadpool_size=config.get("clnt_threadpool_size", 10),
            enable_profile=config.get("enable_profile", False),
        )

    @staticmethod
    def load_from_extra_config(extra_config: dict) -> "SiMMConfig":
        """Load config from extra_config dictionary."""
        if "manager_address" not in extra_config:
            raise ValueError("manager_address is required in extra_config")

        return SiMMConfig(
            manager_address=extra_config.get("manager_address"),
            clnt_threadpool_size=extra_config.get("clnt_threadpool_size", 10),
            enable_profile=extra_config.get("enable_profile", False),
        )


def get_current_process_numa() -> int:
    """
    Return value: numa_node of current process, failed return -1
    """
    try:
        # get current cpu
        with open("/proc/self/stat", "r") as f:
            stat_data = f.read()

        # the 39th field is processor
        fields = stat_data.split()
        if len(fields) < 39:
            return -1
        current_cpu = int(fields[38])
        numa_path = f"/sys/devices/system/cpu/cpu{current_cpu}/node0"
        if os.path.exists(numa_path) and os.path.islink(numa_path):
            link_target = os.readlink(numa_path)
            # parse numa node from path
            match = re.search(r"node(\d+)$", link_target)
            if match:
                return int(match.group(1))

        return -1
    except Exception:
        return -1


def get_numa_nic_mapping() -> Dict[int, List[str]]:
    """
    Return value: Dict[numa_node, List(rdma_device_name)]
    """
    ib_root = "/sys/class/infiniband"
    device_map = defaultdict(list)

    if not os.path.exists(ib_root):
        logger.error(f"SiMM ERROR: {ib_root} not found. Are RDMA drivers loaded?")
        return []

    for device_name in os.listdir(ib_root):
        numa_path = os.path.join(ib_root, device_name, "device", "numa_node")
        numa_node = -1  # default value, if system is UMA.

        try:
            if os.path.exists(numa_path):
                with open(numa_path, "r") as f:
                    content = f.read().strip()
                    numa_node = int(content)
        except (IOError, ValueError):
            pass
        device_map[numa_node].append(device_name)

    return device_map

class SiMMClient:
    """
    SiMM client. Initialization follows SGLang hicache_simm.py:
    config from extra_config (if manager_address) or from JSON file, then
    NUMA-aware NIC, set_flag, Store(), warmup. MR is registered later via register_mr().
    """

    def __init__(
        self,
        manager_address: Optional[str] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ):
        self.store = None
        self.mr_ext = None
        self.extra_backend_tag = None
        self.config = None

        # 1) Load config: extra_config (with manager_address) > from_file() > manager_address only
        if extra_config is not None and extra_config.get("manager_address") is not None:
            self.config = SiMMConfig.load_from_extra_config(extra_config)
            logger.info("SiMM configuration loaded from extra_config.")
        else:
            try:
                self.config = SiMMConfig.from_file()
                logger.info("SiMM configuration loaded from file.")
            except (RuntimeError, ValueError):
                if manager_address:
                    self.config = SiMMConfig(
                        manager_address=manager_address,
                        clnt_threadpool_size=10,
                        enable_profile=False,
                    )
                    logger.info("SiMM configuration from manager_address.")
                else:
                    raise ValueError(
                        "SiMM config: set {} or pass manager_address or extra_config with manager_address".format(
                        FLEXKV_SIMM_JSON_ENV_VAR
                    )
                    ) from None

        if extra_config and "extra_backend_tag" in extra_config:
            self.extra_backend_tag = extra_config["extra_backend_tag"]
            logger.info("Using extra_backend_tag: %s", self.extra_backend_tag)

        # 2) NUMA-aware NIC (same as hicache_simm.py)
        nic_mapping = get_numa_nic_mapping()
        logger.info("SiMM NUMA-aware allocation: %s", nic_mapping)
        current_numa = get_current_process_numa()
        if current_numa >= 0:
            rdma_devices = nic_mapping.get(current_numa)
            if rdma_devices is not None and len(rdma_devices) > 0:
                rdma_device_str = ",".join(rdma_devices)
                os.environ["SICL_NET_DEVICES"] = rdma_device_str
                logger.info("SiMM using rdma %s", rdma_device_str)

        # 3) Log path and set_flag before Store() (same as hicache_simm.py)
        filename_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_path = f"/var/log/simm/{filename_ts}-{os.getpid()}/simm_clnt.log"
        parts = self.config.manager_address.split(":")
        cm_ip = parts[0] if parts else "127.0.0.1"
        cm_port = parts[1] if len(parts) > 1 else "9400"
        set_flag("cm_primary_node_ip", cm_ip)
        set_flag("cm_primary_node_port", cm_port)
        set_flag("clnt_log_file", log_file_path)
        set_flag("clnt_thread_pool_size", str(self.config.clnt_threadpool_size))

        self.store = Store()
        logger.info("SiMM store setup successfully.")
        self.mr_ext = None

        self.warmup_simm_client()
        logger.info("SiMM store warmup successfully.")

    def register_mr(self, ptr: int, size: int) -> None:
        """Register a memory region with SiMM (zero-copy). Call after __init__ when buffer is ready."""
        self.mr_ext = register_mr(ptr, size)

    def warmup_simm_client(self) -> None:
        """Dry run a key to warmup SiMM client (same as hicache_simm.py warmup())."""
        logger.info("begin warm up SiMM client")
        start_time = time.perf_counter_ns()
        warmup_key = "flexkv_simm_warmup_key" + uuid.uuid4().hex
        warmup_tensor = torch.frombuffer(
            bytearray(warmup_key.encode()), dtype=torch.uint8
        )
        warmup_size = 4 * 1024  # 4 KB
        block = self.store.allocate(warmup_size)
        block_ = block.as_ref()
        block_[: len(warmup_key)] = warmup_tensor
        if self.store.put(warmup_key, block.view()) != 0:
            logger.warning("SiMM client warmup put key %s failed", warmup_key)
        if not self.store.exists(warmup_key):
            logger.warning("SiMM client warmup key %s not exists", warmup_key)
        got_block = self.store.allocate(warmup_size)
        if self.store.get(warmup_key, got_block.view()) < 0:
            logger.warning("SiMM client warmup get key %s failed", warmup_key)
        if not all(got_block.as_ref()[: len(warmup_key)] == warmup_tensor):
            logger.warning("SiMM client warmup key %s data wrong", warmup_key)
        logger.info(
            "finish SiMM client warm up, cost %.2f us",
            (time.perf_counter_ns() - start_time) / 1000,
        ) 

    def exists(self, key) -> bool:
        exist_result = self._batch_exist_impl([key])
        return exist_result[0]

    def batch_exists(
        self, 
        keys_strs: List[str], 
        extra_info: Optional[Dict[str, Any]] = None
    ) -> int:

        exist_result = self._batch_exist_impl(keys_strs)
        for i in range(len(keys_strs)):
            if not exist_result[i]:
                return i
        return len(keys_strs)

    def batch_get_v1(
        self,
        cpu_ptrs: List[int],
        block_sizes: List[int],
        keys: List[str],
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> List[bool]:
        if self.extra_backend_tag is not None:
            keys = [f"{self.extra_backend_tag}_{key}" for key in keys]
        get_results = self._get_batch_zero_copy_impl(
            keys, cpu_ptrs, block_sizes
        )
        return self._check_success(get_results, is_set_operate=False)

    def batch_set_v1(
        self,
        cpu_ptrs: List[int],
        block_sizes_list: List[int],
        keys_strs: List[str],
        extra_info: Optional[Dict[str, Any]] = None,
        ) -> List[bool]:
        if self.extra_backend_tag is not None:
            keys_strs = [f"{self.extra_backend_tag}_{k}" for k in keys_strs]
        exist_result = self._batch_exist_impl(keys_strs)
        # maybe we don't need to check again, since flexkv cache engine will check this
        set_keys = []
        set_buffer_ptrs = []
        set_buffer_sizes = []
        set_indices = []
        set_results = [-1] * len(keys_strs)
        total_size = 0
        for i in range(len(keys_strs)):
            if not exist_result[i]:
                set_keys.append(keys_strs[i])
                set_buffer_ptrs.append(cpu_ptrs[i])
                set_buffer_sizes.append(block_sizes_list[i])
                set_indices.append(i)
                total_size += block_sizes_list[i]
            else:
                set_results[i] = 0

        # Only set non-existing keys to storage
        if len(set_keys) > 0:
            put_results = self._put_batch_zero_copy_impl(
                set_keys, set_buffer_ptrs, set_buffer_sizes
            )
            for i in range(len(set_indices)):
                set_results[set_indices[i]] = put_results[i]

        return self._check_success(set_results, is_set_operate=True)

    def batch_delete_v1(
        self,
        keys_strs: List[str],
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> List[bool]:
        delete_results = self._batch_delete_impl(keys_strs)
        return self._check_success(delete_results, is_set_operate=False)

    def _check_success(self, results: List[int], is_set_operate: bool) -> List[bool]:
        return [k_res > 0 if is_set_operate else k_res == 0 for k_res in results] 

    def _put_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        block_views = []
        for i in range(len(buffer_ptrs)):
            block_view = BlockView.from_buffer(
                buffer_ptrs[i], buffer_sizes[i], self.mr_ext
            )
            block_views.append(block_view)
        return self.store.mput(key_strs, block_views)

    def _get_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        block_views = []
        for i in range(len(buffer_ptrs)):
            block_view = BlockView.from_buffer(
                buffer_ptrs[i], buffer_sizes[i], self.mr_ext
            )
            block_views.append(block_view)
        return self.store.mget(key_strs, block_views)

    def _batch_exist_impl(self, key_strs: List[str]) -> List[bool]:
        return self.store.mexists(key_strs)


    

class SimmCacheEngine:
    """Minimal cache engine for SIMM backend: match via batch_exists, no local index."""

    def __init__(self, cache_config: "CacheConfig"):
        self.tokens_per_block = cache_config.tokens_per_block
        manager_address = (
            getattr(cache_config, "simm_manager_address", None)
            or os.environ.get("FLEXKV_SIMM_MANAGER_ADDRESS", "")
        )
        extra_config = None
        if manager_address:
            extra_config = {
                "manager_address": manager_address,
                "clnt_threadpool_size": getattr(
                    cache_config, "simm_clnt_threadpool_size", 10
                ),
                "enable_profile": getattr(
                    cache_config, "simm_enable_profile", False
                ),
            }
            if getattr(cache_config, "simm_extra_backend_tag", None) is not None:
                extra_config["extra_backend_tag"] = cache_config.simm_extra_backend_tag
        self.simm_client = SiMMClient(
            manager_address=manager_address or None,
            extra_config=extra_config,
        )

    def reset(self) -> None:
        pass

    def start(self) -> None:
        pass

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        # block_hashes has length num_blocks; one key per block
        keys = [str(sequence_meta.block_hashes[i]) for i in range(sequence_meta.num_blocks)]
        matched_length = self.simm_client.batch_exists(keys)
        return MatchResultAccel(num_matched_blocks=matched_length,
                                num_ready_matched_blocks=matched_length,
                                last_ready_node=None,
                                last_node=None,
                                last_node_matched_length=matched_length,
                                physical_blocks=np.arange(matched_length, dtype=np.int64),
                                matched_pos="global")

    # insert is skiped as the real index is managed by simm itself
    def insert(self, sequence_meta: SequenceMeta, physical_block_ids: np.ndarray) -> np.ndarray:
        return np.array([], dtype=np.int64)

    def lock_node(self, node: Any) -> None:
        pass

    def unlock(self, node: Any) -> None:
        pass

    def set_ready(self, node: Any, ready: bool, ready_length: int) -> None:
        pass

    # TAKE is skipped as the real index is managed by simm itself
    def take(
        self,
        num_required_blocks: int,
        protected_node: Optional[Any] = None,
        strict: bool = True,
    ) -> np.ndarray:
        return np.zeros(num_required_blocks, dtype=np.int64)
