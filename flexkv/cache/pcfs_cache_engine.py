from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, List, Dict

import numpy as np
import torch

from flexkv.c_ext import CRadixNode
from flexkv import c_ext
from flexkv.cache.mempool import Mempool
from flexkv.cache.radix_remote import LocalRadixTree, DistributedRadixTree
from flexkv.cache.redis_meta import RedisMetaChannel as _PyRedisMetaChannel
from flexkv.cache.redis_meta import RedisMeta
from flexkv.common.block import SequenceMeta
from flexkv.common.exceptions import InvalidConfigError, NotEnoughSpaceError
if TYPE_CHECKING:
    from flexkv.common.config import CacheConfig
from flexkv.common.transfer import DeviceType
from flexkv.common.type import MatchResultAccel


class PCFSCacheEngine:
    def __init__(self,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                 *,
                 # Optional runtime wiring for remote/local trees
                 local_max_num_blocks: Optional[int] = None,
                 local_lease_ttl_ms: int = 100000,
                 local_renew_lease_ms: int = 0,
                 local_refresh_batch_size: int = 256,
                 local_idle_sleep_ms: int = 10,
                 local_lt_pool_initial_capacity: int = 0,
                 remote_max_num_blocks: Optional[int] = None,
                 remote_node_id: int = 0,
                 remote_lt_pool_initial_capacity: int = 0,
                 remote_refresh_batch_size: int = 128,
                 remote_rebuild_interval_ms: int = 1000,
                 remote_idle_sleep_ms: int = 10,
                 meta: Optional[RedisMeta] = None) -> None:
        if num_total_blocks <= 0:
            raise InvalidConfigError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise InvalidConfigError(
                f"Invalid tokens_per_block: {tokens_per_block}, tokens_per_block must be a power of 2"
            )

        # PCFS cache engine is always REMOTE device type
        self.device_type = DeviceType.REMOTE
        self._meta: Optional[RedisMeta] = meta
        # Mapping: node_id -> list of PCFS file_nodeids
        self.nid_to_file_nodeids: Dict[int, List[int]] = {}
        # Partition parameter used for mapping block_id to file index
        self.round_robin: int = 1

        # Local index (authoritative for mutations)
        self.local_index = LocalRadixTree(
            tokens_per_block=tokens_per_block,
            max_num_blocks=int(local_max_num_blocks or num_total_blocks),
            lease_ttl_ms=int(local_lease_ttl_ms),
            renew_lease_ms=int(local_renew_lease_ms),
            refresh_batch_size=int(local_refresh_batch_size),
            idle_sleep_ms=int(local_idle_sleep_ms),
            lt_pool_initial_capacity=int(local_lt_pool_initial_capacity),
        )


        # Remote reference index (read-only, built from Redis)
        self.remote_index = DistributedRadixTree(
            tokens_per_block=tokens_per_block,
            max_num_blocks=int(remote_max_num_blocks or num_total_blocks),
            node_id=int(remote_node_id),
            lt_pool_initial_capacity=int(remote_lt_pool_initial_capacity),
            refresh_batch_size=int(remote_refresh_batch_size),
            rebuild_interval_ms=int(remote_rebuild_interval_ms),
            idle_sleep_ms=int(remote_idle_sleep_ms),
        )
        # defer channel start to start(meta)

        # Local memory pool for physical blocks on this device
        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio

    def start(self) -> None:
        if self._meta is None:
            raise InvalidConfigError("RedisMeta is not provided; ensure from_cache_config stores it or pass it to start().")
        self.remote_ch = self._meta.get_redis_meta_channel("RPCFSB")
        self.local_ch = self._meta.get_redis_meta_channel("PCFSB")
                # Load and store mapping of node_id -> file_nodeids from Redis
        try:
            self.nid_to_file_nodeids = self._meta.load_pcfs_file_nodeids()
        except Exception:
            raise InvalidConfigError("Failed to load PCFS file nodeids from Redis")
        self.local_index.start(self.local_ch)
        self.remote_index.start(self.remote_ch)

    def stop(self) -> None:
        self.local_index.stop()
        self.remote_index.stop()

    def reset(self) -> None:
        self.local_index.reset()
        self.mempool.reset()

    def match_all(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        block_hashes_t = torch.from_numpy(sequence_meta.block_hashes).to(torch.int64)
        num_blocks = sequence_meta.num_blocks

        # Query both local and remote
        mr_local = self.local_index.match_prefix(block_hashes_t, int(num_blocks), True)
        mr_remote = self.remote_index.match_prefix(block_hashes_t, int(num_blocks), True)

        # Choose the one with the larger matched length; tie-break on ready length
        local_key = (int(mr_local.num_matched_blocks), int(mr_local.num_ready_matched_blocks))
        remote_key = (int(mr_remote.num_matched_blocks), int(mr_remote.num_ready_matched_blocks))
        chosen = mr_local if local_key >= remote_key else mr_remote

        # physical blocks
        phys_np = torch.tensor(chosen.physical_blocks, dtype=torch.int64).numpy()
        #block_node_ids = torch.tensor(chosen.block_node_ids, dtype=torch.uint32).numpy() if chosen.block_node_ids is not None else None
        # optional block_node_ids
        bnids_np = None
        if chosen is mr_remote:
            block_node_ids = torch.tensor(chosen.block_node_ids, dtype=torch.uint32).numpy() if chosen.block_node_ids is not None else None
            if block_node_ids is None:
                raise Exception("Failed to get block_node_ids")
            bnids_np = self.nodeids_to_file_nodeids(block_node_ids, phys_np)
            if bnids_np is None:
                raise Exception("Failed to get file_nodeids")
            bnids_len = bnids_np.shape[0]
            if bnids_len != phys_np.shape[0]:
                raise Exception("bnids_len != phys_np.shape[0]")
        return MatchResultAccel(
            num_ready_matched_blocks=int(chosen.num_ready_matched_blocks),
            num_matched_blocks=int(chosen.num_matched_blocks),
            last_ready_node=chosen.last_ready_node,
            last_node=chosen.last_node,
            last_node_matched_length=int(chosen.last_node_matched_length),
            physical_blocks=phys_np,
            block_node_ids=bnids_np,
        )

    def nodeids_to_file_nodeids(self,
                                 block_node_ids: np.ndarray,
                                 physical_blocks: np.ndarray) -> Optional[np.ndarray]:
        """Convert per-block node ids to per-block PCFS file_nodeids.

        For each i:
          nid = block_node_ids[i]
          file_nodeids_list = self.nid_to_file_nodeids[nid]
          remote_file_num = len(file_nodeids_list)
          block_id = physical_blocks[i]
          f_idx = (block_id // self.round_robin) % remote_file_num
          out[i] = file_nodeids_list[f_idx]
        """
        try:
            bnids_np = np.asarray(block_node_ids, dtype=np.uint32)
            phys_np = np.asarray(physical_blocks, dtype=np.int64)
        except Exception:
            return None
        if bnids_np.shape[0] != phys_np.shape[0]:
            raise Exception("block_node_ids and physical_blocks must have the same length")
        out = np.full(phys_np.shape, fill_value=-1, dtype=np.int64)
        rr = max(1, int(self.round_robin))
        for i in range(bnids_np.shape[0]):
            nid = int(bnids_np[i])
            file_list = self.nid_to_file_nodeids.get(nid)
            if not file_list:
                break
            remote_file_num = len(file_list)
            if remote_file_num <= 0:
                break
            block_id = int(phys_np[i])
            f_idx = (block_id // rr) % remote_file_num
            out[i] = int(file_list[f_idx])
        return out

    def match_local(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        block_hashes_t = torch.from_numpy(sequence_meta.block_hashes).to(torch.int64)
        num_blocks = sequence_meta.num_blocks

        mr_local = self.local_index.match_prefix(block_hashes_t, int(num_blocks), True)

        phys_np = torch.tensor(mr_local.physical_blocks, dtype=torch.int64).numpy()

        return MatchResultAccel(
            num_ready_matched_blocks=int(mr_local.num_ready_matched_blocks),
            num_matched_blocks=int(mr_local.num_matched_blocks),
            last_ready_node=mr_local.last_ready_node,
            last_node=mr_local.last_node,
            last_node_matched_length=int(mr_local.last_node_matched_length),
            physical_blocks=phys_np,
            block_node_ids=None,
        )

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResultAccel] = None) -> Optional[CRadixNode]:
        sequence_meta.gen_hashes()
        phys_t = torch.from_numpy(physical_block_ids).to(torch.int64) if isinstance(physical_block_ids, np.ndarray) else physical_block_ids.to(torch.int64)
        hashes_t = torch.from_numpy(sequence_meta.block_hashes).to(torch.int64)
        if match_result is None:
            return self.local_index.insert(
                phys_t, hashes_t, int(sequence_meta.num_blocks), int(num_insert_blocks), bool(is_ready)
            )
        else:
            return self.local_index.insert(
                phys_t, hashes_t, int(sequence_meta.num_blocks), int(num_insert_blocks), bool(is_ready),
                match_result.last_node, int(match_result.num_matched_blocks), int(match_result.last_node_matched_length)
            )

    def lock_node(self, node: CRadixNode) -> None:
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.lock(node)
        else:
            self.local_index.lock(node)

    def cleanup(self, node: CRadixNode, cleanup_length: int) -> None:
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.unlock(node)
            self.remote_index.set_ready(node, True, int(cleanup_length))
        else:
            self.local_index.unlock(node)
            self.local_index.set_ready(node, True, int(cleanup_length))

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[CRadixNode] = None,
             strict: bool = True) -> torch.Tensor:
        if num_required_blocks > self.mempool.num_free_blocks:
            if protected_node is not None:
                self.local_index.lock(protected_node)
            evict_block_num = max(
                num_required_blocks - self.mempool.num_free_blocks,
                int(self.mempool.num_total_blocks * self.evict_ratio),
            )
            target_blocks = torch.zeros(evict_block_num, dtype=torch.int64)
            num_evicted = self.local_index.evict(target_blocks, evict_block_num)
            if num_evicted != evict_block_num:
                target_blocks.resize_(num_evicted)
            self.mempool.recycle_blocks(target_blocks.numpy())
            if protected_node is not None:
                self.local_index.unlock(protected_node)
        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise NotEnoughSpaceError(
                "Not enough free blocks to take, ", required=num_required_blocks, available=self.mempool.num_free_blocks
            )
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        return self.mempool.allocate_blocks(num_allocated_blocks)

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)


    @classmethod
    def from_cache_config(cls, cache_config: "CacheConfig", node_id: int, meta: Optional[RedisMeta] = None) -> "PCFSCacheEngine":
        """Create a PCFSCacheEngine from CacheConfig.

        This replaces RemotePCFSCacheEngine. It wires both local and remote
        radix trees using parameters from CacheConfig and the provided node_id.
        """
        num_blocks = int(cache_config.num_remote_blocks or 0)

        # 1) Generate unique remote_file_prefix using uuid and build remote_cache_path
        if cache_config.remote_file_prefix is None:
            raise InvalidConfigError("remote_file_prefix must be provided in CacheConfig when enable_remote is True")
        if cache_config.remote_file_num is None or cache_config.remote_file_num <= 0:
            raise InvalidConfigError("remote_file_num must be a positive integer in CacheConfig when enable_remote is True")

        # Prefer uuid from RedisMeta to ensure cluster-wide uniqueness, fallback to Python uuid if meta is None
        try:
            unique_suffix = meta.get_uuid() if meta is not None else __import__("uuid").uuid4().hex
        except Exception:
            unique_suffix = __import__("uuid").uuid4().hex

        new_prefix = f"{cache_config.remote_file_prefix}_{unique_suffix}"
        cache_config.remote_file_prefix = new_prefix
        cache_config.remote_cache_path = [
            f"{cache_config.remote_file_prefix}_{i}" for i in range(cache_config.remote_file_num)
        ]

        # 2) Create PCFS instance and lookup/create files to collect nodeids
        remote_cfg = cache_config.remote_config_custom or {}
        pcfs_fsid = remote_cfg.get("pcfs_fsid")
        pcfs_port = remote_cfg.get("pcfs_port")
        pcfs_ip = remote_cfg.get("pcfs_ip")
        pcfs_parent_nodeid = remote_cfg.get("pcfs_parent_nodeid")
        if None in (pcfs_fsid, pcfs_port, pcfs_ip, pcfs_parent_nodeid):
            raise InvalidConfigError("Some required PCFS config fields are missing: pcfs_fsid, pcfs_port, pcfs_ip, pcfs_parent_nodeid")

        pcfs = c_ext.Pcfs(pcfs_fsid, pcfs_port, pcfs_ip, False, pcfs_parent_nodeid)
        if not pcfs.init():
            raise InvalidConfigError(f"PCFS init failed: fsid={pcfs_fsid}, ip={pcfs_ip}")

        node_ids: List[int] = []
        # Derive file size if available; otherwise, use 0 when not provided (only lookup or create placeholder)
        # Prefer explicit file_size mode
        file_size = 0
        if getattr(cache_config, "remote_cache_size_mode", "file_size") == "file_size":
            file_size = int(cache_config.remote_file_size or 0)

        for remote_path in cache_config.remote_cache_path:
            nodeid = pcfs.lookup_or_create_file(remote_path, file_size, True)
            if nodeid == 0:
                raise InvalidConfigError(f"lookup or create file failed for file: {remote_path}")
            node_ids.append(int(nodeid))

        # 3) Register nodeids into Redis for discovery
        if meta is not None:
            meta.add_node_ids(node_ids)

        # Set global pcfs instance for subsequent C++ remote transfers
        try:
            c_ext.set_pcfs_instance(pcfs)
        except Exception:
            pass

        return cls(
            num_total_blocks=num_blocks,
            tokens_per_block=int(cache_config.tokens_per_block),
            evict_ratio=float(cache_config.evict_ratio),
            local_max_num_blocks=num_blocks,
            local_lease_ttl_ms=int(getattr(cache_config, "lease_ttl_ms", 100000)),
            local_renew_lease_ms=int(getattr(cache_config, "renew_lease_ms", 0)),
            local_refresh_batch_size=int(getattr(cache_config, "refresh_batch_size", 256)),
            local_idle_sleep_ms=int(getattr(cache_config, "idle_sleep_ms", 10)),
            local_lt_pool_initial_capacity=int(getattr(cache_config, "lt_pool_initial_capacity", 0)),
            remote_max_num_blocks=num_blocks,
            remote_node_id=int(node_id),
            remote_lt_pool_initial_capacity=int(getattr(cache_config, "lt_pool_initial_capacity", 0)),
            remote_refresh_batch_size=int(getattr(cache_config, "refresh_batch_size", 128)),
            remote_rebuild_interval_ms=int(getattr(cache_config, "rebuild_interval_ms", 1000)),
            remote_idle_sleep_ms=int(getattr(cache_config, "idle_sleep_ms", 10)),
            meta=meta,
        )

