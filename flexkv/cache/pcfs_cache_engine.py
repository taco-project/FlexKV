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


class HierarchyLRCacheEngine:
    def __init__(self,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                device_type: DeviceType,
                # Optional runtime wiring for remote/local trees
                local_max_num_blocks: Optional[int] = 0,
                local_lease_ttl_ms: int = 100000,
                 local_renew_lease_ms: int = 10000,
                 local_refresh_batch_size: int = 1000,
                 local_idle_sleep_ms: int = 10,
                 remote_max_num_blocks: int = 4000000,
                 redis_node_id: int = 0,
                 remote_refresh_batch_size: int = 1000,
                 remote_rebuild_interval_ms: int = 10000,
                 remote_idle_sleep_ms: int = 10,
                 local_safety_ttl_ms: int = 100,
                 evict_start_threshold: float = 1.0,
                 meta: Optional[RedisMeta] = None) -> None:
        if num_total_blocks <= 0:
            raise InvalidConfigError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise InvalidConfigError(
                f"Invalid tokens_per_block: {tokens_per_block}, tokens_per_block must be a power of 2"
            )

        self.device_type = device_type
        self._meta: Optional[RedisMeta] = meta # todo: define storage type in meta


        # belows are only for 3rd-party remote storage (like pcfs)
        # Mapping: node_id -> list of PCFS file_nodeids
        self.nid_to_file_nodeids: Dict[int, List[int]] = {}
        # Partition parameter used for mapping block_id to file index
        self.round_robin: int = 1

        # Local index (authoritative for mutations)
        self.local_index = LocalRadixTree(
            tokens_per_block=tokens_per_block,
            max_num_blocks=int(num_total_blocks),
            lease_ttl_ms=int(local_lease_ttl_ms),
            renew_lease_ms=int(local_renew_lease_ms),
            refresh_batch_size=int(local_refresh_batch_size),
            idle_sleep_ms=int(local_idle_sleep_ms),
            safety_ttl_ms=int(local_safety_ttl_ms),
            swap_block_threshold=int(evict_ratio * num_total_blocks)
        )


        # Remote reference index (read-only, built from Redis)
        self.remote_index = DistributedRadixTree(
            tokens_per_block=tokens_per_block,
            max_num_blocks=int(remote_max_num_blocks or (num_total_blocks * 10)),
            node_id=int(redis_node_id),
            refresh_batch_size=int(remote_refresh_batch_size),
            rebuild_interval_ms=int(remote_rebuild_interval_ms),
            idle_sleep_ms=int(remote_idle_sleep_ms),
            lease_renew_ms=int(local_renew_lease_ms),
        )
        # defer channel start to start(meta)

        # Local memory pool for physical blocks on this device
        self.mempool = Mempool(num_total_blocks=int(local_max_num_blocks or num_total_blocks))

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

    def start(self) -> None:
        if self._meta is None:
            raise InvalidConfigError("RedisMeta is not provided; ensure from_cache_config stores it or pass it to start().")
        #TODO can we use like this to distinguish the different tree pairs?
        if self.device_type == DeviceType.REMOTE:
            local_ch_block_key = "PCFSB"
            remote_ch_block_key = "PCFSB"
        elif self.device_type == DeviceType.CPU:
            local_ch_block_key = "CPUB"
            remote_ch_block_key = "CPUB"
        elif self.device_type == DeviceType.SSD:
            local_ch_block_key = "SSDB"
            remote_ch_block_key = "SSDB"
        else:
            raise InvalidConfigError(f"Invalid device type: {self.device_type}")
        self.remote_ch = self._meta.get_redis_meta_channel(remote_ch_block_key)
        self.local_ch = self._meta.get_redis_meta_channel(local_ch_block_key)
                # Load and store mapping of node_id -> file_nodeids from Redis
        if self.device_type == DeviceType.REMOTE:
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

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        """Match a sequence against the cache index.
        
        This method provides a simple interface similar to CacheEngine.match(),
        delegating to match_all() for consistency.
        
        Args:
            sequence_meta: The sequence metadata to match
            
        Returns:
            MatchResultAccel: The match result
        """
        return self.match_all(sequence_meta)

    def match_all(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        block_hashes_t = torch.from_numpy(sequence_meta.block_hashes).to(torch.int64)
        num_blocks = sequence_meta.num_blocks

        # Query both local and remote
        mr_local = self.local_index.match_prefix(block_hashes_t, int(num_blocks), True)
        mr_remote = self.remote_index.match_prefix(block_hashes_t, int(num_blocks), True)
        print(f"[MATCH {self.device_type.name}] local: {mr_local.num_matched_blocks}/{mr_local.num_ready_matched_blocks}, remote: {mr_remote.num_matched_blocks}/{mr_remote.num_ready_matched_blocks}")
        # For simplicy, we choose the one with the larger matched length; tie-break on ready length
        # We should allow to combine the two results in the future.
        local_key = (int(mr_local.num_matched_blocks), int(mr_local.num_ready_matched_blocks))
        remote_key = (int(mr_remote.num_matched_blocks), int(mr_remote.num_ready_matched_blocks))
        matched_pos = "local" if local_key >= remote_key else "remote"
        chosen = mr_local if local_key >= remote_key else mr_remote

        # physical blocks
        bnids_np = None
        if chosen is mr_remote:
            #尝试使用DistributedRadixTree的block_node_ids
            #如果检查失败，则使用LocalRadixTree的匹配结果
            nids = chosen.block_node_ids
            nps = chosen.physical_blocks
            # Convert tensors to numpy views (CPU) if present
            if isinstance(nids, torch.Tensor) and nids.numel() > 0:
                # For P2P mode (CPU/SSD), no PCFS conversion is needed
                # Only convert to PCFS file_nodeids if device_type is REMOTE
                if self.device_type == DeviceType.REMOTE:
                    bnids_np = self.nodeids_to_file_nodeids(nids.cpu().numpy(), nps.cpu().numpy())
                    if bnids_np is None:
                        chosen = mr_local
                else:
                    # For P2P mode, use node_ids directly
                    bnids_np = nids.cpu().numpy().astype(np.uint32)
                    #print(f"[REMOTE_MATCH {self.device_type.name}] Using remote data: block_ids={nps.cpu().numpy()[:min(4, len(nps))]}, node_ids={bnids_np[:min(4, len(bnids_np))]}")
            else:
                bnids_np = None
                if mr_remote.num_matched_blocks > 0:
                    #print(f"[REMOTE_MATCH {self.device_type.name}] Warning: remote matched but block_node_ids is empty, falling back to local")
                    chosen = mr_local
        phys_np = chosen.physical_blocks.cpu().numpy()
        if self.device_type == DeviceType.CPU and matched_pos == "remote" and mr_local.num_matched_blocks > 0:
            insert_to_local_cpu_index = False
        else:
            insert_to_local_cpu_index = True
        #TODO A big question is how to get the node id for peer_cpu and peer_ssd?
        return MatchResultAccel(
            num_ready_matched_blocks=int(chosen.num_ready_matched_blocks),
            num_matched_blocks=int(chosen.num_matched_blocks),
            last_ready_node=chosen.last_ready_node,
            last_node=chosen.last_node,
            last_node_matched_length=int(chosen.last_node_matched_length),
            physical_blocks=phys_np,
            block_node_ids=bnids_np,
            matched_pos=matched_pos,
            matched_node_ids=bnids_np,  # Set matched_node_ids for P2P transfer
            insert_to_local_cpu_index=insert_to_local_cpu_index,
        )

    def nodeids_to_file_nodeids(self,
                                 bnids_np: np.ndarray,
                                 phys: np.ndarray) -> Optional[np.ndarray]:
        """Convert per-block node ids to per-block PCFS file_nodeids.

        Args:
            bnids_np: block_node_ids from MatchResultAccel.block_node_ids
            phys: physical_blocks from MatchResultAccel.physical_blocks

        Returns:
            file_nodeids array with dtype=uint32, or None if conversion fails
        """
        if bnids_np is None or phys is None:
            return None
        try:
            bnids_np = np.asarray(bnids_np, dtype=np.uint32)
            phys_np = np.asarray(phys, dtype=np.int64)
        except Exception:
            return None
        if bnids_np.shape[0] != phys_np.shape[0]:
            return None
        out = np.full(phys_np.shape, fill_value=0, dtype=np.uint32)
        rr = max(1, int(self.round_robin))
        
        for i in range(bnids_np.shape[0]):
            nid = int(bnids_np[i])
            #检查节点是否活跃
            is_active = self._meta.is_node_active(nid)
            if not is_active:
                print(f"[DEBUG] Node {nid} is not active, returning None")
                return None
            file_list = self.nid_to_file_nodeids.get(nid)
            #检查文件列表是否为空
            if not file_list:
                return None
            remote_file_num = len(file_list)
            if remote_file_num <= 0:
                return None
            block_id = int(phys_np[i])
            f_idx = (block_id // rr) % remote_file_num
            out[i] = np.uint32(file_list[f_idx])
        return out

    def match_local(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        block_hashes_t = torch.from_numpy(sequence_meta.block_hashes).to(torch.int64)
        num_blocks = sequence_meta.num_blocks

        mr_local = self.local_index.match_prefix(block_hashes_t, int(num_blocks), True)

        phys_np = mr_local.physical_blocks.cpu().numpy()

        return MatchResultAccel(
            num_ready_matched_blocks=int(mr_local.num_ready_matched_blocks),
            num_matched_blocks=int(mr_local.num_matched_blocks),
            last_ready_node=mr_local.last_ready_node,
            last_node=mr_local.last_node,
            last_node_matched_length=int(mr_local.last_node_matched_length),
            physical_blocks=phys_np,
            block_node_ids=None,
            matched_pos="local",
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
            node = self.local_index.insert(
                phys_t, hashes_t, int(sequence_meta.num_blocks), int(num_insert_blocks), bool(is_ready)
            )
        else:
            node = self.local_index.insert(
                phys_t, hashes_t, int(sequence_meta.num_blocks), int(num_insert_blocks), bool(is_ready),
                match_result.last_node, int(match_result.num_matched_blocks), int(match_result.last_node_matched_length)
            )
        # NOTE: Do NOT lock the node here, because the caller (put() method) will lock it
        # The node will be unlocked in _transfer_callback after data transfer completes
        return node

    def lock_node(self, node: CRadixNode) -> None:
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.lock(node) #TODO why do we need to lock the remote node?
        else:
            self.local_index.lock(node)

    def unlock(self, node: CRadixNode) -> None:
        """Unlock a node in the appropriate index (local or remote).
        
        Args:
            node: The radix node to unlock
        """
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.unlock(node)
        else:
            self.local_index.unlock(node)

    def cleanup(self, node: CRadixNode, cleanup_length: int) -> None:
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.unlock(node)
        else:
            self.local_index.unlock(node)

    def set_ready(self, node: CRadixNode, ready: bool = True, ready_length: int = -1) -> None:
        """Set the ready state of a node in the appropriate index (local or remote).
        
        Args:
            node: The radix node to set ready state
            ready: Whether the node is ready (default: True)
            ready_length: The ready length (default: -1, meaning use node's current length)
        """
        if node is None:
            return
        try:
            is_remote_node = bool(node.has_block_node_ids())
        except Exception:
            is_remote_node = False
        if is_remote_node:
            self.remote_index.set_ready(node, ready, ready_length)
        else:
            self.local_index.set_ready(node, ready, ready_length)

    def cleanup(self, node: CRadixNode, cleanup_length: int) -> None:
        unlock(node)
        set_ready(node, ready, ready_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[CRadixNode] = None,
             strict: bool = True) -> torch.Tensor:
        # Calculate current utilization
        utilization = (self.mempool.num_total_blocks - self.mempool.num_free_blocks) / self.mempool.num_total_blocks if self.mempool.num_total_blocks > 0 else 0
        
        # Proactive eviction: trigger when utilization exceeds threshold OR when blocks are needed
        should_evict = (utilization >= self.evict_start_threshold) or (num_required_blocks > self.mempool.num_free_blocks)
        
        if should_evict:
            if protected_node is not None:
                self.local_index.lock(protected_node)
            
            # Calculate how many blocks to evict
            # Goal: maintain free blocks above (1 - evict_start_threshold) ratio
            target_free_blocks = int(self.mempool.num_total_blocks * (1.0 - self.evict_start_threshold))
            evict_to_reach_target = max(0, target_free_blocks - self.mempool.num_free_blocks)
            
            evict_block_num = max(
                num_required_blocks - self.mempool.num_free_blocks,  # At least meet current demand
                evict_to_reach_target,                               # Or reach target free ratio
                int(self.mempool.num_total_blocks * self.evict_ratio) if self.evict_ratio > 0 else 0  # Or minimum evict_ratio
            )
            
            if evict_block_num > 0:
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
        #print(f"[TAKE STATISTICS] device type: {self.device_type.name}, utilization: {utilization}, ",
        #      f"should_evict: {should_evict}, num_required_blocks: {num_required_blocks}, ",
        #      f"num_allocated_blocks: {num_allocated_blocks}, num_free_blocks: {self.mempool.num_free_blocks}")
        return self.mempool.allocate_blocks(num_allocated_blocks)

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)


    @classmethod
    def pcfs_ce_from_cache_config(cls, cache_config: "CacheConfig", node_id: int, meta: Optional[RedisMeta] = None) -> "HierarchyLRCacheEngine":
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
            device_type=DeviceType.REMOTE,
            local_lease_ttl_ms=int(getattr(cache_config, "lease_ttl_ms", 100000)),
            local_renew_lease_ms=int(getattr(cache_config, "renew_lease_ms", 10)),
            local_refresh_batch_size=int(getattr(cache_config, "refresh_batch_size", 256)),
            local_idle_sleep_ms=int(getattr(cache_config, "idle_sleep_ms", 10)),
            remote_max_num_blocks=num_blocks,
            redis_node_id=int(node_id),
            remote_refresh_batch_size=int(getattr(cache_config, "refresh_batch_size", 256)),
            remote_rebuild_interval_ms=int(getattr(cache_config, "rebuild_interval_ms", 1000)),
            remote_idle_sleep_ms=int(getattr(cache_config, "idle_sleep_ms", 10)),
            local_safety_ttl_ms=int(getattr(cache_config, "safety_ttl_ms", 100)),
            meta=meta,
        )

    #TODO is this enough for peercpu and peerssd?
    @classmethod
    def from_cache_config(cls, cache_config: "CacheConfig", node_id: int, device_type: DeviceType, meta: Optional[RedisMeta] = None) -> "HierarchyLRCacheEngine":
        if device_type == DeviceType.REMOTE:
            return cls.pcfs_ce_from_cache_config(cache_config, node_id, meta)
        else:
            # 根据device_type选择正确的blocks配置
            if device_type == DeviceType.CPU:
                local_max_num_blocks = int(cache_config.num_cpu_blocks)
            elif device_type == DeviceType.SSD:
                local_max_num_blocks = int(cache_config.num_ssd_blocks)
            else:
                local_max_num_blocks = int(cache_config.num_local_blocks or 0)
            
            return cls(
                num_total_blocks=int(cache_config.num_remote_blocks or 0),
                tokens_per_block=int(cache_config.tokens_per_block),
                evict_ratio=float(cache_config.evict_ratio),
                device_type=device_type,
                local_max_num_blocks=local_max_num_blocks,
                local_lease_ttl_ms=int(getattr(cache_config, "lease_ttl_ms", 100000)),
                local_renew_lease_ms=int(getattr(cache_config, "renew_lease_ms", 0)),
                local_refresh_batch_size=int(getattr(cache_config, "refresh_batch_size", 256)),
                local_idle_sleep_ms=int(getattr(cache_config, "idle_sleep_ms", 10)),
                # local_lt_pool_initial_capacity=int(getattr(cache_config, "lt_pool_initial_capacity", 0)),
                remote_max_num_blocks=int(cache_config.num_remote_blocks or 0),
                redis_node_id=int(node_id),
                # remote_node_id=int(node_id),
                # remote_lt_pool_initial_capacity=int(getattr(cache_config, "lt_pool_initial_capacity", 0)),
                remote_refresh_batch_size=int(getattr(cache_config, "refresh_batch_size", 128)),
                remote_rebuild_interval_ms=int(getattr(cache_config, "rebuild_interval_ms", 10000)),
                remote_idle_sleep_ms=int(getattr(cache_config, "idle_sleep_ms", 10)),
                local_safety_ttl_ms=int(getattr(cache_config, "safety_ttl_ms", 100)),
                evict_start_threshold=float(cache_config.evict_start_threshold),
                meta=meta,
            )
            raise InvalidConfigError("Invalid device type: {cache_config.device_type}")

