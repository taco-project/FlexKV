from typing import Optional, List, Tuple

from typing import Optional, Any
from flexkv.cache.redis_meta import RedisMetaChannel as _PyRedisMetaChannel
import torch

from flexkv.c_ext import DistributedRadixTree as _CDistributedRadixTree
from flexkv.c_ext import LocalRadixTree as _CLocalRadixTree
from flexkv.c_ext import RedisMetaChannel as _CRedisMetaChannel
from flexkv.c_ext import CMatchResult
from flexkv.c_ext import CRadixNode
from flexkv.c_ext import RefRadixTree


class DistributedRadixTree:
    def __init__(self,
                 tokens_per_block: int,
                 max_num_blocks: int,
                 node_id: int,
                 refresh_batch_size: int = 128,
                 rebuild_interval_ms: int = 1000,
                 idle_sleep_ms: int = 10,
                 lease_renew_ms: int = 5000,
                 hit_reward_seconds: int = 0) -> None:
        if _CDistributedRadixTree is None:
            raise ImportError("c_ext.DistributedRadixTree is not available")
        self._c = _CDistributedRadixTree(int(tokens_per_block), int(max_num_blocks), int(node_id),
                                         int(refresh_batch_size), int(rebuild_interval_ms), int(idle_sleep_ms), int(lease_renew_ms), int(hit_reward_seconds))
        self._started = False

    def __del__(self) -> None:
        """析构函数，确保在对象被销毁时调用stop方法"""
        try:
            if hasattr(self, '_started') and self._started:
                self.stop()
        except Exception:
            # 忽略析构函数中的异常，避免影响程序退出
            pass

    def start(self, channel: _PyRedisMetaChannel) -> bool:
        """Start the DistributedRadixTree with the given Redis meta channel.
        
        Args:
            channel: RedisMetaChannel instance to use for Redis communication
            
        Returns:
            bool: True if start was successful, False otherwise
        """
        try:
            ch = getattr(channel, "_c", channel)
            if not self._c.start(ch):
                return False
            self._started = True
            return True
        except Exception:
            self._started = False
            return False

    def stop(self) -> None:
        self._c.stop()
        self._started = False

    def remote_tree_refresh(self) -> Optional["RefRadixTree"]:
        """Refresh the remote tree by loading block metadata from Redis.
        
        This method can be called at any time to manually refresh the remote tree,
        regardless of whether the background refresh thread is running.
        
        Returns:
            Optional[RefRadixTree]: The refreshed reference tree, or None if refresh fails
        """
        return self._c.remote_tree_refresh()

    def match_prefix(self, block_hashes: torch.Tensor, num_blocks: int, update_cache_info: bool = True):
        return self._c.match_prefix(block_hashes, int(num_blocks), bool(update_cache_info))

    def lock(self, node: "CRadixNode") -> None:
        self._c.lock(node)

    def unlock(self, node: "CRadixNode") -> None:
        self._c.unlock(node)

    def is_empty(self) -> bool:
        return bool(self._c.is_empty())

    def set_ready(self, node: "CRadixNode", ready: bool = True, ready_length: int = -1) -> None:
        self._c.set_ready(node, bool(ready), int(ready_length))


class LocalRadixTree:
    def __init__(self,
                 tokens_per_block: int,
                 max_num_blocks: int = 1000000,
                 lease_ttl_ms: int = 100000,
                 renew_lease_ms: int = 0,
                 refresh_batch_size: int = 256,
                 idle_sleep_ms: int = 10,
                 safety_ttl_ms: int = 100,
                 swap_block_threshold: int = 1024,
                 hit_reward_seconds: int = 0) -> None:
        if _CLocalRadixTree is None:
            raise ImportError("c_ext.LocalRadixTree is not available")
        self._c = _CLocalRadixTree(
            int(tokens_per_block),
            int(max_num_blocks),
            int(lease_ttl_ms),
            int(renew_lease_ms),
            int(refresh_batch_size),
            int(idle_sleep_ms),
            int(safety_ttl_ms),
            int(swap_block_threshold),
            int(hit_reward_seconds)
        )
        self._started = False

    def __del__(self) -> None:
        """析构函数，确保在对象被销毁时调用stop方法"""
        try:
            if hasattr(self, '_started') and self._started:
                self.stop()
        except Exception:
            # 忽略析构函数中的异常，避免影响程序退出
            pass

    def set_meta_channel(self, channel: _PyRedisMetaChannel) -> None:
        """Set the Redis meta channel for this LocalRadixTree.
        
        Args:
            channel: RedisMetaChannel instance to use for Redis communication
        """
        ch = getattr(channel, "_c", channel)
        self._c.set_meta_channel(ch)

    def start(self, channel: _PyRedisMetaChannel) -> bool:
        """Start the LocalRadixTree with the given Redis meta channel.
        
        Args:
            channel: RedisMetaChannel instance to use for Redis communication
            
        Returns:
            bool: True if start was successful, False otherwise
        """
        try:
            ch = getattr(channel, "_c", channel)
            if not self._c.start(ch):
                return False
            self._started = True
            return True
        except Exception:
            self._started = False
            return False

    def stop(self) -> None:
        self._c.stop()
        self._started = False

    # Mirror base class methods on LocalRadixTree
    def match_prefix(self, block_hashes: torch.Tensor, num_blocks: int, update_cache_info: bool = True):
        return self._c.match_prefix(block_hashes, int(num_blocks), bool(update_cache_info))

    def total_unready_blocks(self) -> int:
        return int(self._c.total_unready_blocks())

    def total_ready_blocks(self) -> int:
        return int(self._c.total_ready_blocks())

    def total_cached_blocks(self) -> int:
        return int(self._c.total_cached_blocks())

    def total_node_num(self) -> int:
        return int(self._c.total_node_num())

    def reset(self) -> None:
        self._c.reset()

    def is_root(self, node: "CRadixNode") -> bool:
        # Note: CRadixNode pointer type is opaque in Python; in practice this is used internally
        return bool(self._c.is_root(node))

    def remove_node(self, node: "CRadixNode") -> None:
        self._c.remove_node(node)

    def remove_leaf(self, node: "CRadixNode") -> None:
        self._c.remove_leaf(node)

    def add_node(self, node: "CRadixNode") -> None:
        self._c.add_node(node)

    def add_leaf(self, node: "CRadixNode") -> None:
        self._c.add_leaf(node)

    def lock(self, node: "CRadixNode") -> None:
        self._c.lock(node)

    def unlock(self, node: "CRadixNode") -> None:
        self._c.unlock(node)

    def is_empty(self) -> bool:
        return bool(self._c.is_empty())

    def inc_node_count(self) -> None:
        self._c.inc_node_count()

    def dec_node_count(self) -> None:
        self._c.dec_node_count()

    def set_ready(self, node: "CRadixNode", ready: bool = True, ready_length: int = -1) -> None:
        self._c.set_ready(node, bool(ready), int(ready_length))

    def insert(self, physical_block_ids: torch.Tensor, block_hashes: torch.Tensor, 
               num_blocks: int, num_insert_blocks: int = -1, ready: bool = True, 
               node: "CRadixNode" = None, num_matched_blocks: int = -1, 
               last_node_matched_length: int = -1) -> "CRadixNode":
        """Insert blocks into the LocalRadixTree.
        
        Args:
            physical_block_ids: Tensor containing physical block IDs
            block_hashes: Tensor containing block hash values
            num_blocks: Total number of blocks
            num_insert_blocks: Number of blocks to insert (-1 for all)
            ready: Whether the inserted blocks are ready
            node: Last node for continuation (-1 for auto-match)
            num_matched_blocks: Number of matched blocks (-1 for auto-match)
            last_node_matched_length: Length of last node match (-1 for auto-match)
            
        Returns:
            CRadixNode: The newly inserted node, or None if no insertion occurred
        """
        return self._c.insert(
            physical_block_ids, block_hashes, int(num_blocks), int(num_insert_blocks),
            bool(ready), node, int(num_matched_blocks), int(last_node_matched_length)
        )

    def evict(self, evicted_blocks: torch.Tensor, num_evicted: int) -> int:
        """Evict blocks from the LocalRadixTree.
        
        Args:
            evicted_blocks: Tensor to store evicted block IDs
            num_evicted: Number of blocks to evict
            
        Returns:
            int: Number of blocks actually evicted
        """
        return int(self._c.evict(evicted_blocks, int(num_evicted)))

    def insert_and_publish(self, node: "CRadixNode") -> bool:
        if not self._started:
            raise RuntimeError("LocalRadixTree must be started before calling insert_and_publish")
        return bool(self._c.insert_and_publish(node))

    def drain_pending_queues(self) -> int:
        return int(self._c.drain_pending_queues())


