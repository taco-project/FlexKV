from __future__ import annotations

from typing import Optional, Any
from flexkv.cache.redis_meta import RedisMetaChannel as _PyRedisMetaChannel
import torch

from c_ext import DistributedRadixTree as _CDistributedRadixTree
from c_ext import LocalRadixTree as _CLocalRadixTree
from c_ext import RedisMetaChannel as _CRedisMetaChannel
from c_ext import CMatchResult
from c_ext import CRadixNode


class DistributedRadixTree:
    def __init__(self,
                 tokens_per_block: int,
                 max_num_blocks: int,
                 node_id: int,
                 lt_pool_initial_capacity: int = 0,
                 refresh_batch_size: int = 128,
                 rebuild_interval_ms: int = 1000,
                 idle_sleep_ms: int = 10) -> None:
        if _CDistributedRadixTree is None:
            raise ImportError("c_ext.DistributedRadixTree is not available")
        self._c = _CDistributedRadixTree(int(tokens_per_block), int(max_num_blocks), int(node_id),
                                         int(lt_pool_initial_capacity), int(refresh_batch_size), int(rebuild_interval_ms), int(idle_sleep_ms))

    def start(self, channel: _PyRedisMetaChannel) -> None:
        ch = getattr(channel, "_c", channel)
        self._c.start(ch)

    def stop(self) -> None:
        self._c.stop()

    def remote_tree_refresh(self):
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
                 lt_pool_initial_capacity: int = 0) -> None:
        if _CLocalRadixTree is None:
            raise ImportError("c_ext.LocalRadixTree is not available")
        self._c = _CLocalRadixTree(int(tokens_per_block), int(max_num_blocks), int(lease_ttl_ms), int(renew_lease_ms), int(refresh_batch_size), int(idle_sleep_ms), int(lt_pool_initial_capacity))

    def set_meta_channel(self, channel: _PyRedisMetaChannel) -> None:
        ch = getattr(channel, "_c", channel)
        self._c.set_meta_channel(ch)

    def start(self, channel: _PyRedisMetaChannel) -> None:
        ch = getattr(channel, "_c", channel)
        self._c.start(ch)

    def stop(self) -> None:
        self._c.stop()

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

    def insert_and_publish(self, node: "CRadixNode") -> None:
        self._c.insert_and_publish(node)


