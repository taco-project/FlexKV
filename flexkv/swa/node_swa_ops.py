"""Shared SWA radix-tree operations for in-process and server-side paths."""
from typing import Optional, Union

import numpy as np
import torch

from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig
from flexkv.common.debug import flexkv_logger


def engine_tree(engine):
    tree = getattr(engine, "index", None)
    if tree is None:
        tree = getattr(engine, "local_index", None)
    return tree


def swa_unavailable_reason(cache_config: CacheConfig, engine) -> Optional[str]:
    if cache_config.swa is None or not cache_config.swa.enabled:
        return "SWAPoolConfig is None or disabled"
    if engine is None:
        return "cpu_cache_engine is None"
    tree = engine_tree(engine)
    if tree is None:
        return "cpu_cache_engine has no index/local_index radix tree"
    if not hasattr(tree, "drain_freed_swa_slots"):
        return "radix tree lacks drain_freed_swa_slots (node-attached SWA unsupported)"
    if not hasattr(engine, "init_swa"):
        return "cpu_cache_engine lacks init_swa()"
    return None


def get_or_init_swa_manager(cache_config: CacheConfig, engine):
    reason = swa_unavailable_reason(cache_config, engine)
    if reason is not None:
        flexkv_logger.warning(f"[SWA] get_or_init_swa_manager: {reason}")
        return None
    if getattr(engine, "swa_manager", None) is None:
        engine.init_swa(cache_config.swa)
    return engine.swa_manager


def match_radix_node(cache_config: CacheConfig, engine, token_ids: np.ndarray):
    tree = engine_tree(engine)
    if tree is None:
        return None
    seq = SequenceMeta(
        token_ids=np.asarray(token_ids, dtype=np.int64),
        tokens_per_block=cache_config.tokens_per_block,
    )
    seq.gen_hashes()
    if seq.num_blocks == 0:
        return None
    block_hashes_t = torch.from_numpy(seq.block_hashes).to(torch.int64)
    mr = tree.match_prefix(block_hashes_t, int(seq.num_blocks), False)
    if mr is None or int(mr.num_matched_blocks) == 0:
        return None
    return mr.last_node


def _normalize_swa_data(swa_data: Union[torch.Tensor, np.ndarray, bytes]) -> Union[np.ndarray, bytes]:
    if isinstance(swa_data, torch.Tensor):
        if swa_data.is_cuda:
            swa_data = swa_data.cpu()
        return swa_data.numpy()
    return swa_data


def swa_put_on_engine(
    cache_config: CacheConfig,
    engine,
    token_ids: np.ndarray,
    swa_data: Union[torch.Tensor, np.ndarray, bytes],
) -> bool:
    prod_mgr = get_or_init_swa_manager(cache_config, engine)
    if prod_mgr is None:
        return False
    node = match_radix_node(cache_config, engine, token_ids)
    if node is None:
        flexkv_logger.info(
            f"[SWA] swa_put_on_engine: no radix node for token_count={len(token_ids)}"
        )
        return False
    return prod_mgr.put(node, _normalize_swa_data(swa_data))


def swa_available_on_engine(
    cache_config: CacheConfig,
    engine,
    token_ids: np.ndarray,
) -> bool:
    prod_mgr = get_or_init_swa_manager(cache_config, engine)
    if prod_mgr is None:
        return False
    node = match_radix_node(cache_config, engine, token_ids)
    if node is None:
        return False
    return prod_mgr.has(node)


def swa_get_on_engine(
    cache_config: CacheConfig,
    engine,
    token_ids: np.ndarray,
) -> Optional[Union[torch.Tensor, np.ndarray]]:
    prod_mgr = get_or_init_swa_manager(cache_config, engine)
    if prod_mgr is None:
        return None
    node = match_radix_node(cache_config, engine, token_ids)
    if node is None:
        return None
    return prod_mgr.get(node)
