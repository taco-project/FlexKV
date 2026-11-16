"""One pool-wide prefetch byte policy shared by backend and serving adapters."""

from dataclasses import dataclass
from typing import Optional


PREFETCH_BYTE_BUDGET = 128 * 1024**3


@dataclass(frozen=True)
class PrefetchBudget:
    tokens_per_block: int
    block_bytes: int
    swa_slot_bytes: int
    byte_limit: int


def resolve_prefetch_budget(model, cache, *, cpu_blocks: Optional[int] = None,
                            swa_aware: bool = True) -> PrefetchBudget:
    # Import lazily: the geometry module itself needs no native/GPU stack.
    from flexkv.common.config import (
        GLOBAL_CONFIG_FROM_ENV,
        RankInfo,
        block_size_in_bytes_for_cache,
    )

    replicas = 1
    if model.num_kv_heads == 1 and GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode == "all_write":
        replicas = model.effective_tp_size_per_node
    block_bytes = replicas * block_size_in_bytes_for_cache(model, cache, RankInfo(model))
    swa_bytes = 0
    swa = cache.swa
    if swa_aware and swa is not None and swa.enabled:
        if swa.multi_group and swa.snapshot_bytes is None:
            raise ValueError("multi-group SWA prefetch requires registered snapshot geometry")
        swa_bytes = replicas * (
            swa.snapshot_bytes
            or swa.num_swa_layers * cache.tokens_per_block * swa.bytes_per_token_per_layer
        )
    if cpu_blocks is None:
        cpu_blocks = cache.num_cpu_blocks
    # Keep a Full-KV block outside speculative ownership. Sidecars consume the
    # same credit conservatively; the physical allocator still applies pressure.
    limit = min(PREFETCH_BYTE_BUDGET, max(0, cpu_blocks - 1) * block_bytes)
    return PrefetchBudget(cache.tokens_per_block, block_bytes, swa_bytes, limit)
