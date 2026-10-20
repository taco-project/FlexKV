"""Thin abstract base for chunked-prefetch backends.

``PrefetchCoordinator`` drives policy/seal/drain and calls these methods.
Concrete planners (Mooncake REMOTE2H graphs, radixshmem ``pull_async``) own
cache plans and transfer mechanics.

Optional: ``poll_completions()`` — backends without a transfer-engine pipe
(e.g. radixshmem) may expose this; the coordinator probes via ``hasattr``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from flexkv.prefetch.types import PrefetchOptions


class ChunkPlannerBase(ABC):
    """Shared helpers + interface contract for chunked prefetch planners."""

    def _init_common(
        self,
        engine,
        max_pinned_bytes: int,
        *,
        apply_kv_replicas: bool = False,
        executor_name: str = "flexkv-prefetch-query",
    ) -> None:
        self.engine = engine
        self.cache = engine.cache_engine
        self.block_size = engine.cache_config.tokens_per_block
        self.block_bytes = self._compute_block_bytes(apply_kv_replicas=apply_kv_replicas)
        self.max_pinned_bytes = max_pinned_bytes
        self.pinned_bytes = 0
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=executor_name
        )

    def _compute_block_bytes(self, *, apply_kv_replicas: bool = False) -> int:
        from flexkv.common.config import (
            GLOBAL_CONFIG_FROM_ENV,
            RankInfo,
            block_size_in_bytes_for_cache,
        )

        nbytes = block_size_in_bytes_for_cache(
            self.engine.model_config,
            self.engine.cache_config,
            RankInfo(self.engine.model_config),
        )
        if not apply_kv_replicas:
            return nbytes
        replicas = 1
        if (
            self.engine.model_config.num_kv_heads == 1
            and GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode == "all_write"
        ):
            replicas = self.engine.model_config.effective_tp_size_per_node
        return nbytes * replicas

    @staticmethod
    def validate_prefetch_tokens(tokens, candidate_start_token: int) -> np.ndarray:
        tokens = np.asarray(tokens)
        if tokens.ndim != 1 or tokens.dtype != np.int64:
            raise ValueError("prefetch token_ids must be a one-dimensional int64 array")
        if candidate_start_token > len(tokens):
            raise ValueError("candidate_start_token exceeds request length")
        return tokens

    @abstractmethod
    def begin(
        self,
        tokens,
        namespace: Optional[List[str]],
        client_id: int,
        options: PrefetchOptions,
    ) -> Any:
        ...

    @abstractmethod
    def resolve(self, context: Any) -> Optional[Tuple[int, int]]:
        ...

    @abstractmethod
    def reserve(
        self,
        context: Any,
        begin: int,
        target: int,
        options: PrefetchOptions,
        available_bytes: int,
    ) -> Any:
        ...

    @abstractmethod
    def submit_batch(self, chunks: Sequence[Any]) -> None:
        ...

    @abstractmethod
    def complete(self, chunk: Any, completion: Any) -> None:
        ...

    @abstractmethod
    def commit(self, context: Any, chunk: Any) -> int:
        ...

    @abstractmethod
    def discard(self, context: Any, chunk: Any) -> None:
        ...

    @abstractmethod
    def release(self, context: Any) -> None:
        ...

    @abstractmethod
    def result_end(self, context: Any, committed: int) -> int:
        ...

    def cancel_query(self, context: Any) -> None:
        cancel = getattr(context, "cancel_query", None)
        if cancel is not None and hasattr(cancel, "set"):
            cancel.set()
        future = getattr(context, "future", None)
        if future is not None:
            future.cancel()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
