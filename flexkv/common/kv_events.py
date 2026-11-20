# SPDX-License-Identifier: Apache-2.0
"""Framework-neutral KV placement events for FlexKV's own cache tiers.

A KV-aware router only knows what the inference engine tells it, and the
engine never sees FlexKV's tiers: to sglang, a block that lands in the
radixshmem CPU tier simply left the device. This queue is how the tier says
what it holds, in a form the engine can forward on its existing channel —
so the router still sees one publisher per engine, the engine itself.

Two things distinguish it from `flexkv.integration.dynamo.collector`:

* no framework import. That collector builds `vllm.distributed.kv_events`
  structs directly, which makes it unusable from the sglang path.
* a store carries its `token_ids`. Every framework in the chain hashes
  differently (sglang, FlexKV and radixshmem each have their own scheme),
  so a router that hashes content itself — llm-d and dynamo both do — can
  only map an engine hash onto its own key if the tokens come with it.
  Removals need no tokens: the router resolves them through the mapping
  the store established.

The queue is written by whichever thread runs the tier's insert path and
drained by the engine's event pump, hence the lock.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass
class BlocksStored:
    """`block_hashes[i]` covers `token_ids[i]`, in path order.

    `parent_block_hash` is the hash of the block immediately before the
    first one here, or None when this batch starts at the path root. It is
    what lets the consumer rebuild a prefix chain out of a tail insert.
    """
    block_hashes: List[int] = field(default_factory=list)
    token_ids: List[List[int]] = field(default_factory=list)
    parent_block_hash: Optional[int] = None
    block_size: int = 0
    medium: str = "CPU"


@dataclass
class BlocksRemoved:
    """Hash-only: the tier has no tokens left by the time it evicts."""
    block_hashes: List[int] = field(default_factory=list)
    medium: str = "CPU"


@dataclass
class AllBlocksCleared:
    """The tier's view and the consumer's diverged beyond repair (a reset, or
    an event ring that lapped). Drop everything and start over."""


class KVEventQueue:
    """Placement events awaiting pickup, disabled until someone enables it.

    Disabled is the default because an unread queue is a leak: nothing
    publishes until a consumer has declared itself.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[Any] = []
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
        with self._lock:
            self._events = []

    def publish_stored(self,
                       block_hashes: Sequence[int],
                       token_ids: Sequence[Sequence[int]],
                       *,
                       block_size: int,
                       parent_block_hash: Optional[int] = None,
                       medium: str = "CPU") -> None:
        if not self._enabled or len(block_hashes) == 0:
            return
        if len(block_hashes) != len(token_ids):
            raise ValueError(
                f"kv event: {len(block_hashes)} block hashes but "
                f"{len(token_ids)} token blocks")
        self._append(BlocksStored(
            block_hashes=_as_int_list(block_hashes),
            token_ids=[_as_int_list(t) for t in token_ids],
            parent_block_hash=(None if parent_block_hash is None
                               else int(parent_block_hash)),
            block_size=int(block_size),
            medium=medium,
        ))

    def publish_removed(self,
                        block_hashes: Sequence[int],
                        *,
                        medium: str = "CPU") -> None:
        if not self._enabled or len(block_hashes) == 0:
            return
        self._append(BlocksRemoved(block_hashes=_as_int_list(block_hashes),
                                   medium=medium))

    def publish_all_cleared(self) -> None:
        if not self._enabled:
            return
        self._append(AllBlocksCleared())

    def take(self) -> List[Any]:
        if not self._enabled:
            return []
        with self._lock:
            events, self._events = self._events, []
        return events

    def _append(self, event: Any) -> None:
        with self._lock:
            self._events.append(event)


def _as_int_list(values: Sequence[int]) -> List[int]:
    """Python ints, whatever the source. FlexKV hashes arrive as uint64
    numpy scalars, which msgspec cannot encode."""
    if isinstance(values, np.ndarray):
        return values.tolist()
    return [int(v) for v in values]
