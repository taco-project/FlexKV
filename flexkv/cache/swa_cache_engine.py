# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SWACacheManager — control-plane orchestration of multi-tier SWA caching.

This module owns the *global* SWA control plane over the node-mounted SWA state
(the Full-KV radix nodes carry the SWA slot / tombstone / lock; each
``CacheEngineAccel`` / ``HierarchyLRCacheEngine`` owns an SWA host pool as
``engine.swa_pool`` and exposes ``engine.match_swa`` / ``engine.set_swa``). It is
the SWA counterpart to the full-KV orchestration in
:class:`~flexkv.cache.cache_engine.GlobalCacheEngine`,
and is deliberately kept in a separate module so the SWA peer-op logic does not
clutter the (already large) full-KV engine.

Responsibilities (control plane only — no byte movement):
  * Multi-tier prefix match (CPU / SSD / REMOTE), each clamped to that tier's
    Full-KV hit (SWA must be a subset of Full at every tier).
  * Build SWA *peer* ops into the SAME ``TransferOpGraph`` as the full-KV ops,
    with tier dependencies that mirror the full-KV graph exactly. SWA ops reuse
    the STANDARD transfer types (H2D / D2H / DISK2H / H2DISK / REMOTE2H /
    H2REMOTE) and carry ``is_swa=True`` so the transfer engine routes them to the
    dedicated SWA worker; their src/dst block ids are SWA-pool slot ids:
      - GET: the SWA ``H2D`` depends on the SWA ``DISK2H`` / ``REMOTE2H`` staging
        ops; only the terminal SWA ``H2D`` is reported as a finished op (joins
        the VIRTUAL barrier alongside the full-KV ``H2D``).
      - PUT: the SWA ``H2DISK`` / ``H2REMOTE`` write-through ops depend on the SWA
        ``D2H`` but are fire-and-forget (NOT reported), only the SWA ``D2H`` is
        reported — exactly like the full-KV ``D2H`` / ``H2DISK`` / ``H2REMOTE``.

SWA is a first-class PEER op, NOT a child derived from the full-KV op (see
the swa_design docs): the full-KV ``pending_count`` child model is PP-sibling
replica fan-out, the indexer rides the full op as a layer-group sharing block
ids, and neither fits SWA (independent slot space; the SWA-only case has no full
op to derive from). The data-plane colleague aligned on the ``is_swa`` flag (a
plain ``TransferOp`` field) rather than dedicated SWA transfer types, so the
graph stays homogeneous and routing is a single boolean.

Everything here is gated by ``cache_config.enable_swa_transfer`` (default False):
until the dedicated SWA transfer worker (data plane) is registered, the build
helpers are no-ops so an SWA op never reaches the transfer engine. The byte
movement, kernels, SWA SSD/remote storage and completion callbacks are the data
plane's responsibility.
"""

from typing import List, Optional

import numpy as np

from flexkv.common.transfer import DeviceType, TransferOp, TransferOpGraph, TransferType


class SWACacheManager:
    """Global SWA control plane: multi-tier match + peer-op graph construction.

    Holds a back-reference to the owning ``GlobalCacheEngine`` to reach the
    per-tier cache engines (and their node-mounted SWA state / ``swa_pool``) and
    the cache config. The per-tier SWA primitives (match_swa / set_swa / evict_swa
    via the radix tree) live on the engines themselves; this class only
    orchestrates across tiers.
    """

    def __init__(self, global_cache_engine) -> None:
        self._gce = global_cache_engine

    # --- tier access -------------------------------------------------------

    def _engine(self, device_type: DeviceType):
        return self._gce.cache_engines.get(device_type)

    def _swa_enabled_tier(self, device_type: DeviceType) -> bool:
        """True iff the tier's engine has a node-mounted SWA host pool."""
        engine = self._engine(device_type)
        return bool(getattr(engine, "swa_enabled", False)) if engine is not None else False

    @property
    def enabled(self) -> bool:
        """True when SWA transfer is gated on AND the CPU tier has an SWA pool.

        Gating mirrors the full-KV path: the SWA control-plane match always runs
        (cheap), but graph construction is suppressed until the data plane lands.
        """
        cfg = getattr(self._gce, "cache_config", None)
        return bool(getattr(cfg, "enable_swa_transfer", False)) and \
            self._swa_enabled_tier(DeviceType.CPU)

    # --- peer-op graph construction (gated) --------------------------------

    def build_swa_op(self,
                     graph: TransferOpGraph,
                     transfer_type: TransferType,
                     src_slot_ids: np.ndarray,
                     dst_slot_ids: np.ndarray,
                     dp_client_id: int = 0,
                     mooncake_tail_hashes: Optional[List[str]] = None) -> Optional[int]:
        """Add one peer SWA transfer op (``is_swa=True``) to ``graph``; return op_id.

        ``transfer_type`` is a STANDARD type (H2D / D2H / DISK2H / H2DISK /
        REMOTE2H / H2REMOTE); the ``is_swa`` flag routes it to the SWA worker.
        ``src_slot_ids`` / ``dst_slot_ids`` are SWA-pool slot ids (independent of
        the full-KV block-id space). Returns None (adds nothing) when SWA transfer
        is disabled or the slot arrays are empty, so callers can invoke it
        unconditionally.
        """
        if not self.enabled:
            return None
        src = np.asarray(src_slot_ids, dtype=np.int64)
        dst = np.asarray(dst_slot_ids, dtype=np.int64)
        if src.size == 0 or dst.size == 0:
            return None
        is_remote = transfer_type in (TransferType.H2REMOTE, TransferType.REMOTE2H)
        tail_hashes = (
            [str(h) for h in mooncake_tail_hashes]
            if (is_remote and mooncake_tail_hashes)
            else None
        )
        op = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=transfer_type,
            src_block_ids=src,
            dst_block_ids=dst,
            dp_client_id=dp_client_id,
            is_swa=True,
            mooncake_store_swa_block_hashes=tail_hashes,
        )
        graph.add_transfer_op(op)
        return op.op_id

    def build_get_chain(self,
                        graph: TransferOpGraph,
                        gpu_slot_ids: np.ndarray,
                        cpu_slot_ids: np.ndarray,
                        ssd_slot_ids: Optional[np.ndarray] = None,
                        remote_slot_ids: Optional[np.ndarray] = None,
                        dp_client_id: int = 0,
                        mooncake_tail_hashes: Optional[List[str]] = None) -> Optional[int]:
        """Build the GET-side SWA load chain into ``graph``; return the terminal
        SWA ``H2D`` op_id (to be appended to the graph's finished_ops_ids so it
        joins the VIRTUAL barrier alongside the full-KV H2D).

        Mirrors the full-KV GET graph: the SWA ``H2D`` (CPU SWA slot -> GPU swa
        pool) depends on the staging ops ``DISK2H`` / ``REMOTE2H`` when the SWA
        bytes are sourced from SSD / REMOTE. (CPU-resident SWA needs no staging
        op, like a CPU full-KV hit.) All ops carry ``is_swa=True``. Returns None
        when disabled / empty.
        """
        h2d_id = self.build_swa_op(
            graph, TransferType.H2D, cpu_slot_ids, gpu_slot_ids,
            dp_client_id=dp_client_id,
        )
        if h2d_id is None:
            return None
        if ssd_slot_ids is not None and np.asarray(ssd_slot_ids).size > 0:
            ssd2h_id = self.build_swa_op(
                graph, TransferType.DISK2H, ssd_slot_ids, cpu_slot_ids,
                dp_client_id=dp_client_id,
            )
            if ssd2h_id is not None:
                graph.add_dependency(h2d_id, ssd2h_id)
        if remote_slot_ids is not None and np.asarray(remote_slot_ids).size > 0:
            remote2h_id = self.build_swa_op(
                graph, TransferType.REMOTE2H, remote_slot_ids, cpu_slot_ids,
                dp_client_id=dp_client_id,
                mooncake_tail_hashes=mooncake_tail_hashes,
            )
            if remote2h_id is not None:
                graph.add_dependency(h2d_id, remote2h_id)
        return h2d_id

    def build_put_chain(self,
                        graph: TransferOpGraph,
                        gpu_slot_ids: np.ndarray,
                        cpu_slot_ids: np.ndarray,
                        ssd_slot_ids: Optional[np.ndarray] = None,
                        remote_slot_ids: Optional[np.ndarray] = None,
                        dp_client_id: int = 0,
                        mooncake_tail_hashes: Optional[List[str]] = None) -> Optional[int]:
        """Build the PUT-side SWA store chain into ``graph``; return the SWA
        ``D2H`` op_id (to be appended to the graph's finished_ops_ids).

        Mirrors the full-KV PUT graph: the SWA ``D2H`` (GPU swa pool -> CPU SWA
        slot) is the reported op; the SWA ``H2DISK`` / ``H2REMOTE`` write-through
        ops depend on the SWA ``D2H`` but are fire-and-forget (NOT reported),
        exactly like the full-KV ``D2H`` / ``H2DISK`` / ``H2REMOTE``. All ops
        carry ``is_swa=True``. Returns None when disabled / empty.
        """
        d2h_id = self.build_swa_op(
            graph, TransferType.D2H, gpu_slot_ids, cpu_slot_ids,
            dp_client_id=dp_client_id,
        )
        if d2h_id is None:
            return None
        if ssd_slot_ids is not None and np.asarray(ssd_slot_ids).size > 0:
            h2ssd_id = self.build_swa_op(
                graph, TransferType.H2DISK, cpu_slot_ids, ssd_slot_ids,
                dp_client_id=dp_client_id,
            )
            if h2ssd_id is not None:
                graph.add_dependency(h2ssd_id, d2h_id)
        if remote_slot_ids is not None and np.asarray(remote_slot_ids).size > 0:
            h2remote_id = self.build_swa_op(
                graph, TransferType.H2REMOTE, cpu_slot_ids, remote_slot_ids,
                dp_client_id=dp_client_id,
                mooncake_tail_hashes=mooncake_tail_hashes,
            )
            if h2remote_id is not None:
                graph.add_dependency(h2remote_id, d2h_id)
        return d2h_id
