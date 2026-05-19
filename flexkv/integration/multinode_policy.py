"""Multi-node role decision helpers for the sglang ↔ FlexKV connector.

Design doc §4.5.5 splits ``is_multinode`` into **two independent axes**:

* ``is_multinode_tp``  — one TP group spans >1 physical node.
  Each such node runs a full SD-Remote (``TransferManagerOnRemote``)
  with its own RedisMeta + Mooncake registration.

* ``is_multinode_cp``  — CP > 1 and the CP group spans >1 physical node.
  CP all-gather makes every ``cp_rank``'s KV pool bit-wise identical,
  so non-leader CP ranks **do not** run a full SD-Remote; they only
  need a GPU-registration stub (``KVTPClient``) + receive coordinated
  H2D commands routed by the sync-leader rank.

The master connector (``flexkv_connector.py``) currently conflates the
two under a ``nnodes > 1 and node_rank > 0 and local_rank == 0`` rule
of thumb.  This module provides the policy functions that the
connector **should** call once we can exercise cross-node boots on a
two-machine GPU setup (tracked as §2.4 in
``docs/dist_reuse/implementation_gap_2026-05-11.md``).

Everything here is pure Python / pure logic so it is unit-testable
without torch, CUDA, or a running sglang process.  See
``FlexKV/tests/test_multinode_role_policy.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RemoteProcessRole(str, Enum):
    """What, if anything, should this rank's local ``FlexKVConnector``
    spawn as its ``TransferManagerOnRemote`` process?

    The three roles correspond to the three paths in the design doc:

    * ``MASTER``: this rank is the sync-leader of an instance.  It runs
      the full ``KVManager`` (owns the Master coordinator, writes to
      Redis, owns Mooncake TransferEngine).  No ``TransferManagerOnRemote``
      process is spawned — the master IS the transfer authority.

    * ``SD_REMOTE_FULL``: this rank sits on a non-master node of a
      *cross-node TP/PP* group.  It must spawn a full
      ``TransferManagerOnRemote`` that:

        - registers its local CPU block pool with Mooncake,
        - writes its ``sd:<sd_key>:node/block`` entries to Redis,
        - replies ``RemoteReadyMsg`` so the Master can discover it,
        - serves coordinated GET/PUT commands from the Master.

    * ``CP_PEER_REGISTRATION_ONLY``: this rank is a non-leader CP rank.
      The design doc §4.5.5 + §4.12.2 (5) says these ranks only need
      a lightweight ``TransferManagerOnRemote`` *stub* that:

        - registers its GPU blocks with ``KVTPClient`` (so the sync
          leader's H2D path can write to it), AND
        - listens for coordinated H2D slot-mapping commands.

      It does **NOT** touch RedisMeta or Mooncake — CP all-gather
      already makes this rank's content identical to the sync leader.

    * ``NO_REMOTE``: single-node instance — no ``TransferManagerOnRemote``
      spawn at all.  Legacy single-box behaviour.
    """
    MASTER = "master"
    SD_REMOTE_FULL = "sd_remote_full"
    CP_PEER_REGISTRATION_ONLY = "cp_peer_registration_only"
    NO_REMOTE = "no_remote"


@dataclass(frozen=True)
class RankTopology:
    """Topology facts about a single rank, as seen from the connector.

    All fields are scalars so this is trivially serialisable / hashable
    for tests.  The connector extracts these from ``ModelConfig`` +
    ``server_args``; tests construct them directly.
    """

    # Core dimensions
    nnodes: int
    node_rank: int
    local_rank: int           # 0..gpus_per_node-1

    # Rolled-up FlexKV topology (see ModelConfig docstring)
    is_multinode_tp: bool     # ``tp_node_count > 1``
    is_multinode_cp: bool     # CP > 1 and CP crosses node boundary

    # Optional sync-leader hint.  If the caller already knows whether
    # this rank is the sync leader (from ``sglang`` group metadata),
    # pass ``is_sync_leader``.  Otherwise the default heuristic kicks
    # in: ``(local_rank == 0 and node_rank == 0)``.
    is_sync_leader: Optional[bool] = None


def decide_remote_role(topo: RankTopology) -> RemoteProcessRole:
    """Compute the role of a rank.

    Decision table (see design doc §4.5.5, simplified):

    ===================  ================  ================  =================
    Single-node?         is_multinode_tp   is_multinode_cp   Role
    ===================  ================  ================  =================
    yes (nnodes == 1)    (ignored)         (ignored)         NO_REMOTE
    no, rank 0 box       False             False             NO_REMOTE
    no, rank 0 box       any               any               MASTER
    no, off-master box   is_multinode_tp   any               SD_REMOTE_FULL
    no, off-master box   False             is_multinode_cp   CP_PEER_REGISTRATION_ONLY
    ===================  ================  ================  =================

    Where "rank 0 box" means ``node_rank == 0``.  Across both
    ``is_multinode_tp`` and ``is_multinode_cp`` axes we place the
    Master on ``node_rank==0`` by convention (this matches the current
    ``flexkv_connector.py`` assumption that sync leader is
    ``node_rank==0``).

    Note on CP + TP combined: when BOTH ``is_multinode_tp`` and
    ``is_multinode_cp`` are True for an off-master rank, it runs
    ``SD_REMOTE_FULL`` — the TP-side state requires a full SD-Remote;
    CP-side reduction is handled *inside* that remote's sync leader
    (same as ``is_multinode_tp=True, is_multinode_cp=False``).  We
    never downgrade a TP-remote to a CP-peer-only stub.
    """
    _validate(topo)

    # Single-node instance: nothing to spawn.
    if topo.nnodes <= 1:
        return RemoteProcessRole.NO_REMOTE

    # Master node — spawn nothing, the in-process KVManager IS the
    # transfer authority.
    if topo.node_rank == 0:
        if not topo.is_multinode_tp and not topo.is_multinode_cp:
            # Multi-node deployment but THIS instance spans only one
            # node — e.g. DP > 1 across nodes but each DP instance is
            # single-node.  No remote peer exists in this instance.
            return RemoteProcessRole.NO_REMOTE
        return RemoteProcessRole.MASTER

    # Off-master nodes —
    # TP takes priority: a TP-split SD cannot be served by a CP-only stub.
    if topo.is_multinode_tp:
        return RemoteProcessRole.SD_REMOTE_FULL

    if topo.is_multinode_cp:
        return RemoteProcessRole.CP_PEER_REGISTRATION_ONLY

    # Off-master but neither TP nor CP is multi-node.  Today's legacy
    # code treats this the same as CP-peer (it spawns a
    # ``TransferManagerOnRemote`` on every non-master node when
    # ``nnodes > 1``).  We preserve that behaviour for bug-compat
    # during the migration; the ideal long-term answer is NO_REMOTE,
    # but flipping it here would break the existing code path that
    # the multi-node PP (``is_multinode_tp=False`` but
    # ``pp_size>1`` crossing nodes) relies on.
    #
    # TODO(dist_reuse-§2.4): revisit once ``is_multinode_pp`` has its
    # own property on ModelConfig.
    return RemoteProcessRole.SD_REMOTE_FULL


def is_sync_leader(topo: RankTopology) -> bool:
    """Heuristic used when the caller hasn't provided ``is_sync_leader``.

    Today's ``flexkv_connector.py`` infers it as ``local_rank == 0 and
    node_rank == 0``.  We keep that rule so drop-in replacement is
    byte-for-byte equivalent; callers that know better pass
    ``is_sync_leader`` explicitly on the ``RankTopology``.
    """
    if topo.is_sync_leader is not None:
        return bool(topo.is_sync_leader)
    return topo.node_rank == 0 and topo.local_rank == 0


def _validate(topo: RankTopology) -> None:
    if topo.nnodes <= 0:
        raise ValueError(f"nnodes must be > 0, got {topo.nnodes}")
    if not (0 <= topo.node_rank < topo.nnodes):
        raise ValueError(
            f"node_rank out of range: {topo.node_rank} / {topo.nnodes}"
        )
    if topo.local_rank < 0:
        raise ValueError(f"local_rank must be >= 0, got {topo.local_rank}")
