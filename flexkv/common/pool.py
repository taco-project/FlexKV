"""Which KV pool a transfer addresses.

A "pool" is one slot-id space: a set of blocks that an op's ``src_block_ids`` /
``dst_block_ids`` index into. Full KV and SWA are two of them. They are *not*
two kinds of transfer -- same direction, same tiers, same layout family, same
worker -- so what distinguishes them belongs on the op as a value, and on the
worker as a lookup key, rather than in a parallel set of classes, worker maps
and dispatch branches.

``is_swa: bool`` said the same thing in a shape that could only ever say two
things, and every consumer had to re-derive the pool from it (``_swa_worker_map``
vs ``_worker_map``, ``swa_cpu_blocks`` vs ``cpu_blocks``, ``PoolKind.SWA`` vs
``PoolKind.KV``). It stays as a derived property so the ~100 call sites that
read it keep working, but ``pool_id`` is the field.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import-time only. ``common.transfer`` imports ``PoolId`` from here, so a
    # module-scope import back would be a cycle; nothing below needs the class
    # object at runtime (``.name`` is read off the instance).
    from flexkv.common.transfer import DeviceType


class PoolId(IntEnum):
    """One slot-id space.

    ``IntEnum`` rather than ``Enum`` so a member sorts and compares as its
    value -- ``sorted(self._pools)`` in the worker relies on it, and so does
    the ``PoolId.FULL_KV == 0`` default below.

    Note it does NOT buy anything at the pickle boundary: measured, a member
    costs 1.53us to pickle against a bare int's 0.26us, because ``IntEnum``
    inherits ``Enum.__reduce_ex__`` and gets no discount for being an int.
    That is why ``WorkerTransferOp`` encodes it as an int by hand (see
    ``worker_op._encode_pool``) rather than letting pickle see the member.

    Values are wire-visible (``WorkerTransferOp`` carries one), so append
    rather than renumber. FULL_KV is 0 so that the historical ``is_swa=False``
    default maps onto it for free.
    """

    FULL_KV = 0
    SWA = 1

    @property
    def is_swa(self) -> bool:
        return self is PoolId.SWA

    @classmethod
    def from_is_swa(cls, is_swa: bool) -> "PoolId":
        return cls.SWA if is_swa else cls.FULL_KV


@dataclass(frozen=True)
class PoolEndpoint:
    """One pool's storage on one side of a transfer edge.

    An edge (CPU<->SSD, say) is served by one worker. That worker holds one
    binding per pool that has an endpoint on *both* of its sides -- which is
    exactly the condition under which a transfer of that pool over that edge is
    possible, and is why this pairs a pool with a tier rather than being a bare
    pool id.

    Frozen and hashable so it can key the engine's endpoint registry; ``str``
    reads as ``SWA@SSD`` in the log lines that report which pools a worker got.
    """

    pool_id: PoolId
    device_type: "DeviceType"

    def __str__(self) -> str:
        return f"{self.pool_id.name}@{self.device_type.name}"
