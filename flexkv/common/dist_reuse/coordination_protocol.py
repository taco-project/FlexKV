"""Wire format for Master ↔ Remote coordination of distributed KV reuse.

Phase D-4 (proposal_unify_with_graph_dispatch_2026-05-15.md): the
``CoordQuery*`` / ``CoordGet*`` / ``CoordPut*`` message types from the
early implementation are **deleted** here.  They were the dataclasses
behind the old per-SD ZMQ coord protocol; the unified graph-dispatch
path replaces them — peer-SD acks now arrive as
``CompletedOp(sd_key, contributing_node_id)`` through the existing
``TransferManagerMultiNodeHandle`` polling thread.

What survives:

* :class:`RemoteReadyMsg` — Remote → Master one-shot bootstrap ack.
  Used during instance startup before the graph-dispatch path is up.
* :class:`FailureReportMsg` — Remote → Master data-plane failure ping
  (Layer-2 closed loop in design doc §4.3.2).  Asynchronous and
  orthogonal to the per-PUT/GET coordination flow, so it stays on the
  ZMQ side channel.

All messages remain ``dataclass``-based so they pickle cleanly through
the current ZMQ payload format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Type, Union


__all__ = [
    "CoordMsgType",
    "RemoteReadyMsg",
    "FailureReportMsg",
    "EpochVerifyError",
    "encode_coord_message",
    "decode_coord_message",
]


class CoordMsgType(str, Enum):
    """Discriminator embedded in every coordination message.

    Phase D-4: only the bootstrap + failure-report types remain.  The
    PUT/GET coordination types (COORD_QUERY / COORD_GET_* / COORD_PUT_*)
    were deleted with the unified graph-dispatch refactor.
    """

    REMOTE_READY = "remote_ready"
    FAILURE_REPORT = "failure_report"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
@dataclass
class _BaseCoordMsg:
    """Common fields embedded in every wire message.

    ``epoch`` carries the *sender's* expected ``session_epoch``; receivers
    cross-check it against their own and raise :class:`EpochVerifyError`
    when they disagree (design doc §4.3.2 anti-replay rule).
    """

    # Class-level discriminator; every concrete subclass overrides this in
    # ``__init_subclass__``.
    type: ClassVar[CoordMsgType]

    # Identifies which FlexKV instance authored the message.  This is the
    # value of ``CacheConfig.instance_id`` of the sender.
    sender_instance_id: str = ""

    # Per-message correlation ID (set by the originator; copied verbatim
    # into the ack).  ``-1`` is the sentinel for "not yet assigned".
    request_id: int = -1

    # Snapshot of the sender's ``session_epoch`` at the time the message was
    # sent.  Required by the receiver to invalidate stale traffic across an
    # instance restart (design doc §4.3.2).  Empty string permitted in tests.
    sender_epoch: str = ""


def _msg_class_with_type(cls: Type[_BaseCoordMsg], type_value: CoordMsgType) -> Type[_BaseCoordMsg]:
    """Helper to attach a ``type`` ClassVar to a dataclass."""
    cls.type = type_value  # type: ignore[assignment]
    return cls


# ---------------------------------------------------------------------------
# Remote → Master: ready handshake
# ---------------------------------------------------------------------------
@dataclass
class RemoteReadyMsg(_BaseCoordMsg):
    """Sent by a Remote once its RedisMeta + Mooncake init finishes.

    The Master collects N-1 of these (one per non-master SD) before the
    instance is considered ready (design doc §4.6.2 step 4).
    """

    sd_key: str = ""
    distributed_node_id: int = -1
    mooncake_addr: str = ""
    zmq_addr: str = ""


_msg_class_with_type(RemoteReadyMsg, CoordMsgType.REMOTE_READY)


# ---------------------------------------------------------------------------
# Layer-2 failure closed loop
# ---------------------------------------------------------------------------
@dataclass
class FailureReportMsg(_BaseCoordMsg):
    """Remote → Master: a Mooncake P2P read or write hit a hard error.

    The Master invalidates the affected prefix in the aggregate radix and
    optionally escalates to ``invalidate_all_by_instance`` after seeing
    enough failures from the same peer (design doc §4.3.2 Layer-2).
    """

    peer_instance_id: str = ""
    failed_block_hashes: List[int] = field(default_factory=list)
    error: str = ""


_msg_class_with_type(FailureReportMsg, CoordMsgType.FAILURE_REPORT)


# ---------------------------------------------------------------------------
# Encoding helpers — protocol-level, not transport-level
# ---------------------------------------------------------------------------
class EpochVerifyError(RuntimeError):
    """Raised when a receiver detects a stale ``sender_epoch``.  The caller
    is expected to translate this into a ``STALE_EPOCH`` response and let
    the sender invalidate its view of the affected peer."""


_TYPE_TO_CLASS: Dict[CoordMsgType, Type[_BaseCoordMsg]] = {
    CoordMsgType.REMOTE_READY: RemoteReadyMsg,
    CoordMsgType.FAILURE_REPORT: FailureReportMsg,
}


AnyCoordMsg = Union[
    RemoteReadyMsg,
    FailureReportMsg,
]


def encode_coord_message(msg: AnyCoordMsg) -> Dict[str, Any]:
    """Convert a dataclass message to a plain ``dict`` (transport-agnostic).

    The result is JSON-serializable as long as the embedded fields are
    (block hashes are signed 64-bit ints, which JSON handles fine).
    """
    if not isinstance(msg, tuple(_TYPE_TO_CLASS.values())):
        raise TypeError(f"encode_coord_message: not a coord message: {type(msg).__name__}")
    out: Dict[str, Any] = {"type": msg.type.value}
    for f in msg.__dataclass_fields__.values():  # type: ignore[attr-defined]
        out[f.name] = getattr(msg, f.name)
    return out


def decode_coord_message(payload: Dict[str, Any]) -> AnyCoordMsg:
    """Inverse of :func:`encode_coord_message`.

    Raises ``ValueError`` if ``payload['type']`` is missing or unknown.
    """
    raw_type = payload.get("type")
    if raw_type is None:
        raise ValueError("decode_coord_message: missing 'type' field")
    try:
        mtype = CoordMsgType(raw_type)
    except ValueError as e:
        raise ValueError(f"decode_coord_message: unknown type {raw_type!r}") from e
    cls = _TYPE_TO_CLASS[mtype]
    # Drop ``type`` before delegating to the dataclass ctor.
    fields_payload = {k: v for k, v in payload.items() if k != "type"}
    # Validate that no unknown extra keys leak in (catches schema drift).
    valid_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    extra = set(fields_payload) - valid_names
    if extra:
        raise ValueError(f"decode_coord_message: unknown fields for {mtype.value}: {sorted(extra)}")
    return cls(**fields_payload)  # type: ignore[return-value]
