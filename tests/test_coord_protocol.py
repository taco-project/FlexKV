"""Unit tests for ``flexkv.cache.coordination_protocol``.

Phase D-4 (proposal_unify_with_graph_dispatch_2026-05-15.md): only
``RemoteReadyMsg`` and ``FailureReportMsg`` survive — the
``CoordQuery*`` / ``CoordGet*`` / ``CoordPut*`` messages were tied to
the old per-SD ZMQ coord protocol and are replaced by the unified
graph-dispatch path (peer SD acks now come back as
``CompletedOp(sd_key, contributing_node_id)`` via
``TransferManagerMultiNodeHandle._completion_sink``).
"""
from __future__ import annotations

import pickle

import pytest

from flexkv.common.dist_reuse.coordination_protocol import (
    CoordMsgType,
    EpochVerifyError,
    FailureReportMsg,
    RemoteReadyMsg,
    decode_coord_message,
    encode_coord_message,
)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class TestType:
    @pytest.mark.parametrize("cls,expected", [
        (RemoteReadyMsg, CoordMsgType.REMOTE_READY),
        (FailureReportMsg, CoordMsgType.FAILURE_REPORT),
    ])
    def test_class_type_attached(self, cls, expected):
        assert cls.type is expected
        # Also accessible via instance
        msg = cls()
        assert msg.type is expected


# ---------------------------------------------------------------------------
# Encode / decode round trips
# ---------------------------------------------------------------------------
class TestRoundTrip:
    @pytest.mark.parametrize("msg", [
        RemoteReadyMsg(
            sender_instance_id="inst1",
            sender_epoch="epoch-1",
            request_id=42,
            sd_key="abc:pp1/2:tpn0/1:nsa0",
            distributed_node_id=7,
            mooncake_addr="10.0.0.1:5555",
            zmq_addr="tcp://10.0.0.1:6666",
        ),
        FailureReportMsg(
            sender_instance_id="inst-r",
            sender_epoch="e",
            request_id=4,
            peer_instance_id="inst1",
            failed_block_hashes=[42, 43],
            error="hca dropped",
        ),
    ])
    def test_encode_decode_round_trip(self, msg):
        payload = encode_coord_message(msg)
        # ``type`` is preserved as the enum value (str)
        assert payload["type"] == msg.type.value
        out = decode_coord_message(payload)
        assert out == msg
        assert out is not msg
        assert type(out) is type(msg)

    @pytest.mark.parametrize("msg", [
        RemoteReadyMsg(),
        FailureReportMsg(error="x"),
    ])
    def test_pickle_round_trip(self, msg):
        # Current ZMQ transport uses pickle, so make sure the dataclasses
        # survive a pickle round-trip (default fields, mutable defaults, etc.)
        out = pickle.loads(pickle.dumps(msg))
        assert out == msg


# ---------------------------------------------------------------------------
# Decode error cases
# ---------------------------------------------------------------------------
class TestDecodeErrors:
    def test_missing_type(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            decode_coord_message({"sender_instance_id": "x"})

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="unknown type"):
            decode_coord_message({"type": "definitely-not-a-real-type"})

    def test_unknown_field(self):
        with pytest.raises(ValueError, match="unknown fields"):
            decode_coord_message({
                "type": CoordMsgType.REMOTE_READY.value,
                "sender_instance_id": "x",
                "this_field_does_not_exist": 1,
            })

    def test_encode_rejects_non_message(self):
        with pytest.raises(TypeError):
            encode_coord_message({"type": "fake"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# EpochVerifyError sanity
# ---------------------------------------------------------------------------
def test_epoch_verify_error_is_runtime_error():
    assert issubclass(EpochVerifyError, RuntimeError)
    with pytest.raises(EpochVerifyError):
        raise EpochVerifyError("stale")


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
def test_default_message_has_empty_lists_not_shared():
    """Make sure mutable default fields (``field(default_factory=list)``) do
    NOT share state between instances — a classic dataclass foot-gun."""
    a = FailureReportMsg()
    b = FailureReportMsg()
    a.failed_block_hashes.append(1)
    assert b.failed_block_hashes == []
