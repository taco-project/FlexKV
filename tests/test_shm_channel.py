"""Round-trip tests for the CE↔TE ShmChannel.

Run with::

    python3 -m pytest tests/test_shm_channel.py -v

The test forks N producer processes and one consumer; each producer sends K
graph-shaped messages and reads back acks via its own result ring.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import threading
import time

import numpy as np
import pytest

from flexkv.common.transfer import CompletedOp
from flexkv.transfer.shm_channel import ShmChannel, ShmControlBlock


SERVER_ID = "shm_channel_test"


def _producer(channel_id: int, n: int, server_id: str) -> None:
    ch = ShmChannel(server_id, channel_id, create=False)
    for i in range(n):
        ch.submit_send({"channel": channel_id, "seq": i, "data": b"x" * 1024})
    # Wait for echoed acks (CompletedOps carrying channel/seq in graph_id/op_id).
    received = 0
    while received < n:
        msgs = ch.result_recv(timeout_s=2.0)
        received += len(msgs)
        for m in msgs:
            assert m.graph_id == channel_id


def _consumer(num_channels: int, total_per_channel: int, server_id: str) -> None:
    ctrl = ShmControlBlock(server_id, create=True)
    channels = [
        ShmChannel(server_id, i, create=True) for i in range(num_channels)
    ]
    ctrl.set_ready()
    pending = num_channels * total_per_channel
    while pending > 0:
        had_work = False
        for ch in channels:
            msgs = ch.submit_recv()
            if msgs:
                had_work = True
                for m in msgs:
                    # Echo back the (channel, seq) as a CompletedOp — the result
                    # ring is now typed to CompletedOp records.
                    ch.result_send([CompletedOp(graph_id=m["channel"],
                                                op_id=m["seq"])])
                    pending -= 1
        if not had_work:
            time.sleep(0.001)
    for ch in channels:
        ch.close()
        ch.unlink()
    ctrl.close()
    ctrl.unlink()


def _cleanup_shm(server_id: str, num_channels: int) -> None:
    for ch_id in range(num_channels):
        path = f"/dev/shm/flexkv_te_ch_{server_id}_{ch_id}"
        if os.path.exists(path):
            os.unlink(path)
    ctrl_path = f"/dev/shm/flexkv_te_ctrl_{server_id}"
    if os.path.exists(ctrl_path):
        os.unlink(ctrl_path)


def test_n_producer_one_consumer():
    server_id = f"{SERVER_ID}_npm"
    num_channels = 4
    total_per_channel = 32

    _cleanup_shm(server_id, num_channels)

    ctx = mp.get_context("spawn")
    consumer = ctx.Process(
        target=_consumer,
        args=(num_channels, total_per_channel, server_id),
    )
    consumer.start()

    # Wait for consumer to set up shm files.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if os.path.exists(f"/dev/shm/flexkv_te_ctrl_{server_id}"):
            break
        time.sleep(0.01)
    else:
        consumer.terminate()
        pytest.fail("consumer never created control block")

    # Wait for ready flag via control block.
    ctrl = ShmControlBlock(server_id, create=False)
    assert ctrl.wait_ready(timeout_s=5.0)
    ctrl.close()

    producers = [
        ctx.Process(
            target=_producer, args=(i, total_per_channel, server_id)
        )
        for i in range(num_channels)
    ]
    for p in producers:
        p.start()
    for p in producers:
        p.join(timeout=10.0)
        assert p.exitcode == 0, f"producer {p.pid} exit code {p.exitcode}"

    consumer.join(timeout=10.0)
    assert consumer.exitcode == 0


def test_single_round_trip_local():
    server_id = f"{SERVER_ID}_local"
    _cleanup_shm(server_id, 1)
    ctrl = ShmControlBlock(server_id, create=True)
    ch = ShmChannel(server_id, 0, create=True)
    try:
        ch.submit_send("hello")
        ch.submit_send({"k": 42})
        msgs = ch.submit_recv()
        assert msgs == ["hello", {"k": 42}]

        # Result ring carries fixed-width CompletedOp records; all fields must
        # round-trip, including the transfer_type string and the -1 sentinel.
        sent = [
            CompletedOp(graph_id=7, op_id=3, transfer_type="H2D",
                        num_blocks=12, num_bytes=98304),
            CompletedOp(graph_id=7, op_id=-1),  # graph-completed sentinel
        ]
        ch.result_send(sent)
        out = ch.result_recv(timeout_s=0.0)
        assert out == sent
        assert out[1].is_graph_completed()
    finally:
        ch.close()
        ch.unlink()
        ctrl.close()
        ctrl.unlink()


def test_submit_fragmentation():
    """A payload larger than one slot must fragment and round-trip intact,
    interleaved with small single-slot messages."""
    server_id = f"{SERVER_ID}_frag"
    _cleanup_shm(server_id, 1)
    ctrl = ShmControlBlock(server_id, create=True)
    # Small slots so a modest payload spans many fragments.
    ch = ShmChannel(server_id, 0, create=True,
                    submit_slots=1024, slot_size=4096)
    try:
        big = {"arr": np.arange(200_000, dtype=np.int64)}  # ~1.5 MB > slot
        small = {"k": 1}
        ch.submit_send(small)
        ch.submit_send(big)
        ch.submit_send(small)
        msgs = ch.submit_recv()
        assert len(msgs) == 3
        assert msgs[0] == small
        assert np.array_equal(msgs[1]["arr"], big["arr"])
        assert msgs[2] == small
    finally:
        ch.close()
        ch.unlink()
        ctrl.close()
        ctrl.unlink()


def test_submit_payload_too_large():
    """A payload that can't fit the whole ring is rejected, not deadlocked."""
    server_id = f"{SERVER_ID}_toobig"
    _cleanup_shm(server_id, 1)
    ctrl = ShmControlBlock(server_id, create=True)
    ch = ShmChannel(server_id, 0, create=True,
                    submit_slots=8, slot_size=4096)
    try:
        with pytest.raises(ValueError):
            ch.submit_send(b"x" * (8 * 4096))  # needs more fragments than slots
    finally:
        ch.close()
        ch.unlink()
        ctrl.close()
        ctrl.unlink()


def _make_big_graph(nbytes: int) -> dict:
    """A graph-shaped payload that pickles to ~nbytes (block-id arrays dominate)."""
    n = nbytes // 16  # two int64 arrays
    return {"src": np.arange(n, dtype=np.int64),
            "dst": np.arange(n, dtype=np.int64)}


def test_submit_big_graph_500kb():
    """A ~500 KB graph fragments across the default 32 KB slots and round-trips."""
    server_id = f"{SERVER_ID}_big500"
    _cleanup_shm(server_id, 1)
    ctrl = ShmControlBlock(server_id, create=True)
    ch = ShmChannel(server_id, 0, create=True)  # default 32 KB / 8192
    try:
        big = _make_big_graph(500 * 1024)
        blob_sz = len(pickle.dumps(big, protocol=pickle.HIGHEST_PROTOCOL))
        assert blob_sz > 500 * 1024, f"payload only {blob_sz} B"
        assert blob_sz > ch.slot_size, "payload must exceed one slot"
        ch.submit_send(big)
        out = ch.submit_recv()
        assert len(out) == 1
        assert np.array_equal(out[0]["src"], big["src"])
        assert np.array_equal(out[0]["dst"], big["dst"])
    finally:
        ch.close()
        ch.unlink()
        ctrl.close()
        ctrl.unlink()


def test_submit_small_graphs_high_rate():
    """4096 small multi-slot graphs at ~4096/s: a producer thread submits while a
    consumer thread drains, verifying no loss, correct order, and no ring-full
    stall at the default 8192-slot capacity."""
    server_id = f"{SERVER_ID}_hirate"
    _cleanup_shm(server_id, 1)
    ctrl = ShmControlBlock(server_id, create=True)
    ch = ShmChannel(server_id, 0, create=True)  # default 32 KB / 8192
    n_msgs = 4096
    small = {"payload": b"x" * (2 * ch.slot_size)}  # spans ~3 slots each
    received: list = []

    def consumer() -> None:
        while len(received) < n_msgs:
            received.extend(ch.submit_recv())

    try:
        t = threading.Thread(target=consumer, daemon=True)
        t.start()
        start = time.monotonic()
        for i in range(n_msgs):
            ch.submit_send({"seq": i, **small})
        t.join(timeout=30.0)
        elapsed = time.monotonic() - start
        assert len(received) == n_msgs, f"got {len(received)}/{n_msgs}"
        assert [m["seq"] for m in received] == list(range(n_msgs)), "order/loss"
        rate = n_msgs / elapsed
        assert rate >= 4096, f"throughput {rate:.0f}/s below 4096/s target"
    finally:
        ch.close()
        ch.unlink()
        ctrl.close()
        ctrl.unlink()
