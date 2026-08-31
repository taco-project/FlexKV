import array
import os
import socket
import struct
import tempfile
import threading

import pytest

from flexkv.integration.vllm.layerwise_sync import (
    LayerwiseCounterPool,
    LayerwiseLoadMetadata,
    LayerwiseStepCoordinator,
)


pytestmark = pytest.mark.unit


def _short_socket_path(prefix):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".sock", dir="/tmp")
    os.close(fd)
    os.unlink(path)
    return path


class _FakeFDs:
    def __init__(self):
        self.next_fd = 10
        self.reads = []
        self.closed = []

    def create(self):
        fd = self.next_fd
        self.next_fd += 1
        return fd

    def read(self, fd):
        self.reads.append(fd)
        return 1

    def close(self, fd):
        self.closed.append(fd)


def _pool(layer_names=("layer.0", "layer.1", "layer.2")):
    fake = _FakeFDs()
    pool = LayerwiseCounterPool(
        layer_names,
        fd_factory=fake.create,
        fd_reader=fake.read,
        fd_closer=fake.close,
    )
    return pool, fake


def test_step_coordinator_round_robins_only_load_steps():
    coordinator = LayerwiseStepCoordinator(enabled=True, num_counters=3)
    assert coordinator.build_metadata(False) == LayerwiseLoadMetadata(enabled=True)
    assert [coordinator.build_metadata(True).counter_id for _ in range(5)] == [
        0, 1, 2, 0, 1]

    disabled = LayerwiseStepCoordinator(enabled=False)
    assert disabled.build_metadata(True) == LayerwiseLoadMetadata(enabled=False)
    assert coordinator.external_match_is_async(True) is False
    assert disabled.external_match_is_async(True) is True
    assert disabled.external_match_is_async(False) is False
    assert coordinator.launch_kwargs(
        LayerwiseLoadMetadata(enabled=True, counter_id=2, has_load=True)
    ) == {
        "as_batch": True,
        "layerwise_transfer": True,
        "counter_id": 2,
    }


def test_counter_pool_waits_once_per_layer_and_releases_on_last_layer():
    pool, fake = _pool()
    pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=1, has_load=True))

    pool.wait("layer.0")
    pool.wait("layer.0")
    pool.wait("layer.1")
    assert fake.reads == [13, 14]

    pool.wait("layer.2")
    assert fake.reads == [13, 14, 15]

    # Last-layer completion releases the active counter; later hooks are no-op.
    pool.wait("layer.0")
    assert fake.reads == [13, 14, 15]


def test_counter_pool_disabled_or_empty_metadata_is_noop():
    pool, fake = _pool()
    pool.bind(LayerwiseLoadMetadata(enabled=False))
    pool.wait("layer.0")
    pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=-1, has_load=False))
    pool.wait("layer.1")
    assert fake.reads == []


def test_counter_pool_rejects_unknown_layer_and_counter():
    pool, _ = _pool()
    with pytest.raises(ValueError, match="counter_id"):
        pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=9, has_load=True))

    pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=0, has_load=True))
    with pytest.raises(KeyError, match="unknown layer_name"):
        pool.wait("layer.99")


def test_failed_wait_poisons_pool_to_prevent_stale_counter_reuse():
    attempts = []

    def flaky_read(fd):
        attempts.append(fd)
        if len(attempts) == 1:
            raise TimeoutError("simulated")
        return 1

    fake = _FakeFDs()
    pool = LayerwiseCounterPool(
        ("layer.0",),
        fd_factory=fake.create,
        fd_reader=flaky_read,
        fd_closer=fake.close,
    )
    pool.bind(LayerwiseLoadMetadata(enabled=True, counter_id=0, has_load=True))
    with pytest.raises(TimeoutError):
        pool.wait("layer.0")
    with pytest.raises(RuntimeError, match="unusable"):
        pool.bind(
            LayerwiseLoadMetadata(enabled=True, counter_id=0, has_load=True))
    assert attempts == [10]


def test_bind_rejects_new_step_until_previous_counter_finishes():
    pool, fake = _pool(layer_names=("a", "b"))
    metadata = LayerwiseLoadMetadata(enabled=True, counter_id=0, has_load=True)
    pool.bind(metadata)
    pool.wait("a")
    with pytest.raises(RuntimeError, match="previous"):
        pool.bind(metadata)
    pool.wait("b")
    pool.bind(metadata)
    pool.wait("a")
    assert fake.reads == [10, 11, 10]


def test_counter_pool_close_is_idempotent():
    pool, fake = _pool(layer_names=("a", "b"))
    expected_fds = [fd for counter in pool.fds for fd in counter]
    pool.close()
    pool.close()
    assert fake.closed == expected_fds


def test_send_to_worker_matches_layerwise_worker_wire_contract():
    socket_path = _short_socket_path("flexkv-lw-")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)

    received = {}

    def receive():
        conn, _ = server.accept()
        with conn:
            metadata = conn.recv(16)
            received["metadata"] = struct.unpack("iiii", metadata)
            counters = {}
            for _ in range(3):
                msg, ancdata, _flags, _addr = conn.recvmsg(
                    4, socket.CMSG_SPACE(2 * struct.calcsize("i")))
                counter_id = struct.unpack("i", msg)[0]
                for level, kind, data in ancdata:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        fds = array.array("i")
                        fds.frombytes(data[:2 * fds.itemsize])
                        counters[counter_id] = list(fds)
            received["counters"] = counters
            conn.sendall(b"\x01")

    server_thread = threading.Thread(target=receive)
    server_thread.start()

    owned_write_fds = []

    def pipe_fd():
        read_fd, write_fd = os.pipe()
        owned_write_fds.append(write_fd)
        return read_fd

    pool = LayerwiseCounterPool(
        ("layer.0", "layer.1"),
        fd_factory=pipe_fd,
        fd_reader=lambda _fd: 1,
    )
    try:
        pool.send_to_worker(
            socket_path,
            tp_rank_per_node=1,
            tp_size_per_node=2,
            timeout_s=2,
            retry_interval_s=0.01,
        )
    finally:
        server_thread.join(timeout=2)
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        for fds in received.get("counters", {}).values():
            for fd in fds:
                os.close(fd)
        pool.close()
        for fd in owned_write_fds:
            os.close(fd)

    assert received["metadata"] == (1, 2, 2, 3)
    assert set(received["counters"]) == {0, 1, 2}
    assert all(len(fds) == 2 for fds in received["counters"].values())


def test_send_to_worker_retries_after_nack():
    socket_path = _short_socket_path("flexkv-retry-")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(2)
    attempts = []

    def receive():
        for ack in (b"\x00", b"\x01"):
            conn, _ = server.accept()
            with conn:
                conn.recv(16)
                for _ in range(3):
                    _msg, ancdata, _flags, _addr = conn.recvmsg(
                        4, socket.CMSG_SPACE(struct.calcsize("i")))
                    for level, kind, data in ancdata:
                        if (level == socket.SOL_SOCKET
                                and kind == socket.SCM_RIGHTS):
                            fds = array.array("i")
                            fds.frombytes(data[:fds.itemsize])
                            for fd in fds:
                                os.close(fd)
                attempts.append(ack)
                conn.sendall(ack)

    thread = threading.Thread(target=receive)
    thread.start()
    write_fds = []

    def pipe_fd():
        read_fd, write_fd = os.pipe()
        write_fds.append(write_fd)
        return read_fd

    pool = LayerwiseCounterPool(
        ("layer.0",), fd_factory=pipe_fd, fd_reader=lambda _fd: 1)
    try:
        pool.send_to_worker(
            socket_path,
            tp_rank_per_node=0,
            tp_size_per_node=1,
            timeout_s=2,
            retry_interval_s=0.01,
        )
    finally:
        thread.join(timeout=2)
        server.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        pool.close()
        for fd in write_fds:
            os.close(fd)

    assert attempts == [b"\x00", b"\x01"]
