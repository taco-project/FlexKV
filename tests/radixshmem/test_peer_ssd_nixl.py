"""NIXL transport of PEERSSD2H (`flexkv.transfer.workers.peer_ssd_nixl`).

* The SSD block -> (file, offset) map agrees with what the io_uring engine
  wrote (multi-device, multi-file layout).
* Two endpoints in one process: arbitrary, scattered block lists read
  byte-exact, concurrent requests from two threads, a bad block id comes back
  as an error, a peer that stopped serving times out -- for a 4 KiB-aligned
  block (served with O_DIRECT) and a 512 B-aligned one (served buffered, like
  FlexKV's own engine writes it).
* A peer never writes after the reader gave up (the slots may be reused), and
  a reader that dropped a peer fails fast, then reconnects.
* A reader whose own endpoint is still starting fails at once.
Skipped without NIXL (UCX + POSIX plugins) or `flexkv.c_ext`.
"""
from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

nixl_api = pytest.importorskip("nixl._api")

try:
    from flexkv import c_ext
    from flexkv.c_ext import transfer_kv_blocks_ssd
    from flexkv.common.pool import PoolId
    from flexkv.transfer.workers.peer_ssd_nixl import (NixlRemoteSSDReader, NixlSSDEndpoint,
                                                       ssd_block_location)
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"flexkv.c_ext / NIXL endpoint unavailable: {exc}", allow_module_level=True)

BLOCK = 8192           # 4 KiB-aligned: served with O_DIRECT
BLOCK_512 = 8192 + 512  # only 512 B-aligned (like DeepSeek-V4's 1671680 B): served buffered
_port = [20000 + os.getpid() % 5000 * 2]


def _pattern(i: int, block: int = BLOCK) -> np.ndarray:
    return np.random.default_rng(1000 + i).integers(0, 255, block, dtype=np.uint8)


def _write_with_io_uring(tmp_path, devices: int, files_per_device: int, blocks_per_file: int,
                         block: int = BLOCK):
    """Lay blocks out with FlexKV's own engine; return (files, num_blocks)."""
    files = {}
    for d in range(devices):
        files[d] = []
        for f in range(files_per_device):
            path = str(tmp_path / f"ssd_{d}_{f}.bin")
            with open(path, "wb") as fh:
                os.truncate(fh.fileno(), blocks_per_file * block)
            files[d].append(path)
    n = devices * files_per_device * blocks_per_file
    cpu = torch.from_numpy(np.concatenate([_pattern(i, block) for i in range(n)]))
    ioctx = c_ext.SSDIOCTX(files, len(files), 64, 0)
    ids = torch.arange(n, dtype=torch.int64)
    transfer_kv_blocks_ssd(ioctx, torch.tensor([0], dtype=torch.int32), cpu.data_ptr(), ids, ids,
                           block, 0, block, 0, block, block, False, blocks_per_file, 1, 4, 1,
                           ssd_io_opt=True)
    return files, n


def test_block_location_matches_the_io_uring_layout(tmp_path):
    files, n = _write_with_io_uring(tmp_path, devices=2, files_per_device=3, blocks_per_file=4)
    for i in range(n):
        path, off = ssd_block_location(files, i, BLOCK)
        with open(path, "rb") as fh:
            fh.seek(off)
            assert np.array_equal(np.frombuffer(fh.read(BLOCK), np.uint8), _pattern(i)), i


@pytest.fixture(params=[BLOCK, BLOCK_512], ids=["direct", "buffered"])
def pair(tmp_path, request):
    block = request.param
    files, n = _write_with_io_uring(tmp_path, devices=1, files_per_device=2, blocks_per_file=16,
                                    block=block)
    port = _port[0]
    _port[0] += 2
    src_buf = np.zeros(block * 4, dtype=np.uint8)
    dst_buf = np.zeros(block * n, dtype=np.uint8)
    server = NixlSSDEndpoint(f"t-srv-{os.getpid()}-{port}", port=port, cpu_ptr=src_buf.ctypes.data,
                             cpu_bytes=src_buf.nbytes, block_bytes=block, ssd_files=files,
                             chunk_blocks=3, num_chunks=2)
    client = NixlSSDEndpoint(f"t-cli-{os.getpid()}-{port}", port=port + 1,
                             cpu_ptr=dst_buf.ctypes.data, cpu_bytes=dst_buf.nbytes,
                             block_bytes=block, ssd_files={})
    assert server.direct_io == (block % 4096 == 0)
    client.connect(server.name, "127.0.0.1", port, 30)
    yield server, client, dst_buf, n, block
    client.close()
    server.close()


def test_scattered_blocks_land_byte_exact(pair):
    server, client, dst, n, bs = pair
    src = [7, 3, 30, 0, 12, 25, 25, 1]
    out = [31, 0, 5, 9, 2, 17, 18, 22]
    client.read(server.name, src, out, 30)
    for s, d in zip(src, out):
        assert np.array_equal(dst[d * bs:(d + 1) * bs], _pattern(s, bs)), (s, d)


def test_a_rewritten_block_is_served_fresh(pair):
    """A slot reused after eviction is rewritten by FlexKV's engine; the next
    served read must see the new bytes, not a cached copy of the old ones."""
    server, client, dst, n, bs = pair
    client.read(server.name, [5], [0], 30)
    assert np.array_equal(dst[:bs], _pattern(5, bs))
    files = server._files
    fresh = _pattern(777, bs)
    ioctx = c_ext.SSDIOCTX(files, len(files), 64, 0)
    one = torch.tensor([5], dtype=torch.int64)
    blocks_per_file = os.path.getsize(files[0][0]) // bs
    transfer_kv_blocks_ssd(ioctx, torch.tensor([0], dtype=torch.int32),
                           torch.from_numpy(fresh).data_ptr(), one, torch.tensor([0]),
                           bs, 0, bs, 0, bs, bs, False, blocks_per_file, 1, 1, 1, ssd_io_opt=True)
    client.read(server.name, [5], [1], 30)
    assert np.array_equal(dst[bs:2 * bs], fresh)


def test_concurrent_requests(pair):
    server, client, dst, n, bs = pair
    errors = []

    def _read(lo, hi):
        try:
            client.read(server.name, list(range(lo, hi)), list(range(lo, hi)), 30)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=_read, args=(lo, lo + n // 4)) for lo in range(0, n, n // 4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for i in range(n):
        assert np.array_equal(dst[i * bs:(i + 1) * bs], _pattern(i, bs)), i


def test_a_bad_block_is_an_error(pair):
    server, client, dst, n, bs = pair
    with pytest.raises(RuntimeError, match="failed the read"):
        client.read(server.name, [0, n + 100], [0, 1], 30)


def test_a_silent_peer_times_out(pair):
    server, client, dst, n, bs = pair
    server.close()  # stops serving; the agent stays reachable
    with pytest.raises(TimeoutError):
        client.read(server.name, [0], [0], 1)


def test_an_expired_request_is_not_served(pair):
    server, client, dst, n, bs = pair
    with pytest.raises(RuntimeError, match="expired"):
        client.read(server.name, [0, 1], [0, 1], 30, write_window_s=-1)
    assert not dst[:2 * bs].any()


def test_a_stalled_peer_does_not_write_late_and_the_reader_reconnects(pair, monkeypatch):
    server, client, dst, n, bs = pair
    serve = server._serve

    def _stalled(sender, req, staging):  # first request: the peer stalls past the timeout
        monkeypatch.setattr(server, "_serve", serve)
        time.sleep(2.0)
        serve(sender, req, staging)
    monkeypatch.setattr(server, "_serve", _stalled)
    client.retry_s = 1.0
    with pytest.raises(TimeoutError):
        client.read(server.name, [3], [3], 1)
    with pytest.raises(RuntimeError, match="dropped"):  # fails fast while backing off
        client.connect(server.name, "127.0.0.1", server.port, 5)
    time.sleep(2.5)                      # the stalled peer has woken up and answered
    assert not dst[3 * bs:4 * bs].any()  # ... without writing into the given-up slot
    client.connect(server.name, "127.0.0.1", server.port, 30)
    client.read(server.name, [4], [5], 30)
    assert np.array_equal(dst[5 * bs:6 * bs], _pattern(4, bs))


def test_reads_fail_at_once_while_the_endpoint_starts():
    reader = NixlRemoteSSDReader.__new__(NixlRemoteSSDReader)
    reader._worker = SimpleNamespace(num_layers=4)
    reader._timeout = 30.0
    reader._ready = threading.Event()          # never set: still starting
    reader._not_ready_logged = -1e9
    reader._error = None
    t0 = time.monotonic()
    for _ in range(3):
        with pytest.raises(RuntimeError, match="still starting"):
            reader.read(1, PoolId.FULL_KV, torch.tensor([0]), torch.tensor([0]), 0, 4)
    assert time.monotonic() - t0 < 1.0


@pytest.mark.parametrize("same_name", [False, True], ids=["new-name", "same-name"])
def test_a_restarted_reader_is_served_at_once(pair, same_name):
    """A reader process restarts (same node): the peer still holds the old process's
    metadata, so the new process must come back under a new agent name."""
    server, client, dst, n, bs = pair
    client.read(server.name, [1], [1], 30)
    name, port = client.name, client.port + 100
    client.close()
    client.agent = None
    buf = np.zeros(bs * n, dtype=np.uint8)
    again = NixlSSDEndpoint(name if same_name else f"{name}@{time.time_ns()}", port=port,
                            cpu_ptr=buf.ctypes.data, cpu_bytes=buf.nbytes, block_bytes=bs, ssd_files={})
    try:
        again.connect(server.name, "127.0.0.1", server.port, 10)
        if same_name:
            with pytest.raises((TimeoutError, RuntimeError)):
                again.read(server.name, [2], [2], 5)
        else:
            again.read(server.name, [2], [2], 5)
            assert np.array_equal(buf[2 * bs:3 * bs], _pattern(2, bs))
    finally:
        again.close()


def _file_block(server, block: int, bs: int) -> np.ndarray:
    path, off = ssd_block_location(server._files, block, bs)
    with open(path, "rb") as fh:
        fh.seek(off)
        return np.frombuffer(fh.read(bs), np.uint8)


def test_scattered_writes_land_byte_exact_and_read_back(pair):
    """H2PEERSSD: the peer stores the writer's CPU blocks in its SSD blocks,
    at the io_uring engine's addresses, and serves them back."""
    server, client, cpu, n, bs = pair
    src = [4, 0, 9, 2, 7]
    dst = [30, 1, 11, 6, 17]
    for i, s in enumerate(src):
        cpu[s * bs:(s + 1) * bs] = _pattern(500 + i, bs)
    client.write(server.name, src, dst, 30)
    for i, d in enumerate(dst):
        assert np.array_equal(_file_block(server, d, bs), _pattern(500 + i, bs)), d
    cpu[:] = 0
    client.read(server.name, dst, list(range(len(dst))), 30)
    for i in range(len(dst)):
        assert np.array_equal(cpu[i * bs:(i + 1) * bs], _pattern(500 + i, bs)), i


def test_a_write_to_a_bad_block_touches_nothing(pair):
    server, client, cpu, n, bs = pair
    cpu[:bs] = _pattern(900, bs)
    with pytest.raises(RuntimeError, match="failed the write"):
        client.write(server.name, [0, 0], [3, n + 100], 30)
    assert np.array_equal(_file_block(server, 3, bs), _pattern(3, bs))


def test_an_expired_write_is_not_stored(pair):
    server, client, cpu, n, bs = pair
    cpu[:bs] = _pattern(901, bs)
    with pytest.raises(RuntimeError, match="expired"):
        client.write(server.name, [0], [2], 30, write_window_s=-1)
    assert np.array_equal(_file_block(server, 2, bs), _pattern(2, bs))
