"""NIXL transport of PEERSSD2H (`flexkv.transfer.workers.peer_ssd_nixl`).

* The SSD block -> (file, offset) map agrees with what the io_uring engine
  wrote (multi-device, multi-file layout).
* Two endpoints in one process: arbitrary, scattered block lists read
  byte-exact, concurrent requests from two threads, a bad block id comes back
  as an error, a peer that stopped serving times out.
Skipped without NIXL (UCX + POSIX plugins) or `flexkv.c_ext`.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest
import torch

nixl_api = pytest.importorskip("nixl._api")

try:
    from flexkv import c_ext
    from flexkv.c_ext import transfer_kv_blocks_ssd
    from flexkv.transfer.workers.peer_ssd_nixl import NixlSSDEndpoint, ssd_block_location
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"flexkv.c_ext / NIXL endpoint unavailable: {exc}", allow_module_level=True)

BLOCK = 8192


def _pattern(i: int) -> np.ndarray:
    return np.random.default_rng(1000 + i).integers(0, 255, BLOCK, dtype=np.uint8)


def _write_with_io_uring(tmp_path, devices: int, files_per_device: int, blocks_per_file: int):
    """Lay blocks out with FlexKV's own engine; return (files, num_blocks)."""
    files = {}
    for d in range(devices):
        files[d] = []
        for f in range(files_per_device):
            path = str(tmp_path / f"ssd_{d}_{f}.bin")
            with open(path, "wb") as fh:
                os.truncate(fh.fileno(), blocks_per_file * BLOCK)
            files[d].append(path)
    n = devices * files_per_device * blocks_per_file
    cpu = torch.from_numpy(np.concatenate([_pattern(i) for i in range(n)]))
    ioctx = c_ext.SSDIOCTX(files, len(files), 64, 0)
    ids = torch.arange(n, dtype=torch.int64)
    transfer_kv_blocks_ssd(ioctx, torch.tensor([0], dtype=torch.int32), cpu.data_ptr(), ids, ids,
                           BLOCK, 0, BLOCK, 0, BLOCK, BLOCK, False, blocks_per_file, 1, 4, 1,
                           ssd_io_opt=True)
    return files, n


def test_block_location_matches_the_io_uring_layout(tmp_path):
    files, n = _write_with_io_uring(tmp_path, devices=2, files_per_device=3, blocks_per_file=4)
    for i in range(n):
        path, off = ssd_block_location(files, i, BLOCK)
        with open(path, "rb") as fh:
            fh.seek(off)
            assert np.array_equal(np.frombuffer(fh.read(BLOCK), np.uint8), _pattern(i)), i


@pytest.fixture
def pair(tmp_path):
    files, n = _write_with_io_uring(tmp_path, devices=1, files_per_device=2, blocks_per_file=16)
    port = 20000 + os.getpid() % 10000
    src_buf = np.zeros(BLOCK * 4, dtype=np.uint8)
    dst_buf = np.zeros(BLOCK * n, dtype=np.uint8)
    server = NixlSSDEndpoint(f"t-srv-{os.getpid()}", port=port, cpu_ptr=src_buf.ctypes.data,
                             cpu_bytes=src_buf.nbytes, block_bytes=BLOCK, ssd_files=files,
                             chunk_blocks=3, num_chunks=2)
    client = NixlSSDEndpoint(f"t-cli-{os.getpid()}", port=port + 1, cpu_ptr=dst_buf.ctypes.data,
                             cpu_bytes=dst_buf.nbytes, block_bytes=BLOCK, ssd_files={})
    client.connect(server.name, "127.0.0.1", port, 30)
    yield server, client, dst_buf, n
    client.close()
    server.close()


def test_scattered_blocks_land_byte_exact(pair):
    server, client, dst, n = pair
    src = [7, 3, 30, 0, 12, 25, 25, 1]
    out = [31, 0, 5, 9, 2, 17, 18, 22]
    client.read(server.name, src, out, 30)
    for s, d in zip(src, out):
        assert np.array_equal(dst[d * BLOCK:(d + 1) * BLOCK], _pattern(s)), (s, d)


def test_concurrent_requests(pair):
    server, client, dst, n = pair
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
        assert np.array_equal(dst[i * BLOCK:(i + 1) * BLOCK], _pattern(i)), i


def test_a_bad_block_is_an_error(pair):
    server, client, dst, n = pair
    with pytest.raises(RuntimeError, match="failed the read"):
        client.read(server.name, [0, n + 100], [0, 1], 30)


def test_a_silent_peer_times_out(pair):
    server, client, dst, n = pair
    server.close()  # stops serving; the agent stays reachable
    with pytest.raises(TimeoutError):
        client.read(server.name, [0], [0], 1)
