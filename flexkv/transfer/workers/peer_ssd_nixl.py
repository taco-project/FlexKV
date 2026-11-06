"""NIXL transport for PEERSSD2H: a peer's SSD blocks -> this node's CPU pool.

NIXL has no one-sided read of a remote file, so the node holding the blocks
does the I/O (the ``remote_storage_example`` pattern of nixl.git): the reader
sends a request notification naming the peer's SSD blocks and where they go in
its own CPU pool; the peer reads them from its SSD files into a staging buffer
with the POSIX file plugin and writes them over UCX straight into the reader's
CPU slots, one chunk while the next is read, then answers OK (or ER + reason).

One :class:`NixlSSDEndpoint` per (node, pool) lives in the PEERSSD2H worker of
that pool. It is both sides: it serves this node's SSD to every peer (a
dispatcher thread routes notifications, server threads run the pipelines) and
reads peers' SSD into this node's CPU pool for the worker's ops.

Opt in with ``FLEXKV_REMOTE_SSD_READER=flexkv.transfer.workers.peer_ssd_nixl:make_reader``.
Every node publishes how to reach its endpoint (agent name, ip, port) next to
its SSD manifest; ``FLEXKV_NIXL_SSD_IP`` / ``FLEXKV_NIXL_SSD_PORT`` (port of the
FULL pool; SWA uses port + 1) / ``FLEXKV_NIXL_UCX_DEVICES`` /
``FLEXKV_NIXL_SSD_TIMEOUT_S`` tune it.
"""

from __future__ import annotations

import json
import mmap
import os
import queue
import socket
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from flexkv.common.debug import flexkv_logger
from flexkv.common.pool import PoolId
from flexkv.transfer import nixlutil
from flexkv.transfer.nixlutil import file_path_for_ssd_block, require_nixl

_REQ, _OK, _ERR = b"RQ", b"OK", b"ER"
_IDLE_SLEEP_S = 50e-6
_DIRECT_IO_ALIGN = 4096
_NOT_READY_LOG_EVERY_S = 30.0


class PeerUnavailable(RuntimeError):
    """An expected miss (this endpoint still starting, the peer not serving yet
    or backing off after a drop): the prefetch misses, no stack trace needed."""
    quiet = True


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, "") or default)


def _default_ip() -> str:
    ip = os.getenv("FLEXKV_NIXL_SSD_IP", "")
    if ip:
        return ip
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def use_direct_io(block_bytes: int) -> bool:
    """Whether served reads bypass the page cache. Same rule as FlexKV's own
    io_uring engine (csrc/transfer_ssd.cpp): O_DIRECT only for 4 KiB-aligned
    blocks. Otherwise that engine writes the blocks through the page cache
    too (DeepSeek-V4's 1671680 B blocks), and serving them buffered reads the
    same, current pages."""
    return int(block_bytes) % _DIRECT_IO_ALIGN == 0


def ssd_block_location(files: Dict[int, List[str]], block: int, block_bytes: int) -> Tuple[str, int]:
    """(file path, byte offset) of SSD block ``block``, as the io_uring engine
    addresses it: device ``block % num_devices``, then round-robin over that
    device's files."""
    num_devices = len(files)
    per_device = len(next(iter(files.values())))
    path, block_in_file = file_path_for_ssd_block(files, int(block), num_devices, per_device, 1)
    return path, block_in_file * int(block_bytes)


class _Staging:
    """Page-aligned host buffers, ``count`` of ``chunk_bytes`` each."""

    def __init__(self, count: int, chunk_bytes: int):
        self.count, self.chunk_bytes = count, chunk_bytes
        self._map = mmap.mmap(-1, count * chunk_bytes)
        self._view = np.frombuffer(self._map, dtype=np.uint8)
        self.base = int(self._view.ctypes.data)

    def ptr(self, i: int) -> int:
        return self.base + i * self.chunk_bytes


class NixlSSDEndpoint:
    """One NIXL agent serving this node's SSD blocks and reading peers' blocks
    into this node's CPU pool (``cpu_ptr``/``cpu_bytes``, ``block_bytes`` per
    block). SSD block ids address ``ssd_files`` exactly like the io_uring
    engine (one device per key, blocks round-robin over its files)."""

    def __init__(self,
                 name: str,
                 *,
                 port: int,
                 cpu_ptr: int,
                 cpu_bytes: int,
                 block_bytes: int,
                 ssd_files: Dict[int, List[str]],
                 ucx_devices: str = "",
                 chunk_blocks: int = 0,
                 num_chunks: int = 4,
                 num_servers: int = 2,
                 posix_params: Optional[Dict[str, str]] = None):
        require_nixl()
        self.name = name
        self.port = int(port)
        self.block_bytes = int(block_bytes)
        self._cpu_ptr, self._cpu_bytes = int(cpu_ptr), int(cpu_bytes)
        self._files = {int(k): list(v) for k, v in ssd_files.items()}
        if chunk_blocks <= 0:  # ~32 MB chunks, at least one block
            chunk_blocks = max(1, (32 << 20) // self.block_bytes)
        self.chunk_blocks = int(chunk_blocks)

        agent = nixlutil.nixl_agent(name, nixlutil.nixl_agent_config(
            True, True, self.port, backends=[]))
        self.agent = agent
        agent.create_backend("UCX", {"ucx_devices": ucx_devices} if ucx_devices else {})
        agent.create_backend("POSIX", dict(posix_params or {}))

        # This node's CPU pool: where peers write the blocks it reads.
        agent.register_memory([(self._cpu_ptr, self._cpu_bytes, 0, "cpu")], "DRAM",
                              backends=["UCX"])
        # This node's SSD files: what it serves, bypassing the page cache
        # whenever FlexKV's own writer does (use_direct_io).
        self.direct_io = use_direct_io(self.block_bytes)
        flags = os.O_RDWR | (os.O_DIRECT if self.direct_io else 0)
        self._fds: Dict[str, int] = {}
        for paths in self._files.values():
            for path in paths:
                if path not in self._fds:
                    self._fds[path] = os.open(path, flags)
        # Blocks each file holds: a block past them must fail, not read short.
        self._file_blocks = {path: os.fstat(fd).st_size // self.block_bytes
                             for path, fd in self._fds.items()}
        if self._fds:
            agent.register_memory([(0, 0, fd, path) for path, fd in self._fds.items()],
                                  "FILE", backends=["POSIX"])
        # Staging: storage reads land here, UCX writes leave from here.
        self._staging = [_Staging(num_chunks, self.chunk_blocks * self.block_bytes)
                         for _ in range(num_servers)]
        for st in self._staging:
            agent.register_memory([(st.base, st.count * st.chunk_bytes, 0, "staging")], "DRAM",
                                  backends=["UCX", "POSIX"])

        self._waiters: Dict[str, List[Any]] = {}  # request id -> [Event, error or None]
        self._lock = threading.Lock()
        self._connected: set = set()
        self._contacts: Dict[str, Tuple[str, int]] = {}
        self._down_until: Dict[str, float] = {}   # peer -> monotonic time of the next reconnect
        self.retry_s = float(os.getenv("FLEXKV_NIXL_SSD_RETRY_S", "") or 10)
        self._requests: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._dispatch, name=f"{name}-notif", daemon=True)]
        self._threads += [threading.Thread(target=self._serve_loop, args=(i,),
                                           name=f"{name}-serve{i}", daemon=True)
                          for i in range(num_servers)]
        for t in self._threads:
            t.start()
        flexkv_logger.info(
            f"[peer-ssd-nixl] endpoint {name} on port {self.port}: CPU pool {self._cpu_bytes >> 20} MB, "
            f"{len(self._fds)} SSD files, block {self.block_bytes} B, chunk {self.chunk_blocks} blocks "
            f"x {num_chunks} per server x {num_servers} servers, ucx_devices={ucx_devices or 'auto'}, "
            f"ssd_io={'direct' if self.direct_io else 'buffered (block not 4 KiB-aligned, like the local writer)'}")

    def close(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass

    # -- addressing -------------------------------------------------------
    def ssd_location(self, block: int) -> Tuple[int, int]:
        """(fd, byte offset) of one of this node's SSD blocks."""
        path, offset = ssd_block_location(self._files, block, self.block_bytes)
        if block < 0 or offset // self.block_bytes >= self._file_blocks[path]:
            raise ValueError(f"SSD block {block} is out of range")
        return self._fds[path], offset

    # -- reader side ------------------------------------------------------
    def connect(self, peer: str, ip: str, port: int, timeout_s: float) -> None:
        with self._lock:
            if peer in self._connected:
                return
            self._contacts[peer] = (ip, int(port))
            wait = self._down_until.get(peer, 0.0) - time.monotonic()
        if wait > 0:
            # Recently dropped: fail at once instead of making every prefetch
            # wait out the metadata timeout of a peer that may still be down.
            raise PeerUnavailable(f"[peer-ssd-nixl] {peer} was dropped; reconnecting in {wait:.0f}s")
        self.agent.send_local_metadata(ip, port)
        self.agent.fetch_remote_metadata(peer, ip, port)
        deadline = time.monotonic() + timeout_s
        while not self.agent.check_remote_metadata(peer):
            if time.monotonic() > deadline:
                self.drop(peer, "no metadata")
                raise TimeoutError(f"[peer-ssd-nixl] no metadata from {peer} at {ip}:{port}")
            time.sleep(0.01)
        with self._lock:
            self._connected.add(peer)
            self._down_until.pop(peer, None)
        flexkv_logger.info(f"[peer-ssd-nixl] {self.name} connected to {peer} ({ip}:{port})")

    def drop(self, peer: str, why: str) -> None:
        """Forget ``peer`` after a timeout or a transport error: NIXL keeps a
        failed UCX endpoint and fails every later send on it, so both sides
        drop each other's metadata and the next `connect` (after `retry_s`)
        builds a fresh connection."""
        with self._lock:
            self._connected.discard(peer)
            self._down_until[peer] = time.monotonic() + self.retry_s
            contact = self._contacts.get(peer)
        for step in (lambda: self.agent.remove_remote_agent(peer),
                     lambda: contact and self.agent.invalidate_local_metadata(*contact)):
            try:
                step()
            except Exception as e:  # noqa: BLE001 - best effort, the peer may be gone
                flexkv_logger.debug(f"[peer-ssd-nixl] dropping {peer}: {e!r}")
        flexkv_logger.warning(f"[peer-ssd-nixl] {self.name} dropped {peer} ({why}); "
                              f"reconnecting after {self.retry_s:.0f}s")

    def read(self, peer: str, src_blocks: Sequence[int], dst_blocks: Sequence[int],
             timeout_s: float, write_window_s: Optional[float] = None) -> None:
        """Have ``peer`` write its SSD blocks ``src_blocks`` into this node's
        CPU blocks ``dst_blocks``. Raises on error or timeout; the destination
        slots then hold undefined bytes.

        The caller recycles the slots once this raises, so the peer must not
        write into them later (a stalled peer would otherwise serve the queued
        request after the timeout and overwrite whatever reused the slots): the
        request carries a wall-clock deadline (``write_window_s``, by default
        the timeout minus a margin) after which the peer posts no more writes.
        The nodes' clocks must agree to well within that margin."""
        if len(src_blocks) != len(dst_blocks):
            raise ValueError("src and dst block lists differ in length")
        if len(src_blocks) == 0:
            return
        if write_window_s is None:
            write_window_s = timeout_s - min(5.0, timeout_s / 3)
        rid = uuid.uuid4().hex[:16]
        waiter = [threading.Event(), None]
        with self._lock:
            self._waiters[rid] = waiter
        body = {"id": rid, "base": self._cpu_ptr, "block": self.block_bytes,
                "expires": time.time() + write_window_s,
                "src": [int(b) for b in src_blocks], "dst": [int(b) for b in dst_blocks]}
        try:
            try:
                self.agent.send_notif(peer, _REQ + json.dumps(body).encode())
            except Exception as e:  # noqa: BLE001
                self.drop(peer, repr(e))
                raise
            if not waiter[0].wait(timeout_s):
                self.drop(peer, f"no answer in {timeout_s:.0f}s")
                raise TimeoutError(
                    f"[peer-ssd-nixl] {peer} did not answer a {len(src_blocks)}-block read "
                    f"in {timeout_s:.0f}s")
        finally:
            with self._lock:
                self._waiters.pop(rid, None)
        if waiter[1] is not None:
            raise RuntimeError(f"[peer-ssd-nixl] {peer} failed the read: {waiter[1]}")

    # -- notification routing ----------------------------------------------
    def _dispatch(self) -> None:
        while not self._stop.is_set():
            try:
                notifs = self.agent.get_new_notifs()
            except Exception as e:  # noqa: BLE001 - keep serving
                flexkv_logger.error(f"[peer-ssd-nixl] get_new_notifs failed: {e}")
                time.sleep(0.01)
                continue
            if not notifs:
                time.sleep(_IDLE_SLEEP_S)
                continue
            for sender, msgs in notifs.items():
                for msg in msgs:
                    kind = bytes(msg[:2])
                    if kind == _REQ:
                        self._requests.put((sender, json.loads(bytes(msg[2:]))))
                    elif kind in (_OK, _ERR):
                        rid, _, err = bytes(msg[2:]).decode(errors="replace").partition(":")
                        with self._lock:
                            waiter = self._waiters.get(rid)
                        if waiter is not None:
                            waiter[1] = err if kind == _ERR else None
                            waiter[0].set()

    # -- server side -------------------------------------------------------
    def _serve_loop(self, index: int) -> None:
        staging = self._staging[index]
        while not self._stop.is_set():
            try:
                sender, req = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                self._serve(sender, req, staging)
                reply = _OK + req["id"].encode()
                flexkv_logger.info(
                    "[FlexKV-IO] operation=peer_ssd_serve act=complete status=ok to=%s blocks=%d "
                    "bytes=%d time_ms=%.1f", sender, len(req["src"]),
                    len(req["src"]) * self.block_bytes, (time.perf_counter() - t0) * 1e3)
            except Exception as e:  # noqa: BLE001 - the reader learns why
                flexkv_logger.error(f"[peer-ssd-nixl] serving {sender} failed: {e!r}")
                reply = _ERR + f"{req.get('id', '')}:{e!r}".encode()
            try:
                self.agent.send_notif(sender, reply)
            except Exception as e:  # noqa: BLE001
                flexkv_logger.error(f"[peer-ssd-nixl] cannot answer {sender}: {e!r}")

    def _post(self, op: str, local: List[Tuple[int, int, int]], remote: List[Tuple[int, int, int]],
              remote_type: str, remote_agent: str) -> Any:
        ldescs = self.agent.get_xfer_descs(local, "DRAM")
        rdescs = self.agent.get_xfer_descs(remote, remote_type)
        handle = self.agent.initialize_xfer(op, ldescs, rdescs, remote_agent)
        state = self.agent.transfer(handle)
        if state == "ERR":
            self.agent.release_xfer_handle(handle)
            raise RuntimeError(f"NIXL {op} to {remote_agent} failed to start")
        return handle

    def _done(self, handle: Any) -> bool:
        state = self.agent.check_xfer_state(handle)
        if state == "ERR":
            self.agent.release_xfer_handle(handle)
            raise RuntimeError("NIXL transfer failed")
        if state == "DONE":
            self.agent.release_xfer_handle(handle)
            return True
        return False

    def _serve(self, sender: str, req: Dict[str, Any], staging: _Staging) -> None:
        """Storage read of chunk k+1 overlaps the network write of chunk k;
        each chunk owns its staging buffer until its write completed."""
        if int(req["block"]) != self.block_bytes:
            raise ValueError(f"block size {req['block']} != this node's {self.block_bytes}")
        src, dst, base = req["src"], req["dst"], int(req["base"])
        expires = float(req.get("expires", float("inf")))
        if time.time() > expires:
            raise TimeoutError("request expired before it was served")
        chunks = [(i, min(i + self.chunk_blocks, len(src))) for i in range(0, len(src), self.chunk_blocks)]
        free = deque(range(staging.count))
        reads: deque = deque()   # (chunk, buffer, handle)
        writes: deque = deque()
        nxt = done = 0
        bb = self.block_bytes
        while done < len(chunks):
            while free and nxt < len(chunks):
                buf = free.popleft()
                lo, hi = chunks[nxt]
                local = [(staging.ptr(buf) + j * bb, bb, 0) for j in range(hi - lo)]
                files = [(off, bb, fd) for fd, off in (self.ssd_location(b) for b in src[lo:hi])]
                reads.append((nxt, buf, self._post("READ", local, files, "FILE", self.name)))
                nxt += 1
            progressed = False
            if reads and self._done(reads[0][2]):
                if time.time() > expires:
                    # The reader has given up and may have reused the slots.
                    for _, _, h in list(reads)[1:] + list(writes):  # reads[0] is released
                        while not self._done(h):
                            time.sleep(_IDLE_SLEEP_S)
                    raise TimeoutError("request expired before it was served")
                c, buf, _ = reads.popleft()
                lo, hi = chunks[c]
                local = [(staging.ptr(buf) + j * bb, bb, 0) for j in range(hi - lo)]
                remote = [(base + int(d) * bb, bb, 0) for d in dst[lo:hi]]
                writes.append((c, buf, self._post("WRITE", local, remote, "DRAM", sender)))
                progressed = True
            if writes and self._done(writes[0][2]):
                _, buf, _ = writes.popleft()
                free.append(buf)
                done += 1
                progressed = True
            if not progressed:
                time.sleep(_IDLE_SLEEP_S)


class NixlRemoteSSDReader:
    """`RemoteSSDReader` of a PEERSSD2H worker over :class:`NixlSSDEndpoint`.
    Whole blocks only (PEERSSD2H ops are never layer-sliced).

    The endpoint is built in a background thread: registering the CPU pool
    with UCX pins it for CUDA too, which takes minutes while the TE's GPU
    workers pin the same pool and the model loads, and FlexKV must not wait
    for that. Until it is up this node neither serves nor reads (a read waits
    up to the timeout, then fails and the prefetch misses); its contact file
    appears only once it serves."""

    def __init__(self, worker: Any, pool_id: PoolId):
        from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
        import radixshmem

        self._worker = worker
        self._pool = "swa" if pool_id is PoolId.SWA else "full"
        self._timeout = float(os.getenv("FLEXKV_NIXL_SSD_TIMEOUT_S", "") or 30)
        self._dir = GLOBAL_CONFIG_FROM_ENV.peer_ssd_manifest_dir
        if not self._dir:
            raise RuntimeError("the NIXL SSD reader needs FLEXKV_PEER_SSD_MANIFEST_DIR to find peers")
        self._client = radixshmem.RadixClient(worker.ssd_server_name)
        self.node = self._client.info.node_name or f"rank{self._client.info.rank}"
        self._node_names: Dict[int, str] = {}
        self.endpoint: Optional[NixlSSDEndpoint] = None
        self._ready = threading.Event()
        self._not_ready_logged = -_NOT_READY_LOG_EVERY_S
        self._error: Optional[BaseException] = None
        threading.Thread(target=self._start, name=f"nixl-ssd-{self._pool}-init", daemon=True).start()

    def _start(self) -> None:
        cpu = self._worker.cpu_blocks
        port = _env_int("FLEXKV_NIXL_SSD_PORT", 5600) + (1 if self._pool == "swa" else 0)
        t0 = time.monotonic()
        try:
            # One agent name per process: a peer still holding a restarted node's
            # old metadata would otherwise serve it over the dead connection.
            self.endpoint = NixlSSDEndpoint(
                f"flexkv-ssd-{self.node}-{self._pool}@{os.getpid()}.{time.time_ns()}",
                port=port,
                cpu_ptr=int(cpu.data_ptr()),
                cpu_bytes=int(cpu.numel() * cpu.element_size()),
                block_bytes=int(self._worker.block_stride_in_bytes),
                ssd_files=self._worker.ssd_files,
                ucx_devices=os.getenv("FLEXKV_NIXL_UCX_DEVICES", ""),
                posix_params={"use_uring": os.getenv("FLEXKV_NIXL_POSIX_URING", "true")},
            )
            os.makedirs(self._dir, exist_ok=True)
            path = os.path.join(self._dir, f"{self.node}.nixl-{self._pool}.json")
            with open(f"{path}.tmp", "w") as f:
                json.dump({"agent": self.endpoint.name, "ip": _default_ip(), "port": port}, f)
            os.replace(f"{path}.tmp", path)
            flexkv_logger.info(f"[peer-ssd-nixl] {self.endpoint.name} serving after "
                               f"{time.monotonic() - t0:.1f}s; contact {path}")
        except BaseException as e:  # noqa: BLE001 - reported by every read
            self._error = e
            flexkv_logger.error(f"[peer-ssd-nixl] endpoint start failed: {e!r}")
        finally:
            self._ready.set()

    def _peer(self, rank: int) -> Tuple[str, str, int]:
        if rank not in self._node_names:
            for rec in self._client.nodes():
                if int(rec.get("rank", -1)) >= 0:
                    self._node_names[int(rec["rank"])] = str(rec.get("node_name", ""))
        node = self._node_names.get(rank)
        if not node:
            raise RuntimeError(f"[peer-ssd-nixl] no SSD index node has rank {rank}")
        path = os.path.join(self._dir, f"{node}.nixl-{self._pool}.json")
        try:
            with open(path) as f:
                c = json.load(f)
        except FileNotFoundError:
            raise PeerUnavailable(f"[peer-ssd-nixl] {node} does not serve its SSD yet ({path})") from None
        return c["agent"], c["ip"], int(c["port"])

    def read(self, peer_rank: int, pool_id: PoolId, src_ssd_blocks: torch.Tensor,
             dst_cpu_blocks: torch.Tensor, layer_id: int, layer_end: int) -> None:
        if layer_id != 0 or layer_end < self._worker.num_layers:
            raise ValueError("the NIXL SSD reader moves whole blocks only")
        if not self._ready.is_set():
            # Fail at once: a prefetch waiting here would only delay its miss.
            now = time.monotonic()
            if now - self._not_ready_logged >= _NOT_READY_LOG_EVERY_S:
                self._not_ready_logged = now
                flexkv_logger.warning(
                    "[peer-ssd-nixl] this node's NIXL endpoint is still starting; "
                    "peer SSD reads fail (prefetch misses) until it serves")
            raise PeerUnavailable("[peer-ssd-nixl] this node's NIXL endpoint is still starting")
        if self._error is not None:
            raise RuntimeError(f"[peer-ssd-nixl] endpoint unavailable: {self._error!r}")
        agent, ip, port = self._peer(int(peer_rank))
        self.endpoint.connect(agent, ip, port, self._timeout)
        self.endpoint.read(agent, src_ssd_blocks.tolist(), dst_cpu_blocks.tolist(), self._timeout)


def make_reader(worker: Any, pool_id: PoolId) -> NixlRemoteSSDReader:
    """``FLEXKV_REMOTE_SSD_READER=flexkv.transfer.workers.peer_ssd_nixl:make_reader``."""
    return NixlRemoteSSDReader(worker, pool_id)
