from __future__ import annotations

import ctypes
import os
import socket
import struct
import threading
import time


_EFD_NONBLOCK = 0x800


def _create_eventfd() -> int:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.eventfd(ctypes.c_uint(0), ctypes.c_int(_EFD_NONBLOCK))
    if fd == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(fd)


def _send_fds(sock: socket.socket, fds: list[int], counter_id: int) -> None:
    packed = struct.pack(f"{len(fds)}i", *fds)
    sock.sendmsg(
        [struct.pack("i", counter_id)],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, packed)],
    )


class EventfdGroup:
    """Minimal SGLang-compatible layer completion counter client."""

    def __init__(self, socket_path: str, tp_size: int, num_layers: int, num_counters: int = 3):
        self.socket_path = socket_path
        self.tp_size = tp_size
        self.num_layers = num_layers
        self.num_counters = num_counters
        self.fds = [
            [[_create_eventfd() for _ in range(num_layers)] for _ in range(tp_size)]
            for _ in range(num_counters)
        ]
        self.threads: list[threading.Thread] = []
        self.errors: list[str] = []

    def start(self) -> None:
        for rank in range(self.tp_size):
            thread = threading.Thread(target=self._connect, args=(rank,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _connect(self, rank: int) -> None:
        try:
            client = None
            for _ in range(240):
                candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    candidate.connect(self.socket_path)
                    client = candidate
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    candidate.close()
                    time.sleep(0.25)
            if client is None:
                raise TimeoutError(f"eventfd socket did not appear: {self.socket_path}")
            with client:
                client.sendall(struct.pack("iiii", rank, self.tp_size, self.num_layers, self.num_counters))
                for counter_id in range(self.num_counters):
                    _send_fds(client, self.fds[counter_id][rank], counter_id)
                client.settimeout(30)
                if client.recv(1) != b"\x01":
                    raise RuntimeError(f"eventfd registration rejected for rank {rank}")
        except Exception as exc:
            self.errors.append(str(exc))

    def wait_ready(self, timeout: float = 65) -> None:
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            thread.join(max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self.threads):
            raise TimeoutError("eventfd registration timed out")
        if self.errors:
            raise RuntimeError("; ".join(self.errors))

    def read_counter(self, counter_id: int) -> list[list[int]]:
        values = []
        for rank_fds in self.fds[counter_id]:
            rank_values = []
            for fd in rank_fds:
                try:
                    rank_values.append(struct.unpack("Q", os.read(fd, 8))[0])
                except BlockingIOError:
                    rank_values.append(0)
            values.append(rank_values)
        return values

    def close(self) -> None:
        for counter in self.fds:
            for rank in counter:
                for fd in rank:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
