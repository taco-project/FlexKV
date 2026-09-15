"""Worker-side synchronization primitives for vLLM layer-wise KV loading.

The FlexKV data plane signals completion of each layer through Linux eventfds.
This module intentionally has no torch/vLLM/c_ext dependency so its counter
state machine and Unix-domain-socket handshake can be tested on CPU-only hosts.
"""

from __future__ import annotations

import ctypes
import errno
import os
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence


_EFD_SEMAPHORE = 0x1


def _linux_eventfd(initval: int = 0, flags: int = _EFD_SEMAPHORE) -> int:
    eventfd_fn = getattr(os, "eventfd", None)
    if eventfd_fn is not None:
        return int(eventfd_fn(initval, flags))

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.eventfd.argtypes = [ctypes.c_uint, ctypes.c_int]
    libc.eventfd.restype = ctypes.c_int
    fd = libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(fd)


def _settle_eventfd(fd: int, timeout_s: float) -> bool:
    """Consume one pending eventfd unit, waiting at most timeout_s."""
    try:
        ready, _, _ = select.select([fd], [], [], timeout_s)
        if not ready:
            return False
        os.read(fd, 8)
        return True
    except OSError:
        return True


def _read_eventfd(fd: int) -> int:
    timeout_s = float(os.getenv("FLEXKV_LAYERWISE_WAIT_TIMEOUT_S", "60"))
    ready, _, _ = select.select([fd], [], [], timeout_s)
    if not ready:
        raise TimeoutError(
            f"timed out after {timeout_s}s waiting for layer-wise KV load")
    payload = os.read(fd, 8)
    if len(payload) != 8:
        raise OSError(errno.EIO, f"short eventfd read: {len(payload)} bytes")
    return int(struct.unpack("Q", payload)[0])


def _send_fds(sock: socket.socket, fds: Sequence[int], counter_id: int) -> None:
    packed_fds = struct.pack(f"{len(fds)}i", *fds)
    sock.sendmsg(
        [struct.pack("i", counter_id)],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, packed_fds)],
    )


@dataclass(frozen=True)
class LayerwiseLoadMetadata:
    """Serializable scheduler-to-worker metadata for one model step."""

    enabled: bool = False
    counter_id: int = -1
    has_load: bool = False


class LayerwiseStepCoordinator:
    """Scheduler-side deterministic counter assignment."""

    def __init__(self, enabled: bool, num_counters: int = 3) -> None:
        if num_counters <= 0:
            raise ValueError("num_counters must be positive")
        self.enabled = enabled
        self.num_counters = num_counters
        self._next_counter = 0

    def build_metadata(self, has_load: bool) -> LayerwiseLoadMetadata:
        if not self.enabled or not has_load:
            return LayerwiseLoadMetadata(enabled=self.enabled)
        counter_id = self._next_counter
        self._next_counter = (self._next_counter + 1) % self.num_counters
        return LayerwiseLoadMetadata(
            enabled=True,
            counter_id=counter_id,
            has_load=True,
        )

    def external_match_is_async(self, needs_load: bool) -> bool:
        """Whether vLLM should enter WAITING_FOR_REMOTE_KVS.

        Layer-wise loads must run in the same model step so attention-layer
        hooks can wait on their eventfds. Non-layer-wise loads retain the
        existing whole-transfer async behavior.
        """
        return bool(needs_load and not self.enabled)

    def launch_kwargs(self, metadata: LayerwiseLoadMetadata) -> dict[str, object]:
        # A no-load step carries counter_id=-1. Clamping that to 0 would make
        # the data plane signal counter 0 -- which a concurrent real load may
        # own -- so its layers could see units that belong to no transfer.
        # Only a metadata that actually has a load may drive a layer-wise
        # launch; otherwise fall back to the non-layer-wise path.
        if not metadata.enabled or not metadata.has_load:
            return {
                "as_batch": metadata.enabled,
                "layerwise_transfer": False,
                "counter_id": 0,
            }
        self._validate_launch_counter(metadata.counter_id)
        return {
            "as_batch": metadata.enabled,
            "layerwise_transfer": metadata.enabled,
            "counter_id": metadata.counter_id,
        }

    def _validate_launch_counter(self, counter_id: int) -> None:
        if counter_id < 0 or counter_id >= self.num_counters:
            raise ValueError(
                f"layer-wise launch counter_id={counter_id} outside "
                f"[0, {self.num_counters})"
            )


class LayerwiseCounterPool:
    """Triple-buffered per-layer eventfd synchronization.

    The scheduler chooses a counter id for each layer-wise batch and sends it
    through connector metadata. The worker binds that id at forward start;
    each attention layer then consumes exactly one semaphore unit from its
    eventfd. A counter can only be reused after its final layer was consumed.
    """

    def __init__(
        self,
        layer_names: Sequence[str],
        num_counters: int = 3,
        fd_factory: Callable[[], int] = _linux_eventfd,
        fd_reader: Callable[[int], int] = _read_eventfd,
        fd_closer: Callable[[int], None] = os.close,
        fd_settler: Callable[[int, float], bool] = _settle_eventfd,
    ) -> None:
        if not layer_names:
            raise ValueError("layer_names must not be empty")
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("layer_names must be unique")
        if num_counters <= 0:
            raise ValueError("num_counters must be positive")

        self.layer_names = tuple(layer_names)
        self.layer_to_index = {
            layer_name: index for index, layer_name in enumerate(self.layer_names)
        }
        self.num_layers = len(self.layer_names)
        self.num_counters = num_counters
        self._fd_reader = fd_reader
        self._fd_closer = fd_closer
        # Retained for compatibility with external test fixtures that inject a
        # settler. Reusing an incomplete asynchronous counter is unsafe, so
        # this pool deliberately no longer invokes it.
        self._fd_settler = fd_settler
        self._fds = [
            [fd_factory() for _ in range(self.num_layers)]
            for _ in range(num_counters)
        ]
        self._waited = [[False] * self.num_layers for _ in range(num_counters)]
        self._active_counter = -1
        self._lock = threading.Lock()
        self._closed = False
        self._failed_error: BaseException | None = None

    @property
    def fds(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(counter_fds) for counter_fds in self._fds)

    def release(self, counter_id: int) -> None:
        self._validate_counter(counter_id)
        with self._lock:
            self._waited[counter_id] = [False] * self.num_layers
            if self._active_counter == counter_id:
                self._active_counter = -1

    def bind(self, metadata: LayerwiseLoadMetadata) -> None:
        """Bind the counter used by the current vLLM forward step."""
        if self._failed_error is not None:
            raise RuntimeError(
                "layer-wise counter pool is unusable after a prior wait failure"
            ) from self._failed_error
        if self._active_counter >= 0:
            # Eventfd units carry no generation. The data plane writes them
            # asynchronously, so an incomplete forward cannot be safely
            # reclaimed without an explicit producer-quiescence acknowledgement
            # or per-generation descriptors.
            raise RuntimeError(
                "received new layer-wise metadata before the previous "
                f"counter {self._active_counter} finished"
            )
        if not metadata.enabled or not metadata.has_load:
            self._active_counter = -1
            return
        self._validate_counter(metadata.counter_id)
        with self._lock:
            self._waited[metadata.counter_id] = [False] * self.num_layers
            self._active_counter = metadata.counter_id

    def wait(self, layer_name: str) -> None:
        counter_id = self._active_counter
        if counter_id < 0:
            return
        try:
            layer_index = self.layer_to_index[layer_name]
        except KeyError as exc:
            raise KeyError(
                f"unknown layer_name={layer_name!r}; registered layers="
                f"{self.layer_names!r}"
            ) from exc

        with self._lock:
            if self._waited[counter_id][layer_index]:
                return
        try:
            self._fd_reader(self._fds[counter_id][layer_index])
        except BaseException as exc:
            self._failed_error = exc
            self._active_counter = -1
            raise
        with self._lock:
            self._waited[counter_id][layer_index] = True
            finished = all(self._waited[counter_id])
        if finished:
            self.release(counter_id)

    def send_to_worker(
        self,
        socket_path: str,
        tp_rank_per_node: int,
        tp_size_per_node: int,
        timeout_s: float = 360.0,
        retry_interval_s: float = 0.05,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Send all counter eventfds to the LayerwiseTransferWorker.

        This method is intended to run in a background thread before GPU-cache
        registration, because registration may start a worker that blocks while
        waiting for this handshake.

        ``cancel_event`` lets a caller abandon the retry loop early. Shutdown
        needs it: otherwise this thread keeps retrying for the full timeout_s
        while holding eventfds that the caller wants to close.
        """
        deadline = time.monotonic() + timeout_s
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                return
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(socket_path)
                    sock.sendall(
                        struct.pack(
                            "iiii",
                            tp_rank_per_node,
                            tp_size_per_node,
                            self.num_layers,
                            self.num_counters,
                        )
                    )
                    for counter_id, fds in enumerate(self._fds):
                        _send_fds(sock, fds, counter_id)
                    sock.settimeout(min(30.0, max(1.0, timeout_s)))
                    ack = sock.recv(1)
                    if ack != b"\x01":
                        raise RuntimeError(
                            f"LayerwiseTransferWorker rejected eventfds: {ack!r}"
                        )
                    return
            except (OSError, RuntimeError, TimeoutError) as exc:
                last_error = exc
                if cancel_event is not None:
                    # Interruptible sleep: a plain time.sleep() would ignore a
                    # cancel that arrives during the backoff.
                    if cancel_event.wait(retry_interval_s):
                        return
                else:
                    time.sleep(retry_interval_s)
        raise RuntimeError(
            f"timed out sending layer-wise eventfds to {socket_path}: {last_error}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for counter_fds in self._fds:
            for fd in counter_fds:
                self._fd_closer(fd)
        self._fds.clear()

    def _validate_counter(self, counter_id: int) -> None:
        if counter_id < 0 or counter_id >= self.num_counters:
            raise ValueError(
                f"counter_id={counter_id} outside [0, {self.num_counters})"
            )

    def __del__(self) -> None:
        self.close()
