"""Communication helpers for the FlexKV connector.

FlexKV runs a single KVManager per DP group (typically the TP/CP/PP
sync leader's process). Every other rank in the same KV-cache-sharing
fan-out must be told the leader's decisions: which prefix matched in
FlexKV, which task id the leader allocated, which slot mappings to
send, etc.

This file provides:

* ``FlexKVComm`` — a 3-axis (PP × CP × TP) hierarchical sync context
  built on torch.distributed (gloo CPU groups). Exposes ``scatter``,
  ``scatter_pp``, ``barrier`` and ``all_reduce_min`` plus role flags
  (``is_sync_leader`` etc.) that the connector branches on.
* libc / ``eventfd`` shims used by the layerwise transfer worker
  socket handshake.
* ``FlexKVLayerLoadingEvent`` and ``FlexKVLayerDoneCounter`` — the
  eventfd-backed per-layer completion structures that the FlexKV
  layerwise transfer worker signals into. Hooked into sglang's
  ``register_layer_transfer_counter`` so each layer's forward waits
  for its own host→device copy.
"""

from __future__ import annotations

import collections
import ctypes
import errno
import io
import logging
import os
import pickle
import socket
import struct
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

CMD_LAYERWISE = 3


class FlexKVScatterChannel(str, Enum):
    """Logical channels multiplexed over the hierarchical P2P fan-out."""

    LOOKUP = "lookup"
    LAYERWISE_MANIFEST = "layerwise_manifest"
    STORE_COMPLETION = "store_completion"
    PREFETCH_START = "prefetch_start"
    PREFETCH_PROGRESS = "prefetch_progress"
    RESET = "reset"
    STORE_START = "store_start"
    STORE_RESET = "store_reset"
    STORE_READY = "store_ready"


_SCATTER_CHANNEL_OFFSETS = {
    FlexKVScatterChannel.LOOKUP: 1,
    FlexKVScatterChannel.LAYERWISE_MANIFEST: 2,
    FlexKVScatterChannel.STORE_COMPLETION: 3,
    FlexKVScatterChannel.PREFETCH_START: 4,
    FlexKVScatterChannel.PREFETCH_PROGRESS: 5,
    FlexKVScatterChannel.RESET: 6,
    FlexKVScatterChannel.STORE_START: 7,
    FlexKVScatterChannel.STORE_RESET: 8,
    FlexKVScatterChannel.STORE_READY: 9,
}
_SCATTER_CHANNEL_TYPES = {
    FlexKVScatterChannel.LOOKUP: dict,
    FlexKVScatterChannel.LAYERWISE_MANIFEST: dict,
    FlexKVScatterChannel.STORE_COMPLETION: list,
    FlexKVScatterChannel.PREFETCH_START: dict,
    FlexKVScatterChannel.PREFETCH_PROGRESS: dict,
    FlexKVScatterChannel.RESET: dict,
    FlexKVScatterChannel.STORE_START: dict,
    FlexKVScatterChannel.STORE_RESET: dict,
    FlexKVScatterChannel.STORE_READY: list,
}

_CONTROL_FRAME_SIZE = 4096
_CONTROL_HEADER = struct.Struct("<q")
_CONTROL_INLINE_SIZE = _CONTROL_FRAME_SIZE - _CONTROL_HEADER.size


def _control_send_timeout_s() -> Optional[float]:
    raw = os.environ.get("FLEXKV_CONTROL_SEND_TIMEOUT_S", "1200")
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "[FlexKV] Invalid FLEXKV_CONTROL_SEND_TIMEOUT_S=%r; using 1200s", raw
        )
        value = 1200.0
    return value if value > 0 else None


class _AsyncSendReaper:
    """Retire gloo ``isend`` works on a daemon thread.

    A gloo ``SendWork`` only becomes complete after the peer posts the
    matching receive, and ``Work.is_completed()`` does not advance until
    ``wait()`` is called.  ``wait(timeout)`` is not a probe: when it expires,
    gloo signals every pair in the context and closes the whole process group.
    This thread therefore uses an untimed ``wait()``, which releases the GIL
    and returns as soon as the peer receives.  Each queued entry keeps its
    frame tensor alive until the send has drained.

    Hang detection is done on the caller side instead: when the oldest send
    has waited longer than ``timeout_s``, the next ``submit`` raises.  Waiting
    for backpressure is bounded by the same deadline.
    """

    def __init__(
        self, world_rank: int, max_pending: int, timeout_s: Optional[float]
    ):
        self._world_rank = world_rank
        self._max_pending = max_pending
        self._timeout_s = timeout_s
        # Entries: (work, frame, dst, tag, enqueued_monotonic).
        self._pending: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._backpressure_logged = False

    def submit(self, works: List, tag: int) -> None:
        if not works:
            return
        with self._cv:
            self._raise_if_failed_locked()
            self._check_oldest_locked()
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name=f"flexkv-send-reaper-{self._world_rank}",
                    daemon=True,
                )
                self._thread.start()
            if len(self._pending) >= self._max_pending:
                if not self._backpressure_logged:
                    self._backpressure_logged = True
                    logger.warning(
                        "[FlexKV] rank=%d has %d undrained control sends; "
                        "waiting for slow followers to receive",
                        self._world_rank,
                        len(self._pending),
                    )
                while (
                    self._error is None
                    and len(self._pending) >= self._max_pending
                ):
                    self._cv.wait(self._remaining_locked())
                    self._check_oldest_locked()
                self._raise_if_failed_locked()
            now = time.monotonic()
            self._pending.extend(
                (work, frame, dst, tag, now) for work, frame, dst in works
            )
            self._cv.notify_all()

    def pending(self) -> int:
        with self._cv:
            return len(self._pending)

    def _remaining_locked(self) -> Optional[float]:
        if self._timeout_s is None or not self._pending:
            return None
        age = time.monotonic() - self._pending[0][4]
        return max(0.0, self._timeout_s - age)

    def _check_oldest_locked(self) -> None:
        if self._error is not None:
            self._raise_if_failed_locked()
        if self._timeout_s is None or not self._pending:
            return
        _work, _frame, dst, tag, enqueued = self._pending[0]
        age = time.monotonic() - enqueued
        if age <= self._timeout_s:
            return
        self._error = TimeoutError(
            f"[FlexKV] control send from rank={self._world_rank} to "
            f"rank={dst} (tag={tag}) not received for {age:.1f}s "
            f"(> FLEXKV_CONTROL_SEND_TIMEOUT_S={self._timeout_s:g}); "
            f"{len(self._pending)} sends pending. The receiver is hung or "
            f"the scatter sequence diverged across ranks."
        )
        logger.error("%s", self._error)
        self._cv.notify_all()
        self._raise_if_failed_locked()

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"[FlexKV] control send failed on rank={self._world_rank}: "
                f"{self._error}"
            ) from self._error

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: bool(self._pending))
                work, _frame, dst, tag, _enqueued = self._pending[0]
            try:
                work.wait()
            except BaseException as exc:  # noqa: BLE001
                logger.error(
                    "[FlexKV] control send from rank=%d to rank=%d (tag=%d) "
                    "failed: %s",
                    self._world_rank,
                    dst,
                    tag,
                    exc,
                )
                with self._cv:
                    if self._error is None:
                        self._error = exc
                    self._cv.notify_all()
                return
            with self._cv:
                self._pending.popleft()
                if len(self._pending) < self._max_pending:
                    self._backpressure_logged = False
                self._cv.notify_all()


class FlexKVComm:
    """3-axis (PP × CP × TP) hierarchical sync for the FlexKV connector.

    Notation:
      * "sync leader" is the unique rank that talks to the FlexKV
        KVManager: pp_rank=0, attn_cp_rank=0, attn_tp_rank=0.
      * "PP stage leader" is the (cp=0, tp=0) rank within a PP stage —
        it does cross-PP P2P (``scatter_pp``).
      * Every rank participates in collective layers it belongs to.

    Communication strategy:
      * P2P (send/recv/isend/irecv) on CPU tensors → ``world_cpu_group``
        (the global gloo group). Sub-group cpu_groups have unreliable
        TCP pairs for direct P2P.
      * Collectives (all_reduce / barrier) → sglang's sub-group
        cpu_groups (fine for collectives).
    """

    # P2P tags. World group is shared with sglang's own P2P, so we pick
    # 4-byte tags that won't collide.
    _TAG_SCATTER = int.from_bytes(b"FxSc", byteorder="big")
    _TAG_PP = int.from_bytes(b"FxPP", byteorder="big")
    _TAG_CP = int.from_bytes(b"FxCP", byteorder="big")
    _TAG_TP = int.from_bytes(b"FxTP", byteorder="big")
    _TAG_PP_AR_MIN = int.from_bytes(b"FxA2", byteorder="big")
    _TAG_PP_BARRIER = int.from_bytes(b"FxB2", byteorder="big")
    _TAG_PP_BARRIER_BCAST = int.from_bytes(b"FxB3", byteorder="big")
    _TAG_AR_BCAST = int.from_bytes(b"FxAR", byteorder="big")

    # Upper bound on undrained non-blocking sends. Beyond it the sender waits
    # for followers to catch up instead of growing without bound.
    _MAX_PENDING_SENDS = 32768

    def __init__(
        self,
        rank_info,
        world_rank: int,
        pp_group=None,
        attn_tp_group=None,
        attn_cp_group=None,
        world_group=None,
    ):
        model_config = rank_info.model_config
        self.world_rank = world_rank
        self._world_group = world_group
        self._send_reaper = _AsyncSendReaper(
            world_rank, self._MAX_PENDING_SENDS, _control_send_timeout_s()
        )

        # Accept either GroupCoordinator wrappers (has ``.cpu_group``) or
        # raw ProcessGroups.
        self.pp_cpu_group = (
            getattr(pp_group, "cpu_group", pp_group) if pp_group is not None else None
        )
        self.attn_tp_cpu_group = (
            getattr(attn_tp_group, "cpu_group", attn_tp_group)
            if attn_tp_group is not None
            else None
        )
        self.attn_cp_cpu_group = (
            getattr(attn_cp_group, "cpu_group", attn_cp_group)
            if attn_cp_group is not None
            else None
        )

        self.pp_size = model_config.pp_size
        self.attn_tp_size = model_config.attn_tp_size
        self.attn_cp_size = model_config.attn_cp_size

        self.pp_rank = rank_info.pp_rank
        self.attn_tp_rank = rank_info.attn_tp_rank
        self.attn_cp_rank = rank_info.attn_cp_rank

        self.is_pp_stage_leader = self.attn_tp_rank == 0 and self.attn_cp_rank == 0
        self.is_sync_leader = self.pp_rank == 0 and self.is_pp_stage_leader
        self.is_pp_leader = self.pp_rank == 0 and self.is_pp_stage_leader
        self.is_cp_leader = self.attn_cp_rank == 0
        self.is_tp_leader = self.attn_tp_rank == 0

        # P2P routing tables (computed once).
        stride = self.attn_tp_size * self.attn_cp_size
        self._pp_stage_leader_ranks = [s * stride for s in range(self.pp_size)]
        pp_stage_offset = self.pp_rank * stride
        self._cp_leader_ranks = (
            [
                pp_stage_offset + cp * self.attn_tp_size
                for cp in range(self.attn_cp_size)
            ]
            if self.attn_cp_size > 1
            else []
        )
        if self.attn_tp_size > 1:
            if self.attn_tp_cpu_group is None:
                raise RuntimeError(
                    f"[FlexKV] attn_tp_size={self.attn_tp_size} > 1 but "
                    f"attn_tp_cpu_group is None — TP CPU group is required "
                    f"for scatter/collectives."
                )
            self._tp_group_ranks = [
                dist.get_global_rank(self.attn_tp_cpu_group, i)
                for i in range(self.attn_tp_cpu_group.size())
            ]
        else:
            self._tp_group_ranks = []
        self._pp_group_global_ranks = (
            [
                dist.get_global_rank(self.pp_cpu_group, i)
                for i in range(self.pp_cpu_group.size())
            ]
            if self.pp_size > 1 and self.pp_cpu_group is not None
            else []
        )
        self._pp_stage_member_ranks = list(
            range(pp_stage_offset, pp_stage_offset + stride)
        )

        self.needs_sync = (
            self.pp_size > 1 or self.attn_tp_size > 1 or self.attn_cp_size > 1
        )

        self._world_cpu_group = self._resolve_world_cpu_group()

        self.pp_group = (
            self.pp_cpu_group
            if (self.pp_size > 1 and self.is_pp_stage_leader)
            else None
        )
        self.is_pp_active = self.pp_size > 1
        self.is_pp_sender = self.is_pp_leader
        self.is_pp_receiver = self.is_pp_stage_leader and not self.is_pp_leader

        self.is_cross_node_pp = self.pp_size > rank_info.pp_size_per_node
        self.should_send_slot_mapping_to_remote = (
            self.is_pp_receiver and self.is_cross_node_pp
        )

        logger.info(
            "[FlexKV] Comm init: rank=%d, pp=%d/%d, tp=%d/%d, cp=%d/%d, "
            "sync_leader=%s, stage_leader=%s, cross_node_pp=%s",
            world_rank,
            self.pp_rank,
            self.pp_size,
            self.attn_tp_rank,
            self.attn_tp_size,
            self.attn_cp_rank,
            self.attn_cp_size,
            self.is_sync_leader,
            self.is_pp_stage_leader,
            self.is_cross_node_pp,
        )

    # ------------------------------------------------------------------
    # World CPU group resolution
    # ------------------------------------------------------------------

    def _resolve_world_cpu_group(self):
        """Return the world CPU group, preferring the injected ``world_group``.

        When ``world_group`` is not injected by the caller (legacy sglang
        path), fall back to sglang's ``get_world_group()`` via a delayed
        import so this module does not require sglang at import time.
        """
        world_group = self._world_group
        if world_group is None:
            from sglang.srt.distributed.parallel_state import get_world_group

            world_group = get_world_group()
        return getattr(world_group, "cpu_group", world_group)

    # ------------------------------------------------------------------
    # Public collectives
    # ------------------------------------------------------------------

    def scatter(
        self,
        data: Any,
        *,
        channel: FlexKVScatterChannel,
        blocking: bool = False,
    ) -> Any:
        """Hierarchical fan-out: sync_leader → PP stage leaders →
        CP leaders → TP ranks. Returns the leader's payload on every rank.

        Logical operations use distinct P2P tags. Scheduler ticks are not a
        collective clock: for example, one CP rank can poll store completion
        before another rank enters prefix lookup. Sharing one tag for both
        operations lets the lookup receive a pending completion list instead
        of its task-id dictionary.

        ``blocking=False`` queues isends and reaps later — fine for the
        hot path; ``True`` blocks until the leader's sends drain (used
        on shutdown / barriers).
        """
        channel = self._normalize_scatter_channel(channel)
        self._validate_scatter_payload(data, channel, source="local")
        if self.pp_size > 1 and self.is_pp_stage_leader:
            data = self._scatter_group(
                data,
                self._pp_stage_leader_ranks,
                self.is_pp_leader,
                self._scatter_tag(self._TAG_PP, channel),
                blocking,
            )
        if self._cp_leader_ranks:
            data = self._scatter_group(
                data,
                self._cp_leader_ranks,
                self.is_cp_leader,
                self._scatter_tag(self._TAG_CP, channel),
                blocking,
            )
        if self._tp_group_ranks:
            data = self._scatter_group(
                data,
                self._tp_group_ranks,
                self.is_tp_leader,
                self._scatter_tag(self._TAG_TP, channel),
                blocking,
            )
        self._validate_scatter_payload(data, channel, source="received")
        return data

    @staticmethod
    def _normalize_scatter_channel(channel: Any) -> FlexKVScatterChannel:
        try:
            return FlexKVScatterChannel(channel)
        except (TypeError, ValueError) as exc:
            valid = ", ".join(item.value for item in FlexKVScatterChannel)
            raise ValueError(
                f"Unknown FlexKV scatter channel {channel!r}; expected one of {valid}"
            ) from exc

    @staticmethod
    def _scatter_tag(base_tag: int, channel: FlexKVScatterChannel) -> int:
        return base_tag + _SCATTER_CHANNEL_OFFSETS[channel]

    def _validate_scatter_payload(
        self,
        data: Any,
        channel: FlexKVScatterChannel,
        *,
        source: str,
    ) -> None:
        expected_type = _SCATTER_CHANNEL_TYPES[channel]
        if isinstance(data, expected_type):
            return
        raise TypeError(
            "[FlexKV] scatter protocol mismatch: "
            f"rank={self.world_rank} channel={channel.value} source={source} "
            f"expected={expected_type.__name__} got={type(data).__name__}"
        )

    def scatter_pp(self, data: Any) -> Any:
        """PP-only fan-out across PP stages (only stage leaders participate)."""
        if not self._pp_group_global_ranks:
            return data
        is_leader = self._pp_group_global_ranks[0] == self.world_rank
        return self._scatter_group(
            data,
            self._pp_group_global_ranks,
            is_leader,
            self._TAG_SCATTER,
            blocking=False,
        )

    def all_reduce_min(self, value: int) -> int:
        """Hierarchical all_reduce(MIN) across TP, CP, PP.

        Used to align FlexKV block-count limits across all ranks that
        will register GPU buffers (each rank computes the maximum it can
        support, and we take the MIN to land on a value everyone can
        honor).
        """
        tensor = torch.tensor(value, dtype=torch.int64)
        if self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.attn_tp_cpu_group)
        if self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self.attn_cp_cpu_group)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            self._pp_all_reduce_min_p2p(tensor)
        if self.pp_size > 1:
            self._bcast_to_stage_members(tensor, self._TAG_AR_BCAST)
        return int(tensor.item())

    def barrier(self) -> None:
        if self.attn_tp_size > 1 and self.attn_tp_cpu_group is not None:
            dist.barrier(group=self.attn_tp_cpu_group)
        if self.attn_cp_size > 1 and self.attn_cp_cpu_group is not None:
            dist.barrier(group=self.attn_cp_cpu_group)
        if self.pp_size > 1 and self.is_pp_stage_leader:
            self._pp_barrier_p2p()
        if self.pp_size > 1:
            dummy = torch.tensor([0], dtype=torch.int64)
            self._bcast_to_stage_members(dummy, self._TAG_PP_BARRIER_BCAST)

    # ------------------------------------------------------------------
    # Internal scatter helper
    # ------------------------------------------------------------------

    def _scatter_group(
        self,
        data: Any,
        group_ranks: List[int],
        is_leader: bool,
        tag: int,
        blocking: bool = False,
    ) -> Any:
        if not group_ranks or self.world_rank not in group_ranks:
            return data
        if is_leader:
            if len(group_ranks) == 1:
                return data
            # Serialize once per fan-out. Each view pins the BytesIO buffer;
            # queued works retain those views until the sends have drained.
            frames = self._encode_control_frames(data)
            works = []
            for dst in group_ranks:
                if dst != self.world_rank:
                    for frame in frames:
                        work = dist.isend(
                            frame, dst=dst, tag=tag, group=self._world_cpu_group,
                        )
                        works.append((work, frame, dst))
            if blocking:
                for work, _frame, _dst in works:
                    work.wait()
            else:
                self._send_reaper.submit(works, tag)
            return data
        return self._recv(group_ranks[0], tag, self._world_cpu_group)

    # ------------------------------------------------------------------
    # Low-level send / recv on the world cpu group
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_control_frames(data: Any) -> List[torch.Tensor]:
        # Reserve a zero-filled inline frame and pickle directly after its
        # signed length header. Large objects grow the same backing buffer.
        buffer = io.BytesIO(bytes(_CONTROL_FRAME_SIZE))
        buffer.seek(_CONTROL_HEADER.size)
        pickle.dump(data, buffer)
        size = buffer.tell() - _CONTROL_HEADER.size
        view = buffer.getbuffer()
        _CONTROL_HEADER.pack_into(view, 0, size)
        tensor = torch.frombuffer(view, dtype=torch.uint8)
        frames = [tensor[:_CONTROL_FRAME_SIZE]]
        if size > _CONTROL_INLINE_SIZE:
            frames.append(tensor[_CONTROL_FRAME_SIZE:])
        return frames

    def _recv(self, src: int, tag: int = 0, group=None) -> Any:
        frame = torch.empty(_CONTROL_FRAME_SIZE, dtype=torch.uint8)
        dist.irecv(frame, src=src, tag=tag, group=group).wait()
        view = memoryview(frame.numpy())
        size = _CONTROL_HEADER.unpack_from(view)[0]
        if size <= 0:
            raise RuntimeError(
                "[FlexKV] received invalid serialized payload size: "
                f"rank={self.world_rank} src={src} tag={tag} size={size}"
            )
        if size <= _CONTROL_INLINE_SIZE:
            return pickle.loads(view[_CONTROL_HEADER.size:_CONTROL_HEADER.size + size])
        payload = torch.empty(size, dtype=torch.uint8)
        # Only the inline prefix needs copying; receive the exact remainder
        # directly into the final payload, then unpickle its buffer view.
        payload[:_CONTROL_INLINE_SIZE].copy_(frame[_CONTROL_HEADER.size:])
        dist.irecv(
            payload[_CONTROL_INLINE_SIZE:], src=src, tag=tag, group=group,
        ).wait()
        return pickle.loads(memoryview(payload.numpy()))

    def _send_tensor(
        self, tensor: torch.Tensor, dst: int, tag: int = 0, group=None
    ) -> None:
        dist.send(tensor, dst=dst, tag=tag, group=group)

    def _recv_tensor(
        self, tensor: torch.Tensor, src: int, tag: int = 0, group=None
    ) -> None:
        dist.recv(tensor, src=src, tag=tag, group=group)

    def _bcast_to_stage_members(self, tensor: torch.Tensor, tag: int) -> None:
        if not self.is_pp_stage_leader:
            self._recv_tensor(
                tensor,
                src=self._pp_stage_leader_ranks[self.pp_rank],
                tag=tag,
                group=self._world_cpu_group,
            )
            return
        for rank in self._pp_stage_member_ranks:
            if rank != self.world_rank:
                self._send_tensor(
                    tensor, dst=rank, tag=tag, group=self._world_cpu_group
                )

    def _pp_all_reduce_min_p2p(self, tensor: torch.Tensor) -> None:
        leader_rank = self._pp_stage_leader_ranks[0]
        other_leaders = self._pp_stage_leader_ranks[1:]
        tag = self._TAG_PP_AR_MIN
        if self.world_rank == leader_rank:
            result = int(tensor.item())
            for src in other_leaders:
                other = torch.tensor(0, dtype=torch.int64)
                self._recv_tensor(other, src=src, tag=tag, group=self._world_cpu_group)
                result = min(result, int(other.item()))
            tensor.fill_(result)
            for dst in other_leaders:
                self._send_tensor(tensor, dst=dst, tag=tag, group=self._world_cpu_group)
        else:
            self._send_tensor(
                tensor, dst=leader_rank, tag=tag, group=self._world_cpu_group
            )
            self._recv_tensor(
                tensor, src=leader_rank, tag=tag, group=self._world_cpu_group
            )

    def _pp_barrier_p2p(self) -> None:
        leader_rank = self._pp_stage_leader_ranks[0]
        other_leaders = self._pp_stage_leader_ranks[1:]
        tag = self._TAG_PP_BARRIER
        dummy = torch.tensor([1], dtype=torch.int64)
        if self.world_rank == leader_rank:
            for src in other_leaders:
                self._recv_tensor(dummy, src=src, tag=tag, group=self._world_cpu_group)
            for dst in other_leaders:
                self._send_tensor(dummy, dst=dst, tag=tag, group=self._world_cpu_group)
        else:
            self._send_tensor(
                dummy, dst=leader_rank, tag=tag, group=self._world_cpu_group
            )
            self._recv_tensor(
                dummy, src=leader_rank, tag=tag, group=self._world_cpu_group
            )


# ----------------------------------------------------------------------
# libc / eventfd / SCM_RIGHTS shims for the layerwise UDS handshake
# ----------------------------------------------------------------------

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.eventfd.argtypes = [ctypes.c_uint, ctypes.c_int]
_libc.eventfd.restype = ctypes.c_int
_libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_libc.read.restype = ctypes.c_ssize_t
_libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_libc.write.restype = ctypes.c_ssize_t

EFD_SEMAPHORE = 0x1
EFD_NONBLOCK = 0x800


def eventfd(initval: int = 0, flags: int = 0) -> int:
    fd = _libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def eventfd_write(fd: int, val: int) -> None:
    v = ctypes.c_uint64(val)
    n = _libc.write(fd, ctypes.byref(v), ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd write failed: {os.strerror(err)}")


def eventfd_read(fd: int) -> int:
    v = ctypes.c_uint64()
    n = _libc.read(fd, ctypes.byref(v), ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        if err == errno.EAGAIN:
            return 0
        raise OSError(err, f"eventfd read failed: {os.strerror(err)}")
    return v.value


def send_fds(sock: socket.socket, fds: list, extra_data: bytes = b"x") -> None:
    """SCM_RIGHTS-send a list of file descriptors over a UDS socket."""
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], ancdata)


# ----------------------------------------------------------------------
# Layerwise transfer signaling (eventfd-backed)
# ----------------------------------------------------------------------


class FlexKVLayerLoadingEvent:
    """One per producer slot. Holds ``num_layers`` semaphore eventfds —
    the FlexKV layerwise worker writes 1 to each as the corresponding
    layer's H2D copy completes; the consumer (sglang's
    ``register_layer_transfer_counter`` hook) reads to wait for them."""

    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        # Semaphore mode so each read consumes exactly one signal. NONBLOCK
        # lets ``reset_for_new_transfer`` drain leftover counter values
        # without blocking; ``wait`` re-arms the fd to blocking before
        # reading so consumers still get the desired blocking semantics.
        self.load_event_fds: List[int] = [
            eventfd(0, EFD_SEMAPHORE | EFD_NONBLOCK) for _ in range(num_layers)
        ]
        self._finished = True
        self._wait_started = False
        self.wait_remaining: List[int] = [0] * num_layers

    def _drain(self) -> None:
        """Consume all pending eventfd signals without blocking."""
        import os

        for fd in self.load_event_fds:
            while True:
                try:
                    if not os.read(fd, 8):
                        break
                except BlockingIOError:
                    break
                except OSError:
                    break

    def reset_for_new_transfer(self) -> None:
        """Drain any leftover signals from prior transfers, then arm.

        Without this drain, a previous transfer that wrote N eventfd
        signals but only had N-K reads (e.g. because the attention
        backend skipped a layer's ``get_key_buffer`` call) leaves K
        pending. The next transfer's first ``wait(layer)`` returns
        immediately reading one of those stale signals, even though
        the FlexKV worker hasn't actually finished that layer's H2D
        yet — and forward proceeds with wrong KV data.
        """
        self._drain()
        self._finished = False
        self._wait_started = False
        self.wait_remaining = [0] * self._num_layers

    def add_transfer(self) -> None:
        """Add one load to the current prefill batch.

        Several requests may restore into the same SGLang prefill batch. They
        share one eventfd set, and the layer hook consumes one signal per load
        before allowing that layer's forward to proceed.
        """
        if self._finished or self._wait_started:
            raise RuntimeError("Cannot add a transfer after layer waits have started")
        for layer_index in range(self._num_layers):
            self.wait_remaining[layer_index] += 1

    def mark_reusable(self) -> None:
        """Return this slot to the producer ring after transfer quiescence.

        The caller must first ensure that the FlexKV worker can no longer write
        to this event set. This method is used by the connector's flush path
        after it has completely waited for every launched layerwise task.
        """
        self._drain()
        self._finished = True
        self._wait_started = False
        self.wait_remaining = [0] * self._num_layers

    def pending_transfers(self) -> int:
        """Return the number of signals still expected for every layer."""
        if self._wait_started:
            raise RuntimeError("Layer waits have already started")
        counts = set(self.wait_remaining)
        if len(counts) != 1:
            raise RuntimeError(
                "Layerwise event has inconsistent per-layer transfer counts"
            )
        return self.wait_remaining[0]

    def remove_transfer(self) -> None:
        """Undo one pre-launch ``add_transfer`` during coordinated rollback."""
        if self._wait_started:
            raise RuntimeError(
                "Cannot remove a transfer after layer waits have started"
            )
        if self.pending_transfers() <= 0:
            raise RuntimeError("Layerwise event has no transfer to remove")
        self.wait_remaining = [count - 1 for count in self.wait_remaining]

    def wait(
        self,
        layer_index: int,
        count: int = 1,
        timeout_s: Optional[float] = None,
    ) -> None:
        """Block until the FlexKV worker signals layer ``layer_index``.

        The fd was created with EFD_NONBLOCK so reset can drain it. We
        re-introduce the blocking semantics with ``select.select`` on a
        NONBLOCK fd: the read after select is guaranteed to consume one
        signal.
        """
        import os
        import select

        assert 0 <= layer_index < self._num_layers
        assert count > 0
        self._wait_started = True
        fd = self.load_event_fds[layer_index]
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        for _ in range(count):
            while True:
                timeout = None
                if deadline is not None:
                    timeout = max(0.0, deadline - time.monotonic())
                readable, _, _ = select.select([fd], [], [], timeout)
                if not readable:
                    raise TimeoutError(
                        "Timed out waiting for a FlexKV layerwise eventfd signal"
                    )
                try:
                    buf = os.read(fd, 8)
                    if buf:
                        break
                except BlockingIOError:
                    # Spurious wakeup; loop and re-select.
                    continue
        if layer_index == self._num_layers - 1:
            self._finished = True

    def close(self) -> None:
        for fd in self.load_event_fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self.load_event_fds.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class FlexKVLayerDoneCounter:
    """Triple-buffered slot-based layerwise counter.

    The KV pool calls ``wait_until(layer_id)`` once per layer during forward.
    All restores collected while SGLang builds one prefill batch share a
    producer slot. The layer hook consumes one eventfd signal per restore for
    each layer. Producer rotation therefore happens per prefill batch, not per
    request, and the fixed-size ring remains sufficient for large batches.
    """

    def __init__(self, num_layers: int, num_counters: int = 3, world_rank: int = -1):
        self.num_layers = num_layers
        self.num_counters = num_counters
        self.world_rank = world_rank
        timeout_raw = os.environ.get("FLEXKV_LAYERWISE_WAIT_TIMEOUT_S", "240")
        try:
            timeout_s = float(timeout_raw)
        except ValueError:
            logger.warning(
                "[FlexKV] Invalid FLEXKV_LAYERWISE_WAIT_TIMEOUT_S=%r; using 240s",
                timeout_raw,
            )
            timeout_s = 240.0
        self.wait_timeout_s: Optional[float] = timeout_s if timeout_s > 0 else None
        self.events: List[FlexKVLayerLoadingEvent] = [
            FlexKVLayerLoadingEvent(num_layers) for _ in range(num_counters)
        ]
        self.producer_index = -1
        self.consumer_index = -1
        self._task_to_producer: Dict[int, int] = {}
        self._consumer_task_ids: List[int] = []

    def register_task(self, task_id: int, producer_id: int) -> None:
        if task_id in self._task_to_producer:
            raise RuntimeError(f"Task {task_id} is already registered")
        self._task_to_producer[task_id] = producer_id

    def expected_transfer_count(
        self, counter_id: int, *, allow_unclaimed_active: bool = False
    ) -> int:
        """Validate a counter selection without mutating local state.

        The sync leader may have just armed a fresh producer event, while
        followers only accept either their current open batch or a reusable
        event.  Returning the next signal count lets the connector detect
        rank-skewed batch boundaries before launching FlexKV.
        """
        if not 0 <= counter_id < self.num_counters:
            raise ValueError(
                f"Invalid counter_id={counter_id}, must be in [0, {self.num_counters})"
            )
        event = self.events[counter_id]
        if self.consumer_index == counter_id:
            if event._finished:
                raise RuntimeError(
                    f"Counter {counter_id} is marked finished but is still the consumer"
                )
            if event._wait_started:
                raise RuntimeError(
                    f"Counter {counter_id} has already started layer waits"
                )
            return event.pending_transfers() + 1

        if self.consumer_index >= 0:
            raise RuntimeError(
                f"Counter {self.consumer_index} is still the active consumer; "
                f"cannot select counter {counter_id}"
            )

        if event._finished:
            return 1

        if event._wait_started:
            raise RuntimeError(f"Counter {counter_id} has already started layer waits")

        if allow_unclaimed_active and self.producer_index == counter_id:
            pending = event.pending_transfers()
            if pending != 0:
                raise RuntimeError(
                    f"Unclaimed counter {counter_id} unexpectedly has {pending} transfers"
                )
            return 1

        raise RuntimeError(f"Counter {counter_id} is still active on a previous batch")

    def register_task_with_explicit_counter_id(
        self, task_id: int, counter_id: int
    ) -> None:
        self.expected_transfer_count(counter_id)
        if task_id in self._task_to_producer:
            raise RuntimeError(f"Task {task_id} is already registered")
        event = self.events[counter_id]
        if event._finished:
            event.reset_for_new_transfer()
        self._task_to_producer[task_id] = counter_id

    def update_producer(self) -> int:
        if self.consumer_index >= 0:
            event = self.events[self.consumer_index]
            if not event._finished and not event._wait_started:
                return self.consumer_index

        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ]._finished, "Producer event should be finished before reuse"
        self.events[self.producer_index].reset_for_new_transfer()
        return self.producer_index

    def set_consumer(self, task_id: int) -> None:
        if task_id < 0:
            return
        producer_id = self._task_to_producer.pop(task_id, None)
        if producer_id is None:
            return
        if self.consumer_index < 0:
            self.consumer_index = producer_id
        elif self.consumer_index != producer_id:
            raise RuntimeError(
                "FlexKV restores from different producer slots cannot share "
                "one prefill batch"
            )
        self.events[producer_id].add_transfer()
        self._consumer_task_ids.append(task_id)

    def rollback_prepared_transfer(
        self, task_id: int, producer_id: int, *, transfer_added: bool
    ) -> None:
        """Undo local counter state when an all-rank preflight fails."""
        if not 0 <= producer_id < self.num_counters:
            raise ValueError(
                f"Invalid producer_id={producer_id}, must be in "
                f"[0, {self.num_counters})"
            )
        self._task_to_producer.pop(task_id, None)
        event = self.events[producer_id]
        if transfer_added:
            event.remove_transfer()
            try:
                self._consumer_task_ids.remove(task_id)
            except ValueError as exc:
                raise RuntimeError(
                    f"Task {task_id} is missing from the active consumer batch"
                ) from exc

        if not event._wait_started and event.pending_transfers() == 0:
            if self.consumer_index == producer_id:
                self.consumer_index = -1
            event.mark_reusable()

    def abort_unregistered_selection(self, producer_id: int) -> None:
        """Release a freshly selected leader counter before registration."""
        if not 0 <= producer_id < self.num_counters:
            raise ValueError(
                f"Invalid producer_id={producer_id}, must be in "
                f"[0, {self.num_counters})"
            )
        event = self.events[producer_id]
        if self.consumer_index == producer_id or event._wait_started:
            raise RuntimeError(f"Counter {producer_id} is already being consumed")
        if event.pending_transfers() != 0:
            raise RuntimeError(
                f"Counter {producer_id} already has registered transfers"
            )
        event.mark_reusable()

    def wait_until(self, threshold: int) -> None:
        if self.consumer_index < 0:
            return
        event = self.events[self.consumer_index]
        wait_count = event.wait_remaining[threshold]
        if wait_count <= 0:
            return
        event.wait_remaining[threshold] = 0
        try:
            event.wait(
                threshold,
                wait_count,
                timeout_s=getattr(self, "wait_timeout_s", None),
            )
        except TimeoutError as exc:
            task_ids = list(getattr(self, "_consumer_task_ids", []))
            rank = getattr(self, "world_rank", -1)
            timeout_s = getattr(self, "wait_timeout_s", None)
            logger.error(
                "[ERROR][FlexKV-SGLang] operation=load act=wait "
                "status=timeout direction=H2D flexkv_task_ids=%s "
                "mode=layerwise rank=%d counter_id=%d layer=%d "
                "wait_count=%d timeout=%s",
                task_ids,
                rank,
                self.consumer_index,
                threshold,
                wait_count,
                "-" if timeout_s is None else f"{timeout_s}s",
            )
            raise TimeoutError(
                "FlexKV layerwise wait timed out: "
                f"rank={rank}, task_ids={task_ids}, counter_id={self.consumer_index}, "
                f"layer={threshold}, wait_count={wait_count}"
            ) from exc
        if threshold == self.num_layers - 1:
            self.consumer_index = -1
            self._consumer_task_ids.clear()

    def reset(self) -> None:
        # FlexKVConnector.reset() completely waits for every launched transfer
        # before calling this method, so no worker can signal these eventfds
        # after they are drained and returned to the producer ring.
        for event in self.events:
            event.mark_reusable()
        self.producer_index = -1
        self.consumer_index = -1
        self._task_to_producer.clear()
        self._consumer_task_ids.clear()

    def __del__(self) -> None:
        try:
            for event in self.events:
                event.close()
            self.events.clear()
        except Exception:
            pass
