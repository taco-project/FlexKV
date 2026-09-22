# SPDX-License-Identifier: Apache-2.0
# cython: boundscheck=True, wraparound=True
"""
Shared-memory IPC channel for CacheEngine ↔ TransferEngine communication in
multi-DP FlexKV.

Adapted from PR #144 (commit 5e262ca, originally for DPClient ↔ KVServer).
Slimmed down for the CE↔TE use case:
  - Per-channel size is small (256 KB ring + 256 KB sync) because the only
    payloads are pickled TransferOpGraph (submit) and CompletedOp lists (wait).
  - Sync request/response slot is dropped — CE↔TE is fully fire-and-forget on
    both directions; submit is async, completions are pushed asynchronously.
  - Two SPSC ring buffers per channel: `submit` (CE→TE) and `result` (TE→CE).
    Each side futex-waits on its own counter when the ring is empty.

Layout per channel (one /dev/shm file per CE):
  [0..64)        submit_write_pos  (uint64, CE writes)
  [64..128)      submit_read_pos   (uint64, TE writes)
  [128..192)     submit_wake       (int32, CE bumps + futex_wake)
  [192..256)     result_write_pos  (uint64, TE writes)
  [256..320)     result_read_pos   (uint64, CE writes)
  [320..384)     result_wake       (int32, TE bumps + futex_wake)
  [384..ring_off) reserved
  [ring_off ..)  submit ring (slot * SUBMIT_SLOTS)
                 result ring (slot * RESULT_SLOTS)

A single `ShmControlBlock` (separate /dev/shm file) carries a global wake
counter that the TE polls when it has more than one channel attached, so it
can sleep idly without per-channel futex wait.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import pickle
import platform
import struct
from typing import Any, List, Optional

# ── Linux futex wrappers ────────────────────────────────────────────────

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# futex syscall number is arch-specific; pick at import time.
_MACHINE = platform.machine().lower()
if _MACHINE in ("x86_64", "amd64"):
    _SYS_FUTEX = 202
elif _MACHINE in ("aarch64", "arm64"):
    _SYS_FUTEX = 98
else:  # pragma: no cover
    raise RuntimeError(f"Unsupported machine for futex syscall: {_MACHINE}")
_FUTEX_WAIT = 0
_FUTEX_WAKE = 1


def _futex_wait(addr: int, expected: int, timeout_ns: Optional[int] = None) -> int:
    if timeout_ns is None:
        return _libc.syscall(
            _SYS_FUTEX, ctypes.c_void_p(addr),
            _FUTEX_WAIT, ctypes.c_int(expected),
            ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0),
        )
    # struct timespec
    ts = (ctypes.c_long * 2)(timeout_ns // 1_000_000_000,
                             timeout_ns % 1_000_000_000)
    return _libc.syscall(
        _SYS_FUTEX, ctypes.c_void_p(addr),
        _FUTEX_WAIT, ctypes.c_int(expected),
        ctypes.byref(ts), ctypes.c_void_p(0), ctypes.c_int(0),
    )


def _futex_wake(addr: int, count: int = 1) -> int:
    return _libc.syscall(
        _SYS_FUTEX, ctypes.c_void_p(addr),
        _FUTEX_WAKE, ctypes.c_int(count),
        ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0),
    )


# ── Layout constants ────────────────────────────────────────────────────

_CL = 64  # cache line

# Header lives in the first 6 cache lines; ring data starts on a page boundary.
OFF_SUBMIT_W = 0 * _CL
OFF_SUBMIT_R = 1 * _CL
OFF_SUBMIT_WAKE = 2 * _CL
OFF_RESULT_W = 3 * _CL
OFF_RESULT_R = 4 * _CL
OFF_RESULT_WAKE = 5 * _CL
HEADER_SIZE = 6 * _CL  # 384 B

# Submit ring holds pickled TransferOpGraphs, fragmented across slots so a payload
# larger than one slot spans several (32 KB × 8192 = 256 MB, ~256 MB max message).
# Fragmentation retired the old "must fit one slot" constraint that made high-QPS
# as_batch=True graphs overflow (observed 152 KB @2500 with prefetch).
DEFAULT_SUBMIT_SLOTS = 8192         # power of 2
DEFAULT_SUBMIT_SLOT_SIZE = 32 * 1024
DEFAULT_SLOT_SIZE = DEFAULT_SUBMIT_SLOT_SIZE  # back-compat alias for `slot_size=`

# Result ring holds one fixed-width CompletedOp record per slot (64 B × 65536 = 4 MB).
DEFAULT_RESULT_SLOTS = 65536        # power of 2
DEFAULT_RESULT_SLOT_SIZE = 64       # cache line; one 54 B CompletedOp record

_PAGE = 4096


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


# Submit-ring fragment header (per slot): payload bytes in this slot (u32) + a
# last-fragment flag (u8). A message is the concatenation of fragments up to and
# including the one with is_last=1.
_FRAG_HDR = struct.Struct("<IB")
_FRAG_HDR_SIZE = _FRAG_HDR.size  # 5


# CompletedOp result-ring record: graph_id/op_id (i64), transfer_type (u8 index),
# num_blocks (u32), num_bytes (u64), flags (u8: bit0 = failed), then the
# worker-measured wait_ms / xfer_ms / e2e_ms (f64 each) so the CE side can
# export the same transfer-duration histograms as the in-process path.
# block_results does NOT cross the ring: a failed graph degrades to whole-task
# failure on the client side, which is the correct conservative reading for
# every backend the shm path serves (mooncake's partial success never runs
# through this channel).
_COMPLETED_OP = struct.Struct("<qqBIQBddd")
COMPLETED_OP_WIRE_SIZE = _COMPLETED_OP.size

# transfer_type frozen as a byte index; 0xFF = None (VIRTUAL ops). Every
# TransferType value must be here (tests/test_shm_channel.py checks): the TE
# sends `op.transfer_type.value` for each completed op. Append only, the index
# is the wire format.
_TT_NONE = 0xFF
_TT_NAMES = (
    "H2D", "D2H", "DISK2H", "H2DISK", "DISK2D", "D2DISK",
    "REMOTE2H", "H2REMOTE", "PEERH2H", "H2PEERH", "PEERSSD2H", "H2PEERSSD",
    "VIRTUAL", "LAYERWISE",
)
_TT_NAME_TO_IDX = {name: i for i, name in enumerate(_TT_NAMES)}
_tt_unknown_warned: set = set()


def transfer_type_index(tt: Any) -> int:
    """Wire index of a CompletedOp.transfer_type (a TransferType value, the
    member itself, or None). A name the table does not know goes out as None
    with one warning per name: the CE loses that op's type attribution, which
    beats a KeyError in the TE's result thread that would stop every
    completion on the node."""
    if tt is None:
        return _TT_NONE
    key = getattr(tt, "value", tt)
    idx = _TT_NAME_TO_IDX.get(key)
    if idx is None:
        if key not in _tt_unknown_warned:
            _tt_unknown_warned.add(key)
            from flexkv.common.debug import flexkv_logger
            flexkv_logger.warning(
                f"shm channel: transfer type {key!r} has no wire index; it is sent as "
                f"None (add it to shm_channel._TT_NAMES)")
        return _TT_NONE
    return idx


def encode_completed_op(op: Any) -> bytes:
    """Pack a CompletedOp into its fixed-width record."""
    tt_idx = transfer_type_index(op.transfer_type)
    flags = 1 if getattr(op, "failed", False) else 0
    return _COMPLETED_OP.pack(
        op.graph_id, op.op_id, tt_idx, op.num_blocks, op.num_bytes, flags,
        float(getattr(op, "wait_ms", 0.0)),
        float(getattr(op, "xfer_ms", 0.0)),
        float(getattr(op, "e2e_ms", 0.0)),
    )


def decode_completed_op(buf: Any, off: int) -> Any:
    """Unpack a CompletedOp record from `buf` at byte offset `off`."""
    from flexkv.common.transfer import CompletedOp
    graph_id, op_id, tt_idx, num_blocks, num_bytes, flags, wait_ms, xfer_ms, e2e_ms = \
        _COMPLETED_OP.unpack_from(buf, off)
    tt = None if tt_idx == _TT_NONE or tt_idx >= len(_TT_NAMES) else _TT_NAMES[tt_idx]
    return CompletedOp(
        graph_id=graph_id,
        op_id=op_id,
        transfer_type=tt,
        num_blocks=num_blocks,
        num_bytes=num_bytes,
        wait_ms=wait_ms,
        xfer_ms=xfer_ms,
        e2e_ms=e2e_ms,
        failed=bool(flags & 1),
    )


# ── ShmControlBlock ─────────────────────────────────────────────────────

CTRL_WAKE = 0
CTRL_READY = _CL
CTRL_SIZE = _PAGE


def _safe_id(server_id: str) -> str:
    return server_id.replace("/", "_").replace(":", "_").strip("_")


class ShmControlBlock:
    """Optional global wake counter used by the TE when polling N channels."""

    def __init__(self, server_id: str, create: bool = False):
        self.server_id = server_id
        safe = _safe_id(server_id)
        self.shm_path = f"/dev/shm/flexkv_te_ctrl_{safe}"

        if create:
            fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, CTRL_SIZE)
            self.buf = mmap.mmap(fd, CTRL_SIZE)
            os.close(fd)
            self.buf[:] = b"\x00" * CTRL_SIZE
        else:
            fd = os.open(self.shm_path, os.O_RDWR)
            self.buf = mmap.mmap(fd, CTRL_SIZE)
            os.close(fd)

        self._base = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        self._wake = ctypes.c_int32.from_address(self._base + CTRL_WAKE)
        self._ready = ctypes.c_int32.from_address(self._base + CTRL_READY)

    @property
    def _wake_addr(self) -> int:
        return self._base + CTRL_WAKE

    @property
    def _ready_addr(self) -> int:
        return self._base + CTRL_READY

    def notify(self) -> None:
        # Read-modify-write is not atomic across processes; the TE uses snapshot
        # comparison so any change wakes it.
        self._wake.value += 1
        _futex_wake(self._wake_addr, 1)

    def get_wake(self) -> int:
        return self._wake.value

    def wait(self, expected: int, timeout_ns: Optional[int] = None) -> None:
        _futex_wait(self._wake_addr, expected, timeout_ns)

    def set_ready(self) -> None:
        self._ready.value = 1
        _futex_wake(self._ready_addr, 0x7FFFFFFF)

    def wait_ready(self, timeout_s: float = 60.0) -> bool:
        import time
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ready.value != 0:
                return True
            _futex_wait(self._ready_addr, 0,
                        timeout_ns=int(0.5 * 1_000_000_000))
        return False

    def close(self) -> None:
        if self.buf is not None:
            self.buf.close()
            self.buf = None

    def unlink(self) -> None:
        try:
            os.unlink(self.shm_path)
        except FileNotFoundError:
            pass


# ── ShmChannel (CE ↔ TE) ────────────────────────────────────────────────

class ShmChannel:
    """One bi-directional channel between a single CE and the TE.

    Two SPSC rings: CE→TE submit, TE→CE result. Each side futex-waits on its
    own wake counter when its consumer ring is empty.
    """

    def __init__(self,
                 server_id: str,
                 channel_id: int,
                 create: bool = False,
                 submit_slots: int = DEFAULT_SUBMIT_SLOTS,
                 result_slots: int = DEFAULT_RESULT_SLOTS,
                 slot_size: int = DEFAULT_SUBMIT_SLOT_SIZE,
                 result_slot_size: int = DEFAULT_RESULT_SLOT_SIZE):
        assert submit_slots & (submit_slots - 1) == 0, \
            "submit_slots must be power of 2"
        assert result_slots & (result_slots - 1) == 0, \
            "result_slots must be power of 2"
        assert result_slot_size >= COMPLETED_OP_WIRE_SIZE, \
            f"result_slot_size {result_slot_size} < CompletedOp record " \
            f"{COMPLETED_OP_WIRE_SIZE}"

        self.channel_id = channel_id
        self.submit_slots = submit_slots
        self.result_slots = result_slots
        self.slot_size = slot_size  # submit ring; result ring uses result_slot_size
        self.result_slot_size = result_slot_size

        safe = _safe_id(server_id)
        self.shm_path = f"/dev/shm/flexkv_te_ch_{safe}_{channel_id}"

        # Lay out: header -> aligned to page -> submit ring -> result ring.
        self._submit_off = _round_up(HEADER_SIZE, _PAGE)
        self._result_off = self._submit_off + submit_slots * slot_size
        total = self._result_off + result_slots * result_slot_size

        self.total_size = total

        if create:
            fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, total)
            self.buf = mmap.mmap(fd, total)
            os.close(fd)
            self.buf[:HEADER_SIZE] = b"\x00" * HEADER_SIZE
        else:
            fd = os.open(self.shm_path, os.O_RDWR)
            self.buf = mmap.mmap(fd, total)
            os.close(fd)

        self._base = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        self._submit_w = ctypes.c_uint64.from_address(self._base + OFF_SUBMIT_W)
        self._submit_r = ctypes.c_uint64.from_address(self._base + OFF_SUBMIT_R)
        self._submit_wake = ctypes.c_int32.from_address(self._base + OFF_SUBMIT_WAKE)
        self._result_w = ctypes.c_uint64.from_address(self._base + OFF_RESULT_W)
        self._result_r = ctypes.c_uint64.from_address(self._base + OFF_RESULT_R)
        self._result_wake = ctypes.c_int32.from_address(self._base + OFF_RESULT_WAKE)

    # ---- futex helpers ----

    @property
    def _submit_wake_addr(self) -> int:
        return self._base + OFF_SUBMIT_WAKE

    @property
    def _result_wake_addr(self) -> int:
        return self._base + OFF_RESULT_WAKE

    def _bump_wake(self, ptr: ctypes.c_int32, addr: int) -> None:
        ptr.value += 1
        _futex_wake(addr, 1)

    # ---- ring helpers ----

    def _ring_full(self, w: int, r: int, slots: int) -> bool:
        return ((w + 1) & (slots - 1)) == r

    def _ring_used(self, w: int, r: int, slots: int) -> int:
        return (w - r) & (slots - 1)

    # ---- CE side: submit + recv result ----

    def submit_send(self, payload: Any) -> None:
        """Enqueue a payload to TE, fragmenting it across slots if it exceeds one.

        All fragments are written first, then submit_w is advanced once, so the TE
        never observes a partial message. Spins+yields until enough contiguous
        slots are free."""
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        slots = self.submit_slots
        body = self.slot_size - _FRAG_HDR_SIZE
        nfrag = max(1, (len(blob) + body - 1) // body)
        if nfrag > slots - 1:
            raise ValueError(
                f"payload needs {nfrag} fragments but submit ring holds "
                f"{slots - 1}; raise submit_slots or slot_size")

        wp = self._submit_w.value
        spin = 0
        while self._ring_used(wp, self._submit_r.value, slots) + nfrag > slots - 1:
            if spin > 1_000_000:
                raise RuntimeError("shm channel submit ring full")
            if spin > 1000:
                os.sched_yield()
            spin += 1

        w = wp
        for i in range(nfrag):
            chunk = blob[i * body:(i + 1) * body]
            off = self._submit_off + w * self.slot_size
            self.buf[off:off + _FRAG_HDR_SIZE] = _FRAG_HDR.pack(
                len(chunk), 1 if i == nfrag - 1 else 0)
            self.buf[off + _FRAG_HDR_SIZE:off + _FRAG_HDR_SIZE + len(chunk)] = chunk
            w = (w + 1) & (slots - 1)
        self._submit_w.value = w
        self._bump_wake(self._submit_wake, self._submit_wake_addr)

    def result_recv(self, timeout_s: Optional[float] = None) -> List[Any]:
        """Drain pending TE→CE results, one CompletedOp per slot. Blocks up to
        `timeout_s` if empty."""
        out: List[Any] = []
        rp = self._result_r.value
        wp = self._result_w.value
        slots = self.result_slots
        if rp == wp and timeout_s is not None and timeout_s > 0:
            wake = self._result_wake.value
            # Re-check; TE might have arrived between read and wait.
            wp = self._result_w.value
            if rp == wp:
                if timeout_s == float("inf"):
                    _futex_wait(self._result_wake_addr, wake)
                else:
                    _futex_wait(self._result_wake_addr, wake,
                                timeout_ns=int(timeout_s * 1_000_000_000))
                wp = self._result_w.value

        while rp != wp:
            out.append(decode_completed_op(
                self.buf, self._result_off + rp * self.result_slot_size))
            rp = (rp + 1) & (slots - 1)
        if out:
            self._result_r.value = rp
        return out

    # ---- TE side: recv submit + send result ----

    def submit_recv(self) -> List[Any]:
        """Drain CE→TE submissions (non-blocking), reassembling fragmented
        messages. submit_r is advanced only past fully-received messages."""
        out: List[Any] = []
        rp = self._submit_r.value
        wp = self._submit_w.value
        slots = self.submit_slots
        frags: List[bytes] = []
        while rp != wp:
            off = self._submit_off + rp * self.slot_size
            n, is_last = _FRAG_HDR.unpack_from(self.buf, off)
            frags.append(bytes(self.buf[off + _FRAG_HDR_SIZE:
                                        off + _FRAG_HDR_SIZE + n]))
            rp = (rp + 1) & (slots - 1)
            if is_last:
                payload = b"".join(frags)
                frags.clear()
                self._submit_r.value = rp  # release this message's slots (payload is a copy)
                try:
                    out.append(pickle.loads(payload))
                except Exception as e:  # noqa: BLE001 - a record we cannot decode
                    # Dropping it costs the sender one task; keeping it would
                    # re-raise on every poll and stall the whole ring.
                    from flexkv.common.debug import flexkv_logger
                    flexkv_logger.error(
                        f"shm channel {self.channel_id}: dropping an undecodable "
                        f"{len(payload)}-byte submit record ({e!r})")
        return out

    def result_send(self, ops: List[Any]) -> None:
        """Enqueue a batch of CompletedOps, one fixed-width record per slot, and
        wake the CE once at the end. Spins if the ring fills rather than dropping a
        completion (which would hang the owning task); with 65536 slots that is
        effectively unreachable."""
        if not ops:
            return
        slots = self.result_slots
        slot_sz = self.result_slot_size
        base = self._result_off
        wp = self._result_w.value
        warned = False
        for op in ops:
            if self._ring_full(wp, self._result_r.value, slots):
                # Full mid-batch: publish+wake so the CE drains, then spin.
                self._bump_wake(self._result_wake, self._result_wake_addr)
                spin = 0
                while self._ring_full(wp, self._result_r.value, slots):
                    if spin > 1000:
                        os.sched_yield()
                    spin += 1
                    if spin % 5_000_000 == 0 and not warned:
                        try:
                            from flexkv.common.debug import flexkv_logger
                            flexkv_logger.error(
                                f"shm channel result ring stuck full "
                                f"(slots={slots}); is the CE consumer alive?"
                            )
                        except Exception:
                            pass
                        warned = True
            off = base + wp * slot_sz
            self.buf[off:off + COMPLETED_OP_WIRE_SIZE] = encode_completed_op(op)
            wp = (wp + 1) & (slots - 1)
            self._result_w.value = wp
        self._bump_wake(self._result_wake, self._result_wake_addr)

    # ---- Submit-wake fileno: lets TE selector wait on this channel ----

    @property
    def submit_wake_fd(self) -> int:
        # We don't have a real eventfd; selector users should poll get_wake()
        # delta + futex_wait via ShmControlBlock. Returning -1 signals "no fd".
        return -1

    # ---- Lifecycle ----

    def close(self) -> None:
        if self.buf is not None:
            self.buf.close()
            self.buf = None

    def unlink(self) -> None:
        try:
            os.unlink(self.shm_path)
        except FileNotFoundError:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
