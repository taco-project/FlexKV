"""
Shared Memory IPC Channel for FlexKV server-client communication.

Replaces ZMQ+pickle with shared memory + futex for low-latency IPC.
Works with both mp.Process and subprocess.Popen server modes since
all communication goes through named files in /dev/shm/.

Design:
- ShmControlBlock: global wake counter for server idle sleep (futex-based)
- ShmChannel: per-client channel with:
  - SPSC ring buffer for async (fire-and-forget) requests
  - Sync request/response slot with spin-wait + adaptive futex fallback
- Binary message format: fixed header + raw numpy buffers (no pickle)
"""

import ctypes
import ctypes.util
import mmap
import os
import struct
import numpy as np
from enum import IntEnum
from typing import Optional, Dict, List, Tuple, Union

from flexkv.common.request import KVResponseStatus, KVResponse

# ── Linux futex wrappers ────────────────────────────────────────────────

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

_SYS_FUTEX = 202  # x86_64
_FUTEX_WAIT = 0
_FUTEX_WAKE = 1


def futex_wait(addr: int, expected: int) -> int:
    return _libc.syscall(
        _SYS_FUTEX, ctypes.c_void_p(addr),
        _FUTEX_WAIT, ctypes.c_int(expected),
        ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0),
    )


def futex_wake(addr: int, count: int = 1) -> int:
    return _libc.syscall(
        _SYS_FUTEX, ctypes.c_void_p(addr),
        _FUTEX_WAKE, ctypes.c_int(count),
        ctypes.c_void_p(0), ctypes.c_void_p(0), ctypes.c_int(0),
    )


# ── Message types ───────────────────────────────────────────────────────

class ShmMsgType(IntEnum):
    # Async (fire-and-forget)
    PUT_ASYNC = 1
    GET_ASYNC = 2
    PREFETCH_ASYNC = 3
    LAUNCH_TASKS = 4
    CANCEL_TASKS = 5

    # Sync (round-trip)
    GET_MATCH = 10
    PUT_MATCH = 11
    WAIT = 12
    TRY_WAIT = 13
    IS_READY = 14

    # Control
    START = 50
    REGISTER = 51
    SHUTDOWN = 52


# ── Shared memory layout constants ──────────────────────────────────────

_CL = 64  # cache line size

# Async ring buffer
ASYNC_RING_SLOTS = 4096         # power of 2
ASYNC_SLOT_SIZE = 256 * 1024    # 256KB per slot
ASYNC_RING_SIZE = ASYNC_RING_SLOTS * ASYNC_SLOT_SIZE

# Sync request/response area
SYNC_REQ_SIZE = 256 * 1024      # 256KB
SYNC_RESP_SIZE = 256 * 1024     # 256KB

# Header offsets (cache-line aligned)
OFF_ASYNC_WRITE = 0             # uint64, client writes
OFF_ASYNC_READ = _CL            # uint64, server writes
OFF_SYNC_REQ_FLAG = 2 * _CL    # int32, client sets to 1
OFF_SYNC_RESP_FLAG = 3 * _CL   # int32, server sets to 1

HEADER_SIZE = 4 * _CL  # 256 bytes
ASYNC_RING_OFFSET = HEADER_SIZE
SYNC_REQ_OFFSET = HEADER_SIZE + ASYNC_RING_SIZE
SYNC_RESP_OFFSET = SYNC_REQ_OFFSET + SYNC_REQ_SIZE
TOTAL_SHM_SIZE = SYNC_RESP_OFFSET + SYNC_RESP_SIZE

# Control block layout
CTRL_WAKE_COUNTER = 0       # int32, any client increments
CTRL_SERVER_READY = _CL     # int32, server sets to 1
CTRL_SIZE = 4096             # one page

# Message header format
# msg_type(u8) + dp_client_id(i32) + task_id(i64) + n_tokens(i32)
# + flags(u8) + layer_granularity(i32) + n_task_ids(i32) + batch_id(i64)
# + wait_timeout(f64) + n_namespace(i32)
MSG_HEADER_FMT = "<BiqiBiiqdi"
MSG_HEADER_SIZE = struct.calcsize(MSG_HEADER_FMT)

# Flags bits
FLAG_HAS_SLOT_MAPPING = 0x01
FLAG_HAS_TOKEN_MASK = 0x02
FLAG_HAS_NAMESPACE = 0x04
FLAG_COMPLETELY = 0x08
FLAG_AS_BATCH = 0x10

# Response header format
# status_code(i32) + task_id(i64) + is_ready(u8) + has_mask(u8)
# + mask_len(i32) + n_kv_responses(i32) + error_msg_len(i32)
RESP_HEADER_FMT = "<iqBBiii"
RESP_HEADER_SIZE = struct.calcsize(RESP_HEADER_FMT)


# ── ShmControlBlock ─────────────────────────────────────────────────────

def _sanitize_server_id(server_id: str) -> str:
    """Convert server_id to a safe filename component."""
    return server_id.replace("/", "_").replace(":", "_").strip("_")


class ShmControlBlock:
    """
    Global control block shared between server and all clients.
    Used for:
    - Wake counter: clients increment + futex_wake to wake idle server
    - Server ready flag: server sets after initialization
    """

    def __init__(self, server_id: str, create: bool = False):
        self.server_id = server_id
        safe_id = _sanitize_server_id(server_id)
        self.shm_name = f"flexkv_shm_ctrl_{safe_id}"
        self.shm_path = f"/dev/shm/{self.shm_name}"

        if create:
            fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, CTRL_SIZE)
            self.buf = mmap.mmap(fd, CTRL_SIZE)
            os.close(fd)
            self.buf[:] = b'\x00' * CTRL_SIZE
        else:
            fd = os.open(self.shm_path, os.O_RDWR)
            self.buf = mmap.mmap(fd, CTRL_SIZE)
            os.close(fd)

        self._base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))

    @property
    def _wake_addr(self) -> int:
        return self._base_addr + CTRL_WAKE_COUNTER

    @property
    def _ready_addr(self) -> int:
        return self._base_addr + CTRL_SERVER_READY

    def notify_server(self) -> None:
        """Client calls this to wake the server from idle sleep."""
        # Atomic increment: read-modify-write (safe because each client
        # only needs to signal "something changed", exact count doesn't matter)
        val = struct.unpack_from("<i", self.buf, CTRL_WAKE_COUNTER)[0]
        struct.pack_into("<i", self.buf, CTRL_WAKE_COUNTER, val + 1)
        futex_wake(self._wake_addr, 1)

    def wait_for_work(self, spin_threshold: int = 1000) -> None:
        """Server calls this when idle — spins briefly, then futex sleeps."""
        for _ in range(spin_threshold):
            # Quick check: anything changed?
            return  # in spin mode, just return immediately to re-check
        cur = struct.unpack_from("<i", self.buf, CTRL_WAKE_COUNTER)[0]
        futex_wait(self._wake_addr, cur)

    def get_wake_counter(self) -> int:
        return struct.unpack_from("<i", self.buf, CTRL_WAKE_COUNTER)[0]

    def futex_wait_on_wake(self, expected: int) -> None:
        futex_wait(self._wake_addr, expected)

    def set_server_ready(self) -> None:
        struct.pack_into("<i", self.buf, CTRL_SERVER_READY, 1)
        futex_wake(self._ready_addr, 2147483647)  # wake all waiters

    def wait_server_ready(self, timeout_s: float = 60.0) -> bool:
        """Client polls/waits for server to be ready."""
        import time
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if struct.unpack_from("<i", self.buf, CTRL_SERVER_READY)[0] != 0:
                return True
            futex_wait(self._ready_addr, 0)
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


# ── Message pack/unpack ─────────────────────────────────────────────────

def pack_request(buf, offset: int, msg_type: int, dp_client_id: int,
                 task_id: int = -1,
                 token_ids: Optional[np.ndarray] = None,
                 slot_mapping: Optional[np.ndarray] = None,
                 token_mask: Optional[np.ndarray] = None,
                 layer_granularity: int = -1,
                 task_ids: Optional[List[int]] = None,
                 slot_mappings: Optional[List[np.ndarray]] = None,
                 as_batch: bool = False,
                 batch_id: int = -1,
                 wait_timeout: float = 20.0,
                 completely: bool = False,
                 namespace: Optional[List[str]] = None,
                 model_config_bytes: Optional[bytes] = None) -> int:
    """Pack a request message into buffer at offset. Returns total bytes written."""
    n_tokens = len(token_ids) if token_ids is not None else 0
    n_task_ids = len(task_ids) if task_ids is not None else 0
    n_namespace = len(namespace) if namespace is not None else 0

    flags = 0
    if slot_mapping is not None:
        flags |= FLAG_HAS_SLOT_MAPPING
    if token_mask is not None:
        flags |= FLAG_HAS_TOKEN_MASK
    if namespace is not None:
        flags |= FLAG_HAS_NAMESPACE
    if completely:
        flags |= FLAG_COMPLETELY
    if as_batch:
        flags |= FLAG_AS_BATCH

    struct.pack_into(MSG_HEADER_FMT, buf, offset,
                     msg_type, dp_client_id, task_id, n_tokens,
                     flags, layer_granularity, n_task_ids, batch_id,
                     wait_timeout, n_namespace)
    pos = offset + MSG_HEADER_SIZE

    # token_ids (int64)
    if token_ids is not None and n_tokens > 0:
        nbytes = token_ids.nbytes
        buf[pos:pos + nbytes] = token_ids.tobytes()
        pos += nbytes

    # slot_mapping (int64)
    if slot_mapping is not None and n_tokens > 0:
        nbytes = slot_mapping.nbytes
        buf[pos:pos + nbytes] = slot_mapping.tobytes()
        pos += nbytes

    # token_mask (bool/uint8)
    if token_mask is not None and n_tokens > 0:
        mask_bytes = token_mask.astype(np.uint8).tobytes()
        buf[pos:pos + len(mask_bytes)] = mask_bytes
        pos += len(mask_bytes)

    # task_ids list (int64)
    if task_ids is not None and n_task_ids > 0:
        arr = np.array(task_ids, dtype=np.int64)
        nbytes = arr.nbytes
        buf[pos:pos + nbytes] = arr.tobytes()
        pos += nbytes

    # slot_mappings list (for launch_tasks)
    if slot_mappings is not None and n_task_ids > 0:
        for sm in slot_mappings:
            sm_len = len(sm)
            struct.pack_into("<i", buf, pos, sm_len)
            pos += 4
            nbytes = sm.nbytes
            buf[pos:pos + nbytes] = sm.tobytes()
            pos += nbytes

    # namespace
    if namespace is not None and n_namespace > 0:
        for ns in namespace:
            ns_bytes = ns.encode("utf-8")
            struct.pack_into("<i", buf, pos, len(ns_bytes))
            pos += 4
            buf[pos:pos + len(ns_bytes)] = ns_bytes
            pos += len(ns_bytes)

    # model_config_bytes (for register request)
    if model_config_bytes is not None:
        struct.pack_into("<i", buf, pos, len(model_config_bytes))
        pos += 4
        buf[pos:pos + len(model_config_bytes)] = model_config_bytes
        pos += len(model_config_bytes)

    return pos - offset


def unpack_request(buf, offset: int) -> dict:
    """Unpack a request message from buffer. Returns dict of fields."""
    (msg_type, dp_client_id, task_id, n_tokens,
     flags, layer_gran, n_task_ids, batch_id,
     wait_timeout, n_namespace) = struct.unpack_from(MSG_HEADER_FMT, buf, offset)
    pos = offset + MSG_HEADER_SIZE

    result = {
        "msg_type": msg_type,
        "dp_client_id": dp_client_id,
        "task_id": task_id,
        "n_tokens": n_tokens,
        "flags": flags,
        "layer_granularity": layer_gran,
        "n_task_ids": n_task_ids,
        "batch_id": batch_id,
        "wait_timeout": wait_timeout,
        "completely": bool(flags & FLAG_COMPLETELY),
        "as_batch": bool(flags & FLAG_AS_BATCH),
    }

    # token_ids
    if n_tokens > 0:
        nbytes = n_tokens * 8
        result["token_ids"] = np.frombuffer(buf[pos:pos + nbytes], dtype=np.int64).copy()
        pos += nbytes
    else:
        result["token_ids"] = None

    # slot_mapping
    if (flags & FLAG_HAS_SLOT_MAPPING) and n_tokens > 0:
        nbytes = n_tokens * 8
        result["slot_mapping"] = np.frombuffer(buf[pos:pos + nbytes], dtype=np.int64).copy()
        pos += nbytes
    else:
        result["slot_mapping"] = None

    # token_mask
    if (flags & FLAG_HAS_TOKEN_MASK) and n_tokens > 0:
        nbytes = n_tokens
        raw = np.frombuffer(buf[pos:pos + nbytes], dtype=np.uint8).copy()
        result["token_mask"] = raw.astype(bool)
        pos += nbytes
    else:
        result["token_mask"] = None

    # task_ids
    if n_task_ids > 0:
        nbytes = n_task_ids * 8
        result["task_ids"] = np.frombuffer(buf[pos:pos + nbytes], dtype=np.int64).tolist()
        pos += nbytes
    else:
        result["task_ids"] = None

    # slot_mappings
    if n_task_ids > 0 and (flags & FLAG_HAS_SLOT_MAPPING) and result["token_ids"] is None:
        # slot_mappings for launch_tasks (when there are no token_ids)
        sms = []
        for _ in range(n_task_ids):
            sm_len = struct.unpack_from("<i", buf, pos)[0]
            pos += 4
            nbytes = sm_len * 8
            sms.append(np.frombuffer(buf[pos:pos + nbytes], dtype=np.int64).copy())
            pos += nbytes
        result["slot_mappings"] = sms
    elif n_task_ids > 0 and msg_type == ShmMsgType.LAUNCH_TASKS:
        # launch_tasks always has slot_mappings
        sms = []
        for _ in range(n_task_ids):
            sm_len = struct.unpack_from("<i", buf, pos)[0]
            pos += 4
            nbytes = sm_len * 8
            sms.append(np.frombuffer(buf[pos:pos + nbytes], dtype=np.int64).copy())
            pos += nbytes
        result["slot_mappings"] = sms
    else:
        result["slot_mappings"] = None

    # namespace
    if (flags & FLAG_HAS_NAMESPACE) and n_namespace > 0:
        ns_list = []
        for _ in range(n_namespace):
            ns_len = struct.unpack_from("<i", buf, pos)[0]
            pos += 4
            ns_list.append(buf[pos:pos + ns_len].decode("utf-8") if isinstance(buf[pos:pos + ns_len], bytes)
                           else bytes(buf[pos:pos + ns_len]).decode("utf-8"))
            pos += ns_len
        result["namespace"] = ns_list
    else:
        result["namespace"] = None

    # model_config_bytes (for register)
    if msg_type == ShmMsgType.REGISTER:
        if pos < len(buf):
            try:
                cfg_len = struct.unpack_from("<i", buf, pos)[0]
                pos += 4
                result["model_config_bytes"] = bytes(buf[pos:pos + cfg_len])
                pos += cfg_len
            except (struct.error, IndexError):
                result["model_config_bytes"] = None
        else:
            result["model_config_bytes"] = None

    return result


# ── KVResponse status mapping ───────────────────────────────────────────

_STATUS_TO_INT = {
    KVResponseStatus.SUCCESS: 0,
    KVResponseStatus.NOTFOUND: 1,
    KVResponseStatus.UNREADY: 2,
    KVResponseStatus.TIMEOUT: 3,
    KVResponseStatus.CANCELLED: 4,
    KVResponseStatus.FAILED: 5,
}

_INT_TO_STATUS = {v: k for k, v in _STATUS_TO_INT.items()}


def pack_response(buf, offset: int,
                  status_code: int = 0,
                  task_id: int = -1,
                  is_ready: bool = False,
                  mask: Optional[np.ndarray] = None,
                  kv_responses: Optional[Dict[int, KVResponse]] = None,
                  error_msg: Optional[str] = None) -> int:
    """Pack response into shared memory buffer. Returns bytes written."""
    has_mask = 1 if mask is not None else 0
    mask_len = len(mask) if mask is not None else 0
    n_kv = len(kv_responses) if kv_responses is not None else 0
    error_bytes = error_msg.encode("utf-8") if error_msg else b""
    error_len = len(error_bytes)

    struct.pack_into(RESP_HEADER_FMT, buf, offset,
                     status_code, task_id, int(is_ready),
                     has_mask, mask_len, n_kv, error_len)
    pos = offset + RESP_HEADER_SIZE

    # Pack mask (np.ndarray, typically bool or uint8)
    if mask is not None:
        mask_bytes = mask.astype(np.uint8).tobytes()
        buf[pos:pos + len(mask_bytes)] = mask_bytes
        pos += len(mask_bytes)

    # Pack kv_responses: Dict[int, KVResponse]
    if kv_responses is not None:
        for tid, resp in kv_responses.items():
            status_int = _STATUS_TO_INT.get(resp.status, 5)
            has_return_mask = resp.return_mask is not None
            struct.pack_into("<qiB", buf, pos, tid, status_int, int(has_return_mask))
            pos += 13  # 8 + 4 + 1

            if has_return_mask:
                rm = resp.return_mask
                if isinstance(rm, list):
                    # Batched: list of np.ndarray
                    struct.pack_into("<Bi", buf, pos, 1, len(rm))
                    pos += 5
                    for arr in rm:
                        arr_uint8 = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
                        struct.pack_into("<i", buf, pos, len(arr_uint8))
                        pos += 4
                        buf[pos:pos + arr_uint8.nbytes] = arr_uint8.tobytes()
                        pos += arr_uint8.nbytes
                else:
                    # Single np.ndarray
                    rm_uint8 = rm.astype(np.uint8) if rm.dtype != np.uint8 else rm
                    struct.pack_into("<Bi", buf, pos, 0, len(rm_uint8))
                    pos += 5
                    buf[pos:pos + rm_uint8.nbytes] = rm_uint8.tobytes()
                    pos += rm_uint8.nbytes

    # Pack error message
    if error_len > 0:
        buf[pos:pos + error_len] = error_bytes
        pos += error_len

    return pos - offset


def unpack_response(buf, offset: int) -> dict:
    """Unpack response from shared memory buffer."""
    (status_code, task_id, is_ready, has_mask,
     mask_len, n_kv, error_len) = struct.unpack_from(RESP_HEADER_FMT, buf, offset)
    pos = offset + RESP_HEADER_SIZE

    result = {
        "status_code": status_code,
        "task_id": task_id,
        "is_ready": bool(is_ready),
        "mask": None,
        "kv_responses": None,
        "error_msg": None,
    }

    # Unpack mask
    if has_mask and mask_len > 0:
        result["mask"] = np.frombuffer(buf[pos:pos + mask_len], dtype=np.uint8).copy()
        pos += mask_len

    # Unpack kv_responses
    if n_kv > 0:
        kv_responses = {}
        for _ in range(n_kv):
            tid, status_int, has_rm = struct.unpack_from("<qiB", buf, pos)
            pos += 13
            status_enum = _INT_TO_STATUS.get(status_int, KVResponseStatus.FAILED)
            return_mask = None

            if has_rm:
                is_list, count = struct.unpack_from("<Bi", buf, pos)
                pos += 5
                if is_list:
                    mask_list = []
                    for _ in range(count):
                        arr_len = struct.unpack_from("<i", buf, pos)[0]
                        pos += 4
                        mask_list.append(
                            np.frombuffer(buf[pos:pos + arr_len], dtype=np.uint8).copy())
                        pos += arr_len
                    return_mask = mask_list
                else:
                    return_mask = np.frombuffer(buf[pos:pos + count], dtype=np.uint8).copy()
                    pos += count

            kv_responses[tid] = KVResponse(
                status=status_enum,
                task_id=tid,
                return_mask=return_mask,
            )
        result["kv_responses"] = kv_responses

    # Unpack error message
    if error_len > 0:
        raw = buf[pos:pos + error_len]
        result["error_msg"] = raw.decode("utf-8") if isinstance(raw, bytes) else bytes(raw).decode("utf-8")
        pos += error_len

    return result


# ── ShmChannel ──────────────────────────────────────────────────────────

class ShmChannel:
    """
    A shared memory channel between one client and the server.

    Provides:
    - Async ring buffer: client enqueues, server dequeues (SPSC, lock-free)
    - Sync slot: client writes request + spins on response (one at a time)
    - All notification via futex on shared memory (no eventfd needed)
    """

    def __init__(self, server_id: str, client_id: int, create: bool = False):
        self.client_id = client_id
        safe_id = _sanitize_server_id(server_id)
        self.shm_name = f"flexkv_shm_ch_{safe_id}_{client_id}"
        self.shm_path = f"/dev/shm/{self.shm_name}"

        if create:
            fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, TOTAL_SHM_SIZE)
            self.buf = mmap.mmap(fd, TOTAL_SHM_SIZE)
            os.close(fd)
            self.buf[:HEADER_SIZE] = b'\x00' * HEADER_SIZE
        else:
            fd = os.open(self.shm_path, os.O_RDWR)
            self.buf = mmap.mmap(fd, TOTAL_SHM_SIZE)
            os.close(fd)

        self._base_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))

    @property
    def _sync_req_flag_addr(self) -> int:
        return self._base_addr + OFF_SYNC_REQ_FLAG

    @property
    def _sync_resp_flag_addr(self) -> int:
        return self._base_addr + OFF_SYNC_RESP_FLAG

    # ── Async ring buffer ───────────────────────────────────────────────

    def _get_write_pos(self) -> int:
        return struct.unpack_from("<Q", self.buf, OFF_ASYNC_WRITE)[0]

    def _set_write_pos(self, val: int) -> None:
        struct.pack_into("<Q", self.buf, OFF_ASYNC_WRITE, val)

    def _get_read_pos(self) -> int:
        return struct.unpack_from("<Q", self.buf, OFF_ASYNC_READ)[0]

    def _set_read_pos(self, val: int) -> None:
        struct.pack_into("<Q", self.buf, OFF_ASYNC_READ, val)

    def async_send(self, msg_type: int, dp_client_id: int, task_id: int = -1,
                   **kwargs) -> None:
        """Enqueue an async message into the ring buffer."""
        wp = self._get_write_pos()
        rp = self._get_read_pos()

        next_wp = (wp + 1) & (ASYNC_RING_SLOTS - 1)
        spin = 0
        while next_wp == rp:
            spin += 1
            if spin > 1000000:
                raise RuntimeError("ShmChannel async ring buffer full after spin")
            rp = self._get_read_pos()

        slot_offset = ASYNC_RING_OFFSET + wp * ASYNC_SLOT_SIZE
        pack_request(self.buf, slot_offset, msg_type, dp_client_id, task_id, **kwargs)
        self._set_write_pos(next_wp)

    def async_recv(self) -> Optional[dict]:
        """Dequeue one async message (server side)."""
        rp = self._get_read_pos()
        wp = self._get_write_pos()
        if rp == wp:
            return None

        slot_offset = ASYNC_RING_OFFSET + rp * ASYNC_SLOT_SIZE
        msg = unpack_request(self.buf, slot_offset)

        next_rp = (rp + 1) & (ASYNC_RING_SLOTS - 1)
        self._set_read_pos(next_rp)
        return msg

    # ── Sync request/response ───────────────────────────────────────────

    def sync_send_request(self, msg_type: int, dp_client_id: int,
                          task_id: int = -1, **kwargs) -> None:
        """Write a sync request (client side). Must call sync_wait_response after."""
        # Clear response flag
        struct.pack_into("<i", self.buf, OFF_SYNC_RESP_FLAG, 0)
        # Write request
        pack_request(self.buf, SYNC_REQ_OFFSET, msg_type, dp_client_id, task_id, **kwargs)
        # Set request flag (signals server)
        struct.pack_into("<i", self.buf, OFF_SYNC_REQ_FLAG, 1)

    def sync_wait_response(self, spin_iters: int = 10000) -> dict:
        """Spin-wait for sync response (client side)."""
        resp_addr = self._sync_resp_flag_addr
        for _ in range(spin_iters):
            if struct.unpack_from("<i", self.buf, OFF_SYNC_RESP_FLAG)[0] != 0:
                break
        else:
            while struct.unpack_from("<i", self.buf, OFF_SYNC_RESP_FLAG)[0] == 0:
                futex_wait(resp_addr, 0)

        resp = unpack_response(self.buf, SYNC_RESP_OFFSET)
        # Only clear resp flag; server already cleared req flag after reading
        struct.pack_into("<i", self.buf, OFF_SYNC_RESP_FLAG, 0)
        return resp

    def sync_request(self, msg_type: int, dp_client_id: int,
                     task_id: int = -1, spin_iters: int = 10000,
                     **kwargs) -> dict:
        """Send sync request and wait for response (client side convenience)."""
        self.sync_send_request(msg_type, dp_client_id, task_id, **kwargs)
        return self.sync_wait_response(spin_iters)

    def check_sync_request(self) -> Optional[dict]:
        """Non-blocking check for pending sync request (server side)."""
        if struct.unpack_from("<i", self.buf, OFF_SYNC_REQ_FLAG)[0] == 0:
            return None
        req = unpack_request(self.buf, SYNC_REQ_OFFSET)
        # Clear request flag so we don't re-process on next loop iteration
        struct.pack_into("<i", self.buf, OFF_SYNC_REQ_FLAG, 0)
        return req

    def send_sync_response(self, **kwargs) -> None:
        """Write response and signal client (server side)."""
        pack_response(self.buf, SYNC_RESP_OFFSET, **kwargs)
        struct.pack_into("<i", self.buf, OFF_SYNC_RESP_FLAG, 1)
        futex_wake(self._sync_resp_flag_addr)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def close(self) -> None:
        if self.buf is not None:
            self.buf.close()
            self.buf = None

    def unlink(self) -> None:
        try:
            os.unlink(self.shm_path)
        except FileNotFoundError:
            pass
