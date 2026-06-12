from dataclasses import dataclass
import pickle
import struct
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.request import KVResponse, KVResponseStatus
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.server.request import (
    CancelTaskRequest,
    GetMatchRequest,
    GetRequest,
    IsReadyRequest,
    LaunchTaskRequest,
    PutRequest,
    RegisterTPClientRequest,
    StartRequest,
    ShutdownRequest,
    TryWaitRequest,
    WaitRequest,
)


MAGIC = b"FKAO"
VERSION = 1

OP_REGISTER_DP_CLIENT = 1
OP_IS_READY = 2
OP_PUT = 3
OP_GET = 4
OP_GET_MATCH = 5
OP_LAUNCH_TASKS = 6
OP_CANCEL_TASK = 7
OP_WAIT = 8
OP_TRY_WAIT = 9
OP_SHUTDOWN = 10
OP_START = 11
OP_REGISTER_TP_CLIENT = 12

RESP_IS_READY = 101
RESP_GET_MATCH = 102
RESP_WAIT = 103
RESP_TRY_WAIT = 104
RESP_ERROR = 127


STATUS_TO_CODE = {
    KVResponseStatus.SUCCESS: 0,
    KVResponseStatus.NOTFOUND: 1,
    KVResponseStatus.UNREADY: 2,
    KVResponseStatus.TIMEOUT: 3,
    KVResponseStatus.CANCELLED: 4,
    KVResponseStatus.FAILED: 5,
}

CODE_TO_LAYOUT_TYPE = {
    0: KVCacheLayoutType.LAYERFIRST,
    1: KVCacheLayoutType.BLOCKFIRST,
}

CODE_TO_DTYPE = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    3: torch.int64,
    4: torch.int32,
    5: torch.int16,
    6: torch.int8,
    7: torch.uint8,
    8: torch.bool,
}


@dataclass
class RegisterRawDPClientRequest:
    dp_client_id: int
    client_recv_port: str
    tp_size: int = 1


@dataclass
class _RawTensorHandle:
    ipc_handle: bytes
    shape: tuple[int, ...]
    dtype_code: int
    offset: int

@dataclass
class RecsysKVCacheLayout(KVCacheLayout):
    """Recsys KV layout: [layer, block, kv, token, head, dim]."""

    def __post_init__(self) -> None:
        self._kv_shape = torch.Size(
            [
                self.num_layer,
                self.num_block,
                self._kv_dim,
                self.tokens_per_block,
                self.num_head,
                self.head_size,
            ]
        )

    def get_layer_stride(self) -> int:
        return self.kv_shape[1:].numel()

    def get_block_stride(self) -> int:
        return self.kv_shape[2:].numel()

    def get_kv_stride(self) -> int:
        return self.kv_shape[3:].numel()


def _read_tensor_handle(reader: "BufferReader") -> _RawTensorHandle:
    handle_size = reader.read_i32()
    ipc_handle = bytes(reader._read(handle_size))
    shape_rank = reader.read_i32()
    shape = tuple(reader.read_i64() for _ in range(shape_rank))
    dtype_code = reader.read_i32()
    offset = reader.read_i64()
    return _RawTensorHandle(ipc_handle=ipc_handle, shape=shape, dtype_code=dtype_code, offset=offset)


def _read_layout(reader: "BufferReader") -> KVCacheLayout:
    return RecsysKVCacheLayout(
        type=CODE_TO_LAYOUT_TYPE[reader.read_i32()],
        num_layer=reader.read_i32(),
        num_block=reader.read_i32(),
        tokens_per_block=reader.read_i32(),
        num_head=reader.read_i32(),
        head_size=reader.read_i32(),
        is_mla=reader.read_bool(),
    )


def _build_tensor_shared_handle(raw_handle: _RawTensorHandle, device_id: int) -> TensorSharedHandle:
    return TensorSharedHandle(
        raw_handle.ipc_handle,
        device_id=device_id,
        tensor_shape=raw_handle.shape,
        tensor_dtype=CODE_TO_DTYPE[raw_handle.dtype_code],
        offset=raw_handle.offset,
    )


class BufferReader:
    def __init__(self, data: bytes):
        self._view = memoryview(data)
        self._offset = 0

    def _read(self, size: int) -> memoryview:
        if self._offset + size > len(self._view):
            raise ValueError("Unexpected end of FlexKV AOTI payload")
        start = self._offset
        self._offset += size
        return self._view[start:self._offset]

    def read_u8(self) -> int:
        return struct.unpack_from("<B", self._read(1))[0]

    def read_i32(self) -> int:
        return struct.unpack_from("<i", self._read(4))[0]

    def read_i64(self) -> int:
        return struct.unpack_from("<q", self._read(8))[0]

    def read_f64(self) -> float:
        return struct.unpack_from("<d", self._read(8))[0]

    def read_bool(self) -> bool:
        return bool(self.read_u8())

    def read_string(self) -> str:
        size = self.read_i32()
        return bytes(self._read(size)).decode("utf-8")

    def read_string_list(self) -> List[str]:
        return [self.read_string() for _ in range(self.read_i32())]

    def read_int64_array(self) -> np.ndarray:
        size = self.read_i32()
        if size == 0:
            return np.empty((0,), dtype=np.int64)
        return np.frombuffer(self._read(size * 8), dtype=np.int64).copy()

    def read_bool_array(self) -> np.ndarray:
        size = self.read_i32()
        if size == 0:
            return np.empty((0,), dtype=np.bool_)
        return np.frombuffer(self._read(size), dtype=np.uint8).astype(np.bool_, copy=False)

    def read_optional_bool_array(self) -> Optional[np.ndarray]:
        if not self.read_bool():
            return None
        return self.read_bool_array()

    def read_int64_array_list(self) -> List[np.ndarray]:
        return [self.read_int64_array() for _ in range(self.read_i32())]


class BufferWriter:
    def __init__(self):
        self._chunks: List[bytes] = []

    def write_u8(self, value: int) -> None:
        self._chunks.append(struct.pack("<B", value))

    def write_i32(self, value: int) -> None:
        self._chunks.append(struct.pack("<i", value))

    def write_i64(self, value: int) -> None:
        self._chunks.append(struct.pack("<q", value))

    def write_f64(self, value: float) -> None:
        self._chunks.append(struct.pack("<d", value))

    def write_bool(self, value: bool) -> None:
        self.write_u8(1 if value else 0)

    def write_string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.write_i32(len(encoded))
        self._chunks.append(encoded)

    def write_string_list(self, values: List[str]) -> None:
        self.write_i32(len(values))
        for value in values:
            self.write_string(value)

    def write_int64_array(self, array: np.ndarray) -> None:
        flat = np.asarray(array, dtype=np.int64).reshape(-1)
        self.write_i32(int(flat.size))
        self._chunks.append(flat.tobytes(order="C"))

    def write_bool_array(self, array: np.ndarray) -> None:
        flat = np.asarray(array, dtype=np.bool_).reshape(-1)
        self.write_i32(int(flat.size))
        self._chunks.append(flat.astype(np.uint8, copy=False).tobytes(order="C"))

    def to_bytes(self) -> bytes:
        return b"".join(self._chunks)


def _header(opcode: int) -> BufferWriter:
    writer = BufferWriter()
    writer._chunks.append(MAGIC)
    writer.write_u8(VERSION)
    writer.write_u8(opcode)
    return writer


def is_raw_message(frame: bytes) -> bool:
    return len(frame) >= 6 and frame[:4] == MAGIC and frame[4] == VERSION


def decode_request(frame: bytes):
    if not is_raw_message(frame):
        return pickle.loads(frame), False

    reader = BufferReader(frame)
    reader._read(4)
    reader.read_u8()
    opcode = reader.read_u8()

    if opcode == OP_REGISTER_DP_CLIENT:
        return RegisterRawDPClientRequest(
            dp_client_id=reader.read_i32(),
            client_recv_port=reader.read_string(),
            tp_size=reader.read_i32(),
        ), True

    if opcode == OP_IS_READY:
        return IsReadyRequest(dp_client_id=reader.read_i32()), True

    if opcode == OP_START:
        return StartRequest(dp_client_id=reader.read_i32()), True

    if opcode == OP_PUT:
        return PutRequest(
            dp_client_id=reader.read_i32(),
            token_ids=reader.read_int64_array(),
            slot_mapping=reader.read_int64_array(),
            token_mask=reader.read_optional_bool_array(),
            task_id=reader.read_i64(),
            namespace=reader.read_string_list(),
        ), True

    if opcode == OP_GET:
        return GetRequest(
            dp_client_id=reader.read_i32(),
            token_ids=reader.read_int64_array(),
            slot_mapping=reader.read_int64_array(),
            token_mask=reader.read_optional_bool_array(),
            task_id=reader.read_i64(),
            layer_granularity=reader.read_i32(),
            namespace=reader.read_string_list(),
        ), True

    if opcode == OP_GET_MATCH:
        return GetMatchRequest(
            dp_client_id=reader.read_i32(),
            token_ids=reader.read_int64_array(),
            token_mask=reader.read_optional_bool_array(),
            layer_granularity=reader.read_i32(),
            task_id=reader.read_i64(),
            namespace=reader.read_string_list(),
        ), True

    if opcode == OP_LAUNCH_TASKS:
        dp_client_id = reader.read_i32()
        task_count = reader.read_i32()
        task_ids = [reader.read_i64() for _ in range(task_count)]
        slot_mappings = reader.read_int64_array_list()
        as_batch = reader.read_bool()
        batch_id = reader.read_i64()
        return LaunchTaskRequest(dp_client_id, task_ids, slot_mappings, as_batch, batch_id), True

    if opcode == OP_CANCEL_TASK:
        dp_client_id = reader.read_i32()
        task_count = reader.read_i32()
        return CancelTaskRequest(dp_client_id, [reader.read_i64() for _ in range(task_count)]), True

    if opcode == OP_WAIT:
        dp_client_id = reader.read_i32()
        task_count = reader.read_i32()
        wait_task_ids = [reader.read_i64() for _ in range(task_count)]
        wait_timeout = reader.read_f64()
        completely = reader.read_bool()
        return WaitRequest(dp_client_id, None, wait_task_ids, wait_timeout, completely), True

    if opcode == OP_TRY_WAIT:
        dp_client_id = reader.read_i32()
        task_count = reader.read_i32()
        return TryWaitRequest(dp_client_id, None, [reader.read_i64() for _ in range(task_count)]), True

    if opcode == OP_SHUTDOWN:
        return ShutdownRequest(dp_client_id=reader.read_i32()), True

    if opcode == OP_REGISTER_TP_CLIENT:
        dp_client_id = reader.read_i32()
        device_id = reader.read_i32()
        handle_count = reader.read_i32()
        raw_handles = [_read_tensor_handle(reader) for _ in range(handle_count)]
        gpu_layout = _read_layout(reader)
        handles = [_build_tensor_shared_handle(raw_handle, device_id) for raw_handle in raw_handles]
        return RegisterTPClientRequest(dp_client_id, device_id, handles, gpu_layout), True

    raise ValueError(f"Unknown FlexKV AOTI opcode: {opcode}")


def encode_is_ready_response(is_ready: bool) -> bytes:
    writer = _header(RESP_IS_READY)
    writer.write_bool(is_ready)
    return writer.to_bytes()


def encode_get_match_response(task_id: int, mask: Union[np.ndarray, Dict[int, np.ndarray]]) -> bytes:
    writer = _header(RESP_GET_MATCH)
    writer.write_i64(task_id)
    if isinstance(mask, dict):
        if mask:
            writer.write_bool_array(next(iter(mask.values())))
        else:
            writer.write_bool_array(np.empty((0,), dtype=np.bool_))
    else:
        writer.write_bool_array(mask)
    return writer.to_bytes()


def encode_wait_response(opcode: int, kv_responses: Dict[int, KVResponse]) -> bytes:
    writer = _header(opcode)
    writer.write_i32(len(kv_responses))
    for task_id, response in kv_responses.items():
        writer.write_i64(task_id)
        writer.write_u8(STATUS_TO_CODE[response.status])
    return writer.to_bytes()


def encode_error_response(message: str) -> bytes:
    writer = _header(RESP_ERROR)
    writer.write_string(message)
    return writer.to_bytes()