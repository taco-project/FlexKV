from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from flexkv.common.pool import PoolId
from flexkv.common.transfer import TransferOp, TransferType, LayerwiseTransferOp

# Pickling a numpy array costs ~4.8us REGARDLESS of size -- np.empty(0) and
# np.arange(64) both measure 4.79us, because the cost is the reduce protocol
# (import numpy.core.multiarray, look up _reconstruct, build the args tuple),
# not the buffer copy. Every WorkerTransferOp carries at least two arrays, so
# that is ~9.6us of fixed cost per op paid on the TransferEngine's dispatch
# thread, which is single-threaded and shared by every worker.
#
# Encoding the buffer as bytes bypasses the reduce protocol entirely: pickling
# `bytes` is a single opcode. Measured on the 8-id ops this path actually sees,
# a src+dst pair goes 6.4us -> 0.7us to encode and 3.7us -> 1.5us to decode.
#
# Above a few thousand ids the fixed cost is amortized and tobytes()' extra
# memcpy starts to dominate, so large arrays keep the ndarray path. The
# crossover measured here is ~6-8k ids; 4096 sits below it with margin, and the
# rows above it come out within noise of unmodified behaviour rather than
# regressing.
_ARRAY_ENCODE_MAX_ELEMS = 4096

# Shared placeholder for the slot-id path, whose block ids live in the op
# buffer instead. One module object rather than a fresh np.empty(0) per op:
# pickle memoizes the second reference when src and dst are the same object.
# Read-only because it IS shared -- an in-place write would otherwise be
# visible to every op ever built. Nothing can observe the flag across the pipe:
# it is size 0, so it encodes to bytes and decodes to a fresh writable array.
_EMPTY_INT64 = np.empty(0, dtype=np.int64)
_EMPTY_INT64.flags.writeable = False


def _encode_array(arr: Optional[np.ndarray]) -> Any:
    """Cheap wire form for a small numpy array. Inverse of _decode_array."""
    if arr is None:
        return None
    if arr.size > _ARRAY_ENCODE_MAX_ELEMS or arr.ndim != 1:
        return arr  # let pickle handle it: the fixed cost is amortized here
    # dtype travels with the buffer rather than being assumed int64: these
    # fields are int64 at every construction site today, but a decoder that
    # hardcodes it would silently reinterpret the bytes if one ever is not,
    # which is far worse than the 0.3us the dtype string costs.
    return (arr.dtype.str, arr.tobytes())


def _decode_array(value: Any) -> Any:
    if type(value) is not tuple:
        return value  # None, or a large array pickle handled directly
    dtype_str, buf = value
    # bytearray, not bytes: np.frombuffer over an immutable buffer returns a
    # read-only array, and torch.from_numpy on that warns and yields a tensor
    # whose writes are undefined behaviour. worker.get_transfer_block_ids()
    # feeds these straight into torch.from_numpy.
    return np.frombuffer(bytearray(buf), dtype=np.dtype(dtype_str))


# Pickling an Enum member costs 1.46us -- more than both block-id arrays
# combined after the encoding above -- because it goes through the class's
# __reduce_ex__ and re-looks-up the member. Its `.value` is a plain str at
# 0.55us, and ``TransferType(value)`` returns the interned member itself
# (verified for every member), so identity comparisons like
# ``op.transfer_type is TransferType.LAYERWISE`` in the worker still hold.
def _encode_ttype(t: TransferType) -> str:
    return t.value


def _decode_ttype(v: Any) -> TransferType:
    # Tolerate an already-decoded member so a mixed-version peer, or a state
    # dict that was never encoded, does not turn into a TypeError deep in the
    # worker's transfer dispatch.
    return v if type(v) is TransferType else TransferType(v)


# Same trade as _encode_ttype: PoolId is an IntEnum, but IntEnum inherits
# Enum's __reduce_ex__, so a member still costs 1.53us to pickle against a
# bare int's 0.26us. ``PoolId(int)`` returns the interned member, so
# ``op.pool_id is PoolId.SWA`` in the worker still holds after the trip.
def _encode_pool(p: PoolId) -> int:
    return int(p)


def _decode_pool(v: Any) -> PoolId:
    return v if type(v) is PoolId else PoolId(v)


@dataclass(frozen=True)
class WorkerTransferResult:
    """Worker-to-scheduler completion with optional per-block outcomes."""

    transfer_op_id: int
    block_results: Optional[Tuple[bool, ...]] = None


@dataclass
class WorkerTransferOp:
    transfer_op_id: int
    transfer_graph_id: int
    transfer_type: TransferType
    src_slot_id: int
    dst_slot_id: int
    valid_block_num: int
    src_block_ids: np.ndarray
    dst_block_ids: np.ndarray
    src_block_node_ids: Optional[np.ndarray]
    mooncake_store_block_hashes: Optional[np.ndarray] = None
    mooncake_store_swa_block_hashes: Optional[list] = None
    # Which KV pool this op's block ids index. Not a different kind of
    # transfer -- same direction, same layout, same worker -- so it selects a
    # pool binding inside the worker rather than a separate SWA worker
    # upstream. Encoded as an int on the wire: measured here, pickling an
    # IntEnum member costs 1.53us against 0.26us for the bare int, because
    # IntEnum inherits Enum's __reduce_ex__ and gets no discount for being an
    # int. Same trick, same reason as _encode_ttype.
    pool_id: PoolId = PoolId.FULL_KV
    prof_submitted_ns: int = 0

    @property
    def is_swa(self) -> bool:
        """Kept for the readers that predate ``pool_id``; see PoolId."""
        return self.pool_id is PoolId.SWA

    def __init__(self, transfer_op: TransferOp):
        self.transfer_op_id = transfer_op.op_id
        self.transfer_graph_id = transfer_op.graph_id
        self.transfer_type = transfer_op.transfer_type
        self.src_slot_id = transfer_op.src_slot_id
        self.dst_slot_id = transfer_op.dst_slot_id
        self.valid_block_num = transfer_op.valid_block_num
        # Always preserve optional src_block_node_ids from TransferOp
        self.src_block_node_ids = transfer_op.src_block_node_ids
        self.mooncake_store_block_hashes = transfer_op.mooncake_store_block_hashes
        self.mooncake_store_swa_block_hashes = transfer_op.mooncake_store_swa_block_hashes
        self.pool_id = getattr(transfer_op, "pool_id", PoolId.FULL_KV)

        if self.src_slot_id == -1 or self.dst_slot_id == -1:
            self.src_block_ids = transfer_op.src_block_ids
            self.dst_block_ids = transfer_op.dst_block_ids
        elif (transfer_op.mooncake_store_block_hashes is not None
              or transfer_op.mooncake_store_swa_block_hashes is not None):
            # Mooncake ops need block ids even when slot ids are set.
            self.src_block_ids = transfer_op.src_block_ids
            self.dst_block_ids = transfer_op.dst_block_ids
        else:
            # Both slot ids are set, so get_transfer_block_ids() reads the op
            # buffer and never looks at these. They exist only to keep the
            # field non-optional. int64 rather than np.empty(0)'s float64
            # default so anything that does inspect them (logging, a future
            # reader) sees the same dtype as every populated path.
            self.src_block_ids = _EMPTY_INT64
            self.dst_block_ids = _EMPTY_INT64

    # Pickled on the TransferEngine dispatch thread, once per op, for the trip
    # to the worker process. See _encode_array for why the arrays do not go
    # through pickle's numpy path.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["src_block_ids"] = _encode_array(state["src_block_ids"])
        state["dst_block_ids"] = _encode_array(state["dst_block_ids"])
        state["src_block_node_ids"] = _encode_array(state["src_block_node_ids"])
        state["mooncake_store_block_hashes"] = _encode_array(
            state["mooncake_store_block_hashes"])
        state["transfer_type"] = _encode_ttype(state["transfer_type"])
        state["pool_id"] = _encode_pool(state["pool_id"])
        return state

    def __setstate__(self, state: dict) -> None:
        state["src_block_ids"] = _decode_array(state["src_block_ids"])
        state["dst_block_ids"] = _decode_array(state["dst_block_ids"])
        state["src_block_node_ids"] = _decode_array(state["src_block_node_ids"])
        state["mooncake_store_block_hashes"] = _decode_array(
            state["mooncake_store_block_hashes"])
        state["transfer_type"] = _decode_ttype(state["transfer_type"])
        state["pool_id"] = _decode_pool(state["pool_id"])
        self.__dict__.update(state)


@dataclass
class WorkerLayerwiseTransferOp:
    transfer_op_id: int
    transfer_graph_id: int
    transfer_type: TransferType
    src_block_ids_h2d: np.ndarray
    dst_block_ids_h2d: np.ndarray
    # Always non-None: LayerwiseTransferOp normalizes missing SWA ids to empty
    # np.int64 arrays. Empty arrays signal cpp that this transfer carries no SWA.
    swa_src_block_ids_h2d: np.ndarray
    swa_dst_block_ids_h2d: np.ndarray
    counter_id: int  # Counter set index for triple buffering eventfd notification
    prof_submitted_ns: int = 0

    def __init__(self, transfer_op: LayerwiseTransferOp):
        self.transfer_op_id = transfer_op.op_id
        self.transfer_graph_id = transfer_op.graph_id
        assert transfer_op.transfer_type == TransferType.LAYERWISE
        self.transfer_type = transfer_op.transfer_type
        self.src_block_ids_h2d = transfer_op.src_block_ids_h2d
        self.dst_block_ids_h2d = transfer_op.dst_block_ids_h2d
        self.swa_src_block_ids_h2d = transfer_op.swa_src_block_ids_h2d
        self.swa_dst_block_ids_h2d = transfer_op.swa_dst_block_ids_h2d
        self.counter_id = transfer_op.counter_id

    # Four arrays here rather than WorkerTransferOp's two, so the fixed
    # per-array pickle cost is twice as bad.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for key in _LAYERWISE_ARRAY_FIELDS:
            state[key] = _encode_array(state[key])
        state["transfer_type"] = _encode_ttype(state["transfer_type"])
        return state

    def __setstate__(self, state: dict) -> None:
        for key in _LAYERWISE_ARRAY_FIELDS:
            state[key] = _decode_array(state[key])
        state["transfer_type"] = _decode_ttype(state["transfer_type"])
        self.__dict__.update(state)


# Declared after the class so a field rename that misses this tuple fails at
# import, not at the first layerwise transfer.
_LAYERWISE_ARRAY_FIELDS = tuple(
    f for f in WorkerLayerwiseTransferOp.__annotations__
    if f.endswith("_h2d")
)
assert len(_LAYERWISE_ARRAY_FIELDS) == 4, _LAYERWISE_ARRAY_FIELDS
