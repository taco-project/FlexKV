"""WorkerTransferOp's arrays no longer travel through pickle's numpy path.

``__getstate__`` swaps every small ndarray for a ``(dtype_str, bytes)`` pair,
because pickling a numpy array costs ~4.8us regardless of its size and each op
carries two to eight of them. That is a pure encoding change, so the only thing
that matters is that it is exactly invertible -- and nothing raises if it is
not. A dtype that silently changes int64 -> float64 would reinterpret the same
bytes as garbage block ids and index the wrong KV blocks; a read-only decoded
array would make ``torch.from_numpy`` warn and hand back a tensor whose writes
are undefined.
"""
import pickle

import numpy as np
import pytest

from flexkv.common.transfer import TransferOp, TransferType
from flexkv.transfer.worker_op import (
    _ARRAY_ENCODE_MAX_ELEMS,
    _LAYERWISE_ARRAY_FIELDS,
    WorkerTransferOp,
    _decode_array,
    _encode_array,
)

P = pickle.HIGHEST_PROTOCOL


def _roundtrip(obj):
    return pickle.loads(pickle.dumps(obj, protocol=P))


def _make_op(n=8, ttype=TransferType.DISK2H, **kwargs):
    src = np.arange(n, dtype=np.int64)
    dst = np.arange(n, dtype=np.int64) + 1000
    return TransferOp(graph_id=0, transfer_type=ttype,
                      src_block_ids=src, dst_block_ids=dst, **kwargs)


# --------------------------------------------------------------------------
# the encoder in isolation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 8, 4095, _ARRAY_ENCODE_MAX_ELEMS,
                               _ARRAY_ENCODE_MAX_ELEMS + 1, 20000])
def test_encode_decode_is_exactly_invertible(n):
    """Both sides of the size threshold, since they take different branches."""
    arr = (np.arange(n, dtype=np.int64) * 7919) % 65521
    out = _decode_array(_roundtrip(_encode_array(arr)))
    assert out.dtype == np.int64
    assert np.array_equal(out, arr)


# --------------------------------------------------------------------------
# layerwise
# --------------------------------------------------------------------------

def test_layerwise_array_field_list_is_complete():
    """_LAYERWISE_ARRAY_FIELDS is derived from the annotations by suffix. A
    field that stops matching would silently keep the slow path; one that is
    renamed to match but is not an array would hit _encode_array and throw."""
    from flexkv.transfer.worker_op import WorkerLayerwiseTransferOp
    expected = {
        "src_block_ids_h2d", "dst_block_ids_h2d",
        "swa_src_block_ids_h2d", "swa_dst_block_ids_h2d",
    }
    assert set(_LAYERWISE_ARRAY_FIELDS) == expected
    annotated = set(WorkerLayerwiseTransferOp.__annotations__)
    assert expected <= annotated
    # The SSD read is a standalone DISK2H op now; no disk2h ids ride this one.
    assert not [f for f in annotated if f.endswith("_disk2h")]


def test_roundtrip_over_a_real_pipe():
    """The unit tests above use pickle directly; the production path is
    multiprocessing.Connection, which uses ForkingPickler. Confirm the custom
    __reduce__ hooks survive it and a real fork boundary."""
    import multiprocessing as mp

    def echo(conn):
        got = conn.recv()
        conn.send((got.src_block_ids.tolist(), str(got.src_block_ids.dtype),
                   got.src_block_ids.flags.writeable, got.valid_block_num))
        conn.close()

    op = _make_op(8)
    parent, child = mp.Pipe()
    proc = mp.get_context("fork").Process(target=echo, args=(child,))
    proc.start()
    child.close()
    parent.send(WorkerTransferOp(op))
    ids, dtype, writable, nblocks = parent.recv()
    parent.close()
    proc.join(timeout=10)

    assert ids == op.src_block_ids.tolist()
    assert dtype == "int64"
    assert writable
    assert nblocks == 8


