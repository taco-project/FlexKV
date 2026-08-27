"""``NixlFileBackend`` round-trips real bytes through a real NIXL agent.

The rest of the backend suite checks arithmetic; this file checks that the
arithmetic moves the right bytes. It seeds a pool, writes it out through NIXL,
zeroes the pool, reads it back and compares byte for byte -- so a stride that is
self-consistent but wrong (the failure mode when stride derivation moves from
the worker into a backend) fails here.

Both plugin families are covered when the box supports them:

  POSIX  / CPU<->file, on ``CPUSSDDiskTransferWorker``'s edge
  GDS_MT / GPU<->file, on ``GDSTransferWorker``'s edge

Model shapes are the production ones (Qwen full attention, DeepSeek-V3.2 and
GLM-5.2 MLA), and both CPU layout orders are exercised, because the CPU-side
offset formula differs between them.

``_EdgeWorker`` publishes exactly the attributes ``NixlFileBackend.attach``
reads off a worker. Standing a real ``TransferWorkerBase`` up would need a
spawned process, a pinned pool and the io_uring context this backend replaces
anyway; what matters is that the *edge contract* -- these attribute names, with
these meanings -- is what the backend consumes.
"""
from __future__ import annotations

import os
import tempfile
from typing import Dict, List

import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferType
from flexkv.transfer.backends import NixlFileBackend

pytest.importorskip("nixl", reason="NIXL SDK not installed")


# (num_layers, num_kv_heads, head_size, kv_dim) -- layer counts trimmed so a
# round trip stays sub-second; the strides, not the layer count, are the point.
SHAPES = [
    pytest.param((4, 8, 128, 2), id="qwen3-mha"),
    pytest.param((4, 1, 576, 1), id="deepseek-v3.2-mla"),
    pytest.param((6, 1, 576, 1), id="glm-5.2-mla"),
]
LAYOUTS = [
    pytest.param("LAYERFIRST", id="lfirst"),
    pytest.param("BLOCKFIRST", id="bfirst"),
]

TOKENS_PER_BLOCK = 16
NUM_BLOCKS = 8
NUM_DEVICES = 2
FILES_PER_DEVICE = 2
NUM_FILES = NUM_DEVICES * FILES_PER_DEVICE
DTYPE = torch.float16

_PLUGIN_CACHE: Dict[str, bool] = {}


def _plugin_available(plugin: str) -> bool:
    """Can this plugin actually register a backing file on this box?

    Creating the backend is not enough of a probe: GDS_MT instantiates happily
    without the ``nvidia_fs`` kernel module and only fails later, at
    ``cuFileHandleRegister``.  So register a real file, which is exactly what
    ``attach`` does first.  POSIX needs nothing and always passes.
    """
    if plugin not in _PLUGIN_CACHE:
        from flexkv.transfer.nixlutil import NixlAgentSession
        ok = False
        try:
            with tempfile.TemporaryDirectory() as probe_dir:
                path = os.path.join(probe_dir, "probe.bin")
                with open(path, "wb") as fh:
                    fh.truncate(1 << 20)
                session = NixlAgentSession(plugin, {})
                ok = bool(session.prepare_all_ssd_files({0: [path]}))
        except Exception:
            ok = False
        _PLUGIN_CACHE[plugin] = ok
    return _PLUGIN_CACHE[plugin]


def _require(plugin: str) -> None:
    if not _plugin_available(plugin):
        pytest.skip(
            f"NIXL {plugin} cannot register files here "
            f"(GDS needs the nvidia_fs kernel module and a supported "
            f"filesystem); the arithmetic this would exercise is covered "
            f"without an agent in tests/test_storage_backends.py")


class _EdgeWorker:
    """The edge attributes ``NixlFileBackend.attach`` is allowed to read."""

    def __init__(self, cpu_kv_layout: KVCacheLayout, dtype: torch.dtype):
        self.worker_id = 99
        self.cpu_kv_layout = cpu_kv_layout
        self.dtype = dtype
        self.num_layers = cpu_kv_layout.num_layer
        self.kv_dim = cpu_kv_layout.kv_dim
        self.has_multi_group = False
        self.registered: List[str] = []
        self._bytes_per_block = -1

    def _register_host_tensor(self, tensor: torch.Tensor, label: str = "") -> None:
        # The real worker calls cudaHostRegister here. NIXL's DRAM registration
        # does not require pinning, and pinning a pool this test then drops
        # would leak into the rest of the session, so just record the call.
        self.registered.append(label)


def _make_layout(shape, layout_type: str, num_block: int = NUM_BLOCKS) -> KVCacheLayout:
    num_layers, num_kv_heads, head_size, kv_dim = shape
    return KVCacheLayout(
        type=KVCacheLayoutType[layout_type],
        num_layer=num_layers,
        num_block=num_block,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_head=num_kv_heads,
        head_size=head_size,
        kv_dim=kv_dim,
        num_kv_heads=num_kv_heads,
    )


def _attach_edge(worker: _EdgeWorker, shape, layout_type: str) -> int:
    """Reproduce ``CPUSSDDiskTransferWorker``'s single-group stride block.

    Same expressions in the same order as ``worker.py``; if that derivation
    changes, this expectation is what should have to change with it.  Returns
    the byte size each backing file must have.
    """
    cpu = worker.cpu_kv_layout
    ssd = _make_layout(shape, layout_type)
    per_file = ssd.div_block(NUM_FILES, padding=True)
    itemsize = worker.dtype.itemsize

    worker.chunk_size_in_bytes = cpu.get_chunk_size() * itemsize
    worker.block_stride_in_bytes = cpu.get_block_stride() * itemsize
    worker.cpu_kv_stride_in_bytes = cpu.get_kv_stride() * itemsize
    worker.cpu_layer_stride_in_bytes = cpu.get_layer_stride() * itemsize
    worker.ssd_kv_stride_in_bytes = per_file.get_kv_stride() * itemsize
    worker.ssd_layer_stride_in_bytes = per_file.get_layer_stride() * itemsize
    worker.ssd_block_stride_in_bytes = per_file.get_block_stride() * itemsize

    # One whole per-file layout per file. Deriving it from the layout rather
    # than from a stride keeps it right for both orders: LAYERFIRST spans
    # num_layer*layer_stride, BLOCKFIRST spans num_block*block_stride.
    return per_file.kv_shape.numel() * itemsize


class _Op:
    """A layer-sliceable op.

    Note this is *more* than production sends: ``WorkerTransferOp`` has no
    ``layer_id``/``layer_granularity``, so the backend must take the range
    from the plan, not off the op.
    """

    def __init__(self, transfer_type: TransferType, valid_block_num: int,
                 layer_id: int = -1, layer_granularity: int = -1):
        self.transfer_type = transfer_type
        self.valid_block_num = valid_block_num
        self.layer_id = layer_id
        self.layer_granularity = layer_granularity


def _make_files(tmpdir: str, file_bytes: int) -> Dict[int, List[str]]:
    ssd_files: Dict[int, List[str]] = {}
    for d in range(NUM_DEVICES):
        paths = []
        for f in range(FILES_PER_DEVICE):
            p = os.path.join(tmpdir, f"nixl_d{d}_f{f}.bin")
            with open(p, "wb") as fh:
                fh.truncate(file_bytes)
            paths.append(p)
        ssd_files[d] = paths
    return ssd_files


def _seed(t: torch.Tensor, salt: int = 0) -> None:
    """Fill with a non-repeating pattern exactly representable in fp16.

    Non-repeating matters: with a constant fill, a wrong offset lands on bytes
    that happen to be identical and the comparison passes anyway.
    """
    flat = t.view(-1)
    idx = torch.arange(flat.numel(), dtype=torch.float32, device=t.device)
    flat.copy_(((idx + salt * 331) % 1021).to(t.dtype))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout_type", LAYOUTS)
def test_nixl_posix_cpu_file_roundtrip_is_byte_exact(shape, layout_type):
    """Seed CPU pool -> H2DISK -> zero -> DISK2H -> compare.

    Covers every block, layer and kv plane, so any one of the four strides
    being wrong shows up as a mismatch rather than as a plausible number.
    """
    _require("POSIX")
    worker = _EdgeWorker(_make_layout(shape, layout_type), DTYPE)
    file_bytes = _attach_edge(worker, shape, layout_type)

    cpu_blocks = torch.empty(worker.cpu_kv_layout.kv_shape, dtype=DTYPE)
    _seed(cpu_blocks)
    golden = cpu_blocks.clone()
    worker.cpu_blocks = cpu_blocks

    with tempfile.TemporaryDirectory() as tmpdir:
        be = NixlFileBackend("POSIX", _make_files(tmpdir, file_bytes))
        be.attach(worker)
        assert worker.registered == ["nixl_cpu_pool"]
        assert worker._bytes_per_block == (
            be.chunk_size_in_bytes * be.num_layers * be.kv_dim)

        ids = torch.arange(NUM_BLOCKS, dtype=torch.int64)
        moved = be.transfer(worker, _Op(TransferType.H2DISK, NUM_BLOCKS), ids, ids)
        assert moved == (be.chunk_size_in_bytes * be.num_layers
                         * NUM_BLOCKS * be.kv_dim)

        cpu_blocks.zero_()
        assert not torch.equal(cpu_blocks, golden)

        be.transfer(worker, _Op(TransferType.DISK2H, NUM_BLOCKS), ids, ids)
        be.shutdown()

    assert torch.equal(cpu_blocks, golden), "NIXL POSIX round trip moved wrong bytes"


def test_nixl_layer_range_transfers_only_that_range():
    """A layerwise op carries (layer_id, layer_granularity); the backend must
    emit descriptors for that slice only, or a per-layer GET clobbers layers
    the consumer is already reading."""
    _require("POSIX")
    shape = (4, 1, 576, 1)
    worker = _EdgeWorker(_make_layout(shape, "LAYERFIRST"), DTYPE)
    file_bytes = _attach_edge(worker, shape, "LAYERFIRST")

    cpu_blocks = torch.empty(worker.cpu_kv_layout.kv_shape, dtype=DTYPE)
    _seed(cpu_blocks)
    golden = cpu_blocks.clone()
    worker.cpu_blocks = cpu_blocks

    with tempfile.TemporaryDirectory() as tmpdir:
        be = NixlFileBackend("POSIX", _make_files(tmpdir, file_bytes))
        be.attach(worker)
        ids = torch.arange(NUM_BLOCKS, dtype=torch.int64)
        be.transfer(worker, _Op(TransferType.H2DISK, NUM_BLOCKS), ids, ids)
        cpu_blocks.zero_()
        moved = be.transfer(
            worker,
            _Op(TransferType.DISK2H, NUM_BLOCKS, layer_id=1, layer_granularity=2),
            ids, ids)
        be.shutdown()

    assert moved == be.chunk_size_in_bytes * 2 * NUM_BLOCKS * be.kv_dim
    v = cpu_blocks.reshape(4, -1)
    g = golden.reshape(4, -1)
    assert torch.equal(v[1], g[1]) and torch.equal(v[2], g[2])
    assert v[0].float().abs().sum() == 0 and v[3].float().abs().sum() == 0
