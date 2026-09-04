"""``MooncakeStoreBackend``: the arithmetic that moved when it stopped being a worker.

Demoting mooncake-store from a worker class to a ``StorageBackend`` moved real
key and pointer arithmetic across a file boundary, and none of it had a test
that named the backend -- the suite covered the *native* engines, so a backend
that attached with a subtly wrong block size would have gone green.

Two failure modes are pinned:

  * **block size** -- derived at attach time from the CPU layout. Getting it
    wrong reads or writes a neighbouring block's bytes rather than raising.
  * **partial failure** -- a batch where some keys land and some do not must
    fail the whole op. Publishing the survivors is worse than failing: a
    reader hits blocks whose peers were never written.

``FakeWorker`` plus ``make_geometry`` stand in for the worker's *edge*: the
``EdgeGeometry`` a backend is handed, built by the same expressions
``CPURemoteTransferWorker`` uses. That contract is what the refactor
introduced, so a worker that stops publishing one of these fails here rather
than in production.

``NixlFileBackend`` is covered by ``test_nixl_backend_e2e.py``, which round-trips
real bytes through a real agent.
"""
from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np
import pytest
import torch

from flexkv.common.config import LayerGroupSpec
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferType
from flexkv.transfer.backends import (
    MooncakeStoreBackend,
    NixlFileBackend,
    PcfsRemoteBackend,
    StorageBackend,
)
from flexkv.transfer.geometry import ChunkStrides, EdgeGeometry, HostSide

P = pickle.HIGHEST_PROTOCOL


# ---------------------------------------------------------------------------
# Model shapes. Real configs, so a geometry regression shows up at the scale
# it would in production rather than on a 4-layer toy.
# ---------------------------------------------------------------------------

# (num_layers, num_kv_heads_per_rank, head_size, kv_dim, tokens_per_block)
QWEN3_8B = (36, 8, 128, 2, 16)          # full attention, MHA/GQA
QWEN3_5_397B = (15, 2, 256, 2, 16)      # only the 15 full_attention layers cache
DEEPSEEK_V32 = (61, 1, 576, 1, 16)      # MLA: kv_lora_rank 512 + rope 64, kv_dim=1
GLM_5_2 = (78, 1, 576, 1, 16)           # GLM-5.2 main KV: same MLA shape, 78 layers

MODEL_SHAPES = [
    pytest.param(QWEN3_8B, id="qwen3-8b"),
    pytest.param(QWEN3_5_397B, id="qwen3.5-397b-full-layers"),
    pytest.param(DEEPSEEK_V32, id="deepseek-v3.2"),
    pytest.param(GLM_5_2, id="glm-5.2"),
]


def make_cpu_layout(shape, num_blocks: int, layout_type: str = "LAYERFIRST"):
    num_layers, num_kv_heads, head_size, kv_dim, tpb = shape
    return KVCacheLayout(
        type=KVCacheLayoutType[layout_type],
        num_layer=num_layers, num_block=num_blocks, tokens_per_block=tpb,
        num_head=num_kv_heads, head_size=head_size,
        kv_dim=kv_dim, num_kv_heads=num_kv_heads,
    )


# DSv4-flash: main KV bf16 uncompressed alongside an fp8 indexer compressed
# 128x, both attached to the same transformer layers. The two groups have
# different dtypes *and* different token counts per block, which is exactly
# why the block has no uniform (layer, kv) chunk and ``strides is None``.
DSV4_GROUPS = [
    LayerGroupSpec(num_layers=4, num_kv_heads=1, head_size=576,
                   layer_indices=[0, 1, 2, 3], dtype=torch.bfloat16,
                   compress_ratio=1),
    LayerGroupSpec(num_layers=4, num_kv_heads=1, head_size=128,
                   layer_indices=[0, 1, 2, 3], dtype=torch.uint8,
                   compress_ratio=128),
]


def make_multi_group_layout(num_blocks: int = 8, tp_size: int = 1):
    """A heterogeneous CPU layout: byte-flat blocks, no per-chunk strides.

    ``tokens_per_block=128`` so the 128x-compressed indexer group still stores
    a whole token per block; ``get_block_stride()`` returns BYTES here, which
    is the asymmetry the geometry has to carry correctly.
    """
    return KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=4, num_block=num_blocks, tokens_per_block=128,
        num_head=1, head_size=576, kv_dim=1, num_kv_heads=1,
        layer_groups=DSV4_GROUPS, tp_size=tp_size,
    )


def _hugepage_aligned(shape, dtype: torch.dtype, alignment: int):
    """A tensor of ``shape`` whose data pointer is ``alignment``-aligned.

    Returns ``(backing, view)``; the caller must keep ``backing`` alive, since
    the view does not own the storage.
    """
    nbytes = 1
    for dim in shape:
        nbytes *= int(dim)
    nbytes *= torch.empty(0, dtype=dtype).element_size()
    backing = torch.zeros(nbytes + alignment, dtype=torch.uint8)
    pad = (-backing.data_ptr()) % alignment
    view = backing[pad:pad + nbytes].view(dtype).view(shape)
    assert view.data_ptr() % alignment == 0
    return backing, view


# ---------------------------------------------------------------------------
# The edge a backend is allowed to read.
# ---------------------------------------------------------------------------


class FakeWorker:
    """Exactly the non-geometry worker surface ``attach`` may still read.

    Deliberately not a ``TransferWorkerBase``: the value of this stand-in is
    that touching anything else raises ``AttributeError``.  ``attach`` reaching
    for a private worker method would then fail here rather than silently
    couple the backend to one worker class.

    Everything that describes the *edge* now arrives as an ``EdgeGeometry``
    (see ``make_geometry``); what is left here is the worker itself.
    """

    def __init__(self, cpu_kv_layout: KVCacheLayout, dtype: torch.dtype,
                 cpu_blocks: Optional[torch.Tensor] = None):
        self.worker_id = 0
        self.cpu_kv_layout = cpu_kv_layout
        self.dtype = dtype
        self.cpu_blocks = cpu_blocks
        self.registered: List[str] = []
        # Only ``make_geometry`` reads this; a HugePage-backed pool maps more
        # than it holds, and the geometry is where that gets published.
        self.cpu_blocks_mapped_size: Optional[int] = None

    def _register_host_tensor(self, tensor: torch.Tensor, label: str = "") -> None:
        self.registered.append(label)


def make_geometry(worker: FakeWorker) -> EdgeGeometry:
    """The ``EdgeGeometry`` ``CPURemoteTransferWorker`` would publish.

    Same expressions in the same order as ``workers/remote.py``; a remote edge
    has no ``ssd`` or ``gpu`` side, and under a heterogeneous layout no
    per-chunk strides at all.
    """
    layout = worker.cpu_kv_layout
    dtype = worker.dtype
    itemsize = dtype.itemsize
    has_multi_group = layout.layer_groups is not None
    if has_multi_group:
        block_stride = int(layout.get_block_stride())
        strides = None
        bytes_per_block = block_stride
    else:
        block_stride = layout.get_block_stride() * itemsize
        chunk_bytes = layout.get_chunk_size() * itemsize
        strides = ChunkStrides(
            chunk_bytes=chunk_bytes,
            kv_stride=layout.get_kv_stride() * itemsize,
            layer_stride=layout.get_layer_stride() * itemsize,
            block_stride=block_stride,
        )
        bytes_per_block = chunk_bytes * layout.num_layer * layout.kv_dim
    blocks = worker.cpu_blocks
    layer_ptrs = (
        torch.tensor([blocks.data_ptr()], dtype=torch.int64)
        if blocks is not None else torch.tensor([0], dtype=torch.int64)
    )
    return EdgeGeometry(
        num_layers=layout.num_layer,
        kv_dim=layout.kv_dim,
        num_kv_heads=layout.num_kv_heads,
        dtype=dtype,
        has_multi_group=has_multi_group,
        bytes_per_block=bytes_per_block,
        cpu=HostSide(
            layout=layout,
            blocks=blocks,
            layer_ptrs=layer_ptrs,
            block_stride=block_stride,
            mapped_size=worker.cpu_blocks_mapped_size,
            strides=strides,
        ),
    )


class FakeOp:
    def __init__(self, transfer_type: TransferType, valid_block_num: int,
                 layer_id: int = -1, layer_granularity: int = -1,
                 block_hashes=None, swa_block_hashes=None,
                 src_block_node_ids=None):
        self.transfer_type = transfer_type
        # Only read when a backend reports a failure; the ids name which op
        # died, so they have to survive into the message.
        self.transfer_op_id = 0
        self.transfer_graph_id = 0
        self.valid_block_num = valid_block_num
        self.layer_id = layer_id
        self.layer_granularity = layer_granularity
        self.mooncake_store_block_hashes = block_hashes
        self.mooncake_store_swa_block_hashes = swa_block_hashes
        self.src_block_node_ids = src_block_node_ids


# ---------------------------------------------------------------------------
# The ABC contract itself
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Picklability: the spawn boundary
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PcfsRemoteBackend.attach arithmetic
#
# PCFS (libhifs_client_sdk) is not installed on any box we can test on, and
# builds without ``FLEXKV_ENABLE_CFS=1`` leave ``transfer_kv_blocks_remote``
# as ``None``, so no e2e run can reach this edge. That makes the geometry
# worth pinning here, and pinning it *against the kernel's addressing* rather
# than against the formula that produces it -- restating the formula would
# have re-asserted the LAYERFIRST-only bug rather than catching it.
#
# What the kernel does (csrc/pcfs/pcfs.cpp:304-343), for each layer i and each
# block b, is:
#
#     cpu = cpu_base + i*cpu_layer_stride + b*chunk        (+cpu_kv_stride for V)
#     rem =            i*remote_layer_stride + b*chunk     (+remote_kv_stride for V)
#     transfer ``chunk`` bytes
#
# ``chunk`` is both the block multiplier and the length, so the only layout it
# can address natively is LAYERFIRST. Under BLOCKFIRST a block's layers are
# contiguous and the backend must flatten the block to stay correct.
# ---------------------------------------------------------------------------


def _pcfs_expected_cpu_offsets(layout: KVCacheLayout, dtype: torch.dtype,
                               block_id: int) -> List[int]:
    """Byte offsets of every (layer, kv) chunk of ``block_id``, from the layout.

    An oracle derived from ``kv_shape`` alone -- deliberately independent of
    anything ``attach`` computes.
    """
    itemsize = dtype.itemsize
    chunk = layout.get_chunk_size() * itemsize
    offsets = []
    for layer in range(layout.num_layer):
        for kv in range(layout.kv_dim):
            if layout.type == KVCacheLayoutType.LAYERFIRST:
                # [layer, kv, block, ...]
                idx = ((layer * layout.kv_dim + kv) * layout.num_block
                       + block_id)
            else:  # BLOCKFIRST: [block, layer, kv, ...]
                idx = ((block_id * layout.num_layer + layer) * layout.kv_dim
                       + kv)
            offsets.append(idx * chunk)
    return sorted(offsets)


class _StubPcfs:
    """Records the (offset, ptr, nbytes) triples the kernel would issue."""

    def __init__(self):
        self.reads: List[tuple] = []
        self.writes: List[tuple] = []

    def init(self):
        return True

    def lookup_or_create_file(self, path, size, need_create):
        self.file_size = size
        return 1234  # any non-zero nodeid


def _run_pcfs_kernel(be: PcfsRemoteBackend, cpu_base: int,
                     cpu_block_id: int, remote_block_id: int) -> List[tuple]:
    """Re-implementation of the C++ loop, driven by what ``attach`` produced.

    Mirrors ``_transfer_single_thread_impl`` exactly; the point is that it is
    fed only the backend's own numbers, so a stride the backend got wrong
    shows up as a wrong address here.
    """
    issued = []
    for i in range(be.num_layers):
        cpu_k_layer = cpu_base + i * be.cpu_layer_stride_in_bytes
        remote_layer_off = i * be.remote_layer_stride_in_bytes_per_file
        remote_k = remote_layer_off + remote_block_id * be.remote_block_stride_in_bytes
        issued.append((cpu_k_layer + cpu_block_id * be.remote_block_stride_in_bytes,
                       remote_k, be.chunk_size_in_bytes))
        if be.kv_dim == 1:
            continue
        cpu_v_layer = cpu_k_layer + be.cpu_kv_stride_in_bytes
        issued.append((cpu_v_layer + cpu_block_id * be.remote_block_stride_in_bytes,
                       remote_k + be.remote_kv_stride_in_bytes_per_file,
                       be.chunk_size_in_bytes))
    return issued


def _attach_pcfs(be: PcfsRemoteBackend, worker: FakeWorker, monkeypatch):
    """attach() with the PCFS SDK stubbed, so the geometry half really runs."""
    from flexkv import c_ext

    stub = _StubPcfs()
    monkeypatch.setattr(c_ext, "Pcfs", lambda *a, **k: stub, raising=False)
    monkeypatch.setattr(c_ext, "set_pcfs_instance", lambda *a: None,
                        raising=False)
    # attach() refuses to run when the CFS kernel is absent, which it is on
    # every box here; the geometry it computes is what we are testing.
    import flexkv.transfer.worker as tw
    monkeypatch.setattr(tw, "transfer_kv_blocks_remote", object(),
                        raising=False)
    be.attach(worker, make_geometry(worker))
    return stub


@pytest.mark.parametrize("shape", MODEL_SHAPES)
@pytest.mark.parametrize("layout_type", ["LAYERFIRST", "BLOCKFIRST"])
def test_pcfs_addresses_every_chunk_of_a_block_exactly_once(
        shape, layout_type, monkeypatch):
    """The kernel, fed attach()'s numbers, must cover a block exactly.

    Under BLOCKFIRST the pre-fix backend derived CPU strides as
    ``num_blocks * chunk`` -- the LAYERFIRST formula -- and addressed the pool
    ``num_blocks`` times too far out. BLOCKFIRST is the default
    (``FLEXKV_CPU_LAYOUT``), so this was the live path.
    """
    num_blocks, num_files = 64, 4
    dtype = torch.bfloat16
    layout = make_cpu_layout(shape, num_blocks, layout_type)
    remote_layout = make_cpu_layout(shape, num_blocks, layout_type)
    worker = FakeWorker(layout, dtype)

    be = PcfsRemoteBackend(
        remote_files=[f"/f{i}" for i in range(num_files)],
        remote_kv_layout=remote_layout,
        remote_config_custom={"pcfs_fsid": 1, "pcfs_port": 1,
                              "pcfs_ip": "127.0.0.1",
                              "pcfs_parent_nodeid": 1},
        enable_pcfs_sharing=False,
    )
    _attach_pcfs(be, worker, monkeypatch)

    block_id = 7
    issued = _run_pcfs_kernel(be, cpu_base=0, cpu_block_id=block_id,
                              remote_block_id=block_id)

    # Every byte of the block, once: the CPU offsets the kernel touches must
    # be exactly the chunks the layout says that block owns.
    expected = _pcfs_expected_cpu_offsets(layout, dtype, block_id)
    covered = []
    for cpu_off, _remote_off, nbytes in issued:
        covered.extend(range(cpu_off, cpu_off + nbytes,
                             layout.get_chunk_size() * dtype.itemsize))
    assert sorted(covered) == expected

    # ...and it must stay inside the pool it was given.
    pool_bytes = layout.kv_shape.numel() * dtype.itemsize
    for cpu_off, _remote_off, nbytes in issued:
        assert cpu_off >= 0 and cpu_off + nbytes <= pool_bytes, (
            f"CPU access [{cpu_off}, {cpu_off + nbytes}) escapes the "
            f"{pool_bytes}-byte pool")


@pytest.mark.parametrize("shape", MODEL_SHAPES)
@pytest.mark.parametrize("layout_type", ["LAYERFIRST", "BLOCKFIRST"])
def test_pcfs_blocks_do_not_overlap_on_either_side(shape, layout_type,
                                                   monkeypatch):
    """Two distinct blocks must not touch a shared byte, CPU or remote.

    The overlap check is what a too-large stride cannot survive: it aliases
    block b onto some other block's bytes.
    """
    num_blocks, num_files = 64, 4
    dtype = torch.bfloat16
    layout = make_cpu_layout(shape, num_blocks, layout_type)
    worker = FakeWorker(layout, dtype)

    be = PcfsRemoteBackend(
        remote_files=[f"/f{i}" for i in range(num_files)],
        remote_kv_layout=make_cpu_layout(shape, num_blocks, layout_type),
        remote_config_custom={"pcfs_fsid": 1, "pcfs_port": 1,
                              "pcfs_ip": "127.0.0.1",
                              "pcfs_parent_nodeid": 1},
        enable_pcfs_sharing=False,
    )
    stub = _attach_pcfs(be, worker, monkeypatch)

    def spans(block_id, side):
        out = []
        for cpu_off, remote_off, nbytes in _run_pcfs_kernel(
                be, 0, block_id, block_id):
            start = cpu_off if side == "cpu" else remote_off
            out.append((start, start + nbytes))
        return out

    for side in ("cpu", "remote"):
        a, b = spans(3, side), spans(4, side)
        for (a0, a1) in a:
            for (b0, b1) in b:
                assert a1 <= b0 or b1 <= a0, (
                    f"{side} blocks 3 and 4 overlap: "
                    f"[{a0},{a1}) vs [{b0},{b1})")

    # The file the backend asks PCFS to create must hold every block assigned
    # to it -- a stride that under-counts would size the file short.
    blocks_per_file = num_blocks // num_files
    assert stub.file_size >= blocks_per_file * (
        layout.get_block_stride() if layout_type == "BLOCKFIRST"
        else layout.get_chunk_size() * layout.num_layer * layout.kv_dim
    ) * dtype.itemsize


def test_pcfs_rejects_mismatched_cpu_and_remote_layouts(monkeypatch):
    """The kernel uses one set of block ids for both sides; layouts must agree."""
    layout = make_cpu_layout(QWEN3_8B, 64, "BLOCKFIRST")
    worker = FakeWorker(layout, torch.bfloat16)
    be = PcfsRemoteBackend(
        remote_files=["/f0"],
        remote_kv_layout=make_cpu_layout(QWEN3_8B, 64, "LAYERFIRST"),
        remote_config_custom={"pcfs_fsid": 1, "pcfs_port": 1,
                              "pcfs_ip": "127.0.0.1",
                              "pcfs_parent_nodeid": 1},
        enable_pcfs_sharing=False,
    )
    with pytest.raises(ValueError, match="must match"):
        _attach_pcfs(be, worker, monkeypatch)


# ---------------------------------------------------------------------------
# MooncakeStoreBackend.attach / key building
#
# The mooncake SDK is absent on most boxes, so the client half is stubbed and
# the parts that are FlexKV's own -- block sizing, pointer arithmetic, key
# construction, direction routing -- are asserted directly.
# ---------------------------------------------------------------------------


def _mooncake_cache_config(**overrides):
    from types import SimpleNamespace
    base = dict(
        mooncake_store_config_path="/nonexistent.json",
        mooncake_store_pp_rank=0, mooncake_store_pp_size=1,
        mooncake_store_node_layer_start=0, mooncake_store_node_layer_end=0,
        mooncake_store_total_layers=0,
        mooncake_mr_split_policy="strict",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _StubMooncakeClient:
    """Records what the backend asked the store to do."""

    def __init__(self):
        self.registered: List[int] = []
        self.unregistered: List[int] = []
        self.puts: List[tuple] = []
        self.gets: List[tuple] = []
        self.fail_next = False
        # Indices within the batch that come back False -- the interesting
        # case is a *partial* failure, which is what a single flag cannot say.
        self.fail_indices: set = set()

    # (ptr, size) rather than a tensor: the backend registers *memory
    # regions* now, because a HugePage pool larger than the transport's MR
    # limit has to be split into several, and a tensor cannot say "this half".
    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))

    def unregister_buffer(self, ptr):
        self.unregistered.append(ptr)

    def _results(self, keys):
        return [not (self.fail_next or i in self.fail_indices)
                for i in range(len(keys))]

    def batch_put(self, keys, ptrs, sizes):
        self.puts.append((list(keys), list(ptrs), list(sizes)))
        return self._results(keys)

    def batch_get(self, keys, ptrs, sizes):
        self.gets.append((list(keys), list(ptrs), list(sizes)))
        return self._results(keys)


def _attach_mooncake(be: MooncakeStoreBackend, worker: FakeWorker,
                     monkeypatch) -> _StubMooncakeClient:
    """attach() with the SDK stubbed out, so the FlexKV half really runs."""
    import flexkv.external.mooncake_store_utils as msu

    stub = _StubMooncakeClient()
    monkeypatch.setattr(msu.MooncakeStoreConfig, "from_file",
                        classmethod(lambda cls, cfg, override_global_segment_size=None:
                                    object()))
    monkeypatch.setattr(msu, "MooncakeStoreClient", lambda cfg: stub)
    be.attach(worker, make_geometry(worker))
    return stub


@pytest.mark.parametrize("shape", MODEL_SHAPES)
def test_mooncake_block_size_is_whole_block_bytes(shape, monkeypatch):
    """Opaque whole-block I/O: the store gets one value per CPU block, so its
    size is the block's total bytes -- elements * itemsize for a single-group
    layout."""
    from flexkv.external.mooncake_store_keys import PoolKind

    num_layers, num_kv_heads, head_size, kv_dim, tpb = shape
    dtype = torch.bfloat16
    cpu_layout = make_cpu_layout(shape, num_blocks=8, layout_type="BLOCKFIRST")
    cpu_blocks = torch.zeros(cpu_layout.kv_shape, dtype=dtype)
    worker = FakeWorker(cpu_layout, dtype, cpu_blocks)

    be = MooncakeStoreBackend(cache_config=_mooncake_cache_config(),
                              pool_kind=PoolKind.KV)
    _attach_mooncake(be, worker, monkeypatch)

    want = num_layers * kv_dim * tpb * num_kv_heads * head_size * dtype.itemsize
    assert be.block_size_bytes == want
    # The engine publishes its own whole-block size; the worker reads it back
    # for the transfer trace rather than the backend writing onto the worker.
    assert be.bytes_per_block == want
    # Not registered with CUDA: Mooncake owns the RDMA MR, and a second host
    # registration of the same shared pool exhausts the host mapping budget on
    # a large cache for no gain.
    assert worker.registered == []
    # A plain tensor pool maps exactly what it holds, so it is one whole MR.
    stub = be.mooncake_client
    assert stub.registered == [
        (cpu_blocks.data_ptr(), cpu_blocks.numel() * cpu_blocks.element_size())]


def test_mooncake_serves_a_heterogeneous_block_whose_chunks_do_not_exist(monkeypatch):
    """The multi-group edge: whole-block I/O works where chunk I/O cannot.

    Under a heterogeneous layout the block has no uniform (layer, kv) chunk --
    ``get_chunk_size()`` raises -- so ``strides is None`` on the CPU side and
    the *only* number a backend may read is ``block_stride``, already in bytes.
    Mooncake is opaque and whole-block, so it must attach here; a regression
    that reintroduces chunk arithmetic fails with the layout's own
    "not valid for multi-group layout" rather than silently sizing keys wrong.

    Byte size is asserted against the groups it was built from rather than
    against ``get_block_stride()``, so the test does not simply restate the
    implementation.
    """
    from flexkv.external.mooncake_store_keys import PoolKind

    cpu_layout = make_multi_group_layout(num_blocks=8)
    # Multi-group pools are byte-flat: the buffer dtype is uint8 and the
    # layout's second dim is already the block's byte count.
    cpu_blocks = torch.zeros(cpu_layout.kv_shape, dtype=torch.uint8)
    worker = FakeWorker(cpu_layout, torch.uint8, cpu_blocks)

    geometry = make_geometry(worker)
    assert geometry.has_multi_group
    assert geometry.cpu.strides is None
    with pytest.raises(ValueError, match="NIXL GDS_MT needs.*CPU"):
        geometry.cpu.require_strides("NIXL GDS_MT")

    be = MooncakeStoreBackend(cache_config=_mooncake_cache_config(),
                              pool_kind=PoolKind.KV)
    _attach_mooncake(be, worker, monkeypatch)

    # 4 layers * kv_dim 1 * (128 tokens * 1 head * 576) * 2 bytes  [main KV]
    # + 4 layers * kv_dim 1 * (128/128 tokens * 1 head * 128) * 1 byte [indexer]
    want = 4 * 1 * 128 * 1 * 576 * 2 + 4 * 1 * 1 * 1 * 128 * 1
    assert be.block_size_bytes == want
    assert be.bytes_per_block == want
    assert geometry.bytes_per_block == want
    # And the whole pool is one MR, sized in those same bytes.
    assert be.mooncake_client.registered == [(cpu_blocks.data_ptr(), want * 8)]


def test_pcfs_flattens_a_heterogeneous_block_into_one_io(monkeypatch):
    """PCFS under multi-group: one I/O per block, addressed in bytes.

    PCFS *is* chunk-addressing, but under BLOCKFIRST it flattens the whole
    block into a single pseudo-layer of a single pseudo-KV before it reads any
    stride -- which is what lets it serve a heterogeneous edge at all. The
    numbers below are the flattened ones (``num_layers == kv_dim == 1``), not
    the edge's four layers, and that distinction is the reason the backend may
    not write them back onto the worker.
    """
    cpu_layout = make_multi_group_layout(num_blocks=64)
    cpu_blocks = torch.zeros(cpu_layout.kv_shape, dtype=torch.uint8)
    worker = FakeWorker(cpu_layout, torch.uint8, cpu_blocks)

    be = PcfsRemoteBackend(
        remote_files=["/f0", "/f1"],
        remote_kv_layout=make_multi_group_layout(num_blocks=64),
        remote_config_custom={"pcfs_fsid": 1, "pcfs_port": 1,
                              "pcfs_ip": "127.0.0.1",
                              "pcfs_parent_nodeid": 1},
        enable_pcfs_sharing=False,
    )
    _attach_pcfs(be, worker, monkeypatch)

    want = 4 * 1 * 128 * 1 * 576 * 2 + 4 * 1 * 1 * 1 * 128 * 1
    assert (be.num_layers, be.kv_dim) == (1, 1)
    assert be.chunk_size_in_bytes == want
    assert be.bytes_per_block == want
    # The CPU base pointer is resolved at attach, not per transfer.
    assert be.cpu_base_ptr == cpu_blocks.data_ptr()


@pytest.mark.parametrize(
    "transfer_type, why",
    [
        (TransferType.REMOTE2H,
         "a get that did not run leaves stale or zeroed bytes in a CPU block "
         "the cache is about to hand out as a hit"),
        (TransferType.H2REMOTE,
         "a put that did not run leaves a key the index claims is stored"),
    ],
    ids=["get", "put"],
)
def test_mooncake_partial_failure_is_reported_per_block(transfer_type, why,
                                                        monkeypatch):
    """One False in the batch must reach the engine as *that block's* outcome.

    This test used to assert the opposite -- that a single False raised and
    failed the whole op -- because the engine had no way to be told "blocks 0
    and 2 arrived, block 1 did not".  It does now: an op carries
    ``block_results``, and the engine AND-merges them.  So the contract
    inverted, and the reason the old one existed is preserved: what must never
    happen is a partial batch reported as a *success*, because then the cache
    hands out block 1 as a hit.

    Failure is still not an exception. An op that never completes hangs its
    graph and leaks every cache block the plan holds, which is strictly worse
    than completing with the truth about each block.
    """
    from flexkv.external.mooncake_store_keys import PoolKind

    cpu_layout = make_cpu_layout(QWEN3_8B, num_blocks=8, layout_type="BLOCKFIRST")
    cpu_blocks = torch.zeros(cpu_layout.kv_shape, dtype=torch.bfloat16)
    worker = FakeWorker(cpu_layout, torch.bfloat16, cpu_blocks)
    be = MooncakeStoreBackend(cache_config=_mooncake_cache_config(),
                              pool_kind=PoolKind.KV)
    stub = _attach_mooncake(be, worker, monkeypatch)

    hashes = np.array([7, 8, 9], dtype=np.int64)
    cpu_ids = torch.tensor([2, 6, 4], dtype=torch.int64)
    remote_ids = torch.tensor([0, 1, 2], dtype=torch.int64)
    src, dst = ((remote_ids, cpu_ids) if transfer_type == TransferType.REMOTE2H
                else (cpu_ids, remote_ids))

    stub.fail_indices = {1}  # middle block only: a partial failure
    op = FakeOp(transfer_type, 3, block_hashes=hashes)
    op.transfer_op_id, op.transfer_graph_id = 41, 5

    block_results, moved = be.transfer_blocks(worker, op, src, dst)

    # Non-contiguous: the surviving blocks keep their own positions, which is
    # the whole point -- position 1 is the one the cache must not hand out.
    assert block_results == (True, False, True), why
    # Bytes *actually* moved. Counting the miss would inflate the bandwidth
    # every trace reports and hide the failure from the perf record.
    assert moved == 2 * be.block_size_bytes




def test_mooncake_splits_a_hugepage_pool_that_exceeds_the_mr_limit(monkeypatch):
    """A HugePage pool bigger than the MR limit registers as several regions.

    ``_split_mooncake_registration_regions`` is unit-tested directly in
    ``test_mooncake_large_pool_registration.py``.  What is *not* covered there
    is the wiring: whether ``attach`` reaches the splitter at all, and whether
    it feeds it the mapping's aligned length rather than the logical pool
    size.  Those are two different numbers, and registering the logical one
    leaves the tail of the mapping outside every MR.
    """
    from flexkv.external.mooncake_store_keys import PoolKind

    hugepage = 2 << 20
    cpu_layout = make_cpu_layout(QWEN3_8B, num_blocks=64, layout_type="BLOCKFIRST")
    # A real pool is a HugePage mapping, so its base pointer is HugePage
    # aligned and the splitter rejects anything else. torch.zeros is not, so
    # over-allocate and take an aligned view rather than relaxing the check --
    # that alignment is a precondition of the arithmetic under test.
    backing, cpu_blocks = _hugepage_aligned(cpu_layout.kv_shape,
                                            torch.bfloat16, hugepage)
    worker = FakeWorker(cpu_layout, torch.bfloat16, cpu_blocks)

    logical = cpu_blocks.numel() * cpu_blocks.element_size()
    # A HugePage handle knows its mapping is longer than the pool; a plain
    # tensor does not, which is why the worker publishes this separately.
    mapped = logical + (2 << 20)
    worker.cpu_blocks_mapped_size = mapped

    cfg = _mooncake_cache_config()
    cfg.hugepage_size_bytes = hugepage
    cfg.mooncake_max_mr_size_bytes = logical // 3  # force at least 3 regions

    be = MooncakeStoreBackend(cache_config=cfg, pool_kind=PoolKind.KV)
    stub = _attach_mooncake(be, worker, monkeypatch)

    # Every byte of the *mapping* is covered, and the splitter was fed the
    # mapped length rather than the logical one.
    assert len(stub.registered) > 1
    assert stub.registered[0][0] == cpu_blocks.data_ptr()
    assert sum(size for _, size in stub.registered) == mapped


def test_mooncake_rejects_a_direction_it_cannot_serve(monkeypatch):
    """A key/value store has no D2H: report it, do not raise past the worker."""
    from flexkv.external.mooncake_store_keys import PoolKind

    cpu_layout = make_cpu_layout(QWEN3_8B, num_blocks=8, layout_type="BLOCKFIRST")
    cpu_blocks = torch.zeros(cpu_layout.kv_shape, dtype=torch.bfloat16)
    worker = FakeWorker(cpu_layout, torch.bfloat16, cpu_blocks)
    be = MooncakeStoreBackend(cache_config=_mooncake_cache_config(),
                              pool_kind=PoolKind.KV)
    _attach_mooncake(be, worker, monkeypatch)

    op = FakeOp(TransferType.D2H, 3,
                block_hashes=np.array([7, 8, 9], dtype=np.int64))
    block_results, moved = be.transfer_blocks(
        worker, op,
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([2, 6, 4], dtype=torch.int64))

    assert block_results == (False, False, False)
    assert moved == 0
