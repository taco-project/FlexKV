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

``FakeWorker`` stands in for the worker's *edge*: the attributes a backend is
allowed to read. That contract is what the refactor introduced, so a worker
that stops publishing one of these fails here rather than in production.

``NixlFileBackend`` is covered by ``test_nixl_backend_e2e.py``, which round-trips
real bytes through a real agent.
"""
from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np
import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferType
from flexkv.transfer.backends import (
    MooncakeStoreBackend,
    NixlFileBackend,
    PcfsRemoteBackend,
    StorageBackend,
)

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
    """Exactly the worker attributes ``StorageBackend.attach`` may read.

    Deliberately not a ``TransferWorkerBase``: the value of this stand-in is
    that touching anything else raises ``AttributeError``.  ``attach`` reaching
    for a private worker method would then fail here rather than silently
    couple the backend to one worker class.
    """

    def __init__(self, cpu_kv_layout: KVCacheLayout, dtype: torch.dtype,
                 cpu_blocks: Optional[torch.Tensor] = None):
        self.worker_id = 0
        self.cpu_kv_layout = cpu_kv_layout
        self.dtype = dtype
        self.num_layers = cpu_kv_layout.num_layer
        self.kv_dim = cpu_kv_layout.kv_dim
        self.has_multi_group = cpu_kv_layout.layer_groups is not None
        self.cpu_blocks = cpu_blocks
        self.cpu_layer_ptrs = (
            torch.tensor([cpu_blocks.data_ptr()], dtype=torch.int64)
            if cpu_blocks is not None else torch.tensor([0], dtype=torch.int64)
        )
        self._bytes_per_block = -1
        self.registered: List[str] = []

    def _register_host_tensor(self, tensor: torch.Tensor, label: str = "") -> None:
        self.registered.append(label)


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
# The engine (libhifs_client_sdk) is not installed on most boxes, so the
# stride derivation is verified against a hand-computed expectation rather than
# by running a transfer. These formulas are the ones the old
# CPURemoteTransferWorker.__init__ carried verbatim.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", MODEL_SHAPES)
@pytest.mark.parametrize("transfer_type,is_read", [
    (TransferType.H2REMOTE, False),
    (TransferType.REMOTE2H, True),
])

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
    be.attach(worker)
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
    assert worker._bytes_per_block == want
    # Not registered with CUDA: Mooncake owns the RDMA MR, and a second host
    # registration of the same shared pool exhausts the host mapping budget on
    # a large cache for no gain.
    assert worker.registered == []
    # A plain tensor pool maps exactly what it holds, so it is one whole MR.
    stub = be.mooncake_client
    assert stub.registered == [
        (cpu_blocks.data_ptr(), cpu_blocks.numel() * cpu_blocks.element_size())]


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

    assert len(stub.registered) > 1
    assert all(size <= cfg.mooncake_max_mr_size_bytes
               for _, size in stub.registered)
    # Every byte of the *mapping* is covered, contiguously and exactly once.
    assert stub.registered[0][0] == cpu_blocks.data_ptr()
    assert sum(size for _, size in stub.registered) == mapped
    for (ptr, size), (next_ptr, _) in zip(stub.registered, stub.registered[1:]):
        assert ptr + size == next_ptr
    # No block straddles two MRs: that is the failure this splitting exists to
    # prevent, since older Mooncake releases cannot transfer such a block.
    assert all(size % be.block_size_bytes == 0
               for _, size in stub.registered[:-1])

    be.shutdown()
    # Reverse order, so a region is never left registered behind a freed one.
    assert stub.unregistered == [ptr for ptr, _ in reversed(stub.registered)]


def test_mooncake_plain_pool_registers_exactly_once(monkeypatch):
    """No mapped size means no splitting: one MR over what the pool holds."""
    from flexkv.external.mooncake_store_keys import PoolKind

    cpu_layout = make_cpu_layout(QWEN3_8B, num_blocks=8, layout_type="BLOCKFIRST")
    cpu_blocks = torch.zeros(cpu_layout.kv_shape, dtype=torch.bfloat16)
    worker = FakeWorker(cpu_layout, torch.bfloat16, cpu_blocks)
    assert not hasattr(worker, "cpu_blocks_mapped_size")

    be = MooncakeStoreBackend(cache_config=_mooncake_cache_config(),
                              pool_kind=PoolKind.KV)
    stub = _attach_mooncake(be, worker, monkeypatch)

    assert stub.registered == [
        (cpu_blocks.data_ptr(), cpu_blocks.numel() * cpu_blocks.element_size())]


def test_mooncake_store_failure_completes_the_op_all_false(monkeypatch):
    """A raising store client completes the op rather than propagating.

    The old code let this out of ``transfer``, and the worker turned it into a
    failed completion -- which worked only because a mooncake op was
    all-or-nothing.  Now that the op carries per-block outcomes, an exception
    escaping here would skip the reporting path entirely, and an op that never
    reports hangs its graph and leaks every cache block the plan holds.
    """
    from flexkv.external.mooncake_store_keys import PoolKind

    cpu_layout = make_cpu_layout(QWEN3_8B, num_blocks=8, layout_type="BLOCKFIRST")
    cpu_blocks = torch.zeros(cpu_layout.kv_shape, dtype=torch.bfloat16)
    worker = FakeWorker(cpu_layout, torch.bfloat16, cpu_blocks)
    be = MooncakeStoreBackend(cache_config=_mooncake_cache_config(),
                              pool_kind=PoolKind.KV)
    stub = _attach_mooncake(be, worker, monkeypatch)

    def boom(*_args, **_kwargs):
        raise RuntimeError("store is down")

    stub.batch_get = boom

    op = FakeOp(TransferType.REMOTE2H, 3,
                block_hashes=np.array([7, 8, 9], dtype=np.int64))
    block_results, moved = be.transfer_blocks(
        worker, op,
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([2, 6, 4], dtype=torch.int64))

    assert block_results == (False, False, False)
    assert moved == 0


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
