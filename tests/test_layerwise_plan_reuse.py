"""The cached layerwise request plan, driven through the worker method itself.

``_layerwise_plan`` builds the ``RegionRequest`` list once per ``has_swa``
shape and keeps it; ``_layerwise_transfer_impl`` then rebinds only the two
block-id tensors before each submit.  That trade has exactly two ways to be
wrong, and neither raises:

* **a stale binding** -- if a rebind did not take (pybind11 moving out of the
  vector, an aliased tensor, a plan shared between shapes), the next transfer
  copies the *previous* op's blocks into the current op's destination.  Bytes
  land, the transfer reports success, and the consumer reads another request's
  KV.
* **the wrong pool's id tensor** -- SWA and full KV draw block ids from
  different slot-id spaces, so a request that reads the main tensor for an SWA
  region indexes the SWA pool with a main-pool slot id.

So these run real H2D through ``_layerwise_transfer_impl`` and compare bytes,
twice, with *different* block ids the second time.  ``tests/
test_layerwise_multi_group_swa.py`` covers the same geometry but builds its
requests itself, which is what left this method uncovered: the 9846-case suite
never called it once.

The SWA arm is the one e2e cannot reach -- Qwen3-8B has no SWA pool, and the
models that do need TP=8.

Run:
    pytest tests/test_layerwise_plan_reuse.py -v
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytest
import torch

from flexkv.common.config import LayerGroupSpec
from flexkv.common.pool import PoolId
from flexkv.transfer.completion import CompletionContract
from flexkv.transfer.region_batch import rank_share_mode, region_batch_available
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker, _Pool

from test_layerwise_multi_group_swa import (
    MAIN_HEAD_SIZE,
    TOKENS_PER_BLOCK,
    MultiGroupFixture,
    _seed_main_cpu_layer,
    _seed_swa_cpu_layer,
    build_fixture,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and region_batch_available()),
    reason="CUDA + RegionBatchGroup required",
)

NUM_LAYERS = 4
# Two (cpu_src, gpu_dst) pairs per pool. The second transfer must land the
# second pair's bytes; reusing the first pair would hide a stale binding.
MAIN_BLOCKS = [(2, 5), (1, 4)]
SWA_BLOCKS = [(3, 6), (0, 7)]


def _worker(fx: MultiGroupFixture, *, with_swa_pool: bool) -> GPUCPUTransferWorker:
    """A worker holding only what the two methods under test read.

    ``object.__new__`` because a real one imports CUDA IPC handles from a live
    producer process; every field below is either the fixture's own geometry or
    a transfer knob, and the region indices are laid out the way
    ``_build_region_batch`` lays them out -- full KV's regions first, then
    SWA's -- because those integers are what cpp addresses.
    """
    w = object.__new__(GPUCPUTransferWorker)
    w.worker_id = 0
    w._num_original_layers = fx.num_original_layers
    w.region_batch = fx.group
    w.transfer_num_cta_h2d = 4
    w.use_ce_transfer_h2d = True
    w._rank_share_mode = rank_share_mode("sharded")
    w._completion = CompletionContract.PER_LAYER
    w._layerwise_completion_timeout_s = 120.0
    w._layerwise_plans = {}

    num_groups = len(fx.layer_groups)
    w._pools = {
        PoolId.FULL_KV: _Pool(
            pool_id=PoolId.FULL_KV,
            name="full_kv",
            regions=[object() for _ in range(num_groups)],
            region_indices=list(range(num_groups)),
            layer_members=[list(m) for m in fx.strides["layer_members"]],
            bytes_per_block=1024,
        ),
    }
    if with_swa_pool:
        w._pools[PoolId.SWA] = _Pool(
            pool_id=PoolId.SWA,
            name="swa",
            regions=[object()],
            region_indices=[num_groups],
            # SWA numbers its layers the model's way: one per original layer.
            layer_members=[[(0, layer)] for layer in range(fx.num_original_layers)],
            bytes_per_block=256,
        )
    w._layer_milestones = w._build_layer_milestones()
    return w


def _ids(*values: int) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int64)


def _seed_main(fx: MultiGroupFixture, cpu_block: int) -> Dict[Tuple[int, int, int], torch.Tensor]:
    member_map = fx.strides["layer_member_map"]
    expected: Dict[Tuple[int, int, int], torch.Tensor] = {}
    for orig in range(fx.num_original_layers):
        for gi, local_id in member_map.members_of(orig):
            expected[(orig, gi, local_id)] = _seed_main_cpu_layer(
                fx.cpu_blocks, cpu_block, orig, gi, local_id,
                fx.strides, fx.layer_groups, TOKENS_PER_BLOCK,
            )
    return expected


def _assert_main(fx: MultiGroupFixture, gpu_block: int,
                 expected: Dict[Tuple[int, int, int], torch.Tensor]) -> None:
    for (orig, gi, local_id), exp in expected.items():
        actual = fx.gpu_tensors_per_group[gi][local_id][gpu_block].cpu()
        assert torch.equal(actual, exp), (
            f"main group={gi} orig={orig} local={local_id} gpu_block={gpu_block}")


def _assert_swa(fx: MultiGroupFixture, gpu_block: int,
                expected: Dict[int, torch.Tensor]) -> None:
    assert fx.swa_gpu_tensors is not None
    for orig, exp in expected.items():
        actual = fx.swa_gpu_tensors[orig][gpu_block].cpu()
        assert torch.equal(actual, exp), f"SWA orig={orig} gpu_block={gpu_block}"


def _fixture() -> MultiGroupFixture:
    """DSv4's shape in miniature: a main group plus an indexer that skips
    layer 0, so ``has_swa=False`` really does leave a layer empty."""
    main = LayerGroupSpec(
        num_layers=NUM_LAYERS, num_kv_heads=1, head_size=MAIN_HEAD_SIZE,
        layer_indices=list(range(NUM_LAYERS)), dtype=torch.uint8,
    )
    return build_fixture([main], NUM_LAYERS, has_swa=True)


class TestLayerwisePlanReuse:
    def test_swa_transfer_moves_both_pools(self) -> None:
        """The has_swa=True branch, end to end: main *and* SWA bytes land."""
        fx = _fixture()
        w = _worker(fx, with_swa_pool=True)
        cpu_main, gpu_main = MAIN_BLOCKS[0]
        cpu_swa, gpu_swa = SWA_BLOCKS[0]

        exp_main = _seed_main(fx, cpu_main)
        exp_swa = {
            orig: _seed_swa_cpu_layer(fx.swa_cpu, cpu_swa, orig)
            for orig in range(NUM_LAYERS)
        }

        w._layerwise_transfer_impl(
            _ids(cpu_main), _ids(gpu_main), _ids(cpu_swa), _ids(gpu_swa), 0)
        torch.cuda.synchronize()

        _assert_main(fx, gpu_main, exp_main)
        _assert_swa(fx, gpu_swa, exp_swa)

    def test_second_transfer_rebinds_both_pools(self) -> None:
        """The stale-binding failure: run twice, second op's blocks must win.

        If the cached plan kept the first call's tensors, this passes its
        *source* check and fails here -- the second destination would hold the
        first source's bytes, which is one request reading another's KV.
        """
        fx = _fixture()
        w = _worker(fx, with_swa_pool=True)

        seen = []
        for (cpu_main, gpu_main), (cpu_swa, gpu_swa) in zip(MAIN_BLOCKS, SWA_BLOCKS):
            exp_main = _seed_main(fx, cpu_main)
            exp_swa = {
                # Salt per round so the two rounds' SWA bytes differ even
                # where the block ids happen not to.
                orig: _seed_swa_cpu_layer(
                    fx.swa_cpu, cpu_swa, orig, salt=0xB2 + len(seen))
                for orig in range(NUM_LAYERS)
            }
            w._layerwise_transfer_impl(
                _ids(cpu_main), _ids(gpu_main), _ids(cpu_swa), _ids(gpu_swa), 0)
            torch.cuda.synchronize()
            seen.append(((gpu_main, exp_main), (gpu_swa, exp_swa)))

        # Both rounds, checked after both ran: the first round's destination
        # must still hold the first round's bytes too, or the plan is aliasing
        # the destinations rather than rebinding them.
        for (gpu_main, exp_main), (gpu_swa, exp_swa) in seen:
            _assert_main(fx, gpu_main, exp_main)
            _assert_swa(fx, gpu_swa, exp_swa)

        # One plan built, reused the second time.
        assert list(w._layerwise_plans) == [True]

    def test_the_two_shapes_get_different_plans(self) -> None:
        """has_swa keys the cache because it changes the request list.

        Sharing one plan across shapes would either submit SWA requests bound
        to the main pool's ids (a wrong-slot-space read) or leave the SWA-only
        layers unposted, hanging the consumer on their fds.
        """
        fx = _fixture()
        w = _worker(fx, with_swa_pool=True)

        with_swa = w._layerwise_plan(True)
        without = w._layerwise_plan(False)

        assert w._layerwise_plan(True) is with_swa, "plan rebuilt, not cached"
        assert with_swa is not without

        swa_index = len(fx.layer_groups)
        assert PoolId.SWA in with_swa[1]
        assert PoolId.SWA not in without[1]
        assert swa_index in [r.region_index for r in with_swa[0]]
        assert swa_index not in [r.region_index for r in without[0]]
        # Every layer here has a main member, so nothing is empty with SWA;
        # dropping SWA cannot make a layer empty in this geometry either --
        # what it must not do is silently keep the SWA requests.
        assert len(without[0]) == len(with_swa[0]) - NUM_LAYERS

    def test_swa_only_layers_are_empty_without_swa(self) -> None:
        """A layer whose only state is SWA is empty in the no-SWA plan.

        The consumer waits on every layer's fd regardless of whether this op
        carries state for it, so such a layer has to be posted up front. It is
        per *shape*, not per worker: the same worker leaves layer 0 uncovered
        only when the op has no SWA blocks.
        """
        main = LayerGroupSpec(
            num_layers=NUM_LAYERS - 1, num_kv_heads=1, head_size=MAIN_HEAD_SIZE,
            layer_indices=list(range(1, NUM_LAYERS)), dtype=torch.uint8,
        )
        fx = build_fixture([main], NUM_LAYERS, has_swa=True)
        w = _worker(fx, with_swa_pool=True)

        assert w._layerwise_plan(True)[2] == []
        assert w._layerwise_plan(False)[2] == [0]

    def test_swa_op_at_a_worker_without_an_swa_pool_raises(self) -> None:
        """The dangerous fall-through: SWA slot ids against full-KV regions.

        Nothing downstream can catch it -- the ids are valid integers in the
        main pool's space -- so it has to be refused here.
        """
        fx = _fixture()
        w = _worker(fx, with_swa_pool=False)
        with pytest.raises(RuntimeError, match="no SWA pool registered"):
            w._layerwise_transfer_impl(_ids(2), _ids(5), _ids(3), _ids(6), 0)
