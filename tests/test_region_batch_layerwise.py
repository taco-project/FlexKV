"""Per-layer completion on the region batch: same bytes, one fd per layer.

This is the equivalence evidence for folding ``LayerwiseTransferWorker`` (and
``csrc/layerwise.cpp``) into ``RegionBatchGroup::submit_layerwise``.  The old
worker fused three things -- a stride table, a launch order, and an eventfd
protocol.  The first two are now the region list and the request order, so what
is left to pin is the protocol, plus the fact that changing *when* the consumer
is told did not change *what* it is told.

So there are two kinds of assertion here:

  - bytes: ``submit_layerwise`` must land exactly what ``submit`` lands.  A
    per-layer launch that got a layer's ``layer_id`` or granularity wrong moves
    real data to the wrong place, and no amount of correct signalling hides it.
  - signalling: exactly one semaphore unit per rank per model layer, none
    before the transfer, and a model layer that spans two regions posts once,
    after both -- which is the property the consumer's "layer L is readable"
    actually rests on.

Needs >= 2 GPUs and an extension built with RegionBatchGroup.
"""
import gc
from typing import List

import pytest
import torch

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.transfer.region_batch import (
    build_region_batch,
    make_requests,
    region_batch_available,
)

from eventfd_probe import Fds
from test_region_batch_equivalence import (
    NUM_BLOCKS,
    Region,
    block_ids,
    sync_all,
)

NUM_GPUS = min(4, torch.cuda.device_count()) if torch.cuda.is_available() else 0

pytestmark = [
    pytest.mark.skipif(NUM_GPUS < 2,
                       reason=f"Need at least 2 GPUs, found {NUM_GPUS}"),
    pytest.mark.skipif(not region_batch_available(),
                       reason="extension built without RegionBatchGroup"),
]

# The model has 4 layers. Region "full" covers all of them; region "indexer"
# covers only the last three. Layer 0 is therefore a one-member milestone and
# layers 1..3 are two-member ones -- which is the DSv4 shape, and the case a
# single-region test cannot distinguish from "post whenever anything lands".
NUM_MODEL_LAYERS = 4
FULL_LAYERS = [0, 1, 2, 3]
INDEXER_LAYERS = [1, 2, 3]


@pytest.fixture(autouse=True)
def _cleanup_gpu_mem():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_regions(num_gpus: int, kv_dim: int = 2,
                   cpu_layout_name: str = "LAYERFIRST") -> List[Region]:
    """A "full" region over every layer plus an "indexer" over the last three."""
    return [
        Region("full", len(FULL_LAYERS), 8, kv_dim, cpu_layout_name, num_gpus,
               seed=1),
        Region("indexer", len(INDEXER_LAYERS), 8, kv_dim, cpu_layout_name,
               num_gpus, seed=2),
    ]


def _group(regions, num_gpus, cpu_layout_name="LAYERFIRST"):
    return build_region_batch(
        [r.to_spec() for r in regions], list(range(num_gpus)),
        ce_segment_threshold=GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
        ce_path_opt=GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
        ce_enable_memcpy2d=GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
        is_blockfirst=cpu_layout_name == "BLOCKFIRST",
        num_kv_heads=regions[0].num_kv_heads,
        ce_gather_threads=GLOBAL_CONFIG_FROM_ENV.ce_gather_threads,
        ce_gather_nt=GLOBAL_CONFIG_FROM_ENV.ce_gather_nt,
    )


def _milestones():
    """``[(model_layer, region_index, local_layer)]`` -- what the worker builds.

    Written out here rather than imported so that a change in the worker's
    milestone construction is caught by a disagreement with this table, not
    silently mirrored by it.
    """
    out = []
    for layer in range(NUM_MODEL_LAYERS):
        if layer in FULL_LAYERS:
            out.append((layer, 0, FULL_LAYERS.index(layer)))
        if layer in INDEXER_LAYERS:
            out.append((layer, 1, INDEXER_LAYERS.index(layer)))
    return out


def _layerwise_requests(num_regions, use_ce):
    ids = block_ids(NUM_BLOCKS)
    requests = []
    for model_layer, region_index, local_layer in _milestones():
        req = make_requests(
            num_regions, ids, ids, True,
            transfer_num_cta=4, use_ce_transfer=use_ce,
            region_indices=[region_index],
            layer_id=local_layer,
            layer_granularity=1,  # PER_LAYER
        )[0]
        req.milestone_layer = model_layer
        requests.append(req)
    return requests


def _empty_layers():
    covered = {m[0] for m in _milestones()}
    return [l for l in range(NUM_MODEL_LAYERS) if l not in covered]


def _seed_cpu(regions):
    """Distinct bytes per region so a region mixup shows up in the compare."""
    for i, r in enumerate(regions):
        flat = r.cpu.view(-1)
        ramp = torch.arange(flat.numel(), dtype=torch.float32) % 89
        flat[:] = ((ramp + i * 13) / 89.0).to(r.cpu.dtype)


@pytest.mark.parametrize("use_ce", [False, True], ids=["cuda", "ce"])
@pytest.mark.parametrize("cpu_layout_name", ["LAYERFIRST", "BLOCKFIRST"])
def test_layerwise_h2d_lands_the_same_bytes_as_a_whole_block_submit(
        use_ce, cpu_layout_name):
    """Per-layer launch order must not change what ends up on the GPUs.

    submit() moves every region's every layer in one request per region;
    submit_layerwise() moves the same bytes as one request per (region, layer)
    with granularity 1. Any drift in the per-layer offset arithmetic -- the
    thing layerwise.cpp used to compute for itself -- lands here as a mismatch.
    """
    num_gpus = NUM_GPUS

    ref = _build_regions(num_gpus, cpu_layout_name=cpu_layout_name)
    _seed_cpu(ref)
    for r in ref:
        r.zero_gpu()
    ref_group = _group(ref, num_gpus, cpu_layout_name)
    ref_group.submit(
        make_requests(len(ref), block_ids(NUM_BLOCKS), block_ids(NUM_BLOCKS),
                      True, transfer_num_cta=4, use_ce_transfer=use_ce),
        True)
    sync_all(num_gpus)
    expected = [r.snapshot_gpu() for r in ref]
    del ref_group, ref

    got = _build_regions(num_gpus, cpu_layout_name=cpu_layout_name)
    _seed_cpu(got)
    for r in got:
        r.zero_gpu()
    group = _group(got, num_gpus, cpu_layout_name)
    # No eventfds: submit_layerwise degrades to launch-in-layer-order with a
    # single drain, i.e. exactly CompletionContract.WHOLE. That degradation is
    # itself worth pinning -- it is what a non-sglang consumer gets.
    group.submit_layerwise(_layerwise_requests(len(got), use_ce),
                           _empty_layers(), 0)
    ok, err = group.wait_layer_completion(60.0)
    assert ok, err
    sync_all(num_gpus)

    for ri, (r, exp) in enumerate(zip(got, expected)):
        for g in range(num_gpus):
            for l, (a, b) in enumerate(zip(r.snapshot_gpu()[g], exp[g])):
                assert torch.equal(a, b), (
                    f"region {ri} ({r.name}) rank {g} layer {l}: layerwise H2D "
                    f"differs from the batched whole-block H2D")
    del group


@pytest.mark.parametrize("notify_mode", ["hostfunc", "polling"])
def test_every_model_layer_gets_exactly_one_unit_on_every_rank(notify_mode):
    """The protocol: one semaphore token per rank per layer per transfer.

    sglang reads exactly one. Two would let it run a layer ahead of the data;
    zero hangs it. Both notify paths must agree on this, since the mode is an
    env var a deployment can flip.
    """
    num_gpus = NUM_GPUS
    regions = _build_regions(num_gpus)
    _seed_cpu(regions)
    group = _group(regions, num_gpus)
    fds = Fds(num_counters=3, tp_size=num_gpus, num_layers=NUM_MODEL_LAYERS)
    try:
        group.set_layer_eventfds(fds.tensor(), num_gpus, NUM_MODEL_LAYERS,
                                 notify_mode)
        assert group.layer_notification_enabled

        for layer in range(NUM_MODEL_LAYERS):
            assert fds.units(layer) == [0] * num_gpus, (
                f"layer {layer} was signalled before the transfer started")

        group.submit_layerwise(_layerwise_requests(len(regions), False),
                               _empty_layers(), 0)
        ok, err = group.wait_layer_completion(60.0)
        assert ok, err

        for layer in range(NUM_MODEL_LAYERS):
            assert fds.units(layer) == [1] * num_gpus, (
                f"layer {layer}: expected one unit per rank")
            assert fds.units(layer) == [0] * num_gpus, (
                f"layer {layer}: a second signal arrived after the drain")
    finally:
        del group
        fds.close()


