"""When the consumer's per-layer eventfds are allowed to fire.

Bypasses the SGLang socket handshake by registering real semaphore eventfds on
the region batch directly (``set_layer_eventfds``, which is what
``_init_completion`` does with the fds it receives over the UDS).  Verifies:

  - fds are silent before ``submit_layerwise``
  - each original layer receives exactly one ``write(1)`` semaphore signal
  - a layer with two members (main + indexer) still posts exactly once
  - a layer nothing is launched for is posted anyway, immediately
  - a layer whose only member is SWA waits for that SWA copy

The last two are the same layer in two different transfers, which is the point:
"empty" is a property of the op, not of the model.  An op carrying no SWA
blocks leaves an SWA-only layer uncovered, so it belongs in that op's
``empty_layers``; an op carrying them must not post it early.

Run:
    pytest tests/test_layerwise_eventfd_timing.py -v
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

from flexkv.common.config import LayerGroupSpec

from eventfd_probe import Fds
from test_layerwise_multi_group_swa import (
    CPU_SRC,
    INDEXER_HEAD_SIZE,
    MAIN_HEAD_SIZE,
    SWA_CPU_SRC,
    TOKENS_PER_BLOCK,
    MultiGroupFixture,
    _seed_main_cpu_layer,
    _seed_swa_cpu_layer,
    build_fixture,
    h2d_requests,
)

NOTIFY_MODE = "hostfunc"


def _armed(
    layer_groups: List[LayerGroupSpec],
    num_original_layers: int,
    *,
    has_swa: bool = True,
) -> Tuple[MultiGroupFixture, Fds]:
    """A fixture with the consumer's fd table already registered.

    The caller owns the ``Fds`` and must close it; every test here does so via
    ``with``.  Registering is not optional for these tests: ``LayerNotifier``
    no-ops when no table is present, so an unregistered fixture would exercise
    none of the marker machinery and every assertion would read zero.
    """
    fx = build_fixture(layer_groups, num_original_layers, has_swa=has_swa)
    fds = Fds(num_counters=1, tp_size=1, num_layers=num_original_layers)
    fx.group.set_layer_eventfds(
        fds.tensor(), 1, num_original_layers, NOTIFY_MODE)
    return fx, fds


def _run_h2d(fx: MultiGroupFixture, *, with_swa: bool) -> None:
    requests, empty_layers = h2d_requests(fx, with_swa=with_swa)
    fx.group.submit_layerwise(requests, empty_layers, 0)
    ok, err = fx.group.wait_layer_completion(120.0)
    assert ok, f"layerwise H2D did not complete: {err}"
    torch.cuda.synchronize()


def _dsv4_like_groups(num_c4_layers: int = 4) -> List[LayerGroupSpec]:
    c4_ids = list(range(num_c4_layers))
    c128_ids = list(range(num_c4_layers, num_c4_layers * 2))
    return [
        LayerGroupSpec(
            num_layers=len(c4_ids),
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=c4_ids,
            dtype=torch.uint8,
            compress_ratio=4,
        ),
        LayerGroupSpec(
            num_layers=len(c128_ids),
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE // 2,
            layer_indices=c128_ids,
            dtype=torch.uint8,
            compress_ratio=1,
        ),
        LayerGroupSpec(
            num_layers=len(c4_ids),
            num_kv_heads=1,
            head_size=INDEXER_HEAD_SIZE,
            layer_indices=c4_ids,
            dtype=torch.uint8,
            compress_ratio=4,
        ),
    ]


def _seed_all_layers(fx: MultiGroupFixture, with_swa: bool) -> None:
    member_map = fx.strides["layer_member_map"]
    for orig in range(fx.num_original_layers):
        for gi, local_id in member_map.members_of(orig):  # type: ignore[union-attr]
            _seed_main_cpu_layer(
                fx.cpu_blocks,
                CPU_SRC,
                orig,
                gi,
                local_id,
                fx.strides,
                fx.layer_groups,
                TOKENS_PER_BLOCK,
            )
    if with_swa:
        assert fx.swa_cpu is not None
        for orig in range(fx.num_original_layers):
            _seed_swa_cpu_layer(fx.swa_cpu, SWA_CPU_SRC, orig)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestLayerwiseEventfdTiming:
    def test_eventfd_idle_before_transfer(self) -> None:
        """All layer fds must stay at zero before H2D starts."""
        num_layers = 4
        with _armed(_dsv4_like_groups(num_c4_layers=2), num_layers)[1] as fds:
            for layer in range(num_layers):
                assert fds.units(layer) == [0], (
                    f"layer {layer} signalled before any transfer was submitted")

    def test_eventfd_posts_one_unit_per_layer_after_swa_h2d(self) -> None:
        """Each layer gets one signal, matching SGLang's one semaphore read."""
        num_layers = 8
        fx, fds = _armed(_dsv4_like_groups(num_c4_layers=4), num_layers)
        with fds:
            _seed_all_layers(fx, with_swa=True)
            for layer in range(num_layers):
                assert fds.units(layer) == [0]

            _run_h2d(fx, with_swa=True)

            for layer in range(num_layers):
                assert fds.units(layer) == [1], (
                    f"layer {layer}: expected exactly one semaphore unit")
                # Drained above; a second read must find nothing, or the
                # consumer would take one post as two ready layers.
                assert fds.units(layer) == [0], (
                    f"layer {layer}: extra signal after drain")

    def test_dual_member_layer_single_eventfd_post(self) -> None:
        """c4 layers carry main+indexer members; still only one eventfd post per layer."""
        num_layers = 4
        main = LayerGroupSpec(
            num_layers=num_layers,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=list(range(num_layers)),
            dtype=torch.uint8,
        )
        indexer = LayerGroupSpec(
            num_layers=3,
            num_kv_heads=1,
            head_size=INDEXER_HEAD_SIZE,
            layer_indices=[1, 2, 3],
            dtype=torch.uint8,
        )
        fx, fds = _armed([main, indexer], num_layers)
        with fds:
            _seed_all_layers(fx, with_swa=True)
            _run_h2d(fx, with_swa=True)

            for orig in range(num_layers):
                assert fds.units(orig) == [1], (
                    f"orig layer {orig}: dual-member+SWA must post exactly once")

    def test_empty_member_immediate_eventfd_without_swa(self) -> None:
        """Layer 0 has no members and this op carries no SWA -- posted up front."""
        num_layers = 4
        main = LayerGroupSpec(
            num_layers=3,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=[1, 2, 3],
            dtype=torch.uint8,
        )
        # The pool has SWA regions, but the transfer below names no SWA blocks,
        # so layer 0 is uncovered by *this* op and belongs in its empty list.
        fx, fds = _armed([main], num_layers, has_swa=True)
        with fds:
            member_map = fx.strides["layer_member_map"]
            assert member_map.members_of(0) == ()  # type: ignore[union-attr]

            for gi, local_id in member_map.members_of(1):  # type: ignore[union-attr]
                _seed_main_cpu_layer(
                    fx.cpu_blocks, CPU_SRC, 1, gi, local_id,
                    fx.strides, fx.layer_groups, TOKENS_PER_BLOCK,
                )

            assert fds.units(0) == [0]
            assert fds.units(1) == [0]

            _, empty_layers = h2d_requests(fx, with_swa=False)
            assert 0 in empty_layers, (
                "an SWA-only layer must be reported empty by an op that "
                "carries no SWA blocks, or the consumer waits forever")

            _run_h2d(fx, with_swa=False)

            assert fds.units(0) == [1]
            assert fds.units(1) == [1]

    def test_empty_member_waits_for_swa_before_eventfd(self) -> None:
        """Layer 0 has no main member but this op carries SWA -- no early post."""
        num_layers = 4
        main = LayerGroupSpec(
            num_layers=3,
            num_kv_heads=1,
            head_size=MAIN_HEAD_SIZE,
            layer_indices=[1, 2, 3],
            dtype=torch.uint8,
        )
        fx, fds = _armed([main], num_layers, has_swa=True)
        with fds:
            assert fx.swa_cpu is not None
            for orig in range(num_layers):
                _seed_swa_cpu_layer(fx.swa_cpu, SWA_CPU_SRC, orig)

            assert fds.units(0) == [0]

            _, empty_layers = h2d_requests(fx, with_swa=True)
            assert empty_layers == [], (
                "with SWA active every layer has a member, so nothing may be "
                "posted before its copy lands")

            _run_h2d(fx, with_swa=True)

            for orig in range(num_layers):
                assert fds.units(orig) == [1], (
                    f"orig {orig}: SWA-inclusive layer must post after H2D completes")
