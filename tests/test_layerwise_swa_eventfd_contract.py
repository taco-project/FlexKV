"""The two eventfd invariants, read off the source that now implements them.

These used to assert on the text of ``csrc/layerwise.cpp``.  That file is gone:
per-layer completion is no longer a worker class with its own launch loop, it
is ``RegionBatchGroup::submit_layerwise`` plus ``LayerNotifier``.  The
invariants did not go with it, because they are properties of the protocol
sglang implements, not of any particular launcher:

  1. one semaphore unit per layer, never two -- sglang reads exactly one;
  2. a model layer is signalled after *every* region that covers it, not after
     the first, so "layer L is readable" means all of L.

``tests/test_region_batch_layerwise.py`` checks both against real eventfds on
real GPUs and is the authority.  These are the cheap, GPU-free guards that
catch the two edits most likely to break them silently -- a changed write
value, and a post moved from the per-model-layer marker to a per-region one.
"""
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def test_the_old_layerwise_worker_is_gone() -> None:
    """If it comes back, these guards are pointing at the wrong file."""
    assert not (ROOT / "csrc/layerwise.cpp").exists()
    assert not (ROOT / "csrc/layerwise.h").exists()

    # The *class* is what went away. ``flexkv/transfer/layerwise.py`` still
    # exists as a façade -- see the next test for why it has to.
    source = (ROOT / "flexkv/transfer/layerwise.py").read_text(encoding="utf-8")
    assert "class LayerwiseTransferWorker" not in source


def test_the_layerwise_import_path_still_answers() -> None:
    """sglang's connector imports this exact path, and it is out of tree.

    ``flexkv_connector.py`` does ``from flexkv.transfer.layerwise import
    build_layerwise_eventfd_socket_path`` inside a ``try/except ImportError``
    that re-raises as "FlexKV is not installed". So dropping the module does
    not surface as a rename -- it surfaces as the whole ``--enable-flexkv``
    path refusing to start, with a message pointing at the wrong cause.
    """
    from flexkv.transfer import layer_eventfd, layerwise

    assert (layerwise.build_layerwise_eventfd_socket_path
            is layer_eventfd.build_layerwise_eventfd_socket_path)


def test_exactly_one_semaphore_unit_is_written_per_layer() -> None:
    source = (ROOT / "csrc/layer_notify.cpp").read_text(encoding="utf-8")

    assert "uint64_t val = 1;" in source
    # Any other value desynchronizes sglang's accounting without an error: it
    # would read one unit and leave the rest, so the next transfer's wait
    # returns immediately on a layer that has not landed.
    for bad in ("uint64_t val = 0", "uint64_t val = 2", "uint64_t val = n"):
        assert bad not in source
    # One write site. Two would mean two ways to signal a layer, and only one
    # of them would be covered by the per-model-layer marker.
    assert source.count("write(fd, &val, sizeof(val))") == 1


def test_a_layer_is_posted_from_its_marker_not_from_a_region() -> None:
    """The post must hang off ``milestone_layer``, which spans regions.

    ``submit_layerwise`` groups requests by milestone layer and records one
    marker per (layer, rank) after that layer's *last* request is on the
    stream. A post moved inside the per-request loop would fire once per
    region, which is the exact bug the dual-member DSv4 layers exist to catch.
    """
    source = (ROOT / "csrc/region_batch.cpp").read_text(encoding="utf-8")

    assert "notifier_.begin_layer(layer)" in source
    # The marker is recorded when the milestone layer *changes*, i.e. after the
    # last request of the layer that is closing -- not per request.
    assert "if (layer != open_layer && open_layer >= 0) {" in source
    assert "notifier_.record(open_layer, rank, pool_->stream(rank));" in source
    # ...and once more after the loop, for the final layer.
    assert source.count(
        "notifier_.record(open_layer, rank, pool_->stream(rank));") == 2


def test_layers_this_model_has_no_state_for_are_still_posted() -> None:
    """A consumer waiting on a layer we never write hangs unless we post it."""
    source = (ROOT / "csrc/region_batch.cpp").read_text(encoding="utf-8")

    assert "for (int layer : empty_layers) {" in source
    assert "notifier_.post_empty(layer);" in source
