from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def test_layerwise_posts_one_eventfd_token_per_layer() -> None:
    source = (ROOT / "csrc/layerwise.cpp").read_text(encoding="utf-8")

    assert "uint64_t val = 2" not in source
    assert source.count("uint64_t val = 1;") == 3


def test_dense_layer_waits_for_swa_before_notification() -> None:
    source = (ROOT / "csrc/layerwise.cpp").read_text(encoding="utf-8")

    assert "swa_slots_for_orig_(orig, swa_active) > 0" in source
    assert (
        "if (!layer_members_[orig].empty() ||\n"
        "        swa_slots_for_orig_(orig, swa_active) > 0) {\n"
        "      work_origs.push_back(orig);"
    ) in source
    assert "int slots_per_gpu = members_this_layer + swa_slots;" in source
