"""CUDA_VISIBLE_DEVICES policy of the radixshmem shm TE subprocess."""
import pytest

from flexkv.transfer_manager import shm_te_clears_cuda_visible_devices


@pytest.mark.parametrize("cvd, needed, clears", [
    (None, 2, False),        # nothing set: the TE sees every GPU, ids are physical
    ("2,3", 2, False),       # sglang: one namespace for all TP workers -> inherit
    ("0,1,2,3", 2, False),   # a wider namespace than needed still covers the TE
    ("5", 1, False),         # single-GPU deployment on a restricted device -> inherit
    ("3", 8, True),          # vLLM DP: rank pinned to one device, TE serves 8 -> clear
    ("", 2, True),           # empty value hides every GPU; drop it
    ("GPU-aaaa,GPU-bbbb", 2, False),  # UUID form counts the same way
])
def test_shm_te_cvd_policy(cvd, needed, clears):
    assert shm_te_clears_cuda_visible_devices(cvd, needed) is clears
