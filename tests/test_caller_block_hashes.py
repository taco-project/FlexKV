"""Caller-supplied block hashes: an engine that names its pages itself (for
example SGLang's SHA-256 page chain) hands those names to FlexKV, so the cache
and the engine's KV events share one key space. Without them FlexKV hashes the
tokens itself, exactly as before."""
import numpy as np
import pytest

from flexkv.common import block as block_mod
from flexkv.common.block import SequenceMeta
from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.server.request import GetMatchRequest, PutMatchRequest

TPB = 4


def _tokens(n_blocks, extra=0):
    return np.arange(1, n_blocks * TPB + extra + 1, dtype=np.int64)


@pytest.fixture
def scheme(monkeypatch):
    def set_scheme(value):
        monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "hash_scheme", value)
    set_scheme("")
    return set_scheme


def test_without_hashes_flexkv_hashes_the_tokens_as_before(scheme):
    a = SequenceMeta(_tokens(3), TPB)
    b = SequenceMeta(_tokens(3), TPB)
    assert a.block_hashes.dtype == np.uint64 and a.block_hashes.size == 3
    np.testing.assert_array_equal(a.block_hashes, b.block_hashes)


def test_given_hashes_are_used_verbatim_and_namespace_is_not_mixed_in(scheme):
    given = np.array([-1, 7, 2**62], dtype=np.int64)          # signed, as SGLang emits
    seq = SequenceMeta(_tokens(3), TPB, namespace=["ns"], block_hashes=given)
    assert seq.has_hashes() and seq.block_hashes.dtype == np.uint64
    np.testing.assert_array_equal(seq.block_hashes.view(np.int64), given)
    assert seq.get_hash(0) == 2**64 - 1


def test_hashes_beyond_the_full_blocks_are_dropped(scheme):
    # The trailing partial block is never stored, so its hash is not either.
    seq = SequenceMeta(_tokens(2, extra=3), TPB, block_hashes=np.arange(10, 14, dtype=np.uint64))
    np.testing.assert_array_equal(seq.block_hashes, [10, 11])


def test_too_few_hashes_is_an_error(scheme):
    with pytest.raises(ValueError, match="2 entries for 3 blocks"):
        SequenceMeta(_tokens(3), TPB, block_hashes=np.array([1, 2], dtype=np.int64))


def test_float_hashes_are_rejected(scheme):
    with pytest.raises(TypeError):
        SequenceMeta(_tokens(1), TPB, block_hashes=np.array([1.0]))


def test_a_named_scheme_requires_hashes_on_every_request(scheme):
    scheme("sglang-sha256-i64-v1")
    assert block_mod.caller_hashes_required()
    with pytest.raises(ValueError, match="FLEXKV_HASH_SCHEME"):
        SequenceMeta(_tokens(2), TPB)
    SequenceMeta(_tokens(2), TPB, block_hashes=np.array([1, 2], dtype=np.int64))


def test_ipc_requests_carry_the_hashes_and_default_to_none():
    req = GetMatchRequest(dp_client_id=0, token_ids=_tokens(1), token_mask=None)
    assert req.block_hashes is None
    req = PutMatchRequest(dp_client_id=0, token_ids=_tokens(1), token_mask=None,
                          block_hashes=np.array([5], dtype=np.int64))
    assert int(req.block_hashes[0]) == 5
