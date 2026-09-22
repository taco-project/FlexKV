"""flexkv.common.hash_utils: the numpy hashing path keeps the torch path's contract."""
import numpy as np
import pytest

from flexkv.common.hash_utils import Hasher, gen_hashes

pytestmark = pytest.mark.unit

TPB = 16


def test_gen_hashes_chains_blocks_like_the_incremental_hasher():
    tokens = np.arange(1, 4 * TPB + 1, dtype=np.int64)
    hashes = gen_hashes(tokens, TPB)
    assert hashes.dtype == np.uint64 and hashes.shape == (4,)
    h = Hasher()
    expected = []
    for b in range(4):
        h.update(tokens[b * TPB:(b + 1) * TPB])
        expected.append(int(h.digest()))
    assert hashes.tolist() == expected
    # a strided view hashes the same tokens as its contiguous copy
    assert gen_hashes(np.repeat(tokens, 2)[::2], TPB).tolist() == expected
    # a trailing partial block is not hashed
    assert gen_hashes(tokens[:2 * TPB + 5], TPB).tolist() == expected[:2]
    assert gen_hashes(tokens[:TPB - 1], TPB).size == 0


def test_gen_hashes_rejects_non_int64_tokens():
    """int32 tokens reinterpreted as int64 would hash the neighbouring memory
    (the torch path raised on them too)."""
    with pytest.raises(TypeError, match="int64"):
        gen_hashes(np.arange(2 * TPB, dtype=np.int32), TPB)


def test_gen_hashes_numpy_binding_checks_its_buffers():
    from flexkv import c_ext
    tokens = np.arange(2 * TPB, dtype=np.int64)
    out = np.zeros(2, dtype=np.uint64)
    with pytest.raises(TypeError, match="int64"):
        c_ext.gen_hashes_numpy(Hasher().hasher, tokens.astype(np.int32), TPB, out)
    with pytest.raises(TypeError, match="uint64"):
        c_ext.gen_hashes_numpy(Hasher().hasher, tokens, TPB, np.zeros(2, dtype=np.int64))
    with pytest.raises(ValueError, match="blocks"):          # 3 blocks asked of 32 tokens
        c_ext.gen_hashes_numpy(Hasher().hasher, tokens, TPB, np.zeros(3, dtype=np.uint64))
    with pytest.raises(ValueError, match="contiguous"):
        c_ext.gen_hashes_numpy(Hasher().hasher, np.repeat(tokens, 2)[::2], TPB, out)
    with pytest.raises(ValueError, match="tokens_per_block"):
        c_ext.gen_hashes_numpy(Hasher().hasher, tokens, 0, out)
    c_ext.gen_hashes_numpy(Hasher().hasher, tokens, TPB, out)  # the valid call still works
    assert out.tolist() == gen_hashes(tokens, TPB).tolist()
