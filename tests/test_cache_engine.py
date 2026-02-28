import random
import time
from typing import Union

import pytest
import numpy as np

from flexkv.cache.mempool import Mempool
from flexkv.cache.cache_engine import CacheEngine, CacheEngineAccel
from flexkv.common.transfer import DeviceType
from flexkv.common.block import SequenceMeta

CacheEngineType = Union[CacheEngine, CacheEngineAccel]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENGINE_CLASSES = [CacheEngine, CacheEngineAccel]

DEFAULT_NUM_TOTAL_BLOCKS = 64
DEFAULT_TOKENS_PER_BLOCK = 4
DEFAULT_EVICT_RATIO = 0.05
DEFAULT_DEVICE_TYPE = DeviceType.CPU

LARGE_NUM_TOTAL_BLOCKS = 10_000_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=ENGINE_CLASSES, ids=[cls.__name__ for cls in ENGINE_CLASSES])
def engine_cls(request):
    return request.param


@pytest.fixture
def cache_engine(request: pytest.FixtureRequest, engine_cls) -> CacheEngineType:
    param = getattr(request, 'param', {})
    default_config_kwargs = {
        'device_type': DEFAULT_DEVICE_TYPE,
        'num_total_blocks': DEFAULT_NUM_TOTAL_BLOCKS,
        'tokens_per_block': DEFAULT_TOKENS_PER_BLOCK,
        'evict_ratio': DEFAULT_EVICT_RATIO,
    }
    default_config_kwargs.update(param)
    return engine_cls(**default_config_kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BASE_CONFIG = {
    'num_total_blocks': DEFAULT_NUM_TOTAL_BLOCKS,
    'tokens_per_block': DEFAULT_TOKENS_PER_BLOCK,
    'evict_ratio': DEFAULT_EVICT_RATIO,
    'device_type': DEFAULT_DEVICE_TYPE,
}


def _cfg(**overrides) -> dict:
    """Return a copy of the base config with *overrides* applied."""
    cfg = _BASE_CONFIG.copy()
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Tests – CacheEngine config / init
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "config, should_raise",
    [
        # --- Valid configurations ---
        # Default valid config
        (_cfg(), False),
        # Explicit eviction policies – all 5 should be valid
        (_cfg(eviction_policy='lru'), False),
        (_cfg(eviction_policy='lfu'), False),
        (_cfg(eviction_policy='fifo'), False),
        (_cfg(eviction_policy='mru'), False),
        (_cfg(eviction_policy='filo'), False),
        # Minimal valid: 1 block, tokens_per_block=1
        (_cfg(num_total_blocks=1, tokens_per_block=1), False),
        # Large tokens_per_block (power of 2)
        (_cfg(tokens_per_block=128), False),
        # Different device types
        (_cfg(device_type=DeviceType.GPU), False),
        (_cfg(device_type=DeviceType.SSD), False),

        # --- Invalid: num_total_blocks ---
        # Zero blocks
        (_cfg(num_total_blocks=0, device_type=DeviceType.GPU), True),
        # Negative blocks
        (_cfg(num_total_blocks=-1), True),
        (_cfg(num_total_blocks=-100), True),

        # --- Invalid: tokens_per_block ---
        # Zero tokens per block
        (_cfg(tokens_per_block=0, device_type=DeviceType.SSD), True),
        # Negative tokens per block
        (_cfg(tokens_per_block=-1), True),
        (_cfg(tokens_per_block=-4), True),
        # Not a power of 2
        (_cfg(tokens_per_block=3), True),
        (_cfg(tokens_per_block=5), True),
        (_cfg(tokens_per_block=6), True),
        (_cfg(tokens_per_block=7), True),
        (_cfg(tokens_per_block=9), True),
        (_cfg(tokens_per_block=10), True),
        (_cfg(tokens_per_block=15), True),
        (_cfg(tokens_per_block=17), True),
        (_cfg(tokens_per_block=100), True),

        # --- Invalid: device_type ---
        (_cfg(device_type='Unknown'), True),
        (_cfg(device_type=999), True),
        (_cfg(device_type=None), True),
    ],
)
def test_config_init(engine_cls, config: dict, should_raise: bool):
    if should_raise:
        with pytest.raises(ValueError):
            engine_cls(**config)
    else:
        engine = engine_cls(**config)
        assert isinstance(engine, engine_cls)


# ---------------------------------------------------------------------------
# Tests – Mempool (independent of engine class)
# ---------------------------------------------------------------------------
def test_mempool():
    mempool = Mempool(num_total_blocks=DEFAULT_NUM_TOTAL_BLOCKS)
    assert mempool.num_free_blocks == DEFAULT_NUM_TOTAL_BLOCKS

    # Basic allocate and recycle
    block_ids = mempool.allocate_blocks(16)
    assert isinstance(block_ids, np.ndarray)
    assert block_ids.dtype == np.int64
    assert block_ids.shape == (16,)
    assert mempool.num_free_blocks == DEFAULT_NUM_TOTAL_BLOCKS - 16

    mempool.recycle_blocks(block_ids)
    assert mempool.num_free_blocks == DEFAULT_NUM_TOTAL_BLOCKS

    # Exhaust all blocks
    block_ids = np.concatenate([
        mempool.allocate_blocks(16),
        mempool.allocate_blocks(16),
        mempool.allocate_blocks(16),
        mempool.allocate_blocks(16),
    ])
    assert mempool.num_free_blocks == 0

    with pytest.raises(ValueError):
        mempool.allocate_blocks(1)

    mempool.recycle_blocks(block_ids)
    assert mempool.num_free_blocks == DEFAULT_NUM_TOTAL_BLOCKS

    # Allocate zero blocks
    empty_blocks = mempool.allocate_blocks(0)
    assert empty_blocks.shape == (0,)
    assert empty_blocks.dtype == np.int64
    assert mempool.num_free_blocks == DEFAULT_NUM_TOTAL_BLOCKS

    # Allocate negative raises
    with pytest.raises(ValueError):
        mempool.allocate_blocks(-1)

    # Recycle empty array
    mempool.recycle_blocks(np.array([], dtype=np.int64))
    assert mempool.num_free_blocks == DEFAULT_NUM_TOTAL_BLOCKS

    # Recycle wrong dtype raises
    with pytest.raises(ValueError):
        mempool.recycle_blocks(np.array([1, 2, 3], dtype=np.int32))

    # recycle_blocks no longer raises ValueError for already free blocks
    mempool.recycle_blocks(np.array([1, 2, 3], dtype=np.int64))
    assert mempool.num_free_blocks == DEFAULT_NUM_TOTAL_BLOCKS

    # Recycle wrong ndim raises
    with pytest.raises(ValueError):
        mempool.recycle_blocks(np.array([[1, 2, 3]], dtype=np.int64))

# ---------------------------------------------------------------------------
# Tests – CacheEngine reset
# ---------------------------------------------------------------------------
def test_reset(cache_engine: CacheEngineType):
    cache_engine.reset()
    assert cache_engine.index.is_empty()
    assert cache_engine.mempool.num_used_blocks == 0


# ---------------------------------------------------------------------------
# Tests – CacheEngine match & insert
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cache_engine",
    [
        {'num_total_blocks': LARGE_NUM_TOTAL_BLOCKS, 'tokens_per_block': 1, 'device_type': DEFAULT_DEVICE_TYPE},
        {'num_total_blocks': LARGE_NUM_TOTAL_BLOCKS, 'tokens_per_block': 16, 'device_type': DEFAULT_DEVICE_TYPE},
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_insert", [1, 10, 100])
@pytest.mark.parametrize("seq_len", [1, 10, 16, 32, 10000])
def test_match_and_insert(cache_engine: CacheEngineType, num_insert: int, seq_len: int):
    base_token_ids = np.random.randint(0, 10000, (seq_len,), dtype=np.int64)
    base_num_blocks = seq_len // cache_engine.tokens_per_block
    cache_engine.insert(
        SequenceMeta(token_ids=base_token_ids, tokens_per_block=cache_engine.tokens_per_block),
        np.arange(base_num_blocks, dtype=np.int64),
        is_ready=True,
    )
    cur_cached_blocks = base_num_blocks
    for i in range(num_insert):
        prefix_ratio = random.random()
        prefix_len = int(len(base_token_ids) * prefix_ratio)
        num_prefix_blocks = prefix_len // cache_engine.tokens_per_block
        token_ids = np.concatenate([
            base_token_ids[:prefix_len],
            np.random.randint(
                10000 + i * seq_len,
                10000 + (i + 1) * seq_len,
                (seq_len - prefix_len,),
                dtype=np.int64,
            ),
        ])
        insert_sequence_meta = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=cache_engine.tokens_per_block,
        )
        match_result = cache_engine.match(insert_sequence_meta)
        assert match_result.num_ready_matched_blocks == num_prefix_blocks
        assert match_result.num_matched_blocks == num_prefix_blocks
        assert match_result.last_ready_node is not None
        assert match_result.last_node is not None
        assert match_result.physical_blocks.shape == (num_prefix_blocks,)
        assert match_result.physical_blocks.dtype == np.int64

        num_insert_blocks = insert_sequence_meta.num_blocks - num_prefix_blocks
        cache_engine.insert(
            insert_sequence_meta,
            np.arange(num_insert_blocks, dtype=np.int64),
            is_ready=True,
            match_result=match_result,
        )
        cur_cached_blocks += num_insert_blocks
        assert cache_engine.index.total_cached_blocks() == cur_cached_blocks

        match_result = cache_engine.match(insert_sequence_meta)
        assert match_result.num_matched_blocks == insert_sequence_meta.num_blocks
        assert match_result.num_ready_matched_blocks == insert_sequence_meta.num_blocks


# ---------------------------------------------------------------------------
# Tests – CacheEngine take & recycle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cache_engine",
    [
        {'num_total_blocks': 100, 'tokens_per_block': 16, 'device_type': DEFAULT_DEVICE_TYPE},
    ],
    indirect=True,
)
def test_take_and_recycle(cache_engine: CacheEngineType):
    num_total_blocks = cache_engine.num_total_blocks
    tokens_per_block = cache_engine.tokens_per_block
    seq_blocks = 10
    token_ids = np.random.randint(0, 10000, (seq_blocks * tokens_per_block,), dtype=np.int64)
    sequence_meta = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)
    physical_blocks = cache_engine.take(seq_blocks)
    radixnode = cache_engine.insert(sequence_meta, physical_blocks, is_ready=True)
    assert cache_engine.index.total_cached_blocks() == seq_blocks

    # take(0) should return an empty array
    empty_blocks = cache_engine.take(0)
    assert empty_blocks.shape == (0,)
    assert empty_blocks.dtype == np.int64

    # Negative take raises ValueError
    with pytest.raises(ValueError):
        cache_engine.take(-1)

    # Strict take with protected node raises RuntimeError when insufficient
    with pytest.raises(RuntimeError):
        cache_engine.take(num_total_blocks, protected_node=radixnode, strict=True)

    # Non-strict take returns available blocks only
    physical_blocks2 = cache_engine.take(
        num_total_blocks, protected_node=radixnode, strict=False,
    )
    assert physical_blocks2.shape == (num_total_blocks - seq_blocks,)
    assert physical_blocks2.dtype == np.int64

    cache_engine.recycle(physical_blocks2)

    # Locked node prevents eviction
    cache_engine.lock_node(radixnode)
    with pytest.raises(RuntimeError):
        cache_engine.take(num_total_blocks, protected_node=radixnode, strict=True)
    cache_engine.unlock(radixnode)
    cache_engine.set_ready(radixnode, True, radixnode.size())

    # After unlock, strict take of all blocks succeeds (evicts the node)
    physical_blocks = cache_engine.take(num_total_blocks, protected_node=None, strict=True)
    assert physical_blocks.shape == (num_total_blocks,)
    assert cache_engine.index.total_cached_blocks() == 0
    assert radixnode.parent is None


# ---------------------------------------------------------------------------
# Tests – CacheEngine cleanup (lock / unlock / set_ready)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cache_engine",
    [
        {'num_total_blocks': 100, 'tokens_per_block': 1, 'device_type': DEFAULT_DEVICE_TYPE},
    ],
    indirect=True,
)
def test_cleanup(cache_engine: CacheEngineType):
    if cache_engine.tokens_per_block != 1:
        pytest.skip("tokens_per_block != 1")
    tokens_per_block = cache_engine.tokens_per_block
    token_ids_list = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64),
        np.array([0, 1, 2, 3, 17, 15, 19, 20], dtype=np.int64),
        np.array([0, 23, 22, 21], dtype=np.int64),
    ]
    sequence_meta_list = [
        SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block)
        for token_ids in token_ids_list
    ]

    # Insert first sequence (all unready)
    num_insert_blocks0 = sequence_meta_list[0].num_blocks
    radixnode0 = cache_engine.insert(
        sequence_meta_list[0],
        np.arange(num_insert_blocks0, dtype=np.int64),
        is_ready=False,
    )
    cache_engine.lock_node(radixnode0)
    radixnode0_size = radixnode0.size()

    # Insert second sequence (shares prefix with first)
    match_result = cache_engine.match(sequence_meta_list[1])
    num_insert_blocks1 = sequence_meta_list[1].num_blocks - match_result.num_matched_blocks
    radixnode1 = cache_engine.insert(
        sequence_meta_list[1],
        np.arange(num_insert_blocks1, dtype=np.int64),
        match_result=match_result,
        is_ready=False,
    )
    cache_engine.lock_node(radixnode1)
    radixnode1_size = radixnode1.size()

    # Insert third sequence (shares prefix with first)
    match_result = cache_engine.match(sequence_meta_list[2])
    num_insert_blocks2 = sequence_meta_list[2].num_blocks - match_result.num_matched_blocks
    radixnode2 = cache_engine.insert(
        sequence_meta_list[2],
        np.arange(num_insert_blocks2, dtype=np.int64),
        match_result=match_result,
        is_ready=False,
    )
    cache_engine.lock_node(radixnode2)
    radixnode2_size = radixnode2.size()

    total_insert_blocks = num_insert_blocks0 + num_insert_blocks1 + num_insert_blocks2
    assert cache_engine.index.total_cached_blocks() == total_insert_blocks
    assert cache_engine.index.total_unready_blocks() == total_insert_blocks
    assert cache_engine.index.total_ready_blocks() == 0

    # Unlock & set ready in reverse order, verify incremental ready counts
    cache_engine.unlock(radixnode2)
    cache_engine.set_ready(radixnode2, True, radixnode2_size)
    assert cache_engine.index.total_ready_blocks() == num_insert_blocks2

    cache_engine.unlock(radixnode1)
    cache_engine.set_ready(radixnode1, True, radixnode1_size)
    assert cache_engine.index.total_ready_blocks() == num_insert_blocks1 + num_insert_blocks2

    cache_engine.unlock(radixnode0)
    cache_engine.set_ready(radixnode0, True, radixnode0_size)
    assert cache_engine.index.total_ready_blocks() == total_insert_blocks


# ---------------------------------------------------------------------------
# Tests – Eviction policy (LRU / LFU / FIFO / MRU / FILO)
# ---------------------------------------------------------------------------

# Shared constants for eviction tests
_EVICT_TOKENS_PER_BLOCK = 1
_EVICT_NUM_TOTAL_BLOCKS = 4
_EVICT_RATIO = 0.25  # Evict exactly 1 block when full


def _make_seqs(n: int = 5):
    """Create *n* single-token sequences: A→[0], B→[1], ..."""
    return [
        SequenceMeta(
            token_ids=np.array([i], dtype=np.int64),
            tokens_per_block=_EVICT_TOKENS_PER_BLOCK,
        )
        for i in range(n)
    ]


def _create_engine(engine_cls, policy: str, hit_reward_seconds: int = 0,
                   num_total_blocks: int = _EVICT_NUM_TOTAL_BLOCKS):
    """Helper to create an engine with the given eviction *policy*."""
    return engine_cls(
        device_type=DEFAULT_DEVICE_TYPE,
        num_total_blocks=num_total_blocks,
        tokens_per_block=_EVICT_TOKENS_PER_BLOCK,
        evict_ratio=_EVICT_RATIO,
        eviction_policy=policy,
        hit_reward_seconds=hit_reward_seconds,
    )


def _insert_and_access(engine, seqs, access_pattern):
    """Insert the first 4 sequences, then apply *access_pattern*.

    *access_pattern* is a list of (seq_index, repeat_count) tuples that
    determines how many times each sequence is matched (accessed).  A small
    sleep is injected between groups so that ``last_access_time`` and
    ``creation_time`` differ between sequences.

    After all accesses, the 5th sequence (seqs[4]) is inserted, which
    triggers eviction of exactly 1 block.

    Returns a dict mapping sequence label ('A'..'E') to the number of
    matched blocks after the eviction.
    """
    labels = ['A', 'B', 'C', 'D', 'E']

    # 1. Insert A, B, C, D in order (with small delays for distinct timestamps)
    for seq in seqs[:4]:
        engine.insert(seq, engine.take(1), is_ready=True)
        time.sleep(0.002)

    # 2. Apply access pattern
    for idx, count in access_pattern:
        for _ in range(count):
            engine.match(seqs[idx])
        time.sleep(0.002)

    # 3. Insert E → triggers eviction of 1 block
    engine.insert(seqs[4], engine.take(1), is_ready=True)

    # 4. Check which sequences survived
    return {
        labels[i]: engine.match(seqs[i]).num_matched_blocks
        for i in range(5)
    }


def _assert_only_evicted(result: dict, evicted_label: str):
    """Assert that exactly *evicted_label* was evicted (matched == 0)."""
    for label, matched in result.items():
        if label == evicted_label:
            assert matched == 0, (
                f"{evicted_label} should have been evicted, got result: {result}"
            )
        else:
            assert matched == 1, (
                f"{label} should NOT have been evicted, got result: {result}"
            )


def test_eviction_policy(engine_cls):
    """Test all 5 eviction policies with a unified setup.

    Setup:
      4 blocks total, evict_ratio=0.25 → evict 1 block when full.
      Sequences: A=[0], B=[1], C=[2], D=[3], E=[4].
      Insert A, B, C, D in order, then apply access pattern, then insert E
      to trigger eviction.  Verify which node gets evicted.

    Access pattern: B×4, C×3, D×2, A×5
      → Access order (by time): B → C → D → A
      → Hit counts: A=5, B=4, C=3, D=2
      → Insertion order: A, B, C, D
    """
    access_pattern = [(1, 4), (2, 3), (3, 2), (0, 5)]

    def run_test(policy):
        seqs = _make_seqs()
        engine = _create_engine(engine_cls, policy)
        return _insert_and_access(engine, seqs, access_pattern)

    # LRU: evict oldest access → B
    _assert_only_evicted(run_test('lru'), 'B')

    # LFU: evict lowest hit count (D=2) → D
    _assert_only_evicted(run_test('lfu'), 'D')

    # FIFO: evict earliest inserted → A
    _assert_only_evicted(run_test('fifo'), 'A')

    # MRU: evict most recently accessed → A
    _assert_only_evicted(run_test('mru'), 'A')

    # FILO: evict most recently inserted → D
    _assert_only_evicted(run_test('filo'), 'D')


def test_eviction_policy_edge_cases(engine_cls):
    """Additional edge cases for eviction policies.

    Each sub-test uses a different access pattern to stress a specific aspect.
    """
    def run_test(policy, access_pattern):
        seqs = _make_seqs()
        engine = _create_engine(engine_cls, policy)
        return _insert_and_access(engine, seqs, access_pattern)

    # LFU tie-break: equal hit counts → LRU tie-breaker (oldest access evicted)
    # A×2, B×2, C×2, D×2 — A accessed earliest → A evicted
    _assert_only_evicted(
        run_test('lfu', [(0, 2), (1, 2), (2, 2), (3, 2)]), 'A')

    # FIFO ignores access: A accessed 100 times, still evicted (inserted first)
    _assert_only_evicted(
        run_test('fifo', [(0, 100), (1, 1), (2, 1), (3, 1)]), 'A')

    # FILO ignores access: D accessed 100 times, still evicted (inserted last)
    _assert_only_evicted(
        run_test('filo', [(0, 1), (1, 1), (2, 1), (3, 100)]), 'D')

    # MRU different order: A→C→B→D, most recent = D → D evicted
    _assert_only_evicted(
        run_test('mru', [(0, 5), (2, 3), (1, 4), (3, 2)]), 'D')


# ---- Verify all 5 policies are accepted without error ----
@pytest.mark.parametrize("policy", ['lru', 'lfu', 'fifo', 'mru', 'filo'])
def test_eviction_policy_valid_creation(engine_cls, policy: str):
    """All five policies should be accepted and produce a working engine."""
    engine = _create_engine(engine_cls, policy)
    seqs = _make_seqs(2)
    engine.insert(seqs[0], engine.take(1), is_ready=True)
    mr = engine.match(seqs[0])
    assert mr.num_matched_blocks == 1


# ---- Verify invalid policy raises ValueError ----
@pytest.mark.parametrize("bad_policy", ['random', 'LRU', 'Lfu', '', 'unknown', 'priority_wrong'])
def test_eviction_policy_invalid(engine_cls, bad_policy: str):
    """Invalid eviction policy strings should raise ValueError."""
    with pytest.raises((ValueError, RuntimeError)):
        _create_engine(engine_cls, bad_policy)


def test_eviction_policy_consecutive(engine_cls):
    """Test eviction priority ordering by varying evict_ratio.

    Setup:
      4 blocks total.  Insert A, B, C, D then apply an access pattern.
      Run 3 independent experiments with evict_ratio = 0.25, 0.5, 0.75
      (evicting 1, 2, 3 blocks respectively).  By diffing the cumulative
      evicted sets we derive the eviction *order*.

    Access pattern: B×4, C×3, D×2, A×5
      LRU order (oldest access first): B → C → D
      LFU order (lowest hits first):   D → C → B
      FIFO order (earliest insert):    A → B → C
      MRU order (newest access first): A → D → C
      FILO order (latest insert):      D → C → B

    Using separate evict_ratio values avoids the problem of newly inserted
    sequences participating in subsequent eviction rounds.
    """
    access_pattern = [(1, 4), (2, 3), (3, 2), (0, 5)]
    labels = ['A', 'B', 'C', 'D']

    expected_eviction_order = {
        'lru':  ['B', 'C', 'D'],
        'lfu':  ['D', 'C', 'B'],
        'fifo': ['A', 'B', 'C'],
        'mru':  ['A', 'D', 'C'],
        'filo': ['D', 'C', 'B'],
    }

    def _setup_and_evict(policy, evict_ratio):
        """Create a fresh engine with the given *evict_ratio*, insert A-D,
        apply access pattern, then insert E to trigger eviction.
        Return the set of A-D labels that were evicted."""
        seqs = _make_seqs(5)
        engine = engine_cls(
            device_type=DEFAULT_DEVICE_TYPE,
            num_total_blocks=_EVICT_NUM_TOTAL_BLOCKS,
            tokens_per_block=_EVICT_TOKENS_PER_BLOCK,
            evict_ratio=evict_ratio,
            eviction_policy=policy,
        )

        for seq in seqs[:4]:
            engine.insert(seq, engine.take(1), is_ready=True)
            time.sleep(0.002)

        for idx, count in access_pattern:
            for _ in range(count):
                engine.match(seqs[idx])
            time.sleep(0.002)

        # Insert E → triggers eviction
        engine.insert(seqs[4], engine.take(1), is_ready=True)

        evicted = set()
        for i, label in enumerate(labels):
            if engine.match(seqs[i]).num_matched_blocks == 0:
                evicted.add(label)
        return evicted

    for policy, expected in expected_eviction_order.items():
        # 3 independent experiments: evict 1, 2, 3 blocks respectively
        evicted_1 = _setup_and_evict(policy, 0.25)   # floor(4*0.25)=1
        evicted_2 = _setup_and_evict(policy, 0.50)   # floor(4*0.50)=2
        evicted_3 = _setup_and_evict(policy, 0.75)   # floor(4*0.75)=3

        # Derive ordered eviction list by diffing the cumulative sets
        order = []
        assert len(evicted_1) == 1, (
            f"policy={policy}: evict_ratio=0.25 should evict 1, got {evicted_1}")
        order.append(evicted_1.pop())

        diff2 = evicted_2 - {order[0]}
        assert len(diff2) == 1, (
            f"policy={policy}: evict_ratio=0.50 should evict 1 more, got {diff2}")
        order.append(diff2.pop())

        diff3 = evicted_3 - set(order)
        assert len(diff3) == 1, (
            f"policy={policy}: evict_ratio=0.75 should evict 1 more, got {diff3}")
        order.append(diff3.pop())

        assert order == expected, (
            f"policy={policy}: expected eviction order {expected}, got {order}"
        )


def test_eviction_policy_batch(engine_cls):
    """Test batch eviction with a larger evict_ratio.

    Setup:
      4 blocks total, evict_ratio=0.5 → evict 2 blocks at once when full.
      Insert A, B, C, D.  Then insert E (needs 1 block, but evicts 2).
      Verify the 2 evicted nodes are the lowest-priority pair.

    Access pattern: B×4, C×3, D×2, A×5
      LRU:  evict B, C  (oldest access)
      LFU:  evict D, C  (lowest hit count)
      FIFO: evict A, B  (earliest inserted)
      MRU:  evict A, D  (most recent access)
      FILO: evict D, C  (most recently inserted)
    """
    access_pattern = [(1, 4), (2, 3), (3, 2), (0, 5)]
    labels = ['A', 'B', 'C', 'D', 'E']

    expected_evicted = {
        'lru':  {'B', 'C'},
        'lfu':  {'D', 'C'},
        'fifo': {'A', 'B'},
        'mru':  {'A', 'D'},
        'filo': {'D', 'C'},
    }

    for policy, evicted_set in expected_evicted.items():
        seqs = _make_seqs(5)
        engine = engine_cls(
            device_type=DEFAULT_DEVICE_TYPE,
            num_total_blocks=_EVICT_NUM_TOTAL_BLOCKS,
            tokens_per_block=_EVICT_TOKENS_PER_BLOCK,
            evict_ratio=0.5,  # Evict 2 blocks at once
            eviction_policy=policy,
        )

        # Insert A, B, C, D
        for seq in seqs[:4]:
            engine.insert(seq, engine.take(1), is_ready=True)
            time.sleep(0.002)

        # Apply access pattern
        for idx, count in access_pattern:
            for _ in range(count):
                engine.match(seqs[idx])
            time.sleep(0.002)

        # Insert E → triggers eviction of 2 blocks
        engine.insert(seqs[4], engine.take(1), is_ready=True)

        result = {
            labels[i]: engine.match(seqs[i]).num_matched_blocks
            for i in range(5)
        }

        actually_evicted = {lbl for lbl, m in result.items() if m == 0}
        survived = {lbl for lbl, m in result.items() if m == 1}

        assert actually_evicted == evicted_set, (
            f"policy={policy}: expected evicted {evicted_set}, "
            f"got evicted {actually_evicted}, result={result}"
        )
        # E should always survive
        assert 'E' in survived, (
            f"policy={policy}: E should survive, result={result}"
        )


def test_eviction_policy_reinsert_after_eviction(engine_cls):
    """Verify that a sequence evicted can be re-inserted and matched correctly.

    Setup:
      4 blocks total, evict_ratio=0.25, LRU policy.
      Insert A, B, C, D → access B, C, D, A → insert E (evicts B).
      Then re-insert B and verify it can be matched.
    """
    seqs = _make_seqs(5)
    engine = _create_engine(engine_cls, 'lru')

    # Insert A, B, C, D
    for seq in seqs[:4]:
        engine.insert(seq, engine.take(1), is_ready=True)
        time.sleep(0.002)

    # Access pattern: make B the oldest accessed
    for idx, count in [(1, 4), (2, 3), (3, 2), (0, 5)]:
        for _ in range(count):
            engine.match(seqs[idx])
        time.sleep(0.002)

    # Insert E → evicts B (LRU)
    engine.insert(seqs[4], engine.take(1), is_ready=True)
    assert engine.match(seqs[1]).num_matched_blocks == 0, "B should be evicted"

    # Now evict another to make room, then re-insert B
    # Access so that C becomes LRU candidate
    engine.match(seqs[0])
    engine.match(seqs[3])
    engine.match(seqs[4])
    time.sleep(0.002)

    # Re-insert B (this triggers eviction of C, the current LRU)
    engine.insert(seqs[1], engine.take(1), is_ready=True)

    # B should now be matchable
    assert engine.match(seqs[1]).num_matched_blocks == 1, (
        "Re-inserted B should be matchable"
    )
    # C should have been evicted
    assert engine.match(seqs[2]).num_matched_blocks == 0, (
        "C should be evicted to make room for re-inserted B"
    )
