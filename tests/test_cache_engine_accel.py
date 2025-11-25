from multiprocessing import Value
import random

import pytest
import numpy as np

from flexkv.cache.mempool import Mempool
from flexkv.cache.cache_engine import CacheEngineAccel
from flexkv.common.transfer import DeviceType
from flexkv.common.block import SequenceMeta

@pytest.fixture
def cache_engine(request: pytest.FixtureRequest) -> CacheEngineAccel:
    param = request.param if hasattr(request, 'param') else {}
    default_config_kwargs = {
        'device_type': DeviceType.CPU,
        'num_total_blocks': 64,
        'tokens_per_block': 4,
        'evict_ratio': 0.05,
    }
    default_config_kwargs.update(param)
    return CacheEngineAccel(**default_config_kwargs)

@pytest.mark.parametrize(
    "config, should_raise",
    [
        ({'evict_ratio': 0.05, 'num_total_blocks': 64, 'tokens_per_block': 4, 'device_type': DeviceType.CPU}, False),
        ({'evict_ratio': 0.05, 'num_total_blocks': 0, 'tokens_per_block': 4, 'device_type': DeviceType.GPU}, True),
        ({'evict_ratio': 0.05, 'num_total_blocks': 64, 'tokens_per_block': 0, 'device_type': DeviceType.SSD}, True),
        ({'evict_ratio': 0.05, 'num_total_blocks': 64, 'tokens_per_block': 4, 'device_type': 'Unknown'}, True),
        ({'evict_ratio': 0.05, 'num_total_blocks': 64, 'tokens_per_block': 3, 'device_type': DeviceType.CPU}, True),
    ]
)
def test_config_init(config: dict, should_raise: bool):
    if should_raise:
        with pytest.raises(ValueError) as e:
            CacheEngineAccel(**config)
    else:
        engine = CacheEngineAccel(**config)
        assert isinstance(engine, CacheEngineAccel)

def test_mempool():
    mempool = Mempool(num_total_blocks=64)
    assert mempool.num_free_blocks == 64
    block_ids = mempool.allocate_blocks(16)
    assert isinstance(block_ids, np.ndarray)
    assert block_ids.dtype == np.int64
    assert block_ids.shape == (16,)
    assert mempool.num_free_blocks == 48
    mempool.recycle_blocks(block_ids)
    assert mempool.num_free_blocks == 64

    block_ids = np.concatenate([mempool.allocate_blocks(16),
                           mempool.allocate_blocks(16),
                           mempool.allocate_blocks(16),
                           mempool.allocate_blocks(16)])
    assert mempool.num_free_blocks == 0

    with pytest.raises(ValueError):
        mempool.allocate_blocks(1)

    mempool.recycle_blocks(block_ids)
    assert mempool.num_free_blocks == 64

    empty_blocks = mempool.allocate_blocks(0)
    assert empty_blocks.shape == (0, )
    assert empty_blocks.dtype == np.int64
    assert mempool.num_free_blocks == 64

    with pytest.raises(ValueError):
        mempool.allocate_blocks(-1)

    mempool.recycle_blocks(np.array([], dtype=np.int64))
    assert mempool.num_free_blocks == 64

    with pytest.raises(ValueError):
        mempool.recycle_blocks(np.array([1, 2, 3], dtype=np.int32))
    with pytest.raises(ValueError):
        mempool.recycle_blocks(np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError):
        mempool.recycle_blocks(np.array([[1, 2, 3]], dtype=np.int64))

def test_reset(cache_engine: CacheEngineAccel):
    cache_engine.reset()
    assert cache_engine.index.is_empty()
    assert cache_engine.mempool.num_used_blocks == 0

@pytest.mark.parametrize(
    "cache_engine",
    [
        {'num_total_blocks': 10000000, 'tokens_per_block': 1, 'device_type': DeviceType.CPU},
        {'num_total_blocks': 10000000, 'tokens_per_block': 16, 'device_type': DeviceType.CPU},
    ],
    indirect=True
)
@pytest.mark.parametrize(
    "num_insert",
    [100],
)
@pytest.mark.parametrize(
    "seq_len",
    [1, 10, 16, 32, 10000],
)
def test_match_and_insert(cache_engine: CacheEngineAccel, num_insert: int, seq_len: int):
    base_token_ids = np.random.randint(0, 10000, (seq_len, ), dtype=np.int64)
    base_num_blocks = seq_len // cache_engine.tokens_per_block
    cache_engine.insert(SequenceMeta(token_ids=base_token_ids,
                                     tokens_per_block=cache_engine.tokens_per_block),
                        np.arange(base_num_blocks, dtype=np.int64),
                        is_ready=True)
    cur_cached_blocks = base_num_blocks
    for i in range(num_insert):
        prefix_ratio = random.random()
        prefix_len = int(len(base_token_ids)*prefix_ratio)
        num_prefix_blocks = prefix_len // cache_engine.tokens_per_block
        token_ids = np.concatenate([base_token_ids[:prefix_len],
                               np.random.randint(10000 + i * seq_len,
                                             10000 + (i+1) * seq_len,
                                             (seq_len-prefix_len, ),
                                             dtype=np.int64)])
        insert_sequence_meta = SequenceMeta(token_ids=token_ids,
                                            tokens_per_block=cache_engine.tokens_per_block)
        match_result = cache_engine.match(insert_sequence_meta)
        assert match_result.num_ready_matched_blocks == num_prefix_blocks
        assert match_result.num_matched_blocks == num_prefix_blocks

        num_insert_blocks = insert_sequence_meta.num_blocks - num_prefix_blocks
        cache_engine.insert(insert_sequence_meta,
                            np.arange(num_insert_blocks, dtype=np.int64),
                            is_ready=True,
                            match_result=match_result)
        cur_cached_blocks += num_insert_blocks
        assert cache_engine.index.total_cached_blocks() == cur_cached_blocks

        match_result = cache_engine.match(insert_sequence_meta)
        assert match_result.num_matched_blocks == insert_sequence_meta.num_blocks
        assert match_result.num_ready_matched_blocks == insert_sequence_meta.num_blocks

@pytest.mark.parametrize(
    "cache_engine",
    [
        {'num_total_blocks': 100, 'tokens_per_block': 16, 'device_type': DeviceType.CPU},
    ],
    indirect=True
)
def test_take_and_recycle(cache_engine: CacheEngineAccel):
    num_total_blocks = cache_engine.num_total_blocks
    tokens_per_block = cache_engine.tokens_per_block
    seq_blocks = 10
    token_ids = np.random.randint(0, 10000, (seq_blocks * tokens_per_block, ), dtype=np.int64)
    sequence_meta = SequenceMeta(token_ids=token_ids,
                                 tokens_per_block=tokens_per_block)
    physical_blocks = cache_engine.take(seq_blocks)
    radixnode = cache_engine.insert(sequence_meta, physical_blocks, is_ready=True)
    assert cache_engine.index.total_cached_blocks() == seq_blocks

    empty_blocks = cache_engine.take(0)
    assert empty_blocks.shape == (0, )
    assert empty_blocks.dtype == np.int64

    with pytest.raises(ValueError):
        cache_engine.take(-1)
    with pytest.raises(RuntimeError):
        cache_engine.take(num_total_blocks, protected_node=radixnode, strict=True)

    physical_blocks2 = cache_engine.take(num_total_blocks, protected_node=radixnode, strict=False)
    assert physical_blocks2.shape == (num_total_blocks - seq_blocks, )
    assert physical_blocks2.dtype == np.int64

    cache_engine.recycle(physical_blocks2)

    cache_engine.lock_node(radixnode)
    with pytest.raises(RuntimeError):
        cache_engine.take(num_total_blocks, protected_node=radixnode, strict=True)
    cache_engine.unlock(radixnode)
    cache_engine.set_ready(radixnode, True, radixnode.size())

    physical_blocks = cache_engine.take(num_total_blocks, protected_node=None, strict=True)
    assert physical_blocks.shape == (num_total_blocks, )
    assert cache_engine.index.total_cached_blocks() == 0

@pytest.mark.parametrize(
    "cache_engine",
    [
        {'num_total_blocks': 100, 'tokens_per_block': 1, 'device_type': DeviceType.CPU},
    ],
    indirect=True
)
def test_cleanup(cache_engine: CacheEngineAccel):
    if cache_engine.tokens_per_block != 1:
        pytest.skip("tokens_per_block != 1")
    tokens_per_block = cache_engine.tokens_per_block
    token_ids_list = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64),
                      np.array([0, 1, 2, 3, 17, 15, 19, 20], dtype=np.int64),
                      np.array([0, 23, 22, 21], dtype=np.int64)]
    sequence_meta_list = [SequenceMeta(token_ids=token_ids,
                                      tokens_per_block=tokens_per_block)
                          for token_ids in token_ids_list]
    num_insert_blocks0 = sequence_meta_list[0].num_blocks
    radixnode0 = cache_engine.insert(sequence_meta_list[0],
                                     np.arange(num_insert_blocks0, dtype=np.int64),
                                     is_ready=False)
    cache_engine.lock_node(radixnode0)
    radixnode0_size = radixnode0.size()
    match_result = cache_engine.match(sequence_meta_list[1])
    num_insert_blocks1 = sequence_meta_list[1].num_blocks - match_result.num_matched_blocks
    radixnode1 = cache_engine.insert(sequence_meta_list[1],
                                     np.arange(num_insert_blocks1, dtype=np.int64),
                                     match_result=match_result,
                                     is_ready=False)
    cache_engine.lock_node(radixnode1)
    radixnode1_size = radixnode1.size()
    match_result = cache_engine.match(sequence_meta_list[2])
    num_insert_blocks2 = sequence_meta_list[2].num_blocks - match_result.num_matched_blocks
    radixnode2 = cache_engine.insert(sequence_meta_list[2],
                                     np.arange(num_insert_blocks2, dtype=np.int64),
                                     match_result=match_result,
                                     is_ready=False)
    cache_engine.lock_node(radixnode2)
    radixnode2_size = radixnode2.size()
    total_insert_blocks = num_insert_blocks0 + num_insert_blocks1 + num_insert_blocks2
    assert cache_engine.index.total_cached_blocks() == total_insert_blocks
    assert cache_engine.index.total_unready_blocks() == total_insert_blocks
    assert cache_engine.index.total_ready_blocks() == 0

    cache_engine.unlock(radixnode2)
    cache_engine.set_ready(radixnode2, True, radixnode2_size)
    assert cache_engine.index.total_ready_blocks() == num_insert_blocks2

    cache_engine.unlock(radixnode1)
    cache_engine.set_ready(radixnode1, True, radixnode1_size)
    assert cache_engine.index.total_ready_blocks() == num_insert_blocks1 + num_insert_blocks2

    cache_engine.unlock(radixnode0)
    cache_engine.set_ready(radixnode0, True, radixnode0_size)
    assert cache_engine.index.total_ready_blocks() == num_insert_blocks0 + num_insert_blocks1 + num_insert_blocks2
