import random

import pytest
import torch

from flexkv.cache.mempool import Mempool
from flexkv.cache.cache_engine import CacheEngine
from flexkv.common.transfer import DeviceType
from flexkv.common.exceptions import InvalidConfigError, NotEnoughSpaceError
from flexkv.common.block import SequenceMeta

@pytest.fixture
def cache_engine(request: pytest.FixtureRequest) -> CacheEngine:
    param = request.param if hasattr(request, 'param') else {}
    default_config_kwargs = {
        'device_type': DeviceType.CPU,
        'num_total_blocks': 64,
        'tokens_per_block': 4,
    }
    default_config_kwargs.update(param)
    return CacheEngine(**default_config_kwargs)

@pytest.mark.parametrize(
    "config, should_raise",
    [
        ({'num_total_blocks': 64, 'tokens_per_block': 4, 'device_type': DeviceType.CPU}, False),
        ({'num_total_blocks': 0, 'tokens_per_block': 4, 'device_type': DeviceType.GPU}, True),
        ({'num_total_blocks': 64, 'tokens_per_block': 0, 'device_type': DeviceType.SSD}, True),
        ({'num_total_blocks': 64, 'tokens_per_block': 4, 'device_type': 'Unknown'}, True),
        ({'num_total_blocks': 64, 'tokens_per_block': 3, 'device_type': DeviceType.CPU}, True),
    ]
)
def test_config_init(config: dict, should_raise: bool):
    if should_raise:
        with pytest.raises(InvalidConfigError) as e:
            CacheEngine(**config)
    else:
        engine = CacheEngine(**config)
        assert isinstance(engine, CacheEngine)

def test_mempool():
    mempool = Mempool(num_total_blocks=64)
    assert mempool.num_free_blocks == 64
    block_ids = mempool.allocate_blocks(16)
    assert isinstance(block_ids, torch.Tensor)
    assert block_ids.dtype == torch.int64
    assert block_ids.shape == (16,)
    assert mempool.num_free_blocks == 48
    mempool.recycle_blocks(block_ids)
    assert mempool.num_free_blocks == 64

    block_ids = torch.cat([mempool.allocate_blocks(16),
                           mempool.allocate_blocks(16),
                           mempool.allocate_blocks(16),
                           mempool.allocate_blocks(16)])
    assert mempool.num_free_blocks == 0

    with pytest.raises(NotEnoughSpaceError):
        mempool.allocate_blocks(1)

    mempool.recycle_blocks(block_ids)
    assert mempool.num_free_blocks == 64

    with pytest.raises(ValueError):
        mempool.allocate_blocks(0)
    assert mempool.num_free_blocks == 64

    with pytest.raises(ValueError):
        mempool.allocate_blocks(-1)

    mempool.recycle_blocks(torch.tensor([], dtype=torch.int64))
    assert mempool.num_free_blocks == 64

    with pytest.raises(ValueError):
        mempool.recycle_blocks(torch.tensor([1, 2, 3], dtype=torch.int32))
    with pytest.raises(ValueError):
        mempool.recycle_blocks(torch.tensor([1, 2, 3], dtype=torch.int64))
    with pytest.raises(ValueError):
        mempool.recycle_blocks(torch.tensor([[1, 2, 3]], dtype=torch.int64))

def test_reset(cache_engine: CacheEngine):
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
def test_match_and_insert(cache_engine: CacheEngine, num_insert: int, seq_len: int):
    base_token_ids = torch.randint(0, 10000, (seq_len, ), dtype=torch.int64)
    base_num_blocks = seq_len // cache_engine.tokens_per_block
    cache_engine.insert(SequenceMeta(token_ids=base_token_ids,
                                     tokens_per_block=cache_engine.tokens_per_block),
                        torch.arange(base_num_blocks, dtype=torch.int64),
                        is_ready=True)
    cur_cached_blocks = base_num_blocks
    for i in range(num_insert):
        prefix_ratio = random.random()
        prefix_len = int(len(base_token_ids)*prefix_ratio)
        num_prefix_blocks = prefix_len // cache_engine.tokens_per_block
        token_ids = torch.cat([base_token_ids[:prefix_len],
                               torch.randint(10000 + i * seq_len,
                                             10000 + (i+1) * seq_len,
                                             (seq_len-prefix_len, ),
                                             dtype=torch.int64)])
        insert_sequence_meta = SequenceMeta(token_ids=token_ids,
                                            tokens_per_block=cache_engine.tokens_per_block)
        match_result = cache_engine.match(insert_sequence_meta)
        assert match_result.num_ready_matched_blocks == num_prefix_blocks
        assert match_result.num_matched_blocks == num_prefix_blocks
        assert match_result.last_ready_node is not None
        assert match_result.last_node is not None
        assert match_result.physical_blocks.shape == (num_prefix_blocks, )
        assert match_result.physical_blocks.dtype == torch.int64

        num_insert_blocks = insert_sequence_meta.num_blocks - num_prefix_blocks
        cache_engine.insert(insert_sequence_meta,
                            torch.arange(num_insert_blocks, dtype=torch.int64),
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
def test_take_and_recycle(cache_engine: CacheEngine):
    num_total_blocks = cache_engine.num_total_blocks
    tokens_per_block = cache_engine.tokens_per_block
    seq_blocks = 10
    token_ids = torch.randint(0, 10000, (seq_blocks * tokens_per_block, ), dtype=torch.int64)
    sequence_meta = SequenceMeta(token_ids=token_ids,
                                 tokens_per_block=tokens_per_block)
    physical_blocks = cache_engine.take(seq_blocks)
    radixnode = cache_engine.insert(sequence_meta, physical_blocks, is_ready=True)
    assert cache_engine.index.total_cached_blocks() == seq_blocks

    with pytest.raises(ValueError):
        cache_engine.take(0)
    with pytest.raises(ValueError):
        cache_engine.take(-1)
    with pytest.raises(NotEnoughSpaceError):
        cache_engine.take(num_total_blocks, protected_node=radixnode, strict=True)

    physical_blocks2 = cache_engine.take(num_total_blocks, protected_node=radixnode, strict=False)
    assert physical_blocks2.shape == (num_total_blocks - seq_blocks, )
    assert physical_blocks2.dtype == torch.int64

    cache_engine.recycle(physical_blocks2)

    cache_engine.lock_node(radixnode)
    with pytest.raises(NotEnoughSpaceError):
        cache_engine.take(num_total_blocks, protected_node=radixnode, strict=True)
    cache_engine.cleanup(radixnode, radixnode.size())

    physical_blocks = cache_engine.take(num_total_blocks, protected_node=None, strict=True)
    assert physical_blocks.shape == (num_total_blocks, )
    assert cache_engine.index.total_cached_blocks() == 0
    assert radixnode.parent is None

@pytest.mark.parametrize(
    "cache_engine",
    [
        {'num_total_blocks': 100, 'tokens_per_block': 1, 'device_type': DeviceType.CPU},
    ],
    indirect=True
)
def test_cleanup(cache_engine: CacheEngine):
    if cache_engine.tokens_per_block != 1:
        pytest.skip("tokens_per_block != 1")
    tokens_per_block = cache_engine.tokens_per_block
    token_ids_list = [torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64),
                      torch.tensor([0, 1, 2, 3, 17, 15, 19, 20], dtype=torch.int64),
                      torch.tensor([0, 23, 22, 21], dtype=torch.int64)]
    sequence_meta_list = [SequenceMeta(token_ids=token_ids,
                                      tokens_per_block=tokens_per_block)
                          for token_ids in token_ids_list]
    num_insert_blocks0 = sequence_meta_list[0].num_blocks
    radixnode0 = cache_engine.insert(sequence_meta_list[0],
                                     torch.arange(num_insert_blocks0, dtype=torch.int64),
                                     is_ready=False)
    cache_engine.lock_node(radixnode0)
    radixnode0_size = radixnode0.size()
    match_result = cache_engine.match(sequence_meta_list[1])
    num_insert_blocks1 = sequence_meta_list[1].num_blocks - match_result.num_matched_blocks
    radixnode1 = cache_engine.insert(sequence_meta_list[1],
                                     torch.arange(num_insert_blocks1, dtype=torch.int64),
                                     match_result=match_result,
                                     is_ready=False)
    cache_engine.lock_node(radixnode1)
    radixnode1_size = radixnode1.size()
    match_result = cache_engine.match(sequence_meta_list[2])
    num_insert_blocks2 = sequence_meta_list[2].num_blocks - match_result.num_matched_blocks
    radixnode2 = cache_engine.insert(sequence_meta_list[2],
                                     torch.arange(num_insert_blocks2, dtype=torch.int64),
                                     match_result=match_result,
                                     is_ready=False)
    cache_engine.lock_node(radixnode2)
    radixnode2_size = radixnode2.size()
    total_insert_blocks = num_insert_blocks0 + num_insert_blocks1 + num_insert_blocks2
    assert cache_engine.index.total_cached_blocks() == total_insert_blocks
    assert cache_engine.index.total_unready_blocks() == total_insert_blocks
    assert cache_engine.index.total_ready_blocks() == 0

    cache_engine.cleanup(radixnode2, radixnode2_size)
    assert cache_engine.index.total_ready_blocks() == num_insert_blocks2

    cache_engine.cleanup(radixnode1, radixnode1_size)
    assert cache_engine.index.total_ready_blocks() == num_insert_blocks1 + num_insert_blocks2

    cache_engine.cleanup(radixnode0, radixnode0_size)
    assert cache_engine.index.total_ready_blocks() == num_insert_blocks0 + num_insert_blocks1 + num_insert_blocks2
