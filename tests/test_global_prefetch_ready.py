from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.cache.radixtree import MatchResult
from flexkv.common.block import SequenceMeta
from flexkv.common.transfer import TransferType


pytestmark = pytest.mark.unit


def _match(blocks=(), *, last_node=None):
    physical_blocks = np.asarray(blocks, dtype=np.int64)
    return MatchResult(
        num_ready_matched_blocks=len(physical_blocks),
        num_matched_blocks=len(physical_blocks),
        last_ready_node=last_node,
        last_node=last_node,
        physical_blocks=physical_blocks,
    )


def _prefetch_engine(*, use_mooncake: bool):
    engine = GlobalCacheEngine.__new__(GlobalCacheEngine)
    engine.cache_config = SimpleNamespace(
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=True,
        enable_kv_sharing=False,
    )
    engine.index_accel = False
    engine.use_mooncake_store_backend = use_mooncake
    engine._metrics_collector = None

    cpu_root = MagicMock()
    remote_node = MagicMock()
    cpu_node = MagicMock()
    cpu_node.size.return_value = 2
    engine.cpu_cache_engine = MagicMock()
    engine.cpu_cache_engine.take.return_value = np.asarray([10, 11], dtype=np.int64)
    engine.cpu_cache_engine.insert.return_value = cpu_node
    engine.ssd_cache_engine = None
    engine.remote_cache_engine = MagicMock()
    engine.match_all = MagicMock(
        return_value=(
            _match(last_node=cpu_root),
            _match(),
            _match([20, 21], last_node=remote_node),
        )
    )
    return engine, cpu_node


def _plan_remote_prefetch(engine):
    sequence = SequenceMeta(
        token_ids=np.arange(32, dtype=np.int64),
        tokens_per_block=16,
    )
    strategy = SimpleNamespace(ignore_gpu=True, ignore_remote=False)
    return engine._get_impl_global(
        request_id=1,
        sequence_meta=sequence,
        block_mask_start=0,
        block_mask_end=2,
        gpu_block_ids=np.asarray([0, 0], dtype=np.int64),
        temp_cache_strategy=strategy,
        dp_client_id=0,
    )


def test_remote_prefetch_publishes_cpu_node_and_waits_for_remote2h():
    engine, cpu_node = _prefetch_engine(use_mooncake=False)
    plan = _plan_remote_prefetch(engine)

    remote2h = [
        op
        for op in plan.transfer_graph._op_map.values()
        if op.transfer_type == TransferType.REMOTE2H
    ]
    assert len(remote2h) == 1
    assert plan.finished_ops_ids == [remote2h[0].op_id]
    assert remote2h[0].op_id in plan.op_callback_dict
    assert not plan.deferred_inserts

    plan.op_callback_dict[remote2h[0].op_id]()
    engine.cpu_cache_engine.set_ready.assert_called_once_with(cpu_node, True, 2)


def test_mooncake_remote_prefetch_defers_cpu_insert():
    engine, _cpu_node = _prefetch_engine(use_mooncake=True)
    plan = _plan_remote_prefetch(engine)

    remote2h = [
        op
        for op in plan.transfer_graph._op_map.values()
        if op.transfer_type == TransferType.REMOTE2H
    ]
    assert len(remote2h) == 1
    assert plan.finished_ops_ids == [remote2h[0].op_id]
    assert remote2h[0].op_id in plan.op_callback_dict
    engine.cpu_cache_engine.insert.assert_not_called()
    assert len(plan.deferred_inserts) == 1
    assert plan.deferred_inserts[0].load_result is not None
