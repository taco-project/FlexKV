"""Metadata observation with real two-rank Gloo and no transfer endpoints."""

import copy
import tempfile
import time
from datetime import timedelta
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from test_sglang_read_only_prefix_probe import Manager, connector

from flexkv.integration.sglang.comm import FlexKVComm
from flexkv.integration.sglang.connector import FlexKVConnector

pytestmark = pytest.mark.unit


def _probe_trace(rank, store):
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=15))
    try:
        info = NS(model_config=NS(pp_size=1, attn_tp_size=2, attn_cp_size=1),
                  pp_rank=0, attn_tp_rank=rank, attn_cp_rank=0, pp_size_per_node=1)
        comm = FlexKVComm(info, rank, attn_tp_group=dist.group.WORLD,
                         world_group=dist.group.WORLD)
        manager = Manager(193) if rank == 0 else None
        cache = connector(manager, comm)
        original = copy.deepcopy((cache._pending_lookups, cache._pending_lookup_contexts,
                                  cache._prefetch_result_sessions))
        now = time.time()
        times = [0, now] if rank == 0 else [now + 10000, 0]
        plan = FlexKVConnector.pending_probe_plan(cache, times)
        assert plan["indices"] == [0]
        assert FlexKVConnector.peek_cpu_prefix_tokens(
            cache, range(256), np.ones(256), namespace=["tenant"]
        ) == 192
        if rank == 0:
            assert manager.calls[-1]["namespace"] == ["tenant"]
            assert manager.calls[-1]["cpu_only"] is True
            assert manager.cancels == [[42]]
            manager.hit = 0
        assert FlexKVConnector.peek_cpu_prefix_tokens(cache, range(64), np.ones(64)) == 0
        if rank == 0:
            manager.hit, manager.fail_sum = 64, True
        assert FlexKVConnector.peek_cpu_prefix_tokens(cache, range(64), np.ones(64)) is None
        if rank == 0:
            manager.fail_sum, manager.fail_cancel = False, True
            queries_before = len(manager.calls)
        for _ in range(3):
            assert FlexKVConnector.peek_cpu_prefix_tokens(cache, range(64), np.ones(64)) is None
        if rank == 0:
            assert len(manager.calls) == queries_before + 1
            assert cache._metadata_probe_task == 42

        # Reset cancellation failure must be broadcast before any rank exits.
        cache._launched_load_tids = []
        with pytest.raises(RuntimeError, match="metadata probe cancellation"):
            FlexKVConnector.reset(cache)
        if rank == 0:
            manager.fail_cancel = False
        assert FlexKVConnector.peek_cpu_prefix_tokens(cache, range(64), np.ones(64)) == 64
        assert (cache._pending_lookups, cache._pending_lookup_contexts,
                cache._prefetch_result_sessions) == original
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_two_rank_read_only_probe_and_cancellation_recovery():
    with tempfile.TemporaryDirectory(prefix="flexkv-probe-") as directory:
        mp.spawn(_probe_trace, args=(f"file://{directory}/store",), nprocs=2, join=True)
