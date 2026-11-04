"""Multi-rank Gloo smoke test for SGLang store ownership and control frames.

Run manually with::

    torchrun --standalone --nproc-per-node=3 \
        tests/run_sglang_store_protocol_distributed.py
"""

import json
from concurrent.futures import Future
from datetime import timedelta
import os
import pickle
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from flexkv.integration.sglang.comm import FlexKVComm, FlexKVScatterChannel
from flexkv.integration.sglang.connector import FlexKVConnector


class _FakeKVManager:
    def put_match_batch(self, requests):
        return [(17 + i, np.ones(len(req["token_ids"]), dtype=np.bool_), True)
                for i, req in enumerate(requests)]

    def put_match_batch_async(self, requests):
        self.pending = Future()
        self.requests = requests
        return self.pending

    def launch(self, **kwargs):
        del kwargs

    def launch_async(self, **kwargs):
        self.pending_launch = Future()
        return self.pending_launch

    def wait(self, task_ids, timeout, completely):
        assert completely and timeout == 0
        return {tid: SimpleNamespace(status=SimpleNamespace(value="success")) for tid in task_ids}


def _make_comm(rank: int, world_size: int, axis: str) -> FlexKVComm:
    rank_info = SimpleNamespace(
        model_config=SimpleNamespace(
            pp_size=world_size if axis == "pp" else 1,
            attn_tp_size=world_size if axis == "tp" else 1,
            attn_cp_size=world_size if axis == "cp" else 1,
        ),
        pp_rank=rank if axis == "pp" else 0,
        attn_tp_rank=rank if axis == "tp" else 0,
        attn_cp_rank=rank if axis == "cp" else 0,
        pp_size_per_node=world_size if axis == "pp" else 1,
    )
    return FlexKVComm(
        rank_info,
        rank,
        pp_group=dist.group.WORLD if axis == "pp" else None,
        attn_tp_group=dist.group.WORLD if axis == "tp" else None,
        attn_cp_group=dist.group.WORLD if axis == "cp" else None,
        world_group=dist.group.WORLD,
    )


def _check_control_frames(comm: FlexKVComm) -> None:
    inline_size = 4096 - 8
    boundary = {"blob": b"b" * inline_size}
    overhead = len(pickle.dumps(boundary)) - inline_size
    boundary["blob"] = boundary["blob"][:inline_size - overhead]
    overflow = {"blob": boundary["blob"] + b"x"}
    payloads = [
        {"tasks": [17, 18], "error": None},
        boundary,
        overflow,
        {"blob": bytes(range(256)) * 257, "tasks": [17, 18]},
    ]
    assert len(pickle.dumps(payloads[0])) < inline_size
    assert len(pickle.dumps(boundary)) == inline_size
    assert len(pickle.dumps(overflow)) == inline_size + 1
    assert len(pickle.dumps(payloads[3])) > 16 * inline_size

    for blocking in (False, True):
        if not blocking and comm.is_sync_leader:
            for payload in payloads:
                outgoing = dict(payload)
                assert comm.scatter(
                    outgoing, channel=FlexKVScatterChannel.LOOKUP,
                ) == payload
                # Queued transfers own the serialized snapshot, not this dict.
                outgoing.clear()
        if not blocking:
            # No receiver has entered scatter yet: submission must not wait
            # for a receive, including when the frame has an overflow tail.
            dist.barrier()
        if blocking or not comm.is_sync_leader:
            for payload in payloads:
                assert comm.scatter(
                    payload if comm.is_sync_leader else {},
                    channel=FlexKVScatterChannel.LOOKUP,
                    blocking=blocking,
                ) == payload
        dist.barrier()

    channels = [
        FlexKVScatterChannel.LOOKUP,
        FlexKVScatterChannel.STORE_START,
        FlexKVScatterChannel.PREFETCH_START,
        FlexKVScatterChannel.RESET,
    ]
    messages = list(zip(channels, payloads))
    if comm.is_sync_leader:
        for channel, payload in messages:
            comm.scatter(payload, channel=channel)
    dist.barrier()
    if not comm.is_sync_leader:
        # With three ranks, the followers consume in opposite orders. Rank 1
        # also receives opposite to the sender when run with only two ranks.
        ordered = reversed(messages) if comm.world_rank % 2 else messages
        for channel, payload in ordered:
            assert comm.scatter({}, channel=channel) == payload
    dist.barrier()

    if comm.pp_size > 1:
        # The PP-only sideband is untyped and uses its own existing P2P tag.
        sideband = ("pp-sideband", payloads, {1, 3}, None)
        if comm.is_sync_leader:
            assert comm.scatter_pp(sideband) == sideband
        dist.barrier()
        if not comm.is_sync_leader:
            assert comm.scatter_pp(None) == sideband
        dist.barrier()


def main() -> None:
    dist.init_process_group(backend="gloo", timeout=timedelta(seconds=60))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size >= 2, "Run this smoke test with at least two ranks"
    for axis in ("pp", "cp"):
        _check_control_frames(_make_comm(rank, world_size, axis))
    comm = _make_comm(rank, world_size, "tp")
    _check_control_frames(comm)

    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector._sync_ctx = comm
    connector.kv_manager = _FakeKVManager() if rank == 0 else None
    connector._inflight_stores = {}
    connector._inflight_store_contexts = {}
    connector._build_swa_slot_mapping = lambda _indices: None
    connector._log_cache_op = lambda *_args, **_kwargs: None

    task_id = connector.store_kv(
        "request",
        [1, 2, 3, 4],
        torch.arange(4, dtype=torch.int64),
    )
    assert task_id == 17
    assert connector._inflight_stores == {"request": 17}

    completed = connector.check_completed_stores()
    assert completed == ["request"]
    assert connector._inflight_stores == {}
    ready = ["one", "two"] if rank == 0 else ["one"]
    assert connector.sync_ready_store_rids(ready, require_all_ranks=True) == ["one"]
    stores = [
        dict(rid=rid, token_ids=[1, 2, 3, 4], kv_indices=torch.arange(4),
             swa_indices=torch.tensor([0, 0, 8, 9]))
        for rid in ("one", "two")
    ]
    assert connector.store_kv_batch(stores) == [17, 18]
    assert connector.check_completed_stores() == ["one", "two"]
    stores[0]["validation_error"] = "bad snapshot" if rank == 1 else ""
    try:
        connector.store_kv_batch(stores)
    except ValueError:
        pass
    else:
        raise AssertionError("rank-local validation must reject every rank")
    stores[0].pop("validation_error")
    admission = connector.submit_store_kv_batch(stores)
    assert connector.poll_store_kv_batch(admission) is None
    if rank == 0:
        connector.kv_manager.pending.set_result(
            connector.kv_manager.put_match_batch(connector.kv_manager.requests)
        )
    assert connector.poll_store_kv_batch(admission) is None
    assert not connector._inflight_stores
    if rank == 0:
        connector.kv_manager.pending_launch.set_result([17, 18])
    assert connector.poll_store_kv_batch(admission) == [17, 18]
    assert connector.check_completed_stores() == ["one", "two"]
    admission = connector.submit_store_kv_batch(stores)
    assert connector.poll_store_kv_batch(admission) is None
    if rank == 0:
        connector.kv_manager.pending.set_exception(RuntimeError("leader match failed"))
    try:
        connector.poll_store_kv_batch(admission)
    except RuntimeError as error:
        assert "leader match failed" in str(error)
    else:
        raise AssertionError("every rank must observe the leader's queued failure")
    assert connector._pending_store_batch is admission
    connector._drain_pending_store_batch()
    assert connector._pending_store_batch is None
    print(
        json.dumps(
            {
                "rank": rank,
                "task_id": task_id,
                "completed": completed,
                "tracked_after_completion": connector._inflight_stores,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    comm.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
