"""Two-rank Gloo smoke test for the SGLang store ownership protocol.

Run manually with::

    torchrun --standalone --nproc-per-node=2 \
        tests/run_sglang_store_protocol_distributed.py
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from flexkv.integration.sglang.comm import FlexKVComm
from flexkv.integration.sglang.connector import FlexKVConnector


class _FakeKVManager:
    def put_match(self, *, token_ids, token_mask):
        del token_mask
        return 17, np.ones(len(token_ids), dtype=np.bool_)

    def launch(self, **kwargs):
        del kwargs

    def try_wait(self, *, task_ids):
        assert task_ids == [17]
        return {17: SimpleNamespace(status=SimpleNamespace(value="success"))}


def main() -> None:
    dist.init_process_group(backend="gloo")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank_info = SimpleNamespace(
        model_config=SimpleNamespace(
            pp_size=1,
            attn_tp_size=world_size,
            attn_cp_size=1,
        ),
        pp_rank=0,
        attn_tp_rank=rank,
        attn_cp_rank=0,
        pp_size_per_node=1,
    )
    comm = FlexKVComm(
        rank_info,
        rank,
        attn_tp_group=dist.group.WORLD,
        world_group=dist.group.WORLD,
    )

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
