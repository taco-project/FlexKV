"""Real four-rank Gloo verification of FlexKVConnector load outcomes.

Run: torchrun --standalone --nproc-per-node=4 tests/run_sglang_load_failure_distributed.py
Uses the real hierarchical FlexKVComm; only the KVManager I/O is synthetic.
"""
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist

from flexkv.integration.sglang.comm import FlexKVComm
from flexkv.integration.sglang.connector import FlexKVConnector


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        assert dist.get_world_size() == 4
        for pp, tp, cp in ((1, 4, 1), (2, 2, 1), (1, 2, 2)):
            tp_group = None
            for first_rank in range(0, 4, tp):
                members = list(range(first_rank, first_rank + tp))
                group = dist.new_group(members, backend="gloo")
                if rank in members:
                    tp_group = group
            rank_info = SimpleNamespace(model_config=SimpleNamespace(pp_size=pp, attn_tp_size=tp, attn_cp_size=cp),
                pp_rank=rank // (cp * tp), attn_tp_rank=rank % tp,
                attn_cp_rank=(rank % (cp * tp)) // tp, pp_size_per_node=pp)
            comm = FlexKVComm(rank_info, rank, attn_tp_group=tp_group, world_group=dist.group.WORLD)
            for iteration in range(10):
                for case in ("success", "failed", "timeout", "launch_error", "wait_error", "unsafe"):
                    c = FlexKVConnector.__new__(FlexKVConnector)
                    c._sync_ctx = comm
                    c._pending_lookups = {"r": 17}
                    c._pending_lookup_contexts = {}
                    c._pop_context = Mock(return_value=SimpleNamespace(task_id=17))
                    c._log_cache_op = Mock()
                    c._to_cpu_int64 = lambda x: x
                    c._build_swa_slot_mapping = lambda x: None
                    c.kv_manager = Mock() if rank == 0 else None
                    c.tp_client = Mock()
                    c.tp_client.gpu_route_control.return_value = {"safe": case != "unsafe", "generation": iteration}
                    if rank == 0:
                        responses = [
                            {17: SimpleNamespace(status="success" if case == "success" else "failed")},
                            {17: SimpleNamespace(status="cancelled")},
                        ]
                        if case == "launch_error":
                            c.kv_manager.launch.side_effect = RuntimeError("launch failed")
                        elif case == "wait_error":
                            responses[0] = RuntimeError("wait failed")
                        elif case in ("timeout", "unsafe"):
                            responses[0] = {17: SimpleNamespace(status="timeout")}
                        c.kv_manager.wait.side_effect = responses
                    try:
                        n = c.retrieve_kv("r", torch.arange(8))
                        outcome = {"loaded": n, "bypass": c._cache_bypassed, "error": None}
                    except RuntimeError as exc:
                        outcome = {"loaded": None, "bypass": c._cache_bypassed, "error": str(exc)}
                    gathered = [None] * 4
                    dist.all_gather_object(gathered, outcome)
                    assert all(value == gathered[0] for value in gathered), (case, gathered)
                    if case == "success":
                        assert outcome == {"loaded": 8, "bypass": False, "error": None}
                    elif case == "unsafe":
                        assert "DMA drain" in outcome["error"]
                    else:
                        assert outcome == {"loaded": 0, "bypass": True, "error": None}
            for work in comm._async_works:
                work.wait()
            dist.barrier()
        dist.barrier()
        if rank == 0:
            print("PASS: 180 load outcomes agreed across TP4, PP2xTP2 and CP2xTP2", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
