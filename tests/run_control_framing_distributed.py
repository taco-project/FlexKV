"""Four-rank Gloo check for control framing and async send buffer lifetime.

Run: torchrun --standalone --nproc-per-node=4 tests/run_control_framing_distributed.py
No GPU, native extension, or Mooncake SDK is required.
"""

import gc
from datetime import timedelta

import torch
import torch.distributed as dist

from flexkv.integration.sglang.comm import FlexKVComm


def payload(iteration, kind):
    if kind == 0:
        return []
    if kind == 1:
        return {"task_id": iteration, "hit": 97280, "rid": str(iteration)}
    return {"iteration": iteration, "data": list(range(1024)), "raw": bytes(range(256))}


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == 4
        comm = FlexKVComm.__new__(FlexKVComm)
        comm.world_rank = rank
        comm._world_cpu_group = None
        comm._async_works = []
        comm._reap_high = 64
        comm._reap_calls = 0
        comm._reap_drained_total = 0
        for inline in [0, 256, 1024]:
            comm._control_inline_bytes = inline
            dist.barrier()
            checks = 0
            for iteration in range(200):
                for kind in range(3):
                    value = payload(iteration, kind)
                    actual = comm._scatter_group(
                        value if rank == 0 else None, list(range(world)), rank == 0, 100 + kind
                    )
                    if rank != 0:
                        assert actual == value, (rank, iteration, kind)
                        checks += 1
                if rank == 0 and iteration % 20 == 0:
                    gc.collect()
            for work in comm._async_works:
                work.wait()
            comm._async_works.clear()
            total = torch.tensor(checks, dtype=torch.int64)
            dist.all_reduce(total)
            assert total.item() == 1800
            if rank == 0:
                print(f"inline_bytes={inline}: {total.item()} receiver comparisons passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
