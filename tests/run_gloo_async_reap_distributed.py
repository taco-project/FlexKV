"""Four-rank CPU Gloo: delayed receivers, async backpressure and framing.

Run with torchrun --standalone --nproc-per-node=4.
"""
import gc
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from flexkv.integration.sglang.comm import FlexKVComm


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=90))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == 4
        for inline in (0, 1024):
            comm = FlexKVComm.__new__(FlexKVComm)
            comm.world_rank = rank
            comm._world_cpu_group = None
            comm._async_works = []
            comm._reap_high = 4
            comm._REAP_MAX_DRAIN = 8
            comm._reap_calls = 0
            comm._reap_drained_total = 0
            comm._reap_stuck_total = 0
            comm._control_inline_bytes = inline
            dist.barrier()
            if rank:
                # Longer than the former 1 ms probe, inside a private test group.
                time.sleep(0.03 * rank)
            comparisons = 0
            for iteration in range(60):
                for kind in range(3):
                    value = {'iteration': iteration, 'kind': kind,
                             'payload': bytes(range(256)) * (8 if kind == 2 else 1)}
                    actual = comm._scatter_group(
                        value if rank == 0 else None, list(range(world)), rank == 0,
                        tag=100 + kind, blocking=False,
                    )
                    if rank:
                        assert actual == value
                        comparisons += 1
                if rank == 0 and iteration % 10 == 0:
                    gc.collect()
            for work in comm._async_works:
                work.wait()
            comm._async_works.clear()
            count = torch.tensor(comparisons, dtype=torch.int64)
            dist.all_reduce(count)
            assert count.item() == 540
            if rank == 0:
                assert comm._reap_calls > 0
                print(f'inline={inline}: delayed async reaper, 540 comparisons passed', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
