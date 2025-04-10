import torch
import time
from flexkv.core.kv_manager import KVManager
from flexkv.core.debug_utils import debuginfo

if __name__ == "__main__":
    debuginfo.set_level("INFO")
    num_turns = 4
    tokens_per_block = 16
    block_per_turn = 100
    total_seq_len = num_turns * block_per_turn * tokens_per_block
    conversation_seqs = torch.randint(0, 1000, (total_seq_len,))

    num_layers = 32
    num_blocks = total_seq_len // tokens_per_block
    num_kv_heads = 8
    head_size = 128
    block_size = tokens_per_block * num_kv_heads * head_size
    dtype = torch.float16

    all_blocks_size = num_layers * 2 * num_blocks * block_size * dtype.itemsize
    print("prepare blocks")
    print(f"total size of the gpu blocks is {all_blocks_size / 1e9:.2f} GB")

    gpu_blocks = [
        torch.randn(
            2,
            num_blocks,
            tokens_per_block,
            num_kv_heads,
            head_size,
            dtype=dtype,
        ).cuda()
        for _ in range(num_layers)
    ]

    print("prepare kvpool")
    kvpool = KVManager(
        num_cpu_blocks=num_blocks,
        num_ssd_blocks=num_blocks,
        tokens_per_block=tokens_per_block,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
        gpu_physical_blocks=[],
    )
    assert not kvpool.is_ready()
    kvpool.add_gpu_blocks(gpu_blocks)
    assert kvpool.is_ready()

    tokens_per_turn = block_per_turn * tokens_per_block

    print(
        f"total sequence length: {total_seq_len} "
        f"tokens per turn: {tokens_per_turn} "
        f"block per turn: {block_per_turn} "
        f"num layers: {num_layers} "
        f"num blocks: {num_blocks} "
    )

    get_request_list = []
    put_request_list = []

    for turn_id in range(num_turns):
        print("\n")
        print(f"Turn {turn_id} starts")
        start_time = time.time()

        token_seq_slice = conversation_seqs[0 : (turn_id + 1) * tokens_per_turn]

        gpu_block_ids = torch.arange(0, (turn_id + 1) * block_per_turn)

        # print(f"Turn {turn_id} get request {user_id} "
        #      f"with token mask: {token_mask}")
        get_request_list.append(
            kvpool.async_get(
                token_seq_slice,
                token_mask=None,
                gpu_physical_block_ids=gpu_block_ids.pin_memory(),
            )
        )

        kvpool.wait_until_finished(get_request_list)
        get_request_list = []

        end_time = time.time()
        tranfered_data_in_GB = (
            block_per_turn
            * tokens_per_block
            * num_kv_heads
            * 2
            * head_size
            * num_layers
            * dtype.itemsize
        ) / 1e9
        get_time = end_time - start_time
        print(
            f"Turn {turn_id} get time: {get_time:.6f} s,"
            f"data size: {tranfered_data_in_GB * turn_id:.2f} GB,"
            f"bandwidth:{tranfered_data_in_GB * turn_id / get_time:.2f} GB/s"
        )

    kvpool.shutdown()
