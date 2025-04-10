import torch
import time
from flexkv.core.kv_manager import KVManager
from flexkv.core.debug_utils import debuginfo

if __name__ == "__main__":
    debuginfo.set_level("INFO")
    num_users = 8
    num_turns = 2
    tokens_per_block = 16
    block_per_turn = 10
    total_seq_len = num_turns * block_per_turn * tokens_per_block
    all_users_conversation_seqs = [
        torch.randint(0, 1000, (total_seq_len,)) for _ in range(num_users)
    ]

    num_layers = 32
    num_blocks = total_seq_len * num_users // tokens_per_block
    num_kv_heads = 48
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

    gpu_physical_block_ids = [
        torch.tensor([], dtype=torch.int64) for _ in range(num_users)
    ]
    for turn_id in range(num_turns):
        print("\n")
        print(f"Turn {turn_id} starts")
        start_time = time.time()
        for user_id in range(num_users):
            token_seq_slice = all_users_conversation_seqs[user_id][
                0 : (turn_id + 1) * tokens_per_turn
            ]
            start_index = (
                turn_id * (num_blocks // num_turns) + user_id * block_per_turn
            )
            end_index = start_index + block_per_turn
            gpu_physical_block_ids[user_id] = torch.cat(
                [
                    gpu_physical_block_ids[user_id],
                    torch.arange(start_index, end_index),
                ]
            )
            token_mask = torch.ones_like(token_seq_slice, dtype=torch.bool)
            token_mask[-tokens_per_turn:] = False
            # print(f"Turn {turn_id} get request {user_id} "
            #      f"with token mask: {token_mask}")
            get_request_list.append(
                kvpool.async_get(
                    token_seq_slice,
                    token_mask,
                    gpu_physical_block_ids[user_id].pin_memory(),
                )
            )
        """
        while len(get_request_list) > 0:
            completed_responses = kvpool.get_results(get_request_list)
            for response in completed_responses:
                get_request_list.remove(response.request_id)
                print(f"Turn {turn_id} get request {response.request_id} "
                      f"completed with return totally read tokens: "
                      f"{response.return_mask.sum()}")
            time.sleep(0.001)
        """
        kvpool.wait_until_finished(get_request_list)
        get_request_list = []

        end_time = time.time()
        tranfered_data_in_GB = (
            num_users
            * block_per_turn
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

        start_time = time.time()
        for user_id in range(num_users):
            token_seq_slice = all_users_conversation_seqs[user_id][
                0 : (turn_id + 1) * tokens_per_turn
            ]
            block_ids = gpu_physical_block_ids[user_id].pin_memory()
            put_request_list.append(
                kvpool.async_put(
                    token_seq_slice,
                    token_mask=None,
                    gpu_physical_block_ids=block_ids,
                )
            )

        kvpool.wait_until_finished(put_request_list)
        put_request_list = []

        end_time = time.time()
        put_time = end_time - start_time
        print(
            f"Turn {turn_id} put time: {put_time:.6f} s,"
            f"data size: {tranfered_data_in_GB:.2f} GB,"
            f"bandwidth: {tranfered_data_in_GB / put_time:.2f} GB/s"
        )

    kvpool.shutdown()
    cpu_blocks = kvpool.get_physical_blocks()

    for i in range(num_layers):
        for j in range(num_blocks):
            gpu_tensor_k = gpu_blocks[i][0, j, :].flatten().cpu()
            gpu_tensor_v = gpu_blocks[i][1, j, :].flatten().cpu()
            cpu_tensor_k = cpu_blocks[i][0, j, :]
            cpu_tensor_v = cpu_blocks[i][1, j, :]
            assert torch.allclose(gpu_tensor_k, cpu_tensor_k)
            assert torch.allclose(gpu_tensor_v, cpu_tensor_v)
