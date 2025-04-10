import torch
import time
from flexkv.core.kv_manager import KVManager
from flexkv.core.debug_utils import debuginfo

if __name__ == "__main__":
    debuginfo.set_level("INFO")

    tokens_per_block = 16
    block_per_request = 20
    seq_len = block_per_request * tokens_per_block
    req_seqs_1 = torch.randint(0, 100, (seq_len,))
    req_seqs_2 = torch.randint(0, 100, (seq_len,))

    cpu_block_num = block_per_request * 30 // 2
    ssd_block_num = block_per_request * 200

    num_layers = 32
    num_kv_heads = 8
    head_size = 128
    block_size = tokens_per_block * num_kv_heads * head_size
    dtype = torch.float16

    gpu_blocks = [
        torch.randn(
            2,
            block_per_request * 2,
            tokens_per_block,
            num_kv_heads,
            head_size,
            dtype=dtype,
        ).cuda()
        for _ in range(num_layers)
    ]

    gpu_blocks_gt = [block.clone().cuda() for block in gpu_blocks]

    print("prepare kvpool")
    kvpool = KVManager(
        num_cpu_blocks=cpu_block_num,
        num_ssd_blocks=ssd_block_num,
        tokens_per_block=tokens_per_block,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
        gpu_physical_blocks=gpu_blocks,
    )

    gpu_block_ids_1 = torch.arange(0, block_per_request, dtype=torch.int64)
    gpu_block_ids_2 = torch.arange(
        block_per_request, block_per_request * 2, dtype=torch.int64
    )

    request_1 = kvpool.async_put(
        token_ids=req_seqs_1,
        token_mask=None,
        gpu_physical_block_ids=gpu_block_ids_1.pin_memory(),
    )
    time.sleep(1)
    # kvpool.wait_until_finished([request_1])
    for i in range(30):
        request_2 = kvpool.async_put(
            token_ids=torch.randint(
                0, 100, (block_per_request * tokens_per_block,)
            ),
            token_mask=None,
            gpu_physical_block_ids=gpu_block_ids_2.pin_memory(),
        )
    time.sleep(1)
    # kvpool.wait_until_finished([request_2])

    debuginfo.set_level("INFO")

    request_3 = kvpool.async_get(
        token_ids=req_seqs_1,
        token_mask=None,
        gpu_physical_block_ids=gpu_block_ids_1.pin_memory(),
    )
    # time.sleep(0.5)

    print("all requests finished")
    kvpool.wait_until_finished([request_3])
    kvpool.shutdown()

    for i in range(num_layers):
        gpu_k = gpu_blocks[i][0, :block_per_request, :, :, :]
        gpu_v = gpu_blocks[i][1, :block_per_request, :, :, :]
        gpu_k_gt = gpu_blocks_gt[i][0, :block_per_request, :, :, :]
        gpu_v_gt = gpu_blocks_gt[i][1, :block_per_request, :, :, :]
        # print(f"gpu_k: {gpu_k.cpu()[0][0][0]}, "
        #      f"gpu_k_gt: {gpu_k_gt.cpu()[0][0][0]}")
        # print(f"gpu_v: {gpu_v.cpu()[0][0][0]}, "
        #      f"gpu_v_gt: {gpu_v_gt.cpu()[0][0][0]}")
        assert torch.allclose(gpu_k, gpu_k_gt)
        assert torch.allclose(gpu_v, gpu_v_gt)
