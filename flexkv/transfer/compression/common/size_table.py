from typing import Tuple

import torch

from flexkv.common.debug import flexkv_logger
from flexkv.transfer.host_buffer import cudaHostRegister


def bind_size_table(
    table,
    label,
    *,
    register: bool = False,
    expected_dims: Tuple[int, ...] = (3, 4),
):
    """Return (ptr, rank_stride, block_stride, layer_stride) for a uint32 table."""
    if table is None:
        return 0, 0, 0, 0
    assert table.dtype == torch.uint32, f"{label} must be uint32"
    assert table.dim() in expected_dims, \
        f"{label}: expected dim in {expected_dims}, got {table.dim()}"
    if register:
        cudaHostRegister(table)
    if table.dim() == 3:
        rank_stride, block_stride, layer_stride = 0, table.stride(0), table.stride(1)
    else:
        rank_stride, block_stride, layer_stride = (
            table.stride(0), table.stride(1), table.stride(2))
    tbl_mb = table.numel() * table.element_size() / (1024 ** 2)
    flexkv_logger.info(
        f"{label}: shape={tuple(table.shape)} dtype=uint32 size={tbl_mb:.2f} MB")
    return table.data_ptr(), rank_stride, block_stride, layer_stride
