from __future__ import annotations

from typing import Dict

from flexkv.transfer.compression.ans import ans_utils
from flexkv.transfer.compression.common.strategy import (
    CompressionStrategy,
    NullCompressionStrategy,
)


# "gpu_cpu_tp" is gone: GPUCPUTransferWorker now covers tp==1 as num_gpus==1,
# so there is one CPU<->GPU compressor, not one per fan-out width.
WORKER_KINDS = (
    "gpu_cpu",
    "cpu_ssd",
)


def _null_compressors() -> Dict[str, CompressionStrategy]:
    return {kind: NullCompressionStrategy() for kind in WORKER_KINDS}


def build_compressors(
    *,
    cpu_handle,
    ssd_handle,
    cache_config,
    model_config,
    gpu_handle_groups,
    layerwise_enabled: bool = False,
) -> Dict[str, CompressionStrategy]:
    enable_nvcomp = ans_utils.check_engine_nvcomp_enable(
        gpu_handle_groups,
        layerwise_enabled=layerwise_enabled,
        cpu_handle=cpu_handle,
    )
    if not enable_nvcomp:
        return _null_compressors()

    (cpu_table, cpu_table_tp,
     ssd_table, ssd_table_tp) = ans_utils.allocate_engine_size_tables(
        cpu_handle=cpu_handle,
        ssd_handle=ssd_handle,
        cache_config=cache_config,
        model_config=model_config,
    )

    from flexkv.transfer.compression.ans.ans_strategy import (
        NvcompCpuSsdStrategy,
        NvcompGpuCpuStrategy,
    )

    tp_size = model_config.effective_tp_size_per_node

    # allocate_engine_size_tables() returns exactly one of the two: the
    # canonical 3-D table (tp==1 or KV replicated across ranks) or the
    # per-rank 4-D one. The strategy checks the dim it got against the
    # worker's own shape, so handing it whichever exists is safe.
    compressors = _null_compressors()
    compressors["gpu_cpu"] = NvcompGpuCpuStrategy(
        cpu_size_table=(cpu_table if cpu_table is not None else cpu_table_tp))
    compressors["cpu_ssd"] = NvcompCpuSsdStrategy(
        cpu_size_table=cpu_table,
        ssd_size_table=ssd_table,
        cpu_size_table_tp=cpu_table_tp,
        ssd_size_table_tp=ssd_table_tp,
        tp_size=tp_size,
    )
    return compressors
