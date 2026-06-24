from __future__ import annotations

from typing import Dict

from flexkv.transfer.compression.ans import ans_utils
from flexkv.transfer.compression.common.strategy import (
    CompressionStrategy,
    NullCompressionStrategy,
)


WORKER_KINDS = (
    "gpu_cpu",
    "gpu_cpu_tp",
    "cpu_ssd",
    "indexer_gpu_cpu",
    "indexer_gpu_cpu_tp",
    "indexer_cpu_ssd",
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
        NvcompGpuCpuTpStrategy,
    )

    is_mla = cpu_handle.kv_layout.is_mla
    tp_size = model_config.effective_tp_size_per_node

    compressors = _null_compressors()
    compressors["gpu_cpu"] = NvcompGpuCpuStrategy(cpu_size_table=cpu_table)
    compressors["gpu_cpu_tp"] = NvcompGpuCpuTpStrategy(
        cpu_size_table_tp=(cpu_table if is_mla else cpu_table_tp))
    compressors["cpu_ssd"] = NvcompCpuSsdStrategy(
        cpu_size_table=cpu_table,
        ssd_size_table=ssd_table,
        cpu_size_table_tp=cpu_table_tp,
        ssd_size_table_tp=ssd_table_tp,
        tp_size=tp_size,
    )
    return compressors
