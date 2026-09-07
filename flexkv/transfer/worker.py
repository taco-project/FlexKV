"""Compatibility façade over the :mod:`flexkv.transfer.workers` package.

This module was a single 4345-line file holding every transfer worker. It was
split one module per resource edge; see ``flexkv/transfer/workers/__init__.py``
for the layout. The split was a pure move -- no symbol changed -- and this
module re-exports the workers, the helpers, and the module-scope ``trace``
handle, so both ``from flexkv.transfer.worker import X`` and the
``worker.trace``-style attribute access the tests use keep working.

New code should import from ``flexkv.transfer.workers``.
"""

from flexkv.transfer import trace  # noqa: F401
from flexkv.transfer.workers import (  # noqa: F401
    CPURemoteTransferWorker,
    CPUSSDDiskTransferWorker,
    GDSTransferWorker,
    GPUCPUTransferWorker,
    MooncakeStoreTransferWorker,
    NixlTransferWorker,
    PEER2CPUTransferWorker,
    TransferWorkerBase,
    WorkerHandle,
    _register_mooncake_regions,
    _split_mooncake_registration_regions,
    _unregister_mooncake_regions,
    _validate_multi_group_chunk_layout,
    ensure_cuda_device,
    import_tensor_handles,
    tpGDSTransferWorker,
    tpGPUCPUTransferWorker,
)

__all__ = [
    "CPURemoteTransferWorker",
    "CPUSSDDiskTransferWorker",
    "GDSTransferWorker",
    "GPUCPUTransferWorker",
    "MooncakeStoreTransferWorker",
    "NixlTransferWorker",
    "PEER2CPUTransferWorker",
    "TransferWorkerBase",
    "WorkerHandle",
    "_register_mooncake_regions",
    "_split_mooncake_registration_regions",
    "_unregister_mooncake_regions",
    "_validate_multi_group_chunk_layout",
    "ensure_cuda_device",
    "import_tensor_handles",
    "tpGDSTransferWorker",
    "tpGPUCPUTransferWorker",
    "trace",
]
