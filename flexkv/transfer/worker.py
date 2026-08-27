"""Compatibility façade over :mod:`flexkv.transfer.workers`.

The worker classes used to live here, in one file, and a lot of code -- inside
the repo and outside it -- imports them from this path. They now live one per
edge under ``flexkv/transfer/workers/``; this module re-exports them so both
paths keep working.

Two names below are *not* workers and are re-exported for a different reason:
``transfer_kv_blocks_remote`` and ``shared_transfer_kv_blocks_remote_read`` are
``c_ext`` entry points that ``backends.py`` imports from this module lazily
(inside ``PcfsRemoteBackend.attach``/``transfer``), and that tests monkeypatch
here. They stay module-level attributes of the façade, ``None`` when the build
has no CFS support, exactly as before.

New code should import from ``flexkv.transfer.workers``.
"""

# ``c_ext`` re-exports. Kept at module level, and imported the same defensive
# way as before: a build without FLEXKV_ENABLE_CFS=1 has no such symbols, and
# PcfsRemoteBackend.attach checks for None rather than catching ImportError.
try:
    from flexkv.c_ext import (
        shared_transfer_kv_blocks_remote_read,
        transfer_kv_blocks_remote,
    )
except ImportError:
    transfer_kv_blocks_remote = None
    shared_transfer_kv_blocks_remote_read = None

from flexkv.transfer.workers import (
    CPURemoteTransferWorker,
    CPUSSDDiskTransferWorker,
    GDSTransferWorker,
    GPUCPUTransferWorker,
    PEER2CPUTransferWorker,
    TransferWorkerBase,
    WorkerHandle,
    ensure_cuda_device,
    import_tensor_handles,
)

# Not workers either. These are the geometry check and the Mooncake external-MR
# helpers, which live beside the code that runs them (``workers/gpu_cpu.py``
# and ``backends.py``) but are re-exported here because the regression suites
# import them from this path.
from flexkv.transfer.backends import (
    _register_mooncake_regions,
    _split_mooncake_registration_regions,
    _unregister_mooncake_regions,
)
from flexkv.transfer.workers import _validate_multi_group_chunk_layout

__all__ = [
    "CPURemoteTransferWorker",
    "CPUSSDDiskTransferWorker",
    "GDSTransferWorker",
    "GPUCPUTransferWorker",
    "PEER2CPUTransferWorker",
    "TransferWorkerBase",
    "WorkerHandle",
    "ensure_cuda_device",
    "import_tensor_handles",
    "shared_transfer_kv_blocks_remote_read",
    "transfer_kv_blocks_remote",
    "_validate_multi_group_chunk_layout",
    "_register_mooncake_regions",
    "_split_mooncake_registration_regions",
    "_unregister_mooncake_regions",
]
