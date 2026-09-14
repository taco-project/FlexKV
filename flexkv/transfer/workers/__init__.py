"""Transfer workers, one module per physical resource edge.

Each concrete worker owns one edge and nothing else:

    gpu_cpu.py   GPU <-> CPU        (main KV + SWA pools, layerwise delivery)
    cpu_ssd.py   CPU <-> local SSD  (io_uring, or a StorageBackend)
    remote.py    CPU <-> remote     (PCFS / mooncake-store; backend-only)
    gds.py       GPU <-> SSD        (GPUDirect Storage)
    peer.py      CPU <-> peer CPU / peer SSD (own ZMQ + Redis control plane)

``runtime.py`` holds what they share (``TransferWorkerBase``: process entry,
run loop, host pinning, backend attach/run) and ``handle.py`` the parent-side
``WorkerHandle``.

Import direction is strictly one-way -- runtime does not import concrete
workers, and concrete workers do not import each other -- so a build missing
one edge's dependency cannot break the others.

``flexkv.transfer.worker`` re-exports every name here; both paths stay valid.
"""

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.transfer import trace

# Before any worker module is imported. ``worker.py`` did this at module scope,
# and ``handle.py`` reads ``trace._TRACE_ON`` on the submit path without going
# through ``runtime``, so the package -- which every import path executes first
# -- is the only place that covers both.
trace.configure(GLOBAL_CONFIG_FROM_ENV.enable_transfer_trace)

from flexkv.transfer.workers.cpu_ssd import CPUSSDDiskTransferWorker  # noqa: E402
from flexkv.transfer.workers.gds import GDSTransferWorker  # noqa: E402
from flexkv.transfer.workers.gpu_cpu import (  # noqa: E402
    GPUCPUTransferWorker,
    _validate_multi_group_chunk_layout,
)
from flexkv.transfer.workers.handle import WorkerHandle  # noqa: E402
from flexkv.transfer.workers.peer import PEER2CPUTransferWorker  # noqa: E402
from flexkv.transfer.workers.remote import CPURemoteTransferWorker  # noqa: E402
from flexkv.transfer.workers.runtime import (  # noqa: E402
    TransferWorkerBase,
    ensure_cuda_device,
    import_tensor_handles,
)

__all__ = [
    "CPURemoteTransferWorker",
    "CPUSSDDiskTransferWorker",
    "GDSTransferWorker",
    "GPUCPUTransferWorker",
    "_validate_multi_group_chunk_layout",
    "PEER2CPUTransferWorker",
    "TransferWorkerBase",
    "WorkerHandle",
    "ensure_cuda_device",
    "import_tensor_handles",
]
