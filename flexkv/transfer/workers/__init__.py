"""Transfer workers, one module per physical resource edge.

Each concrete worker owns one edge and nothing else:

    gpu_cpu.py   GPU <-> CPU
    cpu_ssd.py   CPU <-> local SSD  (io_uring)
    remote.py    CPU <-> remote     (PCFS / mooncake-store)
    gds.py       GPU <-> SSD        (GPUDirect Storage)
    nixl.py      CPU/GPU <-> file   (NIXL agent)
    peer.py      CPU <-> peer CPU / peer SSD (own ZMQ + Redis control plane)

``runtime.py`` holds what they share (``TransferWorkerBase``: process entry,
run loop, host pinning) and ``handle.py`` the parent-side ``WorkerHandle``.

Import direction is strictly one-way -- runtime does not import concrete
workers, and concrete workers do not import each other -- so a build missing
one edge's dependency cannot break the others.

This commit is a pure move: every symbol below is byte-identical to the one it
had in ``flexkv/transfer/worker.py``, which now re-exports them all, so both
import paths stay valid and no caller has to change.
"""

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.transfer import trace

# ``worker.py`` ran this at module scope, before any worker class was defined.
# The package is the equivalent point now: it is what every import path
# executes first, whichever module the caller actually wanted.
trace.configure(GLOBAL_CONFIG_FROM_ENV.enable_transfer_trace)

from flexkv.transfer.workers.cpu_ssd import CPUSSDDiskTransferWorker  # noqa: E402
from flexkv.transfer.workers.gds import GDSTransferWorker, tpGDSTransferWorker  # noqa: E402
from flexkv.transfer.workers.gpu_cpu import (  # noqa: E402
    GPUCPUTransferWorker,
    _validate_multi_group_chunk_layout,
    tpGPUCPUTransferWorker,
)
from flexkv.transfer.workers.handle import WorkerHandle  # noqa: E402
from flexkv.transfer.workers.nixl import NixlTransferWorker  # noqa: E402
from flexkv.transfer.workers.peer import PEER2CPUTransferWorker  # noqa: E402
from flexkv.transfer.workers.remote import (  # noqa: E402
    CPURemoteTransferWorker,
    MooncakeStoreTransferWorker,
    _register_mooncake_regions,
    _split_mooncake_registration_regions,
    _unregister_mooncake_regions,
)
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
]
