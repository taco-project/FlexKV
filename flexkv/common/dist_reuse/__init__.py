"""Distributed-reuse data structures and wire protocols.

This subpackage is **pure Python** and free of GPU / C++ dependencies, so it
can be imported on CPU-only nodes (e.g. CI workers, lightweight test
environments) without pulling in ``flexkv.c_ext``.  See design doc
``docs/dist_reuse/dist_reuse_with_cp_pp_multinode_tp_simplified.md`` and
``docs/dist_reuse/proposal_unify_with_graph_dispatch_2026-05-15.md``.

Phase D-4: ``CoordinationCoordinator`` / ``RemoteCoordHandler`` /
``BlockIndex`` / ``CoordQueryMsg`` / ``CoordGetCmdMsg`` / ``CoordPutCmdMsg``
(plus their ack siblings) were deleted in this refactor.  Per-SD
coordination is now expressed as multi-target ops on a single
``TransferOpGraph`` broadcast through the existing ``_launch_task``
graph-dispatch path; peer-SD acks come back via ``CompletedOp(sd_key,
contributing_node_id)`` to the master polling worker.
"""

from .sharing_domain import (
    DEFAULT_MODEL_ID,
    SharingDomainKey,
    derive_model_id,
)
from .sharing_domain_namespace import (
    INSTANCE_KEY_PREFIX,
    SD_KEY_PREFIX,
    SharingDomainNamespace,
)
from .aggregate_radix import (
    AggregateMatchResult,
    AggregateRadixTree,
    BlockNotTrackedError,
    ReadyEntry,
)
from .coordination_protocol import (
    CoordMsgType,
    EpochVerifyError,
    FailureReportMsg,
    RemoteReadyMsg,
    decode_coord_message,
    encode_coord_message,
)
from .failure_detector import (
    FailureDetector,
    InstanceSession,
    RedisSessionClient,
    make_redis_client_from_cache_config,
    make_session_epoch,
)
from .remote_init import (
    BootstrapResult,
    RemoteDistReuseInitializer,
)
from .master_coordinator import (
    MasterCoordinator,
    SharingDomainHandleSpec,
    build_sharing_domain_handles,
    find_endpoint_for_sd,
    graph_needs_gpu_clear,
)


__all__ = [
    # sharing_domain
    "DEFAULT_MODEL_ID",
    "SharingDomainKey",
    "derive_model_id",
    # namespace
    "INSTANCE_KEY_PREFIX",
    "SD_KEY_PREFIX",
    "SharingDomainNamespace",
    # aggregate_radix
    "AggregateMatchResult",
    "AggregateRadixTree",
    "BlockNotTrackedError",
    "ReadyEntry",
    # coordination_protocol (Phase D-4: trimmed to RemoteReady + FailureReport)
    "CoordMsgType",
    "EpochVerifyError",
    "FailureReportMsg",
    "RemoteReadyMsg",
    "decode_coord_message",
    "encode_coord_message",
    # failure_detector
    "FailureDetector",
    "InstanceSession",
    "RedisSessionClient",
    "make_redis_client_from_cache_config",
    "make_session_epoch",
    # remote_init (Batch C: task 0-F)
    "BootstrapResult",
    "RemoteDistReuseInitializer",
    # master_coordinator
    "MasterCoordinator",
    "SharingDomainHandleSpec",
    "build_sharing_domain_handles",
    "find_endpoint_for_sd",
    "graph_needs_gpu_clear",
]
