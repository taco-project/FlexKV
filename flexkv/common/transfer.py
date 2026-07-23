import threading
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import ClassVar, List, Set, Dict, Callable, Tuple, Optional

import numpy as np

from flexkv.common.debug import flexkv_logger


@dataclass(frozen=True)
class WorkerKey:
    """Immutable, hashable key that uniquely identifies a worker by
    ``(dp_client_id, pp_rank)``."""
    dp_client_id: int = 0
    pp_rank: int = 0


@dataclass(frozen=True)
class CompletedOp:
    graph_id: int
    op_id: int
    # Transfer metrics fields (populated when op completes, for post-completion metrics)
    transfer_type: Optional[str] = None
    num_blocks: int = 0
    num_bytes: int = 0

    def is_graph_completed(self) -> bool:
        return self.op_id == -1

    def to_tuple(self) -> Tuple[int, int]:
        return (self.graph_id, self.op_id)

    @classmethod
    def from_tuple(cls, data: Tuple[int, int]) -> 'CompletedOp':
        return cls(graph_id=data[0], op_id=data[1])

    @classmethod
    def completed_graph(cls, graph_id: int) -> 'CompletedOp':
        return cls(graph_id=graph_id, op_id=-1)


class DeviceType(IntEnum):
    CPU = 0
    GPU = 1
    SSD = 2
    REMOTE = 3
    PEERCPU = 4
    PEERSSD = 5

class TransferType(Enum):
    H2D    = "H2D"
    D2H    = "D2H"
    DISK2H = "DISK2H"
    H2DISK = "H2DISK"
    DISK2D = "DISK2D"
    D2DISK = "D2DISK"
    REMOTE2H = "REMOTE2H"
    H2REMOTE = "H2REMOTE"
    PEERH2H = "PEERH2H"
    H2PEERH = "H2PEERH"
    PEERSSD2H = "PEERSSD2H"
    H2PEERSSD = "H2PEERSSD"

    # if we need to return a results when trasnfer op 1 and op 2 are completed
    # we can add a virtual transfer op 3 that depends on op 1 and op 2
    # so that the op 3 will not be executed actually, but can indicate the completion of
    # a group of transfer ops
    VIRTUAL = "Virtual"
    LAYERWISE = "LAYERWISE"

# class DistType(Enum):
#     DISTH = "DISTH"
#     DISTSSD = "DISTSSD"

class PartitionBlockType(Enum):
    ROUND_ROBIN = 0
    SEQUENTIAL = 1

class TransferOpStatus(Enum):
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2

@dataclass
class TransferOp:
    _next_op_id: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    op_id: int = field(init=False)
    graph_id: int
    transfer_type: TransferType
    src_block_ids: np.ndarray
    dst_block_ids: np.ndarray
    # src_block_node_ids: Optional[np.ndarray] = None
    # this will change dynamically as transfer ops executed
    predecessors: Set[int] = field(default_factory=set)
    # this will keep the full info
    successors: Set[int] = field(default_factory=set)
    status: TransferOpStatus = TransferOpStatus.PENDING
    dp_client_id: int = 0
    # used for get block ids inner worker process
    src_slot_id: int = -1
    dst_slot_id: int = -1
    valid_block_num: int = 0
    remote_node_ids: Optional[np.ndarray] = None
    # used for distributed cpu and ssd
    src_block_node_ids: Optional[np.ndarray] = None
    pending_count: int = 0
    # ---- SWA (Sliding Window Attention) routing -------------------------------
    # When True, this op moves SWA KV (an independent GPU/CPU/SSD/REMOTE pool with
    # its own slot-id space), so the transfer engine routes it to the dedicated
    # SWA worker (_swa_worker_map) instead of the main-KV worker. The op reuses the
    # standard transfer_type (D2H/H2D/DISK2H/H2DISK/REMOTE2H/H2REMOTE); src/dst
    # block ids are SWA-pool slot ids, NOT full-KV block ids.
    is_swa: bool = False
    # Block content hashes for mooncake-store key-based addressing (main KV).
    mooncake_store_block_hashes: Optional[np.ndarray] = None
    # Tail-hash list for SWA mooncake REMOTE2H/H2REMOTE (one entry per SWA slot).
    mooncake_store_swa_block_hashes: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.transfer_type != TransferType.VIRTUAL and \
            self.src_block_ids.size != self.dst_block_ids.size:
            raise ValueError(f"src_block_ids and dst_block_ids must have the same number of physical blocks, but got "
                             f"src_block_ids.size={self.src_block_ids.size}, "
                             f"dst_block_ids.size={self.dst_block_ids.size}")
        with TransferOp._lock:
            self.op_id = TransferOp._next_op_id
            TransferOp._next_op_id += 1
        assert self.src_block_ids.dtype == np.int64
        assert self.dst_block_ids.dtype == np.int64
        self.valid_block_num = self.src_block_ids.size

@dataclass
class LayerwiseTransferOp(TransferOp):

    src_block_ids_h2d: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    dst_block_ids_h2d: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    src_block_ids_disk2h: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    dst_block_ids_disk2h: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    # SWA fields
    swa_src_block_ids_h2d: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    swa_dst_block_ids_h2d: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    swa_src_block_ids_disk2h: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    swa_dst_block_ids_disk2h: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    counter_id: int = 0  # Counter set index for triple buffering eventfd notification

    def __init__(self,
                graph_id: int,
                src_block_ids_h2d: np.ndarray,
                dst_block_ids_h2d: np.ndarray,
                src_block_ids_disk2h: np.ndarray,
                dst_block_ids_disk2h: np.ndarray,
                swa_src_block_ids_h2d: Optional[np.ndarray] = None,
                swa_dst_block_ids_h2d: Optional[np.ndarray] = None,
                swa_src_block_ids_disk2h: Optional[np.ndarray] = None,
                swa_dst_block_ids_disk2h: Optional[np.ndarray] = None,
                dp_client_id: int = 0,
                counter_id: int = 0) -> None:
        self.src_block_ids_h2d = src_block_ids_h2d
        self.dst_block_ids_h2d = dst_block_ids_h2d
        self.src_block_ids_disk2h = src_block_ids_disk2h
        self.dst_block_ids_disk2h = dst_block_ids_disk2h
        # SWA ids default to empty arrays so callers that only need the main-KV
        # path can omit them and __post_init__ assertions still hold (.size on None
        # would AttributeError). Empty SWA arrays drive the cpp fused layer loop
        # to skip the SWA branch entirely (has_swa=False at submit time).
        _empty = lambda: np.array([], dtype=np.int64)
        self.swa_src_block_ids_h2d = swa_src_block_ids_h2d if swa_src_block_ids_h2d is not None else _empty()
        self.swa_dst_block_ids_h2d = swa_dst_block_ids_h2d if swa_dst_block_ids_h2d is not None else _empty()
        self.swa_src_block_ids_disk2h = swa_src_block_ids_disk2h if swa_src_block_ids_disk2h is not None else _empty()
        self.swa_dst_block_ids_disk2h = swa_dst_block_ids_disk2h if swa_dst_block_ids_disk2h is not None else _empty()
        self.counter_id = counter_id

        super().__init__(
            graph_id=graph_id,
            transfer_type=TransferType.LAYERWISE,
            src_block_ids=np.array([], dtype=np.int64),
            dst_block_ids=np.array([], dtype=np.int64),
            dp_client_id=dp_client_id,
        )

    def __post_init__(self) -> None:
        super().__post_init__()

        assert self.src_block_ids_h2d.size == self.dst_block_ids_h2d.size
        assert self.src_block_ids_disk2h.size == self.dst_block_ids_disk2h.size
        assert self.swa_src_block_ids_h2d.size == self.swa_dst_block_ids_h2d.size
        assert self.swa_src_block_ids_disk2h.size == self.swa_dst_block_ids_disk2h.size

        assert self.src_block_ids_h2d.dtype == np.int64
        assert self.dst_block_ids_h2d.dtype == np.int64
        assert self.src_block_ids_disk2h.dtype == np.int64
        assert self.dst_block_ids_disk2h.dtype == np.int64
        assert self.swa_src_block_ids_h2d.dtype == np.int64
        assert self.swa_dst_block_ids_h2d.dtype == np.int64
        assert self.swa_src_block_ids_disk2h.dtype == np.int64
        assert self.swa_dst_block_ids_disk2h.dtype == np.int64


class TransferOpGraph:
    _next_graph_id = 0
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.graph_id = self._get_graph_id()
        self._op_map: Dict[int, TransferOp] = {}
        self._ready_ops: Set[int] = set()
        self._trigger_ops: Set[int] = set()
        self._gpu_transfer_op_id: List[int] = []
        # SWA GPU-transfer ops (is_swa=True, GPU on one side). Bound LATE via
        # set_swa_gpu_blocks() from the request's SWA-pool slot_mapping, exactly
        # as _gpu_transfer_op_id is bound from the full-KV slot_mapping. Kept in a
        # SEPARATE list because SWA lives in its own GPU pool / slot-id space and
        # must never be rebound with full-KV block ids (see add_transfer_op).
        self._swa_gpu_transfer_op_id: List[int] = []

    @classmethod
    def _get_graph_id(cls) -> int:
        with cls._lock:
            graph_id = cls._next_graph_id
            cls._next_graph_id += 1
            return graph_id

    def set_graph_id(self, graph_id: int) -> None:
        self.graph_id = graph_id

    @classmethod
    def create_empty_graph(cls) -> "TransferOpGraph":
        return cls()

    def add_virtual_op(self, op: TransferOp, need_trigger: bool = False) -> None:
        op.graph_id = self.graph_id
        op.transfer_type = TransferType.VIRTUAL
        self._op_map[op.op_id] = op
        if need_trigger:
            self._trigger_ops.add(op.op_id)
        else:
            self._ready_ops.add(op.op_id)

    def trigger_op(self, op_id: int) -> None:
        self._trigger_ops.remove(op_id)
        self._ready_ops.discard(op_id)
        self.mark_completed(op_id)

    def add_transfer_op(self, op: TransferOp) -> None:
        op.graph_id = self.graph_id
        self._op_map[op.op_id] = op
        # Unified GPU-transfer list (colleague's PR#191 model): every op that
        # touches a GPU pool — full-KV OR SWA — is tracked in _gpu_transfer_op_id,
        # and set_gpu_blocks(gpu_blocks, swa_gpu_blocks) sorts them by op.is_swa at
        # bind time (full-KV <- gpu_blocks; SWA <- swa_gpu_blocks, or preserved when
        # swa_gpu_blocks is None). This is what test_set_gpu_blocks_swa exercises.
        if op.transfer_type == TransferType.H2D or \
            op.transfer_type == TransferType.D2H or \
            op.transfer_type == TransferType.D2DISK or \
            op.transfer_type == TransferType.DISK2D:
            self._gpu_transfer_op_id.append(op.op_id)
        # SWA GPU-transfer ops are ALSO tracked separately so the node-mount
        # late-bind entry point set_swa_gpu_blocks() (used by kvtask + the
        # control-plane unit tests) keeps working. Only H2D (GPU dst) / D2H
        # (GPU src) touch the GPU SWA pool; CPU<->SSD/REMOTE staging has no GPU slot.
        # Membership in BOTH lists is safe: set_gpu_blocks() leaves is_swa ops
        # untouched when swa_gpu_blocks is None (the kvtask two-call path), so only
        # set_swa_gpu_blocks() binds them there; set_gpu_blocks(gpu, swa_gpu) binds
        # them directly when a caller supplies the second arg.
        if op.is_swa and (
            op.transfer_type == TransferType.H2D or
            op.transfer_type == TransferType.D2H):
            self._swa_gpu_transfer_op_id.append(op.op_id)
        self._ready_ops.add(op.op_id)

    def add_dependency(self, successor_op_id: int, predecessor_op_id: int) -> None:
        """successor_op_id depends on predecessor_op_id"""
        assert successor_op_id in self._op_map and predecessor_op_id in self._op_map
        self._op_map[successor_op_id].predecessors.add(predecessor_op_id)
        self._op_map[predecessor_op_id].successors.add(successor_op_id)
        self._ready_ops.discard(successor_op_id)

    def mark_completed(self, op_id: int) -> None:
        """mark an op as completed"""
        if op_id in self._op_map:
            assert self._op_map[op_id].status == TransferOpStatus.RUNNING
            self._op_map[op_id].status = TransferOpStatus.COMPLETED
            my_successors = self._op_map[op_id].successors
            for successor_id in my_successors:
                self._op_map[successor_id].predecessors.remove(op_id)

    def take_ready_ops(self) -> List[int]:
        """get a list of op ids that are ready to execute"""
        ready_ops = []
        to_remove = []
        to_add = []
        for op_id in self._ready_ops:
            op = self._op_map[op_id]
            if op.status == TransferOpStatus.COMPLETED:
                to_remove.append(op_id)
                for successor_id in op.successors:
                    if (self._op_map[successor_id].status == TransferOpStatus.PENDING and
                        len(self._op_map[successor_id].predecessors) == 0):
                        ready_ops.append(successor_id)
                        self._op_map[successor_id].status = TransferOpStatus.RUNNING
                        to_add.append(successor_id)
            elif op.status == TransferOpStatus.PENDING: # not supposed to happen now
                ready_ops.append(op_id)
                self._op_map[op_id].status = TransferOpStatus.RUNNING
                to_add.append(op_id)

        self._ready_ops.difference_update(to_remove)
        self._ready_ops.update(to_add)
        return ready_ops

    def all_transfer_ops_completed(self) -> bool:
        """check if all transfer ops are completed"""
        return all(op.status == TransferOpStatus.COMPLETED
                   for op in self._op_map.values())

    def set_gpu_blocks(self,
                       gpu_blocks: np.ndarray,
                       swa_gpu_blocks: Optional[np.ndarray] = None) -> None:
        swa_offset = 0
        for op_id in self._gpu_transfer_op_id:
            op = self._op_map[op_id]
            target_gpu_blocks = gpu_blocks
            # Optional SWA launch-time binding path: if caller passes
            # swa_gpu_blocks, fill SWA ops from that array; otherwise preserve
            # graph-built SWA ids.
            if getattr(op, "is_swa", False):
                if swa_gpu_blocks is None:
                    continue
            transfer_type = op.transfer_type
            if getattr(op, "is_swa", False):
                count = op.dst_block_ids.size if transfer_type.name.endswith("2D") \
                    else op.src_block_ids.size
                next_swa_offset = swa_offset + count
                if next_swa_offset > swa_gpu_blocks.size:
                    raise ValueError(
                        f"not enough SWA GPU blocks to bind op {op_id}: "
                        f"need {next_swa_offset}, got {swa_gpu_blocks.size}"
                    )
                target_gpu_blocks = swa_gpu_blocks[swa_offset:next_swa_offset]
                swa_offset = next_swa_offset
            if transfer_type.name.endswith("2D"):
                if transfer_type == TransferType.DISK2D:
                    op.dst_block_ids = target_gpu_blocks[-op.dst_block_ids.size:]
                else:
                    op.dst_block_ids = target_gpu_blocks[:op.dst_block_ids.size]
            else:
                if transfer_type == TransferType.D2DISK:
                    op.src_block_ids = target_gpu_blocks[-op.src_block_ids.size:]
                else:
                    op.src_block_ids = target_gpu_blocks[:op.src_block_ids.size]
            assert op.src_block_ids.size == op.dst_block_ids.size, \
                f"src_block_ids.size={op.src_block_ids.size}, dst_block_ids.size={op.dst_block_ids.size}"

    def clear_gpu_blocks(self) -> None:
        """Clear GPU block_ids from the graph.
        """
        for op_id in self._gpu_transfer_op_id:
            op = self._op_map[op_id]
            if getattr(op, "is_swa", False):
                continue
            # Replace with empty arrays; set_gpu_blocks() will fill them later
            if op.src_block_ids.size > 0:
                op.src_block_ids = np.array([], dtype=op.src_block_ids.dtype)
            if op.dst_block_ids.size > 0:
                op.dst_block_ids = np.array([], dtype=op.dst_block_ids.dtype)

    def set_swa_gpu_blocks(self, swa_gpu_blocks: np.ndarray) -> None:
        """Bind the GPU-side SWA-pool slot ids for every SWA GPU-transfer op.

        Mirror of :meth:`set_gpu_blocks` for the SWA channel: for an SWA ``H2D``
        the GPU pool is the destination, for an SWA ``D2H`` it is the source.
        ``swa_gpu_blocks`` are SWA-pool slot ids (already converted from the
        connector's swa_slot_mapping). The CPU/SSD/REMOTE side of each SWA op was
        set at build time (node-mounted radix slot) and is left untouched."""
        offset = 0
        for op_id in self._swa_gpu_transfer_op_id:
            op = self._op_map[op_id]
            count = op.dst_block_ids.size if op.transfer_type.name.endswith("2D") \
                else op.src_block_ids.size
            next_offset = offset + count
            if next_offset > swa_gpu_blocks.size:
                raise ValueError(
                    f"not enough SWA GPU blocks to bind op {op_id}: "
                    f"need {next_offset}, got {swa_gpu_blocks.size}"
                )
            gpu_slice = swa_gpu_blocks[offset:next_offset]
            if op.transfer_type.name.endswith("2D"):   # H2D: GPU is dst
                op.dst_block_ids = gpu_slice
            else:                                       # D2H: GPU is src
                op.src_block_ids = gpu_slice
            offset = next_offset
            assert op.src_block_ids.size == op.dst_block_ids.size, \
                f"swa src.size={op.src_block_ids.size}, dst.size={op.dst_block_ids.size}"

    def clear_swa_gpu_blocks(self) -> None:
        """Clear the GPU-side SWA slot ids (mirror of clear_gpu_blocks for SWA)."""
        for op_id in self._swa_gpu_transfer_op_id:
            op = self._op_map[op_id]
            if op.transfer_type.name.endswith("2D"):
                if op.dst_block_ids.size > 0:
                    op.dst_block_ids = np.array([], dtype=op.dst_block_ids.dtype)
            else:
                if op.src_block_ids.size > 0:
                    op.src_block_ids = np.array([], dtype=op.src_block_ids.dtype)

    @property
    def num_ops(self) -> int:
        return len(self._op_map)

    def visualize(self) -> str:
        """
        Visualize the transfer op graph in a readable format.
        Returns a string representation of the graph.
        """
        lines = []
        lines.append(f"╔{'═' * 70}╗")
        lines.append(f"║ TransferOpGraph (graph_id={self.graph_id}, num_ops={self.num_ops})".ljust(71) + "║")
        lines.append(f"╠{'═' * 70}╣")

        if not self._op_map:
            lines.append("║ (empty graph)".ljust(71) + "║")
            lines.append(f"╚{'═' * 70}╝")
            return "\n".join(lines)

        # Sort ops by op_id for consistent display
        sorted_ops = sorted(self._op_map.values(), key=lambda op: op.op_id)

        for i, op in enumerate(sorted_ops):
            # Op header
            status_symbol = {"PENDING": "○", "RUNNING": "◐", "COMPLETED": "●"}.get(op.status.name, "?")
            lines.append(f"║ [{status_symbol}] Op {op.op_id}: {op.transfer_type.value}".ljust(71) + "║")

            # Dependencies
            if op.predecessors:
                pred_str = ", ".join(str(p) for p in sorted(op.predecessors))
                lines.append(f"║     ├─ predecessors: [{pred_str}]".ljust(71) + "║")
            else:
                lines.append("║     ├─ predecessors: (none - ready)".ljust(71) + "║")

            if op.successors:
                succ_str = ", ".join(str(s) for s in sorted(op.successors))
                lines.append(f"║     ├─ successors:   [{succ_str}]".ljust(71) + "║")

            # Block info (truncate if too long)
            if op.transfer_type != TransferType.VIRTUAL:
                src_size = op.src_block_ids.size
                dst_size = op.dst_block_ids.size

                # Show first few and last few block ids
                def format_blocks(block_ids, max_show=4):
                    if block_ids.size == 0:
                        return "[]"
                    elif block_ids.size <= max_show * 2:
                        return str(block_ids.tolist())
                    else:
                        first = block_ids[:max_show].tolist()
                        last = block_ids[-max_show:].tolist()
                        return f"{first[:-1]}...{last[-1]}] (n={block_ids.size})"

                src_str = format_blocks(op.src_block_ids)
                dst_str = format_blocks(op.dst_block_ids)
                lines.append(f"║     ├─ src_blocks:   {src_str}".ljust(71) + "║")
                lines.append(f"║     ├─ dst_blocks:   {dst_str}".ljust(71) + "║")
                lines.append(f"║     └─ dp_client_id={op.dp_client_id}".ljust(71) + "║")
            else:
                lines.append("║     └─ (VIRTUAL - no blocks)".ljust(71) + "║")

            # Separator between ops
            if i < len(sorted_ops) - 1:
                lines.append(f"║{'-' * 70}║")

        # Show ready ops
        lines.append(f"╠{'═' * 70}╣")
        ready_str = ", ".join(str(op_id) for op_id in sorted(self._ready_ops)) if self._ready_ops else "(none)"
        lines.append(f"║ Ready ops: [{ready_str}]".ljust(71) + "║")

        if self._trigger_ops:
            trigger_str = ", ".join(str(op_id) for op_id in sorted(self._trigger_ops))
            lines.append(f"║ Trigger ops: [{trigger_str}]".ljust(71) + "║")

        lines.append(f"╚{'═' * 70}╝")

        result = "\n".join(lines)
        print(result)
        return result

def _make_combined_callback(callbacks: List[Callable]) -> Callable:
    def combined_callback(*args, **kwargs):
        for cb in callbacks:
            cb(*args, **kwargs)
    return combined_callback


def _attach_combined_callback(op: TransferOp,
                              callbacks: List[Callable],
                              op_callback_dict: Dict[int, Callable]) -> None:
    if not callbacks:
        return
    if len(callbacks) == 1:
        op_callback_dict[op.op_id] = callbacks[0]
    else:
        op_callback_dict[op.op_id] = _make_combined_callback(callbacks)


def add_virtual_op_for_multiple_finished_ops(
    graph: TransferOpGraph,
    finished_ops_ids: List[int],
    dp_client_id: int,
) -> Tuple[TransferOpGraph, int]:
    """Return one task-end op for zero, one, or multiple terminal ops."""
    if len(finished_ops_ids) == 0:
        return graph, -1
    if len(finished_ops_ids) == 1:
        return graph, finished_ops_ids[0]

    op = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.VIRTUAL,
        src_block_ids=np.array([], dtype=np.int64),
        dst_block_ids=np.array([], dtype=np.int64),
        dp_client_id=dp_client_id,
    )
    graph.add_transfer_op(op)
    for op_id in finished_ops_ids:
        graph.add_dependency(op.op_id, op_id)
    return graph, op.op_id


def _merge_ops(ops: List[TransferOp], transfer_type: TransferType,
               graph: TransferOpGraph, callbacks: List[Callable],
               op_callback_dict: Dict[int, Callable]) -> Optional[TransferOp]:
    """Merge main-KV ops. Concatenates block ids and ``mooncake_store_block_hashes``
    (H2REMOTE / REMOTE2H) in the same order. Rejects SWA ops (use ``_merge_swa_ops``)
    and refuses to mix mooncake and non-mooncake ops in the same batch.
    """
    if not ops:
        return None
    if any(getattr(op, "is_swa", False) for op in ops):
        raise ValueError(
            f"_merge_ops[{transfer_type.name}]: SWA ops must go through _merge_swa_ops"
        )
    if any(op.mooncake_store_swa_block_hashes is not None for op in ops):
        raise ValueError(
            f"_merge_ops[{transfer_type.name}]: unexpected mooncake_store_swa_block_hashes "
            f"on main-KV op; use _merge_swa_ops"
        )
    src_blocks = np.concatenate([op.src_block_ids for op in ops])
    dst_blocks = np.concatenate([op.dst_block_ids for op in ops])

    merged_kv_hashes: Optional[np.ndarray] = None
    has_hashes = [op.mooncake_store_block_hashes is not None for op in ops]
    if any(has_hashes):
        if not all(has_hashes):
            raise ValueError(
                f"_merge_ops[{transfer_type.name}]: cannot merge mooncake and "
                f"non-mooncake ops in the same batch (has_hashes={has_hashes})"
            )
        merged_kv_hashes = np.concatenate(
            [np.asarray(op.mooncake_store_block_hashes) for op in ops]
        )

    merged_op = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=transfer_type,
        src_block_ids=src_blocks,
        dst_block_ids=dst_blocks,
        dp_client_id=ops[0].dp_client_id,
        mooncake_store_block_hashes=merged_kv_hashes,
    )
    if callbacks:
        if len(callbacks) == 1:
            op_callback_dict[merged_op.op_id] = callbacks[0]
        else:
            op_callback_dict[merged_op.op_id] = _make_combined_callback(callbacks)
    return merged_op


def _merge_swa_ops(ops: List[TransferOp], transfer_type: TransferType,
                   graph: TransferOpGraph, callbacks: List[Callable],
                   op_callback_dict: Dict[int, Callable]) -> Optional[TransferOp]:
    """Merge SWA-lane ops. Local lanes carry no mooncake hash. Remote lanes
    (H2REMOTE / REMOTE2H) concatenate ``mooncake_store_swa_block_hashes``.
    """
    if not ops:
        return None
    if any(not getattr(op, "is_swa", False) for op in ops):
        raise ValueError(
            f"_merge_swa_ops[{transfer_type.name}]: all ops must have is_swa=True"
        )
    if any(op.mooncake_store_block_hashes is not None for op in ops):
        raise ValueError(
            f"_merge_swa_ops[{transfer_type.name}]: unexpected mooncake_store_block_hashes "
            f"on SWA op; use main-KV _merge_ops"
        )
    src_blocks = np.concatenate([op.src_block_ids for op in ops])
    dst_blocks = np.concatenate([op.dst_block_ids for op in ops])

    merged_swa_hashes: Optional[List[str]] = None
    if transfer_type in (TransferType.H2REMOTE, TransferType.REMOTE2H):
        merged_swa_hashes = []
        for op in ops:
            tails = op.mooncake_store_swa_block_hashes
            if tails is None:
                raise ValueError(
                    f"_merge_swa_ops[{transfer_type.name}]: SWA mooncake op missing "
                    f"mooncake_store_swa_block_hashes (op_id={op.op_id})"
                )
            merged_swa_hashes.extend(str(h) for h in tails)
    elif any(op.mooncake_store_swa_block_hashes is not None for op in ops):
        raise ValueError(
            f"_merge_swa_ops[{transfer_type.name}]: mooncake_store_swa_block_hashes "
            f"only allowed on H2REMOTE / REMOTE2H"
        )

    merged_op = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=transfer_type,
        src_block_ids=src_blocks,
        dst_block_ids=dst_blocks,
        dp_client_id=ops[0].dp_client_id,
        is_swa=True,
        mooncake_store_swa_block_hashes=merged_swa_hashes,
    )
    if callbacks:
        if len(callbacks) == 1:
            op_callback_dict[merged_op.op_id] = callbacks[0]
        else:
            op_callback_dict[merged_op.op_id] = _make_combined_callback(callbacks)
    return merged_op


def _bucket_has(*types: TransferType, ops_by_type: Dict[TransferType, List[TransferOp]],
                     swa_ops_by_type: Dict[TransferType, List[TransferOp]]) -> bool:
    return any(ops_by_type[t] or swa_ops_by_type[t] for t in types)


def _pick_dp_client_id(ops_by_type: Dict[TransferType, List[TransferOp]],
                       swa_ops_by_type: Dict[TransferType, List[TransferOp]]) -> int:
    for tt in (TransferType.H2D, TransferType.D2H):
        if ops_by_type[tt]:
            return ops_by_type[tt][0].dp_client_id
        if swa_ops_by_type[tt]:
            return swa_ops_by_type[tt][0].dp_client_id
    return 0


def _add_batch_sink(graph: TransferOpGraph, terminals: List[int],
                    dp_client_id: int) -> int:
    """Materialize the batch sink.

    * 0 terminals    -> -1 (nothing scheduled).
    * 1 terminal     -> return it directly (no VIRTUAL needed).
    * >= 2 terminals -> add a VIRTUAL op depending on all of them.
    """
    terminals = [t for t in terminals if t is not None and t >= 0]
    if not terminals:
        return -1
    if len(terminals) == 1:
        return terminals[0]
    sink = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.VIRTUAL,
        src_block_ids=np.array([], dtype=np.int64),
        dst_block_ids=np.array([], dtype=np.int64),
        dp_client_id=dp_client_id,
    )
    graph.add_transfer_op(sink)
    for term in terminals:
        graph.add_dependency(sink.op_id, term)
    return sink.op_id


def merge_to_batch_graph(batch_id: int,
                         transfer_graphs: List[TransferOpGraph],
                         task_end_op_ids: List[int],
                         op_callback_dict: Dict[int, Callable],
                         layerwise_transfer: bool = False,
                         counter_id: int = 0,
                         ) -> Tuple[TransferOpGraph, int, Dict[int, Callable]]:
    """
    Merge multiple TransferOpGraphs into a single batch graph.

    Supported patterns:
      GET: DISK2H / REMOTE2H (optional) -> H2D
      PUT: D2H -> H2DISK / H2REMOTE (optional)
      layerwise GET: fused LAYERWISE (+ REMOTE2H predecessors when present);
          SWA local lanes always fold into LAYERWISE when layerwise is on.

    Args:
        batch_id: ID for the new batch graph
        transfer_graphs: List of graphs to merge
        task_end_op_ids: List of end op IDs for each task (one per graph)
        op_callback_dict: Dict mapping old op_id -> callback
        layerwise_transfer: Whether to merge the graphs into a layerwise transfer op
        fuse_swa_into_layerwise: When True (default / production), SWA/state
            H2D block ids are carried on the fused LAYERWISE op (uniform SWA
            via launch_swa_h2d_layer_, multi-group via launch_swa_mg_h2d_layer_).
            When False, SWA/state ops stay as standalone predecessors
            (legacy heterogeneous fallback for non-fused debugging).

    Returns:
        (merged_graph, batch_end_op_id, new_op_callback_dict)

    The input task-end ids belong to graphs that are discarded during fusion;
    they are retained in the API for caller compatibility.  The merged graph
    rebuilds its completion contract from its supported shape:
      GET: full H2D + SWA H2D
      PUT: full D2H + SWA D2H
      layerwise GET: the fused LAYERWISE op
    """
    if not transfer_graphs:
        empty_graph = TransferOpGraph()
        empty_graph.set_graph_id(batch_id)
        return empty_graph, -1, {}
    if len(transfer_graphs) != len(task_end_op_ids):
        raise ValueError(
            "transfer_graphs and task_end_op_ids must have the same length")

    merged_graph = TransferOpGraph()
    merged_graph.set_graph_id(batch_id)

    ops_by_type: Dict[TransferType, List[TransferOp]] = {}
    callbacks_by_type: Dict[TransferType, List[Callable]] = {}
    swa_ops_by_type: Dict[TransferType, List[TransferOp]] = {}
    swa_callbacks_by_type: Dict[TransferType, List[Callable]] = {}
    supported_types = {TransferType.DISK2H, TransferType.H2D,
                       TransferType.D2H, TransferType.H2DISK,
                       TransferType.H2REMOTE, TransferType.REMOTE2H}

    for tt in supported_types:
        ops_by_type[tt] = []
        callbacks_by_type[tt] = []
        swa_ops_by_type[tt] = []
        swa_callbacks_by_type[tt] = []

    for graph in transfer_graphs:
        for op_id, op in graph._op_map.items():
            if op.transfer_type == TransferType.VIRTUAL:
                continue
            if op.transfer_type not in supported_types:
                raise NotImplementedError(
                    f"Batch merge does not support transfer type: {op.transfer_type}. "
                    f"Only DISK2H, H2D, D2H, H2DISK, REMOTE2H, and H2REMOTE are supported."
                )
            if getattr(op, "is_swa", False):
                swa_ops_by_type[op.transfer_type].append(op)
                if op.op_id in op_callback_dict:
                    swa_callbacks_by_type[op.transfer_type].append(
                        op_callback_dict[op.op_id])
            else:
                ops_by_type[op.transfer_type].append(op)
                if op.op_id in op_callback_dict:
                    callbacks_by_type[op.transfer_type].append(
                        op_callback_dict[op.op_id])

    new_op_callback_dict: Dict[int, Callable] = {}
    # Layerwise folds local DISK2H/H2D into LAYERWISE: merge those ops with a
    # throwaway callback dict (block ids only), then reattach callbacks onto
    # LAYERWISE. REMOTE2H stays on new_op_callback_dict as a predecessor.
    layerwise_local_cb_dict: Dict[int, Callable] = {}
    local_cb_dict = (layerwise_local_cb_dict if layerwise_transfer
                     else new_op_callback_dict)

    has_get = _bucket_has(TransferType.H2D, TransferType.DISK2H, TransferType.REMOTE2H,
                          ops_by_type=ops_by_type, swa_ops_by_type=swa_ops_by_type)
    has_put = _bucket_has(TransferType.D2H, TransferType.H2DISK, TransferType.H2REMOTE,
                          ops_by_type=ops_by_type, swa_ops_by_type=swa_ops_by_type)

    if layerwise_transfer:
        assert has_get, "layerwise batch must contain GET ops"
        assert not has_put, "layerwise batch must not mix PUT ops"
    else:
        assert not (has_get and has_put), \
            "batch merge does not support mixed GET/PUT ops in the same batch"

    dp_client_id = _pick_dp_client_id(
        ops_by_type=ops_by_type, swa_ops_by_type=swa_ops_by_type)

    if has_get:
        merged_disk2h_op = _merge_ops(
            ops_by_type[TransferType.DISK2H], TransferType.DISK2H,
            merged_graph, callbacks_by_type[TransferType.DISK2H],
            local_cb_dict)
        merged_h2d_op = _merge_ops(
            ops_by_type[TransferType.H2D], TransferType.H2D,
            merged_graph, callbacks_by_type[TransferType.H2D],
            local_cb_dict)
        merged_remote2h_op = _merge_ops(
            ops_by_type[TransferType.REMOTE2H], TransferType.REMOTE2H,
            merged_graph, callbacks_by_type[TransferType.REMOTE2H],
            new_op_callback_dict)
        merged_swa_disk2h_op = _merge_swa_ops(
            swa_ops_by_type[TransferType.DISK2H], TransferType.DISK2H,
            merged_graph, swa_callbacks_by_type[TransferType.DISK2H],
            local_cb_dict)
        merged_swa_h2d_op = _merge_swa_ops(
            swa_ops_by_type[TransferType.H2D], TransferType.H2D,
            merged_graph, swa_callbacks_by_type[TransferType.H2D],
            local_cb_dict)
        merged_swa_remote2h_op = _merge_swa_ops(
            swa_ops_by_type[TransferType.REMOTE2H], TransferType.REMOTE2H,
            merged_graph, swa_callbacks_by_type[TransferType.REMOTE2H],
            new_op_callback_dict)
        if layerwise_transfer:
            for op in (merged_remote2h_op, merged_swa_remote2h_op):
                if op is not None:
                    merged_graph.add_transfer_op(op)

            assert merged_h2d_op is not None or merged_swa_h2d_op is not None, \
                "layerwise GET requires an H2D (main or SWA)"

            layerwise_transfer_op = LayerwiseTransferOp(
                graph_id=merged_graph.graph_id,
                src_block_ids_h2d=merged_h2d_op.src_block_ids if merged_h2d_op is not None
                    else np.array([], dtype=np.int64),
                dst_block_ids_h2d=merged_h2d_op.dst_block_ids if merged_h2d_op is not None
                    else np.array([], dtype=np.int64),
                src_block_ids_disk2h=merged_disk2h_op.src_block_ids
                    if merged_disk2h_op is not None
                    else np.array([], dtype=np.int64),
                dst_block_ids_disk2h=merged_disk2h_op.dst_block_ids
                    if merged_disk2h_op is not None
                    else np.array([], dtype=np.int64),
                swa_src_block_ids_h2d=merged_swa_h2d_op.src_block_ids
                    if merged_swa_h2d_op is not None
                    else np.array([], dtype=np.int64),
                swa_dst_block_ids_h2d=merged_swa_h2d_op.dst_block_ids
                    if merged_swa_h2d_op is not None
                    else np.array([], dtype=np.int64),
                swa_src_block_ids_disk2h=merged_swa_disk2h_op.src_block_ids
                    if merged_swa_disk2h_op is not None
                    else np.array([], dtype=np.int64),
                swa_dst_block_ids_disk2h=merged_swa_disk2h_op.dst_block_ids
                    if merged_swa_disk2h_op is not None
                    else np.array([], dtype=np.int64),
                dp_client_id=dp_client_id,
                counter_id=counter_id,
            )
            merged_graph.add_transfer_op(layerwise_transfer_op)

            if merged_remote2h_op is not None:
                merged_graph.add_dependency(
                    layerwise_transfer_op.op_id, merged_remote2h_op.op_id)
            if merged_swa_remote2h_op is not None:
                merged_graph.add_dependency(
                    layerwise_transfer_op.op_id, merged_swa_remote2h_op.op_id)

            layerwise_callbacks: List[Callable] = []
            layerwise_callbacks.extend(callbacks_by_type[TransferType.DISK2H])
            layerwise_callbacks.extend(callbacks_by_type[TransferType.H2D])
            layerwise_callbacks.extend(swa_callbacks_by_type[TransferType.DISK2H])
            layerwise_callbacks.extend(swa_callbacks_by_type[TransferType.H2D])
            _attach_combined_callback(
                layerwise_transfer_op, layerwise_callbacks, new_op_callback_dict)
            batch_end_op_id = layerwise_transfer_op.op_id
        else:
            for op in (merged_disk2h_op, merged_h2d_op, merged_remote2h_op,
                       merged_swa_disk2h_op, merged_swa_h2d_op,
                       merged_swa_remote2h_op):
                if op is not None:
                    merged_graph.add_transfer_op(op)

            if merged_h2d_op is not None:
                if merged_disk2h_op is not None:
                    merged_graph.add_dependency(
                        merged_h2d_op.op_id, merged_disk2h_op.op_id)
                if merged_remote2h_op is not None:
                    merged_graph.add_dependency(
                        merged_h2d_op.op_id, merged_remote2h_op.op_id)
            if merged_swa_h2d_op is not None:
                if merged_swa_disk2h_op is not None:
                    merged_graph.add_dependency(
                        merged_swa_h2d_op.op_id, merged_swa_disk2h_op.op_id)
                if merged_swa_remote2h_op is not None:
                    merged_graph.add_dependency(
                        merged_swa_h2d_op.op_id, merged_swa_remote2h_op.op_id)

            get_sinks: List[int] = []
            if merged_h2d_op is not None:
                get_sinks.append(merged_h2d_op.op_id)
            if merged_swa_h2d_op is not None:
                get_sinks.append(merged_swa_h2d_op.op_id)
            if not get_sinks:
                # No GPU sink (e.g. prefetch / CPU-only): every independent
                # full-KV and SWA leaf must be a terminal. Taking only the first
                # would mark the batch done while another REMOTE2H/DISK2H lane
                # is still in flight.
                for op in (merged_remote2h_op, merged_swa_remote2h_op,
                           merged_disk2h_op, merged_swa_disk2h_op):
                    if op is not None:
                        get_sinks.append(op.op_id)
            batch_end_op_id = _add_batch_sink(
                merged_graph, get_sinks, dp_client_id)

    elif has_put:
        merged_d2h_op = _merge_ops(
            ops_by_type[TransferType.D2H], TransferType.D2H,
            merged_graph, callbacks_by_type[TransferType.D2H],
            new_op_callback_dict)
        merged_h2disk_op = _merge_ops(
            ops_by_type[TransferType.H2DISK], TransferType.H2DISK,
            merged_graph, callbacks_by_type[TransferType.H2DISK],
            new_op_callback_dict)
        merged_h2remote_op = _merge_ops(
            ops_by_type[TransferType.H2REMOTE], TransferType.H2REMOTE,
            merged_graph, callbacks_by_type[TransferType.H2REMOTE],
            new_op_callback_dict)
        merged_swa_d2h_op = _merge_swa_ops(
            swa_ops_by_type[TransferType.D2H], TransferType.D2H,
            merged_graph, swa_callbacks_by_type[TransferType.D2H],
            new_op_callback_dict)
        merged_swa_h2disk_op = _merge_swa_ops(
            swa_ops_by_type[TransferType.H2DISK], TransferType.H2DISK,
            merged_graph, swa_callbacks_by_type[TransferType.H2DISK],
            new_op_callback_dict)
        merged_swa_h2remote_op = _merge_swa_ops(
            swa_ops_by_type[TransferType.H2REMOTE], TransferType.H2REMOTE,
            merged_graph, swa_callbacks_by_type[TransferType.H2REMOTE],
            new_op_callback_dict)

        for op in (merged_d2h_op, merged_swa_d2h_op, merged_h2disk_op,
                   merged_swa_h2disk_op, merged_h2remote_op, merged_swa_h2remote_op):
            if op is not None:
                merged_graph.add_transfer_op(op)

        if merged_d2h_op is not None:
            if merged_h2disk_op is not None:
                merged_graph.add_dependency(merged_h2disk_op.op_id, merged_d2h_op.op_id)
            if merged_h2remote_op is not None:
                merged_graph.add_dependency(merged_h2remote_op.op_id, merged_d2h_op.op_id)
        if merged_swa_d2h_op is not None:
            if merged_swa_h2disk_op is not None:
                merged_graph.add_dependency(
                    merged_swa_h2disk_op.op_id, merged_swa_d2h_op.op_id)
            if merged_swa_h2remote_op is not None:
                merged_graph.add_dependency(
                    merged_swa_h2remote_op.op_id, merged_swa_d2h_op.op_id)

        put_sinks: List[int] = []
        if merged_d2h_op is not None:
            put_sinks.append(merged_d2h_op.op_id)
        if merged_swa_d2h_op is not None:
            put_sinks.append(merged_swa_d2h_op.op_id)
        if not put_sinks:
            # No D2H sink: wait for every independent full-KV / SWA leaf
            # (H2DISK and/or H2REMOTE). 
            for op in (merged_h2disk_op, merged_swa_h2disk_op,
                       merged_h2remote_op, merged_swa_h2remote_op):
                if op is not None:
                    put_sinks.append(op.op_id)
        batch_end_op_id = _add_batch_sink(merged_graph, put_sinks, dp_client_id)

    else:
        batch_end_op_id = -1

    return merged_graph, batch_end_op_id, new_op_callback_dict



def get_nvtx_default_color() -> int:
    return 0xD3D3D3

def get_nvtx_range_color(number: int) -> int:
    color = (number * 0x9e3779b1) % 0xffffff
    return color
