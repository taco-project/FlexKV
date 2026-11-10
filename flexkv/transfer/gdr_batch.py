"""Bounded coalescing of already-arrived transfer requests; no scheduler dependency."""
import copy
import time
import numpy as np

_IDS = ("src_block_ids_h2d", "dst_block_ids_h2d",
        "swa_src_block_ids_h2d", "swa_dst_block_ids_h2d")


def prepare_batch(worker, ops):
    from flexkv.common.transfer import TransferType
    from flexkv.transfer.worker_op import WorkerTransferOp
    config = worker._gdr_config
    direction = config["d2h"]
    if (not config["enabled"] or direction["path"] not in ("gdr", "auto")
            or direction["batch_requests"] <= 1 or worker.region_batch is None
            or getattr(getattr(worker, "_compressor", None), "needs_gpu_cpu_thread_group", False)):
        return
    def snapshot(op):
        src, dst = worker.get_transfer_block_ids(op, pinned=False)
        return src.numpy().copy(), dst.numpy().copy()
    groups = coalesce_d2h(ops, WorkerTransferOp, TransferType.D2H,
                          direction["batch_requests"], direction["max_batch_blocks"], snapshot)
    if config["diagnostics"]["summary"]:
        from flexkv.common.debug import flexkv_logger
        for count, ids in groups:
            flexkv_logger.info("[GDR-BATCH] direction=d2h requests=%d op_ids=%s", count, ids)


class Batch:
    """Execute once, report once per original op using the existing run loop."""
    def __init__(self, children, ids=_IDS):
        self.op = copy.copy(children[0])
        # copy.copy inherits child 0's arrival time, but the D2H producer fence
        # keys on the latest one: a child that arrived after the last fence must
        # not ride it just because an earlier sibling was covered.
        self.op._received_ns = max((getattr(c, "_received_ns", 0) for c in children),
                                   default=0)
        for name in ids:
            setattr(self.op, name, np.concatenate([getattr(c, name) for c in children]))
        if ids == _IDS:
            self.op._gdr_signal_count = len(children)
        else:
            self.op.valid_block_num = sum(c.valid_block_num for c in children)
        self.done = False
        self.result = None
        self.error = None

    def run(self, launch):
        if not self.done:
            self.started_ns = time.perf_counter_ns()
            try:
                self.result = launch(self.op)
            except Exception as exc:
                self.error = exc
            finally:
                self.finished_ns = time.perf_counter_ns()
                self.done = True
        if self.error is not None:
            raise self.error
        return self.result


def coalesce(ops, op_type, limit, max_blocks):
    """Attach a shared execution to original ops. Keep IDs/order/control barriers.

    Originals are never replaced, so success/failure fanout and graph resource
    ownership stay in the existing worker loop. Prepare all groups before
    attaching anything, so allocation errors leave the original list usable.
    """
    groups, group = [], []
    def flush():
        if len(group) > 1:
            groups.append((list(group), Batch(group)))
        group.clear()
    for op in ops:
        if not isinstance(op, op_type):
            flush()
            continue
        if group:
            incompatible = (op.counter_id != group[0].counter_id or len(group) >= limit or
                            sum(len(g.src_block_ids_h2d) for g in group) +
                            len(op.src_block_ids_h2d) > max_blocks)
            if not incompatible:
                incompatible = any(np.intersect1d(getattr(op, name), getattr(g, name)).size
                                   for g in group for name in _IDS[1::2])
            if incompatible:
                flush()
        group.append(op)
    flush()
    for children, batch in groups:
        for child in children:
            child._gdr_batch = batch
    return [(len(children), [c.transfer_op_id for c in children]) for children, _ in groups]


def coalesce_d2h(ops, op_type, d2h_type, limit, max_blocks, resolve_ids=None):
    """Snapshot slot descriptors, merge disjoint destinations, retain original leases.

    Source GPU IDs may overlap (reads); destination CPU IDs must not overlap.
    The existing synchronous launch and Batch fanout retain all source leases
    until the combined transfer finishes, including on failure.
    """
    groups, group = [], []
    def flush():
        if len(group) > 1:
            originals, descriptors = zip(*group)
            batch = Batch(descriptors, ("src_block_ids", "dst_block_ids"))
            batch.op.src_slot_id = batch.op.dst_slot_id = -1
            groups.append((originals, batch))
        group.clear()
    for op in ops:
        eligible = (isinstance(op, op_type) and op.transfer_type == d2h_type and
                    (resolve_ids is not None or (op.src_slot_id == -1 and op.dst_slot_id == -1)) and
                    op.src_block_node_ids is None and
                    getattr(op, "mooncake_store_block_hashes", None) is None and
                    getattr(op, "mooncake_store_swa_block_hashes", None) is None and op.valid_block_num > 0)
        if not eligible:
            flush()
            continue
        descriptor = copy.copy(op)
        if resolve_ids is not None:
            descriptor.src_block_ids, descriptor.dst_block_ids = resolve_ids(op)
        if not (descriptor.valid_block_num == len(descriptor.src_block_ids) == len(descriptor.dst_block_ids)):
            flush()
            continue
        if group and (getattr(op, "pool_id", None) != getattr(group[0][0], "pool_id", None) or len(group) >= limit or
                      sum(g.valid_block_num for _, g in group)+op.valid_block_num > max_blocks or
                      any(np.intersect1d(descriptor.dst_block_ids,g.dst_block_ids).size for _, g in group)):
            flush()
        group.append((op, descriptor))
    flush()
    for children, batch in groups:
        for child in children:
            child._gdr_batch = batch
    return [(len(children), [c.transfer_op_id for c in children]) for children, _ in groups]
