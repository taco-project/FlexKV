"""Worker loss containment. All methods run under the engine scheduler lock.

A process sentinel proves death, not CUDA quiescence. Only a worker's explicit
post-drain acknowledgement permits rollback of a transfer whose send was
attempted. Otherwise its graph and buffers remain quarantined.
"""
import threading
import time
from typing import Any, Dict, List, Set

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import TransferOp, TransferType
from flexkv.transfer import trace
from flexkv.transfer.workers.handle import WorkerHandle


class WorkerResilience:
    def _init_resilience(self) -> None:
        self._resilience_lock = threading.RLock()
        self._op_workers = {}
        self._dead_workers = set()
        self._disabled_routes = set()
        self._route_generation = {}

    def _route_workers(self, dp_client_id: int) -> List[WorkerHandle]:
        result = {}
        for worker_map in self._workers.values():
            for entry in worker_map.values():
                if isinstance(entry, dict):
                    for key, handle in entry.items():
                        if key.dp_client_id == dp_client_id:
                            result[id(handle)] = handle
        return list(result.values())

    def _worker_routes(self, handle: WorkerHandle) -> Set[int]:
        return {key.dp_client_id
                for worker_map in self._workers.values()
                for entry in worker_map.values() if isinstance(entry, dict)
                for key, value in entry.items() if value is handle}

    def _retire_worker(self, handle: WorkerHandle) -> bool:
        """Resolve only transfers with a positive drain acknowledgement."""
        self._dead_workers.add(id(handle))
        routes = self._worker_routes(handle)
        self._disabled_routes.update(routes)
        quiesced = getattr(handle, "quiesced_event", None)
        safe = quiesced is not None and quiesced.is_set()
        owned = [op_id for op_id, owner in self._op_workers.items()
                 if owner is handle]
        if owned and not safe:
            flexkv_logger.error(
                "Worker %s lost without DMA drain acknowledgement; "
                "quarantining ops=%s routes=%s", handle.worker_id, owned, routes)
            return False
        for op_id in owned:
            self._op_workers.pop(op_id, None)
            trace.dec_inflight()
            trace.consume_submit_ns(op_id)
            self._handle_failed_op(op_id)
        return True

    def _check_exited_workers(self) -> None:
        for handle in self._collect_worker_handles():
            if (hasattr(handle, "process") and id(handle) not in self._dead_workers
                    and not handle.process.is_alive()):
                flexkv_logger.error("Transfer worker %s exited (exitcode=%s)",
                                     handle.worker_id, handle.process.exitcode)
                self._retire_worker(handle)

    def _submit_tracked(self, handle: WorkerHandle, op: TransferOp) -> None:
        if not hasattr(self, "_op_workers"):
            self._op_workers = {}
        if (id(handle) in getattr(self, "_dead_workers", ())
                or (hasattr(handle, "process") and not handle.process.is_alive())):
            # This op was never sent. Defer its failure until PP fan-out has
            # finished establishing parent pending counts.
            self.finished_ops_queue.put((op.op_id, False, None))
            return
        self._op_workers[op.op_id] = handle
        try:
            handle.submit_transfer(op)
        except Exception:
            # A failed send is ambiguous: the peer might have accepted bytes.
            # Stop and drain it before synthesizing a failure completion.
            flexkv_logger.exception("Dispatch to worker %s failed", handle.worker_id)
            try:
                handle.abort()
            finally:
                self._disabled_routes.update(self._worker_routes(handle))
            # The next scheduler tick retires it after the entire PP fan-out
            # has established parent pending counts.

    def _fail_dispatch(self, op: TransferOp) -> None:
        self._failed_graph_ids.add(op.graph_id)
        self.scheduler.fail_graph(op.graph_id)
        self._failed_parent_op_ids.add(op.op_id)
        if op.pending_count == 0:
            self._discard_failed_op(op)

    def fence_gpu_route(self, dp_client_id: int) -> Dict[str, Any]:
        """Stop both GPU directions for ONE DP client; never infer safety from timeout.

        A false safe result requires the inference instance to restart. Its
        outstanding task's memory must remain allocated until then.
        """
        with self._resilience_lock:
            workers = self._route_workers(dp_client_id)
            if not workers:
                raise ValueError(f"Unknown DP client {dp_client_id}")
            self._disabled_routes.add(dp_client_id)
            for worker in workers:
                worker.abort()
                self._retire_worker(worker)
            self._emit_drained_graph_failures()
            safe = not any(owner in workers for owner in self._op_workers.values())
            return {"safe": safe, "generation": self._route_generation.get(dp_client_id, 0)}

    def recover_gpu_route(self, dp_client_id: int, generation: int) -> Dict[str, Any]:
        """Re-import FullKV/SWA into fresh workers for the SAME live exporter.

        This API does not accept new inference-process GPU pointers. A new
        inference instance still requires a registration/session protocol.
        """
        with self._resilience_lock:
            current = self._route_generation.get(dp_client_id, 0)
            if (int(generation) == current - 1 and current > 0
                    and dp_client_id not in self._disabled_routes):
                # Retry after a lost control reply: rebuilding the same route
                # again would destroy the workers just acknowledged by peers.
                return {"safe": True, "generation": current}
            if int(generation) != current:
                raise ValueError("Stale GPU route generation")
            if dp_client_id not in self._disabled_routes:
                raise RuntimeError("Fence GPU route before recovery")
            if self.fence_gpu_route(dp_client_id)["safe"] is not True:
                raise RuntimeError("GPU route has quarantined DMA; restart inference instance")
            if any(op.dp_client_id == dp_client_id for op in self.op_id_to_op.values()):
                raise RuntimeError("GPU route still has transfers draining")
            if TransferType.LAYERWISE in self._worker_map:
                raise NotImplementedError("Live layerwise recovery requires new eventfd consumers")
            # Only ordinary H2D/D2H can currently be rebuilt. Reject other GPU
            # backends rather than resuming a partially repaired route.
            if any(key.dp_client_id == dp_client_id
                   for wm in self._workers.values() for kind, entry in wm.items()
                   if kind not in (TransferType.H2D, TransferType.D2H)
                   and isinstance(entry, dict) for key in entry):
                raise NotImplementedError("Recovery supports CPU/GPU workers only")
            replacements = []
            try:
                for mapping in (self.h2d_workers, self.d2h_workers):
                    for key in mapping:
                        if key.dp_client_id != dp_client_id:
                            continue
                        worker = self._create_gpu_cpu_worker(key, self.gpu_handle_groups[key])
                        replacements.append((mapping, key, worker))
                        deadline = time.monotonic() + 60.0
                        while not worker.ready_event.wait(0.1):
                            if not worker.process.is_alive() or time.monotonic() >= deadline:
                                raise RuntimeError("Replacement GPU worker failed readiness")
                        if not worker.process.is_alive():
                            raise RuntimeError("Replacement GPU worker died after readiness")
            except BaseException:
                for _, _, worker in replacements:
                    worker.abort()
                raise
            for mapping, key, worker in replacements:
                self._dead_workers.discard(id(worker))
                mapping[key] = worker
            self._route_generation[dp_client_id] = current + 1
            self._disabled_routes.discard(dp_client_id)
            return {"safe": True, "generation": current + 1}
