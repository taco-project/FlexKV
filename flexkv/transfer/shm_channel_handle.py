# SPDX-License-Identifier: Apache-2.0
# cython: boundscheck=True, wraparound=True
"""
Shared-memory variant of TransferManagerHandle for the multi-DP path.

Architecture:
- N CE processes each hold one `TransferManagerShmChannelHandle` connected to
  the single TE process via a `ShmChannel` named after the (server_id,
  channel_id).
- The TE process runs a multi-channel dispatcher loop (`_te_shm_main`) that
  polls all N submit rings, hands graphs to the underlying TransferManager,
  and routes each completed op back to its originating channel via a
  `graph_id → channel_id` map.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import nvtx

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import CompletedOp, TransferOpGraph
from flexkv.transfer.shm_channel import ShmChannel, ShmControlBlock


# Wire format: small dicts so we don't pay for full TransferOpGraph re-pickling
# more than necessary. Submit messages carry the graph itself; we'll let pickle
# handle the structure.

class _SubmitMsg:
    __slots__ = ("graph", "task_end_op_id", "is_batch")

    def __init__(self, graph, task_end_op_id: int = -1, is_batch: bool = False):
        self.graph = graph
        self.task_end_op_id = task_end_op_id
        self.is_batch = is_batch


# The result ring carries fixed-width CompletedOp records directly (no wrapper).


# CE-side handle ---------------------------------------------------------

class TransferManagerShmChannelHandle:
    """CE-side handle that submits transfer graphs to a shared TE via shmem."""

    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 server_id: str,
                 channel_id: int,
                 file_wait_timeout_s: float = 60.0):
        from flexkv.transfer.shm_channel import _safe_id

        self.model_config = model_config
        self.cache_config = cache_config
        self.server_id = server_id
        self.channel_id = channel_id

        safe = _safe_id(server_id)
        self._ctrl_path = f"/dev/shm/flexkv_te_ctrl_{safe}"
        self._ch_path = f"/dev/shm/flexkv_te_ch_{safe}_{channel_id}"
        self._file_wait_timeout_s = file_wait_timeout_s

        # Poll for shm files (created in TE subprocess by setup_channels()).
        # Existence does NOT mean the TM is initialized — that's signalled
        # by the ctrl ready flag and observed via `is_ready()`.
        deadline = time.time() + file_wait_timeout_s
        while time.time() < deadline:
            if os.path.exists(self._ctrl_path) and os.path.exists(self._ch_path):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                f"Timed out waiting for TE shm files: "
                f"{self._ctrl_path}, {self._ch_path}"
            )
        # Attaching is non-blocking once files exist.
        self._ctrl = ShmControlBlock(server_id, create=False)
        self._channel = ShmChannel(server_id, channel_id, create=False)

    # TransferManagerHandleBase interface -------------------------------

    def start(self) -> None:
        # Nothing to do — TE creates and starts the channel.
        pass

    def is_ready(self) -> bool:
        # The TE subprocess flips the ctrl ready flag after the
        # TransferManager (incl. GPU registration) finishes initializing.
        return self._ctrl._ready.value != 0

    def submit(self, transfer_graph: TransferOpGraph,
               task_end_op_id: int = -1) -> None:
        nvtx_range = nvtx.start_range(
            message="TransferManagerShmChannelHandle.submit", color="green"
        )
        self._channel.submit_send(_SubmitMsg(transfer_graph, task_end_op_id))
        self._ctrl.notify()  # wake the TE: an idle _poll_submits parks on the ctrl futex for up to 100 ms
        nvtx.end_range(nvtx_range)

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        # Send each graph as its own submit message — keeps the TE side
        # simple. Could be batched into a list for fewer pickle calls if
        # benchmarks show it matters.
        for g in transfer_graphs:
            self._channel.submit_send(_SubmitMsg(g, -1, is_batch=True))
        if transfer_graphs:
            self._ctrl.notify()   # see submit()

    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        if timeout is None:
            timeout = 0.0
        out: List[CompletedOp] = self._channel.result_recv(timeout_s=timeout)
        if out and os.environ.get("FLEXKV_TRACE_TE", "0") == "1":
            completed_graphs = sorted({op.graph_id for op in out
                                        if op.is_graph_completed()})
            flexkv_logger.info(
                f"[TE-TRACE] CE recv ch={self.channel_id} "
                f"completed_graphs={completed_graphs} n_ops={len(out)}"
            )
        return out

    def shutdown(self) -> None:
        try:
            self._channel.close()
        except Exception:
            pass
        try:
            self._ctrl.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass


# TE-side multi-channel dispatcher --------------------------------------

class _TEShmDispatcher:
    """Polls N channels, forwards submits to TransferManager, routes results.

    Two-phase startup:
      1) `setup_channels()` creates the control block + per-channel shm files
         and sets the ready flag. CE-side handles can attach as soon as this
         returns. TM does not need to exist yet.
      2) `start_dispatch(transfer_manager)` launches the polling threads. Call
         this after the TM has finished initializing.
    """

    def __init__(self, server_id: str, num_channels: int):
        self._tm = None
        self._server_id = server_id
        self._num_channels = num_channels
        self._ctrl: Optional[ShmControlBlock] = None
        self._channels: List[ShmChannel] = []
        # graph_id -> channel_id (submitter)
        self._graph_owner: Dict[int, int] = {}
        self._owner_lock = threading.Lock()
        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._result_thread: Optional[threading.Thread] = None

    def setup_channels(self) -> None:
        """Create shm control block + per-channel files. Idempotent w.r.t. CE
        attaches — CE-side handles only need these files to exist."""
        self._ctrl = ShmControlBlock(self._server_id, create=True)
        self._channels = [
            ShmChannel(self._server_id, ch_id, create=True)
            for ch_id in range(self._num_channels)
        ]
        flexkv_logger.info(
            f"TE shm dispatcher: {self._num_channels} channels created on "
            f"server_id={self._server_id}"
        )

    def start_dispatch(self, transfer_manager) -> None:
        """Bind the TransferManager and start the polling threads. The ctrl
        ready flag is flipped so CEs that were spinning on `wait_ready`
        can proceed."""
        self._tm = transfer_manager
        assert self._ctrl is not None, "setup_channels() must run first"
        self._ctrl.set_ready()
        self._poll_thread = threading.Thread(
            target=self._poll_submits, daemon=True, name="te-shm-poll"
        )
        self._result_thread = threading.Thread(
            target=self._poll_results, daemon=True, name="te-shm-result"
        )
        self._poll_thread.start()
        self._result_thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        # Wake up futex waiters so threads can exit.
        if self._ctrl is not None:
            self._ctrl.notify()
        for ch in self._channels:
            try:
                ch.close()
            except Exception:
                pass
        self._channels = []
        if self._ctrl is not None:
            try:
                self._ctrl.close()
                self._ctrl.unlink()
            except Exception:
                pass
            self._ctrl = None

    def _poll_submits(self) -> None:
        idle_spins = 0
        while not self._stop.is_set():
            had_work = False
            for ch in self._channels:
                msgs = ch.submit_recv()
                if not msgs:
                    continue
                had_work = True
                for m in msgs:
                    if not isinstance(m, _SubmitMsg):
                        flexkv_logger.warning(
                            f"TE got unexpected submit msg type: {type(m)}"
                        )
                        continue
                    graph = m.graph
                    with self._owner_lock:
                        self._graph_owner[graph.graph_id] = ch.channel_id
                    try:
                        self._tm.submit(graph)
                    except Exception as e:  # noqa: BLE001 - one bad graph must not stop the node
                        flexkv_logger.error(
                            f"TE could not submit graph {graph.graph_id} from channel "
                            f"{ch.channel_id}: {e!r}; failing it", exc_info=True)
                        self._fail_graph(ch, graph.graph_id)
            if had_work:
                idle_spins = 0
                continue
            idle_spins += 1
            if idle_spins >= 1000:
                # Idle: futex wait on ctrl wake counter.
                snapshot = self._ctrl.get_wake() if self._ctrl else 0
                # Re-check after snapshot — necessary to avoid lost wakeup.
                any_pending = any(
                    ch._submit_r.value != ch._submit_w.value
                    for ch in self._channels
                )
                if any_pending:
                    idle_spins = 0
                    continue
                if self._ctrl is not None:
                    self._ctrl.wait(snapshot,
                                    timeout_ns=int(0.1 * 1_000_000_000))
                idle_spins = 0

    def _poll_results(self) -> None:
        while not self._stop.is_set():
            try:
                completed = self._tm.wait(timeout=0.05)
            except Exception as e:  # pragma: no cover
                flexkv_logger.error(f"TE result poll error: {e}")
                time.sleep(0.01)
                continue
            if not completed:
                continue
            # Group completed ops by owner channel.
            by_channel: Dict[int, List[CompletedOp]] = {}
            for op in completed:
                with self._owner_lock:
                    owner = self._graph_owner.get(op.graph_id)
                    if op.op_id == -1:
                        # Terminal message (completed OR failed) — drop the
                        # mapping after we've grouped, or failed graphs leak it.
                        self._graph_owner.pop(op.graph_id, None)
                if owner is None:
                    flexkv_logger.warning(
                        f"TE got completed op for unknown graph {op.graph_id}"
                    )
                    continue
                by_channel.setdefault(owner, []).append(op)
            for ch_id, ops in by_channel.items():
                if not (0 <= ch_id < len(self._channels)):
                    continue
                try:
                    self._channels[ch_id].result_send(ops)
                except Exception as e:  # noqa: BLE001 - keep serving the other channels
                    flexkv_logger.error(
                        f"TE could not deliver {len(ops)} completion(s) to channel "
                        f"{ch_id} (graphs {sorted({op.graph_id for op in ops})}): {e!r}",
                        exc_info=True)

    def _fail_graph(self, ch, graph_id: int) -> None:
        """Tell the owning CE that `graph_id` is over and failed when the TE could
        not run it, so its task errors out instead of waiting forever."""
        from flexkv.common.transfer import CompletedOp
        with self._owner_lock:
            self._graph_owner.pop(graph_id, None)
        try:
            ch.result_send([CompletedOp(graph_id=graph_id, op_id=-1, transfer_type=None,
                                        num_blocks=0, num_bytes=0, wait_ms=0.0,
                                        xfer_ms=0.0, e2e_ms=0.0, failed=True)])
        except Exception as e:  # noqa: BLE001
            flexkv_logger.error(
                f"TE could not report the failure of graph {graph_id} to channel "
                f"{ch.channel_id}: {e!r}")


def te_shm_main(model_config: ModelConfig,
                cache_config: CacheConfig,
                gpu_register_port: str,
                server_id: str,
                num_channels: int,
                start_event,
                ready_event,
                stop_event) -> None:
    """Entrypoint for the TE subprocess in `mode="shm"`.

    Mirrors `TransferManagerInterProcessHandle._process_worker` but replaces
    the single mp.Pipe with N shm channels.

    Critical ordering: shm channel files (`flexkv_te_ctrl_*`,
    `flexkv_te_ch_*_*`) must exist before any CE attaches. We therefore
    create the dispatcher's channels FIRST (so CEs can open the files), then
    bring up the TransferManager (which blocks on GPU registration), then
    bind the TM into the dispatcher and flip the ready flag.
    """
    from flexkv.transfer_manager import TransferManager
    dispatcher = None
    tm = None
    try:
        os.environ["MPI4PY_RC_INITIALIZE"] = "false"

        # Phase 1: create shm channels — CE side can attach now.
        dispatcher = _TEShmDispatcher(server_id, num_channels)
        dispatcher.setup_channels()
        # Signal start (but not ready) so the parent's `_start_event.wait()`
        # returns. Ready flag is set later by start_dispatch().
        start_event.set()

        # Phase 2: build and start the TransferManager. This blocks waiting
        # for GPU clients to register over the zmq gpu_register_port.
        tm = TransferManager(model_config, cache_config, gpu_register_port)
        tm.initialize_transfer_engine()
        tm.start()
        # TransferManager binds a GPU-control REP socket in __init__; every
        # deployment mode must service it or a client suspend/resume call
        # stalls for the full 120s RCVTIMEO.
        tm.start_gpu_control_listener()

        # Phase 3: bind TM, flip ready flag, launch poll threads.
        dispatcher.start_dispatch(tm)
        ready_event.set()

        # Block until parent terminates us.
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except Exception as e:
        flexkv_logger.error(f"te_shm_main failed: {e}", exc_info=True)
    finally:
        if dispatcher is not None:
            try:
                dispatcher.shutdown()
            except Exception:
                pass
        if tm is not None:
            try:
                tm.shutdown()
            except Exception:
                pass
