"""Receiving the consumer's per-layer eventfds, on its own.

This is the handshake half of "layerwise": before any transfer happens, the
inference framework (sglang today) opens a UDS to us and passes, per TP rank
and per counter set, one eventfd per model layer.  Those fds are the only
channel by which "layer L has landed" can be told to a process that is not
ours.

It lived inside the old ``LayerwiseTransferWorker`` because that was the only
worker that did per-layer notification.  Now that per-layer completion is a
*contract* a normal CPU<->GPU worker can be asked to honour
(``CompletionContract.PER_LAYER``), the handshake belongs to ``worker.py``
rather than to a worker class of its own.  Hence this module: no worker class,
no CUDA, no C++ handle, just a socket and a tensor.

The returned tensor is shaped ``[num_counters, tp_size, num_layers]`` flat,
int32, with ``-1`` for a rank/counter/layer the consumer did not supply.  That
is exactly what ``LayerEventfdTable`` on the C++ side consumes.  An empty
tensor means "nobody asked for notification", which every layer of the stack
treats as ``CompletionContract.WHOLE``.
"""
import os
import socket
import struct
import time
from typing import Dict, List, Tuple

import torch

from flexkv.common.config import ModelConfig
from flexkv.common.debug import flexkv_logger

# Counter sets the consumer rotates between (triple buffering): a transfer
# names one, so a layer's fd for counter 0 can still be un-consumed while
# counter 1's transfer is already in flight. The consumer tells us the real
# number in its handshake header; this is only the pre-handshake default.
_DEFAULT_NUM_COUNTERS = 3


def build_layerwise_eventfd_socket_path(
    dp_client_id: int,
    pp_rank: int,
    model_config: ModelConfig,
) -> str:
    """Construct the per-(pp, dp) UDS path both ends derive independently.

    Both this process and the consumer compute it from the same ModelConfig
    fields, so there is no env var to keep in sync between them.
    """
    base = os.environ.get(
        'FLEXKV_LAYERWISE_EVENTFD_SOCKET',
        '/tmp/flexkv_layerwise_eventfd.sock',
    )
    suffix = ""
    if model_config.pp_size > 1:
        suffix += f"_pp{pp_rank}"
    if model_config.instance_num > 1 or model_config.dp_size > 1:
        suffix += f"_dp{dp_client_id}"
    if not suffix:
        return base
    root, ext = os.path.splitext(base)
    return f"{root}{suffix}{ext}"


def _recv_fds(sock: socket.socket, num_fds: int) -> Tuple[List[int], bytes]:
    """Receive ``num_fds`` fds plus the inline header, via SCM_RIGHTS."""
    data_buf = bytearray(256)
    anc_buf_size = socket.CMSG_SPACE(num_fds * struct.calcsize("i"))

    nbytes, ancdata, flags, addr = sock.recvmsg_into([data_buf], anc_buf_size, 0)
    data = bytes(data_buf[:nbytes])

    fds: List[int] = []
    for level, ctype, cdata in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            num_received = len(cdata) // struct.calcsize("i")
            fds = list(struct.unpack(
                f"{num_received}i", cdata[:num_received * struct.calcsize("i")]))
            break
    if not fds:
        raise RuntimeError("did not receive fds via SCM_RIGHTS")
    return fds, data


def receive_layer_eventfds(
    socket_path: str,
    tp_group_size: int,
    num_layers: int,
    *,
    log_prefix: str = "[layer-eventfd]",
    max_retries: int = 180,
    retry_interval: float = 1.0,
) -> torch.Tensor:
    """Serve the UDS until every rank has handed over its fds.

    Returns ``[num_counters, tp_size, num_layers]`` flat int32, or an empty
    tensor if nobody connected before the deadline -- the caller decides
    whether that is fatal (PER_LAYER) or fine (WHOLE).
    """

    def cleanup_socket() -> None:
        try:
            if os.path.exists(socket_path):
                os.unlink(socket_path)
        except OSError:
            pass

    cleanup_socket()
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_sock.bind(socket_path)
        # Backlog well above tp_group_size: a client whose first attempt
        # raced our bind() retries, and a full backlog turns that retry into
        # a connection refused rather than a queued connection.
        server_sock.listen(tp_group_size * 3)
        os.chmod(socket_path, 0o777)
        flexkv_logger.info(
            f"{log_prefix} eventfd server created: socket={socket_path}, "
            f"waiting for {tp_group_size} connection(s)")
    except Exception as e:  # noqa: BLE001
        flexkv_logger.error(
            f"{log_prefix} failed to bind/listen on {socket_path}: {e}")
        server_sock.close()
        return torch.empty(0, dtype=torch.int32)

    # Per-connection timeout rather than one global one: a client that fails
    # mid-handshake retries, and a global timeout would give up on the whole
    # group because of one bad connection. The total deadline still bounds it.
    per_conn_timeout = 30  # seconds per accept()
    total_deadline = time.time() + max_retries * retry_interval
    server_sock.settimeout(per_conn_timeout)
    all_rank_eventfds: Dict[int, Dict[int, List[int]]] = {}
    num_counters = _DEFAULT_NUM_COUNTERS
    conn_idx = 0

    try:
        while len(all_rank_eventfds) < tp_group_size:
            if time.time() > total_deadline:
                flexkv_logger.error(
                    f"{log_prefix} deadline exceeded on {socket_path}, "
                    f"received {len(all_rank_eventfds)}/{tp_group_size} ranks")
                break

            remaining = total_deadline - time.time()
            server_sock.settimeout(min(per_conn_timeout, max(remaining, 1)))

            try:
                conn, _ = server_sock.accept()
                conn_idx += 1
                flexkv_logger.info(
                    f"{log_prefix} accepted connection {conn_idx} "
                    f"(registered {len(all_rank_eventfds)}/{tp_group_size}) "
                    f"on {socket_path}")
            except socket.timeout:
                flexkv_logger.warning(
                    f"{log_prefix} timeout waiting for connection on "
                    f"{socket_path}, registered "
                    f"{len(all_rank_eventfds)}/{tp_group_size}, retrying...")
                continue

            try:
                with conn:
                    # 16-byte header: effective_tp_rank,
                    # effective_tp_size_per_node, num_layers, num_counters.
                    metadata = conn.recv(16)
                    if len(metadata) < 16:
                        flexkv_logger.error(
                            f"{log_prefix} incomplete metadata on "
                            f"{socket_path}: expected 16 bytes, got "
                            f"{len(metadata)}")
                        continue

                    (rank_key, tp_size_recv, recv_num_layers,
                     recv_num_counters) = struct.unpack("iiii", metadata[:16])

                    if not all_rank_eventfds:
                        # First rank to arrive defines the table shape; the
                        # consumer's own count wins over ours because it is
                        # the one that will be waiting on these fds.
                        num_layers = recv_num_layers
                        num_counters = recv_num_counters

                    flexkv_logger.debug(
                        f"{log_prefix} connection {conn_idx}: "
                        f"effective_tp_rank={rank_key}, "
                        f"effective_tp_size_per_node={tp_size_recv}, "
                        f"num_layers={recv_num_layers}, "
                        f"num_counters={recv_num_counters}")

                    rank_eventfds: Dict[int, List[int]] = {}
                    for _ in range(recv_num_counters):
                        fds, extra_data = _recv_fds(conn, recv_num_layers)
                        counter_id = struct.unpack("i", extra_data[:4])[0]
                        rank_eventfds[counter_id] = fds
                        flexkv_logger.debug(
                            f"{log_prefix} received counter_id={counter_id}, "
                            f"num_fds={len(fds)} from tp_rank={rank_key}")

                    all_rank_eventfds[rank_key] = rank_eventfds
                    try:
                        conn.sendall(b"\x01")  # ACK: fds landed
                    except Exception:  # noqa: BLE001
                        pass
                    flexkv_logger.info(
                        f"{log_prefix} received all eventfds from "
                        f"effective_tp_rank={rank_key} on {socket_path}")
            except Exception as e:  # noqa: BLE001
                try:
                    conn.sendall(b"\x00")  # NACK: client should retry
                except Exception:  # noqa: BLE001
                    pass
                flexkv_logger.warning(
                    f"{log_prefix} failed to receive eventfds from connection "
                    f"{conn_idx} on {socket_path}: {e}. Client will retry, "
                    f"continuing accept loop...")
                continue
    except Exception as e:  # noqa: BLE001
        flexkv_logger.error(
            f"{log_prefix} fatal error in accept loop on {socket_path}: {e}")
    finally:
        server_sock.close()
        cleanup_socket()

    if not all_rank_eventfds:
        flexkv_logger.warning(
            f"{log_prefix} no connections received on {socket_path}")
        return torch.empty(0, dtype=torch.int32)

    # [num_counters, tp_size, num_layers], flat. A rank that never connected
    # contributes -1s rather than being dropped: the table is indexed by rank,
    # so a short row would silently shift every later rank's fds.
    eventfds_list: List[int] = []
    for counter_id in range(num_counters):
        for tp_rank in range(tp_group_size):
            fds = all_rank_eventfds.get(tp_rank, {}).get(
                counter_id, [-1] * num_layers)
            eventfds_list.extend(fds)

    tensor = torch.tensor(eventfds_list, dtype=torch.int32)
    flexkv_logger.info(
        f"{log_prefix} eventfd setup complete: socket={socket_path}, "
        f"tensor_shape={tensor.shape}, counters={num_counters}, "
        f"tp_size={tp_group_size}, layers={num_layers}")
    return tensor
