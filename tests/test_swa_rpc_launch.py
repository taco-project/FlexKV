# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import threading
from types import SimpleNamespace

import pytest

from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType
from flexkv.transfer_manager import TransferManagerOnRemote

from flexkv.server.request import LaunchTaskRequest


def test_launch_request_carries_swa_slot_mappings():
    full = [np.arange(4, dtype=np.int64)]
    swa = [np.array([7], dtype=np.int64)]

    req = LaunchTaskRequest(
        dp_client_id=3,
        task_ids=[11],
        slot_mappings=full,
        swa_slot_mappings=swa,
        as_batch=True,
        batch_id=99,
        layerwise_transfer=True,
        counter_id=2,
    )

    assert req.swa_slot_mappings is swa
    np.testing.assert_array_equal(req.swa_slot_mappings[0], swa[0])


def test_server_launch_handler_uses_keywords_for_swa_slot_mappings():
    full = [np.arange(4, dtype=np.int64)]
    swa = [np.array([9], dtype=np.int64)]
    req = LaunchTaskRequest(
        dp_client_id=0,
        task_ids=[1],
        slot_mappings=full,
        swa_slot_mappings=swa,
        as_batch=False,
        batch_id=-1,
        layerwise_transfer=False,
        counter_id=5,
    )

    calls = {}

    class _Engine:
        def launch_tasks(self, **kwargs):
            calls.update(kwargs)

    class _Server:
        kv_task_engine = _Engine()

    from flexkv.server.server import KVServer

    KVServer._handle_launch_task_request(_Server(), req)

    assert calls == {
        "task_ids": [1],
        "slot_mappings": full,
        "swa_slot_mappings": swa,
        "as_batch": False,
        "batch_id": -1,
        "layerwise_transfer": False,
        "counter_id": 5,
    }


@pytest.mark.parametrize("mapping_first", [True, False])
def test_remote_pp_store_binds_local_full_and_swa_only_after_both_arrive(mapping_first):
    graph = TransferOpGraph()
    graph.stage_local_task_id = 17001
    full = TransferOp(
        graph_id=graph.graph_id, transfer_type=TransferType.D2H,
        src_block_ids=np.array([1, 2]), dst_block_ids=np.array([10, 11]),
    )
    swa = TransferOp(
        graph_id=graph.graph_id, transfer_type=TransferType.D2H,
        src_block_ids=np.array([3]), dst_block_ids=np.array([20]), is_swa=True,
    )
    graph.add_transfer_op(full)
    graph.add_transfer_op(swa)
    submitted = []
    remote = SimpleNamespace(
        _pending_graphs={}, _pending_slot_mappings={}, _pending_lock=threading.Lock(),
        _active_graphs={}, _active_graphs_lock=threading.Lock(), submit=submitted.append,
    )
    mapping = lambda: TransferManagerOnRemote._handle_set_slot_mapping(
        remote, graph.stage_local_task_id, np.array([101, 102]), np.array([201, 202]),
    )
    arrival = lambda: TransferManagerOnRemote._handle_submit(remote, graph)
    first, second = (mapping, arrival) if mapping_first else (arrival, mapping)
    first()
    assert submitted == []
    second()
    assert submitted == [graph]
    assert full.src_block_ids.tolist() == [101, 102]
    assert swa.src_block_ids.tolist() == [202]
    assert full.dst_block_ids.tolist() == [10, 11]
    assert swa.dst_block_ids.tolist() == [20]
    assert remote._pending_graphs == remote._pending_slot_mappings == {}


def test_confirmed_launch_failure_returns_error_instead_of_orphaning_rpc():
    from flexkv.server.server import KVServer

    replies = []
    def fail(**kwargs):
        raise ValueError("invalid SWA mapping")
    server = SimpleNamespace(
        kv_task_engine=SimpleNamespace(launch_tasks=fail),
        client_manager=SimpleNamespace(
            get_zmq=lambda _: SimpleNamespace(send_pyobj=replies.append),
        ),
    )
    request = LaunchTaskRequest(
        dp_client_id=0, task_ids=[17], slot_mappings=[np.arange(4)],
        confirm_launch=True,
    )
    KVServer._handle_launch_task_request(server, request)
    assert len(replies) == 1
    assert "invalid SWA mapping" in replies[0].error_msg
