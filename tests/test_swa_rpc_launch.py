# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

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
