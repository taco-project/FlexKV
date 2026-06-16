"""Lifecycle tests for batched KVTasks (as_batch=True).

These cover the behaviour introduced when the engine takes ownership of
sub-tasks at batch-merge time (eager pop) instead of the old back-reference
+ refcount tracing:

* launch(as_batch=True) returns a single batch handle;
* the merged sub-tasks are removed from the engine immediately;
* waiting on a merged-away sub-task id returns NOTFOUND;
* waiting on the batch id yields one response whose return_mask is the
  per-sub-task list;
* the batch task and all sub-tasks are reclaimed afterwards (no leak).

The existing test_kvmanager suite only exercises batched GET; here we also
cover batched PUT, which has no other coverage.
"""
import time
import multiprocessing as mp

import pytest
import torch

from flexkv.common.request import KVResponseStatus
from flexkv.kvmanager import KVManager

from common_utils import (
    generate_request_pair,
    block_ids_2_slot_mapping,
    skip_if_insufficient_gpus,
)
from test_kvmanager import run_tp_client, shutdown_tp_client

GPU_LAYOUT_TYPE = 0
NUM_REQUESTS = 4
BLOCK_PER_REQUEST = 2


@pytest.fixture
def running_kvmanager(model_config, cache_config, test_config):
    """Bring up a direct-mode KVManager with one registered TP client."""
    skip_if_insufficient_gpus(model_config.tp_size * model_config.dp_size)

    num_gpu_blocks = test_config["num_gpu_blocks"]
    kvmanager = KVManager(model_config, cache_config)
    kvmanager.start()

    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe()
    tp_proc = mp_ctx.Process(
        target=run_tp_client,
        args=(0, 0, kvmanager.gpu_register_port, model_config, cache_config,
              num_gpu_blocks, child_conn, GPU_LAYOUT_TYPE),
        daemon=True,
    )
    tp_proc.start()
    # Block until the TP client has registered its GPU blocks with the server.
    assert parent_conn.recv() is not None, "TP client failed to register GPU blocks"
    parent_conn.close()

    while not kvmanager.is_ready():
        time.sleep(0.5)

    try:
        yield kvmanager, num_gpu_blocks
    finally:
        shutdown_tp_client([tp_proc])
        kvmanager.shutdown()


def _assert_batch_lifecycle(kvmanager, sub_ids, launched_ids, num_sub):
    """Assert the eager-pop / single-handle / no-leak invariants for one batch."""
    engine = kvmanager.kv_task_engine

    # launch merged the sub-tasks into a single, distinct batch handle.
    assert len(launched_ids) == 1
    batch_id = launched_ids[0]
    assert batch_id not in sub_ids

    # The sub-tasks were popped from the engine at merge time.
    for sub_id in sub_ids:
        assert sub_id not in engine.tasks, f"sub-task {sub_id} was not eager-popped"

    # New contract: waiting on a merged-away sub-task id is NOTFOUND.
    notfound = kvmanager.wait([sub_ids[0]], timeout=2.0)
    assert notfound[sub_ids[0]].status == KVResponseStatus.NOTFOUND

    # Waiting on the batch id returns one response; return_mask is the per-sub list.
    result = kvmanager.wait([batch_id], completely=True)
    assert result[batch_id].status == KVResponseStatus.SUCCESS
    assert isinstance(result[batch_id].return_mask, list)
    assert len(result[batch_id].return_mask) == num_sub

    # No leak: the batch task and its graph mapping are gone after observation.
    assert batch_id not in engine.tasks
    assert batch_id not in engine.graph_to_task.values()
    return batch_id


@pytest.mark.parametrize(
    "cache_config",
    [{"enable_cpu": True, "enable_ssd": False, "num_cpu_blocks": 1024}],
    indirect=True,
)
def test_batched_put_and_get_lifecycle(running_kvmanager, cache_config):
    kvmanager, num_gpu_blocks = running_kvmanager
    engine = kvmanager.kv_task_engine
    tokens_per_block = cache_config.tokens_per_block

    pairs = [
        generate_request_pair(i, BLOCK_PER_REQUEST, num_gpu_blocks, tokens_per_block, 1)
        for i in range(NUM_REQUESTS)
    ]

    # ---- batched PUT (first write -> every sub-task has a non-empty graph) ----
    put_ids, put_slot_mappings = [], []
    for token_ids, block_ids, dp_id in pairs:
        task_id, _ = kvmanager.put_match(token_ids=token_ids, token_mask=None, dp_id=dp_id)
        put_ids.append(task_id)
        put_slot_mappings.append(block_ids_2_slot_mapping(block_ids, tokens_per_block))
    launched_put = kvmanager.launch(put_ids, put_slot_mappings, as_batch=True)
    _assert_batch_lifecycle(kvmanager, put_ids, launched_put, NUM_REQUESTS)

    # ---- batched GET of the same tokens -> should fully hit the cache ----
    get_ids, get_slot_mappings = [], []
    for token_ids, block_ids, dp_id in pairs:
        task_id, _ = kvmanager.get_match(token_ids=token_ids, token_mask=None, dp_id=dp_id)
        get_ids.append(task_id)
        get_slot_mappings.append(block_ids_2_slot_mapping(block_ids, tokens_per_block))
    launched_get = kvmanager.launch(get_ids, get_slot_mappings, as_batch=True)
    _assert_batch_lifecycle(kvmanager, get_ids, launched_get, NUM_REQUESTS)

    # ---- global no-leak guard: every task and graph mapping reclaimed ----
    # engine.tasks is an ExpiringDict whose keys() also exposes internal
    # attributes (max_age_seconds / max_len), so only count real (int) task ids.
    leaked_tasks = [k for k in engine.tasks.keys() if isinstance(k, int)]
    assert leaked_tasks == [], f"leaked tasks: {leaked_tasks}"
    assert len(engine.graph_to_task) == 0, f"leaked graphs: {engine.graph_to_task}"
