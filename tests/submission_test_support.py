"""CPU-only loaders for actual submission code; no torch/native dependency.

Only imports, sockets and completion sources are replaced. Task state,
graph merging, submission bookkeeping, wait and cleanup run production code.
"""
import ast
import copy
from dataclasses import dataclass, field, replace
from enum import Enum
import logging
import os
from pathlib import Path
import sys
import threading
import time
import types
import typing
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NS = types.SimpleNamespace
noop = lambda *args, **kwargs: None
logger = NS(info=noop, error=noop, warning=noop, debug=noop,
            exception=noop, is_enabled_for=lambda *args: False)


def load_nodes(path, names, namespace=None):
    module = types.ModuleType('submission_test_' + path.stem + '_' + str(len(sys.modules)))
    sys.modules[module.__name__] = module
    module.__dict__.update({k: v for k, v in (namespace or {}).items()
                            if not k.startswith('__')})
    nodes = [node for node in ast.parse(path.read_text()).body
             if getattr(node, 'name', None) in names or
             (isinstance(node, ast.Assign) and any(
                 getattr(t, 'id', None) in names for t in node.targets))]
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    return module


def method(path, owner, name, namespace):
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == owner)
    node = next(node for node in cls.body if getattr(node, 'name', None) == name)
    node.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ns = dict(namespace)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    return ns[name]


base = dict(threading=threading, dataclass=dataclass, field=field, replace=replace,
            Enum=Enum, np=np, flexkv_logger=logger)
base.update({k: v for k, v in vars(typing).items() if not k.startswith('_')})
# The transfer module is pure Python except for its logger import.
transfer_path = ROOT / 'flexkv/common/transfer.py'
transfer_names = {getattr(n, 'name', '') for n in ast.parse(transfer_path.read_text()).body}
from enum import IntEnum
base['IntEnum'] = IntEnum
tr = load_nodes(transfer_path, transfer_names - {''}, base)
req = load_nodes(ROOT / 'flexkv/common/request.py', {'KVResponse', 'KVResponseStatus'}, base)
trace = NS(ENABLED=False, CACHE_DETAILS=False, task_event=noop, graph_event=noop)
remote_ns = dict(TransferSubmissionError=tr.TransferSubmissionError, flexkv_logger=logger)


class RemoteEndpoint:
    submit = method(ROOT / 'flexkv/transfer_manager.py', 'TransferManagerMultiNodeHandle',
                    'submit', remote_ns)
    submit_batch = method(ROOT / 'flexkv/transfer_manager.py', 'TransferManagerMultiNodeHandle',
                          'submit_batch', remote_ns)

    def __init__(self, send):
        self._connected = True
        self.command_socket = NS(send_pyobj=send)


ns = dict(vars(tr))
ns.update(base, time=time, os=os, copy=copy, logging=logging,
    KVResponse=req.KVResponse, KVResponseStatus=req.KVResponseStatus,
    TransferManagerMultiNodeHandle=RemoteEndpoint,
    DEFAULT_CACHE_STRATEGY=NS(), CPUONLY_CACHE_STRATEGY=NS(),
    GLOBAL_CONFIG_FROM_ENV=NS(enable_layerwise_transfer=True),
    put_trace=trace, get_global_collector=lambda: None,
    get_nvtx_default_color=lambda: 'red',
    nvtx=NS(start_range=noop, end_range=noop, mark=noop, push_range=noop, pop_range=noop))
km = load_nodes(ROOT / 'flexkv/kvtask.py', {
    'TaskStatus', 'TaskType', 'KVTask', 'TASK_STATUS_TO_RESPONSE_STATUS',
    'convert_to_response_status', '_longest_success_prefix', 'KVTaskManager', 'KVTaskEngine'}, ns)


class Handle:
    def __init__(self, remote=False):
        self.events = []
        self.sent = []
        self.before_send = None
        self.after_send = None
        self._handle = RemoteEndpoint(self.send) if remote else object()

    def send(self, message):
        if self.before_send:
            self.before_send(message)
        graphs = message['graphs'] if 'graphs' in message else [message['graph']]
        self.sent.extend(graph.graph_id for graph in graphs)
        if self.after_send:
            self.after_send(message)

    def submit(self, graph, task_end_op_id=-1):
        self._handle.submit(graph, task_end_op_id)

    def submit_batch(self, graphs):
        if isinstance(self._handle, RemoteEndpoint):
            self._handle.submit_batch(graphs)
        else:
            self.send({'graphs': graphs})

    def wait(self, timeout=0):
        events, self.events = self.events, []
        return events

    def shutdown(self):
        pass


class Plan:
    def __init__(self):
        self.abort = Mock()
        self.commit = Mock()

    def __call__(self):
        self.commit()


def engine(ids=(17,), handles=2, pp=True):
    obj = object.__new__(km.KVTaskEngine)
    obj.tasks, obj.graph_to_task = {}, {}
    obj._terminal_tasks, obj.prefetch_tasks = {}, {}
    obj.uncompleted_graphs, obj.uncompleted_ops, obj.uncompleted_op_results = {}, {}, {}
    obj.transfer_handles = [Handle()] + [Handle(remote=True) for _ in range(handles - 1)]
    obj.required_completed_count = handles
    obj.model_config = NS(nnodes=handles, pp_size=2 if pp else 1)
    obj.cache_config = NS(tokens_per_block=1)
    obj.cache_engine = NS(slot_mapping_to_block_ids=lambda slots, size: slots[::size] // size)
    obj.tracer = NS(trace_launch_tasks=noop, trace_wait_request=noop)
    obj._log_task_terminal = Mock()
    obj.task_id_lock, obj.task_id_counter = threading.Lock(), 99
    obj._returned_pending_tasks = set()
    plans = {}
    for tid in ids:
        graph = tr.TransferOpGraph()
        op = tr.TransferOp(graph_id=graph.graph_id, transfer_type=tr.TransferType.H2D,
                           src_block_ids=np.arange(4), dst_block_ids=np.zeros(4, dtype=np.int64))
        graph.add_transfer_op(op)
        graph.slot_mapping_task_id = tid
        plans[tid] = Plan()
        obj.tasks[tid] = km.KVTask(tid, km.TaskType.GET, op.op_id, False,
            km.TaskStatus.UNREADY, np.arange(4), np.zeros(4, dtype=np.int64),
            np.ones(4, dtype=bool), graph, np.ones(4, dtype=bool), plans[tid], {})
        obj.graph_to_task[graph.graph_id] = tid
    return obj, plans


def terminal(handle, gid, failed=False):
    handle.events.append(tr.CompletedOp.failed_graph(gid) if failed
                         else tr.CompletedOp.completed_graph(gid))
