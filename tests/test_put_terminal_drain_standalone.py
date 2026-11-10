"""CPU lifecycle regressions using production methods with fake transport/tier.
No accelerator performance claims.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
from enum import Enum
from contextlib import nullcontext
import time
import os
import contextlib
import unittest
ROOT=Path(__file__).resolve().parents[1]
class Status(Enum):
    RUNNING=1; COMPLETED=2; CANCELLED=3; FAILED=4; UNREADY=5
class Kind(Enum):
    PUT=1; PREFETCH=2; GET=3; BATCH_GET=4
class Device(Enum):
    CPU=1; SSD=2; REMOTE=3
noop=lambda *a,**k: None
trace=NS(task_event=noop, ENABLED=False, CACHE_DETAILS=False, poll_stage=lambda *a:nullcontext())
ns=dict(get_nvtx_default_color=lambda: 'red', time=time, os=os, TaskStatus=Status, TaskType=Kind, DeviceType=Device,
        TransferType=NS(REMOTE2H=NS(value='remote')), get_global_collector=lambda:None,
        put_trace=trace, invoke_op_callback=lambda cb,op:cb(), flexkv_logger=NS(error=noop,warning=noop,info=noop), logger=NS(exception=noop),
        nvtx=NS(start_range=noop,end_range=noop,push_range=noop,pop_range=noop),
        convert_to_response_status=lambda s:'success', KVResponse=lambda **k:NS(**k),
        _status_value=lambda r:r.status, _is_terminal_status=lambda s:s=='success',
        np=NS(zeros_like=lambda x: [0]*len(x)))
def method(file,name):
    tree=ast.parse((ROOT/file).read_text())
    f=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name)
    f.decorator_list=[]
    m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[])
    ast.fix_missing_locations(m);exec(compile(m,str(ROOT/file),'exec'),ns)
    return ns[name]
km='flexkv/kvtask.py'; ce='flexkv/cache/cache_engine.py'
methods={n:method(km,n) for n in ['_fail_task','progress_returned_tasks','_update_tasks','check_completed','_mark_completed','_release_task','_remember_terminal','_reap_completed_tasks','_wait_impl','get_match']}
opcb=method(ce,'_op_callback'); graphcb=method(ce,'_transfer_callback')
poll=method('flexkv/integration/sglang/connector.py','check_completed_stores')
connector_has_pending=method('flexkv/integration/sglang/connector.py','has_pending_store_graphs')
connector_progress=method('flexkv/integration/sglang/connector.py','progress_store_graphs')
facade_progress=method('flexkv/kvmanager.py','progress_returned_tasks')
class NSArray(list):
    def __getitem__(self,key):
        return self if isinstance(key,list) else super().__getitem__(key)
class Tests(unittest.TestCase):
    def setup_case(self,p2p=False):
        self.node=NS(ready=False,locked=True,published=False)
        def publish(n):n.published=True
        engine=NS(unlock=lambda n:setattr(n,'locked',False),set_ready=lambda n,*a:setattr(n,'ready',True),local_index=NS(insert_and_publish=publish))
        self.cache=NS(cpu_cache_engine=engine,cache_config=NS(enable_p2p_cpu=p2p))
        self.t=NS(status=Status.RUNNING,callback=lambda:graphcb(self.cache,{Device.CPU:(self.node,3)},is_put=True),
            graph=NS(graph_id=10,_op_map={},num_ops=1), task_type=Kind.PUT,transfer_failed=False, submission_error=None, cancel_requested=False, prefetch_key=None, task_end_op_id=4,
            task_end_op_finished=False,request_returned=False,return_mask=[True],
            op_callback_dict={4:lambda:opcb(self.cache,Device.CPU,self.node,3)})
        self.t.is_completed=lambda:self.t.status in [Status.COMPLETED,Status.FAILED,Status.CANCELLED]
        self.t.shed_heavy_resources=noop
        self.q=[];self.drains=0
        def drain(timeout):
            self.drains+=1; out=self.q[:];self.q.clear();return out
        self.manager=NS(tasks={7:self.t},graph_to_task={10:7},uncompleted_graphs={},_get_completed_ops=drain,_process_empty_graph=noop,
                        _terminal_tasks={},prefetch_tasks={},_returned_pending_tasks=set(),
                        _abort_task_plans=noop,_log_task_terminal=noop,tracer=NS(trace_request=noop))
        for n,f in methods.items():setattr(self.manager,n,MethodType(f,self.manager))
        self.manager.try_wait=lambda task_ids:self.manager._wait_impl(task_ids,only_return_finished=True)
        facade=NS(server_client_mode=False,kv_task_engine=self.manager,try_wait=self.manager.try_wait)
        facade.progress_returned_tasks=MethodType(facade_progress,facade)
        self.connector=NS(_sync_ctx=NS(is_sync_leader=True,needs_sync=False),kv_manager=facade,
                          _inflight_stores={'rid':7},_returned_store_graphs_pending=False,
                          _pop_context=lambda *a:None,_log_cache_op=noop)
        self.connector.has_pending_store_graphs=MethodType(
            connector_has_pending,self.connector)
        self.connector.progress_store_graphs=MethodType(
            connector_progress,self.connector)
    def event(self,op):
        return NS(graph_id=10,op_id=op,transfer_type=None,block_results=None,
                  is_graph_failed=lambda:False,is_graph_completed=lambda:op==-1)
    def early(self):
        self.q.append(self.event(4));self.assertEqual(poll(self.connector),['rid'])
        self.assertTrue(self.node.ready);self.assertTrue(self.node.locked)
        self.assertTrue(self.t.request_returned);self.assertIn(7,self.manager.tasks)
    def test_late_terminal_drained_after_gpu_release_and_idle_stops_polling(self):
        self.setup_case();self.early();self.q.append(self.event(-1))
        self.assertFalse(self.connector.progress_store_graphs())
        self.assertFalse(self.node.locked);self.assertNotIn(7,self.manager.tasks)
        n=self.drains
        for _ in range(100):self.assertFalse(self.connector.progress_store_graphs())
        self.assertEqual(self.drains,n)
    def test_delayed_terminal_remains_tracked_across_empty_polls(self):
        self.setup_case();self.early()
        for _ in range(10):self.assertTrue(self.connector.progress_store_graphs())
        self.assertTrue(self.node.locked);self.assertIn(7,self.manager.tasks)
        self.q.append(self.event(-1));self.connector.progress_store_graphs()
        self.assertFalse(self.node.locked);self.assertNotIn(7,self.manager.tasks)
    def test_reset_or_expired_task_stops_background_polling(self):
        for expired in (False,True):
            self.setup_case();self.early()
            if expired:self.manager.tasks.clear()
            else:self.t.status=Status.CANCELLED
            n=self.drains;self.connector.progress_store_graphs()
            self.assertEqual(self.drains,n)
            self.assertFalse(self.manager._returned_pending_tasks)
    def test_next_local_lookup_drains_before_matching(self):
        self.setup_case();self.early();self.q.append(self.event(-1))
        def match(*a,**k):
            self.assertTrue(self.node.ready);self.assertFalse(self.node.locked);self.assertNotIn(7,self.manager.tasks)
            return 8,[True]
        self.manager._get_match_impl=match
        self.assertEqual(self.manager.get_match(NSArray([1]),token_mask=[True]),(8,[True]))
    def test_p2p_publish_waits_for_terminal(self):
        self.setup_case(p2p=True);self.early();self.assertFalse(self.node.published)
        self.q.append(self.event(-1));self.manager._update_tasks(timeout=0)
        self.assertTrue(self.node.published);self.assertFalse(self.node.locked);self.assertNotIn(7,self.manager.tasks)
    def test_empty_wait_can_pump_terminal(self):
        self.setup_case();self.early();self.q.append(self.event(-1))
        self.assertEqual(self.manager.try_wait([]),{});self.assertFalse(self.node.locked);self.assertNotIn(7,self.manager.tasks)
    def test_late_failure_releases_returned_task(self):
        self.setup_case();self.early()
        e=self.event(-1);e.is_graph_failed=lambda:True
        self.q.append(e);self.connector.progress_store_graphs()
        self.assertEqual(self.t.status,Status.FAILED)
        self.assertNotIn(7,self.manager.tasks)
        self.assertFalse(self.manager._returned_pending_tasks)
    def test_multiple_graphs_finalize_out_of_order_without_duplicate_gpu_release(self):
        self.setup_case()
        other=Tests();other.setup_case()
        other.t.graph.graph_id=11
        self.manager.tasks[8]=other.t;self.manager.graph_to_task[11]=8
        self.connector._inflight_stores['rid2']=8
        e=self.event(4);e.graph_id=11
        self.q.extend([self.event(4),e])
        self.assertEqual(poll(self.connector),['rid','rid2'])
        self.assertEqual(self.manager._returned_pending_tasks,{7,8})
        e=self.event(-1);e.graph_id=11;self.q.append(e)
        self.assertTrue(self.connector.progress_store_graphs())
        self.assertEqual(self.manager._returned_pending_tasks,{7})
        self.assertTrue(self.node.locked);self.assertFalse(other.node.locked)
        self.q.append(self.event(-1))
        self.assertFalse(self.connector.progress_store_graphs())
        self.assertFalse(self.node.locked);self.assertFalse(self.manager.tasks)
    def test_progress_error_does_not_enter_collective(self):
        self.setup_case();self.early();calls=[]
        def fail():raise RuntimeError('transport failure')
        self.connector.kv_manager.progress_returned_tasks=fail
        self.connector._sync_ctx.needs_sync=True
        self.connector._sync_ctx.scatter=lambda rows,**kw:calls.append(rows) or rows
        ns['FlexKVScatterChannel']=NS(STORE_COMPLETION='store')
        self.assertTrue(self.connector.progress_store_graphs())
        self.assertEqual(calls,[])
        self.assertTrue(self.connector._store_finalize_error_reported)
    def test_empty_store_poll_does_not_enter_collective(self):
        self.setup_case();calls=[]
        self.connector._inflight_stores.clear()
        self.connector._sync_ctx.needs_sync=True
        self.connector._sync_ctx.scatter=lambda rows,**kw:calls.append(rows) or rows
        ns['FlexKVScatterChannel']=NS(STORE_COMPLETION='store')
        self.assertEqual(poll(self.connector),[])
        self.assertEqual(calls,[])
    def test_remote_facade_does_not_issue_idle_rpc(self):
        def fail():raise AssertionError('unexpected RPC')
        self.assertFalse(facade_progress(
            NS(server_client_mode=True,dp_client=NS(try_wait=fail))))
    def test_server_pumps_while_idle_and_polls_for_shutdown(self):
        class Start:dp_client_id=0
        class Stop:dp_client_id=0
        ns.update(StartRequest=Start,ShutdownRequest=Stop,zmq=NS(ZMQError=RuntimeError),
                  contextlib=contextlib,signal=NS(SIGINT=2,SIGTERM=15,SIG_IGN=1,signal=noop))
        run=method('flexkv/server/server.py','run')
        progress=iter([True,True,False]);requests=iter([Start(),Stop()]);polls=[];handled=[]
        available=iter([True,False,False,True])
        def poll(**kw):
            polls.append(kw)
            return next(available)
        obj=NS(start_server=noop,
               kv_task_engine=NS(progress_returned_tasks=lambda:next(progress),shutdown=noop),
               recv_from_client=NS(recv_pyobj=lambda:next(requests),poll=poll),
               request_handlers={Stop:lambda req:handled.append(req)})
        run(obj)
        self.assertEqual(polls,[{'timeout':500}]*4);self.assertEqual(len(handled),1)
    def test_diagnostics_only_emit_when_completion_events_are_consumed(self):
        self.setup_case();rows=[]
        old=ns['put_trace']
        ns['put_trace']=NS(ENABLED=True,now=time.perf_counter_ns,
                          task_event=lambda event,*a,**kw:rows.append((event,kw)),
                          emit=lambda event,**kw:rows.append((event,kw)))
        try:
            for _ in range(100):self.manager._update_tasks(timeout=0)
            self.assertEqual(rows,[])
            self.q.extend([self.event(4),self.event(-1)])
            self.manager._update_tasks(timeout=0)
            drains=[kw for event,kw in rows if event=='completion_drain']
            finals=[kw for event,kw in rows if event=='graph_finalize']
            self.assertEqual(len(drains),1);self.assertEqual(drains[0]['ops'],2)
            self.assertEqual(len(finals),1);self.assertGreaterEqual(finals[0]['duration_ns'],0)
        finally:ns['put_trace']=old
    def test_same_drain_has_no_stranded_task(self):
        self.setup_case();self.q.extend([self.event(4),self.event(-1)])
        self.assertEqual(poll(self.connector),['rid']);self.assertTrue(self.node.ready)
        self.assertFalse(self.node.locked);self.assertNotIn(7,self.manager.tasks)
if __name__=='__main__':unittest.main(verbosity=2)
