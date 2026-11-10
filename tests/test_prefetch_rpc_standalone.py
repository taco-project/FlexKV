"""Real client/server request code over a pickle queue instead of ZMQ sockets."""
import pickle
import queue
import threading
import types
import unittest
from submission_test_support import ROOT,NS,base,load_nodes,method,logger,req
from test_hy4_cache_pipeline_standalone import Pipeline

wire=load_nodes(ROOT/'flexkv/server/request.py',
    {'PrefetchRequest','WaitRequest','TryWaitRequest','Response','GetMatchRequest',
     'PutMatchRequest','LaunchTaskRequest','CancelTaskRequest'},base)
ns=dict(base)
ns.update({k:v for k,v in vars(wire).items() if not k.startswith('_')})
ns.update(flexkv_logger=logger,KVResponseStatus=req.KVResponseStatus)
client_class=load_nodes(ROOT/'flexkv/server/client.py',{'KVDPClient'},ns).KVDPClient

class Endpoint:
    def __init__(self,mail):self.mail=mail;self.after_send_error=False
    def send_pyobj(self,value):
        self.mail.put(pickle.dumps(value))
        if self.after_send_error:
            self.after_send_error=False
            raise RuntimeError('send accepted then raised')
    def recv_pyobj(self):return pickle.loads(self.mail.get(timeout=5))

class ServerHarness:
    def __init__(self,pipeline=None):
        self.pipeline=pipeline or Pipeline();self.inbox=queue.Queue();self.outbox=queue.Queue();self.errors=[]
        self.complete_on_wait=False
        self.send,self.recv=Endpoint(self.inbox),Endpoint(self.outbox)
        self.client=client_class.__new__(client_class)
        self.client.dp_client_id=0;self.client._task_id_range=(0,10000000)
        self.client._task_id_counter=0;self.client._task_id_lock=threading.Lock()
        self.client._rpc_lock=threading.Lock()
        self.client.send_to_server=self.send;self.client.recv_from_server=self.recv
        self.server=NS(kv_task_engine=self.pipeline.engine,
                       client_manager=NS(get_zmq=lambda dp:self.recv))
        self.handlers={kind:method(ROOT/'flexkv/server/server.py','KVServer',name,ns)
            for kind,name in ((wire.PrefetchRequest,'_handle_prefetch_request'),
                              (wire.TryWaitRequest,'_handle_try_wait_request'),
                              (wire.WaitRequest,'_handle_wait_request'),
                              (wire.GetMatchRequest,'_handle_get_match_request'),
                              (wire.PutMatchRequest,'_handle_put_match_request'),
                              (wire.LaunchTaskRequest,'_handle_launch_task_request'),
                              (wire.CancelTaskRequest,'_handle_cancel_task_request'))}
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def run(self):
        while True:
            item=self.inbox.get()
            if item is None:return
            if isinstance(item,tuple):
                if len(item)==3:
                    function,done,result=item
                    try:result.append(function())
                    except BaseException as exc:result.append(exc)
                    finally:done.set()
                    continue
                tid,done=item
                try:self.pipeline.complete(tid,consume=False)
                except Exception as exc:self.errors.append(exc)
                finally:done.set()
                continue
            request=pickle.loads(item)
            try:
                if self.complete_on_wait and isinstance(request,wire.WaitRequest):
                    from serving_cpu_test_support import finish
                    for tid in request.wait_task_ids:finish(self.pipeline,tid,consume=False)
                self.handlers[type(request)](self.server,request)
            except Exception as exc:self.errors.append(exc)  # Async request has no ACK.
    def complete(self,tid):
        done=threading.Event();self.inbox.put((tid,done))
        assert done.wait(5),'server did not progress completion'
    def close(self):
        self.inbox.put(None);self.thread.join(5)
        assert not self.thread.is_alive(),'server test thread leaked'
    def call(self,function):
        done=threading.Event();result=[];self.inbox.put((function,done,result))
        assert done.wait(5),'server owner command timed out'
        if isinstance(result[0],BaseException):raise result[0]
        return result[0]

class PrefetchRPCTests(unittest.TestCase):
    def setUp(self):self.h=ServerHarness()
    def tearDown(self):self.h.close()
    def launch(self):
        p=self.h.pipeline
        return self.h.client.prefetch_async(p.ids,namespace=p.namespace)

    def test_pickle_request_response_roundtrip_uses_real_engine_mask(self):
        tid=self.launch()
        self.assertEqual(self.h.client.try_wait([tid]),{})
        self.h.complete(tid)
        response=self.h.client.try_wait([tid])[tid]
        self.assertEqual(response.status.value,'success')
        self.assertEqual(sum(response.return_mask),16)
        self.assertFalse(self.h.errors or self.h.pipeline.engine.tasks)

    def test_async_pre_submit_rejection_is_observable_without_stray_ack(self):
        tid=self.h.client.prefetch_async([[1,2],[3,4]])
        response=self.h.client.try_wait([tid])[tid]
        self.assertEqual(response.status.value,'failed')
        self.assertEqual(len(self.h.errors),1)
        self.assertTrue(self.h.outbox.empty())
        self.assertEqual(self.h.pipeline.manager.cpu_cache_engine.mempool.num_used_blocks,0)

    def test_ambiguous_accepted_send_keeps_actual_client_id_drainable(self):
        self.h.send.after_send_error=True
        with self.assertRaisesRegex(RuntimeError,'accepted then raised') as caught:self.launch()
        tid=caught.exception.launched_task_ids[0]
        self.assertEqual(self.h.client.try_wait([tid]),{})
        self.assertEqual(self.h.pipeline.manager.cpu_cache_engine.mempool.num_used_blocks,4)
        self.h.complete(tid)
        self.assertEqual(self.h.client.wait([tid],wait_timeout=0,completely=True)[tid].status.value,'success')
        self.assertFalse(self.h.errors)

    def test_many_async_requests_get_unique_ids_and_consumed_terminals(self):
        ids=[]
        for _ in range(20):
            tid=self.launch();ids.append(tid)
            response=self.h.client.try_wait([tid])
            if not response:
                self.h.complete(tid)
                response=self.h.client.try_wait([tid])
            self.assertEqual(response[tid].status.value,'success')
        self.assertEqual(ids,list(range(20)))
        self.assertFalse(self.h.errors or self.h.pipeline.engine.tasks or self.h.pipeline.engine.prefetch_tasks)
        self.assertEqual(self.h.pipeline.manager.cpu_cache_engine.mempool.num_used_blocks,4)

if __name__=='__main__':unittest.main()
