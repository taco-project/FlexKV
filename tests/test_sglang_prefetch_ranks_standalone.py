"""Actual prefetch control methods over concurrently running rank fan-out."""
import time
import unittest
from unittest.mock import Mock
import numpy as np
from submission_test_support import NS,req
from cpu_collective_test_support import Topology
from test_sglang_prefetch_standalone import cache_connector

class PrefetchRankTests(unittest.TestCase):
    def test_prefetch_start_cancel_progress_error_and_reset_across_slow_ranks(self):
        for dp,cp,tp in ((1,8,1),(2,2,2),(4,4,2)):
            topology=Topology(dp=dp,cp=cp,tp=tp)
            channels=topology.module.FlexKVScatterChannel
            def actor(index,comm):
                d,p,c,t=index;obj=cache_connector(leader=comm.is_sync_leader)
                obj._sync_ctx=comm;obj._prefetch_max_inflight=3
                for name in ('prefetch_async','poll_prefetches','_drain_prefetches_for_reset'):
                    getattr(obj,name).__func__.__globals__['FlexKVScatterChannel']=channels
                base=100*d
                obj.kv_manager.prefetch_async=Mock(side_effect=[base+1,base+2,base+3])
                # No clock or allocation on a follower may drive its control flow.
                obj._prefetch_next_poll=float('inf') if not comm.is_sync_leader else 0
                for key in ('a','b','c'):
                    materialize=(lambda:np.arange(128)) if comm.is_sync_leader else Mock(side_effect=AssertionError('follower allocation'))
                    self.assertEqual(obj.prefetch_async(key,materialize,token_count=128),base+1+'abc'.index(key))
                    if not comm.is_sync_leader:materialize.assert_not_called()
                obj.cancel_prefetch('b')
                obj.kv_manager.try_wait.return_value={base+1:NS(status=req.KVResponseStatus.SUCCESS,
                    return_mask=np.array([False]*64+[True]*64))}
                time.sleep(.0002*(comm.world_rank%4))
                obj.poll_prefetches()
                self.assertEqual(obj.pop_prefetch_loaded_span('a'),(64,64))
                self.assertEqual(set(obj._ongoing_prefetches),{'b','c'})
                self.assertTrue(obj.check_prefetch_progress('b',poll=False))
                self.assertFalse(obj.check_prefetch_progress('c',poll=False))
                obj._prefetch_next_poll=0 if comm.is_sync_leader else float('inf')
                obj.kv_manager.try_wait.side_effect=RuntimeError('backend response lost')
                with self.assertRaisesRegex(RuntimeError,'backend response lost'):obj.poll_prefetches()
                self.assertEqual(set(obj._ongoing_prefetches),{'b','c'})
                obj.kv_manager.wait.return_value={base+2:NS(status=req.KVResponseStatus.FAILED),
                                                base+3:NS(status=req.KVResponseStatus.SUCCESS)}
                obj._drain_prefetches_for_reset()
                self.assertFalse(obj._ongoing_prefetches or obj._detached_prefetches)
                self.assertEqual(obj.kv_manager.prefetch_async.call_count,3 if comm.is_sync_leader else 0)
                self.assertEqual(obj.kv_manager.try_wait.call_count,2 if comm.is_sync_leader else 0)
                return True
            topology.run(actor);topology.assert_drained()

if __name__=='__main__':unittest.main()
