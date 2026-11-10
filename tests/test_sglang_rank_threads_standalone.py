"""Concurrent real hierarchical scatter/reduction over independent DP replicas."""
import unittest
from types import SimpleNamespace as NS
from cpu_collective_test_support import Topology

class RankThreadsTests(unittest.TestCase):
    def scatter(self,topology):
        channel=topology.module.FlexKVScatterChannel
        def actor(index,comm):
            d,p,c,t=index
            payload={'task_id':17+d,'hit':64*(d+1)} if comm.is_sync_leader else {}
            result=comm.scatter(payload,channel=channel.LOOKUP,blocking=True)
            if result!={'task_id':17+d,'hit':64*(d+1)}:
                raise AssertionError(f'replica {d}, rank {comm.world_rank}: {result}')
            return result
        result=topology.run(actor);topology.assert_drained();return result

    def test_cp8_native_scatter(self):self.scatter(Topology(cp=8))
    def test_two_dp_replicas_with_cp_and_tp(self):self.scatter(Topology(dp=2,cp=2,tp=2))
    def test_noncontiguous_global_group_members(self):self.scatter(Topology(dp=2,cp=2,tp=2,permuted=True))
    def test_pp_cp_tp_replica_scatter(self):self.scatter(Topology(dp=2,pp=2,cp=2,tp=2))

    def test_minimum_and_barrier_are_replica_local(self):
        for permuted in (False,True):
            topology=Topology(dp=2,pp=2,cp=2,tp=2,permuted=permuted)
            def actor(index,comm):
                d,p,c,t=index
                result=comm.all_reduce_min(d*100+p*4+c*2+t)
                comm.barrier()
                self.assertEqual(result,d*100)
                return result
            topology.run(actor);topology.assert_drained()

    def test_sustained_cp_scatter_reaps_async_works_without_stale_payload(self):
        topology=Topology(cp=8)
        def actor(index,comm):
            for step in range(400):
                data={'task_id':step,'hit':step*64} if comm.is_sync_leader else {}
                result=comm.scatter(data,channel=topology.module.FlexKVScatterChannel.LOOKUP)
                self.assertEqual(result,{'task_id':step,'hit':step*64})
            return len(comm._async_works)
        result=topology.run(actor);topology.assert_drained()
        self.assertGreater(topology.comms[(0,0,0,0)]._reap_drained_total,0)
        self.assertLessEqual(max(result.values()),topology.module.FlexKVComm._REAP_HIGH_MAX+8)

    def test_disabled_sync_adds_no_startup_or_runtime_collectives(self):
        topology=Topology(pp=2,cp=2,tp=2)
        def actor(index,comm):
            clone=topology.module.FlexKVComm(comm_rank_info(comm),comm.world_rank,
                pp_group=comm.pp_cpu_group,attn_tp_group=comm.attn_tp_cpu_group,
                attn_cp_group=comm.attn_cp_cpu_group,world_group=topology.world,
                enable_collective_sync=False)
            self.assertFalse(clone.needs_sync)
            self.assertEqual(clone.all_reduce_min(comm.world_rank),comm.world_rank)
            clone.barrier()
            data={'task_id':comm.world_rank,'hit':0}
            self.assertIs(clone.scatter(data,channel=topology.module.FlexKVScatterChannel.LOOKUP),data)
            return True
        topology.run(actor)
        self.assertEqual(len(topology.transport.collectives),8)  # Original PP/CP/TP ctor only.
        self.assertFalse(topology.transport.sent)
        topology.assert_drained()

    def test_missing_or_wrong_cp_group_fails_before_any_data_message(self):
        topology=Topology(cp=2)
        def actor(index,comm):
            with self.assertRaisesRegex(RuntimeError,'requires its CPU ProcessGroup'):
                topology.module.FlexKVComm(comm_rank_info(comm),comm.world_rank,
                    world_group=topology.world)
            bad=NS(model_config=comm_rank_info(comm).model_config,pp_rank=0,attn_tp_rank=0,
                   attn_cp_rank=1-index[2],pp_size_per_node=1)
            with self.assertRaisesRegex(RuntimeError,'group/rank mismatch'):
                topology.module.FlexKVComm(bad,comm.world_rank,
                    attn_cp_group=comm.attn_cp_cpu_group,world_group=topology.world)
            return True
        topology.run(actor);self.assertFalse(topology.transport.sent)


def comm_rank_info(comm):
    return NS(model_config=NS(pp_size=comm.pp_size,attn_tp_size=comm.attn_tp_size,
                             attn_cp_size=comm.attn_cp_size),
              pp_rank=comm.pp_rank,attn_tp_rank=comm.attn_tp_rank,
              attn_cp_rank=comm.attn_cp_rank,pp_size_per_node=1)

if __name__=='__main__':unittest.main()
