"""Use actual payload validation and serialized frames, not an identity scatter."""
import unittest
from test_sglang_compact_control_standalone import load_comm
from test_sglang_restore_terminal_standalone import connector


class StoreErrorProtocolTests(unittest.TestCase):
    def test_error_envelope_round_trips_on_every_axis(self):
        comm, channel, messages = load_comm()
        payload = {'error': 'writer uncertain ' * 1000}
        for base in (comm._TAG_PP, comm._TAG_CP, comm._TAG_TP):
            comm._validate_scatter_payload(payload, channel.STORE_COMPLETION, source='local')
            tag = comm._scatter_tag(base, channel.STORE_COMPLETION)
            self.assertEqual(len(comm._isend(1, payload, tag)), 2)
            result = comm._recv(0, tag)
            comm._validate_scatter_payload(result, channel.STORE_COMPLETION, source='received')
            self.assertEqual(result, payload)
            self.assertFalse(messages)

    def test_invalid_error_shapes_and_wrong_channels_still_fail(self):
        comm, channel, _ = load_comm()
        for payload in ({'error':''}, {'error':None}, {'error':17},
                        {'error':'bad', 'task_id':17}, {'task_id':17}):
            with self.assertRaises(TypeError):
                comm._validate_scatter_payload(payload, channel.STORE_COMPLETION, source='local')
        with self.assertRaises(TypeError):
            comm._validate_scatter_payload({'error':'bad'}, channel.STORE_READY, source='local')

    def test_normal_completion_list_keeps_original_wire_format(self):
        comm, channel, _ = load_comm()
        payload = ['a', 'b']
        comm._validate_scatter_payload(payload, channel.STORE_COMPLETION, source='local')
        tag = comm._scatter_tag(comm._TAG_TP, channel.STORE_COMPLETION)
        self.assertEqual(len(comm._isend(1, payload, tag)), 2)
        self.assertEqual(comm._recv(0, tag), payload)

    def test_connector_failure_reaches_both_ranks_through_real_codec(self):
        comm, channel, messages = load_comm()
        tag = comm._scatter_tag(comm._TAG_TP, channel.STORE_COMPLETION)
        leader, follower = connector(), connector(leader=False)
        def scatter_leader(payload, **kwargs):
            comm._validate_scatter_payload(payload, channel.STORE_COMPLETION, source='local')
            comm._isend(1, payload, tag)
            return payload
        def scatter_follower(payload, **kwargs):
            comm._validate_scatter_payload(payload, channel.STORE_COMPLETION, source='local')
            result = comm._recv(0, tag)
            comm._validate_scatter_payload(result, channel.STORE_COMPLETION, source='received')
            return result
        leader._sync_ctx.scatter, follower._sync_ctx.scatter = scatter_leader, scatter_follower
        leader.kv_manager.try_wait.side_effect = RuntimeError('writer uncertain')
        for obj in (leader, follower):
            obj._sync_ctx.needs_sync = True
            obj._inflight_stores = {'a':17}
            with self.assertRaisesRegex(RuntimeError, 'writer uncertain'):
                obj.check_completed_stores()
            self.assertEqual(obj._inflight_stores, {'a':17})
        follower.kv_manager.try_wait.assert_not_called()
        self.assertFalse(messages)


if __name__ == '__main__':
    unittest.main()
