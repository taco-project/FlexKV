"""Run production FlexKV P2P framing on CPU, without torch/native imports."""
import ast
from enum import Enum
from datetime import timedelta
from pathlib import Path
import pickle
from types import SimpleNamespace as NS
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_comm():
    path = ROOT / "flexkv/integration/sglang/comm.py"
    nodes = []
    for n in ast.parse(path.read_text()).body:
        if getattr(n, "name", "") == "FlexKVScatterChannel": nodes.append(n)
        elif isinstance(n, ast.Assign) and any(getattr(t, "id", "") in {"_SCATTER_CHANNEL_OFFSETS", "_SCATTER_CHANNEL_TYPES"} for t in n.targets): nodes.append(n)
        elif getattr(n, "name", "") == "FlexKVComm":
            n.body = [m for m in n.body if isinstance(m, ast.Assign)
                      or getattr(m, "name", "") in {
                          "_scatter_tag", "_compact_fields", "_pack_control_payload",
                          "_isend_buffers", "_isend", "_recv", "_scatter_group",
                          "_reap_completed_async_works", "_validate_scatter_payload"}]
            nodes.append(n)
    messages = []
    def send(tensor, **kw):
        messages.append((kw, tensor.copy()))
        return NS(wait=lambda: None)
    def recv(tensor, **kw):
        metadata, data = messages.pop(0)
        assert metadata["tag"] == kw["tag"]
        assert data.shape == tensor.shape
        tensor[:] = data
        return NS(wait=lambda: None)
    torch = NS(int64=np.int64, long=np.int64, uint8=np.uint8,
               tensor=lambda data, dtype, **kw: np.array(data, dtype=dtype),
               zeros=lambda n, dtype, **kw: np.zeros(n, dtype=dtype),
               empty=lambda n, dtype, **kw: np.empty(n, dtype=dtype),
               frombuffer=lambda data, dtype, **kw: np.frombuffer(data, dtype=dtype))
    # recv's legacy pickle branch calls Tensor.numpy(), so provide that method.
    class Tensor(np.ndarray):
        def numpy(self): return np.asarray(self)
    torch.empty = lambda n, dtype, **kw: np.empty(n, dtype=dtype).view(Tensor)
    import struct
    ns = dict(Enum=Enum, timedelta=timedelta, torch=torch, dist=NS(isend=send, irecv=recv), pickle=pickle, struct=struct)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
                        names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    comm = ns["FlexKVComm"]()
    comm.world_rank = 1
    comm._world_cpu_group = "cpu"
    comm._async_works = []
    comm._reap_high = 64
    return comm, ns["FlexKVScatterChannel"], messages


class CompactControlTests(unittest.TestCase):
    def test_lookup_and_manifest_take_one_message_for_each_axis(self):
        comm, channel, messages = load_comm()
        for base in (comm._TAG_PP, comm._TAG_CP, comm._TAG_TP):
            for kind, payload in ((channel.LOOKUP, dict(task_id=(1 << 60), hit=65536)),
                                  (channel.LAYERWISE_MANIFEST, dict(cmd=3, task_id=71,
                                    producer_id=4, generation=9, slot_count=128,
                                    expected_signal_count=8, error=None))):
                tag = comm._scatter_tag(base, kind)
                works = comm._isend(1, payload, tag, "cpu")
                self.assertEqual(len(works), 1)
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0][1].nbytes, 56)
                self.assertEqual(comm._recv(0, tag, "cpu"), payload)

    def test_errors_schema_extensions_and_out_of_range_ints_preserve_full_payload(self):
        comm, channel, messages = load_comm()
        for kind, payload in ((channel.LAYERWISE_MANIFEST, dict(cmd=3, task_id=-1,
                producer_id=-1, generation=2, slot_count=64, expected_signal_count=-1,
                error="remote producer failed " * 100)),
                (channel.LOOKUP, dict(task_id=-1, hit=0, extra="new schema")),
                (channel.LOOKUP, dict(task_id=1 << 70, hit=0))):
            tag = comm._scatter_tag(comm._TAG_TP, kind)
            self.assertEqual(len(comm._isend(1, payload, tag)), 3)
            self.assertEqual(comm._recv(0, tag), payload)
            self.assertFalse(messages)

    def test_store_poll_keeps_original_channel_and_pickle_protocol(self):
        comm, channel, messages = load_comm()
        tag = comm._scatter_tag(comm._TAG_CP, channel.STORE_COMPLETION)
        self.assertEqual(len(comm._isend(1, ["rid"], tag)), 2)
        self.assertEqual(comm._recv(0, tag), ["rid"])
        self.assertFalse(messages)

    def test_invalid_header_fails_before_receiving_other_operations(self):
        comm, channel, messages = load_comm()
        tag = comm._scatter_tag(comm._TAG_TP, channel.LOOKUP)
        messages.append((dict(tag=tag), np.array([8, 0, 0, 0, 0, 0, 0], dtype=np.int64)))
        with self.assertRaisesRegex(RuntimeError, "Invalid FlexKV compact header"):
            comm._recv(0, tag)

    def test_pr326_single_serialize_fanout_and_inline_control_framing(self):
        import unittest.mock
        comm, channel, messages = load_comm()
        comm.world_rank = 0
        tag = comm._scatter_tag(comm._TAG_CP, channel.PREFETCH_START)
        payload = {"task_id": 42, "error": ""}

        # 1) _scatter_group serializes payload only ONCE across 7 followers
        with unittest.mock.patch.object(pickle, "dumps", wraps=pickle.dumps) as spy_dumps:
            comm._scatter_group(payload, list(range(8)), is_leader=True, tag=tag, blocking=True)
            self.assertEqual(spy_dumps.call_count, 1)
        # 7 followers * 2 messages (size + data) = 14 messages
        self.assertEqual(len(messages), 14)
        for _ in range(7):
            self.assertEqual(comm._recv(0, tag, "cpu"), payload)
        self.assertFalse(messages)

        # 2) Inline control framing (e.g. 256 bytes): small payload takes 1 message,
        #    large payload (> 248 bytes) takes 2 messages (header + body)
        comm._control_inline_bytes = 256
        works_small = comm._isend(1, payload, tag, "cpu")
        self.assertEqual(len(works_small), 1)
        self.assertEqual(messages[0][1].nbytes, 256)
        self.assertEqual(comm._recv(0, tag, "cpu"), payload)

        big_payload = {"data": list(range(200)), "raw": b"x" * 300}
        works_big = comm._isend(1, big_payload, tag, "cpu")
        self.assertEqual(len(works_big), 2)
        self.assertEqual(comm._recv(0, tag, "cpu"), big_payload)
        self.assertFalse(messages)


if __name__ == "__main__": unittest.main()
